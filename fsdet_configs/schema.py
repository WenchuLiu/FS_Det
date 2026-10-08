"""Edit this file first when adapting FS_Det to a new dataset."""
import os
import json

# All train/validation pipelines use a shared square resize and pad size.
# ConvNeXt is fully convolutional; keep the value divisible by its total
# downsampling factor so feature maps align with the FPN/anchor strides.
image_size = int(os.environ.get('FSDET_IMAGE_SIZE', '800'))
if image_size <= 0 or image_size % 32 != 0:
    raise ValueError('FSDET_IMAGE_SIZE must be a positive multiple of 32.')

# Example split inherited from the current SARDet/Ours2 setup.
sar_base_classes = ('ship', 'car', 'tank', 'bridge', 'harbor')
sar_novel_classes = ('aircraft', )
sar_all_classes = sar_base_classes + sar_novel_classes

# Example RGB taxonomy in the standard DOTA order.
rgb_classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')

# Map each semantic concept to its local label ID in SAR and RGB annotations.
# A list means several RGB labels are grouped as the same SAR concept.
semantic_mapping = {
    'ship': {
        'sar': 0,
        'rgb': 6
    },
    'car': {
        'sar': 1,
        'rgb': [4, 5]
    },
    'tank': {
        'sar': 2,
        'rgb': 9
    },
    'bridge': {
        'sar': 3,
        'rgb': 2
    },
    'harbor': {
        'sar': 4,
        'rgb': 12
    },
}

# Manual CLIP names, in exactly the annotation-label order above. Edit these
# independently of dataset class names (e.g. annotation "0" -> text "ship").
sar_base_text_classes = ('ship', 'car', 'tank', 'bridge', 'harbor')
sar_novel_text_classes = ('aircraft', )
sar_all_text_classes = sar_base_text_classes + sar_novel_text_classes
rgb_text_classes = ('plane', 'baseball diamond', 'bridge',
                    'ground track field', 'small vehicle', 'large vehicle',
                    'ship', 'tennis court', 'basketball court', 'storage tank',
                    'soccer ball field', 'roundabout', 'harbor',
                    'swimming pool', 'helicopter')
for _labels, _texts in ((sar_base_classes, sar_base_text_classes),
                        (sar_novel_classes, sar_novel_text_classes),
                        (rgb_classes, rgb_text_classes)):
    if len(_labels) != len(_texts) or any(not name.strip() for name in _texts):
        raise ValueError(
            'CLIP text names must be nonempty and match label count/order.')

# Text-align maintains an independent prompt/class row table for each modality.
base_modal_categories = {
    'sar': sar_base_text_classes,
    'rgb': rgb_text_classes,
}
all_modal_categories = {
    'sar': sar_all_text_classes,
    'rgb': rgb_text_classes,
}
novel_sar_label_ids = tuple(
    sar_all_classes.index(c) for c in sar_novel_classes)

sar_base_num_classes = len(sar_base_classes)
sar_all_num_classes = len(sar_all_classes)
rgb_num_classes = len(rgb_classes)

base_source_ratio = [6, 2]

# Fine-tuning can use either modality. SAR is the default. SAR annotations may
# be COCO JSON with horizontal boxes or DOTA text files with rotated boxes.
ft_modality = os.environ.get('FSDET_FT_MODALITY', 'sar').strip().lower()
if ft_modality not in ('sar', 'rgb'):
    raise ValueError('FSDET_FT_MODALITY must be either "sar" or "rgb".')

sar_ann_format = os.environ.get('FSDET_SAR_ANN_FORMAT',
                                'hbb_json').strip().lower()
if sar_ann_format not in ('hbb_json', 'dota'):
    raise ValueError(
        'FSDET_SAR_ANN_FORMAT must be either "hbb_json" or "dota".')

_default_sar_head = 'gfl' if sar_ann_format == 'hbb_json' else 'rotated_two_stage'
sar_head_mode = os.environ.get('FSDET_SAR_HEAD',
                               _default_sar_head).strip().lower()
if sar_head_mode not in ('gfl', 'rotated_two_stage'):
    raise ValueError('FSDET_SAR_HEAD must be "gfl" or "rotated_two_stage".')
if sar_ann_format == 'dota' and sar_head_mode != 'rotated_two_stage':
    raise ValueError('DOTA SAR annotations contain rotated boxes; use '
                     'FSDET_SAR_HEAD=rotated_two_stage (or leave it unset).')
if sar_ann_format == 'hbb_json' and sar_head_mode == 'rotated_two_stage':
    raise ValueError('COCO JSON SAR annotations contain horizontal boxes; use '
                     'FSDET_SAR_HEAD=gfl.')

# Shared model config suffix for the two supported SAR annotation formats.
sar_model_format = {'hbb_json': 'gfl', 'dota': 'dota'}[sar_ann_format]

# Dataset paths. Override FSDET_DATA_ROOT or edit these paths.
data_root = os.environ.get('FSDET_DATA_ROOT', 'data/FS_Det')
sar_data_root = os.environ.get('FSDET_SAR_DATA_ROOT',
                               os.path.join(data_root, 'SAR'))
if sar_ann_format == 'hbb_json':
    _sar_default_base_ann = os.path.join(sar_data_root,
                                         'annotations/base_train.json')
    _sar_default_val_ann = os.path.join(sar_data_root,
                                        'annotations/base_val.json')
    _sar_default_test_ann = os.path.join(sar_data_root,
                                         'annotations/test.json')
    _sar_default_base_img_root = os.path.join(sar_data_root, 'images/train')
    _sar_default_val_img_root = os.path.join(sar_data_root, 'images/val')
    _sar_default_test_img_root = os.path.join(sar_data_root, 'images/test')
    _sar_default_fewshot_ann = ''
    _sar_default_fewshot_img_root = ''
    _sar_default_manifest = ''
else:
    _sar_default_base_ann = os.path.join(sar_data_root,
                                         'train/annotations_base')
    _sar_default_val_ann = os.path.join(sar_data_root, 'val/annfiles')
    _sar_default_test_ann = os.path.join(sar_data_root, 'test/annfiles')
    _sar_default_base_img_root = os.path.join(sar_data_root, 'train/images')
    _sar_default_val_img_root = os.path.join(sar_data_root, 'val/images')
    _sar_default_test_img_root = os.path.join(sar_data_root, 'test/images')
    _sar_default_fewshot_ann = os.path.join(sar_data_root,
                                            'train/annotations_few_shot')
    _sar_default_fewshot_img_root = os.path.join(sar_data_root, 'train/images')
    _sar_default_manifest = os.path.join(sar_data_root,
                                         'train/split_manifest.json')

sar_train_ann = os.environ.get('FSDET_SAR_BASE_ANN', _sar_default_base_ann)
sar_val_ann = os.environ.get('FSDET_SAR_VAL_ANN', _sar_default_val_ann)
sar_test_ann = os.environ.get('FSDET_SAR_TEST_ANN', _sar_default_test_ann)
sar_base_img_root = os.environ.get('FSDET_SAR_BASE_IMG_ROOT',
                                   _sar_default_base_img_root)
sar_val_img_root = os.environ.get('FSDET_SAR_VAL_IMG_ROOT',
                                  _sar_default_val_img_root)
sar_test_img_root = os.environ.get('FSDET_SAR_TEST_IMG_ROOT',
                                   _sar_default_test_img_root)
sar_fewshot_ann = os.environ.get('FSDET_SAR_FEWSHOT_ANN',
                                 _sar_default_fewshot_ann)
sar_fewshot_img_root = os.environ.get('FSDET_SAR_FEWSHOT_IMG_ROOT',
                                      _sar_default_fewshot_img_root)
sar_split_manifest = os.environ.get('FSDET_SAR_SPLIT_MANIFEST',
                                    _sar_default_manifest)

_sar_manifest_base_count = None
if sar_split_manifest and os.path.isfile(sar_split_manifest):
    try:
        with open(sar_split_manifest, encoding='utf-8') as _manifest_file:
            _sar_manifest_base_count = json.load(_manifest_file).get(
                'num_base_images')
    except (OSError, ValueError, TypeError):
        _sar_manifest_base_count = None
sar_base_num_images = int(
    os.environ.get(
        'FSDET_SAR_BASE_IMAGES',
        str(_sar_manifest_base_count if _sar_manifest_base_count else 1000)))

rgb_train_ann_default = os.path.join(data_root, 'RGB/train/annotations_base')
if not os.path.isdir(rgb_train_ann_default):
    rgb_train_ann_default = os.path.join(data_root, 'RGB/train/annfiles')
rgb_train_ann = os.environ.get('FSDET_RGB_BASE_ANN', rgb_train_ann_default)
rgb_fewshot_ann = os.environ.get(
    'FSDET_RGB_FEWSHOT_ANN',
    os.path.join(data_root, 'RGB/train/annotations_few_shot'))
rgb_split_manifest = os.environ.get(
    'FSDET_RGB_SPLIT_MANIFEST',
    os.path.join(data_root, 'RGB/train/split_manifest.json'))
rgb_val_ann = os.environ.get('FSDET_RGB_VAL_ANN',
                             os.path.join(data_root, 'RGB/val/annfiles'))
rgb_test_ann = os.environ.get('FSDET_RGB_TEST_ANN', rgb_val_ann)
rgb_train_img_root = os.environ.get(
    'FSDET_RGB_IMG_ROOT', os.path.join(data_root, 'RGB/train/images'))
rgb_val_img_root = os.environ.get('FSDET_RGB_VAL_IMG_ROOT',
                                  os.path.join(data_root, 'RGB/val/images'))
rgb_test_img_root = os.environ.get('FSDET_RGB_TEST_IMG_ROOT', rgb_val_img_root)

_manifest_base_count = None
if os.path.isfile(rgb_split_manifest):
    try:
        with open(rgb_split_manifest, encoding='utf-8') as _manifest_file:
            _manifest_base_count = json.load(_manifest_file).get(
                'num_base_images')
    except (OSError, ValueError, TypeError):
        _manifest_base_count = None
rgb_base_num_images = int(
    os.environ.get(
        'FSDET_RGB_BASE_IMAGES',
        str(_manifest_base_count if _manifest_base_count else 1000)))

# Evaluation follows the annotation geometry.
sar_eval_metric = {'hbb_json': 'bbox', 'dota': 'mAP'}[sar_ann_format]
ft_eval_metric = {'sar': sar_eval_metric, 'rgb': 'mAP'}[ft_modality]
