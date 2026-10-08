"""Fixed SAR 18-shot datasets."""
import fsdet_configs.schema as _schema
from configs._base_.datasets.fsdet_data import build_fewshot_config as _build_fewshot_config

num_shot = 18
# Shot-specific paths; edit these here when this split is stored separately.
# For DOTA, ann_file is the annotations_few_shot directory, img_prefix is the
# shared image directory, and manifest_file is this split's split_manifest.json.
fewshot_ann_file = _schema.sar_fewshot_ann
fewshot_img_prefix = _schema.sar_fewshot_img_root
fewshot_manifest_file = _schema.sar_split_manifest
data, ft_num_images = _build_fewshot_config(
    num_shot,
    modality='sar',
    ann_file=fewshot_ann_file,
    img_root=fewshot_img_prefix,
    manifest_file=fewshot_manifest_file)
total_images = ft_num_images
