"""Dataset definitions and pipelines; metadata is read from schema.py."""
import os
import json
import copy as _copy
import fsdet_configs.schema as _schema


def build_base_data():
    """Build SAR/RGB datasets from the paths and labels above."""
    dataset_type_sar = 'SARDet_hbb' if _schema.sar_ann_format == 'hbb_json' else 'DOTADataset'
    dataset_type_rgb = 'DOTADataset'
    img_size = _schema.image_size
    angle_version = 'le90'
    branch_field = ['sar', 'rgb']
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True)
    sar_train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', img_scale=(img_size, img_size), keep_ratio=True),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=(img_size, img_size)),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        dict(type='MultiBranch', branch_field=branch_field, sar=True)
    ]
    rgb_train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RResize', img_scale=(img_size, img_size)),
        dict(
            type='RRandomFlip',
            flip_ratio=[0.25, 0.25, 0.25],
            direction=['horizontal', 'vertical', 'diagonal'],
            version=angle_version),
        dict(
            type='PolyRandomRotate',
            rotate_ratio=0.5,
            angles_range=180,
            auto_bound=False,
            rect_classes=[4, 5, 7, 8, 10, 14],
            version=angle_version),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size=(img_size, img_size)),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        dict(type='MultiBranch', branch_field=branch_field, rgb=True)
    ]
    sar_test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(img_size, img_size),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size=(img_size, img_size)),
                dict(type='ImageToTensor', keys=['img']),
                dict(
                    type='Collect_subdataset', keys=['img'], subdataset='sar')
            ])
    ]
    if _schema.sar_ann_format == 'dota':
        sar_train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(img_size, img_size)),
            dict(
                type='RRandomFlip',
                flip_ratio=[0.25, 0.25, 0.25],
                direction=['horizontal', 'vertical', 'diagonal'],
                version=angle_version),
            dict(
                type='PolyRandomRotate',
                rotate_ratio=0.5,
                angles_range=180,
                auto_bound=False,
                rect_classes=[],
                version=angle_version),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(img_size, img_size)),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
            dict(type='MultiBranch', branch_field=branch_field, sar=True)
        ]
        sar_test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(img_size, img_size),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size=(img_size, img_size)),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect_subdataset',
                        keys=['img'],
                        subdataset='sar')
                ])
        ]
    rgb_test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(img_size, img_size),
            flip=False,
            transforms=[
                dict(type='RResize'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size=(img_size, img_size)),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect_subdataset', keys=['img'], subdataset='rgb')
            ])
    ]
    sar_train_dataset = dict(
        type=dataset_type_sar,
        ann_file=_schema.sar_train_ann,
        img_prefix=_schema.sar_base_img_root,
        classes=_schema.sar_base_classes,
        pipeline=sar_train_pipeline,
        **dict(
            version=angle_version,
            difficulty=100,
            cache_annotations=None,
            cache_filtered=None,
            filter_unknown_classes=True)
        if _schema.sar_ann_format == 'dota' else {})
    rgb_train_dataset = dict(
        type=dataset_type_rgb,
        ann_file=_schema.rgb_train_ann,
        img_prefix=_schema.rgb_train_img_root,
        classes=_schema.rgb_classes,
        pipeline=rgb_train_pipeline,
        version=angle_version,
        cache_annotations=None,
        cache_filtered=None)
    rgb_val_dataset = dict(
        type=dataset_type_rgb,
        ann_file=_schema.rgb_val_ann,
        img_prefix=_schema.rgb_val_img_root,
        classes=_schema.rgb_classes,
        pipeline=rgb_test_pipeline,
        version=angle_version,
        cache_annotations=None,
        cache_filtered=None)
    data = dict(
        samples_per_gpu=sum(_schema.base_source_ratio),
        workers_per_gpu=4,
        train=dict(
            type='ConcatDataset',
            datasets=[sar_train_dataset, rgb_train_dataset]),
        val=dict(
            type=dataset_type_sar,
            ann_file=_schema.sar_val_ann,
            img_prefix=_schema.sar_val_img_root,
            classes=_schema.sar_base_classes,
            pipeline=sar_test_pipeline,
            **dict(
                version=angle_version,
                difficulty=100,
                cache_annotations=None,
                cache_filtered=None,
                filter_unknown_classes=True)
            if _schema.sar_ann_format == 'dota' else {}),
        val_2=rgb_val_dataset,
        test=dict(
            type=dataset_type_sar,
            ann_file=_schema.sar_test_ann,
            img_prefix=_schema.sar_test_img_root,
            classes=_schema.sar_base_classes,
            pipeline=sar_test_pipeline,
            **dict(
                version=angle_version,
                difficulty=100,
                cache_annotations=None,
                cache_filtered=None,
                filter_unknown_classes=True)
            if _schema.sar_ann_format == 'dota' else {}),
        train_dataloader=dict(
            multi_datasets=True, source_ratio=_schema.base_source_ratio),
        val_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4),
        test_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4))
    return data


def _sar_test_pipeline():
    return [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(_schema.image_size, _schema.image_size),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(
                    type='Pad', size=(_schema.image_size, _schema.image_size)),
                dict(type='ImageToTensor', keys=['img']),
                dict(
                    type='Collect_subdataset', keys=['img'], subdataset='sar')
            ])
    ]


def _rgb_train_pipeline(branch_field=('rgb', )):
    return [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='RResize',
            img_scale=(_schema.image_size, _schema.image_size)),
        dict(
            type='RRandomFlip',
            flip_ratio=[0.25, 0.25, 0.25],
            direction=['horizontal', 'vertical', 'diagonal'],
            version='le90'),
        dict(
            type='PolyRandomRotate',
            rotate_ratio=0.5,
            angles_range=180,
            auto_bound=False,
            rect_classes=[4, 5, 7, 8, 10, 14],
            version='le90'),
        dict(
            type='Normalize',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True),
        dict(type='Pad', size=(_schema.image_size, _schema.image_size)),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        dict(type='MultiBranch', branch_field=list(branch_field), rgb=True)
    ]


def _sar_dota_train_pipeline():
    """Training pipeline for SAR DOTA polygon annotations and rotated boxes."""
    pipeline = _rgb_train_pipeline(branch_field=('sar', ))
    for transform in pipeline:
        if transform.get('type') == 'PolyRandomRotate':
            transform['rect_classes'] = []
        if transform.get('type') == 'MultiBranch':
            transform.pop('rgb', None)
            transform['sar'] = True
    return pipeline


def _sar_dota_test_pipeline():
    pipeline = _rgb_test_pipeline()
    for transform in pipeline[1]['transforms']:
        if transform.get('type') == 'Collect_subdataset':
            transform['subdataset'] = 'sar'
    return pipeline


def _load_sar_manifest(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as manifest_file:
            return json.load(manifest_file)
    except (OSError, ValueError, TypeError):
        return None


def _rgb_test_pipeline():
    return [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(_schema.image_size, _schema.image_size),
            flip=False,
            transforms=[
                dict(type='RResize'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(
                    type='Pad', size=(_schema.image_size, _schema.image_size)),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect_subdataset', keys=['img'], subdataset='rgb')
            ])
    ]


def fewshot_image_count(num_shot, modality=None, manifest_file=None):
    """Resolve the support image count without constructing a dataset."""
    modality = modality or _schema.ft_modality
    if modality == 'sar':
        count_env = f'FSDET_FT{num_shot}_IMAGES'
        default_manifest = (_schema.sar_split_manifest
                            if manifest_file is None else manifest_file)
        sar_manifest = _load_sar_manifest(
            os.environ.get(f'FSDET_SAR_FT{num_shot}_MANIFEST',
                           default_manifest))
        if _schema.sar_ann_format == 'dota':
            default_count = (sar_manifest or {}).get(
                'num_few_shot_images',
                num_shot * len(_schema.sar_novel_classes))
        else:
            default_count = num_shot * len(_schema.sar_novel_classes)
        image_count = int(os.environ.get(count_env, str(default_count)))
    else:
        count_env = f'FSDET_RGB_FT{num_shot}_IMAGES'
        manifest_count = None
        default_manifest = (_schema.rgb_split_manifest
                            if manifest_file is None else manifest_file)
        rgb_manifest_path = os.environ.get(
            f'FSDET_RGB_FT{num_shot}_MANIFEST', default_manifest)
        if os.path.isfile(rgb_manifest_path):
            try:
                with open(
                        rgb_manifest_path,
                        encoding='utf-8') as rgb_manifest_file:
                    manifest_count = json.load(rgb_manifest_file).get(
                        'num_few_shot_images')
            except (OSError, ValueError, TypeError):
                manifest_count = None
        default_count = manifest_count or num_shot * len(_schema.rgb_classes)
        image_count = int(os.environ.get(count_env, str(default_count)))
    return image_count


def build_fewshot_config(num_shot,
                         modality=None,
                         ann_file=None,
                         img_root=None,
                         manifest_file=None):
    """Return few-shot data and count, accepting shot-specific paths.

    ``ann_file``, ``img_root`` and ``manifest_file`` override shared schema
    paths; per-shot modality-specific environment variables override those
    values.
    """
    modality = modality or _schema.ft_modality
    image_count = fewshot_image_count(
        num_shot, modality=modality, manifest_file=manifest_file)
    if modality == 'sar':
        shot_ann_env = f'FSDET_SAR_FT{num_shot}_ANN'
        shot_img_env = f'FSDET_SAR_FT{num_shot}_IMG_ROOT'
        if _schema.sar_ann_format == 'dota':
            ann_file = os.environ.get(
                shot_ann_env, ann_file or _schema.sar_fewshot_ann)
            img_root = os.environ.get(
                shot_img_env, img_root or _schema.sar_fewshot_img_root)
            dataset = dict(
                type='DOTADataset',
                ann_file=ann_file,
                img_prefix=img_root,
                classes=_schema.sar_all_classes,
                pipeline=_sar_dota_train_pipeline(),
                version='le90',
                difficulty=100,
                cache_annotations=None,
                cache_filtered=None,
                filter_unknown_classes=True)
            val_test_dataset = dict(
                type='DOTADataset',
                ann_file=_schema.sar_test_ann,
                img_prefix=_schema.sar_test_img_root,
                classes=_schema.sar_all_classes,
                pipeline=_sar_dota_test_pipeline(),
                version='le90',
                difficulty=100,
                cache_annotations=None,
                cache_filtered=None,
                filter_unknown_classes=True)
        else:
            default_ann = os.path.join(
                _schema.data_root, f'SAR/annotations/fewshot_{num_shot}.json')
            default_img_root = os.path.join(_schema.data_root,
                                            f'SAR/images/fewshot_{num_shot}')
            ann_file = os.environ.get(
                shot_ann_env, ann_file or _schema.sar_fewshot_ann
                or default_ann)
            img_root = os.environ.get(
                shot_img_env, img_root or _schema.sar_fewshot_img_root
                or default_img_root)
            dataset = dict(
                type='SARDet_hbb',
                ann_file=ann_file,
                img_prefix=img_root,
                classes=_schema.sar_all_classes,
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='Resize',
                        img_scale=(_schema.image_size, _schema.image_size),
                        keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(
                        type='Pad',
                        size=(_schema.image_size, _schema.image_size)),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect', keys=['img', 'gt_bboxes',
                                              'gt_labels']),
                    dict(type='MultiBranch', branch_field=['sar'], sar=True)
                ])
            val_test_dataset = dict(
                type='SARDet_hbb',
                ann_file=_schema.sar_test_ann,
                img_prefix=_schema.sar_test_img_root,
                classes=_schema.sar_all_classes,
                pipeline=_sar_test_pipeline())
    else:
        shot_ann_env = f'FSDET_RGB_FT{num_shot}_ANN'
        shot_img_env = f'FSDET_RGB_FT{num_shot}_IMG_ROOT'
        ann_file = os.environ.get(
            shot_ann_env, ann_file or _schema.rgb_fewshot_ann)
        img_root = os.environ.get(
            shot_img_env, img_root or _schema.rgb_train_img_root)
        dataset = dict(
            type='DOTADataset',
            ann_file=ann_file,
            img_prefix=img_root,
            classes=_schema.rgb_classes,
            pipeline=_rgb_train_pipeline(branch_field=('rgb', )),
            version='le90',
            difficulty=100,
            cache_annotations=None,
            cache_filtered=None)
        val_test_dataset = dict(
            type='DOTADataset',
            ann_file=_schema.rgb_test_ann,
            img_prefix=_schema.rgb_test_img_root,
            classes=_schema.rgb_classes,
            pipeline=_rgb_test_pipeline(),
            version='le90',
            difficulty=100,
            cache_annotations=None,
            cache_filtered=None)
    val_dataset = _copy.deepcopy(val_test_dataset)
    val_dataset[
        'ann_file'] = _schema.sar_val_ann if modality == 'sar' else _schema.rgb_val_ann
    val_dataset[
        'img_prefix'] = _schema.sar_val_img_root if modality == 'sar' else _schema.rgb_val_img_root
    val_dataset['test_mode'] = True
    val_test_dataset['test_mode'] = True
    data = dict(
        samples_per_gpu=1,
        workers_per_gpu=4,
        train=dataset,
        val=val_dataset,
        test=val_test_dataset,
        train_dataloader=dict(multi_datasets=False),
        val_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4),
        test_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4))
    return (data, image_count)
