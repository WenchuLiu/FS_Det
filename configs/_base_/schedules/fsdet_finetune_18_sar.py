"""Shared training schedule, with dataset counts resolved from _base_."""
import os
import fsdet_configs.schema as _schema
import configs._base_.datasets.fsdet_18shot_sar as _ft_dataset

_schedule_total_images = _ft_dataset.ft_num_images
gpu_number = int(os.environ.get('FSDET_GPUS', '1'))
total_epoch = int(os.environ.get('FSDET_FT_EPOCHS', '12'))
source_ratio = [1]
batch_size = 1
steps_per_epoch = max(
    1, _schedule_total_images // max(batch_size * gpu_number, 1))
total_iters = steps_per_epoch * total_epoch
optimizer = dict(type='AdamW', lr=5e-05, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=0.3333333333333333,
    step=[
        max(1, int(total_iters * 0.7)),
        max(2, int(total_iters * 0.9))
    ])
runner = dict(type='IterBasedRunner', max_iters=total_iters)
checkpoint_config = dict(interval=steps_per_epoch)
evaluation = dict(
    interval=steps_per_epoch, metric=_schema.sar_eval_metric,
    **dict(classwise=True) if _schema.sar_ann_format == 'hbb_json' else {})
auto_scale_lr = dict(base_batch_size=batch_size * gpu_number, enable=False)
fp16 = dict(loss_scale='dynamic')
