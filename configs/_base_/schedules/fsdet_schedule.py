"""Shared training schedule, with dataset counts resolved from _base_."""
import os
import fsdet_configs.schema as _schema

_schedule_total_images = _schema.sar_base_num_images + _schema.rgb_base_num_images
gpu_number = int(os.environ.get('FSDET_GPUS', '1'))
total_epoch = int(os.environ.get('FSDET_BASE_EPOCHS', '12'))
source_ratio = list(_schema.base_source_ratio)
batch_size = sum(source_ratio)
steps_per_epoch = max(
    1, _schedule_total_images // max(batch_size * gpu_number, 1))
step_epoch = max(1, int(total_epoch * 2 / 3))
if _schema.sar_head_mode == 'gfl':
    sar_optimizer_keys = dict(sar_bbox_head=dict(lr_mult=1.0))
    sar_reweight_losses = dict(
        sar_loss_cls='sar_bbox_head',
        sar_loss_bbox='sar_bbox_head',
        sar_loss_dfl='sar_bbox_head')
else:
    sar_optimizer_keys = dict(
        sar_rpn_head=dict(lr_mult=1.0), sar_roi_head=dict(lr_mult=1.0))
    sar_reweight_losses = dict(
        sar_loss_rpn_cls='sar_rpn_head',
        sar_loss_rpn_bbox='sar_rpn_head',
        sar_loss_cls='sar_roi_head',
        sar_loss_bbox='sar_roi_head')
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            **sar_optimizer_keys,
            **dict(
                rgb_rpn_head=dict(lr_mult=1.0),
                rgb_roi_head=dict(lr_mult=1.0),
                ifr_rpn_head=dict(lr_mult=1.0),
                ifr_roi_head=dict(lr_mult=1.0),
                backbone=dict(lr_mult=1.0),
                neck=dict(lr_mult=1.0))
        }))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='dynamic',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[steps_per_epoch * step_epoch, steps_per_epoch * (total_epoch - 1)],
    extra_args=dict(
        T=3,
        b=0.4,
        ema=0.001,
        backbone_policy='sigmoid_kl',
        head_policy='normal'),
    reweight_losses={
        **sar_reweight_losses,
        **dict(
            rgb_loss_rpn_cls='rgb_rpn_head',
            rgb_loss_rpn_bbox='rgb_rpn_head',
            rgb_loss_cls='rgb_roi_head',
            rgb_loss_bbox='rgb_roi_head')
    })
runner = dict(type='IterBasedRunner', max_iters=steps_per_epoch * total_epoch)
checkpoint_config = dict(interval=steps_per_epoch)
evaluation = dict(
    interval=steps_per_epoch,
    start=8 * steps_per_epoch,
    metric=_schema.sar_eval_metric,
    **dict(classwise=True) if _schema.sar_ann_format == 'hbb_json' else {})
evaluation2 = dict(
    interval=steps_per_epoch, start=8 * steps_per_epoch, metric='mAP')
auto_scale_lr = dict(base_batch_size=batch_size * 8, enable=True)
fp16 = dict(loss_scale='dynamic')
