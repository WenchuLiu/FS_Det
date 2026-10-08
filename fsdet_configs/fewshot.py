"""Compatibility entrypoints; dataset definitions live in configs/_base_/datasets."""
import os
from configs._base_.datasets.fsdet_data import build_fewshot_config as _build_fewshot_config


def build_fewshot_config(num_shot):
    data, image_count = _build_fewshot_config(num_shot)
    data['_delete_'] = True
    return data, image_count


def build_finetune_schedule(image_count, total_epoch):
    """Return the common fine-tuning schedule without a dataset config base."""
    gpu_number = int(os.environ.get('FSDET_GPUS', '1'))
    batch_size = 1
    steps_per_epoch = max(1,
                          int(image_count) // max(batch_size * gpu_number, 1))
    return dict(
        gpu_number=gpu_number,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        optimizer=dict(
            type='AdamW', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.05),
        optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2)),
        lr_config=dict(
            policy='step',
            warmup='linear',
            warmup_iters=100,
            warmup_ratio=1.0 / 3,
            step=[
                max(1, int(steps_per_epoch * 0.7)),
                max(2, int(steps_per_epoch * 0.9)),
            ]),
        runner=dict(
            type='IterBasedRunner',
            max_iters=max(1, steps_per_epoch * int(total_epoch))),
        checkpoint_config=dict(interval=max(1, steps_per_epoch)),
        evaluation=dict(
            interval=max(1, steps_per_epoch), metric='bbox', classwise=True),
        auto_scale_lr=dict(base_batch_size=gpu_number, enable=False))
