#!/usr/bin/env python
"""Create a shape-compatible few-shot initialization checkpoint.

Matching tensors are copied, and SAR prediction layers are expanded by class
name. Base-class and softmax-background rows retain their trained weights;
only novel rows keep the target model's initialization.
"""

import argparse
import os

import torch
from mmcv import Config

from mmrotate.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Few-shot config for the target method')
    parser.add_argument('base_checkpoint', help='Base model checkpoint')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def load_state(path):
    checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict):
        raise TypeError(f'{path} is not a checkpoint dictionary')
    state = checkpoint.get('state_dict', checkpoint)
    normalized = {}
    for key, value in state.items():
        clean_key = key[7:] if key.startswith('module.') else key
        normalized[clean_key] = value.detach().cpu() if torch.is_tensor(value) else value
    return checkpoint, normalized


def expand_sar_prediction(key, source, target, base_classes, all_classes):
    """Remap foreground rows and relocate the final softmax background row."""
    softmax = key.startswith('sar_roi_head.') and '.fc_cls.' in key
    sigmoid = key.startswith('sar_bbox_head.') and '.gfl_cls.' in key
    regression = key.startswith('sar_roi_head.') and '.fc_reg.' in key
    if not (softmax or sigmoid or regression):
        return None
    old_count, new_count = len(base_classes), len(all_classes)
    if not old_count or not new_count or source.shape[1:] != target.shape[1:]:
        return None
    if softmax:
        if source.shape[0] != old_count + 1 or target.shape[0] != new_count + 1:
            raise ValueError(f'{key}: classifier shape disagrees with SAR class names')
        block = 1
    elif sigmoid:
        if source.shape[0] != old_count or target.shape[0] != new_count:
            raise ValueError(f'{key}: classifier shape disagrees with SAR class names')
        block = 1
    else:
        # Class-agnostic regression is copied through the normal shape path.
        if source.shape == target.shape:
            return None
        if source.shape[0] != old_count * 5 or target.shape[0] != new_count * 5:
            raise ValueError(f'{key}: unsupported class-specific regression shape')
        block = 5
    result = target.clone()
    for old_id, name in enumerate(base_classes):
        if name not in all_classes:
            raise ValueError(f'Base class {name!r} is absent from target SAR classes')
        new_id = all_classes.index(name)
        result[new_id * block:(new_id + 1) * block] = source[
            old_id * block:(old_id + 1) * block].to(dtype=target.dtype)
    if softmax:
        result[-1] = source[-1].to(dtype=target.dtype)
    return result


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model_cfg = cfg.model.copy()
    model_cfg.pretrained = None
    model_cfg.backbone.init_cfg = None

    # Avoid downloading the optional prototype/text encoder just to discover
    # target tensor shapes. The actual training config still builds it normally.
    detector_type = model_cfg.get('type')
    if detector_type == 'TriSourceAlignDetectorV6':
        # Keep learned method3 text adapters/temperature. Removing the whole
        # aligner would silently randomize the frozen adapter during fine-tuning.
        align_cfg = model_cfg.get('align_cfg')
        if align_cfg is not None:
            align_cfg = align_cfg.copy()
            align_cfg.initialize_text_features = False
            model_cfg.align_cfg = align_cfg
    elif detector_type in ('TriSourceDetectorV16', 'TriSourceDetectorV16Ours2'):
        align_cfg = model_cfg.get('align_cfg')
        if align_cfg is not None:
            align_cfg = align_cfg.copy()
            align_cfg.enable_backbone_align_inputs = False
            align_cfg.auto_backbone_text_from_aligner = False
            model_cfg.align_cfg = align_cfg

    target_model = build_detector(model_cfg, test_cfg=cfg.get('test_cfg'))
    target_state = target_model.state_dict()
    source_checkpoint, source_state = load_state(args.base_checkpoint)
    base_classes = list(cfg.get('sar_base_classes', ()))
    all_classes = list(cfg.get('sar_all_classes', ()))
    source_config = source_checkpoint.get('meta', {}).get('config')
    if source_config:
        base_classes = list(Config.fromstring(source_config, '.py').get(
            'sar_base_classes', base_classes))
    routing_cfg = cfg.model.get('routing_transfer_cfg', {})
    if routing_cfg.get('enable_transfer', False) and routing_cfg.get('prior_mode', 'semantic_topk') != 'none':
        counts = source_state.get('base_routing_counts')
        if counts is None or counts.numel() == 0 or not bool((counts > 0).all()):
            raise RuntimeError(
                'Base checkpoint has missing routing signatures. Enable '
                'collect_base_signatures in base training, or run '
                'tools/misc/collect_ours2_routing_signatures.py once for an old checkpoint.')
        for signature_key in ('base_routing_counts', 'base_routing_sums'):
            source_signature = source_state.get(signature_key)
            target_signature = target_state.get(signature_key)
            if (source_signature is None or target_signature is None or
                    source_signature.shape != target_signature.shape or
                    not bool(torch.isfinite(source_signature).all())):
                raise RuntimeError(
                    f'Incompatible {signature_key}; recollect signatures for the target backbone/classes.')
        print('Reusing base routing signatures from the checkpoint; no collection pass needed.')
    copied = []
    expanded = []
    skipped_shape = []
    missing = []
    for key, target_value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            missing.append(key)
            continue
        remapped = expand_sar_prediction(
            key, source_value, target_value, base_classes, all_classes)
        if remapped is not None:
            target_state[key] = remapped
            expanded.append(key)
            continue
        if tuple(source_value.shape) != tuple(target_value.shape):
            skipped_shape.append((key, tuple(source_value.shape), tuple(target_value.shape)))
            continue
        target_state[key] = source_value.to(dtype=target_value.dtype)
        copied.append(key)

    target_model.load_state_dict(target_state, strict=True)
    output_checkpoint = dict(
        meta=dict(
            initialized_from=os.path.abspath(args.base_checkpoint),
            target_config=os.path.abspath(args.config),
            copied_tensor_count=len(copied),
            skipped_shape_count=len(skipped_shape),
            missing_tensor_count=len(missing),
            expanded_prediction_tensors=expanded,
            sar_base_classes=base_classes,
            sar_all_classes=all_classes,
            note='SAR base/background rows are preserved; novel rows retain target initialization.'),
        state_dict={
            key: value.detach().cpu()
            for key, value in target_model.state_dict().items()
        })
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary_path = args.output + '.tmp'
    torch.save(output_checkpoint, temporary_path)
    os.replace(temporary_path, args.output)
    print(f'Saved {args.output}')
    print(f'Copied matching tensors: {len(copied)}')
    print(f'Remapped SAR prediction tensors: {expanded}')
    print(f'Skipped shape mismatches: {len(skipped_shape)}')
    print(f'Target-only tensors: {len(missing)}')
    for key, source_shape, target_shape in skipped_shape:
        print(f'  shape changed: {key}: {source_shape} -> {target_shape}')


if __name__ == '__main__':
    main()
