#!/usr/bin/env python
"""Attach collected Ours2 routing buffers to an existing Ours FT checkpoint.

This keeps the original Ours detector weights, including its expanded SAR
classifier initialization, unchanged while adding the base routing signatures
collected from the matching raw base checkpoint.
"""

import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Attach Ours2 SAR routing signatures to an Ours checkpoint')
    parser.add_argument('--signature-checkpoint', required=True)
    parser.add_argument('--init-checkpoint', required=True,
                        help='Existing Ours few-shot initialization checkpoint')
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def _load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        raise ValueError(f'{path} is not an MMCV model checkpoint.')
    return checkpoint


def main():
    args = parse_args()
    signature_checkpoint = _load_checkpoint(args.signature_checkpoint)
    init_checkpoint = _load_checkpoint(args.init_checkpoint)

    signature_state = signature_checkpoint['state_dict']
    target_state = init_checkpoint['state_dict']
    signature_keys = ('base_routing_sums', 'base_routing_counts')
    for key in signature_keys:
        if key not in signature_state:
            raise KeyError(
                f'{args.signature_checkpoint} does not contain {key}.')

    sums = signature_state['base_routing_sums'].detach().cpu()
    counts = signature_state['base_routing_counts'].detach().cpu()
    if counts.numel() == 0 or not bool((counts > 0).all()):
        raise ValueError('Collected signatures are missing one or more base classes.')
    if sums.dim() != 3 or counts.shape != sums.shape[:2]:
        raise ValueError(
            f'Incompatible signature shapes: sums={tuple(sums.shape)}, '
            f'counts={tuple(counts.shape)}.')

    target_state['base_routing_sums'] = sums
    target_state['base_routing_counts'] = counts
    visual_keys = ('base_visual_sums', 'base_visual_counts')
    present_visual_keys = [key for key in visual_keys if key in signature_state]
    if present_visual_keys and len(present_visual_keys) != len(visual_keys):
        raise KeyError(
            'The signature checkpoint must contain both base visual '
            'prototype buffers or neither.')
    if len(present_visual_keys) == len(visual_keys):
        visual_sums = signature_state['base_visual_sums'].detach().cpu()
        visual_counts = signature_state['base_visual_counts'].detach().cpu()
        if (visual_sums.dim() != 2 or
                visual_counts.shape != visual_sums.shape[:1] or
                visual_counts.numel() == 0 or
                not bool((visual_counts > 0).all())):
            raise ValueError(
                'Collected SAR visual prototypes are missing or have '
                f'incompatible shapes: sums={tuple(visual_sums.shape)}, '
                f'counts={tuple(visual_counts.shape)}.')
        target_state['base_visual_sums'] = visual_sums
        target_state['base_visual_counts'] = visual_counts
    meta = init_checkpoint.get('meta')
    if not isinstance(meta, dict):
        meta = {}
        init_checkpoint['meta'] = meta
    meta['ours2_routing_signature_collection'] = dict(
        source_checkpoint=os.path.abspath(args.signature_checkpoint),
        source_modality='sar',
        visual_prototypes_attached=(len(present_visual_keys) == len(visual_keys)),
        base_classes=signature_checkpoint.get('meta', {}).get(
            'ours2_routing_signature_collection', {}).get('base_classes', []),
        init_checkpoint=os.path.abspath(args.init_checkpoint),
        collection_mode='eval_no_grad',
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = args.output + '.tmp'
    torch.save(init_checkpoint, tmp_path)
    os.replace(tmp_path, args.output)
    print(f'Attached routing signatures to original Ours checkpoint: {args.output}')
    print(f'SAR class signature counts per MoE layer: {counts.tolist()}')


if __name__ == '__main__':
    main()
