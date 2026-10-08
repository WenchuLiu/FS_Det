#!/usr/bin/env python
"""Collect SAR base routing signatures from a trained base checkpoint.

This is an offline post-training pass. It loads the SAR child of the base
training dataset, runs the model in eval/no-grad mode, and saves a checkpoint
with per-class routing signature sums and counts populated.
"""

import argparse
import copy
import os
import random

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmrotate.datasets import build_dataloader, build_dataset
from mmrotate.models import build_detector
from mmrotate.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect SAR-only Ours2 base routing signatures')
    parser.add_argument('config', help='Ours2 base config for the split')
    parser.add_argument('checkpoint', help='trained base checkpoint')
    parser.add_argument('--output', required=True,
                        help='checkpoint path to write after collection')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--samples-per-gpu', type=int, default=4)
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-interval', type=int, default=100)
    return parser.parse_args()


def _build_sar_only_dataset(cfg):
    train_cfg = copy.deepcopy(cfg.data.train)
    if train_cfg.get('type') != 'ConcatDataset':
        raise ValueError(
            'Expected cfg.data.train to be a ConcatDataset with SAR first.')
    datasets = train_cfg.get('datasets', [])
    if not datasets:
        raise ValueError('The base training config has no train datasets.')
    train_cfg['datasets'] = [datasets[0]]
    return build_dataset(train_cfg)


def _checkpoint_with_model_state(checkpoint_path, model, output_path,
                                 num_images):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint_obj = checkpoint
    else:
        checkpoint_obj = dict(meta={}, state_dict=checkpoint)

    state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    checkpoint_obj['state_dict'] = state_dict
    if not isinstance(checkpoint_obj.get('meta'), dict):
        checkpoint_obj['meta'] = {}
    meta = checkpoint_obj['meta']
    meta['ours2_routing_signature_collection'] = dict(
        source_modality='sar',
        num_images=int(num_images),
        base_classes=list(model.base_routing_classes),
        collection_mode='eval_no_grad',
        visual_prototypes_collected=bool(
            hasattr(model, 'base_visual_counts') and
            bool((model.base_visual_counts > 0).all())),
    )

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(checkpoint_obj, output_path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Signature collection requires a CUDA device.')
    if args.samples_per_gpu <= 0 or args.workers_per_gpu < 0:
        raise ValueError('Invalid dataloader batch size or worker count.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device_id)

    cfg = Config.fromfile(args.config)
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    routing_cfg = cfg.model.get('routing_transfer_cfg')
    if routing_cfg is None:
        raise ValueError('The model config must define routing_transfer_cfg.')
    routing_cfg.collect_base_signatures = True
    routing_cfg.enable_transfer = False

    dataset = _build_sar_only_dataset(cfg)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=args.samples_per_gpu,
        workers_per_gpu=args.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
        seed=args.seed,
        runner_type='EpochBasedRunner',
        persistent_workers=False,
        multi_datasets=False,
    )

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
    model.cuda(args.device_id)
    model.eval()
    if not hasattr(model, 'set_base_signature_collection'):
        raise TypeError('The config must build TriSourceDetectorV16Ours2.')
    model.reset_base_routing_signatures()
    model.set_base_signature_collection(True, in_eval=True)
    model._routing_signature_collection_only = True
    parallel_model = MMDataParallel(model, device_ids=[args.device_id])

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            parallel_model(return_loss=True, **batch)
            if args.log_interval > 0 and (batch_index + 1) % args.log_interval == 0:
                instance_count = int(model.base_routing_counts[0].sum().item())
                print(
                    f'Collected {batch_index + 1}/{len(data_loader)} batches; '
                    f'SAR GT instances counted for layer 0: {instance_count}',
                    flush=True)

    model._routing_signature_collection_only = False
    model.set_base_signature_collection(False)
    signatures, valid = model._base_signatures()
    counts = model.base_routing_counts
    if not bool(valid.all()):
        raise RuntimeError('Some base classes have no SAR routing signatures.')
    if (model.base_visual_counts.numel() == 0 or
            not bool((model.base_visual_counts > 0).all())):
        raise RuntimeError(
            'Some base classes have no SAR visual prototypes. Check that the '
            'Ours2 backbone returns spatial feature maps.')

    _checkpoint_with_model_state(
        args.checkpoint, model, args.output, num_images=len(dataset))
    print(f'Collected signatures for classes: {model.base_routing_classes}')
    print(f'Per-layer instance counts: {counts.cpu().tolist()}')
    print(f'Visual prototype counts: {model.base_visual_counts.cpu().tolist()}')
    print(f'Saved signature checkpoint: {args.output}')


if __name__ == '__main__':
    main()
