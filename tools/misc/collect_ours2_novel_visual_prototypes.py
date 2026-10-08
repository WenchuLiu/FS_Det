#!/usr/bin/env python
"""Collect a novel-class visual prototype from the SAR few-shot support set."""

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
        description='Collect SAR novel-class visual prototypes for Ours2')
    parser.add_argument('config', help='Ours2 few-shot config for the shot count')
    parser.add_argument('checkpoint', help='Ours/Ours2 few-shot initialization checkpoint')
    parser.add_argument('--output', required=True,
                        help='prototype file path to write')
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--samples-per-gpu', type=int, default=4)
    parser.add_argument('--workers-per-gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def _build_sar_only_dataset(cfg):
    train_cfg = copy.deepcopy(cfg.data.train)
    if train_cfg.get('type') != 'ConcatDataset':
        raise ValueError('Expected cfg.data.train to be a ConcatDataset.')
    datasets = train_cfg.get('datasets', [])
    if len(datasets) != 1:
        raise ValueError(
            'Novel visual prototype collection requires the SAR-only '
            f'few-shot config; found {len(datasets)} train datasets.')
    return build_dataset(train_cfg)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Visual prototype collection requires CUDA.')
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
    routing_cfg.prior_mode = 'visual_topk'
    routing_cfg.collect_base_signatures = False
    routing_cfg.enable_transfer = False
    routing_cfg.pop('novel_visual_prototypes_path', None)

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
    if not hasattr(model, 'set_novel_visual_prototype_collection'):
        raise TypeError('The config must build TriSourceDetectorV16Ours2.')
    model.reset_novel_visual_prototypes()
    model.set_novel_visual_prototype_collection(True)
    parallel_model = MMDataParallel(model, device_ids=[args.device_id])

    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            parallel_model(return_loss=True, **batch)
            print(
                f'Collected novel visual prototypes from '
                f'{batch_index + 1}/{len(data_loader)} batches', flush=True)

    model.set_novel_visual_prototype_collection(False)
    payload = model.get_novel_visual_prototypes()
    payload.update(
        source_modality='sar',
        num_images=len(dataset),
        checkpoint=os.path.abspath(args.checkpoint),
        collection_mode='eval_no_grad',
        feature='final_backbone_gt_region_mean',
    )
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = args.output + '.tmp'
    torch.save(payload, tmp_path)
    os.replace(tmp_path, args.output)
    print(
        f"Saved SAR support prototypes for {payload['class_names']} to "
        f'{args.output}; counts={payload["counts"].tolist()}')


if __name__ == '__main__':
    main()
