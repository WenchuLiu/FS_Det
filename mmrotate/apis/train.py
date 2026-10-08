# Copyright (c) OpenMMLab. All rights reserved.
# Copied from mmdet, only modified `get_root_logger`.
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataset, replace_ImageToTensor)

from mmrotate.utils import compat_cfg, find_latest_checkpoint, get_root_logger
from mmrotate.datasets import build_dataloader

def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    logger.info('Startup: building training dataloader')
    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]
    logger.info('Startup: training dataloader ready')

    # put model on gpus
    logger.info('Startup: moving detector to GPU')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    logger.info('Startup: building optimizer')
    optimizer = build_optimizer(model, cfg.optimizer)
    logger.info('Startup: optimizer ready')

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
            
        for data_key, evaluation_key in (
                ('val', 'evaluation'), ('val_2', 'evaluation2')):
            val_cfg = cfg.data.get(data_key)
            if val_cfg is None:
                continue
            if val_dataloader_args['samples_per_gpu'] > 1:
                val_cfg.pipeline = replace_ImageToTensor(val_cfg.pipeline)
            val_dataset = build_dataset(val_cfg, dict(test_mode=True))
            val_dataloader = build_dataloader(
                val_dataset, **val_dataloader_args)

            eval_cfg = cfg.get(evaluation_key, {})
            eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
            eval_hook = DistEvalHook if distributed else EvalHook
            runner.register_hook(
                eval_hook(val_dataloader, **eval_cfg), priority='LOW')
            
    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    logger.info('Startup: entering training loop')
    runner.run(data_loaders, cfg.workflow)
