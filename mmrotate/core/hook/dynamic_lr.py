"""Loss-aware LR scheduling with bounded, smoothed multipliers."""
import math
from typing import List, Optional, Union

import mmcv
import torch
import torch.nn.functional as F
from mmcv.runner.hooks import LrUpdaterHook
from mmcv.runner.hooks.hook import HOOKS


class EMA_meter:
    def __init__(self, beta):
        self.beta = beta
        self.ema = None
        self.steps = 0

    def update(self, value):
        if not math.isfinite(value) or value < 0:
            return
        if self.ema is None:
            self.ema = value
        else:
            self.ema = (1 - self.beta) * self.ema + self.beta * value
        self.steps += 1

    def get(self):
        return self.ema if self.ema is not None else 1e-3


@HOOKS.register_module()
class DynamicLrUpdaterHook(LrUpdaterHook):
    """Step LR with loss-aware multipliers applied before optimizer updates.

    ``extra_args`` retains T, b, ema, backbone_policy and head_policy. Safety
    options are min_lr_scale (0.1), max_lr_scale (2.0), lr_scale_ema (0.1),
    max_loss_ratio (10.0), and loss_eps (1e-8). Multipliers are relative to
    the current step schedule, not accumulated into the previous LR.
    Invalid loss observations reset multipliers to one without updating EMA.
    This hook cannot repair model/optimizer state that is already non-finite.
    """

    def __init__(self,
                 step: Union[int, List[int]],
                 gamma: float = 0.1,
                 min_lr: Optional[float] = None,
                 extra_args: Optional[dict] = None,
                 reweight_losses: Optional[dict] = None,
                 **kwargs) -> None:
        if isinstance(step, list):
            assert mmcv.is_list_of(step, int)
            assert all(s > 0 for s in step)
            assert step == sorted(step)
        elif isinstance(step, int):
            assert step > 0
        else:
            raise TypeError('"step" must be a list or integer')
        self.step = step
        self.gamma = gamma
        self.min_lr = min_lr
        self.extra_args = dict(T=5, b=0.5, ema=0.005,
                               backbone_policy='min', head_policy='normal')
        self.extra_args.update(extra_args or {})
        if reweight_losses is None:
            reweight_losses = {
                'sar_loss_cls': 'sar_bbox_head',
                'sar_loss_bbox': 'sar_bbox_head',
                'sar_loss_dfl': 'sar_bbox_head',
                'rgb_loss_rpn_cls': 'rgb_rpn_head',
                'rgb_loss_rpn_bbox': 'rgb_rpn_head',
                'rgb_loss_cls': 'rgb_roi_head',
                'rgb_loss_bbox': 'rgb_roi_head',
                'ifr_loss_rpn_cls': 'ifr_rpn_head',
                'ifr_loss_rpn_bbox': 'ifr_rpn_head',
                'ifr_loss_cls': 'ifr_roi_head',
                'ifr_loss_bbox': 'ifr_roi_head',
            }
        self.reweight_losses = dict(reweight_losses)
        self.T = float(self.extra_args['T'])
        self.b = float(self.extra_args['b'])
        self.min_lr_scale = float(self.extra_args.get('min_lr_scale', 0.1))
        self.max_lr_scale = float(self.extra_args.get('max_lr_scale', 2.0))
        self.lr_scale_ema = float(self.extra_args.get('lr_scale_ema', 0.1))
        self.max_loss_ratio = float(self.extra_args.get('max_loss_ratio', 10.0))
        self.loss_eps = float(self.extra_args.get('loss_eps', 1e-8))
        values = (self.T, self.b, self.min_lr_scale, self.max_lr_scale,
                  self.lr_scale_ema, self.max_loss_ratio, self.loss_eps,
                  self.extra_args['ema'], gamma)
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Dynamic LR options must be finite')
        if (self.T <= 0 or self.loss_eps <= 0 or self.max_loss_ratio <= 0
                or not 0 < self.lr_scale_ema <= 1
                or not 0 < self.extra_args['ema'] <= 1
                or not 0 < self.min_lr_scale <= 1 <= self.max_lr_scale
                or not 0 < gamma <= 1):
            raise ValueError('Invalid dynamic LR bounds or smoothing options')
        if min_lr is not None and (not math.isfinite(min_lr) or min_lr < 0):
            raise ValueError('min_lr must be finite and non-negative')
        self.history_ema_loss = [EMA_meter(self.extra_args['ema'])
                                 for _ in self.reweight_losses]
        # Match EMA by loss name; log order and missing modalities may vary.
        self._loss_meters = dict(zip(self.reweight_losses, self.history_ema_loss))
        self.param_groups_param_names_mapping = {}
        self._lr_scales = []
        super().__init__(**kwargs)

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter
        exp = (progress // self.step if isinstance(self.step, int)
               else sum(progress >= s for s in self.step))
        lr = base_lr * self.gamma**exp
        return max(lr, self.min_lr) if self.min_lr is not None else lr

    @staticmethod
    def _loss_value(value):
        if isinstance(value, (list, tuple)):
            return sum(DynamicLrUpdaterHook._loss_value(v) for v in value)
        if isinstance(value, torch.Tensor):
            value = value.detach().double().sum().item()
        return float(value)

    def get_dynamic_lr(self, runner):
        regular_lr = self.get_regular_lr(runner)
        losses = getattr(runner, 'outputs', {}).get('log_vars', {})
        names = [name for name in self.reweight_losses if name in losses]
        values = [self._loss_value(losses[name]) for name in names]
        if not names or any(not math.isfinite(v) or v < 0 for v in values):
            self._lr_scales = [1.0] * len(regular_lr)
            if names and hasattr(runner, 'logger'):
                runner.logger.warning(
                    'Dynamic LR: invalid loss observation; using scheduled LR '
                    'and skipping EMA update. Check loss/gradients/model state.')
            return regular_lr

        current = torch.tensor(values, dtype=torch.float64)
        history = torch.tensor([self._loss_meters[n].get() for n in names],
                               dtype=torch.float64)
        ready = all(self._loss_meters[n].steps >= self.warmup_iters
                    and self._loss_meters[n].steps > 0 for n in names)
        warming_up = self.warmup is not None and runner.iter < self.warmup_iters
        weights = torch.ones(len(names), dtype=torch.float64)
        if ready and not warming_up and self.extra_args['head_policy'] != 'None':
            numerator, denominator = ((current, history)
                                      if self.extra_args['head_policy'] == 'reverse'
                                      else (history, current))
            ratios = (numerator / denominator.clamp_min(self.loss_eps))
            ratios = ratios.clamp(0, self.max_loss_ratio)
            weights = len(names) * F.softmax(ratios / self.T, dim=-1)

        subnet_scales = {}
        for subnet in set(self.reweight_losses.values()):
            indices = [i for i, n in enumerate(names)
                       if self.reweight_losses[n] == subnet]
            # An absent modality targets its scheduled LR.
            subnet_scales[subnet] = (weights[indices].mean().item()
                                     if indices else 1.0)
        policy = self.extra_args['backbone_policy']
        scales = list(subnet_scales.values())
        shared = 1.0
        if ready and not warming_up:
            if policy == 'min':
                shared = min(scales)
            elif policy == 'avg':
                shared = sum(scales) / len(scales)
            elif policy == 'max':
                shared = max(scales)
            elif policy in ('kl', 'sigmoid_kl'):
                # log_softmax avoids log(0) when a probability underflows.
                log_history = F.log_softmax(history, dim=-1)
                log_current = F.log_softmax(current, dim=-1)
                divergence = (log_history.exp() *
                              (log_history - log_current)).mean().clamp_min(0)
                if policy == 'kl':
                    shared = 1 + (1 - divergence.item()) / math.sqrt(self.T)
                else:
                    shared = (2 * torch.sigmoid(
                        (1 - divergence - self.b) * self.T)).item()

        for name, value in zip(names, values):
            self._loss_meters[name].update(value)

        if len(self._lr_scales) != len(regular_lr):
            self._lr_scales = [1.0] * len(regular_lr)
        for index in range(len(regular_lr)):
            param_names = self.param_groups_param_names_mapping.get(index, ())
            targets = []
            for name in param_names:
                matches = [v for subnet, v in subnet_scales.items() if subnet in name]
                targets.append(matches[0] if matches else shared)
            target = sum(targets) / len(targets) if targets else shared
            if not math.isfinite(target):
                target = 1.0
            target = min(self.max_lr_scale, max(self.min_lr_scale, target))
            self._lr_scales[index] += self.lr_scale_ema * (
                target - self._lr_scales[index])
        return self._apply_scales(regular_lr)

    def _apply_scales(self, regular_lr):
        result = [lr * scale for lr, scale in zip(regular_lr, self._lr_scales)]
        if self.min_lr is not None:
            result = [max(lr, self.min_lr) if base > 0 else 0.0
                      for lr, base in zip(result, self.base_lr)]
        return result

    def before_run(self, runner):
        if isinstance(runner.optimizer, dict):
            raise TypeError('Dynamic LR requires a single optimizer')
        # Parameter identity is reliable even for equal-valued/zero parameters.
        names = {id(param): name for name, param in runner.model.named_parameters()}
        for index, group in enumerate(runner.optimizer.param_groups):
            self.param_groups_param_names_mapping[index] = tuple(
                names[id(param)] for param in group['params'] if id(param) in names)
        super().before_run(runner)
        if any(not math.isfinite(lr) or lr < 0 for lr in self.base_lr):
            raise ValueError('Base learning rates must be finite and non-negative')
        self._lr_scales = [1.0] * len(self.base_lr)

    def before_train_iter(self, runner):
        self.regular_lr = self.get_regular_lr(runner)
        if self.warmup is not None and runner.iter < self.warmup_iters:
            lr = self.get_warmup_lr(runner.iter)
        else:
            lr = self._apply_scales(self.regular_lr)
        # Apply previous training observations immediately before this update.
        # Do not consume runner.outputs here: validation can overwrite it.
        self.dynamic_lr = lr
        self._set_lr(runner, lr)

    def after_train_iter(self, runner):
        # Store factors for the next iteration; keep the schedule aligned with
        # the next iteration's step/epoch and preserve the current logged LR.
        self.get_dynamic_lr(runner)
