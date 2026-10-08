"""Regression checks using the real MMCV hook and PyTorch optimizer."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

spec = importlib.util.spec_from_file_location(
    'dynamic_lr_under_test',
    Path(__file__).resolve().parents[2] / 'mmrotate/core/hook/dynamic_lr.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DynamicLrUpdaterHook = module.DynamicLrUpdaterHook


class DynamicLrTests(unittest.TestCase):
    def make_runner(self, **options):
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(1, 1, bias=False)
        model.head_a = torch.nn.Linear(1, 1, bias=False)
        model.head_b = torch.nn.Linear(1, 1, bias=False)
        for param in model.parameters():
            param.data.zero_()  # Equal values must not confuse identity mapping.
        optimizer = torch.optim.SGD([
            {'params': list(model.backbone.parameters()), 'lr': 0.01},
            {'params': list(model.head_a.parameters()), 'lr': 0.01},
            {'params': list(model.head_b.parameters()), 'lr': 0.0},
        ])
        runner = SimpleNamespace(model=model, optimizer=optimizer, iter=0, epoch=0)
        args = dict(step=[10], by_epoch=False,
                    reweight_losses={'a': 'head_a', 'b': 'head_b'})
        args.update(options)
        hook = DynamicLrUpdaterHook(**args)
        hook.before_run(runner)
        return hook, runner

    def observe(self, hook, runner, losses):
        runner.outputs = {'log_vars': losses}
        hook.after_train_iter(runner)

    def test_dynamic_scales_survive_before_iter_and_decay_boundary(self):
        hook, runner = self.make_runner(extra_args={'backbone_policy': 'max'})
        self.observe(hook, runner, {'a': 1.0, 'b': 1.0})
        runner.iter = 9
        self.observe(hook, runner, {'a': 0.0, 'b': 1.0})
        runner.iter = 10
        hook.before_train_iter(runner)
        self.assertGreater(runner.optimizer.param_groups[1]['lr'], 0.001)
        self.assertLessEqual(runner.optimizer.param_groups[1]['lr'], 0.002)
        self.assertEqual(runner.optimizer.param_groups[2]['lr'], 0.0)
        self.assertEqual(hook.param_groups_param_names_mapping[0], ('backbone.weight',))

    def test_bad_losses_do_not_poison_ema_and_recover(self):
        hook, runner = self.make_runner()
        self.observe(hook, runner, {'a': 1.0, 'b': 2.0})
        for value in (float('nan'), float('inf'), -1.0):
            before = [(m.get(), m.steps) for m in hook.history_ema_loss]
            self.observe(hook, runner, {'a': value, 'b': 1.0})
            self.assertEqual(before, [(m.get(), m.steps) for m in hook.history_ema_loss])
            self.assertEqual(hook._lr_scales, [1.0, 1.0, 1.0])
        self.observe(hook, runner, {'a': 0.5, 'b': 1.0})
        hook.before_train_iter(runner)
        self.assertTrue(all(math.isfinite(g['lr']) for g in runner.optimizer.param_groups))

    def test_missing_losses_and_order_match_ema_by_name(self):
        hook, runner = self.make_runner()
        self.observe(hook, runner, {'b': 20.0, 'a': 10.0})
        self.observe(hook, runner, {'b': [10.0, 10.0]})
        self.assertEqual(hook._loss_meters['a'].get(), 10.0)
        self.assertEqual(hook._loss_meters['a'].steps, 1)
        self.assertEqual(hook._loss_meters['b'].get(), 20.0)
        self.assertEqual(hook._loss_meters['b'].steps, 2)
        self.observe(hook, runner, {})
        self.assertEqual(hook._lr_scales, [1.0, 1.0, 1.0])

    def test_extreme_losses_all_policies_stay_bounded(self):
        for policy in ('min', 'avg', 'max', 'kl', 'sigmoid_kl'):
            for head_policy in ('normal', 'reverse', 'None'):
                hook, runner = self.make_runner(extra_args={
                    'backbone_policy': policy, 'head_policy': head_policy})
                for i in range(40):
                    runner.iter = i
                    hook.before_train_iter(runner)
                    for group, base in zip(runner.optimizer.param_groups, hook.base_lr):
                        scheduled = hook.get_lr(runner, base)
                        self.assertTrue(math.isfinite(group['lr']))
                        self.assertGreaterEqual(group['lr'], 0.1 * scheduled - 1e-15)
                        self.assertLessEqual(group['lr'], 2 * scheduled + 1e-15)
                    self.observe(hook, runner, {'a': 1e300 if i % 2 else 0.0,
                                                'b': 0.0 if i % 2 else 1e300})

    def test_warmup_and_actual_optimizer_update(self):
        hook, runner = self.make_runner(warmup='linear', warmup_iters=2,
                                        warmup_ratio=0.5)
        for i, expected_lr in enumerate((0.005, 0.0075, 0.01)):
            runner.iter = i
            hook.before_train_iter(runner)
            self.assertAlmostEqual(runner.optimizer.param_groups[0]['lr'], expected_lr)
            runner.optimizer.zero_grad()
            loss = sum((p - 1).square().sum() for p in runner.model.parameters())
            loss.backward()
            runner.optimizer.step()
            self.observe(hook, runner, {'a': loss.detach(), 'b': loss.detach()})
        self.assertTrue(all(torch.isfinite(p).all() for p in runner.model.parameters()))

    def test_multi_parameter_groups_and_epoch_schedule(self):
        hook, runner = self.make_runner(by_epoch=True)
        runner.optimizer.param_groups[0]['params'].append(runner.model.head_a.weight)
        hook.before_run(runner)
        self.assertEqual(len(hook.param_groups_param_names_mapping[0]), 2)
        runner.epoch = 10
        hook.before_train_epoch(runner)
        hook.before_train_iter(runner)
        self.assertAlmostEqual(runner.optimizer.param_groups[0]['lr'], 0.001)

    def test_invalid_options_fail_early(self):
        for args in ({'T': 0}, {'ema': 2}, {'max_lr_scale': float('inf')},
                     {'lr_scale_ema': 0}, {'min_lr_scale': -1}):
            with self.assertRaises(ValueError):
                self.make_runner(extra_args=args)


if __name__ == '__main__':
    unittest.main()
