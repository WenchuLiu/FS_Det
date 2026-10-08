# Copyright (c) OpenMMLab. All rights reserved.
from itertools import chain
from functools import partial
from timm.models.layers import DropPath
from collections import OrderedDict

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.fft

from torch.distributions.normal import Normal

from mmengine.model import ModuleList, Sequential
from mmengine.logging import MMLogger 
from mmengine.runner.checkpoint import CheckpointLoader

from mmcv.cnn import (build_activation_layer,
                      constant_init, trunc_normal_init)
import torch
import torch.nn as nn
from ..builder import ROTATED_BACKBONES
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, trunc_normal_

from typing import Sequence, List, Dict
from .convnext_moe_origin import FFN
from .utils import build_LayerNorm2d_layer
import copy
# 引入上面定义的 Gate
from .gates import MMSplitGate, OriginGate, FocusGate, ShareGate, SharedExpertGate
import math

GATES = dict(
    mmsplitgate=MMSplitGate,
    focusgate=FocusGate,
    origingate=OriginGate,
    sharegate=ShareGate,
    sharedexpertgate=SharedExpertGate
)


class AdaptiveFFTFrequencyDecoupler(nn.Module):
    """
    自适应动态阈值 RFFT 解耦 (Modality-Agnostic Version)
    """
    def __init__(self, dim): 
        super().__init__()
        self.dim = dim
        rdim = self.get_reduction_dim(dim)

        self.stat_encoder = nn.Sequential(
            nn.Linear(3, rdim),
            nn.GELU(),
            nn.Linear(rdim, rdim // 2),
            nn.GELU(),
        )
        self.mask_predictor = nn.Linear(rdim // 2, 2)
        
        self.alpha_h = torch.nn.Parameter(torch.tensor(0.5))
        self.alpha_w = torch.nn.Parameter(torch.tensor(0.5))
        
        # 针对 SAR 特征的开关
        self.use_log_transform = False 

    def get_reduction_dim(self, dim):
        if dim < 8:
            return max(2, dim)
        log_dim = math.log2(dim)
        reduction = max(2, int(dim // log_dim))
        return reduction

    def _extract_freq_stats(self, magnitude):
        """提取稳定的频域统计特征"""
        B, C, H, W_r = magnitude.shape
        
        # 1. 平均幅度（归一化）
        mean_val = magnitude.mean(dim=[1, 2, 3]) / (magnitude.max() + 1e-8)
        
        # 2. 高频能量比（核心指标）
        y = torch.arange(H, device=magnitude.device).view(1, 1, -1, 1) - H // 2
        x = torch.arange(W_r, device=magnitude.device).view(1, 1, 1, -1)
        dist = torch.sqrt((y / (H//2))**2 + (x / (W_r-1))**2)  # 归一化距离 [0, 1]
        
        high_freq_mask = (dist > 0.5).float()
        total_energy = magnitude.pow(2).sum(dim=[1, 2, 3]) + 1e-8
        high_energy = (magnitude.pow(2) * high_freq_mask).sum(dim=[1, 2, 3])
        high_freq_ratio = (high_energy / total_energy).clamp(0, 1)
        
        # 3. 频谱集中度（可选）
        prob_dist = magnitude.reshape(B, -1)
        prob_dist = prob_dist / (prob_dist.sum(dim=1, keepdim=True) + 1e-8)
        # 使用 stable log
        spectral_concentration = (prob_dist * torch.log(prob_dist + 1e-8)).sum(dim=1).neg()
        
        return torch.stack([mean_val, high_freq_ratio, spectral_concentration], dim=1)  # [B, 3]

    def forward(self, x):
        x_perm = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x_perm.shape
        
        # 可选：对数变换
        if self.use_log_transform:
            x_perm = torch.sign(x_perm) * torch.log1p(torch.abs(x_perm))
        
        # 1. RFFT 变换
        fft = torch.fft.rfft2(x_perm, norm='backward')
        fft_shift = torch.fft.fftshift(fft, dim=-2) 
        
        # 提取特征并预测阈值
        freq_stats = self._extract_freq_stats(fft_shift) 
        stat_embed = self.stat_encoder(freq_stats) 
        raw_threshold = self.mask_predictor(stat_embed).sigmoid() 
        
        blended_threshold_h = self.alpha_h * raw_threshold[:, 0] + (1 - self.alpha_h) * raw_threshold[:, 1]
        blended_threshold_w = self.alpha_w * raw_threshold[:, 0] + (1 - self.alpha_w) * raw_threshold[:, 1]
        
        # 🔥 关键修复：构造可微的软掩码 (Differentiable Soft Mask)
        _, _, _, W_r = fft_shift.shape
        
        # 生成坐标网格
        y_grid = torch.arange(H, device=fft_shift.device).view(1, 1, -1, 1) - H // 2
        x_grid = torch.arange(W_r, device=fft_shift.device).view(1, 1, 1, -1)
        
        # 将相对阈值映射到绝对频域尺度 (保持张量运算，坚决不用 .int() 或 .item())
        # reshape 为 [B, 1, 1, 1] 以便利用广播机制 (Broadcasting)
        h_thresh = (blended_threshold_h * (H / 2)).view(B, 1, 1, 1)
        w_thresh = (blended_threshold_w * W_r).view(B, 1, 1, 1)
        
        # 使用带温度系数的 Sigmoid 构造掩码 (temperature 越大，边缘越锐利)
        temperature = 10.0
        # H 维度：中心低频，向两边衰减
        mask_y = torch.sigmoid(temperature * (h_thresh - torch.abs(y_grid)))
        # W 维度：rfft 只有一半，低频在左侧 (0 索引)，向右侧衰减
        mask_x = torch.sigmoid(temperature * (w_thresh - x_grid))
        
        # 最终掩码 [B, 1, H, W_r]
        mask = mask_y * mask_x
            
        # 3. 分离频率
        fft_low = fft_shift * mask
        fft_high = fft_shift * (1 - mask)
        
        # 4. iRFFT 还原
        fft_low_unshift = torch.fft.ifftshift(fft_low, dim=-2)
        fft_high_unshift = torch.fft.ifftshift(fft_high, dim=-2)
        
        x_low = torch.fft.irfft2(fft_low_unshift, s=(H, W), norm='backward')
        x_high = torch.fft.irfft2(fft_high_unshift, s=(H, W), norm='backward')
        
        if self.use_log_transform:
            x_low = torch.sign(x_low) * torch.expm1(torch.abs(x_low))
            x_high = torch.sign(x_high) * torch.expm1(torch.abs(x_high))
            
        return x_low.permute(0, 2, 3, 1), x_high.permute(0, 2, 3, 1)


class FD_MoE(BaseModule):
    """
    频域解耦 MoE (FD-MoE) - 规范化版
    结合了动态模块配置与直接相加的极简融合机制。
    """
    def __init__(self, 
                 in_channels: int, 
                 mid_channels: int, 
                 moe_args: List[dict], 
                 pw_conv: nn.Module = nn.Conv2d, 
                 act_cfg: nn.Module = nn.GELU, 
                 use_grn: bool = False):
        super().__init__()
        self.in_channels = in_channels
        
        # 必须确保有两个配置，分别对应 [低频, 高频]
        # 【防御性编程】使用深拷贝，防止 config 列表被意外污染
        if len(moe_args) == 1:
            self.moe_args = [copy.deepcopy(moe_args[0]), copy.deepcopy(moe_args[0])]
        else:
            self.moe_args = copy.deepcopy(moe_args[:2])

        self.decoupler = AdaptiveFFTFrequencyDecoupler(in_channels)
        
        # 按照规范模板，使用 getattr / setattr 动态挂载
        for i, moe_arg in enumerate(self.moe_args):
            mid_scale = moe_arg.get('mid_scale', 1.0)
            expert_mid_channels = int(mid_channels * mid_scale)
            
            # 🔥【关键修复 1】必须获取共享专家的数量，并相加得到真实总数！
            n_routed = moe_arg.get('n_routed_experts', 4)
            n_shared = moe_arg.get('n_shared_experts', 0)
            n_total_experts = n_routed + n_shared
            
            gate_type = moe_arg.get('type', 'origingate').lower()
            
            # 动态构建 Gate
            gate_func = GATES[gate_type]
            # 🔥【关键修复 2】传入 moe_arg 的深拷贝，防止 gate 内部的 pop() 毁掉原字典
            gate = gate_func(in_channels, copy.deepcopy(moe_arg))
            
            # 动态构建 Experts
            experts = nn.ModuleList([
                FFN(self.in_channels, 
                    expert_mid_channels, 
                    pw_conv, 
                    act_cfg, 
                    use_grn)
                for _ in range(n_total_experts) # 🔥 这里必须用 n_total_experts 总数！
            ])
            
            # 🔥 修复：强制与 init_weights 中的 expected keys 保持一致
            gate_name = 'gate_low' if i == 0 else 'gate_high'
            expert_name = 'experts_low' if i == 0 else 'experts_high'
            
            setattr(self, gate_name, gate)
            setattr(self, expert_name, experts)

    def forward(self, x: torch.Tensor, batch_indices=None):
        # 1. 频域解耦
        x_low, x_high = self.decoupler(x)
        
        loss_low, out_low = self.gate_low(x_low, self.experts_low, batch_indices)
        
        # 3. 高频分支处理 (对应 Index 1)
        loss_high, out_high = self.gate_high(x_high, self.experts_high, batch_indices)

        # 5. 特征直接相加融合
        out = out_low + out_high
        
        loss_dict = {
            'gate_loss': loss_low + loss_high,
        }
        
        return out, loss_dict


class ConvNeXtBlock(BaseModule):
    def __init__(self,
                 in_channels,
                 dw_conv_cfg=dict(kernel_size=7, 
                                  padding=3),
                 norm_cfg=dict(type='LN2d', 
                               eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 mlp_ratio=4.,
                 linear_pw_conv=True,
                 moe_args:List[dict]=[],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, groups=in_channels, **dw_conv_cfg)
        self.linear_pw_conv = linear_pw_conv
        self.norm = build_LayerNorm2d_layer(norm_cfg, in_channels)
        mid_channels = int(mlp_ratio * in_channels)
        
        if self.linear_pw_conv:
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)
        
        # --- FD-MoE 判断逻辑 ---
        if len(moe_args) > 0:
            self.ffn = FD_MoE(in_channels, mid_channels, moe_args, pw_conv, act_cfg, use_grn)
        else:
            self.ffn = FFN(in_channels, mid_channels, pw_conv, act_cfg, use_grn)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, batch_indices=None):
        def _inner_forward(x, batch_indices):
            shortcut = x
            x = self.depthwise_conv(x)
            
            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1).contiguous() # [N, C, H, W] -> [N, H, W, C]
                x = self.norm(x, data_format='channel_last')
                if isinstance(self.ffn, FD_MoE):
                    x, loss = self.ffn(x, batch_indices)
                else:
                    x = self.ffn(x)
                    loss = torch.tensor(0., device=x.device)
                x = x.permute(0, 3, 1, 2).contiguous()
            else:
                x = self.norm(x, data_format='channel_first')
                # 假设 Conv2d 模式下 FD_MoE 内部处理维度 (目前 FD_MoE 默认是 channel last 逻辑)
                # 为了安全，这里做一个 permute
                if isinstance(self.ffn, FD_MoE):
                    x = x.permute(0, 2, 3, 1)
                    x, loss = self.ffn(x, batch_indices)
                    x = x.permute(0, 3, 1, 2)
                else:
                    x = self.ffn(x)
                    loss = torch.tensor(0., device=x.device)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))
            x = shortcut + self.drop_path(x)
            return x, loss

        if self.with_cp and x.requires_grad:
            x, loss = cp.checkpoint(_inner_forward, x, batch_indices)
        else:
            x, loss = _inner_forward(x, batch_indices)
        return x, loss


@ROTATED_BACKBONES.register_module()
class ConvNeXt_FD_MoE(BaseModule):
    arch_settings = {
        'tiny': {'depths': [3, 3, 9, 3], 'channels': [96, 192, 384, 768]},
        'small': {'depths': [3, 3, 27, 3], 'channels': [96, 192, 384, 768]},
        'base': {'depths': [3, 3, 27, 3], 'channels': [128, 256, 512, 1024]},
        'large': {'depths': [3, 3, 27, 3], 'channels': [192, 384, 768, 1536]},
    }

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 stem_patch_size=4,
                 norm_cfg=dict(type='LN2d', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 linear_pw_conv=True,
                 use_grn=False,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 out_indices=[0, 1, 2, 3],
                 moe_indices=[[],[],[],[]],
                 moe_args: List[Dict] = [],
                 frozen_stages=-1,
                 unfreeze_gate=False,
                 freeze_backbone_except_moe=False,
                 gap_before_final_norm=False,
                 with_cp=False,
                 return_gate_loss=True,
                 init_cfg=[
                     dict(type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
                     dict(type='Constant', layer=['LayerNorm'], val=1., bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            arch = self.arch_settings[arch]
        
        self.depths = arch['depths']
        self.channels = arch['channels']
        self.num_stages = len(self.depths)
        self.out_indices = out_indices
        self.moe_indices = moe_indices
        self.moe_args = moe_args
        
        self.frozen_stages = frozen_stages
        self.unfreeze_gate = unfreeze_gate
        self.freeze_backbone_except_moe = freeze_backbone_except_moe
        self.gap_before_final_norm = gap_before_final_norm
        self.return_gate_loss = return_gate_loss

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        block_idx = 0

        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, self.channels[0], kernel_size=stem_patch_size, stride=stem_patch_size),
            build_LayerNorm2d_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        self.stages = ModuleList()
        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_LayerNorm2d_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(self.channels[i - 1], channels, kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)
            
            cur_moe_idx = [list(range(depth))[q] for q in self.moe_indices[i] if q < depth]
            
            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    moe_args=moe_args if j in cur_moe_idx else [],
                    linear_pw_conv=linear_pw_conv,
                    layer_scale_init_value=layer_scale_init_value,
                    use_grn=use_grn,
                    with_cp=with_cp) for j in range(depth)
            ])
            block_idx += depth
            self.stages.append(stage)

            if i in self.out_indices:
                norm_layer = build_LayerNorm2d_layer(norm_cfg, channels)
                self.add_module(f'norm{i}', norm_layer)
            
        self.downsample_layers[0] = nn.Sequential(build_LayerNorm2d_layer(norm_cfg, self.channels[0]))

        self.dataset_stems = nn.Conv2d(
            in_channels,
            self.channels[0],
            kernel_size=stem_patch_size,
            stride=stem_patch_size
        )
        
        self._freeze_stages()

    def forward(self, inputs, dataset_names=['single']):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        
        if len(dataset_names) > len(inputs):
            dataset_names = dataset_names[:len(inputs)]
            
        batch_indices = {}
        current_idx = 0
        for i, inp in enumerate(inputs):
            B = inp.shape[0]
            batch_indices[dataset_names[i]] = list(range(current_idx, current_idx + B))
            current_idx += B
            
        inputs = torch.cat(inputs, dim=0)
        features = self.dataset_stems(inputs)
        
        outs = []
        gate_losses = []
        
        for i, stage in enumerate(self.stages):
            features = self.downsample_layers[i](features) # 加上这个修复
            
            for each_layer in stage:
                features, gate_loss = each_layer(features, batch_indices)
                if gate_loss is not None:
                    gate_losses.append(gate_loss)
            
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = features.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(features))
                    
        if len(gate_losses) > 0:
            total_gate_loss = 0.0
            moe_count = 0
            
            for l in gate_losses:
                if isinstance(l, dict):
                    total_gate_loss = total_gate_loss + l['gate_loss']
                    moe_count += 1
                else:
                    pass
            
            if moe_count > 0:
                avg_gate_loss = total_gate_loss / moe_count
                final_loss = {
                    'gate_loss': avg_gate_loss,
                }
            else:
                final_loss = features.sum() * 0
        else:
            final_loss = features.sum() * 0
        
        if self.return_gate_loss:
            return tuple(outs), final_loss
        else:
            return tuple(outs)

    def _freeze_stages(self):
        # 1. 基础冻结逻辑
        if self.frozen_stages >= 0:
            if hasattr(self, 'dataset_stems'):
                self.dataset_stems.eval()
                for param in self.dataset_stems.parameters():
                    param.requires_grad = False
            
            for i in range(self.frozen_stages):
                downsample_layer = self.downsample_layers[i]
                stage = self.stages[i]
                downsample_layer.eval()
                stage.eval()
                for param in chain(downsample_layer.parameters(), stage.parameters()):
                    param.requires_grad = False
                
                # 2. 解冻 Gate 逻辑
                if self.unfreeze_gate:
                    for name, module in stage.named_modules():
                        # 这里检查是否是我们定义的 Gate 类型
                        if isinstance(module, (OriginGate, ShareGate)):
                            module.train()
                            for param in module.parameters():
                                param.requires_grad = True
        
        # 3. 冻结除 MoE 外的所有参数
        if self.freeze_backbone_except_moe:
            for name, module in self.named_modules():
                # 默认所有都冻结
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
            
            # 重新解冻 MoE 相关部分
            for name, module in self.named_modules():
                if isinstance(module, FD_MoE): # 或者 OriginGate, ShareGate
                    module.train()
                    for param in module.parameters():
                        param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()

    def init_weights(self):
        # --- 修复并增强的初始化逻辑 ---
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            return

        ckpt = CheckpointLoader.load_checkpoint(self.init_cfg.checkpoint, logger=logger, map_location='cpu')
        _state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
        
        state_dict = OrderedDict()
        for k, v in _state_dict.items():
            if k.startswith('backbone.'): k = k[9:]
            
            if 'downsample_layers.0.0' in k:
                state_dict[k.replace('downsample_layers.0.0', 'dataset_stems')] = v
            elif 'downsample_layers.0.1' in k:
                new_k = k.replace('downsample_layers.0.1', 'downsample_layers.0.0')
                state_dict[new_k] = v
            elif 'pointwise_conv' in k or 'grn' in k:
                stage_splits = k.split('.')
                try:
                    stage_ind = int(stage_splits[1])
                    block_ind = int(stage_splits[2])
                except ValueError:
                    state_dict[k] = v
                    continue
                
                # 判断当前层是否是 MoE 层
                cur_moe_idx = [list(range(self.depths[stage_ind]))[q] for q in self.moe_indices[stage_ind]]
                if block_ind in cur_moe_idx:
                    if self.moe_args:
                        shared_cfg = self.moe_args[0]
                        n_shared = shared_cfg.get('n_routed_experts', 2)
                        for i in range(n_shared):
                            new_k = k.replace('pointwise_conv', f'ffn.experts_low.{i}.pointwise_conv') \
                                     .replace('grn', f'ffn.experts_low.{i}.grn')
                            if "grn.gamma" in new_k or "grn.beta" in new_k: v = v.reshape(-1)
                            state_dict[new_k] = v
                    if self.moe_args:
                        origin_cfg = self.moe_args[1] if len(self.moe_args) > 1 else self.moe_args[0]
                        n_origin = origin_cfg.get('n_routed_experts', 6)
                        for i in range(n_origin):
                            new_k = k.replace('pointwise_conv', f'ffn.experts_high.{i}.pointwise_conv') \
                                     .replace('grn', f'ffn.experts_high.{i}.grn')
                            if "grn.gamma" in new_k or "grn.beta" in new_k: v = v.reshape(-1)
                            state_dict[new_k] = v
                else:
                    new_k = k.replace('pointwise_conv', 'ffn.pointwise_conv').replace('grn', 'ffn.grn')
                    if "grn.gamma" in new_k or "grn.beta" in new_k: v = v.reshape(-1)
                    state_dict[new_k] = v
            else:
                state_dict[k] = v
        
        # 加载
        warnings = self.load_state_dict(state_dict, strict=False)
        if warnings.missing_keys or warnings.unexpected_keys:
            logger.warn(f'Warnings when loading state_dict: {warnings}')
        