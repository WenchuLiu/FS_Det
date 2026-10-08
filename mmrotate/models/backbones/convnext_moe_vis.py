# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from itertools import chain
from typing import Sequence
from functools import partial
from collections import OrderedDict

from typing import List

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

import numpy as np


from mmengine.model import ModuleList, Sequential
from mmengine.logging import MMLogger 
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.model import BaseModule

import torch
import torch.nn as nn
from ..builder import ROTATED_BACKBONES

from mmcv.runner import BaseModule
from timm.models.layers import DropPath, trunc_normal_

from .convnext_moe_origin import FFN

from .gates import MMSplitGate, OriginGate, MMSplitGateV2, ShareGate
from .utils import build_LayerNorm2d_layer


GATES = dict(
    mmsplitgate=MMSplitGate,
    mmsplitgatev2=MMSplitGateV2,
    origingate=OriginGate,
    sharegate=ShareGate
)


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, 
                 in_channels, 
                 mid_channels,
                 moe_args: List[dict],
                 pw_conv: nn.Module = nn.Conv2d,
                 act_cfg: nn.Module = nn.GELU,
                 use_grn: bool = False):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.in_channels = in_channels
        self.moe_args = moe_args
                
        for i, moe_arg in enumerate(moe_args):
            expert_mid_channels = int(mid_channels * moe_arg.mid_scale)
                        
            gate_func = GATES[moe_arg["type"]]
            gate = gate_func(in_channels, moe_arg)
            experts = nn.ModuleList([FFN(self.in_channels,
                                         expert_mid_channels,
                                         pw_conv,
                                         act_cfg,
                                         use_grn)
                                     for _ in range(moe_arg.n_routed_experts)])        
            setattr(self, f'gate_{i}', gate)
            setattr(self, f'expert_{i}', experts)
    
    
    def random_k_gating(self, features):
        for i, moe_arg in enumerate(self.moe_args):    
            experts = getattr(self, f'expert_{i}')
            idx = np.random.randint(0, moe_arg.n_routed_experts, (1, ))[0]  
            features = experts[idx](features)
        return features
    
    def forward(self, x: torch.Tensor, batch_indices=None) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.
            batch_indices (dict): Dictionary containing batch indices for each dataset.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        total_gate_loss = 0
        y = x
        for i in range(len(self.moe_args)):
            gate = getattr(self, f'gate_{i}')
            experts = getattr(self, f'expert_{i}')
            gate_loss, y = gate(y, experts, batch_indices)
            total_gate_loss = total_gate_loss + gate_loss
        return y, total_gate_loss / len(self.moe_args)


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
                 moe_cfgs:List[dict]=[],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 use_grn=False,
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.depthwise_conv = nn.Conv2d(in_channels, 
                                        in_channels,
                                        groups=in_channels,
                                        **dw_conv_cfg)
        self.linear_pw_conv = linear_pw_conv
        self.norm = build_LayerNorm2d_layer(norm_cfg, in_channels)
        mid_channels = int(mlp_ratio * in_channels) 
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)
        self.ffn = FFN(in_channels,mid_channels,pw_conv,act_cfg,use_grn)

        self.moe_cfgs = moe_cfgs
        if len(moe_cfgs) != 0:
            
            self.ffn = MoE(in_channels,
                           mid_channels,
                           moe_cfgs,
                           pw_conv=pw_conv,
                           use_grn=use_grn,
                           act_cfg=act_cfg)
        else:
            self.ffn = FFN(in_channels,mid_channels,pw_conv,act_cfg,use_grn)


        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        
        
        
    def forward(self, x, batch_indices=None):
        
        def _inner_forward(x, batch_indices):
            shortcut = x
            loss = None
            x = self.depthwise_conv(x)
            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                if len(self.moe_cfgs) > 0:
                    x, loss = self.ffn(x, batch_indices)
                else: 
                    x = self.ffn(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                if len(self.moe_cfgs) > 0:
                    x, loss = self.ffn(x)
                else: 
                    x = self.ffn(x)
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
class ConvNeXtMoE(BaseModule):
    arch_settings = {
        'atto': {
            'depths': [2, 2, 6, 2],
            'channels': [40, 80, 160, 320]
        },
        'femto': {
            'depths': [2, 2, 6, 2],
            'channels': [48, 96, 192, 384]
        },
        'pico': {
            'depths': [2, 2, 6, 2],
            'channels': [64, 128, 256, 512]
        },
        'nano': {
            'depths': [2, 2, 8, 2],
            'channels': [80, 160, 320, 640]
        },
        'tiny': {
            'depths': [3, 3, 9, 3],
            'channels': [96, 192, 384, 768]
        },
        'small': {
            'depths': [3, 3, 27, 3],
            'channels': [96, 192, 384, 768]
        },
        'base': {
            'depths': [3, 3, 27, 3],
            'channels': [128, 256, 512, 1024]
        },
        'swin_large': {
            'depths': [2,  2, 18,  2],
            'channels': [192, 384, 768, 1536]
        },
        'large': {
            'depths': [3, 3, 27, 3],
            'channels': [192, 384, 768, 1536]
        },
        'xlarge': {
            'depths': [3, 3, 27, 3],
            'channels': [256, 512, 1024, 2048]
        },
        'huge': {
            'depths': [3, 3, 27, 3],
            'channels': [352, 704, 1408, 2816]
        }
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
                 moe_indices = [[],[],[],[]],
                 moe_cfgs:List[dict] = [],
                 frozen_stages=0,
                 gap_before_final_norm=False,
                 with_cp=False,
                 
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ]):
        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'depths' in arch and 'channels' in arch, \
                f'The arch dict must have "depths" and "channels", ' \
                f'but got {list(arch.keys())}.'

        self.depths = arch['depths']
        self.channels = arch['channels']
        assert (isinstance(self.depths, Sequence)
                and isinstance(self.channels, Sequence)
                and len(self.depths) == len(self.channels)), \
            f'The "depths" ({self.depths}) and "channels" ({self.channels}) ' \
            'should be both sequence with the same length.'

        self.num_stages = len(self.depths)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 4 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        self.moe_indices = moe_indices
        self.moe_cfgs = moe_cfgs        
        
        self.frozen_stages = frozen_stages
        self.gap_before_final_norm = gap_before_final_norm

        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        block_idx = 0

        # 4 downsample layers between stages, including the stem layer.
        self.downsample_layers = ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.channels[0],
                kernel_size=stem_patch_size,
                stride=stem_patch_size),
            build_LayerNorm2d_layer(norm_cfg, self.channels[0]),
        )
        self.downsample_layers.append(stem)

        # 4 feature resolution stages, each consisting of multiple residual
        # blocks
        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            depth = self.depths[i]
            channels = self.channels[i]

            if i >= 1:
                downsample_layer = nn.Sequential(
                    build_LayerNorm2d_layer(norm_cfg, self.channels[i - 1]),
                    nn.Conv2d(
                        self.channels[i - 1],
                        channels,
                        kernel_size=2,
                        stride=2),
                )
                self.downsample_layers.append(downsample_layer)
            moe_id = [list(range(depth))[q] for q in self.moe_indices[i] if q < depth]
            stage = Sequential(*[
                ConvNeXtBlock(
                    in_channels=channels,
                    drop_path_rate=dpr[block_idx + j],
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    moe_cfgs = moe_cfgs if j in moe_id else [],
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

        self._freeze_stages()
        
        self.downsample_layers[0] = nn.Sequential(build_LayerNorm2d_layer(norm_cfg, self.channels[0]))
        
        self.dataset_stems = nn.Conv2d(in_channels,
                                       self.channels[0],
                                       kernel_size=stem_patch_size,
                                       stride=stem_patch_size)
        
    def forward(self, inputs, dataset_names=['single']):
        
        if len(dataset_names) == 1:
            inputs = [inputs] 
        
        batch_indices = {
            dataset_name: list(range(final_index, final_index + inputs[i].shape[0]))
            for i, dataset_name in enumerate(dataset_names)
            for final_index in [sum(inputs[j].shape[0] for j in range(i))]
        }
        
        inputs = torch.cat(inputs, dim=0)
        inputs = self.dataset_stems(inputs)
        
        gate_losses = []
        outs = []
        
        for i, stage in enumerate(self.stages): 
            inputs = self.downsample_layers[i](inputs)
            for each_layer in stage:
                inputs, gate_loss = each_layer(inputs, batch_indices)
                if gate_loss is not None:
                    gate_losses.append(gate_loss)  
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = inputs.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(inputs))
        if len(gate_losses) > 0:
            return tuple(outs), sum(gate_losses)/len(gate_losses)
        return tuple(outs)
        
    def _freeze_stages(self):
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(),
                               stage.parameters()):
                param.requires_grad = False

    def train(self, mode=True):
        super(ConvNeXtMoE, self).train(mode)
        self._freeze_stages()

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.
        """

        max_layer_id = 12 if self.depths[-2] > 9 else 6

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return max_layer_id + 1, max_layer_id + 2

        param_name = param_name[len(prefix):]
        if param_name.startswith('downsample_layers'):
            stage_id = int(param_name.split('.')[1])
            if stage_id == 0:
                layer_id = 0
            elif stage_id == 1 or stage_id == 2:
                layer_id = stage_id + 1
            else:  # stage_id == 3:
                layer_id = max_layer_id

        elif param_name.startswith('stages'):
            stage_id = int(param_name.split('.')[1])
            block_id = int(param_name.split('.')[2])
            if stage_id == 0 or stage_id == 1:
                layer_id = stage_id + 1
            elif stage_id == 2:
                layer_id = 3 + block_id // 3
            else:  # stage_id == 3:
                layer_id = max_layer_id

        # final norm layer
        else:
            layer_id = max_layer_id + 1

        return layer_id, max_layer_id + 2

    def init_weights(self): 
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training starts from scratch')
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support specifying `Pretrained` in `init_cfg` for {self.__class__.__name__}'
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            _state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
            
            if self.init_cfg.reformat:
                model_state_dict = self.state_dict()
                state_dict = OrderedDict()

                for k, v in _state_dict.items():
                    if k.startswith('backbone.'):
                        k = k[9:]
                        if 'downsample_layers.0.0' in k:
                            state_dict[k.replace('downsample_layers.0.0', 'dataset_stems')] = v
                        elif 'downsample_layers.0.1' in k:
                            state_dict[k.replace('downsample_layers.0.1', 'downsample_layers.0.0')] = v
                        elif 'pointwise_conv' in k or 'grn' in k:
                            stage_splits = k.split('.')
                            stage_ind = int(stage_splits[1])
                            block_ind = int(stage_splits[2])
                            if block_ind in self.moe_indices[stage_ind]:
                                for expert_idx, moe_arg in enumerate(self.moe_cfgs):
                                    for i in range(moe_arg.n_routed_experts):
                                        new_k = k.replace('pointwise_conv', f'ffn.expert_{expert_idx}.{i}.pointwise_conv') \
                                                if 'pointwise_conv' in k else \
                                                k.replace('grn', f'ffn.expert_{expert_idx}.{i}.grn')
                                        if new_k in model_state_dict and model_state_dict[new_k].shape == v.shape:
                                            state_dict[new_k] = v
                            else:
                                new_k = k.replace('pointwise_conv', 'ffn.pointwise_conv') if 'pointwise_conv' in k else k.replace('grn', 'ffn.grn')
                                state_dict[new_k] = v
                        else:
                            state_dict[k] = v

                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
            else:
                state_dict = OrderedDict()
                for k, v in _state_dict.items():
                    if k.startswith('backbone.'):
                        k = k[9:]
                        state_dict[k] = v
                            
            warnings = self.load_state_dict(state_dict, strict=False)
            if warnings.missing_keys or warnings.unexpected_keys:
                logger.warn(f'Warnings when loading state_dict: {warnings}')
