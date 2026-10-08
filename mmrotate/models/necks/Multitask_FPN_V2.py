# Copyright (c) OpenMMLab. All rights reserved.
# Modified from csuhan: https://github.com/csuhan/ReDet
from mmcv.runner import BaseModule, auto_fp16
from ..builder import ROTATED_NECKS

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from torch import Tensor

class TextGuidedFusionBlock(nn.Module):
    """
    Simple Cross-Attention Block for fusing Text Embeddings into Image Features.
    Inspired by YOLO-World's T-CSPLayer logic.
    """
    def __init__(self, img_channels, text_dim=512):
        super().__init__()
        # Projections
        self.img_proj = nn.Conv2d(img_channels, img_channels, 1)
        self.text_proj = nn.Linear(text_dim, img_channels)
        
        # Scale factor for attention
        self.scale = img_channels ** -0.5
        
        # Output projection (optional, but good for stability)
        self.out_proj = nn.Conv2d(img_channels, img_channels, 1)

        # Zero-init for residual connection (start with identity mapping)
        nn.init.constant_(self.out_proj.weight, 0)
        nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, img_feat, text_embeds):
        """
        img_feat: [B, C, H, W]
        text_embeds: [B, K, text_dim] or [K, text_dim] (will broadcast)
        """
        B, C, H, W = img_feat.shape
        
        # 1. Prepare Q (Image)
        # [B, C, H, W] -> [B, C, HW] -> [B, HW, C]
        q = self.img_proj(img_feat).flatten(2).transpose(1, 2)
        
        # 2. Prepare K, V (Text)
        # Handle case where text_embeds is [K, D] (shared across batch)
        if text_embeds.dim() == 2:
            text_embeds = text_embeds.unsqueeze(0).expand(B, -1, -1)
        
        # [B, K, text_dim] -> [B, K, C]
        k = self.text_proj(text_embeds)
        v = k 
        
        # 3. Cross Attention: Image querying Text
        # [B, HW, C] x [B, C, K] -> [B, HW, K]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # [B, HW, K] x [B, K, C] -> [B, HW, C]
        x = attn @ v
        
        # 4. Reshape back
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # 5. Residual Connection
        return img_feat + self.out_proj(x)


@ROTATED_NECKS.register_module()
class TextGuidedMultitaskFPN(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 text_dim=512,  # [NEW] Dimension of input text embeddings
                 start_level=0,
                 end_level=-1,
                 extra_level=0,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(TextGuidedMultitaskFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        
        # [NEW] Check for text dimension
        self.text_dim = text_dim

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.extra_level = extra_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        # [NEW] ModuleList for text fusion blocks
        # One block per output level
        self.text_fusion_blocks = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            
            # [NEW] Add fusion block for this level
            self.text_fusion_blocks.append(
                TextGuidedFusionBlock(out_channels, text_dim)
            )

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.extra_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
                
                # [NEW] Add fusion block for extra levels too
                self.text_fusion_blocks.append(
                    TextGuidedFusionBlock(out_channels, text_dim)
                )

    @auto_fp16()
    def forward(self, inputs, text_embeds=None): # [NEW] Accept text_embeds
        """
        Args:
            inputs (list[Tensor]): Image features.
            text_embeds (Tensor): Text embeddings [B, K, D] or [K, D].
        """
        assert text_embeds is not None, "Text embeddings required for TextGuidedFPN"

        # 1. Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 2. Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # 3. Build outputs & Inject Text
        outs = []
        
        # Part 1: Original backbone levels
        for i in range(used_backbone_levels):
            # Standard FPN conv
            feat = self.fpn_convs[i](laterals[i])
            
            # [NEW] Fuse with Text
            feat = self.text_fusion_blocks[i](feat, text_embeds)
            
            outs.append(feat)

        # Part 2: Extra levels
        if self.num_outs > len(outs):
            # use max pool
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add extra convs
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                
                # Calculate extra levels
                # Note: fpn_convs list includes standard + extra convs
                # The index for extra convs starts at used_backbone_levels
                
                # First extra level
                feat = self.fpn_convs[used_backbone_levels](extra_source)
                feat = self.text_fusion_blocks[used_backbone_levels](feat, text_embeds)
                outs.append(feat)

                # Subsequent extra levels
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        feat = self.fpn_convs[i](F.relu(outs[-1]))
                    else:
                        feat = self.fpn_convs[i](outs[-1])
                    
                    # [NEW] Fuse
                    feat = self.text_fusion_blocks[i](feat, text_embeds)
                    outs.append(feat)
                    
        return tuple(outs)