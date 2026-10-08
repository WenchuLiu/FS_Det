# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from collections import OrderedDict
from collections.abc import Mapping
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmrotate.core.bbox.transforms import obb2xyxy
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

import numpy as np

from mmengine.model import ModuleList, Sequential
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.model import BaseModule

from ..builder import ROTATED_BACKBONES

from mmcv.ops import roi_align
from mmcv.runner import BaseModule
from timm.models.layers import DropPath, trunc_normal_

from .convnext_moe_origin import FFN

from .gates import (
    MMSplitGate,
    OriginGate,
    FocusGate,
    ShareGate,
    FusionOriginGate,
    DualLogitFusionGate,
    SharedExpertGate,
)
from .utils import build_LayerNorm2d_layer


GATES = dict(
    mmsplitgate=MMSplitGate,
    focusgate=FocusGate,
    origingate=OriginGate,
    fusionorigingate=FusionOriginGate,
    duallogitfusiongate=DualLogitFusionGate,
    sharegate=ShareGate,
    sharedexpertgate=SharedExpertGate,
)


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_to_dict(cfg):
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, 'items'):
        return dict(cfg.items())
    return dict(cfg.__dict__)


class ModalityRoutingBranch(nn.Module):
    """Build a modality-invariant routing feature and fuse it with raw features.

    The branch uses a modality-specific 1x1 convolution to project each modality
    into a shared low-dimensional space, then fuses the shared feature with the
    original feature for MoE routing.
    """

    def __init__(self, in_channels: int, routing_cfg: Optional[dict] = None):
        super().__init__()
        routing_cfg = routing_cfg or {}
        self.in_channels = in_channels
        self.shared_dim = int(routing_cfg.get('shared_dim', max(16, in_channels // 4)))
        self.dataset_names = list(routing_cfg.get('dataset_names', []))
        self.fuse_mode = routing_cfg.get('fuse_mode', 'concat')
        self.align_loss_weight = float(routing_cfg.get('align_loss_weight', 0.0))
        self.align_loss_type = routing_cfg.get('align_loss_type', 'cosine')
        self.stop_align_grad_to_backbone = bool(routing_cfg.get('stop_align_grad_to_backbone', False))
        self.align_use_shared_norm = bool(routing_cfg.get('align_use_shared_norm', False))
        default_grad_scale = 0.0 if self.stop_align_grad_to_backbone else 1.0
        self.align_backbone_grad_scale = float(
            routing_cfg.get('align_backbone_grad_scale', default_grad_scale))
        self.align_backbone_grad_scale = max(0.0, min(self.align_backbone_grad_scale, 1.0))
        align_temp = max(float(routing_cfg.get('align_loss_temperature', 0.07)), 1e-4)
        self.align_loss_label_smoothing = float(
            routing_cfg.get('align_loss_label_smoothing', 0.0))
        self.align_loss_label_smoothing = max(0.0, min(self.align_loss_label_smoothing, 0.999))
        self.align_logit_scale_max = float(routing_cfg.get('align_logit_scale_max', 100.0))
        # Keep the forward values unchanged while allowing a configurable amount
        # of alignment gradient to flow back into the backbone feature map.
        self.register_buffer(
            'align_logit_scale',
            torch.tensor(np.log(1.0 / align_temp), dtype=torch.float32),
            persistent=False)

        route_alpha = float(routing_cfg.get('route_alpha', 0.5))
        route_alpha = max(min(route_alpha, 1.0 - 1e-4), 1e-4)
        self.learnable_alpha = bool(routing_cfg.get('learnable_alpha', True))
        if self.learnable_alpha:
            logit_alpha = torch.log(torch.tensor(route_alpha / (1.0 - route_alpha)))
            self.route_alpha = nn.Parameter(logit_alpha)
        else:
            self.register_buffer('route_alpha', torch.tensor(route_alpha), persistent=False)

        self.projectors = nn.ModuleDict({
            name: nn.Conv2d(in_channels, self.shared_dim, kernel_size=1, bias=True)
            for name in self.dataset_names
        })
        self.shared_norm = nn.BatchNorm2d(self.shared_dim) if routing_cfg.get('use_shared_norm', True) else nn.Identity()

        if self.fuse_mode == 'concat':
            self.fuser = nn.Sequential(
                nn.Conv2d(in_channels + self.shared_dim, in_channels, kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False))
        elif self.fuse_mode == 'add':
            self.fuser = nn.Conv2d(self.shared_dim, in_channels, kernel_size=1, bias=False)
        else:
            raise ValueError(f'Unsupported fuse_mode: {self.fuse_mode}, expected `concat` or `add`.')

        self.align_roi_pool_size = int(routing_cfg.get('align_roi_pool_size', 7))
        self.align_roi_sampling_ratio = int(routing_cfg.get('align_roi_sampling_ratio', 0))
        text_dim = int(routing_cfg.get('text_dim', 0) or 0)
        self.text_adapter = nn.Linear(text_dim, self.shared_dim) if text_dim > 0 else None

    def _get_projector(self, dataset_name: str) -> nn.Module:
        if dataset_name in self.projectors:
            return self.projectors[dataset_name]
        raise KeyError(
            f'Unknown modality `{dataset_name}` for routing branch. '
            f'Known modalities: {list(self.projectors.keys())}')

    def _to_channel_first(self, x: torch.Tensor, data_format: str) -> torch.Tensor:
        if data_format == 'channel_last':
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def _to_origin_format(self, x: torch.Tensor, data_format: str) -> torch.Tensor:
        if data_format == 'channel_last':
            return x.permute(0, 2, 3, 1).contiguous()
        return x

    def _safe_zero(self, ref: torch.Tensor) -> torch.Tensor:
        return ref.sum().detach() * 0

    def _graph_zero_from(self, tensors) -> Optional[torch.Tensor]:
        if tensors is None:
            return None
        if torch.is_tensor(tensors):
            return tensors.sum() * 0
        if isinstance(tensors, (list, tuple)):
            zero_terms = [self._graph_zero_from(t) for t in tensors]
            zero_terms = [z for z in zero_terms if z is not None]
            if len(zero_terms) == 0:
                return None
            return sum(zero_terms)
        return None

    def _get_modality_text_bank(self, text_features, modality: str) -> Optional[torch.Tensor]:
        if text_features is None:
            return None
        if isinstance(text_features, dict):
            text_features = text_features.get(modality, text_features.get('all', None))
        if text_features is None or (not torch.is_tensor(text_features)) or text_features.numel() == 0:
            return None

        if text_features.dim() == 1:
            text_features = text_features.unsqueeze(0)
        elif text_features.dim() > 2:
            text_features = text_features.reshape(-1, text_features.shape[-1])

        if self.text_adapter is not None:
            if text_features.shape[-1] != self.text_adapter.in_features:
                return None
            text_features = self.text_adapter(text_features)
        elif text_features.shape[-1] != self.shared_dim:
            return None
        return text_features

    def _get_modality_roi_entry(self, roi_features, modality: str):
        if roi_features is None:
            return None
        if isinstance(roi_features, dict):
            return roi_features.get(modality, roi_features.get('all', None))
        return None

    def _get_modality_data(self, data, modality: str):
        if data is None:
            return None
        if isinstance(data, dict):
            return data.get(modality, data.get('all', None))
        return data

    def _get_modality_img_shape(self, modality_img_shapes, modality: str):
        shape = self._get_modality_data(modality_img_shapes, modality)
        if shape is None:
            return None
        if isinstance(shape, torch.Tensor):
            shape = shape.tolist()
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            h = int(shape[-2])
            w = int(shape[-1])
            if h > 0 and w > 0:
                return h, w
        return None

    def _print_align_debug(self, message: str) -> None:
        if not getattr(self, '_align_debug_printed', False):
            print(message)
            self._align_debug_printed = True

    def _to_hbb(self, bboxes: torch.Tensor) -> Optional[torch.Tensor]:
        if not torch.is_tensor(bboxes) or bboxes.numel() == 0:
            return None
        if bboxes.dim() == 1:
            bboxes = bboxes.reshape(1, -1)
        if bboxes.size(-1) == 4:
            return bboxes[:, :4]
        if bboxes.size(-1) == 5:
            return obb2xyxy(bboxes, version='le90')
        return None

    def _build_modality_roi_entry_from_gt(self,
                                          modality: str,
                                          gt_bboxes=None,
                                          gt_labels=None,
                                          modality_img_shapes=None):
        boxes_list = self._get_modality_data(gt_bboxes, modality)
        labels_list = self._get_modality_data(gt_labels, modality)
        if boxes_list is None or labels_list is None:
            return None
        if not isinstance(boxes_list, (list, tuple)) or not isinstance(labels_list, (list, tuple)):
            return None
        if len(boxes_list) == 0 or len(labels_list) == 0:
            return None

        img_shape = self._get_modality_img_shape(modality_img_shapes, modality)
        if img_shape is None:
            return None
        img_h, img_w = img_shape

        rois_all = []
        labels_all = []
        num_imgs = min(len(boxes_list), len(labels_list))
        for img_id in range(num_imgs):
            boxes = boxes_list[img_id]
            labels = labels_list[img_id]
            if (not torch.is_tensor(boxes)) or (not torch.is_tensor(labels)):
                continue
            if boxes.numel() == 0 or labels.numel() == 0:
                continue

            hbb = self._to_hbb(boxes)
            if hbb is None:
                continue
            labels = labels.reshape(-1).long().to(hbb.device)
            n = min(hbb.size(0), labels.size(0))
            if n <= 0:
                continue
            hbb = hbb[:n]
            labels = labels[:n]

            x1 = torch.minimum(hbb[:, 0], hbb[:, 2])
            y1 = torch.minimum(hbb[:, 1], hbb[:, 3])
            x2 = torch.maximum(hbb[:, 0], hbb[:, 2])
            y2 = torch.maximum(hbb[:, 1], hbb[:, 3])
            x1 = (x1 / float(img_w)).clamp(0, 1 - 1.0 / max(float(img_w), 1.0))
            y1 = (y1 / float(img_h)).clamp(0, 1 - 1.0 / max(float(img_h), 1.0))
            x2 = (x2 / float(img_w)).clamp(0, 1)
            y2 = (y2 / float(img_h)).clamp(0, 1)
            x2 = torch.maximum(x2, x1 + 1.0 / max(float(img_w), 1.0)).clamp(max=1)
            y2 = torch.maximum(y2, y1 + 1.0 / max(float(img_h), 1.0)).clamp(max=1)
            rois = torch.stack([
                x1.new_full((n, ), float(img_id)),
                x1,
                y1,
                x2,
                y2
            ], dim=1)
            rois_all.append(rois)
            labels_all.append(labels)

        if len(rois_all) == 0:
            return None
        return dict(
            rois=torch.cat(rois_all, dim=0),
            labels=torch.cat(labels_all, dim=0))

    def _align_roi_with_text(self,
                             shared_feature: torch.Tensor,
                             modality_indices: List[int],
                             roi_entry,
                             text_bank: torch.Tensor,
                             modality: Optional[str] = None) -> Optional[torch.Tensor]:
        if not isinstance(roi_entry, dict):
            self._print_align_debug(
                f'[BackboneAlignDebug] {modality}: roi_entry is not a dict')
            return None
        rois = roi_entry.get('rois', None)
        labels = roi_entry.get('labels', None)
        if (not torch.is_tensor(rois)) or (not torch.is_tensor(labels)):
            self._print_align_debug(
                f'[BackboneAlignDebug] {modality}: rois/labels are invalid')
            return None
        if rois.numel() == 0 or labels.numel() == 0:
            self._print_align_debug(
                f'[BackboneAlignDebug] {modality}: empty rois or labels')
            return None
        if rois.dim() != 2 or rois.size(-1) != 5:
            self._print_align_debug(
                f'[BackboneAlignDebug] {modality}: unexpected rois shape {tuple(rois.shape)}')
            return None

        labels = labels.reshape(-1).long()
        n = min(rois.size(0), labels.size(0))
        if n <= 0:
            self._print_align_debug(
                f'[BackboneAlignDebug] {modality}: n <= 0 after truncation')
            return None
        rois = rois[:n]
        labels = labels[:n]

        index_tensor = torch.as_tensor(modality_indices, device=shared_feature.device, dtype=torch.long)
        if index_tensor.numel() == 0:
            self._print_align_debug(
                f'[BackboneAlignDebug] {modality}: empty modality_indices')
            return None
        modal_feature = shared_feature[index_tensor]
        b, _, h, w = modal_feature.shape
        if b == 0 or h == 0 or w == 0:
            self._print_align_debug(
                f'[BackboneAlignDebug] {modality}: invalid modal_feature shape {tuple(modal_feature.shape)}')
            return None

        valid_img = (rois[:, 0] >= 0) & (rois[:, 0] < b)
        valid_label = (labels >= 0) & (labels < text_bank.size(0))
        valid = valid_img & valid_label
        if not valid.any():
            self._print_align_debug(
                f'[BackboneAlignDebug] {modality}: no valid roi-text pairs; '
                f'num_rois={rois.size(0)}, batch={b}, text_bank={tuple(text_bank.shape)}, '
                f'label_range=({int(labels.min().item())}, {int(labels.max().item())})')
            return None

        rois = rois[valid].to(device=shared_feature.device, dtype=modal_feature.dtype)
        labels = labels[valid].to(device=shared_feature.device, dtype=torch.long)

        x1 = rois[:, 1].clamp(0, 1)
        y1 = rois[:, 2].clamp(0, 1)
        x2 = rois[:, 3].clamp(0, 1)
        y2 = rois[:, 4].clamp(0, 1)
        x1 = x1.clamp(max=1 - 1.0 / max(float(w), 1.0))
        y1 = y1.clamp(max=1 - 1.0 / max(float(h), 1.0))
        x2 = torch.maximum(x2, x1 + 1.0 / max(float(w), 1.0)).clamp(max=1)
        y2 = torch.maximum(y2, y1 + 1.0 / max(float(h), 1.0)).clamp(max=1)

        rois_abs = torch.stack([
            rois[:, 0],
            x1 * w,
            y1 * h,
            x2 * w,
            y2 * h
        ], dim=1)
        roi_shared = roi_align(
            modal_feature,
            rois_abs,
            (self.align_roi_pool_size, self.align_roi_pool_size),
            1.0,
            self.align_roi_sampling_ratio,
            'avg',
            True)
        roi_shared = roi_shared.mean(dim=(2, 3))
        roi_shared = F.normalize(roi_shared, dim=1)

        text_bank = F.normalize(
            text_bank.to(device=shared_feature.device, dtype=roi_shared.dtype), dim=1)
        text_targets = text_bank[labels]
        if self.align_loss_type == 'mse':
            loss = F.mse_loss(roi_shared, text_targets)
        elif self.align_loss_type == 'cosine':
            loss = 1 - F.cosine_similarity(roi_shared, text_targets, dim=1).mean()
        elif self.align_loss_type in ('ce', 'contrastive'):
            logit_scale = self.align_logit_scale.exp().clamp(max=self.align_logit_scale_max)
            logit_scale = logit_scale.to(device=roi_shared.device, dtype=roi_shared.dtype)
            logits = torch.matmul(roi_shared, text_bank.t()) * logit_scale
            loss = F.cross_entropy(
                logits,
                labels,
                label_smoothing=self.align_loss_label_smoothing)
        else:
            raise ValueError(
                f'Unsupported align_loss_type: {self.align_loss_type}, expected '
                '`mse`, `cosine`, `ce`, or `contrastive`.')
        return loss

    def _alignment_loss(self,
                        shared_feature: torch.Tensor,
                        batch_indices: Optional[Dict[str, List[int]]],
                        roi_features=None,
                        text_features=None,
                        gt_bboxes=None,
                        gt_labels=None,
                        modality_img_shapes=None) -> torch.Tensor:
        if self.align_loss_weight <= 0:
            return self._safe_zero(shared_feature)
        if batch_indices is None or len(batch_indices) == 0:
            return self._safe_zero(shared_feature)

        losses = []
        text_zero_terms = []
        for modality, indices in batch_indices.items():
            if len(indices) == 0:
                continue
            roi_entry = self._get_modality_roi_entry(roi_features, modality)
            if roi_entry is None:
                roi_entry = self._build_modality_roi_entry_from_gt(
                    modality,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    modality_img_shapes=modality_img_shapes)
            text_bank = self._get_modality_text_bank(text_features, modality)
            if text_bank is not None:
                text_zero = self._graph_zero_from(text_bank)
                if text_zero is not None:
                    text_zero_terms.append(text_zero)
            if roi_entry is None or text_bank is None:
                continue
            loss = self._align_roi_with_text(
                shared_feature, indices, roi_entry, text_bank, modality=modality)
            if loss is not None:
                losses.append(loss)

        if len(losses) == 0:
            self._print_align_debug(
                '[BackboneAlignDebug] alignment loss list is empty; returning zero')
            zero = self._safe_zero(shared_feature)
            if len(text_zero_terms) > 0:
                zero = zero + sum(text_zero_terms)
            return zero
        return sum(losses) / len(losses) * self.align_loss_weight

    def _project_to_shared_space(self,
                                 x_chw: torch.Tensor,
                                 batch_indices: Optional[Dict[str, List[int]]]) -> torch.Tensor:
        b, _, h, w = x_chw.shape
        shared_feature = x_chw.new_zeros((b, self.shared_dim, h, w))
        if batch_indices is None or len(batch_indices) == 0:
            raise ValueError(
                'ModalityRoutingBranch requires non-empty batch_indices after '
                'removing default_projector.')

        assigned = torch.zeros(b, device=x_chw.device, dtype=torch.bool)
        for dataset_name, indices in batch_indices.items():
            if len(indices) == 0:
                continue
            index_tensor = torch.as_tensor(indices, device=x_chw.device, dtype=torch.long)
            projector = self._get_projector(dataset_name)
            proj_out = projector(x_chw[index_tensor]).to(dtype=shared_feature.dtype)
            shared_feature[index_tensor] = proj_out
            assigned[index_tensor] = True
        if (~assigned).any():
            missing_indices = torch.nonzero(~assigned, as_tuple=False).flatten().tolist()
            raise ValueError(
                'Some samples were not assigned to any modality projector: '
                f'{missing_indices}')
        return shared_feature

    def forward(self,
                x: torch.Tensor,
                batch_indices: Optional[Dict[str, List[int]]] = None,
                roi_features=None,
                text_features=None,
                gt_bboxes=None,
                gt_labels=None,
                modality_img_shapes=None,
                data_format: str = 'channel_last') -> tuple:
        x_chw = self._to_channel_first(x, data_format)
        shared_feature = self._project_to_shared_space(x_chw, batch_indices)
        shared_feature = self.shared_norm(shared_feature)
        if self.fuse_mode == 'concat':
            fused = self.fuser(torch.cat([x_chw, shared_feature], dim=1))
        else:
            fused = x_chw + self.fuser(shared_feature)

        route_alpha = torch.sigmoid(self.route_alpha) if self.learnable_alpha else self.route_alpha
        route_alpha = route_alpha.to(dtype=x_chw.dtype, device=x_chw.device)
        routed_feature = route_alpha * x_chw + (1.0 - route_alpha) * fused
        shared_feature_align = shared_feature
        if self.align_backbone_grad_scale < 1.0:
            align_input = x_chw.detach()
            if self.align_backbone_grad_scale > 0:
                align_input = align_input + self.align_backbone_grad_scale * (x_chw - x_chw.detach())
            shared_feature_align = self._project_to_shared_space(align_input, batch_indices)
            if self.align_use_shared_norm:
                shared_feature_align = self.shared_norm(shared_feature_align)

        align_loss = self._alignment_loss(
            shared_feature_align,
            batch_indices,
            roi_features,
            text_features,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            modality_img_shapes=modality_img_shapes)
        routed_feature = self._to_origin_format(routed_feature, data_format)
        return routed_feature, align_loss


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
                 use_grn: bool = False,
                 linear_pw_conv: bool = True):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.in_channels = in_channels
        self.moe_args = moe_args
        self.linear_pw_conv = linear_pw_conv
                
        for i, moe_arg in enumerate(moe_args):
            expert_mid_channels = int(mid_channels * float(_cfg_get(moe_arg, 'mid_scale', 1.0)))
                        
            gate_type = _cfg_get(moe_arg, 'type', None)
            gate_func = GATES[gate_type]
            gate_args = _cfg_to_dict(moe_arg)
            gate_args.pop('routing_fusion_cfg', None)
            gate = gate_func(in_channels, gate_args)
            routing_cfg = _cfg_get(moe_arg, 'routing_fusion_cfg', {}) or {}
            if 'dataset_names' not in routing_cfg and _cfg_get(moe_arg, 'dataset_names', None) is not None:
                routing_cfg = dict(routing_cfg)
                routing_cfg['dataset_names'] = _cfg_get(moe_arg, 'dataset_names')
            route_branch = ModalityRoutingBranch(in_channels, routing_cfg)
            n_routed_experts = int(_cfg_get(moe_arg, 'n_routed_experts', 1))
            n_shared_experts = int(_cfg_get(moe_arg, 'n_shared_experts', 0))
            total_experts = n_routed_experts + n_shared_experts
            experts = nn.ModuleList([FFN(self.in_channels,
                                         expert_mid_channels,
                                         pw_conv,
                                         act_cfg,
                                         use_grn)
                                     for _ in range(total_experts)])
            setattr(self, f'gate_{i}', gate)
            setattr(self, f'expert_{i}', experts)
            setattr(self, f'route_branch_{i}', route_branch)
    
    
    def random_k_gating(self, features):
        for i, moe_arg in enumerate(self.moe_args):
            experts = getattr(self, f'expert_{i}')
            n_routed_experts = int(_cfg_get(moe_arg, 'n_routed_experts', 1))
            n_shared_experts = int(_cfg_get(moe_arg, 'n_shared_experts', 0))
            idx = np.random.randint(0, n_routed_experts, (1, ))[0] + n_shared_experts
            features = experts[idx](features)
        return features
    
    def forward(self,
                x: torch.Tensor,
                batch_indices=None,
                roi_features=None,
                text_features=None,
                gt_bboxes=None,
                gt_labels=None,
                modality_img_shapes=None) -> torch.Tensor:
        """Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor
            batch_indices (dict): Dictionary containing batch indices for each dataset

        Returns:
            torch.Tensor: Output tensor after expert routing and computation
        """
        total_gate_loss = x.sum().detach() * 0
        total_align_loss = x.sum().detach() * 0
        y = x
        for i in range(len(self.moe_args)):
            gate = getattr(self, f'gate_{i}')
            experts = getattr(self, f'expert_{i}')
            route_branch = getattr(self, f'route_branch_{i}')
            data_format = 'channel_last' if self.linear_pw_conv else 'channel_first'
            route_feature, align_loss = route_branch(
                y,
                batch_indices=batch_indices,
                roi_features=roi_features,
                text_features=text_features,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                modality_img_shapes=modality_img_shapes,
                data_format=data_format)
            if 'expert_input' in inspect.signature(gate.forward).parameters:
                gate_loss, y = gate(
                    route_feature,
                    experts,
                    batch_indices,
                    expert_input=y)
            else:
                gate_loss, y = gate(y, experts, batch_indices)
            total_gate_loss = total_gate_loss + gate_loss
            total_align_loss = total_align_loss + align_loss
        avg_gate_loss = total_gate_loss / len(self.moe_args)
        avg_align_loss = total_align_loss / len(self.moe_args)
        return y, avg_gate_loss, avg_align_loss



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

        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels, groups=in_channels, **dw_conv_cfg)
        self.linear_pw_conv = linear_pw_conv
        self.norm = build_LayerNorm2d_layer(norm_cfg, in_channels)
        mid_channels = int(mlp_ratio * in_channels) 
        if self.linear_pw_conv:
            # Use linear layer to do pointwise conv.
            pw_conv = nn.Linear
        else:
            pw_conv = partial(nn.Conv2d, kernel_size=1)
        self.ffn = FFN(in_channels,mid_channels,pw_conv,act_cfg,use_grn)

        self.moe_args = moe_args
        if len(moe_args) != 0:
            
            self.ffn = MoE(in_channels,
                           mid_channels,
                           moe_args,
                           pw_conv=pw_conv,
                           use_grn=use_grn,
                           act_cfg=act_cfg,
                           linear_pw_conv=linear_pw_conv)
        else:
            self.ffn = FFN(in_channels,mid_channels,pw_conv,act_cfg,use_grn)


        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        
        
        
    def forward(self,
                x,
                batch_indices=None,
                roi_features=None,
                text_features=None,
                gt_bboxes=None,
                gt_labels=None,
                modality_img_shapes=None):
        def _inner_forward(x,
                           batch_indices,
                           roi_features,
                           text_features,
                           gt_bboxes,
                           gt_labels,
                           modality_img_shapes):
            shortcut = x
            gate_loss = x.sum().detach() * 0
            align_loss = x.sum().detach() * 0

            x = self.depthwise_conv(x)
            if self.linear_pw_conv:
                x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x = self.norm(x, data_format='channel_last')
                if len(self.moe_args) != 0:
                    x, gate_loss, align_loss = self.ffn(
                        x,
                        batch_indices,
                        roi_features,
                        text_features,
                        gt_bboxes=gt_bboxes,
                        gt_labels=gt_labels,
                        modality_img_shapes=modality_img_shapes)
                else:
                    x = self.ffn(x)
                x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            else:
                x = self.norm(x, data_format='channel_first')
                if len(self.moe_args) != 0:
                    x, gate_loss, align_loss = self.ffn(
                        x,
                        batch_indices,
                        roi_features,
                        text_features,
                        gt_bboxes=gt_bboxes,
                        gt_labels=gt_labels,
                        modality_img_shapes=modality_img_shapes)
                else:
                    x = self.ffn(x)

            if self.gamma is not None:
                x = x.mul(self.gamma.view(1, -1, 1, 1))

            x = shortcut + self.drop_path(x)
            return x, gate_loss, align_loss

        if self.with_cp and x.requires_grad:
            x, gate_loss, align_loss = cp.checkpoint(
                _inner_forward,
                x,
                batch_indices,
                roi_features,
                text_features,
                gt_bboxes,
                gt_labels,
                modality_img_shapes)
        else:
            x, gate_loss, align_loss = _inner_forward(
                x,
                batch_indices,
                roi_features,
                text_features,
                gt_bboxes,
                gt_labels,
                modality_img_shapes)
        return x, gate_loss, align_loss


@ROTATED_BACKBONES.register_module()
class ConvNeXtMoEV2(BaseModule):
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
                 moe_args:List[dict] = [],
                 frozen_stages=-1,
                 unfreeze_gate=False,  # <--- [新增参数] 默认为 False，保持原有逻辑
                 freeze_backbone_except_moe=False, # <--- [Modify 1] 新增参数
                 gap_before_final_norm=False,
                 with_cp=False,
                 return_gate_loss=True,
                 routing_fusion_cfg: Optional[dict] = None,
                 init_cfg=[
                     dict(
                         type='TruncNormal',
                         layer=['Conv2d', 'Linear'],
                         std=.02,
                         bias=0.),
                     dict(
                         type='Constant', layer=['LayerNorm'], val=1.,
                         bias=0.),
                 ],
                 **kwargs: Any):
        super().__init__(init_cfg=init_cfg)
        _ = kwargs

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
        self.moe_args = moe_args
        if routing_fusion_cfg is not None and len(self.moe_args) > 0:
            patched_args = []
            for moe_arg in self.moe_args:
                if moe_arg is None:
                    patched_args.append(moe_arg)
                    continue
                m = dict(moe_arg)
                if 'routing_fusion_cfg' not in m:
                    m['routing_fusion_cfg'] = dict(routing_fusion_cfg)
                patched_args.append(m)
            self.moe_args = patched_args
        
        self.frozen_stages = frozen_stages
        self.unfreeze_gate = unfreeze_gate  # <--- [新增] 保存这个配置
        self.freeze_backbone_except_moe = freeze_backbone_except_moe  # <--- [Modify 2] 保存这个配置
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
                    moe_args=self.moe_args if j in moe_id else [],
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
        
        self.dataset_stems = nn.Conv2d(in_channels,
                                       self.channels[0],
                                       kernel_size=stem_patch_size,
                                       stride=stem_patch_size)
        self.return_gate_loss = return_gate_loss
        self._freeze_stages()
        
        
    def forward(self,
                inputs,
                dataset_names=['single'],
                roi_features=None,
                text_features=None,
                gt_bboxes=None,
                gt_labels=None):
        
        # if len(dataset_names) == 1:
        #     inputs = [inputs] 
        
        # --- 修改开始：自动对齐输入数据与 dataset_names ---
        # 1. 如果输入是单个 Tensor (单模态情况)，强制转为列表，防止后续索引报错
        if not isinstance(inputs, list) and not isinstance(inputs, tuple):
            inputs = [inputs]
            
        # 2. 强制对齐 dataset_names 的长度与 inputs 的实际长度
        # 如果配置写了3个模态，但实际只送进来了1个模态(SAR)，则截断 dataset_names
        if len(dataset_names) > len(inputs):
            # 打印一个警告是个好习惯，或者直接截断
            dataset_names = dataset_names[:len(inputs)]
        # --- 修改结束 ---
        modality_img_shapes = {}
        for i, dataset_name in enumerate(dataset_names):
            if torch.is_tensor(inputs[i]) and inputs[i].dim() >= 4:
                modality_img_shapes[dataset_name] = tuple(inputs[i].shape[-2:])
        batch_indices = {
            dataset_name: list(range(final_index, final_index + inputs[i].shape[0]))
            for i, dataset_name in enumerate(dataset_names)
            for final_index in [sum(inputs[j].shape[0] for j in range(i))]
        }
        
        inputs = torch.cat(inputs, dim=0)
        features = self.dataset_stems(inputs)
        outs = []
        gate_losses = []
        align_losses = []
        for i, stage in enumerate(self.stages):
            features = self.downsample_layers[i](features)
            for each_layer in stage:
                features, gate_loss, align_loss = each_layer(
                    features,
                    batch_indices=batch_indices,
                    roi_features=roi_features,
                    text_features=text_features,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    modality_img_shapes=modality_img_shapes)
                gate_losses.append(gate_loss)
                align_losses.append(align_loss)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                if self.gap_before_final_norm:
                    gap = features.mean([-2, -1], keepdim=True)
                    outs.append(norm_layer(gap).flatten(1))
                else:
                    outs.append(norm_layer(features))
        gate_loss = sum(gate_losses) / len(gate_losses) if len(gate_losses) > 0 else features.sum().detach() * 0
        align_loss = sum(align_losses) / len(align_losses) if len(align_losses) > 0 else features.sum().detach() * 0
        if self.return_gate_loss:
            return tuple(outs), dict(gate_loss=gate_loss, align_loss=align_loss)
        else:
            return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0 and hasattr(self, 'dataset_stems'):
            self.dataset_stems.eval()
            for param in self.dataset_stems.parameters():
                param.requires_grad = False
        gate_module_types = (
            OriginGate,
            MMSplitGate,
            FocusGate,
            ShareGate,
            FusionOriginGate,
            DualLogitFusionGate,
        )
        # 1. 先执行标准的冻结流程：把指定 Stage 内的所有东西都冻结
        for i in range(self.frozen_stages):
            downsample_layer = self.downsample_layers[i]
            stage = self.stages[i]
            downsample_layer.eval()
            stage.eval()
            for param in chain(downsample_layer.parameters(), stage.parameters()):
                param.requires_grad = False
            # 2. [新增逻辑] 如果配置了解冻 Gate，则把 Gate 和 routing
            # branch 一并重新打开，确保当前的跨模态对齐分支也能学习。
            if self.unfreeze_gate:
                # 遍历当前 Stage 的所有子模块
                for name, module in stage.named_modules():
                    if isinstance(module, gate_module_types + (ModalityRoutingBranch, )):
                        # 关键：必须显式设为 train()，否则 noisy gating 等逻辑可能失效
                        module.train() 
                        # 打开梯度
                        for param in module.parameters():
                            param.requires_grad = True
                        
                        # 可选：打印日志确认解冻成功
                        # print(f"Info: Unfreezing Gate layer at stage {i}: {name}")
            if self.freeze_backbone_except_moe:
                for name, module in stage.named_modules():
                    if isinstance(module, MoE):
                        module.train()
                        for param in module.parameters():
                            param.requires_grad = True

    def train(self, mode=True):
        super(ConvNeXtMoEV2, self).train(mode)
        self._freeze_stages()
        return self

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
                                for expert_idx, moe_arg in enumerate(self.moe_args):
                                    n_routed_experts = int(_cfg_get(moe_arg, 'n_routed_experts', 1))
                                    for i in range(n_routed_experts):
                                        new_k = k.replace('pointwise_conv', f'ffn.expert_{expert_idx}.{i}.pointwise_conv') \
                                                if 'pointwise_conv' in k else \
                                                k.replace('grn', f'ffn.expert_{expert_idx}.{i}.grn')
                                        if "grn.gamma" in new_k or "grn.beta" in new_k:
                                            v = v.reshape(-1)
                                        if new_k in model_state_dict and model_state_dict[new_k].shape == v.shape:
                                            state_dict[new_k] = v
                            else:
                                new_k = k.replace('pointwise_conv', 'ffn.pointwise_conv') if 'pointwise_conv' in k else k.replace('grn', 'ffn.grn')
                                if "grn.gamma" in new_k or "grn.beta" in new_k:
                                    v = v.reshape(-1)
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
