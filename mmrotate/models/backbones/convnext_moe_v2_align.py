# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.ops import roi_align

from ..builder import ROTATED_BACKBONES
from .convnext_moe_v2 import ConvNeXtMoEV2, ModalityRoutingBranch, MoE


class ModalityRoutingBranchAlign(ModalityRoutingBranch):
    """Experimental routing branch with normalized align inputs and
    bidirectional multi-positive contrastive loss.
    """

    def __init__(self, in_channels: int, routing_cfg: Optional[dict] = None):
        routing_cfg = routing_cfg or {}
        super().__init__(in_channels, routing_cfg)
        self.use_shared_norm = bool(routing_cfg.get('use_shared_norm', True))
        self.align_use_shared_norm = bool(
            routing_cfg.get('align_use_shared_norm', self.use_shared_norm))
        self.align_contrastive_i2t_weight = max(
            float(routing_cfg.get('align_contrastive_i2t_weight', 1.0)), 0.0)
        self.align_contrastive_t2i_weight = max(
            float(routing_cfg.get('align_contrastive_t2i_weight', 1.0)), 0.0)

    @classmethod
    def from_existing(cls, branch: ModalityRoutingBranch) -> "ModalityRoutingBranchAlign":
        route_alpha = (
            float(torch.sigmoid(branch.route_alpha.detach()).cpu().item())
            if branch.learnable_alpha
            else float(branch.route_alpha.detach().cpu().item())
        )
        align_temp = float(torch.exp(-branch.align_logit_scale.detach()).cpu().item())
        routing_cfg = dict(
            shared_dim=branch.shared_dim,
            dataset_names=list(branch.dataset_names),
            fuse_mode=branch.fuse_mode,
            align_loss_weight=branch.align_loss_weight,
            align_loss_type=branch.align_loss_type,
            stop_align_grad_to_backbone=branch.stop_align_grad_to_backbone,
            align_use_shared_norm=getattr(
                branch, 'align_use_shared_norm', not isinstance(branch.shared_norm, nn.Identity)),
            align_backbone_grad_scale=branch.align_backbone_grad_scale,
            align_loss_temperature=align_temp,
            align_loss_label_smoothing=branch.align_loss_label_smoothing,
            align_logit_scale_max=branch.align_logit_scale_max,
            route_alpha=route_alpha,
            learnable_alpha=branch.learnable_alpha,
            use_shared_norm=not isinstance(branch.shared_norm, nn.Identity),
            align_roi_pool_size=branch.align_roi_pool_size,
            align_roi_sampling_ratio=branch.align_roi_sampling_ratio,
            text_dim=branch.text_adapter.in_features if branch.text_adapter is not None else 0,
            align_contrastive_i2t_weight=getattr(branch, 'align_contrastive_i2t_weight', 1.0),
            align_contrastive_t2i_weight=getattr(branch, 'align_contrastive_t2i_weight', 1.0),
        )
        new_branch = cls(branch.in_channels, routing_cfg)
        new_branch.load_state_dict(branch.state_dict(), strict=True)
        return new_branch

    def _apply_shared_norm(self, shared_feature: torch.Tensor) -> torch.Tensor:
        return self.shared_norm(shared_feature)

    def _compute_align_logits(self,
                              roi_shared: torch.Tensor,
                              text_bank: torch.Tensor) -> torch.Tensor:
        logit_scale = self.align_logit_scale.exp().clamp(max=self.align_logit_scale_max)
        logit_scale = logit_scale.to(device=roi_shared.device, dtype=roi_shared.dtype)
        return torch.matmul(roi_shared, text_bank.t()) * logit_scale

    def _text_to_roi_contrastive_loss(self,
                                      logits: torch.Tensor,
                                      labels: torch.Tensor) -> torch.Tensor:
        active_labels = torch.unique(labels)
        if active_labels.numel() == 0:
            return logits.sum() * 0

        logits_t2i = logits.transpose(0, 1)[active_labels]
        positive_mask = labels.unsqueeze(0).eq(active_labels.unsqueeze(1))
        log_probs = F.log_softmax(logits_t2i, dim=1)
        denom = positive_mask.sum(dim=1).clamp(min=1).to(dtype=log_probs.dtype)
        loss_per_text = -(positive_mask.to(dtype=log_probs.dtype) * log_probs).sum(dim=1) / denom
        return loss_per_text.mean()

    def _bidirectional_contrastive_loss(self,
                                        logits: torch.Tensor,
                                        labels: torch.Tensor) -> torch.Tensor:
        weighted_losses = []
        weight_sum = 0.0

        if self.align_contrastive_i2t_weight > 0:
            loss_i2t = F.cross_entropy(
                logits,
                labels,
                label_smoothing=self.align_loss_label_smoothing)
            weighted_losses.append(loss_i2t * self.align_contrastive_i2t_weight)
            weight_sum += self.align_contrastive_i2t_weight

        if self.align_contrastive_t2i_weight > 0:
            loss_t2i = self._text_to_roi_contrastive_loss(logits, labels)
            weighted_losses.append(loss_t2i * self.align_contrastive_t2i_weight)
            weight_sum += self.align_contrastive_t2i_weight

        if weight_sum <= 0:
            return logits.sum() * 0
        return sum(weighted_losses) / weight_sum

    def _align_roi_with_text(self,
                             shared_feature: torch.Tensor,
                             modality_indices,
                             roi_entry,
                             text_bank: torch.Tensor,
                             modality: Optional[str] = None):
        if self.align_loss_type not in (
            'bidirectional_contrastive',
            'symmetric_contrastive',
            'clip',
        ):
            return super()._align_roi_with_text(
                shared_feature, modality_indices, roi_entry, text_bank, modality=modality)

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

        index_tensor = torch.as_tensor(
            modality_indices, device=shared_feature.device, dtype=torch.long)
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

        logits = self._compute_align_logits(roi_shared, text_bank)
        loss = self._bidirectional_contrastive_loss(logits, labels)

        return loss

    def forward(self,
                x: torch.Tensor,
                batch_indices=None,
                roi_features=None,
                text_features=None,
                gt_bboxes=None,
                gt_labels=None,
                modality_img_shapes=None,
                data_format: str = 'channel_last') -> tuple:
        x_chw = self._to_channel_first(x, data_format)
        shared_feature = self._project_to_shared_space(x_chw, batch_indices)
        shared_feature = self._apply_shared_norm(shared_feature)
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
                align_input = align_input + self.align_backbone_grad_scale * (
                    x_chw - x_chw.detach())
            shared_feature_align = self._project_to_shared_space(align_input, batch_indices)
            if self.align_use_shared_norm:
                shared_feature_align = self._apply_shared_norm(shared_feature_align)

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


@ROTATED_BACKBONES.register_module()
class ConvNeXtMoEV2Align(ConvNeXtMoEV2):
    """ConvNeXtMoEV2 with isolated experimental routing alignment branch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._upgrade_route_branches()
        self._freeze_stages()

    def _upgrade_route_branches(self):
        for module in list(self.modules()):
            if not isinstance(module, MoE):
                continue
            for idx in range(len(module.moe_args)):
                branch_name = f'route_branch_{idx}'
                old_branch = getattr(module, branch_name, None)
                if old_branch is None or isinstance(old_branch, ModalityRoutingBranchAlign):
                    continue
                if isinstance(old_branch, ModalityRoutingBranch):
                    setattr(
                        module,
                        branch_name,
                        ModalityRoutingBranchAlign.from_existing(old_branch))
