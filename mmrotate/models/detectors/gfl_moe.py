# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.detectors import GFL
from mmdet.models.detectors.base import BaseDetector

from ..builder import ROTATED_DETECTORS


@ROTATED_DETECTORS.register_module()
class GFLMoE(GFL):
    """GFL with support for MoE backbone auxiliary gate losses."""

    def _unpack_backbone_output(self, backbone_out):
        if (
            isinstance(backbone_out, tuple)
            and len(backbone_out) == 2
            and isinstance(backbone_out[0], (list, tuple))
        ):
            return backbone_out[0], backbone_out[1]
        return backbone_out, None

    def extract_feat(self, img, return_gate_loss=False):
        backbone_out = self.backbone(img)
        x, gate_loss = self._unpack_backbone_output(backbone_out)
        if self.with_neck:
            x = self.neck(x)
        if return_gate_loss:
            return x, gate_loss
        return x

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        BaseDetector.forward_train(self, img, img_metas)
        x, gate_loss = self.extract_feat(img, return_gate_loss=True)
        losses = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        if gate_loss is not None:
            losses.update({"gate_loss": gate_loss})
        return losses
