# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ROTATED_LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


def kd_quality_focal_loss(pred,
                          target,
                          weight=None,
                          beta=1,
                          reduction='mean',
                          avg_factor=None):
    num_classes = pred.size(1)
    if weight is not None:
        weight = weight[:, None].repeat(1, num_classes)

    # Ensure target is detached and create a new tensor to avoid in-place issues
    target_sigmoid = torch.sigmoid(target.detach())
    loss = F.binary_cross_entropy_with_logits(pred, target_sigmoid, reduction='none')
    focal_weight = torch.abs(pred.sigmoid() - target_sigmoid).pow(beta)
    loss = loss * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss



@ROTATED_LOSSES.register_module()
class KDQualityFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(KDQualityFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss = self.loss_weight * kd_quality_focal_loss(
                pred,
                target,
                weight,
                beta=self.beta,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss