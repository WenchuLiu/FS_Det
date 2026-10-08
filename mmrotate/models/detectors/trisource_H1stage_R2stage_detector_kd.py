# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional
import torch.nn as nn
import torch
import torch.nn.functional as F

from mmdet.models.losses.gfocal_loss import QualityFocalLoss

from mmdet.core import bbox2result, multi_apply, reduce_mean
from mmcv.runner import force_fp32
import numpy as np
from ..builder import ROTATED_DETECTORS, build_loss
from .trisource_H1stage_R2stage_detector import TriSourceDetector

from mmcv import Config
from mmrotate.models import build_detector
from mmcv.runner import load_checkpoint

from mmdet.models.losses.kd_loss import KnowledgeDistillationKLDivLoss
from mmdet.models.dense_heads.gfl_head import GFLHead

@ROTATED_DETECTORS.register_module()
class TriSourceKDDetector(TriSourceDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """ 

    def __init__(self,
                 *args,
                 teacher_cfg,
                 kd_cfg,
                 teacher_ckpt: Optional[str] = None,
                 **kwargs):
        super(TriSourceKDDetector, self).__init__(*args, **kwargs)

        cfg = Config.fromfile(teacher_cfg)
        self.teacher: TriSourceDetector = build_detector(cfg.model,
                                      train_cfg=cfg.get('train_cfg'),
                                      test_cfg=cfg.get('test_cfg'))
        if teacher_ckpt is not None:
            load_checkpoint(self.teacher, teacher_ckpt, map_location='cpu')
        self.freeze(self.teacher)
        self.loss_cls_kd = build_loss(kd_cfg['loss_cls_kd'])
        self.loss_reg_kd = build_loss(kd_cfg['loss_reg_kd'])
        self.loss_feat_kd = build_loss(kd_cfg['loss_feat_kd'])
        self.reused_teacher_head_idx = kd_cfg['reused_teacher_head_idx']
    
    
        
    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        assert gt_bboxes_ignore is None
        img = self.gather_dict_values(img) 
        img_metas = self.gather_dict_values(img_metas)
        gt_bboxes = self.gather_dict_values(gt_bboxes, ignore_tensor=True)
        gt_labels = self.gather_dict_values(gt_labels, ignore_tensor=True)

        sar_img_metas = img_metas['sar'] 
        rgb_img_metas = img_metas['rgb']
        ifr_img_metas = img_metas['ifr']

        sar_gt_bboxes = gt_bboxes['sar']
        rgb_gt_bboxes = gt_bboxes['rgb']
        ifr_gt_bboxes = gt_bboxes['ifr']

        sar_gt_labels = gt_labels['sar']
        rgb_gt_labels = gt_labels['rgb']
        ifr_gt_labels = gt_labels['ifr']

        self.dataset_names = ['sar', 'rgb', 'ifr']
        
        self.source_ratio = [len(sar_gt_labels), len(rgb_gt_labels), len(ifr_gt_labels)]
        self.teacher.source_ratio = self.source_ratio


        batch_inputs = []  
        for each in self.train_datasets:
            if len(img[each])>0:
                batch_inputs.append(img[each])

        # try:
        feats, gate_loss = self.extract_feat(batch_inputs, self.train_datasets, is_train=True) 
        teacher_feats, _ = self.teacher.extract_feat(batch_inputs, self.train_datasets, is_train=True)

        losses_feat_kd_total = {}    
        for i, (feat, teacher_feat) in enumerate(zip(feats, teacher_feats)):
            if self.dataset_names[i] != 'sar':
                continue
            losses_feat_kd = []
            for x, tea_x in zip(feat, teacher_feat):
                losses_feat_kd.append(self.loss_feat_kd(x, tea_x))
            losses_feat_kd_total[f'loss_feat_kd_{self.dataset_names[i]}'] = losses_feat_kd

        losses = dict()

        losses.update(losses_feat_kd_total)


        if gate_loss is not None:
            losses.update({'gate_loss': gate_loss})

        sar_x, rgb_x, ifr_x = feats
        sar_teacher_x, rgb_teacher_x, ifr_teacher_x = teacher_feats


        if len(sar_gt_labels) > 0:
            batch_input_shape = tuple(img['sar'][0].size()[-2:])
            for img_meta in sar_img_metas:
                img_meta['batch_input_shape'] = batch_input_shape

            stu_cls_scores, stu_bbox_preds, stu_cls_hold, stu_reg_hold = multi_apply(
                self.forward_crosskd_single, sar_x, self.sar_bbox_head.scales, bbox_head=self.sar_bbox_head
            )
            tea_cls_scores, tea_bbox_preds, tea_cls_hold, tea_reg_hold = multi_apply(
                self.forward_crosskd_single,
                sar_teacher_x, 
                self.teacher.sar_bbox_head.scales, 
                bbox_head=self.teacher.sar_bbox_head
            )

            reused_cls_scores, reused_bbox_preds = multi_apply(self.reuse_teacher_head,
                tea_cls_hold, tea_reg_hold, stu_cls_hold, stu_reg_hold, self.sar_bbox_head.scales)

            single_stage_losses = self.sar_loss_by_feat(
                tea_cls_scores, 
                tea_bbox_preds, 
                stu_cls_scores, 
                stu_bbox_preds,
                reused_cls_scores, 
                reused_bbox_preds, 
                sar_gt_bboxes, 
                sar_gt_labels,
                sar_img_metas, 
                gt_bboxes_ignore=gt_bboxes_ignore)

            if single_stage_losses is not None:
                losses.update({'sar_' + k: v for k, v in single_stage_losses.items()})

        if len(rgb_gt_labels) > 0:
            if self.with_rgb_rpn:
                proposal_cfg = self.rgb_train_cfg.get('rpn_proposal',
                                                self.rgb_test_cfg.rpn)
                rpn_losses, proposal_list = self.rgb_rpn_head.forward_train(
                    rgb_x,
                    rgb_img_metas,
                    rgb_gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
                losses.update({'rgb_' + k: v for k, v in rpn_losses.items()})
            else:
                proposal_list = proposals

            roi_losses = self.rgb_roi_head.forward_train(rgb_x, rgb_img_metas, proposal_list,
                                                    rgb_gt_bboxes, rgb_gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            losses.update({'rgb_' + k: v for k, v in roi_losses.items()})

        if len(ifr_gt_labels) > 0:
            if self.with_ifr_rpn:
                proposal_cfg = self.ifr_train_cfg.get('rpn_proposal',
                                                self.ifr_test_cfg.rpn)
                rpn_losses, proposal_list = self.ifr_rpn_head.forward_train(
                    ifr_x,
                    ifr_img_metas,
                    ifr_gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
                losses.update({'ifr_' + k: v for k, v in rpn_losses.items()})
            else:
                proposal_list = proposals

            roi_losses = self.ifr_roi_head.forward_train(ifr_x, ifr_img_metas, proposal_list,
                                                    ifr_gt_bboxes, ifr_gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            losses.update({'ifr_' + k: v for k, v in roi_losses.items()})


        loss_sum = 0 
        if self.multi_tasks_reweight is not None:
            reweight_losses = {}
            cur_losses = []
            for i, (k, loss) in enumerate(losses.items()):
                if k not in self.reweight_losses:
                    reweight_losses.update({k:loss})
                    continue
                elif isinstance(loss, list):
                    loss = sum(loss)
                cur_losses.append(loss)
            cur_losses = torch.stack(cur_losses) 

        if self.multi_tasks_reweight=='uncertainty':
            for i, loss in enumerate(cur_losses):
                loss_sum += 0.5 / (self.mtl_sigma[i] ** 2) * loss + torch.log(1 + self.mtl_sigma[i] ** 2)   
            reweight_losses.update({'reweighted_total_losses':loss_sum})
            return reweight_losses
        
        elif self.multi_tasks_reweight=='dwa':   
            
            if self.history_loss is not None:
                w_i = cur_losses/torch.tensor(self.history_loss).cuda()
                batch_weight = len(self.reweight_losses)*torch.nn.functional.softmax(w_i/self.T, dim=-1)
            else:
                batch_weight = torch.ones_like(cur_losses).cuda()
            loss_sum = torch.mul(cur_losses, batch_weight).sum()
        
            reweight_losses.update({'reweighted_total_losses':loss_sum})
            self.history_loss = cur_losses.detach().cpu().numpy()
            
            return reweight_losses
        

        return losses
    
    def forward_crosskd_single(self, x, scale, bbox_head: GFLHead):
        """Forward feature of a single scale level, aligned with GFL forward_single.
        
        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            bbox_head (GFLHead): The GFL head module.
                
        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
                cls_feat_hold (Tensor): Intermediate classification features for KD.
                reg_feat_hold (Tensor): Intermediate regression features for KD.
        """
        cls_feat = x
        reg_feat = x
        cls_feat_hold, reg_feat_hold = x, x
        
        # Process classification features - aligned with GFL forward_single
        for i, cls_conv in enumerate(bbox_head.cls_convs):
            cls_feat = cls_conv(cls_feat)  # Apply conv + activation directly like GFL
            # Save intermediate features for knowledge distillation
            if i + 1 == self.reused_teacher_head_idx:
                cls_feat_hold = cls_feat
        
        # Process regression features - aligned with GFL forward_single  
        for i, reg_conv in enumerate(bbox_head.reg_convs):
            reg_feat = reg_conv(reg_feat)  # Apply conv + activation directly like GFL
            # Save intermediate features for knowledge distillation
            if i + 1 == self.reused_teacher_head_idx:
                reg_feat_hold = reg_feat
        
        # Final predictions - exactly same as GFL forward_single
        cls_score = bbox_head.gfl_cls(cls_feat)
        bbox_pred = scale(bbox_head.gfl_reg(reg_feat)).float()
        
        return cls_score, bbox_pred, cls_feat_hold, reg_feat_hold

    def reuse_teacher_head(self, tea_cls_feat, tea_reg_feat, stu_cls_feat,
                           stu_reg_feat, scale):
        """Reuse teacher head with aligned forward process.
        
        Args:
            tea_cls_feat (Tensor): Teacher classification features.
            tea_reg_feat (Tensor): Teacher regression features.
            stu_cls_feat (Tensor): Student classification features.
            stu_reg_feat (Tensor): Student regression features.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module.
                
        Returns:
            tuple:
                reused_cls_score (Tensor): Reused classification scores.
                reused_bbox_pred (Tensor): Reused bbox predictions.
        """
        # Align student features with teacher features
        reused_cls_feat = self.align_scale(stu_cls_feat, tea_cls_feat)
        reused_reg_feat = self.align_scale(stu_reg_feat, tea_reg_feat)
        
        # Apply activation if needed (aligned with GFL forward_single)
        if self.reused_teacher_head_idx != 0:
            reused_cls_feat = F.relu(reused_cls_feat)
            reused_reg_feat = F.relu(reused_reg_feat)

        # Continue with remaining teacher head layers - aligned with GFL forward_single
        module = self.teacher.sar_bbox_head
        for i in range(self.reused_teacher_head_idx, module.stacked_convs):
            reused_cls_feat = module.cls_convs[i](reused_cls_feat)  # Apply conv + activation directly
            reused_reg_feat = module.reg_convs[i](reused_reg_feat)  # Apply conv + activation directly
        
        # Final predictions - exactly same as GFL forward_single
        reused_cls_score = module.gfl_cls(reused_cls_feat)
        reused_bbox_pred = scale(module.gfl_reg(reused_reg_feat)).float()
        
        return reused_cls_score, reused_bbox_pred

    def align_scale(self, stu_feat, tea_feat):
        N, C, H, W = stu_feat.size()
        # normalize student feature
        stu_feat = stu_feat.permute(1, 0, 2, 3).reshape(C, -1)
        stu_mean = stu_feat.mean(dim=-1, keepdim=True)
        stu_std = stu_feat.std(dim=-1, keepdim=True)
        stu_feat = (stu_feat - stu_mean) / (stu_std + 1e-6)
        #
        tea_feat = tea_feat.permute(1, 0, 2, 3).reshape(C, -1)
        tea_mean = tea_feat.mean(dim=-1, keepdim=True)
        tea_std = tea_feat.std(dim=-1, keepdim=True)
        stu_feat = stu_feat * tea_std + tea_mean
        return stu_feat.reshape(C, N, H, W).permute(1, 0, 2, 3)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def sar_loss_by_feat(
            self,
            tea_cls_scores,
            tea_bbox_preds,
            cls_scores,
            bbox_preds,
            reused_cls_scores,
            reused_bbox_preds,
            gt_bboxes,
            gt_labels,
            img_metas,
            gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.sar_bbox_head.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.sar_bbox_head.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.sar_bbox_head.cls_out_channels if self.sar_bbox_head.use_sigmoid_cls else 1


        cls_reg_targets = self.sar_bbox_head.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_dfl,\
            avg_factor = multi_apply(
                self.sar_bbox_head.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.sar_bbox_head.prior_generator.strides,
                num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor)
        avg_factor = max(float(reduce_mean(avg_factor)), 1.0)
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

        losses = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

        losses_cls_kd, losses_reg_kd, kd_avg_factor = multi_apply(
            self.pred_mimicking_loss_single,
            tea_cls_scores,
            tea_bbox_preds,
            reused_cls_scores,
            reused_bbox_preds,
            label_weights_list,
            avg_factor=avg_factor)
        kd_avg_factor = sum(kd_avg_factor)
        losses_reg_kd = list(map(lambda x: x / kd_avg_factor, losses_reg_kd))
        losses.update(dict(loss_cls_kd=losses_cls_kd, loss_reg_kd=losses_reg_kd))
        return losses

    def pred_mimicking_loss_single(self, tea_cls_score, tea_bbox_pred, reused_cls_score, reused_bbox_pred, label_weights, avg_factor):
        # classification branch distillation
        tea_cls_score = tea_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.sar_bbox_head.cls_out_channels)
        reused_cls_score = reused_cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.sar_bbox_head.cls_out_channels)
        label_weights = label_weights.reshape(-1)
        loss_cls_kd = self.loss_cls_kd(
            reused_cls_score,
            tea_cls_score.detach(),
            label_weights,
            avg_factor=avg_factor)
        # regression branch distillation
        reg_max = self.sar_bbox_head.reg_max
        tea_bbox_pred = tea_bbox_pred.permute(0, 2, 3,
                                              1).reshape(-1, reg_max + 1)
        reused_bbox_pred = reused_bbox_pred.permute(0, 2, 3, 1).reshape(
            -1, reg_max + 1)
        reg_weights = tea_cls_score.max(dim=1)[0].sigmoid()
        # Use torch.where instead of in-place assignment
        reg_weights = torch.where(label_weights == 0, torch.zeros_like(reg_weights), reg_weights)
        loss_reg_kd = self.loss_reg_kd(
            reused_bbox_pred,
            tea_bbox_pred.detach(),
            weight=reg_weights[:, None].expand(-1, 4).reshape(-1)) / 4

        return loss_cls_kd, loss_reg_kd, reg_weights.sum()