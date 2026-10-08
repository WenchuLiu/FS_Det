# Copyright (c) OpenMMLab. All rights reserved.
# 添加获取prototype区域的代码，在forward train中会返回roi部分的feats和labels，对应tri_align_detector共同使用
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, bbox_overlaps, build_assigner,
                        build_sampler, images_to_levels, multi_apply,
                        reduce_mean, unmap)
from mmdet.core.utils import filter_scores_and_topk
from ..builder import ROTATED_HEADS, build_loss
from mmdet.models.dense_heads.anchor_head import AnchorHead


class Integral(nn.Module):
    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x

@ROTATED_HEADS.register_module()
class GFLAlignHeadV3(AnchorHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 bbox_coder=dict(type='DistancePointBBoxCoder'),
                 reg_max=16,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='gfl_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.neg_sample_num = kwargs.pop('neg_sample_num', 32)
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        super(GFLAlignHeadV3, self).__init__(
            num_classes,
            in_channels,
            bbox_coder=bbox_coder,
            init_cfg=init_cfg,
            **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.integral = Integral(self.reg_max)
        self.loss_dfl = build_loss(loss_dfl)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.gfl_cls(cls_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred, cls_feat

    def anchor_center(self, anchors):
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    # =========================================================================
    # 🔥 [新增] 核心函数：基于 MAP (Masked Average Pooling) 提取原型
    # =========================================================================
    def get_prototypes_via_map(self, cls_feat, gt_bboxes_list, gt_labels_list, stride):
        """
        Args:
            cls_feat (Tensor): [B, C, H, W] 当前层特征图
            gt_bboxes_list (list[Tensor]): Batch 中每张图的 GT 框
            gt_labels_list (list[Tensor]): Batch 中每张图的 GT 标签
            stride (int): 当前层下采样倍率
        """
        B, C, H, W = cls_feat.shape
        device = cls_feat.device
        
        # 1. 生成网格坐标 (用于生成 Mask)
        # 注意: meshgrid 参数顺序 (y, x)
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device)
        )
        # [1, 1, H, W]
        xx = xx[None, None, :, :].float()
        yy = yy[None, None, :, :].float()

        pos_prototypes = []
        pos_proto_labels = []
        neg_prototypes = []
        neg_proto_labels = [] # 背景标签通常不需要，或者是 ignore_index

        for b_idx in range(B):
            feat = cls_feat[b_idx] # [C, H, W]
            gts = gt_bboxes_list[b_idx] # [N_gt, 4]
            labels = gt_labels_list[b_idx] # [N_gt]

            if gts.numel() == 0:
                # 只有背景
                bg_proto = feat.mean(dim=(1, 2)) # 全局平均
                neg_prototypes.append(bg_proto)
                continue

            # 缩放 GT 到特征图尺度
            scaled_gts = gts / stride

            # 2. 生成前景 Mask (Broadcasting)
            # scaled_gts: [N, 4] -> [N, 1, 1] for compare
            x1 = scaled_gts[:, 0].view(-1, 1, 1)
            y1 = scaled_gts[:, 1].view(-1, 1, 1)
            x2 = scaled_gts[:, 2].view(-1, 1, 1)
            y2 = scaled_gts[:, 3].view(-1, 1, 1)

            # instance_masks: [N_gt, H, W]
            # 只有像素中心落在 bbox 内才算前景 (或者 >= x1 & <= x2)
            # 这里简单处理，使用 grid 坐标比较
            instance_masks = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
            instance_masks = instance_masks.squeeze(1).float() 

            # 3. 提取前景原型 (P_f)
            # 对每个实例计算 MAP
            for i in range(len(gts)):
                mask = instance_masks[i] # [H, W]
                area = mask.sum()
                if area > 0:
                    # MAP: sum(feat * mask) / sum(mask)
                    proto = (feat * mask).sum(dim=(1, 2)) / area
                    pos_prototypes.append(proto)
                    pos_proto_labels.append(labels[i])
            
            # 4. 提取背景原型 (P_b)
            # 计算全局前景掩码 (Union of all boxes)
            global_fg_mask = instance_masks.max(dim=0)[0] # [H, W]
            global_bg_mask = 1.0 - global_fg_mask

            bg_area = global_bg_mask.sum()
            if bg_area > 0:
                bg_proto = (feat * global_bg_mask).sum(dim=(1, 2)) / bg_area
                neg_prototypes.append(bg_proto)
                # neg_proto_labels.append(self.num_classes) # 可选：添加背景标签

        # Stack results
        if len(pos_prototypes) > 0:
            pos_prototypes = torch.stack(pos_prototypes)
            pos_proto_labels = torch.stack(pos_proto_labels)
        else:
            pos_prototypes = cls_feat.new_zeros((0, C))
            pos_proto_labels = cls_feat.new_zeros((0,), dtype=torch.long)

        if len(neg_prototypes) > 0:
            neg_prototypes = torch.stack(neg_prototypes)
            # 这里为了保持一致，生成全0的背景label占位，或者你需要具体的背景类ID
            neg_proto_labels = pos_proto_labels.new_full((len(neg_prototypes),), self.num_classes)
        else:
            neg_prototypes = cls_feat.new_zeros((0, C))
            neg_proto_labels = pos_proto_labels.new_zeros((0,))

        return pos_prototypes, pos_proto_labels, neg_prototypes, neg_proto_labels

    def loss_single(self, anchors, cls_score, bbox_pred, cls_feat, labels, label_weights,
                    bbox_targets, stride, num_total_samples, 
                    gt_bboxes_list, gt_labels_list): # 🔥 [修改 2] 接收 GTs
        """
        Compute loss of a single scale level.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # =====================================================================
        # 🔥 [修改 3] 使用 MAP 提取 Prototype (替换原有的 Instance Aggregation)
        # =====================================================================
        pos_prototypes, pos_proto_labels, neg_prototypes, neg_proto_labels = \
            self.get_prototypes_via_map(cls_feat, gt_bboxes_list, gt_labels_list, stride[0])
        # =====================================================================

        # 计算 GFL Loss (保持不变)
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)
        
        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum(), \
               pos_prototypes, pos_proto_labels, neg_prototypes, neg_proto_labels

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             cls_feat,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head."""

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        
        cls_reg_targets = self.get_targets(
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

        # 🔥 [修改 1] 准备 multi_apply 参数
        # loss_single 是针对每一层金字塔调用的，但我们需要用到全局的 GT
        # 因此需要把 gt_bboxes 和 gt_labels 重复 num_levels 次传进去
        num_levels = len(self.prior_generator.strides)
        gt_bboxes_list_repeated = [gt_bboxes for _ in range(num_levels)]
        gt_labels_list_repeated = [gt_labels for _ in range(num_levels)]

        losses_cls, losses_bbox, losses_dfl, \
        avg_factor, pos_feats_list, pos_labels_list, \
        neg_feats_list, neg_labels_list = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            cls_feat, # 注意：这里如果 cls_feat 是 tuple，multi_apply 会自动拆解
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.prior_generator.strides,
            num_total_samples=num_total_samples,
            gt_bboxes_list=gt_bboxes_list_repeated, # 👈 传入重复的列表
            gt_labels_list=gt_labels_list_repeated  # 👈 传入重复的列表
        )

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        
        return dict(
            loss_cls=losses_cls, 
            loss_bbox=losses_bbox, 
            loss_dfl=losses_dfl,
            proto_feats=pos_feats_list, 
            proto_labels=pos_labels_list,
            neg_proto_feats=neg_feats_list,
            neg_proto_labels=neg_labels_list
        )
        
    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. GFL head does not need this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list,
                    self.prior_generator.strides, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            bboxes = self.bbox_coder.decode(
                self.anchor_center(priors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        return self._bbox_post_process(
            mlvl_scores,
            mlvl_labels,
            mlvl_bboxes,
            img_meta['scale_factor'],
            cfg,
            rescale=rescale,
            with_nms=with_nms)

    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of positive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside
