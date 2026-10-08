# Copyright (c) OpenMMLab. All rights reserved.
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
from mmdet.core.utils import filter_scores_and_topk, select_single_mlvl



class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4) # type: ignore
        return x


@ROTATED_HEADS.register_module()
class GFLFewshotHead(AnchorHead):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    """
    # 【修改】__init__函数：增加了 num_novel_classes 参数
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
                     override=[dict(
                         type='Normal',
                         name='gfl_cls', # 【修改】确保两个分类头都能被正确初始化
                         std=0.01,
                         bias_prob=0.01),
                               dict(
                        type='Normal',
                        name='gfl_cls_novel',  # 为 gfl_cls_novel 添加同样的初始化
                        std=0.01,
                        bias_prob=0.01)]),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        # 【新增】保存新类数量
        self.num_novel_classes = 5
        
        # 【修改】父类的 num_classes 现在代表基类的数量
        super(GFLFewshotHead, self).__init__(
            num_classes,
            in_channels,
            bbox_coder=bbox_coder,
            init_cfg=init_cfg,
            **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.integral = Integral(self.reg_max)
        self.loss_dfl = build_loss(loss_dfl)

    # 【修改】_init_layers函数：增加了 gfl_cls_novel 层
    def _init_layers(self):
        """Initialize layers of the head."""
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
        
        # 【修改】基类分类头，输出通道数为基类数量
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
            
        # 【新增】新类分类头，输出通道数为新类数量
        if self.num_novel_classes > 0:
            self.gfl_cls_novel = nn.Conv2d(
                self.feat_channels, self.num_novel_classes, 3, padding=1)

        # 回归头保持不变，因为回归是类别无关的
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    # 【修改】forward函数：现在返回三个结果
    def forward(self, feats):
        """Forward features from the upstream network."""
        # 【修改】multi_apply现在会返回三个列表
        return multi_apply(self.forward_single, feats, self.scales)

    # 【修改】forward_single函数：分别通过两个分类头
    def forward_single(self, x, scale):
        """Forward feature of a single scale level."""
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        
        # 基类分数
        cls_score_base = self.gfl_cls(cls_feat)
        
        # 新类分数
        cls_score_novel = None
        if self.num_novel_classes > 0:
            cls_score_novel = self.gfl_cls_novel(cls_feat)
            
        # 边界框预测（共享）
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        
        return cls_score_base, cls_score_novel, bbox_pred

    def anchor_center(self, anchors):
        """Get anchor centers from anchors."""
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    # 【重大修改】loss_single函数：分别计算基类和新类的损失
    def loss_single(self, 
                    anchors, 
                    cls_score_base, 
                    cls_score_novel, 
                    bbox_pred, 
                    labels,
                    label_weights, 
                    bbox_targets, 
                    stride, 
                    num_total_samples):
        """Compute loss of a single scale level."""
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        
        # Reshape & Permute a Tensors
        cls_score_base = cls_score_base.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        if cls_score_novel is not None:
            cls_score_novel = cls_score_novel.permute(0, 2, 3, 1).reshape(-1, self.num_novel_classes)

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1] for base, [num_classes, num_classes + num_novel_classes -1] for novel
        # BG cat_id: num_classes + num_novel_classes
        bg_class_ind = self.num_classes + self.num_novel_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        
        score = label_weights.new_zeros(labels.shape)
        
        loss_bbox = torch.tensor(0., device=bbox_pred.device)
        loss_dfl = torch.tensor(0., device=bbox_pred.device)
        weight_targets_sum = torch.tensor(0., device=bbox_pred.device)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]


            #=============================================SAE-FSDet
            # 统一计算所有正样本的IoU，作为QFL的target
            weight_targets = cls_score_base.detach().sigmoid()
            if cls_score_novel is not None:
                prob_is_novel = weight_targets[:, 1:2]
                # 如果有新类，需要拼接两个分类器的输出来找到max score
                weight_targets_novel = cls_score_novel.detach().sigmoid()
                weight_targets_novel = prob_is_novel * weight_targets_novel
                weight_targets[:, 1:2] = 0
                weight_targets = torch.cat([weight_targets, weight_targets_novel], dim=1)
            #=============================================SAE-FSDet End
            
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            weight_targets_sum = weight_targets.sum()

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
            # regression losses (GIoU and DFL)
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

        # --- 分别计算分类损失 ---
        # 整体的labels和scores用于QFL
        # (labels, score)是QFL的target，其中score是预测框和GT框的IoU
        qfl_targets = (labels, score)
        
        # 基类损失
        loss_cls_base = self.loss_cls(
            cls_score_base,
            qfl_targets,
            weight=label_weights,
            avg_factor=num_total_samples)
        
        # 新类损失
        loss_cls_novel = torch.tensor(0., device=bbox_pred.device)
        if cls_score_novel is not None:
            # 对于新类的QFL，需要将标签进行映射
            # 基类标签视为背景（-1），新类标签映射到[0, num_novel_classes-1]
            novel_labels = labels.clone() - self.num_classes
            # 将基类和原始背景标签都设为一个无效值，以便在loss中被忽略
            novel_labels[novel_labels < 0] = -1 
            novel_qfl_targets = (novel_labels, score)
            
            loss_cls_novel = self.loss_cls(
                cls_score_novel,
                novel_qfl_targets,
                weight=label_weights,
                avg_factor=num_total_samples)

        return loss_cls_base, loss_cls_novel, loss_bbox, loss_dfl, weight_targets_sum


    # 【重大修改】loss函数：处理多个输出并聚合损失
    @force_fp32(apply_to=('cls_scores_base', 'cls_scores_novel', 'bbox_preds'))
    def loss(self,
             cls_scores_base,
             cls_scores_novel,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head."""
        # print(cls_scores_base.shape, cls_scores_novel.shape, bbox_preds.shape)
        # print(gt_bboxes)
        
        # 【修改】由于forward返回了三个值，这里也需要对应接收
        cls_scores = cls_scores_base # 保持队形，便于后续处理

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        
        # 【修改】总类别数现在是基类和新类之和
        label_channels = self.cls_out_channels + self.num_novel_classes if self.use_sigmoid_cls else 1

        # print(f"loss function input gt_bboxes: {gt_bboxes}")

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
        
        # 如果没有新类，创建一个None的列表以匹配multi_apply的输入
        if self.num_novel_classes == 0:
            cls_scores_novel = [None] * len(cls_scores_base)

        # 【修改】调用修改后的loss_single
        losses_cls_base, losses_cls_novel, losses_bbox, losses_dfl, avg_factor = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores_base,
            cls_scores_novel,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.prior_generator.strides,
            num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        
        # 【修改】返回独立的分类损失
        losses = dict(
            loss_cls=losses_cls_base+losses_cls_novel,
            loss_bbox=losses_bbox,
            loss_dfl=losses_dfl)
            
        return losses
    
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        
        # if gt_labels is None:
        #     loss_inputs = outs + (gt_bboxes, img_metas)
        # else:
        #     loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        # import pdb
        # pdb.set_trace()
        
        losses = self.loss(
            *outs,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            img_metas=img_metas,
            gt_bboxes_ignore=gt_bboxes_ignore)
        # losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
        
    @force_fp32(apply_to=('cls_scores_base', 'cls_scores_novel', 'bbox_preds'))
    def get_bboxes(self,
                cls_scores_base,     # [修改] 接收基类分数
                cls_scores_novel,    # [修改] 接收新类分数
                bbox_preds,
                score_factors=None,
                img_metas=None,
                cfg=None,
                rescale=False,
                with_nms=True,
                **kwargs):
        """
        将一个批次的网络输出转换为边界框结果。
        (文档字符串保持不变)
        """
        
        # [修改] 将所有的 cls_scores 替换为 cls_scores_base 作为基准
        assert len(cls_scores_base) == len(bbox_preds)

        if score_factors is None:
            with_score_factors = False
        else:
            with_score_factors = True
            assert len(cls_scores_base) == len(score_factors)

        num_levels = len(cls_scores_base)

        featmap_sizes = [cls_scores_base[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores_base[0].dtype,
            device=cls_scores_base[0].device)

        result_list = []

        for img_id in range(len(img_metas)):
            img_meta = img_metas[img_id]

            # [核心修改] 分别提取 base 和 novel 的分数，然后打包
            cls_score_base_list = select_single_mlvl(cls_scores_base, img_id)
            cls_score_novel_list = select_single_mlvl(cls_scores_novel, img_id)
            packaged_cls_scores = (cls_score_base_list, cls_score_novel_list)

            # 提取其他预测（这部分保持不变）
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id)
            if with_score_factors:
                score_factor_list = select_single_mlvl(score_factors, img_id)
            else:
                score_factor_list = [None for _ in range(num_levels)]

            # [修改] 调用 _get_bboxes_single 时，传递打包好的分数
            results = self._get_bboxes_single(
                packaged_cls_scores, # <--- 使用打包后的元组
                bbox_pred_list,
                score_factor_list,
                mlvl_priors,
                img_meta,
                cfg,
                rescale,
                with_nms,
                **kwargs)
            result_list.append(results)
            
        return result_list


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
    
    # 【修改】_get_bboxes_single函数：拼接两个分类头的分数
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
        """Transform outputs of a single image into bbox predictions."""
        # 【修改】这里的cls_score_list实际上是 (cls_score_base_list, cls_score_novel_list)
        cls_score_base_list, cls_score_novel_list = cls_score_list
        
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1) # type: ignore

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        
        # 【修改】同时遍历基类和新类的分数列表
        zipped_scores = zip(cls_score_base_list, cls_score_novel_list) if self.num_novel_classes > 0 else zip(cls_score_base_list, [None]*len(cls_score_base_list))

        for level_idx, ((cls_score_base, cls_score_novel), bbox_pred, stride, priors) in enumerate(
                zip(zipped_scores, bbox_pred_list,
                    self.prior_generator.strides, mlvl_priors)):
            assert cls_score_base.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            scores_base = cls_score_base.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            
            # 【修改】拼接基类和新类的分数
            if self.num_novel_classes > 0 and cls_score_novel is not None:
                scores_novel = cls_score_novel.permute(1, 2, 0).reshape(
                    -1, self.num_novel_classes).sigmoid()
                # a. 获取“属于新类的概率”，即基类分类器对 category_id: 1 的打分
                prob_is_novel = scores_base[:, 1:2]
                final_scores_novel = scores_novel * prob_is_novel
                scores_base[:, 1:2] = 0
                scores = torch.cat([scores_base, final_scores_novel], dim=1)
            else:
                scores = scores_base

            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre, # type: ignore
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred'] # type: ignore
            priors = filtered_results['priors'] # type: ignore

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