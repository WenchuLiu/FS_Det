import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule, Scale
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, bbox_overlaps, build_assigner,
                        build_sampler, images_to_levels, multi_apply,
                        reduce_mean, unmap)
from mmdet.core.utils import filter_scores_and_topk
from ..builder import ROTATED_HEADS, build_loss # 注意这里是用 HEADS 还是 ROTATED_HEADS，根据你的库决定
from mmdet.models.dense_heads.anchor_head import AnchorHead
from inspect import signature 
from mmcv.ops import batched_nms

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution."""
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
class TextGuidedGFLAlignHead(AnchorHead):
    """
    Text-Guided GFL Head with Prototype Alignment and CLIP Distillation.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 text_dim=512,  # [Text] Text feature dimension
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
                 # 🔥 [Distill] New config for distillation
                 loss_distill=dict(type='MSELoss', loss_weight=1.0),
                 **kwargs):
        
        # [Align] 获取负样本采样数
        self.neg_sample_num = kwargs.pop('neg_sample_num', 32)
        
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        self.text_dim = text_dim # [Text]
        
        super(TextGuidedGFLAlignHead, self).__init__(
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
        
        # [Text] 温度系数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 🔥 [新增] 限制最大温度，防止梯度爆炸 (比如限制在 4.6，即 exp(4.6)=100)
        self.max_logit_scale = 4.6

        # 🔥 [Distill] Distillation Projection Layer (Student -> Teacher)
        # Assuming Teacher (CLIP) is 512 dim. If Student (feat_channels) != 512, project it.
        # Ideally, both are 512 so Identity is used.
        self.loss_distill_weight = loss_distill.get('loss_weight', 1.0)
        
        # We assume teacher_dim is same as text_dim (e.g. 512)
        teacher_dim = self.text_dim 
        
        # ✅ 修正：将输入维度改为 self.text_dim
        # 此时 self.text_dim (512) == teacher_dim (512)，条件不成立，
        # 代码会正确地进入 else 分支，使用 Identity()
        if self.text_dim != teacher_dim:
            self.distill_proj = nn.Linear(self.text_dim, teacher_dim)
            nn.init.kaiming_uniform_(self.distill_proj.weight, a=1)
            nn.init.constant_(self.distill_proj.bias, 0)
        else:
            self.distill_proj = nn.Identity()

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
        
        # [Text] 修改分类卷积：输出 text_dim
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.text_dim, 3, padding=1)
        
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * (self.reg_max + 1), 3, padding=1)
        
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

    def forward(self, feats, text_embeds):
        """
        Args:
            feats (tuple[Tensor]): Image features.
            text_embeds (Tensor): [K, text_dim] embeddings.
        """
        # [Text] 必须传入 text_embeds
        return multi_apply(self.forward_single, feats, self.scales, text_embeds=text_embeds)

    def forward_single(self, x, scale, text_embeds):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
            
        # [Text] 1. 生成 Visual Embeddings [B, text_dim, H, W]
        img_feat = self.gfl_cls(cls_feat)
        
        # [Text] 2. 归一化 (L2 Norm) - 非常重要，为了 Prototype 质量
        img_feat_norm = img_feat / (img_feat.norm(dim=1, keepdim=True) + 1e-6)
        
        # [Text] 3. 归一化 Text Embeddings
        text_embeds_norm = text_embeds / (text_embeds.norm(dim=-1, keepdim=True) + 1e-6)
        
        # [Text] 4. 计算 Scores (Dot Product)
        # [B, C, H, W] x [K, C] -> [B, K, H, W]
        cls_score = torch.einsum('bdhw, kd -> bkhw', img_feat_norm, text_embeds_norm)
        cls_score = cls_score * self.logit_scale.exp().clamp(min=1.0,max=100.0)  # 🔥 [新增] 限制最大温度，防止梯度爆炸
        
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        
        # [Align] 返回 img_feat_norm 作为 cls_feat
        # 这样后续提取的 Prototype 就是归一化后的特征，方便做 Cosine Loss
        return cls_score, bbox_pred, img_feat_norm
    
    def simple_test(self, feats, img_metas, rescale=False, text_embeds=None):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
            text_embeds (Tensor): [K, C] Text embeddings.
        """
        assert text_embeds is not None, "TextGuidedHead requires 'text_embeds' during inference!"
        
        # 🔥 [透传] text_embeds 给 simple_test_bboxes
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale, text_embeds=text_embeds)

    def simple_test_bboxes(self, feats, img_metas, rescale=False, text_embeds=None):
        """Test det bboxes without test-time augmentation."""
        
        # 1. 🔥 [关键修改] 执行 forward (传入 text_embeds)
        # outs 返回: (cls_scores, bbox_preds, img_feat_norm)
        outs = self.forward(feats, text_embeds=text_embeds)
        
        # 2. 解包结果 (忽略 img_feat_norm，推理不需要)
        # get_bboxes 只需要前两个: cls_scores, bbox_preds
        # 注意：这里的 outs 是 tuple，我们取前两个元素
        bbox_inputs = outs[:2] 
        
        # 3. 生成检测框
        results_list = self.get_bboxes(
            *bbox_inputs, img_metas=img_metas, rescale=rescale)
            
        return results_list
    def aug_test(self, feats, img_metas, rescale=False, text_embeds=None):
        """Test function with test time augmentation.
        
        Args:
            text_embeds (Tensor): [K, C] Text embeddings.
        """
        assert text_embeds is not None, "TextGuidedHead requires 'text_embeds' during inference!"
        
        # 🔥 [透传] text_embeds
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale, text_embeds=text_embeds)

    def aug_test_bboxes(self, feats, img_metas, rescale=False, text_embeds=None):
        """Test det bboxes with test time augmentation."""
        
        # 1. 检查 get_bboxes 是否支持 with_nms 参数 (保持原有安全性检查)
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            
            # 🔥 [关键修改] 传入 text_embeds
            outs = self.forward(x, text_embeds=text_embeds)
            
            # 获取结果 (with_nms=False 表示只解码不筛选，等待合并)
            bbox_outputs = self.get_bboxes(
                *outs,
                img_metas=img_meta,
                cfg=self.test_cfg,
                rescale=False,
                with_nms=False)[0]
            
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])
            if len(bbox_outputs) >= 3:
                aug_labels.append(bbox_outputs[2])

        # 2. 合并增强结果 (使用父类的 merge_aug_bboxes)
        # 注意：merge_aug_bboxes 会处理翻转、旋转后的框坐标还原
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)
            
        merged_labels = torch.cat(aug_labels, dim=0) if aug_labels else None

        if merged_bboxes.numel() == 0:
            det_bboxes = torch.cat([merged_bboxes, merged_scores[:, None]], -1)
            return [
                (det_bboxes, merged_labels),
            ]

        # 3. 执行 Batched NMS
        # 🔥 [注意] 如果之前的 forward 拼接了背景类，这里可能需要切片
        # 但是通常 get_bboxes 内部已经根据 num_classes 处理过了
        
        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                            merged_labels, self.test_cfg.nms)
        det_bboxes = det_bboxes[:self.test_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.test_cfg.max_per_img]

        # 4. Rescale 到原图尺寸
        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        return [
            (_det_bboxes, det_labels),
        ]

    def anchor_center(self, anchors):
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)
    
    # 🔥 [新增] 覆盖父类方法，接收 distill_feats
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      text_embeds=None,    # Receive text embeddings
                      distill_feats=None,  # Receive teacher features
                      **kwargs):
        """
        Override forward_train to support passing text_embeds to forward()
        and distill_feats to loss().
        """
        # 1. Call forward with text_embeds
        # x is the image features (tuple)
        outs = self(x, text_embeds=text_embeds) 
        
        # 2. Prepare inputs for loss()
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            
        # 3. Call loss with additional arguments
        losses = self.loss(*loss_inputs, 
                           gt_bboxes_ignore=gt_bboxes_ignore,
                           distill_feats=distill_feats) # Pass distill_feats here
        
        return losses

    def loss_single(self, anchors, cls_score, bbox_pred, cls_feat, labels, label_weights,
                    bbox_targets, gt_bboxes, teacher_feats, stride, num_total_samples):
        """
        Args:
            teacher_feats (Tensor | None): 单张图的 Teacher 特征，可能为 None
        """
        # ... (前面的 reshape 和索引获取代码保持不变) ...
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        current_num_classes = cls_score.shape[1]
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, current_num_classes)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
        neg_inds = ((labels == bg_class_ind) & (label_weights > 0)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        B, C, H, W = cls_feat.shape
        cls_feat_flat = cls_feat.permute(0, 2, 3, 1).reshape(-1, C)

        # 初始化返回值
        pos_prototypes = cls_feat_flat.new_zeros((0, C))
        pos_proto_labels = labels.new_zeros((0,), dtype=labels.dtype)
        
        loss_distill = torch.tensor(0.0, device=cls_feat.device)
        
        # ==================== 1. 判断是否满足蒸馏条件 ====================
        # 条件：有正样本 AND 接收到了 Teacher 特征 AND Teacher 特征不为空
        can_distill = (len(pos_inds) > 0) and \
                      (teacher_feats is not None) and \
                      (teacher_feats.numel() > 0)

        # ==================== 2. 正样本处理 ====================
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            # 提取正样本特征 (用于 Prototype 和 Distill)
            raw_pos_feats = cls_feat_flat[pos_inds]
            raw_pos_labels = labels[pos_inds]

            # 🔥 [分支 A] 执行蒸馏
            if can_distill:
                # 1. 投影 + 归一化 Student 特征
                student_feats = self.distill_proj(raw_pos_feats)
                student_feats_norm = F.normalize(student_feats, dim=-1)

                # 2. 匹配 GT
                pos_centers = (anchors[pos_inds, :2] + anchors[pos_inds, 2:]) / 2.0
                if gt_bboxes.shape[-1] == 5: 
                     gt_centers = gt_bboxes[:, :2]
                else: 
                     gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:4]) / 2.0
                
                dists = torch.cdist(pos_centers, gt_centers)
                _, match_gt_inds = dists.min(dim=1)
                
                # 3. 取出对应的 Teacher 特征
                target_feats = teacher_feats[match_gt_inds]
                
                # 4. 计算 Loss
                sim = (student_feats_norm * target_feats).sum(dim=1)
                loss_distill = (1.0 - sim).mean() * self.loss_distill_weight

            # 常规 BBox Loss 计算 (保持不变)
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
            
            # Prototype 聚合 (保持不变)
            gt_coords_f32 = pos_decode_bbox_targets.float()
            gt_keys = (gt_coords_f32 * 1000).long()
            unique_gts, inverse_indices = torch.unique(gt_keys, dim=0, return_inverse=True)
            num_unique_instances = unique_gts.size(0)
            agg_feats_list = []
            agg_labels_list = []
            for i in range(num_unique_instances):
                mask = (inverse_indices == i)
                instance_feat = raw_pos_feats[mask].mean(dim=0)
                instance_label = raw_pos_labels[mask][0]
                agg_feats_list.append(instance_feat)
                agg_labels_list.append(instance_label)
            pos_prototypes = torch.stack(agg_feats_list)
            pos_proto_labels = torch.stack(agg_labels_list)

        else:
            # 无正样本时的常规处理
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # ==================== 3. 统一处理不满足蒸馏条件的情况 ====================
        # 🔥 [分支 B] 梯度占位 (为了防止 DDP 报错)
        # 如果没有执行蒸馏 (可能是因为没正样本，也可能是因为没 teacher_feats)
        if not can_distill:
            # 建立 cls_feat 的梯度连接
            loss_distill = loss_distill + cls_feat.sum() * 0.0
            # 建立 distill_proj 权重的梯度连接 (防止 DDP 认为该层参数未使用)
            if hasattr(self.distill_proj, 'weight'):
                loss_distill = loss_distill + self.distill_proj.weight.sum() * 0.0

        # ==================== 4. 负样本处理 (Hard Mining) ====================
        if len(neg_inds) > 0:
            if len(neg_inds) > self.neg_sample_num:
                neg_scores = cls_score[neg_inds].sigmoid().max(dim=1)[0]
                _, topk_idx = neg_scores.topk(self.neg_sample_num)
                sampled_neg_inds = neg_inds[topk_idx]
            else:
                sampled_neg_inds = neg_inds
            
            neg_prototypes = cls_feat_flat[sampled_neg_inds]
            neg_proto_labels = labels[sampled_neg_inds] 
        else:
            neg_prototypes = cls_feat_flat.new_zeros((0, C))
            neg_proto_labels = labels.new_zeros((0,), dtype=labels.dtype)

        # cls loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, loss_distill, weight_targets.sum(), \
               pos_prototypes, pos_proto_labels, neg_prototypes, neg_proto_labels

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             cls_feats,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None,
             distill_feats=None): # 🔥 接收参数
        """
        Args:
            distill_feats (list[Tensor] | None): Teacher features from forward_train.
        """
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

        # 🔥 [分支判断核心] 
        # 如果 forward_train 没有传 distill_feats (为None)，
        # 构造一个全为 None 的列表，长度等于 batch size
        if distill_feats is None:
            distill_feats = [None] * len(img_metas)

        # multi_apply 会自动把 distill_feats 列表拆开，传给 loss_single 的 teacher_feats
        losses_cls, losses_bbox, losses_dfl, losses_distill, \
        avg_factor, pos_feats_list, pos_labels_list, \
        neg_feats_list, neg_labels_list = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            cls_feats,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            gt_bboxes,
            distill_feats,  # <--- 这里传入处理好的列表
            self.prior_generator.strides,
            num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        
        return dict(
            loss_cls=losses_cls, 
            loss_bbox=losses_bbox, 
            loss_dfl=losses_dfl,
            loss_distill=losses_distill,
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
