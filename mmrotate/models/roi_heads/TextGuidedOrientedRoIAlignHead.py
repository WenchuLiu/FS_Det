from mmrotate.core import rbbox2roi, rbbox2result
from ..builder import ROTATED_HEADS
from .rotate_standard_roi_head import RotatedStandardRoIHead
import torch

@ROTATED_HEADS.register_module()
class TextGuidedOrientedRoIAlignHead(RotatedStandardRoIHead):
    
    def __init__(self, **kwargs):
        super(TextGuidedOrientedRoIAlignHead, self).__init__(**kwargs)

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      text_embeds,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        
        # 1. Assign & Sampler (这部分保持原样，省略以节省空间...)
        if self.with_bbox:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                
                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]
                sampling_results.append(sampling_result)

        losses = dict()
        
        # 2. BBox Forward & Loss
        if self.with_bbox:
            # 🔥 [修改] 正确解包 3 个返回值
            bbox_results, prototypes, proto_labels = self._bbox_forward_train(
                x, sampling_results, gt_bboxes, gt_labels, img_metas, text_embeds
            )
            
            # 更新 loss
            losses.update(bbox_results['loss_bbox'])
            
            # 🔥 返回 3 个值给 Detector
            return losses, prototypes, proto_labels
            
        # 异常处理 (无 bbox head)
        device = x[0].device
        empty_dim = text_embeds.size(-1)
        return losses, torch.zeros((0, empty_dim), device=device), torch.zeros((0,), dtype=torch.long, device=device)

    def _bbox_forward(self, x, rois, text_embeds):
        """
        🔥 [修改] 接收 BBox Head 返回的 3 个值，并保存 visual_feat
        """
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        
        # 🔥 接收 visual_feat (512维)
        cls_score, bbox_pred, visual_feat = self.bbox_head(bbox_feats, text_embeds)

        bbox_results = dict(
            cls_score=cls_score, 
            bbox_pred=bbox_pred, 
            bbox_feats=bbox_feats,
            visual_feat=visual_feat # 🔥 存入字典，供下一步使用
        )
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,    
                            img_metas, text_embeds):
        
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        
        # 1. Forward (调用上面修改过的函数)
        bbox_results = self._bbox_forward(x, rois, text_embeds)

        # 2. Targets
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                gt_labels, self.train_cfg)
        
        # 3. Detection Loss
        loss_bbox = self.bbox_head.loss(
            bbox_results['cls_score'],
            bbox_results['bbox_pred'],
            rois,
            *bbox_targets
        )

        bbox_results.update(loss_bbox=loss_bbox)

        # =================================================================
        # 🔥 [新增] 从 visual_feat 中筛选正样本 (512维)
        # =================================================================
        labels_all = bbox_targets[0]
        # 背景类的 label 通常等于 num_classes，前景 < num_classes
        pos_mask = (labels_all >= 0) & (labels_all < self.bbox_head.num_classes)

        prototypes = None
        proto_labels = None

        if pos_mask.any():
            # 取出 BBox Head 算好的 512 维特征
            feats_aligned = bbox_results['visual_feat'] # [N_total, 512]
            
            # 筛选正样本
            prototypes = feats_aligned[pos_mask] # [N_pos, 512]
            proto_labels = labels_all[pos_mask]
        else:
            # 无正样本
            device = x[0].device
            # 维度直接取 text_embeds 的维度 (512)
            prototypes = torch.zeros((0, text_embeds.size(-1)), device=device)
            proto_labels = torch.zeros((0,), dtype=torch.long, device=device)

        # 🔥 返回 3 个值: 结果字典, 原型, 标签
        return bbox_results, prototypes, proto_labels

    # Inference 部分保持不变 (Simple Test)
    def simple_test(self, x, proposal_list, img_metas, rescale=False, text_embeds=None):
        assert self.with_bbox, 'Bbox head must be implemented.'
        
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, 
            rescale=rescale, text_embeds=text_embeds)

        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                         self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def simple_test_bboxes(self, x, img_metas, proposals, rcnn_test_cfg, rescale=False, text_embeds=None):
        rois = rbbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois, text_embeds)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        if bbox_pred is not None:
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels