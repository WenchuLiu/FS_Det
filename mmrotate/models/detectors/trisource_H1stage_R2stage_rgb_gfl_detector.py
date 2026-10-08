# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import bbox2result

from ..builder import ROTATED_DETECTORS, build_head
from .trisource_H1stage_R2stage_detector import TriSourceDetector


@ROTATED_DETECTORS.register_module()
class TriSourceRGBGFLDetector(TriSourceDetector):
    """Tri-source detector with one-stage GFL head for the RGB branch."""

    def __init__(self,
                 backbone,
                 neck=None,
                 rgb_rpn_head=None,
                 rgb_roi_head=None,
                 rgb_bbox_head=None,
                 rgb_train_cfg=None,
                 rgb_test_cfg=None,
                 ifr_rpn_head=None,
                 ifr_roi_head=None,
                 ifr_train_cfg=None,
                 ifr_test_cfg=None,
                 sar_bbox_head=None,
                 sar_train_cfg=None,
                 sar_test_cfg=None,
                 multi_tasks_reweight=None,
                 reweight_losses=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        if rgb_bbox_head is None:
            raise ValueError("rgb_bbox_head must be provided for TriSourceRGBGFLDetector.")
        if rgb_rpn_head is not None or rgb_roi_head is not None:
            raise ValueError(
                "TriSourceRGBGFLDetector does not use rgb_rpn_head or rgb_roi_head."
            )

        super(TriSourceRGBGFLDetector, self).__init__(
            backbone=backbone,
            neck=neck,
            rgb_rpn_head=None,
            rgb_roi_head=None,
            rgb_train_cfg=None,
            rgb_test_cfg=None,
            ifr_rpn_head=ifr_rpn_head,
            ifr_roi_head=ifr_roi_head,
            ifr_train_cfg=ifr_train_cfg,
            ifr_test_cfg=ifr_test_cfg,
            sar_bbox_head=sar_bbox_head,
            sar_train_cfg=sar_train_cfg,
            sar_test_cfg=sar_test_cfg,
            multi_tasks_reweight=multi_tasks_reweight,
            reweight_losses=reweight_losses,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )

        rgb_bbox_head_ = rgb_bbox_head.copy()
        rgb_bbox_head_.update(train_cfg=rgb_train_cfg)
        rgb_bbox_head_.update(test_cfg=rgb_test_cfg)
        self.rgb_bbox_head = build_head(rgb_bbox_head_)

        self.rgb_train_cfg = rgb_train_cfg
        self.rgb_test_cfg = rgb_test_cfg

    @property
    def with_rgb_bbox_head(self):
        """bool: whether the detector has a one-stage RGB bbox head."""
        return hasattr(self, "rgb_bbox_head") and self.rgb_bbox_head is not None

    def forward_dummy(self, img):
        """Used for computing network flops."""
        outs = ()

        x, _ = self.extract_feat(img, ["rgb"])
        rgb_outs = self.rgb_bbox_head.forward(x)
        outs = outs + (rgb_outs, )

        if hasattr(self, "ifr_rpn_head"):
            x, _ = self.extract_feat(img, ["ifr"])
            rpn_outs = self.ifr_rpn_head(x)
            outs = outs + (rpn_outs, )
            proposals = torch.randn(1000, 5).to(img.device)
            roi_outs = self.ifr_roi_head.forward_dummy(x, proposals)
            outs = outs + (roi_outs, )

        x, _ = self.extract_feat(img, ["sar"])
        sar_outs = self.sar_bbox_head.forward(x)
        outs = outs + (sar_outs, )
        return outs

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

        sar_img_metas = img_metas["sar"]
        rgb_img_metas = img_metas["rgb"]
        ifr_img_metas = img_metas["ifr"]

        sar_gt_bboxes = gt_bboxes["sar"]
        rgb_gt_bboxes = gt_bboxes["rgb"]
        ifr_gt_bboxes = gt_bboxes["ifr"]

        sar_gt_labels = gt_labels["sar"]
        rgb_gt_labels = gt_labels["rgb"]
        ifr_gt_labels = gt_labels["ifr"]

        self.source_ratio = [len(sar_gt_labels), len(rgb_gt_labels), len(ifr_gt_labels)]

        batch_inputs = []
        for each in self.train_datasets:
            if len(img[each]) > 0:
                batch_inputs.append(img[each])

        x, gate_loss = self.extract_feat(batch_inputs, self.train_datasets, is_train=True)

        losses = dict()
        if gate_loss is not None:
            losses.update({"gate_loss": gate_loss})

        sar_x, rgb_x, ifr_x = x

        if len(sar_gt_labels) > 0:
            batch_input_shape = tuple(img["sar"][0].size()[-2:])
            for img_meta in sar_img_metas:
                img_meta["batch_input_shape"] = batch_input_shape

            single_stage_losses = self.sar_bbox_head.forward_train(
                sar_x, sar_img_metas, sar_gt_bboxes, sar_gt_labels, gt_bboxes_ignore)
            losses.update({"sar_" + k: v for k, v in single_stage_losses.items()})

        if len(rgb_gt_labels) > 0:
            batch_input_shape = tuple(img["rgb"][0].size()[-2:])
            for img_meta in rgb_img_metas:
                img_meta["batch_input_shape"] = batch_input_shape

            single_stage_losses = self.rgb_bbox_head.forward_train(
                rgb_x, rgb_img_metas, rgb_gt_bboxes, rgb_gt_labels, gt_bboxes_ignore)
            losses.update({"rgb_" + k: v for k, v in single_stage_losses.items()})

        if len(ifr_gt_labels) > 0:
            if self.with_ifr_rpn:
                proposal_cfg = self.ifr_train_cfg.get("rpn_proposal", self.ifr_test_cfg.rpn)
                rpn_losses, proposal_list = self.ifr_rpn_head.forward_train(
                    ifr_x,
                    ifr_img_metas,
                    ifr_gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
                losses.update({"ifr_" + k: v for k, v in rpn_losses.items()})
            else:
                proposal_list = proposals

            roi_losses = self.ifr_roi_head.forward_train(
                ifr_x,
                ifr_img_metas,
                proposal_list,
                ifr_gt_bboxes,
                ifr_gt_labels,
                gt_bboxes_ignore,
                gt_masks,
                **kwargs)
            losses.update({"ifr_" + k: v for k, v in roi_losses.items()})

        loss_sum = 0
        if self.multi_tasks_reweight is not None:
            reweight_losses = {}
            cur_losses = []
            for k, loss in losses.items():
                if k not in self.reweight_losses:
                    reweight_losses.update({k: loss})
                    continue
                if isinstance(loss, list):
                    loss = sum(loss)
                cur_losses.append(loss)
            cur_losses = torch.stack(cur_losses)

        if self.multi_tasks_reweight == "uncertainty":
            for i, loss in enumerate(cur_losses):
                loss_sum += (
                    0.5 / (self.mtl_sigma[i] ** 2) * loss
                    + torch.log(1 + self.mtl_sigma[i] ** 2)
                )
            reweight_losses.update({"reweighted_total_losses": loss_sum})
            return reweight_losses

        if self.multi_tasks_reweight == "dwa":
            if self.history_loss is not None:
                w_i = cur_losses / torch.tensor(self.history_loss).cuda()
                batch_weight = len(self.reweight_losses) * torch.nn.functional.softmax(
                    w_i / self.T, dim=-1
                )
            else:
                batch_weight = torch.ones_like(cur_losses).cuda()
            loss_sum = torch.mul(cur_losses, batch_weight).sum()

            reweight_losses.update({"reweighted_total_losses": loss_sum})
            self.history_loss = cur_losses.detach().cpu().numpy()
            return reweight_losses

        return losses

    def simple_test(self, img, img_metas, subdataset, proposals=None, rescale=False):
        """Test without augmentation."""
        assert isinstance(subdataset[0], list) and len(subdataset) == 1
        assert all(x == subdataset[0][0] for x in subdataset[0])
        subdataset = subdataset[0][0]

        x = self.extract_feat(img, [subdataset])
        if isinstance(x, tuple):
            x, experts_id = x
        else:
            experts_id = None

        if subdataset == "rgb":
            results_list = self.rgb_bbox_head.simple_test(
                x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.rgb_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results

        if subdataset == "sar":
            results_list = self.sar_bbox_head.simple_test(
                x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results

        if proposals is None:
            proposal_list = self.ifr_rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        return self.ifr_roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, subdataset, rescale=False):
        """Test with augmentations."""
        assert isinstance(subdataset[0], list) and len(subdataset) == 1
        assert all(x == subdataset[0][0] for x in subdataset[0])
        subdataset = subdataset[0][0]

        x = self.extract_feat(imgs, [subdataset])
        if isinstance(x, tuple):
            x, experts_id = x
        else:
            experts_id = None

        if subdataset == "rgb":
            results_list = self.rgb_bbox_head.aug_test(
                x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.rgb_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results

        if subdataset == "sar":
            results_list = self.sar_bbox_head.aug_test(
                x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results

        proposal_list = self.ifr_rpn_head.aug_test_rpn(x, img_metas)
        return self.ifr_roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
