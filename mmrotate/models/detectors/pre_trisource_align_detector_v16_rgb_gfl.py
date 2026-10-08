# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import bbox2result

from ..builder import ROTATED_DETECTORS, build_head
from ..utils import TextPrototypeAlignerV10
from .pre_trisource_align_detector_v16 import TriSourceDetectorV16


@ROTATED_DETECTORS.register_module()
class TriSourceDetectorV16RGBGFL(TriSourceDetectorV16):
    """TriSourceDetectorV16 variant with a one-stage GFL RGB branch."""

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
                 align_cfg=None,
                 init_cfg=None):
        if rgb_bbox_head is None:
            raise ValueError(
                "rgb_bbox_head must be provided for TriSourceDetectorV16RGBGFL."
            )
        if rgb_rpn_head is not None or rgb_roi_head is not None:
            raise ValueError(
                "TriSourceDetectorV16RGBGFL does not use rgb_rpn_head or rgb_roi_head."
            )

        original_align_cfg = align_cfg if align_cfg is not None else {}
        super_align_cfg = dict(original_align_cfg)
        if super_align_cfg.get("auto_backbone_text_from_aligner", True):
            super_align_cfg["auto_backbone_text_from_aligner"] = False

        super(TriSourceDetectorV16RGBGFL, self).__init__(
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
            align_cfg=super_align_cfg,
            init_cfg=init_cfg,
        )

        rgb_bbox_head_ = rgb_bbox_head.copy()
        rgb_bbox_head_.update(train_cfg=rgb_train_cfg)
        rgb_bbox_head_.update(test_cfg=rgb_test_cfg)
        self.rgb_bbox_head = build_head(rgb_bbox_head_)

        self.rgb_train_cfg = rgb_train_cfg
        self.rgb_test_cfg = rgb_test_cfg

        self.align_cfg = original_align_cfg
        self.auto_backbone_text_from_aligner = self.align_cfg.get(
            "auto_backbone_text_from_aligner", True
        )
        if self.auto_backbone_text_from_aligner:
            self.prototype_aligner = self._build_prototype_aligner(self.align_cfg)
        else:
            self.prototype_aligner = None

    @property
    def with_rgb_bbox_head(self):
        """bool: whether the detector has a one-stage RGB bbox head."""
        return hasattr(self, "rgb_bbox_head") and self.rgb_bbox_head is not None

    def _build_prototype_aligner(self, align_cfg):
        semantic_mapping = align_cfg.get("semantic_mapping", None)

        sar_cat = align_cfg.get("sar_category", [])
        rgb_cat = align_cfg.get("rgb_category", [])
        ifr_cat = align_cfg.get("ifr_category", [])

        modal_num_classes = {}
        if hasattr(self, "sar_bbox_head") and hasattr(self.sar_bbox_head, "num_classes"):
            modal_num_classes["sar"] = self.sar_bbox_head.num_classes
        else:
            modal_num_classes["sar"] = len(sar_cat)

        if hasattr(self, "rgb_bbox_head") and hasattr(self.rgb_bbox_head, "num_classes"):
            modal_num_classes["rgb"] = self.rgb_bbox_head.num_classes
        else:
            modal_num_classes["rgb"] = len(rgb_cat)

        if hasattr(self, "ifr_roi_head") and hasattr(self.ifr_roi_head, "bbox_head"):
            modal_num_classes["ifr"] = self.ifr_roi_head.bbox_head.num_classes
        else:
            modal_num_classes["ifr"] = len(ifr_cat)

        aligner_args = dict(
            modal_num_classes=modal_num_classes,
            semantic_mapping=semantic_mapping,
            sar_category=sar_cat,
            rgb_category=rgb_cat,
            ifr_category=ifr_cat,
            in_dims=align_cfg.get(
                "in_dims", {"sar": 256, "rgb": 256, "ifr": 256, "text": 512}
            ),
            modalities=list(align_cfg.get("in_dims", {}).keys()),
            align_dim=align_cfg.get("align_dim", 256),
            ema_beta=align_cfg.get("ema_beta", 0.01),
            vol_weight=align_cfg.get("vol_weight", 0.5),
            is_base_training=align_cfg.get("is_base_training", False),
            sar_clip_name=align_cfg.get("sar_clip_name", "ViT-B-16"),
            sar_clip_path=align_cfg.get("sar_clip_path", ""),
            sar_clip_cache=align_cfg.get("sar_clip_cache", ""),
            text_encoder_type=align_cfg.get("text_encoder_type", "sarclip"),
            clip_model_name=align_cfg.get("clip_model_name", "lcybuaa/Git-RSCLIP"),
            clip_processor_name=align_cfg.get("clip_processor_name", None),
            text_prompt_template=align_cfg.get("text_prompt_template", None),
            prompt_file=align_cfg.get("prompt_file", None),
            use_llm_prompt=align_cfg.get("use_llm_prompt", False),
            eps=1e-6,
        )

        return TextPrototypeAlignerV10(**aligner_args)

    def _check_local_label_ranges(self, sar_gt_labels, rgb_gt_labels, ifr_gt_labels):
        if self._label_range_warned:
            return
        label_dict = {"sar": sar_gt_labels, "rgb": rgb_gt_labels, "ifr": ifr_gt_labels}
        num_cls_dict = {
            "sar": getattr(self.sar_bbox_head, "num_classes", None),
            "rgb": getattr(self.rgb_bbox_head, "num_classes", None)
            if hasattr(self, "rgb_bbox_head")
            else None,
            "ifr": getattr(getattr(self.ifr_roi_head, "bbox_head", None), "num_classes", None)
            if hasattr(self, "ifr_roi_head")
            else None,
        }
        msgs = []
        for modality, labels_list in label_dict.items():
            num_cls = num_cls_dict.get(modality, None)
            if num_cls is None:
                continue
            for labels in labels_list:
                if not torch.is_tensor(labels) or labels.numel() == 0:
                    continue
                max_label = int(labels.max().item())
                min_label = int(labels.min().item())
                if min_label < 0 or max_label >= num_cls:
                    msgs.append(
                        f"{modality}: label range [{min_label}, {max_label}] "
                        f"exceeds num_classes={num_cls}"
                    )
                    break
        if len(msgs) > 0:
            import warnings
            warnings.warn("Local label space mismatch detected: " + "; ".join(msgs))
            self._label_range_warned = True

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
        self._check_local_label_ranges(sar_gt_labels, rgb_gt_labels, ifr_gt_labels)

        self.source_ratio = [len(sar_gt_labels), len(rgb_gt_labels), len(ifr_gt_labels)]

        batch_inputs = []
        active_datasets = []
        for each in self.train_datasets:
            if len(img[each]) > 0:
                batch_inputs.append(img[each])
                active_datasets.append(each)
        if len(batch_inputs) == 0:
            raise RuntimeError("No non-empty modality batch found in forward_train.")

        roi_features, text_features = (None, None)
        if self.enable_backbone_align_inputs:
            roi_features, text_features = self._collect_backbone_align_inputs(kwargs)
            need_auto_text = self.auto_backbone_text_from_aligner and text_features is None
            if need_auto_text:
                target_device = (
                    batch_inputs[0].device
                    if len(batch_inputs) > 0
                    else next(self.backbone.parameters()).device
                )
                text_features = self._build_backbone_text_features_from_aligner(
                    target_device
                )
            target_device = (
                batch_inputs[0].device
                if len(batch_inputs) > 0
                else next(self.backbone.parameters()).device
            )
            roi_features = self._move_align_features_to_device(roi_features, target_device)
            text_features = self._move_align_features_to_device(
                text_features, target_device
            )
        backbone_gt_bboxes = dict(sar=sar_gt_bboxes, rgb=rgb_gt_bboxes, ifr=ifr_gt_bboxes)
        backbone_gt_labels = dict(sar=sar_gt_labels, rgb=rgb_gt_labels, ifr=ifr_gt_labels)
        x, backbone_loss = self.extract_feat(
            batch_inputs,
            active_datasets,
            is_train=True,
            roi_features=roi_features,
            text_features=text_features,
            gt_bboxes=backbone_gt_bboxes,
            gt_labels=backbone_gt_labels,
        )

        losses = dict()
        if backbone_loss is not None:
            if isinstance(backbone_loss, dict):
                if "gate_loss" in backbone_loss:
                    losses["loss_moe_gate"] = backbone_loss["gate_loss"] * 1.0
                if "align_loss" in backbone_loss:
                    losses["loss_moe_align"] = backbone_loss["align_loss"] * 1.0
            else:
                losses["loss_moe_gate"] = backbone_loss

        sar_x, rgb_x, ifr_x = x

        if len(sar_gt_labels) > 0:
            batch_input_shape = tuple(img["sar"][0].size()[-2:])
            for img_meta in sar_img_metas:
                img_meta["batch_input_shape"] = batch_input_shape

            single_stage_losses = self.sar_bbox_head.forward_train(
                sar_x, sar_img_metas, sar_gt_bboxes, sar_gt_labels, gt_bboxes_ignore
            )
            losses.update({"sar_" + k: v for k, v in single_stage_losses.items()})

        if len(rgb_gt_labels) > 0:
            batch_input_shape = tuple(img["rgb"][0].size()[-2:])
            for img_meta in rgb_img_metas:
                img_meta["batch_input_shape"] = batch_input_shape

            single_stage_losses = self.rgb_bbox_head.forward_train(
                rgb_x, rgb_img_metas, rgb_gt_bboxes, rgb_gt_labels, gt_bboxes_ignore
            )
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
                    **kwargs,
                )
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
                **kwargs,
            )
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

        if subdataset == "sar":
            results_list = self.sar_bbox_head.simple_test(x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results

        if subdataset == "rgb":
            results_list = self.rgb_bbox_head.simple_test(x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.rgb_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results

        if proposals is None:
            proposal_list = self.ifr_rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        return self.ifr_roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale
        )

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

        if subdataset == "sar":
            results_list = self.sar_bbox_head.aug_test(x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results

        if subdataset == "rgb":
            results_list = self.rgb_bbox_head.aug_test(x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.rgb_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results

        proposal_list = self.ifr_rpn_head.aug_test_rpn(x, img_metas)
        return self.ifr_roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale
        )
