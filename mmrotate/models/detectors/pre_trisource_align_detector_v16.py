# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import warnings

import torch
from mmrotate.core.bbox.transforms import obb2xyxy
import torch.distributed as dist
from mmdet.core import bbox2result
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector
from ..utils import TextPrototypeAlignerV10

@ROTATED_DETECTORS.register_module()
class TriSourceDetectorV16(RotatedBaseDetector):
    """Base class for rotated two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """ 

    def __init__(self,
                 backbone,
                 neck=None,
                 rgb_rpn_head=None,
                 rgb_roi_head=None,
                 rgb_train_cfg=None,
                 rgb_test_cfg=None,
                 ifr_rpn_head=None,
                 ifr_roi_head=None,
                 ifr_train_cfg=None,
                 ifr_test_cfg=None,
                 sar_bbox_head=None,
                 sar_rpn_head=None,
                 sar_roi_head=None,
                 sar_train_cfg=None,
                 sar_test_cfg=None,
                 multi_tasks_reweight=None,
                 reweight_losses=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 align_cfg=None,
                 init_cfg=None,
                 train_modalities=None):
        super(TriSourceDetectorV16, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        # GFL/horizontal RPN start at stride 8; oriented RPN starts at 4.
        self.sar_fpn_start_level = (
            0 if sar_rpn_head is not None and
            sar_rpn_head.get('type') == 'OrientedRPNHead' else 1)
        self.train_datasets = list(train_modalities or ['sar', 'rgb'])
        if not self.train_datasets or any(m not in ('sar', 'rgb', 'ifr') for m in self.train_datasets):
            raise ValueError('train_modalities must be a non-empty subset of sar/rgb/ifr.')

        if neck is not None:
            self.neck = build_neck(neck)
            if neck.get('type') == 'MultitaskFPN':
                start_levels = [
                    self.sar_fpn_start_level if modality == 'sar'
                    else self.neck.start_level
                    for modality in self.train_datasets]
                self.neck.freeze_unused_levels(start_levels)

        if rgb_rpn_head is not None:
            rgb_rpn_train_cfg = rgb_train_cfg.rpn if rgb_train_cfg is not None else None
            rgb_rpn_head_ = rgb_rpn_head.copy()
            rgb_rpn_head_.update(train_cfg=rgb_rpn_train_cfg, test_cfg=rgb_test_cfg.rpn)
            self.rgb_rpn_head = build_head(rgb_rpn_head_)

        if rgb_roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rgb_rcnn_train_cfg = rgb_train_cfg.rcnn if rgb_train_cfg is not None else None
            rgb_roi_head.update(train_cfg=rgb_rcnn_train_cfg)
            rgb_roi_head.update(test_cfg=rgb_test_cfg.rcnn)
            rgb_roi_head.pretrained = pretrained
            self.rgb_roi_head = build_head(rgb_roi_head)

        self.rgb_train_cfg = rgb_train_cfg
        self.rgb_test_cfg = rgb_test_cfg
     
        if ifr_rpn_head is not None:
            ifr_rpn_train_cfg = ifr_train_cfg.rpn if ifr_train_cfg is not None else None
            ifr_rpn_head_ = ifr_rpn_head.copy()
            ifr_rpn_head_.update(train_cfg=ifr_rpn_train_cfg, test_cfg=ifr_test_cfg.rpn)
            self.ifr_rpn_head = build_head(ifr_rpn_head_)

        if ifr_roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            ifr_rcnn_train_cfg = ifr_train_cfg.rcnn if ifr_train_cfg is not None else None
            ifr_roi_head.update(train_cfg=ifr_rcnn_train_cfg)
            ifr_roi_head.update(test_cfg=ifr_test_cfg.rcnn)
            ifr_roi_head.pretrained = pretrained
            self.ifr_roi_head = build_head(ifr_roi_head)

        self.ifr_train_cfg = ifr_train_cfg
        self.ifr_test_cfg = ifr_test_cfg

        #################################################### SAR HBB heads
        self.sar_bbox_head = None
        self.sar_rpn_head = None
        self.sar_roi_head = None
        if sar_rpn_head is not None:
            sar_rpn_train_cfg = (
                sar_train_cfg.rpn if sar_train_cfg is not None else None)
            sar_rpn_head_ = sar_rpn_head.copy()
            sar_rpn_head_.update(
                train_cfg=sar_rpn_train_cfg,
                test_cfg=sar_test_cfg.rpn)
            self.sar_rpn_head = build_head(sar_rpn_head_)

        if sar_roi_head is not None:
            sar_rcnn_train_cfg = (
                sar_train_cfg.rcnn if sar_train_cfg is not None else None)
            sar_roi_head_ = sar_roi_head.copy()
            sar_roi_head_.update(
                train_cfg=sar_rcnn_train_cfg,
                test_cfg=sar_test_cfg.rcnn)
            sar_roi_head_.pretrained = pretrained
            self.sar_roi_head = build_head(sar_roi_head_)

        if sar_bbox_head is not None:
            sar_bbox_head_ = sar_bbox_head.copy()
            sar_bbox_head_.update(train_cfg=sar_train_cfg)
            sar_bbox_head_.update(test_cfg=sar_test_cfg)
            self.sar_bbox_head = build_head(sar_bbox_head_)

        self.sar_train_cfg = sar_train_cfg
        self.sar_test_cfg = sar_test_cfg

        if self.sar_bbox_head is None and not (
                self.sar_rpn_head is not None and self.sar_roi_head is not None):
            raise ValueError(
                'Configure either sar_bbox_head or both sar_rpn_head and '
                'sar_roi_head.')

        self.multi_tasks_reweight = multi_tasks_reweight
        self.reweight_losses = reweight_losses
        if self.multi_tasks_reweight=='uncertainty':
            task_num = len(self.reweight_losses) 
            self.mtl_sigma = torch.nn.Parameter(torch.ones(task_num, requires_grad=True))
        elif self.multi_tasks_reweight=='dwa':
            self.T = 3
            self.history_loss = None

        self.align_cfg = dict(align_cfg or {})
        has_align_cfg = align_cfg is not None
        if has_align_cfg:
            configured_encoder = str(
                self.align_cfg.get('text_encoder_type', 'sarclip')).lower()
            if configured_encoder not in ('sarclip', 'sar_clip'):
                warnings.warn(
                    'TriSourceDetectorV16 uses the local SAR-CLIP encoder; '
                    f'overriding text_encoder_type={configured_encoder!r}.',
                    UserWarning,
                    stacklevel=2)
            self.align_cfg['text_encoder_type'] = 'sarclip'
        self.enable_backbone_align_inputs = self.align_cfg.get(
            'enable_backbone_align_inputs', has_align_cfg)
        self.auto_backbone_roi_from_gt = self.align_cfg.get(
            'auto_backbone_roi_from_gt', has_align_cfg)
        self.auto_backbone_text_from_aligner = self.align_cfg.get(
            'auto_backbone_text_from_aligner', has_align_cfg)
        self.backbone_roi_pool_size = int(self.align_cfg.get('backbone_roi_pool_size', 7))
        self.backbone_roi_max_samples = int(self.align_cfg.get('backbone_roi_max_samples', 128))

        self.prototype_aligner = None
        if self.auto_backbone_text_from_aligner:
            self.prototype_aligner = self._build_prototype_aligner(self.align_cfg)
        self._label_range_warned = False
        self._freeze_inactive_modalities()

    def _freeze_inactive_modalities(self):
        """Keep checkpoint branches, but exclude inactive heads/projectors from DDP."""
        active = set(self.train_datasets)
        for modality in ('sar', 'rgb', 'ifr'):
            if modality in active:
                continue
            for suffix in ('bbox_head', 'rpn_head', 'roi_head'):
                head = getattr(self, f'{modality}_{suffix}', None)
                if head is not None:
                    head.requires_grad_(False)
        for module in self.backbone.modules():
            projectors = getattr(module, 'projectors', None)
            if isinstance(projectors, torch.nn.ModuleDict):
                for modality, projector in projectors.items():
                    if modality not in active:
                        projector.requires_grad_(False)

    # ------------------------------------------------------------------ #
    # Fix: skip loading text_feats_* and banks.* from checkpoint.
    #
    # These buffers are correctly generated during __init__ from the
    # category names in the config using the text encoder (SAR-CLIP).
    # In V16 they are NEVER updated during training – they are frozen
    # buffers used only for backbone MoE routing.
    #
    # Loading them from a checkpoint is dangerous: when
    # initialize_bbox_head.py expands the checkpoint for novel classes,
    # the novel-class text features may be left as zeros (e.g. CLIP
    # encoding fails silently), overwriting the correctly-initialized
    # features and breaking the MoE alignment routing.
    # ------------------------------------------------------------------ #
    _ALIGNER_SKIP_PREFIXES = (
        'prototype_aligner.text_feats_',
        'prototype_aligner.banks.',
    )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Hook into PyTorch checkpoint loading so that text-feature and
        prototype-bank buffers generated by the aligner during __init__ are
        never overwritten by potentially-blank checkpoint values.

        IMPORTANT: we mutate ``state_dict`` in-place (deleting keys) because
        mmengine's recursive ``load()`` reuses the same state_dict for child
        modules, and a filtered copy would NOT propagate to child calls.
        """
        if self.prototype_aligner is not None:
            keys_to_drop = [
                k for k in state_dict
                if any(k.startswith(prefix + skip)
                       for skip in self._ALIGNER_SKIP_PREFIXES)
            ]
            for k in keys_to_drop:
                del state_dict[k]

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                       strict, missing_keys,
                                       unexpected_keys, error_msgs)

    @property
    def with_rgb_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rgb_rpn_head') and self.rgb_rpn_head is not None

    @property
    def with_sar_rpn(self):
        return self.sar_rpn_head is not None

    @property
    def with_sar_roi_head(self):
        return self.sar_roi_head is not None

    @property
    def sar_num_classes(self):
        if self.with_sar_roi_head:
            return self.sar_roi_head.bbox_head.num_classes
        return self.sar_bbox_head.num_classes

    
    @property
    def with_rgb_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'rgb_roi_head') and self.rgb_roi_head is not None
    
    @property
    def with_ifr_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'ifr_rpn_head') and self.ifr_rpn_head is not None

    
    @property
    def with_ifr_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'ifr_roi_head') and self.ifr_roi_head is not None

    def _forward_backbone(self,
                          batch_inputs,
                          datasets,
                          roi_features=None,
                          text_features=None,
                          gt_bboxes=None,
                          gt_labels=None):
        """Run backbone while being compatible with old/new forward signatures."""
        if not hasattr(self, '_forward_backbone_debug_printed'):
            self._forward_backbone_debug_printed = False
        if not self._forward_backbone_debug_printed:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            print(
                f'[BackboneAlignDebug] detector_rank={rank}, '
                f'model_training={self.training}, '
                f'backbone_training={self.backbone.training}, '
                f'roi_features_none={roi_features is None}, '
                f'text_features_none={text_features is None}, '
                f'gt_bboxes_none={gt_bboxes is None}, '
                f'gt_labels_none={gt_labels is None}, '
                f'datasets={datasets}')
            self._forward_backbone_debug_printed = True
        if (roi_features is None and text_features is None and
                gt_bboxes is None and gt_labels is None):
            return self.backbone(batch_inputs, datasets)

        try:
            sig = inspect.signature(self.backbone.forward)
            params = sig.parameters
            supports_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            kwargs = {}
            if 'roi_features' in params or supports_kwargs:
                kwargs['roi_features'] = roi_features
            if 'text_features' in params or supports_kwargs:
                kwargs['text_features'] = text_features
            if 'gt_bboxes' in params or supports_kwargs:
                kwargs['gt_bboxes'] = gt_bboxes
            if 'gt_labels' in params or supports_kwargs:
                kwargs['gt_labels'] = gt_labels
            if len(kwargs) > 0:
                return self.backbone(batch_inputs, datasets, **kwargs)
            return self.backbone(batch_inputs, datasets)
        except (ValueError, TypeError):
            # Fallback for backbones without introspectable signatures.
            try:
                return self.backbone(
                    batch_inputs,
                    datasets,
                    roi_features=roi_features,
                    text_features=text_features,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels)
            except TypeError:
                return self.backbone(batch_inputs, datasets)

    def _collect_backbone_align_inputs(self, kwargs):
        """Collect optional alignment features for ConvNeXtMoEV2 routing."""
        roi_features = kwargs.pop('backbone_roi_features', None)
        if roi_features is None:
            roi_features = kwargs.pop('roi_features', None)
        if roi_features is None:
            roi_features = self.align_cfg.get('backbone_roi_features', None)

        text_features = kwargs.pop('backbone_text_features', None)
        if text_features is None:
            text_features = kwargs.pop('text_features', None)
        if text_features is None:
            text_features = self.align_cfg.get('backbone_text_features', None)

        # Guard against malformed external inputs that would bypass auto-build
        # and silently disable alignment in backbone routing branches.
        if roi_features is not None and (not self._is_valid_backbone_roi_features(roi_features)):
            roi_features = None
        if text_features is not None and (not self._is_valid_backbone_text_features(text_features)):
            text_features = None

        return roi_features, text_features

    def _is_valid_backbone_roi_features(self, roi_features):
        if not isinstance(roi_features, dict):
            return False
        for _, entry in roi_features.items():
            if not isinstance(entry, dict):
                continue
            rois = entry.get('rois', None)
            labels = entry.get('labels', None)
            if torch.is_tensor(rois) and torch.is_tensor(labels):
                if rois.numel() > 0 and labels.numel() > 0:
                    return True
        return False

    def _is_valid_backbone_text_features(self, text_features):
        if torch.is_tensor(text_features):
            return text_features.numel() > 0
        if not isinstance(text_features, dict):
            return False
        for _, entry in text_features.items():
            if torch.is_tensor(entry) and entry.numel() > 0:
                return True
        return False

    def _build_prototype_aligner(self, align_cfg):
        semantic_mapping = align_cfg.get('semantic_mapping', None)
        in_dims = align_cfg.get(
            'in_dims', {'sar': 256, 'rgb': 256, 'text': 512})

        sar_cat = align_cfg.get('sar_category', [])
        rgb_cat = align_cfg.get('rgb_category', [])
        ifr_cat = align_cfg.get('ifr_category', [])

        modal_num_classes = {}
        if self.sar_bbox_head is not None and hasattr(self.sar_bbox_head, 'num_classes'):
            modal_num_classes['sar'] = self.sar_bbox_head.num_classes
        elif self.with_sar_roi_head:
            modal_num_classes['sar'] = self.sar_roi_head.bbox_head.num_classes
        else:
            modal_num_classes['sar'] = len(sar_cat)

        if hasattr(self, 'rgb_roi_head') and hasattr(self.rgb_roi_head, 'bbox_head'):
            modal_num_classes['rgb'] = self.rgb_roi_head.bbox_head.num_classes
        else:
            modal_num_classes['rgb'] = len(rgb_cat)

        if hasattr(self, 'ifr_roi_head') and hasattr(self.ifr_roi_head, 'bbox_head'):
            modal_num_classes['ifr'] = self.ifr_roi_head.bbox_head.num_classes
        else:
            modal_num_classes['ifr'] = len(ifr_cat)

        aligner_args = dict(
            modal_num_classes=modal_num_classes,
            semantic_mapping=semantic_mapping,
            sar_category=sar_cat,
            rgb_category=rgb_cat,
            ifr_category=ifr_cat,
            in_dims=in_dims,
            modalities=list(align_cfg.get('modalities', in_dims.keys())),
            align_dim=align_cfg.get('align_dim', 256),
            ema_beta=align_cfg.get('ema_beta', 0.01),
            vol_weight=align_cfg.get('vol_weight', 0.5),
            is_base_training=align_cfg.get('is_base_training', False),
            sar_clip_name=align_cfg.get('sar_clip_name', 'ViT-B-16'),
            sar_clip_path=align_cfg.get('sar_clip_path', ''),
            sar_clip_cache=align_cfg.get('sar_clip_cache', ''),
            text_encoder_type=align_cfg.get('text_encoder_type', 'sarclip'),
            clip_model_name=align_cfg.get('clip_model_name', None),
            clip_processor_name=align_cfg.get('clip_processor_name', None),
            text_prompt_template=align_cfg.get('text_prompt_template', None),
            prompt_file=align_cfg.get('prompt_file', None),
            use_llm_prompt=align_cfg.get('use_llm_prompt', False),
            eps=1e-6)

        return TextPrototypeAlignerV10(**aligner_args)

    def _to_hbb(self, bboxes: torch.Tensor) -> torch.Tensor:
        if bboxes.numel() == 0:
            return bboxes.new_zeros((0, 4))
        if bboxes.size(-1) == 4:
            return bboxes[:, :4]
        if bboxes.size(-1) == 5:
            return obb2xyxy(bboxes, version='le90')
        raise ValueError(f'Unsupported bbox dim: {bboxes.size(-1)}, expected 4 or 5')

    def _build_modality_backbone_roi_inputs(self, modal_imgs, gt_bboxes, gt_labels):
        if (not torch.is_tensor(modal_imgs)) or modal_imgs.size(0) == 0:
            return None
        if len(gt_bboxes) == 0 or len(gt_labels) == 0:
            return None

        rois_all = []
        labels_all = []
        num_imgs = min(modal_imgs.size(0), len(gt_bboxes), len(gt_labels))
        for img_id in range(num_imgs):
            boxes = gt_bboxes[img_id]
            labels = gt_labels[img_id]
            if (not torch.is_tensor(boxes)) or (not torch.is_tensor(labels)):
                continue
            if boxes.numel() == 0 or labels.numel() == 0:
                continue

            hbb = self._to_hbb(boxes)
            labels = labels.reshape(-1).long().to(hbb.device)
            n = min(hbb.size(0), labels.size(0))
            if n <= 0:
                continue
            hbb = hbb[:n]
            labels = labels[:n]

            h, w = modal_imgs[img_id].shape[-2:]
            if h <= 0 or w <= 0:
                continue
            x1 = torch.minimum(hbb[:, 0], hbb[:, 2])
            y1 = torch.minimum(hbb[:, 1], hbb[:, 3])
            x2 = torch.maximum(hbb[:, 0], hbb[:, 2])
            y2 = torch.maximum(hbb[:, 1], hbb[:, 3])
            x1 = (x1 / float(w)).clamp(0, 1 - 1.0 / max(float(w), 1.0))
            y1 = (y1 / float(h)).clamp(0, 1 - 1.0 / max(float(h), 1.0))
            x2 = (x2 / float(w)).clamp(0, 1)
            y2 = (y2 / float(h)).clamp(0, 1)
            x2 = torch.maximum(x2, x1 + 1.0 / max(float(w), 1.0)).clamp(max=1)
            y2 = torch.maximum(y2, y1 + 1.0 / max(float(h), 1.0)).clamp(max=1)
            rois = torch.stack([
                x1.new_full((n, ), float(img_id)),
                x1,
                y1,
                x2,
                y2
            ], dim=1)
            rois_all.append(rois)
            labels_all.append(labels)

        if len(rois_all) == 0:
            return None

        rois = torch.cat(rois_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
        if self.backbone_roi_max_samples > 0 and rois.size(0) > self.backbone_roi_max_samples:
            idx = torch.randperm(rois.size(0), device=rois.device)[:self.backbone_roi_max_samples]
            rois = rois[idx]
            labels = labels[idx]
        return dict(rois=rois.detach(), labels=labels.detach())

    def _build_backbone_roi_features_from_gt(self,
                                             sar_imgs,
                                             rgb_imgs,
                                             ifr_imgs,
                                             sar_gt_bboxes,
                                             rgb_gt_bboxes,
                                             ifr_gt_bboxes,
                                             sar_gt_labels,
                                             rgb_gt_labels,
                                             ifr_gt_labels):
        roi_features = {}
        sar_roi = self._build_modality_backbone_roi_inputs(
            sar_imgs, sar_gt_bboxes, sar_gt_labels)
        if sar_roi is not None:
            roi_features['sar'] = sar_roi
        rgb_roi = self._build_modality_backbone_roi_inputs(
            rgb_imgs, rgb_gt_bboxes, rgb_gt_labels)
        if rgb_roi is not None:
            roi_features['rgb'] = rgb_roi
        ifr_roi = self._build_modality_backbone_roi_inputs(
            ifr_imgs, ifr_gt_bboxes, ifr_gt_labels)
        if ifr_roi is not None:
            roi_features['ifr'] = ifr_roi
        return roi_features if len(roi_features) > 0 else None

    def _check_local_label_ranges(self, sar_gt_labels, rgb_gt_labels, ifr_gt_labels):
        if self._label_range_warned:
            return
        label_dict = {
            'sar': sar_gt_labels,
            'rgb': rgb_gt_labels,
            'ifr': ifr_gt_labels
        }
        num_cls_dict = {
            'sar': self.sar_num_classes,
            'rgb': getattr(getattr(self.rgb_roi_head, 'bbox_head', None), 'num_classes', None) if hasattr(self, 'rgb_roi_head') else None,
            'ifr': getattr(getattr(self.ifr_roi_head, 'bbox_head', None), 'num_classes', None) if hasattr(self, 'ifr_roi_head') else None
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
                        f'{modality}: label range [{min_label}, {max_label}] '
                        f'exceeds num_classes={num_cls}')
                    break
        if len(msgs) > 0:
            warnings.warn('Local label space mismatch detected: ' + '; '.join(msgs))
            self._label_range_warned = True

    def _build_backbone_text_features_from_aligner(self, device):
        if self.prototype_aligner is None:
            return None
        text_features = {}
        for modality in self.train_datasets:
            attr = f'text_feats_{modality}'
            if hasattr(self.prototype_aligner, attr):
                text_feat = getattr(self.prototype_aligner, attr)
                if torch.is_tensor(text_feat) and text_feat.numel() > 0:
                    text_features[modality] = text_feat.to(device).detach()
        return text_features if len(text_features) > 0 else None

    def _move_align_features_to_device(self, data, device):
        """Move tensor-like alignment inputs to target device recursively."""
        if data is None:
            return None
        if torch.is_tensor(data):
            return data.to(device)
        if isinstance(data, dict):
            return {
                k: self._move_align_features_to_device(v, device)
                for k, v in data.items()
            }
        if isinstance(data, tuple):
            return tuple(self._move_align_features_to_device(v, device) for v in data)
        if isinstance(data, list):
            return [self._move_align_features_to_device(v, device) for v in data]
        return data

    def extract_feat(self,
                     batch_inputs,
                     datasets,
                     is_train=False,
                     roi_features=None,
                     text_features=None,
                     gt_bboxes=None,
                     gt_labels=None):
        """Directly extract features from the backbone+neck."""
        x = self._forward_backbone(
            batch_inputs,
            datasets,
            roi_features=roi_features,
            text_features=text_features,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels)
        if isinstance(x, tuple) and len(x) ==2:
            x, loss = x
            experts_id=None
        ###
        elif isinstance(x, tuple) and len(x) ==3:
            x,loss,experts_id=x
        ###
        else :
            loss = None
            experts_id=None
        if is_train:
            modality_features = self.split_batch(x)
            x = {}
            for modality, features, batch_count in zip(
                    self.train_datasets, modality_features, self.source_ratio):
                if batch_count == 0:
                    x[modality] = []
                    continue
                if self.with_neck:
                    if modality == 'sar' and self.sar_fpn_start_level == 1:
                        features = self.neck(
                            features, start_level=1, add_extra_convs='on_output')
                    else:
                        features = self.neck(features)
                x[modality] = features
        elif self.with_neck:
            assert len(datasets) == 1
            if datasets[0] == 'sar':
                x = self.neck(
                    x, start_level=self.sar_fpn_start_level,
                    add_extra_convs='on_output')
            elif datasets[0] in ('rgb', 'ifr'):
                x = self.neck(x)
            else:
                raise ValueError(f'Invalid dataset: {datasets[0]}')

        if is_train:
            return x, loss
        return x,experts_id
    
    def split_batch(self, x, is_list=False): 
        if is_list:
            slices = []
            start = 0
            for length in self.source_ratio:
                end = start + length
                slices.append(x[start:end])
                start = end

            return slices
        else:
            slices = [torch.split(x_, self.source_ratio, dim=0) for x_ in x]
            return tuple(map(list, zip(*slices))) 
        
   
    def gather_dict_values(self, data, ignore_tensor=False):
        gathered_info = {namespace: [] for namespace in self.train_datasets}

        for item in data:
            for namespace in self.train_datasets:
                if item.get(namespace) is not None:
                    gathered_info[namespace].append(item[namespace])

        for namespace in self.train_datasets:
            if gathered_info[namespace] and isinstance(gathered_info[namespace][0], torch.Tensor):
                # Move each tensor to CUDA
                if ignore_tensor:
                    gathered_info[namespace] = [tensor.cuda() for tensor in gathered_info[namespace]]
                else:                
                    gathered_info[namespace] = torch.stack(gathered_info[namespace]).cuda()

        return gathered_info
    
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        if hasattr(self, 'rgb_rpn_head'):
            x,_ = self.extract_feat(img, ['rgb'])
            rpn_outs = self.rgb_rpn_head(x)
            outs = outs + (rpn_outs, )
            proposals = torch.randn(1000, 5).to(img.device)
            roi_outs = self.rgb_roi_head.forward_dummy(x, proposals)
            outs = outs + (roi_outs, )
        
        if hasattr(self, 'ifr_rpn_head'):
            x,_ = self.extract_feat(img, ['ifr'])
            rpn_outs = self.ifr_rpn_head(x)
            outs = outs + (rpn_outs, )
            proposals = torch.randn(1000, 5).to(img.device)
            roi_outs = self.ifr_roi_head.forward_dummy(x, proposals)
            outs = outs + (roi_outs, )

        x,_ = self.extract_feat(img, ['sar'])
       
        if self.with_sar_rpn:
            rpn_outs = self.sar_rpn_head(x)
            outs = outs + (rpn_outs, )
            proposals = torch.randn(1000, 4).to(img.device)
            roi_outs = self.sar_roi_head.forward_dummy(x, proposals)
            outs = outs + (roi_outs, )
        else:
            results = self.sar_bbox_head.forward(x)
            outs = outs + (results, )
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
        # assert False, (img,img_metas)
        # print(img_metas)
        # print('==+'*10) 
        assert gt_bboxes_ignore is None
        img = self.gather_dict_values(img) 
        img_metas = self.gather_dict_values(img_metas)
        gt_bboxes = self.gather_dict_values(gt_bboxes, ignore_tensor=True)
        gt_labels = self.gather_dict_values(gt_labels, ignore_tensor=True)

        modality_img_metas = {m: img_metas.get(m, []) for m in self.train_datasets}
        modality_gt_bboxes = {m: gt_bboxes.get(m, []) for m in self.train_datasets}
        modality_gt_labels = {m: gt_labels.get(m, []) for m in self.train_datasets}
        sar_img_metas = modality_img_metas.get('sar', [])
        rgb_img_metas = modality_img_metas.get('rgb', [])
        ifr_img_metas = modality_img_metas.get('ifr', [])
        sar_gt_bboxes = modality_gt_bboxes.get('sar', [])
        rgb_gt_bboxes = modality_gt_bboxes.get('rgb', [])
        ifr_gt_bboxes = modality_gt_bboxes.get('ifr', [])
        sar_gt_labels = modality_gt_labels.get('sar', [])
        rgb_gt_labels = modality_gt_labels.get('rgb', [])
        ifr_gt_labels = modality_gt_labels.get('ifr', [])
        self._check_local_label_ranges(sar_gt_labels, rgb_gt_labels, ifr_gt_labels)
        self.source_ratio = [len(modality_gt_labels[m]) for m in self.train_datasets]

        active_datasets = [m for m in self.train_datasets if len(img.get(m, [])) > 0]
        batch_inputs = [img[m] for m in active_datasets]
        if len(batch_inputs) == 0:
            raise RuntimeError('No non-empty modality batch found in forward_train.')

        roi_features, text_features = (None, None)
        if self.enable_backbone_align_inputs:
            roi_features, text_features = self._collect_backbone_align_inputs(kwargs)
            need_auto_text = self.auto_backbone_text_from_aligner and text_features is None
            if need_auto_text:
                target_device = batch_inputs[0].device if len(batch_inputs) > 0 else next(self.backbone.parameters()).device
                text_features = self._build_backbone_text_features_from_aligner(
                    target_device)
            target_device = batch_inputs[0].device if len(batch_inputs) > 0 else next(self.backbone.parameters()).device
            roi_features = self._move_align_features_to_device(roi_features, target_device)
            text_features = self._move_align_features_to_device(text_features, target_device)
        backbone_gt_bboxes = dict(
            sar=sar_gt_bboxes,
            rgb=rgb_gt_bboxes,
            ifr=ifr_gt_bboxes)
        backbone_gt_labels = dict(
            sar=sar_gt_labels,
            rgb=rgb_gt_labels,
            ifr=ifr_gt_labels)
        x, backbone_loss = self.extract_feat(
            batch_inputs,
            active_datasets,
            is_train=True,
            roi_features=roi_features,
            text_features=text_features,
            gt_bboxes=backbone_gt_bboxes,
            gt_labels=backbone_gt_labels)
        
        losses = dict()
        if not hasattr(self, '_backbone_loss_debug_printed'):
            self._backbone_loss_debug_printed = False
        if not self._backbone_loss_debug_printed:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            print(f'[BackboneAlignDebug] detector_rank={rank}, backbone_loss={backbone_loss}')
            self._backbone_loss_debug_printed = True
        if backbone_loss is not None:
            if isinstance(backbone_loss, dict):
                if 'gate_loss' in backbone_loss:
                    losses['loss_moe_gate'] = backbone_loss['gate_loss'] * 1.0
                if 'align_loss' in backbone_loss:
                    losses['loss_moe_align'] = backbone_loss['align_loss'] * 1.0
            else:
                losses['loss_moe_gate'] = backbone_loss
 
        sar_x = x.get('sar', [])
        rgb_x = x.get('rgb', [])
        ifr_x = x.get('ifr', [])

        if len(sar_gt_labels) > 0:
            batch_input_shape = tuple(img['sar'][0].size()[-2:])
            for img_meta in sar_img_metas:
                img_meta['batch_input_shape'] = batch_input_shape

            if self.with_sar_rpn:
                proposal_cfg = self.sar_train_cfg.get(
                    'rpn_proposal', self.sar_test_cfg.rpn)
                rpn_losses, proposal_list = self.sar_rpn_head.forward_train(
                    sar_x,
                    sar_img_metas,
                    sar_gt_bboxes,
                    gt_labels=None,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposal_cfg=proposal_cfg,
                    **kwargs)
                losses.update({'sar_' + k: v for k, v in rpn_losses.items()})
                roi_losses = self.sar_roi_head.forward_train(
                    sar_x,
                    sar_img_metas,
                    proposal_list,
                    sar_gt_bboxes,
                    sar_gt_labels,
                    gt_bboxes_ignore,
                    gt_masks,
                    **kwargs)
                losses.update({'sar_' + k: v for k, v in roi_losses.items()})
            else:
                single_stage_losses = self.sar_bbox_head.forward_train(
                    sar_x, sar_img_metas, sar_gt_bboxes, sar_gt_labels,
                    gt_bboxes_ignore)
                losses.update({
                    'sar_' + k: v for k, v in single_stage_losses.items()})

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
                batch_weight = torch.ones_like(losses).cuda()
            loss_sum = torch.mul(cur_losses, batch_weight).sum()
        
            reweight_losses.update({'reweighted_total_losses':loss_sum})
            self.history_loss = cur_losses.detach().cpu().numpy()
            
            return reweight_losses
        return losses

    def simple_test(self, img, img_metas, subdataset, proposals=None, rescale=False):
        """Test without augmentation."""
        assert isinstance(subdataset[0],list) and len(subdataset)==1 # subdataset: [['sar']]
        assert all(x == subdataset[0][0] for x in subdataset[0]), "Not all elements in subdataset are the same: " + str(subdataset)
        subdataset = subdataset[0][0]

        x = self.extract_feat(img, [subdataset]) 

        if isinstance(x,tuple):
            x,experts_id=x
        else:
            experts_id=None
        if subdataset == 'sar':
            if self.with_sar_rpn:
                if proposals is None:
                    proposal_list = self.sar_rpn_head.simple_test_rpn(
                        x, img_metas)
                else:
                    proposal_list = proposals
                return self.sar_roi_head.simple_test(
                    x, proposal_list, img_metas, rescale=rescale)
            results_list = self.sar_bbox_head.simple_test(
                x, img_metas, rescale=rescale)
            return [
                bbox2result(det_bboxes, det_labels, self.sar_num_classes)
                for det_bboxes, det_labels in results_list
            ]
        elif subdataset == 'rgb':
            if proposals is None:
                 proposal_list = self.rgb_rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            return self.rgb_roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)
        

        elif subdataset == 'ifr':
            if proposals is None:
                proposal_list = self.ifr_rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            return self.ifr_roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas,subdataset, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        assert isinstance(subdataset[0],list) and len(subdataset)==1
        assert all(x == subdataset[0][0] for x in subdataset[0]), "Not all elements in subdataset are the same: " + str(subdataset)
        subdataset = subdataset[0][0]
        x = self.extract_feat(imgs, [subdataset]) 
        if subdataset == 'sar':
            if self.with_sar_rpn:
                proposal_list = self.sar_rpn_head.aug_test_rpn(x, img_metas)
                return self.sar_roi_head.aug_test(
                    x, proposal_list, img_metas, rescale=rescale)
            results_list = self.sar_bbox_head.aug_test(
                x, img_metas, rescale=rescale)
            return [
                bbox2result(det_bboxes, det_labels, self.sar_num_classes)
                for det_bboxes, det_labels in results_list
            ]
        elif subdataset == 'rgb':
            proposal_list = self.rgb_rpn_head.aug_test_rpn(x, img_metas)
            return self.rgb_roi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale)

        elif subdataset == 'ifr':
            proposal_list = self.ifr_rpn_head.aug_test_rpn(x, img_metas)
            return self.ifr_roi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale)
