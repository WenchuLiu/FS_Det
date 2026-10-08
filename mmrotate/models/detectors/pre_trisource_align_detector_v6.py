# Copyright (c) OpenMMLab. All rights reserved.
from math import nan
import warnings
# 是v5的更改，改成了项目要求的内容，为全SARDet的base类+细粒度飞机的训练模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2result
import numpy as np
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector 
from ..utils import PrototypeAlignerV3, TextPrototypeAlignerV3
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 让 CUDA 错误同步报出
 
class EMA_meter:
    def __init__(self, beta):
        self.beta = beta
        self.ema = None
        self.steps = 0

    def update(self, value):
        if self.ema is None:
            self.ema = value
        else:
            self.ema = (1 - self.beta) * self.ema + self.beta * value
        self.steps += 1
    
    def get(self):
        return self.ema


@ROTATED_DETECTORS.register_module()
class TriSourceAlignDetectorV6(RotatedBaseDetector):
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
                 align_cfg = None,
                 init_cfg=None,
                 train_modalities=None):
        super(TriSourceAlignDetectorV6, self).__init__(init_cfg)
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
                self.neck.freeze_unused_levels([
                    self.sar_fpn_start_level if modality == 'sar'
                    else self.neck.start_level
                    for modality in self.train_datasets])

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
        
        # Alignment settings support two-modality base training and SAR-only fine-tuning.
        self.align_cfg = dict(align_cfg or {})
        self.is_base_training = self.align_cfg.get('is_base_training', True)
        self.use_text = bool(self.align_cfg.get('init_clip', False))
        self.use_llm_prompt = bool(self.align_cfg.get('use_llm_prompts', False))
        self.prompt_file = self.align_cfg.get('prompt_file', None)
        self.semantic_mapping = self.align_cfg.get('semantic_mapping', None)
        self.num_classes = int(getattr(self.sar_bbox_head, 'num_classes',
                                       self.sar_roi_head.bbox_head.num_classes
                                       if self.sar_roi_head is not None else
                                       self.align_cfg.get('num_classes', 0)))
        if align_cfg is not None:
            in_dims = self.align_cfg.get('in_dims',
                {m: 256 for m in self.train_datasets})
            align_dim = self.align_cfg.get('align_dim', 256)
            ema_beta = self.align_cfg.get('ema_beta', 0.05)
            vol_weight = self.align_cfg.get('vol_weight', 0.5)
            modalities = list(self.align_cfg.get('modalities', self.train_datasets))
            if self.use_text and 'text' not in modalities:
                modalities.append('text')
            modal_num_classes = {'sar': self.sar_num_classes}
            if self.with_rgb_roi_head and hasattr(self.rgb_roi_head, 'bbox_head'):
                modal_num_classes['rgb'] = self.rgb_roi_head.bbox_head.num_classes
            if self.with_ifr_roi_head and hasattr(self.ifr_roi_head, 'bbox_head'):
                modal_num_classes['ifr'] = self.ifr_roi_head.bbox_head.num_classes
            aligner_kwargs = dict(
                num_classes=self.num_classes,
                modal_num_classes=modal_num_classes,
                in_dims=in_dims,
                semantic_mapping=self.semantic_mapping,
                modalities=tuple(modalities),
                align_dim=align_dim,
                ema_beta=ema_beta,
                vol_weight=vol_weight,
                is_base_training=self.is_base_training)
            if self.use_text:
                self.prototype_aligner = TextPrototypeAlignerV3(
                    **aligner_kwargs,
                    init_clip=self.align_cfg.get('initialize_text_features', True),
                    sar_clip_name=self.align_cfg.get('sar_clip_name', 'ViT-B-16'),
                    sar_clip_path=self.align_cfg.get('sar_clip_path'),
                    sar_clip_cache=self.align_cfg.get('sar_clip_cache'),
                    text_encoder_type=self.align_cfg.get(
                        'text_encoder_type', 'sarclip'),
                    clip_model_name=self.align_cfg.get('clip_model_name'),
                    clip_processor_name=self.align_cfg.get(
                        'clip_processor_name'),
                    use_llm_prompt=self.use_llm_prompt,
                    prompt_file=self.prompt_file,
                    base_modal_categories=self.align_cfg.get(
                        'base_modal_categories'),
                    all_modal_categories=self.align_cfg.get(
                        'all_modal_categories'))
            else:
                self.prototype_aligner = PrototypeAlignerV3(**aligner_kwargs)
        else:
            self.prototype_aligner = None

        self.novel_class_ids = list(self.align_cfg.get('novel_class_ids', []))

        self.multi_tasks_reweight = multi_tasks_reweight
        self.reweight_losses = reweight_losses
        if self.multi_tasks_reweight=='uncertainty':
            task_num = len(self.reweight_losses) 
            self.mtl_sigma = torch.nn.Parameter(torch.ones(task_num, requires_grad=True))
        elif self.multi_tasks_reweight=='dwa':
            self.T = 3
            self.history_loss = None
        # Preserve inactive branches in checkpoints while excluding them from
        # the optimizer and DDP reducer during single-modality fine-tuning.
        for modality in ('sar', 'rgb', 'ifr'):
            if modality in self.train_datasets:
                continue
            for suffix in ('bbox_head', 'rpn_head', 'roi_head'):
                head = getattr(self, f'{modality}_{suffix}', None)
                if head is not None:
                    head.requires_grad_(False)
      

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

    def extract_feat(self, batch_inputs, datasets, is_train=False):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(batch_inputs, datasets)
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
    
    def _gt_roi_prototypes(self, features, head, boxes, labels):
        from mmdet.core import bbox2roi
        from mmrotate.core import rbbox2roi
        if not boxes or not any(box.numel() for box in boxes):
            return None, None
        rois = rbbox2roi(boxes) if boxes[0].size(-1) == 5 else bbox2roi(boxes)
        extractor = head.bbox_roi_extractor
        pooled = extractor(features[:extractor.num_inputs], rois)
        return F.adaptive_avg_pool2d(pooled, 1).flatten(1), torch.cat(labels)

    def compute_ita_loss(self, feats, labels, text_feats, modal_name):
        """Compare local labels only against that modality's text prototypes."""
        device = feats.device
        if text_feats is None or text_feats.numel() == 0:
            return feats.new_tensor(0.0)
        text_feats = F.normalize(text_feats.to(device).float(), dim=-1)
        valid = (labels >= 0) & (labels < text_feats.size(0))
        if not valid.any():
            return feats.new_tensor(0.0)
        valid_feats = F.normalize(feats[valid].float(), dim=-1)
        # A one-column softmax is always 1: CE would be zero with no gradient.
        if text_feats.size(0) == 1:
            return (1 - (valid_feats * text_feats[0]).sum(dim=-1)).mean()
        with torch.cuda.amp.autocast(enabled=False):
            logit_scale = self.prototype_aligner.temperature.float().clamp(max=np.log(100.0)).exp()
            logits = torch.matmul(valid_feats, text_feats.T) * logit_scale
            return F.cross_entropy(logits, labels[valid])

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
        self.source_ratio = [len(modality_gt_labels[m]) for m in self.train_datasets]

        active_datasets = [m for m in self.train_datasets if len(img.get(m, [])) > 0]
        batch_inputs = [img[m] for m in active_datasets]
        if not batch_inputs:
            raise RuntimeError('No non-empty modality batch found in forward_train.')
        x, gate_loss = self.extract_feat(
            batch_inputs, active_datasets, is_train=True)
        
        losses = dict()
        if gate_loss is not None:
            losses.update({'gate_loss': gate_loss})
 
        sar_x = x.get('sar', [])
        rgb_x = x.get('rgb', [])
        ifr_x = x.get('ifr', [])
        
        # 为收集对齐所需的 embeddings & labels
        batch_protos = {}
        batch_labels = {}

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
                roi_result = self.sar_roi_head.forward_train(
                    sar_x,
                    sar_img_metas,
                    proposal_list,
                    sar_gt_bboxes,
                    sar_gt_labels,
                    gt_bboxes_ignore,
                    gt_masks,
                    **kwargs)
                if isinstance(roi_result, tuple):
                    roi_losses = roi_result[0]
                    if len(roi_result) >= 3:
                        sar_prototypes, sar_proto_labels = roi_result[1:3]
                        if (torch.is_tensor(sar_prototypes) and
                                torch.is_tensor(sar_proto_labels) and
                                sar_prototypes.numel() > 0):
                            batch_protos['sar'] = sar_prototypes
                            batch_labels['sar'] = sar_proto_labels
                else:
                    roi_losses = roi_result
                losses.update({'sar_' + k: v for k, v in roi_losses.items()})
                if self.prototype_aligner is not None and 'sar' not in batch_protos:
                    feats, labels = self._gt_roi_prototypes(
                        sar_x, self.sar_roi_head, sar_gt_bboxes, sar_gt_labels)
                    if feats is not None:
                        batch_protos['sar'], batch_labels['sar'] = feats, labels
            else:
                single_stage_losses = self.sar_bbox_head.forward_train(
                    sar_x, sar_img_metas, sar_gt_bboxes, sar_gt_labels,
                    gt_bboxes_ignore)
                losses.update({
                    'sar_' + k: v for k, v in single_stage_losses.items()
                    if k not in ['proto_feats', 'proto_labels']})

                feats_list = single_stage_losses.get('proto_feats')
                labels_list = single_stage_losses.get('proto_labels')
                if feats_list is not None and labels_list is not None:
                    valid_protos = [p for p in feats_list if p is not None]
                    valid_labels = [l for l in labels_list if l is not None]
                    if valid_protos:
                        batch_protos['sar'] = torch.cat(valid_protos, dim=0)
                        batch_labels['sar'] = torch.cat(valid_labels, dim=0)


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

            roi_losses, rgb_prototypes, rgb_proto_labels = self.rgb_roi_head.forward_train(rgb_x, rgb_img_metas, proposal_list,
                                                    rgb_gt_bboxes, rgb_gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)            
            losses.update({'rgb_' + k: v for k, v in roi_losses.items()})
            # losses.update({'rgb_' + k: v for k, v in roi_losses.items() if 'prototypes' not in k and 'proto_labels' not in k})

            batch_protos['rgb'] = rgb_prototypes
            batch_labels['rgb'] = rgb_proto_labels

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

            roi_losses, ifr_prototypes, ifr_proto_labels = self.ifr_roi_head.forward_train(ifr_x, ifr_img_metas, proposal_list,
                                                    ifr_gt_bboxes, ifr_gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            losses.update({'ifr_' + k: v for k, v in roi_losses.items()})
            # losses.update({'ifr_' + k: v for k, v in roi_losses.items() if 'prototypes' not in k and 'proto_labels' not in k})

            batch_protos['ifr'] = ifr_prototypes
            batch_labels['ifr'] = ifr_proto_labels


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

        # 计算三种模态的对齐损失
        if self.prototype_aligner is not None and len(batch_protos) > 0:
            align_losses = self.prototype_aligner.update_and_compute_loss(batch_protos, batch_labels)
            # print(5)
            losses.update(align_losses)

        # 计算 ita loss
        if self.use_text:
            ita_losses = []
            text_proj = self.prototype_aligner.proj['text']
            device = next(text_proj.parameters()).device
            if self.is_base_training:
                # 对所有类计算ita_loss
                for modal_name, feats in batch_protos.items():
                    labels = batch_labels[modal_name]
                    if feats is not None and feats.numel() > 0:
                        raw_text_feats = self.prototype_aligner.text_feats_by_modality.get(
                            modal_name)
                        if raw_text_feats is None:
                            continue
                        text_feats = text_proj(raw_text_feats.to(device))
                        ita_loss = self.compute_ita_loss(
                            feats, labels, text_feats, modal_name
                        )
                        ita_losses.append(ita_loss)
                if ita_losses:
                    losses.update({'ita_loss': 0.05 * sum(ita_losses)})
            elif not self.is_base_training:
                for modal_name, feats in batch_protos.items():
                    labels = batch_labels[modal_name]
                    if feats is not None and feats.numel() > 0:
                        if modal_name == "sar":
                            raw_text_feats = self.prototype_aligner.text_feats_by_modality.get(
                                modal_name)
                            if raw_text_feats is None:
                                continue
                            text_feats = text_proj(raw_text_feats.to(device))
                            # 只保留 新类 的样本
                            novel_ids = torch.as_tensor(
                                self.novel_class_ids, dtype=torch.long,
                                device=labels.device)
                            mask = torch.isin(labels, novel_ids) if novel_ids.numel() else torch.zeros_like(labels, dtype=torch.bool)
                            if mask.any():
                                ita_loss = self.compute_ita_loss(
                                    feats[mask], labels[mask], text_feats, modal_name
                                )
                            else:
                                ita_loss = feats.new_tensor(0.0)
                            ita_losses.append(ita_loss)
                if ita_losses:
                    losses.update({'ita_loss': 0.05 * sum(ita_losses)})
                else:
                    losses.update({'ita_loss': next(self.parameters()).new_tensor(0.0)})
            # A rank may have only base GTs during few-shot training. The
            # learnable temperature must still participate in reduction.
            temperature_zero = self.prototype_aligner.temperature.sum() * 0.0
            losses['ita_loss'] = losses.get('ita_loss', temperature_zero) + temperature_zero
            
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
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results
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
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results
        elif subdataset == 'rgb':
            proposal_list = self.rgb_rpn_head.aug_test_rpn(x, img_metas)
            return self.rgb_roi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale)

        elif subdataset == 'ifr':
            proposal_list = self.ifr_rpn_head.aug_test_rpn(x, img_metas)
            return self.ifr_roi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale)

    def forward_for_gradcam(self, img, subdataset):
        """
        Forward pass for Grad-CAM: returns raw classification logits (before sigmoid/softmax/NMS).
        Only supports 'sar' modality for now.
        
        Args:
            img (Tensor): Input image tensor of shape (1, C, H, W)
            subdataset (str): Modality name, e.g., 'sar'
        
        Returns:
            Tensor: Raw classification logits of shape (N, num_classes)
        """
        assert subdataset == 'sar', "Grad-CAM currently only supports SAR modality."
        

        feats, _ = self.extract_feat(img, [subdataset])
        if feats[0].dim() == 3:
            feats = [f.unsqueeze(0) for f in feats]
        cls_scores, bbox_preds, _ = self.sar_bbox_head(feats)
        all_cls = []
        for score in cls_scores:
            B, C, H, W = score.shape
            score = score.permute(0, 2, 3, 1).reshape(-1, C)  # (H*W, num_classes)
            all_cls.append(score)
    
        result = torch.cat(all_cls, dim=0)

        
        return result
