# Copyright (c) OpenMMLab. All rights reserved.
'''
整体pipelline如下: 

sar_img -> backbone -> neck ->
        /-> sar-clip ->  4*4 img feats -> 
做实验发现ov的head效果很差，很难学到新知识，感觉模型是更新不动了
对比v13，不使用ov的head，同时使用
'''


from math import nan
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2result
import numpy as np
from ..builder import ROTATED_DETECTORS, build_backbone, build_head, build_neck
from .base import RotatedBaseDetector 
from ..utils import TextPrototypeAlignerV9, TwoStageProtoInjector
import os
from mmcv.ops import roi_align
from mmdet.core import bbox2roi

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
class TriSourceAlignDetectorV14(RotatedBaseDetector):
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
                 sar_train_cfg=None,
                 sar_test_cfg=None,
                 multi_tasks_reweight=None,
                 reweight_losses=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 align_cfg = None,
                 init_cfg=None):
        super(TriSourceAlignDetectorV14, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained

        self.backbone = build_backbone(backbone)
        self.train_datasets = ['sar', 'rgb', 'ifr']

        if neck is not None:
            self.neck = build_neck(neck)

        if rgb_rpn_head is not None:
            rgb_rpn_train_cfg = rgb_train_cfg.rpn if rgb_train_cfg is not None else None
            rgb_rpn_head_ = rgb_rpn_head.copy()
            rgb_rpn_head_.update(train_cfg=rgb_rpn_train_cfg, test_cfg=rgb_test_cfg.rpn)
            self.rgb_rpn_head = build_head(rgb_rpn_head_)

        if rgb_roi_head is not None:
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
            ifr_rcnn_train_cfg = ifr_train_cfg.rcnn if ifr_train_cfg is not None else None
            ifr_roi_head.update(train_cfg=ifr_rcnn_train_cfg)
            ifr_roi_head.update(test_cfg=ifr_test_cfg.rcnn)
            ifr_roi_head.pretrained = pretrained
            self.ifr_roi_head = build_head(ifr_roi_head)

        self.ifr_train_cfg = ifr_train_cfg
        self.ifr_test_cfg = ifr_test_cfg

        #################################################### HBB
        sar_bbox_head.update(train_cfg=sar_train_cfg)
        sar_bbox_head.update(test_cfg=sar_test_cfg)
        self.sar_bbox_head = build_head(sar_bbox_head)

        self.sar_train_cfg = sar_train_cfg
        self.sar_test_cfg = sar_test_cfg

        self.multi_tasks_reweight = multi_tasks_reweight
        self.reweight_losses = reweight_losses
        if self.multi_tasks_reweight=='uncertainty':
            task_num = len(self.reweight_losses) 
            self.mtl_sigma = torch.nn.Parameter(torch.ones(task_num, requires_grad=True))
        elif self.multi_tasks_reweight=='dwa':
            self.T = 3
            self.history_loss = None
            
        # ==========================================================
        # 🔥 [优化后] Aligner 初始化逻辑
        # ==========================================================
        self.prototype_aligner = None
        
        if align_cfg is not None:
            # 1. 提取 Semantic Mapping (用于 Loss 计算)
            semantic_mapping = align_cfg.get('semantic_mapping', None)
            assert semantic_mapping is not None, "align_cfg MUST contain 'semantic_mapping'!"

            # 2. 提取各模态具体的类别列表 (用于初始化 text_feats_{m})
            # 🔥 [修改] 显式提取列表
            sar_cat = align_cfg.get('sar_category', [])
            rgb_cat = align_cfg.get('rgb_category', [])
            ifr_cat = align_cfg.get('ifr_category', [])

            # 3. 计算各模态的局部类别数
            modal_num_classes = {}
            
            # 优先从 Head 获取 num_classes，获取不到则用列表长度
            if hasattr(self, 'sar_bbox_head') and hasattr(self.sar_bbox_head, 'num_classes'):
                modal_num_classes['sar'] = self.sar_bbox_head.num_classes
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

            # 4. 准备 Aligner 参数
            self.use_text = align_cfg.get('init_clip', True)
            self.is_base_training = align_cfg.get('is_base_training', False)
            self.use_llm_prompt = align_cfg.get('use_llm_prompt', False)
            
            aligner_args = dict(
                modal_num_classes=modal_num_classes,
                semantic_mapping=semantic_mapping,   # 用于 Loss
                
                # 🔥 [修改] 传入具体的类别列表
                sar_category=sar_cat,
                rgb_category=rgb_cat,
                ifr_category=ifr_cat,
                
                in_dims=align_cfg.get('in_dims', {'sar': 512, 'rgb': 512, 'ifr': 512, 'text': 512}),
                modalities=list(align_cfg.get('in_dims', {}).keys()),
                align_dim=align_cfg.get('align_dim', 256),
                ema_beta=align_cfg.get('ema_beta', 0.01),
                vol_weight=align_cfg.get('vol_weight', 0.5),
                is_base_training=self.is_base_training,
                eps=1e-6,
            )

            # 5. CLIP 参数
            sar_clip_args = dict(
                sar_clip_name=align_cfg.get('sar_clip_name', 'ViT-B-16'),
                sar_clip_path=align_cfg.get('sar_clip_path', ''),
                sar_clip_cache=align_cfg.get('sar_clip_cache', ''),
            )
            aligner_args.update(sar_clip_args)

            if self.use_text:
                aligner_args['prompt_file'] = align_cfg.get('prompt_file', None)
                aligner_args['use_llm_prompt'] = self.use_llm_prompt
            
            self.prototype_aligner = TextPrototypeAlignerV9(**aligner_args)
            
            self.proto_injector = TwoStageProtoInjector()

    @property
    def with_rgb_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rgb_rpn_head') and self.rgb_rpn_head is not None

    
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
        if self.with_neck:
            if len(datasets)>1:
                assert is_train
                sar_x, rgb_x, ifr_x = self.split_batch(x)
                sar_x = self.neck(sar_x, start_level=1, add_extra_convs='on_output')
                rgb_x = self.neck(rgb_x)
                ifr_x = self.neck(ifr_x)
                x = (sar_x, rgb_x, ifr_x)
            else:
                assert not is_train
                if datasets[0] == 'sar':
                    x = self.neck(x, start_level=1, add_extra_convs='on_output')
                elif datasets[0] in ['rgb', 'ifr']:
                    x = self.neck(x)
                else:
                    assert False, 'Invalid dataset'

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
       
        results = self.sar_bbox_head.forward(x)
        outs = outs + (results, )
        return outs
        
    def _extract_and_update_prototypes(self, feats, img_metas, gt_bboxes, gt_labels, modality):
        """
        从 feats (Tuple) 中提取特征，整理成字典，并调用 aligner.update_banks
        """
        # 定义不同模态对应的 Stride
        # SAR 从 P3(8) 开始, RGB 从 P2(4) 开始
        if modality == 'sar':
            strides = [8, 16, 32, 64, 128][:len(feats)]
        else:
            strides = [4, 8, 16, 32, 64][:len(feats)]
            
        if len(gt_bboxes) == 0: return {}

        rois = bbox2roi(gt_bboxes)
        
        batch_labels = torch.cat(gt_labels)
        
        update_dict = {} # {stride: (feats, labels)}

        # 遍历每一层特征图，手动提取
        for i, feat_map in enumerate(feats):
            stride = strides[i]
            # 🔥 [修复关键点] 强制类型对齐：如果特征图是 FP16，rois 也必须转为 FP16
            if rois.dtype != feat_map.dtype:
                rois = rois.to(feat_map.dtype)
            # 注意 spatial_scale = 1/stride
            # roi_feats = roi_align(feat_map, rois, (7, 7), spatial_scale=1.0/stride, sampling_ratio=0)
            roi_feats = roi_align(feat_map, rois, (7, 7), 1.0/stride, 0, 'avg', True)
            # GAP -> [N, C]
            roi_vectors = roi_feats.mean(dim=[2, 3])
            # 存入字典
            update_dict[stride] = (roi_vectors.detach(), batch_labels) # Detach!
            
        return update_dict

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
        
        self.source_ratio = [len(sar_gt_labels), len(rgb_gt_labels), len(ifr_gt_labels)]

        batch_inputs = []  
        for each in self.train_datasets:
            if len(img[each])>0:
                batch_inputs.append(img[each])

        # try:
        x, gate_loss = self.extract_feat(batch_inputs, self.train_datasets, is_train=True)        
            # raise Exception("test")
        
        losses = dict()
        if gate_loss is not None:
            losses.update({'gate_loss': gate_loss})
 
        sar_x, rgb_x, ifr_x = x
        
        # =======================================================
        # [Step 1] 更新 Prototypes
        # =======================================================
        batch_bank_updates = {} # 汇总所有模态的更新数据

        if len(sar_gt_labels) > 0:
            batch_bank_updates['sar'] = self._extract_and_update_prototypes(
                sar_x, sar_img_metas, sar_gt_bboxes, sar_gt_labels, 'sar')
        
        if len(rgb_gt_labels) > 0:
            batch_bank_updates['rgb'] = self._extract_and_update_prototypes(
                rgb_x, rgb_img_metas, rgb_gt_bboxes, rgb_gt_labels, 'rgb')
                
        if len(ifr_gt_labels) > 0:
            batch_bank_updates['ifr'] = self._extract_and_update_prototypes(
                ifr_x, ifr_img_metas, ifr_gt_bboxes, ifr_gt_labels, 'ifr')
        
        # 统一执行 Bank 更新
        self.prototype_aligner.update_banks(batch_bank_updates)    
        
        # =======================================================
        # [Step 2] 知识注入 (Feature Injection)
        # =======================================================
        enhanced_sar_x = []
        sar_strides = [8, 16, 32, 64, 128] # SAR FPN 对应的 Stride

        if len(sar_gt_labels) > 0:
            for i, feat in enumerate(sar_x):
                current_stride = sar_strides[i]
                
                # 调用 Injector
                new_feat = self.proto_injector(
                    sar_feat=feat, 
                    aligner=self.prototype_aligner, 
                    stride=current_stride
                )
                enhanced_sar_x.append(new_feat)
            
            # 用增强后的特征替换原始特征
            # 注意: 如果你是 tuple, 需要转回 tuple
            sar_x = tuple(enhanced_sar_x)
        
        if len(sar_gt_labels) > 0:
            batch_input_shape = tuple(img['sar'][0].size()[-2:])
            for img_meta in sar_img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
    
            single_stage_losses = self.sar_bbox_head.forward_train(sar_x, sar_img_metas, sar_gt_bboxes,
                                                        sar_gt_labels, gt_bboxes_ignore)

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
                batch_weight = torch.ones_like(losses).cuda()
            loss_sum = torch.mul(cur_losses, batch_weight).sum()
        
            reweight_losses.update({'reweighted_total_losses':loss_sum})
            self.history_loss = cur_losses.detach().cpu().numpy()
            
            return reweight_losses

        return losses

    def _inject_sar_features_for_test(self, x):
        """
        在推理阶段对 SAR 特征进行增强 (Cross Attention / Feature Injection)
        """
        enhanced_sar_x = []
        # 必须与 forward_train 保持一致的 strides 设置
        sar_strides = [8, 16, 32, 64, 128] 
        
        # 遍历 FPN 的每一层特征
        for i, feat in enumerate(x):
            if i >= len(sar_strides): 
                enhanced_sar_x.append(feat)
                continue
                
            current_stride = sar_strides[i]
            
            # 🔥 关键：调用 injector
            # 注意：这里不需要 update_banks，直接利用已有的 prototype_aligner.bank
            new_feat = self.proto_injector(
                sar_feat=feat, 
                aligner=self.prototype_aligner, 
                stride=current_stride
            )
            enhanced_sar_x.append(new_feat)
            
        return tuple(enhanced_sar_x)

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
            x = self._inject_sar_features_for_test(x)
            results_list = self.sar_bbox_head.simple_test(
                x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_bbox_head.num_classes)
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
            # =======================================================
            # 🔥 [新增] 测试时特征注入 (针对 aug_test 的列表结构)
            # =======================================================
            # 如果 x 是列表 (多个 aug)，则对每个 aug 的特征都做注入
            if isinstance(x, list):
                new_x = []
                for feat_tuple in x:
                    # 如果 feat_tuple 里面还包着 expert_id，要拆包 (视具体实现而定)
                    if isinstance(feat_tuple, tuple) and len(feat_tuple)==2 and isinstance(feat_tuple[1], list):
                        feats, _ = feat_tuple
                    else:
                        feats = feat_tuple
                        
                    injected_feats = self._inject_sar_features(feats)
                    new_x.append(injected_feats)
                x = new_x
            else:
                # 如果 x 只是单个 tuple (虽然 aug_test 不常见)
                x = self._inject_sar_features_for_test(x)
            results_list = self.sar_bbox_head.aug_test(
            x, img_metas, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_bbox_head.num_classes)
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