# Copyright (c) OpenMMLab. All rights reserved.
'''
整体pipelline如下: 

sar_img -> backbone -> neck ->
        /-> sar-clip ->  4*4 img feats -> 
做实验发现ov的head效果很差，很难学到新知识，感觉模型是更新不动了
对比v14，把每层都有prototype改成共用一个prototype
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
from ..utils import TextPrototypeAlignerV10, FrequencySpectralInjector
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
class TriSourceAlignDetectorV15(RotatedBaseDetector):
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
        super(TriSourceAlignDetectorV15, self).__init__(init_cfg)
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
            
        # # ==========================================================
        # # 🔥 [优化后] Aligner 初始化逻辑
        # # ==========================================================
        # self.prototype_aligner = None
        
        # if align_cfg is not None:
        #     # 1. 提取 Semantic Mapping (用于 Loss 计算)
        #     semantic_mapping = align_cfg.get('semantic_mapping', None)
        #     assert semantic_mapping is not None, "align_cfg MUST contain 'semantic_mapping'!"

        #     # 2. 提取各模态具体的类别列表 (用于初始化 text_feats_{m})
        #     # 🔥 [修改] 显式提取列表
        #     sar_cat = align_cfg.get('sar_category', [])
        #     rgb_cat = align_cfg.get('rgb_category', [])
        #     ifr_cat = align_cfg.get('ifr_category', [])

        #     # 3. 计算各模态的局部类别数
        #     modal_num_classes = {}
            
        #     # 优先从 Head 获取 num_classes，获取不到则用列表长度
        #     if hasattr(self, 'sar_bbox_head') and hasattr(self.sar_bbox_head, 'num_classes'):
        #         modal_num_classes['sar'] = self.sar_bbox_head.num_classes
        #     else:
        #         modal_num_classes['sar'] = len(sar_cat)

        #     if hasattr(self, 'rgb_roi_head') and hasattr(self.rgb_roi_head, 'bbox_head'):
        #         modal_num_classes['rgb'] = self.rgb_roi_head.bbox_head.num_classes
        #     else:
        #          modal_num_classes['rgb'] = len(rgb_cat)

        #     if hasattr(self, 'ifr_roi_head') and hasattr(self.ifr_roi_head, 'bbox_head'):
        #         modal_num_classes['ifr'] = self.ifr_roi_head.bbox_head.num_classes
        #     else:
        #         modal_num_classes['ifr'] = len(ifr_cat)

        #     # 4. 准备 Aligner 参数
        #     self.use_text = align_cfg.get('init_clip', True)
        #     self.is_base_training = align_cfg.get('is_base_training', False)
        #     self.use_llm_prompt = align_cfg.get('use_llm_prompt', False)
            
        #     aligner_args = dict(
        #         modal_num_classes=modal_num_classes,
        #         semantic_mapping=semantic_mapping,   # 用于 Loss
                
        #         # 🔥 [修改] 传入具体的类别列表
        #         sar_category=sar_cat,
        #         rgb_category=rgb_cat,
        #         ifr_category=ifr_cat,
                
        #         in_dims=align_cfg.get('in_dims', {'sar': 512, 'rgb': 512, 'ifr': 512, 'text': 512}),
        #         modalities=list(align_cfg.get('in_dims', {}).keys()),
        #         align_dim=align_cfg.get('align_dim', 256),
        #         ema_beta=align_cfg.get('ema_beta', 0.01),
        #         vol_weight=align_cfg.get('vol_weight', 0.5),
        #         is_base_training=self.is_base_training,
        #         eps=1e-6,
        #     )

        #     # 5. CLIP 参数
        #     sar_clip_args = dict(
        #         sar_clip_name=align_cfg.get('sar_clip_name', 'ViT-B-16'),
        #         sar_clip_path=align_cfg.get('sar_clip_path', ''),
        #         sar_clip_cache=align_cfg.get('sar_clip_cache', ''),
        #     )
        #     aligner_args.update(sar_clip_args)

        #     if self.use_text:
        #         aligner_args['prompt_file'] = align_cfg.get('prompt_file', None)
        #         aligner_args['use_llm_prompt'] = self.use_llm_prompt

        #     self.prototype_aligner = TextPrototypeAlignerV10(**aligner_args)

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
        x, backbone_loss = self.extract_feat(batch_inputs, self.train_datasets, is_train=True)        
            # raise Exception("test")
        
        losses = dict()
        # --- [修改核心] 判断并解包 Loss ---
        if backbone_loss is not None:
            if isinstance(backbone_loss, dict):
                # 1.1 路由损失 (Gate Loss) - 保持默认或小权重
                if 'gate_loss' in backbone_loss:
                    losses['loss_moe_gate'] = backbone_loss['gate_loss'] * 1.0 
                
                # 1.2 解耦损失 (Decoupling Loss) - 🔥 加大权重！
                # 这是防止掉点的关键，强迫高低频彻底分开
                if 'dec_loss' in backbone_loss:
                    losses['loss_moe_dec'] = backbone_loss['dec_loss'] * 0.5 
            else:
                # 兼容旧逻辑
                losses['loss_moe_gate'] = backbone_loss
 
        sar_x, rgb_x, ifr_x = x
        
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