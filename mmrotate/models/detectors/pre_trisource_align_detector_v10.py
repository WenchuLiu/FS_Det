# Copyright (c) OpenMMLab. All rights reserved.
'''
整体pipelline如下: 

sar_img -> backbone -> neck ->
        /-> sar-clip ->  4*4 img feats -> 

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
from ..utils import TextPrototypeAlignerV6
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
class TriSourceAlignDetectorV10(RotatedBaseDetector):
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
        super(TriSourceAlignDetectorV10, self).__init__(init_cfg)
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
        self._modal_classes = {}
        self._modal_class_maps = {}
        
        if align_cfg is not None:
            # 1. 确定类别数 (优先用 Head 的，没有则用 Config 的)
            if hasattr(self, 'sar_bbox_head') and hasattr(self.sar_bbox_head, 'num_classes'):
                self.num_classes = self.sar_bbox_head.num_classes
            else:
                self.num_classes = align_cfg.get('num_classes', None)
                assert self.num_classes is not None, "num_classes not found in head or config!"

            # 2. 提取基础配置
            self.use_text = align_cfg.get('init_clip', True)
            self.is_base_training = align_cfg.get('is_base_training', False)
            self.use_llm_prompt = align_cfg.get('use_llm_prompt', True)
            self.new_class_start_id = align_cfg.get('new_class_start_id', None)
            
            # 检查增量训练配置
            if self.new_class_start_id is None and not self.is_base_training:
                warnings.warn("new_class_start_id not provided in align_cfg for incremental training!")

            # 3. 准备 Aligner 的通用参数
            # 使用字典解包传递参数，代码更整洁
            aligner_args = dict(
                num_classes=self.num_classes,
                in_dims=align_cfg.get('in_dims', {'sar': 256, 'rgb': 256, 'ifr': 256, 'text': 768}),
                semantic_mapping=align_cfg.get('semantic_mapping', None),
                modalities=align_cfg.get('modalities', ('sar', 'rgb', 'ifr', 'text')), # 默认带 text
                align_dim=align_cfg.get('align_dim', 256),
                ema_beta=align_cfg.get('ema_beta', 0.05),
                vol_weight=align_cfg.get('vol_weight', 0.5),
                is_base_training=self.is_base_training,
                fpn_channels=align_cfg.get('fpn_channels', 256),
                eps=align_cfg.get('eps', 1e-6)
            )

            # 4. 准备 SAR-CLIP & Text 相关参数
            # 即使 use_text=False，为了使用 fuse_sar_features，我们依然需要加载 CLIP (Aligner内部逻辑)
            # 但可以通过不传 category_list 来禁用 Text Loss
            
            sar_clip_args = dict(
                sar_clip_name=align_cfg.get('sar_clip_name', 'ViT-B/16'),
                sar_clip_path=align_cfg.get('sar_clip_path', r'./sar_clip/model_configs/ViT-B-16.pt'),
                sar_clip_cache=align_cfg.get('sar_clip_cache', r"./sar_clip/model_configs"),
                sar_all_category=align_cfg.get('sar_all_category', ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']),    
                modal_class_mapping=align_cfg.get('modal_class_mapping', None), # 🔥 传入 Mapping
            )
            aligner_args.update(sar_clip_args)

            # 5. 处理 Text Loss 相关参数
            # 如果 use_text 为 True，传入 category_list，Aligner 会计算 Text Loss
            # 如果 use_text 为 False，传入 None，Aligner 只做 Visual 对齐和 SAR Fuse
            if self.use_text:
                aligner_args['category_list'] = align_cfg.get('category_list', None)
                aligner_args['prompt_file'] = align_cfg.get('prompt_file', None)
                aligner_args['use_llm_prompt'] = self.use_llm_prompt
            else:
                aligner_args['category_list'] = None # 显式关闭 Text Loss

            # 6. 统一实例化 (只用一个类！)
            # 假设你已经把新的统一类命名为 TextPrototypeAlignerV6 并 import 进来了
            from ..utils import TextPrototypeAlignerV6 
            self.prototype_aligner = TextPrototypeAlignerV6(**aligner_args)

            # 7. 处理 Novel Classes Tensor (用于 Loss Mask)
            novel_ids = align_cfg.get('novel_class_ids', None)
            if novel_ids is not None:
                # 如果有值，注册为 buffer (会自动保存到 state_dict)
                self.register_buffer('novel_classes_tensor', torch.tensor(novel_ids, dtype=torch.long))
            else:
                # 如果没有值，赋值为 None (不会保存到 state_dict，训练时不参与设备移动)
                self.novel_classes_tensor = None
            
            # 8. 处理 Modal Class Mapping (用于 compute_ita_loss 的映射)
            modal_mapping = align_cfg.get('modal_class_mapping', None)
            if modal_mapping:
                for modal, global_ids in modal_mapping.items():
                    # 转 Tensor
                    if isinstance(global_ids, list):
                        cls_tensor = torch.tensor(global_ids, dtype=torch.long)
                    else:
                        cls_tensor = global_ids
                    
                    self._modal_classes[modal] = cls_tensor
                    
                    # 创建映射表 [num_classes] -> [-1, 0, 1, ...]
                    # 作用：把 Global Label 映射回该模态内的局部索引
                    mapping_tensor = torch.full((self.num_classes,), -1, dtype=torch.long)
                    if cls_tensor.numel() > 0:
                        # 确保 cls_tensor 在合法范围内
                        valid_cls = cls_tensor[cls_tensor < self.num_classes]
                        mapping_tensor[valid_cls] = torch.arange(len(valid_cls), dtype=torch.long)
                    
                    self._modal_class_maps[modal] = mapping_tensor
            else:
                warnings.warn("modal_class_mapping is missing in align_cfg! Text Loss might fail.")


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
    
    def compute_ita_loss(self, feats, labels, text_feats, modal_name, 
                         neg_feats=None): # 🔥 移除了 batch_idx 参数
        """
        Intra-Modal InfoNCE Loss (Batch-level Negatives)
        允许正样本与当前 Batch 中所有的负样本进行对比。
        """
        device = feats.device
        
        # 1. 基础检查 & 映射 ID (保持不变)
        if not feats.requires_grad and self.iter % 100 == 0:
            print(f"[Warning] {modal_name} feats requires_grad is False!")

        modal_classes = self._modal_classes[modal_name].to(device)
        class_map = self._modal_class_maps[modal_name].to(device)
        modal_text_feats = text_feats[modal_classes]
        modal_labels = class_map[labels]
        
        valid_mask = modal_labels >= 0
        if not valid_mask.all():
            feats = feats[valid_mask]
            modal_labels = modal_labels[valid_mask]
            if feats.numel() == 0: return torch.tensor(0.0, device=device, requires_grad=True)

        # 2. 归一化 (保持不变)
        feats = F.normalize(feats, dim=-1)
        modal_text_feats = F.normalize(modal_text_feats, dim=-1)
        if neg_feats is not None:
            neg_feats = F.normalize(neg_feats, dim=-1)

        logit_scale = self.prototype_aligner.temperature.exp().clamp(max=100.0)

        # ==============================================================
        # Part 1: 正样本 vs. 文本 (Positive Logits)
        # ==============================================================
        # logits: (N_pos, N_cls)
        logits_text_all = torch.matmul(feats, modal_text_feats.T) * logit_scale
        # 取出正确类别的 Logit -> (N_pos, 1)
        pos_logits = logits_text_all.gather(1, modal_labels.view(-1, 1))

        # ==============================================================
        # Part 2: 正样本 vs. 全Batch负样本 (Batch-level Negatives)
        # ==============================================================
        if neg_feats is not None and neg_feats.numel() > 0:
            # 🔥 [核心修改] 不再使用 Mask，直接计算所有相似度
            # feats: (N_pos, D)
            # neg_feats: (B * N_neg_per_img, D) -> 这里包含了整个 Batch 的负样本
            # logits_neg_all: (N_pos, Total_Negs)
            logits_neg_all = torch.matmul(feats, neg_feats.T) * logit_scale
            
            # 拼接: [Pos_Logit, Neg_Logit_1, Neg_Logit_2, ... Neg_Logit_All]
            # Shape: (N_pos, 1 + Total_Negs)
            all_logits = torch.cat([pos_logits, logits_neg_all], dim=1)
            
            # InfoNCE Target: 正样本永远在 index 0
            targets = torch.zeros(feats.size(0), dtype=torch.long, device=device)
            loss = F.cross_entropy(all_logits, targets)
            
            # 这里的权重建议给高一点，因为现在负样本很多，任务变难了，Loss 值会变大
            # 也可以保持 0.5 或 1.0，视量级而定
            return loss 
            
        else:
            # 退化回普通 CE Loss
            return F.cross_entropy(logits_text_all, modal_labels)

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
        
        # # ====================== 🕵️‍♂️ Debug Start ======================
        # print("\n" + ">"*20 + " Deep Debugging " + "<"*20)

        # # 1. 检查 RGB 特征是否正常 (防止 MoE 输出全0或NaN)
        # print(f"[Debug] RGB Feat Level 0 Mean: {rgb_x[0].mean().item():.5f}, Max: {rgb_x[0].max().item():.5f}")
        # if torch.isnan(rgb_x[0]).any():
        #     print("!!! CRITICAL: RGB Features contain NaN !!!")

        # # ==========================================================


        # ==========================================================
        # 🔥 [修正版] SAR-CLIP 特征融合 (调用 Aligner 封装好的方法)
        # ==========================================================
        # 我们不再需要在 Detector 里手动写滑动窗口和 CLIP 推理
        # TextPrototypeAlignerV6.fuse_sar_features 会自动处理：
        # 1. Resize & Unfold (Input-level Sliding Window)
        # 2. CLIP Visual Encoder 推理
        # 3. Global/Local Attention 融合到 FPN
        
        if getattr(self, 'prototype_aligner', None) is not None and \
           hasattr(self.prototype_aligner, 'fuse_sar_features'):
            
            # 直接传入原始图像和 FPN 特征
            # img['sar'] 是 [B, C, H, W] 的 Tensor
            # print(img['sar'].shape)
            sar_x = self.prototype_aligner.fuse_sar_features(img['sar'], sar_x)
            # print('after fuse', sar_x[0].shape)
        
        # # 🔥 [修改] 传入 GT 信息用于可视化
        # # 注意：sar_gt_bboxes 是一个 list of tensors，需要传进去
        # if getattr(self, 'prototype_aligner', None) is not None and \
        #    hasattr(self.prototype_aligner, 'fuse_sar_features'):
            
        #     sar_x = self.prototype_aligner.fuse_sar_features(
        #         img['sar'], 
        #         sar_x,
        #         gt_bboxes=sar_gt_bboxes, # 传入 GT
        #         img_metas=sar_img_metas  # 传入 Meta (为了文件名)
        #     )
        
        
        # 为收集对齐所需的 embeddings & labels
        batch_protos = {}
        batch_labels = {}

        # ======================================================
        # 🔥 [修改 1] 处理 SAR Head 返回的负样本
        # ======================================================
        if len(sar_gt_labels) > 0:
            batch_input_shape = tuple(img['sar'][0].size()[-2:])
            for img_meta in sar_img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
    
            single_stage_losses = self.sar_bbox_head.forward_train(sar_x, sar_img_metas, sar_gt_bboxes,
                                                        sar_gt_labels, gt_bboxes_ignore)

            # 过滤掉非 loss 的 key
            losses.update({'sar_' + k: v for k, v in single_stage_losses.items() 
               if k not in ['proto_feats', 'proto_labels', 'neg_proto_feats', 'neg_proto_labels']})
            
            # --- 收集正样本 ---
            feats_list = single_stage_losses.get('proto_feats', None)
            labels_list = single_stage_losses.get('proto_labels', None)

            if feats_list is not None and labels_list is not None:
                valid_protos = [p for p in feats_list if p is not None]
                valid_labels = [l for l in labels_list if l is not None]
                if valid_protos:
                    batch_protos['sar'] = torch.cat(valid_protos, dim=0)
                    batch_labels['sar'] = torch.cat(valid_labels, dim=0)

            # --- 收集负样本 ---
            neg_feats_list = single_stage_losses.get('neg_proto_feats', None)
            # neg_labels 通常是全 num_classes，其实不太需要传，但为了统一接口还是收集
            # neg_labels_list = single_stage_losses.get('neg_proto_labels', None) 

            if neg_feats_list is not None:
                valid_neg_protos = [p for p in neg_feats_list if p is not None]
                if valid_neg_protos:
                    # 存入 batch_protos，用 'sar_neg' 作为 key
                    batch_protos['sar_neg'] = torch.cat(valid_neg_protos, dim=0)
                    # batch_labels['sar_neg'] 可以不存，因为大家都知道它是背景


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

        # # ================== 🕵️‍♂️ Debug Start: 侦探代码 ==================
        # print("\n" + "="*20 + " Debugging Align Loss " + "="*20)
        
        # # 1. 检查各模态是否提取到了特征
        # print(f"[Step 1] Batch Protos Keys: {batch_protos.keys()}")
        
        # # 2. 检查标签数值范围 (这是最可能的嫌疑人!)
        # for modal in ['sar', 'rgb', 'ifr']:
        #     if modal in batch_labels and batch_labels[modal] is not None:
        #         lbls = batch_labels[modal].unique().tolist()
        #         print(f"  -> {modal.upper()} Labels in this batch: {lbls}")
        #     else:
        #         print(f"  -> {modal.upper()} Labels: MISSING")

        # # 3. 检查 Aligner 的 Bank 状态
        # if self.prototype_aligner is not None:
        #     print("[Step 3] Checking Bank Status inside Aligner:")
        #     mapping = self.prototype_aligner.semantic_mapping
        #     # 挑一个关键类别检查，比如 'ship'
        #     target_class = "ship"
        #     if target_class in mapping:
        #         print(f"  Target mapping for '{target_class}': {mapping[target_class]}")
        #         # 检查对应的 ID 在 Bank 里是否 Valid
        #         for m, idx in mapping[target_class].items():
        #             if m == 'text': continue
        #             # 处理列表或单值
        #             indices = idx if isinstance(idx, list) else [idx]
        #             for i in indices:
        #                 # 保护性检查，防止索引越界
        #                 if i < self.prototype_aligner.banks[m].valid.numel():
        #                     is_valid = self.prototype_aligner.banks[m].valid[i].item()
        #                     print(f"    [{m}] Index {i} Valid? -> {is_valid}")
        #                 else:
        #                     print(f"    [{m}] Index {i} OUT OF BOUNDS! (Bank size: {self.prototype_aligner.banks[m].valid.numel()})")
        
        # print("="*60 + "\n")
        # # ================== 🕵️‍♂️ Debug End ==================
        
        # 计算三种模态的对齐损失 (Grammian Volume, Sinkhorn 等)
        # ⚠️ 注意：必须过滤掉 '_neg' 后缀的 Key
        # 因为 Aligner 的 update_and_compute_loss 会遍历字典更新 Bank，
        # 如果传入 'sar_neg'，它会试图更新一个不存在的 'sar_neg' Bank，导致报错。
        
        # 1. 构建过滤后的字典 (只保留正样本)
        align_protos = {k: v for k, v in batch_protos.items() if not k.endswith('_neg')}
        align_labels = {k: v for k, v in batch_labels.items() if not k.endswith('_neg')}

        # 获取设备，用于创建 dummy tensors
        device = img['sar'].device

        if self.prototype_aligner is not None:
            # -------------------------------------------------------
            # Case A: 正常情况 (本 Batch 提取到了正样本)
            # -------------------------------------------------------
            if len(align_protos) > 0:
                # 1. 更新 Banks
                self.prototype_aligner.update_banks(align_protos, align_labels)
                
                # 2. 计算 Align Loss (Volume + OT)
                # 内部已有 dummy hook，安全
                align_losses = self.prototype_aligner.compute_align_loss(align_protos)
                losses.update(align_losses)

                # 3. 计算 ITA Loss
                if self.use_text:
                    total_ita_loss = torch.tensor(0.0, device=device)
                    
                    for modal_name, feats in align_protos.items():
                        if modal_name == 'text': continue
                        labels = align_labels.get(modal_name, None)
                        
                        # 只要有数据，就正常调用
                        # (即使某模态为空，循环进不去，下面的 else 也会兜底吗？
                        #  不，这里是针对整个 batch 有没有正样本。
                        #  只要 align_protos > 0，说明至少有一个模态有数据，text_proj 就会在 align_loss 里被调用，
                        #  所以这里跳过也没事)
                        if feats is not None and labels is not None and feats.numel() > 0:
                            loss = self.prototype_aligner.compute_ita_loss(feats, labels, modal_name)
                            total_ita_loss += loss
                    
                    losses.update({'ita_loss': 0.1 * total_ita_loss})

            # -------------------------------------------------------
            # 🔥 Case B: 幽灵分支 (有 GT 但无正样本) -> 必须强制空转！
            # -------------------------------------------------------
            else:
                # 1. 强制空转 Align Loss
                # 传入空字典，触发 compute_align_loss 内部的 text_feats 计算 + dummy_hook
                dummy_align = self.prototype_aligner.compute_align_loss({})
                losses.update(dummy_align)
                
                # 2. 强制空转 ITA Loss
                if self.use_text:
                    # 创建一个空的 dummy 输入
                    # 假设 align_dim 为 256，这里设为 0 行
                    d_feat = torch.empty((0, 256), device=device)
                    d_label = torch.empty(0, dtype=torch.long, device=device)
                    
                    # 随便传一个合法的 modal_name (例如 'sar')
                    # 这会进入 compute_ita_loss -> 算出 Text Feats -> 发现 feat 为空 -> 返回 dummy_loss
                    dummy_ita = self.prototype_aligner.compute_ita_loss(d_feat, d_label, 'sar')
                    losses.update({'ita_loss': 0.1 * dummy_ita})
            
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
            
        # ==========================================================
        # 🔥 [新增] SAR-CLIP 特征融合 (Test Time)
        # ==========================================================
        # 只有在 SAR 模态下，且 Aligner 初始化成功时才执行
        if subdataset == 'sar':
            if getattr(self, 'prototype_aligner', None) is not None and \
               hasattr(self.prototype_aligner, 'fuse_sar_features'):
                
                # img 在 simple_test 中通常是 [1, C, H, W] 的 Tensor
                # fuse_sar_features 内部会自动处理 resize 和 sliding window
                x = self.prototype_aligner.fuse_sar_features(img, x)
        # ==========================================================
            
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