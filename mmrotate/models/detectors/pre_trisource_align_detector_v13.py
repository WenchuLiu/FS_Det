# Copyright (c) OpenMMLab. All rights reserved.
'''
整体pipelline如下: 

sar_img -> backbone -> neck ->
        /-> sar-clip ->  4*4 img feats -> 

对比v12删去了蒸馏，看蒸馏影响大不大
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
from ..utils import TextPrototypeAlignerV8
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
class TriSourceAlignDetectorV13(RotatedBaseDetector):
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
        super(TriSourceAlignDetectorV13, self).__init__(init_cfg)
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
            
            self.prototype_aligner = TextPrototypeAlignerV8(**aligner_args)

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
    
    def _get_sar_teacher_patches(self, img_raw_tensor, gt_bboxes_list, expand_ratio=1.2):
        """
        Args:
            img_raw_tensor (Tensor): [B, 3, H, W], 范围 0-255
            gt_bboxes_list (list[Tensor]): [N, 5] 或 [N, 4]
            expand_ratio (float): 扩充系数，建议 1.2 ~ 1.5
        """
        teacher_patches_list = []
        
        imgs = img_raw_tensor.float()
        
        for i, (img, gt_bboxes) in enumerate(zip(imgs, gt_bboxes_list)):
            if len(gt_bboxes) == 0:
                teacher_patches_list.append(None)
                continue

            # 1. 处理 Box (旋转转水平)
            if gt_bboxes.shape[-1] == 5:
                from mmrotate.core import rbbox2bbox
                hbboxes = rbbox2bbox(gt_bboxes)
            else:
                hbboxes = gt_bboxes

            patches = []
            img_h, img_w = img.shape[1], img.shape[2]
            
            for bbox in hbboxes:
                x1, y1, x2, y2 = bbox[:4] # 保持 float 精度计算中心点
                
                # 计算中心点和宽高
                w = x2 - x1
                h = y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
                # 🔥 [改进]：计算最大边长并扩展，做成正方形 (Square Crop)
                # 这样 resize 到 224x224 时物体不会变形
                max_side = max(w, h)
                new_side = max_side * expand_ratio
                
                # 计算新的坐标
                new_x1 = cx - new_side / 2
                new_y1 = cy - new_side / 2
                new_x2 = cx + new_side / 2
                new_y2 = cy + new_side / 2
                
                # 转回 int 并处理边界截断
                new_x1 = int(max(0, new_x1))
                new_y1 = int(max(0, new_y1))
                new_x2 = int(min(img_w, new_x2))
                new_y2 = int(min(img_h, new_y2))
                
                # 检查有效性
                if new_x2 - new_x1 < 2 or new_y2 - new_y1 < 2:
                    # Padding 全黑
                    patch = torch.zeros((3, 224, 224), device=img.device)
                else:
                    patch = img[:, new_y1:new_y2, new_x1:new_x2]
                    # 2. Resize 到 224x224
                    # 此时 patch 接近正方形，resize 不会造成严重形变
                    patch = F.interpolate(patch.unsqueeze(0), size=(224, 224), 
                                          mode='bilinear', align_corners=False).squeeze(0)
                                          
                patches.append(patch)
            
            if len(patches) > 0:
                patches = torch.stack(patches) # [N, 3, 224, 224]
                
                # 3. 归一化逻辑 (匹配 readtif)
                # [0, 255] -> [-1, 1]
                patches = patches / 255.0
                patches = (patches - 0.5) * 2.0
                
                feats = self.prototype_aligner.extract_teacher_features(patches)
                teacher_patches_list.append(feats)
            else:
                teacher_patches_list.append(None)
                
        return teacher_patches_list

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

        # if 'img_raw' in kwargs:
        #     img_raw_dict = self.gather_dict_values(kwargs['img_raw'])
            
        #     if 'sar' in img_raw_dict and len(gt_labels['sar']) > 0:
        #         # 获取 Raw Images [B, 3, H, W] (0-255)
        #         sar_raw_imgs = img_raw_dict['sar'] 
                
        #         # 传入我们写的 Crop 函数
        #         sar_distill_feats = self._get_sar_teacher_patches(
        #             sar_raw_imgs,      # <--- 只有这里用 Raw
        #             gt_bboxes['sar'], 
        #             expand_ratio=1.3
        #         )
        
        # 为收集对齐所需的 embeddings & labels
        batch_protos = {}
        batch_labels = {}

        
        if len(sar_gt_labels) > 0:
            batch_input_shape = tuple(img['sar'][0].size()[-2:])
            for img_meta in sar_img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
    
            # single_stage_losses = self.sar_bbox_head.forward_train(sar_x, sar_img_metas, sar_gt_bboxes,
            #                                             sar_gt_labels, gt_bboxes_ignore)
            single_stage_losses = self.sar_bbox_head.forward_train(
                                        sar_x, 
                                        sar_img_metas, 
                                        sar_gt_bboxes,
                                        sar_gt_labels, 
                                        gt_bboxes_ignore,
                                        text_embeds=self.prototype_aligner.text_feats_sar, # <--- You MUST pass this!
                                    )

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

        # 🔥 [新增] 提前把 img_raw 拿出来，防止传给后续不支持的模块
        img_raw = kwargs.pop('img_raw', None)
        
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
                                                    rgb_gt_bboxes, rgb_gt_labels,self.prototype_aligner.text_feats_rgb,
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
                                                    ifr_gt_bboxes, ifr_gt_labels, self.prototype_aligner.text_feats_ifr,
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

        if self.prototype_aligner is not None:
            # -------------------------------------------------------
            # Case A: 正常情况 (本 Batch 提取到了正样本)
            # -------------------------------------------------------
            if len(align_protos) > 0:
                # 1. 更新 Banks
                self.prototype_aligner.update_banks(align_protos, align_labels)
                
                # 2. 计算 Align Loss (Volume + OT)
                # 内部已有 dummy hook，安全
                align_losses = self.prototype_aligner.compute_align_loss(align_protos, align_labels)
                losses.update(align_losses)
            
        return losses


    def simple_test(self, img, img_metas, subdataset, proposals=None, rescale=False):
        """Test without augmentation."""
        assert isinstance(subdataset[0], list) and len(subdataset) == 1
        assert all(x == subdataset[0][0] for x in subdataset[0]), "Not all elements in subdataset are the same: " + str(subdataset)
        subdataset = subdataset[0][0] # e.g. 'sar'

        # 1. 提取视觉特征
        x = self.extract_feat(img, [subdataset]) 

        if isinstance(x, tuple):
            x, experts_id = x
        else:
            experts_id = None
            
        # 2. 🔥 [关键新增] 获取 Text Embeddings (符合训练范式)
        # 训练时你在 forward_train 里传了 self.prototype_aligner.text_feats_xxx
        # 测试时也必须传，否则 Head 里的 Attention 或 Cosine Classifier 无法工作
        text_embeds = None
        if self.prototype_aligner is not None:
            if subdataset == 'sar':
                text_embeds = self.prototype_aligner.text_feats_sar
            elif subdataset == 'rgb':
                text_embeds = self.prototype_aligner.text_feats_rgb
            elif subdataset == 'ifr':
                text_embeds = self.prototype_aligner.text_feats_ifr

        # 3. 推理并传递 text_embeds
        if subdataset == 'sar':
            # 注意：你的 Head (sar_bbox_head) 的 simple_test 需要支持 text_embeds 参数
            # 或者 Head 内部的 forward 能够接收这个参数
            results_list = self.sar_bbox_head.simple_test(
                x, img_metas, rescale=rescale, text_embeds=text_embeds)
            
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
                x, proposal_list, img_metas, rescale=rescale, text_embeds=text_embeds)
        
        elif subdataset == 'ifr':
            if proposals is None:
                proposal_list = self.ifr_rpn_head.simple_test_rpn(x, img_metas)
            else:
                proposal_list = proposals
            
            return self.ifr_roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale, text_embeds=text_embeds)

    def aug_test(self, imgs, img_metas, subdataset, rescale=False):
        """Test with augmentations."""
        assert isinstance(subdataset[0], list) and len(subdataset) == 1
        assert all(x == subdataset[0][0] for x in subdataset[0]), "Not all elements in subdataset are the same: " + str(subdataset)
        subdataset = subdataset[0][0]
        
        # 1. 提取特征
        x = self.extract_feat(imgs, [subdataset]) 
        
        # 2. 🔥 [关键新增] 获取 Text Embeddings
        text_embeds = None
        if self.prototype_aligner is not None:
            if subdataset == 'sar':
                text_embeds = self.prototype_aligner.text_feats_sar
            elif subdataset == 'rgb':
                text_embeds = self.prototype_aligner.text_feats_rgb
            elif subdataset == 'ifr':
                text_embeds = self.prototype_aligner.text_feats_ifr

        # 3. 推理
        if subdataset == 'sar':
            results_list = self.sar_bbox_head.aug_test(
                x, img_metas, rescale=rescale, text_embeds=text_embeds)
            
            bbox_results = [
                bbox2result(det_bboxes, det_labels, self.sar_bbox_head.num_classes)
                for det_bboxes, det_labels in results_list
            ]
            return bbox_results

        elif subdataset == 'rgb':
            proposal_list = self.rgb_rpn_head.aug_test_rpn(x, img_metas)
            return self.rgb_roi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale, text_embeds=text_embeds)

        elif subdataset == 'ifr':
            proposal_list = self.ifr_rpn_head.aug_test_rpn(x, img_metas)
            return self.ifr_roi_head.aug_test(
                x, proposal_list, img_metas, rescale=rescale, text_embeds=text_embeds)

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