# Copyright (c) OpenMMLab. All rights reserved.
'''
尝试使用text moe

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
from ..utils import PrototypeAlignerV5, TextPrototypeAlignerV5
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
class TriSourceAlignDetectorV9(RotatedBaseDetector):
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
        super(TriSourceAlignDetectorV9, self).__init__(init_cfg)
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
            
        # =============new add================
        if align_cfg is not None:
            # 尝试从 sar head 获取 num_classes（以 sar 为准，需与你 heads 保持一致）
            try:
                self.num_classes = self.sar_bbox_head.num_classes
            except Exception:
                # 如果无法得到，则需要使用 cfg 提供
                self.num_classes = align_cfg.get('num_classes', None)
                assert self.num_classes is not None, "align_cfg must include num_classes if sar_bbox_head.num_classes not available"

            in_dims = align_cfg.get('in_dims', {'sar': 256, 'rgb': 256, 'ifr': 256})
            align_dim = align_cfg.get('align_dim', 256)
            ema_beta = align_cfg.get('ema_beta', 0.05)
            vol_weight = align_cfg.get('vol_weight', 0.5)
            self.use_text = align_cfg.get('init_clip', True)
            self.is_base_training = align_cfg.get('is_base_training', False)
            self.use_llm_prompt= align_cfg.get('use_llm_prompt', True)
            self.prompt_file = align_cfg.get('prompt_file', "/liuwenchu/code/detection/category_descriptions.json")
            self.semantic_mapping = align_cfg.get('semantic_mapping', None)
            self.new_class_start_id = align_cfg.get('new_class_start_id', None)
            if self.new_class_start_id is None and not self.is_base_training:
                warnings.warn("new_class_start_id not provided in align_cfg for incremental training!")
            self.category_list = align_cfg.get('category_list', None)         

            if self.use_text:
                self.prototype_aligner = TextPrototypeAlignerV5(
                    num_classes=self.num_classes,
                    in_dims=in_dims,
                    semantic_mapping=self.semantic_mapping,
                    modalities=('sar','rgb','ifr','text'),
                    align_dim=align_dim,
                    ema_beta=ema_beta,
                    vol_weight=vol_weight,
                    is_base_training = self.is_base_training,
                    use_llm_prompt = self.use_llm_prompt,
                    prompt_file=self.prompt_file,
                    category_list=self.category_list,
                    clip_model_name="lcybuaa/Git-RSCLIP",  # CLIP模型名称
                )
            else:
                self.prototype_aligner = PrototypeAlignerV5(
                    num_classes=self.num_classes,
                    in_dims=in_dims,
                    semantic_mapping=self.semantic_mapping,
                    modalities=('sar','rgb','ifr'),
                    align_dim=align_dim,
                    ema_beta=ema_beta,
                    vol_weight=vol_weight,
                    is_base_training = self.is_base_training,
                )
        else:
            self.prototype_aligner = None
            
        if align_cfg is not None:
            # [新增] 读取指定的新类列表
            novel_class_ids = align_cfg.get('novel_class_ids', None)
            
            # 将列表转换为 Tensor 并注册为 buffer，这样它会自动随模型去 GPU
            if novel_class_ids is not None:
                # register_buffer 的第二个参数必须是 Tensor
                self.register_buffer('novel_classes_tensor', torch.tensor(novel_class_ids, dtype=torch.long))
            else:
                # 如果没提供，初始化为 None (注意 register_buffer 不能直接存 None，这里用属性存)
                self.novel_classes_tensor = None
                # 兼容旧逻辑：如果没提供具体列表，但提供了 start_id
                self.new_class_start_id = align_cfg.get('new_class_start_id', None)
            
            modal_class_mapping = align_cfg.get('modal_class_mapping', None)
            assert modal_class_mapping is not None, "modal_class_mapping must be provided in align_cfg"
            
            # ✅ 保存为普通 dict of tensors（未注册为 buffer）
            self._modal_classes = {}
            self._modal_class_maps = {}
            
            for modal, global_cls_ids in modal_class_mapping.items():
                if isinstance(global_cls_ids, list):
                    classes_tensor = torch.tensor(global_cls_ids, dtype=torch.long)
                else:
                    classes_tensor = global_cls_ids  # 已是 tensor
                
                self._modal_classes[modal] = classes_tensor
                
                # 构建映射：全局类别索引 → 本地索引
                mapping = torch.full((self.num_classes,), -1, dtype=torch.long)
                if classes_tensor.numel() > 0:
                    mapping[classes_tensor] = torch.arange(len(classes_tensor), dtype=torch.long)
                self._modal_class_maps[modal] = mapping

            # 注意：这里没有 register_buffer！
        else:
            self._modal_classes = {}
            self._modal_class_maps = {}
            
      
        if self.prototype_aligner is not None and getattr(self, 'use_text', False):
            self._inject_text_to_backbone()

    def _inject_text_to_backbone(self):
        """
        从 Aligner 提取 Text 特征并注入 Backbone
        """
        # [Debug] 打印一下，看看是不是这里抛出异常或者没进去
        print(f"DEBUG: Injecting text features shape: {self.prototype_aligner.text_feats.shape}")
        try:
            # 1. 临时获取 Text Features (此时模型还在 CPU 上)
            # 注意：prototype_aligner 必须已经初始化了 text_feats
            text_features = self.prototype_aligner.text_feats # [N_classes, D_raw]
            # 3. 注入 Backbone
            if hasattr(self.backbone, 'init_text_features'):
                self.backbone.init_text_features(text_features)
                
        except Exception as e:
            print(f"[Warning] Failed to inject text features at init: {e}")

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
    
    def compute_ita_loss(self, feats, labels, text_feats, modal_name):
        """
        Intra-Modal Alignment Loss (模态内对齐)
        只计算当前模态对应的 subset text features 的 CE Loss。
        """
        device = feats.device
        
        # 1. [Debug] 检查梯度链是否断裂 (非常重要!)
        # 如果打印 False，说明 Backbone 被冻结了或者 Head 返回的特征被 detach 了
        if not feats.requires_grad and self.iter % 100 == 0:
            print(f"[Warning] {modal_name} feats requires_grad is False! ITA Loss won't work.")

        # 2. 获取映射表
        # modal_classes: 当前模态包含的全局 ID (e.g. SAR -> [0,1,2,3,4])
        modal_classes = self._modal_classes[modal_name].to(device)
        # class_map: 全局 ID -> 局部 ID (e.g. {0:0, 1:1, ...})
        class_map = self._modal_class_maps[modal_name].to(device)

        # 3. 筛选 Text Features (只取当前模态相关的文本)
        # 维度变化: (Total_Classes, D) -> (N_modal_classes, D)
        modal_text_feats = text_feats[modal_classes]

        # 4. 转换标签 (Global ID -> Local ID)
        # 这是为了配合 softmax 的维度 (N_modal_classes)
        modal_labels = class_map[labels]  # (N,)

        # 5. [安全过滤] 剔除不在当前模态列表中的 Label
        # 正常情况下不应该发生，但为了防止代码崩溃 (如 labels 中混入了背景类或忽略类)
        valid_mask = modal_labels >= 0
        if not valid_mask.all():
            # 过滤掉非法样本
            feats = feats[valid_mask]
            modal_labels = modal_labels[valid_mask]
            
            if feats.numel() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

        # 6. 归一化 (Cosine Similarity 前置步骤)
        feats = F.normalize(feats, dim=-1)
        modal_text_feats = F.normalize(modal_text_feats, dim=-1)
        
        # 7. 计算 Logits
        # 温度系数
        logit_scale = self.prototype_aligner.temperature.exp().clamp(max=100.0)
        
        # (N_batch, N_modal_classes)
        logits = torch.matmul(feats, modal_text_feats.T) * logit_scale
        
        # 8. 计算 Standard Cross Entropy
        # 关键修正: 这里必须传入 modal_labels (局部ID)，而不是 labels (全局ID)
        loss = F.cross_entropy(logits, modal_labels)
        
        return loss

    def compute_proto_contrastive_loss(self):
        """
        [Debug版] Prototype-Text Contrastive (PTC) Loss
        自动过滤无效ID，并打印调试信息。
        """
        loss_total = 0.0
        cnt = 0
        tau = 0.07
        
        # 1. 检查 Text Bank
        if self.prototype_aligner is None or 'text' not in self.prototype_aligner.banks:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        text_bank = self.prototype_aligner.banks['text']
        
        if not text_bank.valid.any():
            # [DEBUG] 如果 Text Bank 没东西，打印一下
            # print("[PTC Debug] Text Bank is empty! Check update_and_compute_loss.")
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        # 获取所有有效的 Text 原型和 ID
        all_text_protos = text_bank.protos[text_bank.valid]
        all_text_ids = text_bank.valid.nonzero(as_tuple=False).flatten()
        
        all_text_protos = F.normalize(all_text_protos, dim=-1)

        for modal in ['sar', 'rgb', 'ifr']:
            if modal not in self.prototype_aligner.banks:
                continue
                
            vis_bank = self.prototype_aligner.banks[modal]
            
            # 如果该模态目前还没有有效原型，跳过
            if not vis_bank.valid.any():
                continue
                
            # 2. 获取 Visual 原型和 ID
            vis_protos_all = vis_bank.protos[vis_bank.valid]
            vis_ids_all = vis_bank.valid.nonzero(as_tuple=False).flatten()
            
            # 3. [关键修复]：只保留在 Text Bank 中存在的 ID (取交集)
            # mask[i] = True 表示 vis_ids_all[i] 在 all_text_ids 中存在
            is_in_text_mask = torch.isin(vis_ids_all, all_text_ids)
            
            if not is_in_text_mask.any():
                # [DEBUG] 只有当完全没有交集时才跳过
                # print(f"[PTC Debug] No intersection between {modal} and Text IDs.")
                continue

            # 筛选：只取能匹配上的
            vis_protos = vis_protos_all[is_in_text_mask]
            vis_ids = vis_ids_all[is_in_text_mask]
            
            # 归一化
            vis_protos = F.normalize(vis_protos, dim=-1)
            
            # 4. 计算 Logits
            # vis_protos: [N_valid, D] vs all_text_protos: [N_text, D]
            logits = torch.matmul(vis_protos, all_text_protos.T) / tau
            
            # 5. 生成标签
            # 找到 vis_ids 在 all_text_ids 中的索引位置
            # 因为我们已经过滤过了 (is_in_text_mask)，所以这里一定能找到匹配
            mask = vis_ids.unsqueeze(1) == all_text_ids.unsqueeze(0)
            target_idx = mask.nonzero(as_tuple=False)[:, 1]
            
            # 6. 计算 Loss
            loss_v2t = F.cross_entropy(logits, target_idx)
            
            loss_total += loss_v2t
            cnt += 1
            
        if cnt == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        return loss_total / cnt
    
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


        # 为收集对齐所需的 embeddings & labels
        batch_protos = {}
        batch_labels = {}

        if len(sar_gt_labels) > 0:
            batch_input_shape = tuple(img['sar'][0].size()[-2:])
            for img_meta in sar_img_metas:
                img_meta['batch_input_shape'] = batch_input_shape
    
            single_stage_losses = self.sar_bbox_head.forward_train(sar_x, sar_img_metas, sar_gt_bboxes,
                                                sar_gt_labels, gt_bboxes_ignore)

            losses.update({'sar_' + k: v for k, v in single_stage_losses.items() 
               if k not in ['proto_feats', 'proto_labels']})
            
            # 1. 从字典中获取变量
            feats_list = single_stage_losses['proto_feats']
            labels_list = single_stage_losses['proto_labels']

            # 2. **核心检查**: 确保变量本身不是 None
            #    这是专门为了解决你提出的新问题
            if feats_list is not None and labels_list is not None:
                # 3. 过滤列表内部的 None 元素 (处理 [Tensor, None, ...] 的情况)
                #    这里的 for 循环现在是安全的，因为我们已经确认了 feats_list 不是 None
                valid_protos = [p for p in feats_list if p is not None]
                valid_labels = [l for l in labels_list if l is not None]
                # 4. 确保过滤后的列表不为空 (处理 [] 的情况)
                if valid_protos:
                    # print(1)
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
        
        
        # 计算三种模态的对齐损失
        if self.prototype_aligner is not None and len(batch_protos) > 0:
            # print(5)
            align_losses = self.prototype_aligner.update_and_compute_loss(batch_protos, batch_labels)
            losses.update(align_losses)

        # 计算 ITA_Loss
        if self.prototype_aligner is not None and self.use_text:
            ita_losses = []
            text_proj = self.prototype_aligner.proj['text']
            device = next(text_proj.parameters()).device
            text_feats = text_proj(self.prototype_aligner.text_feats.to(device))
            
            # [关键修复] 初始化 total_loss 为 text_feats * 0
            # 这样无论后面有没有真实的 loss，text_proj 的参数都在计算图中，
            # 只是梯度为 0，满足 DDP 要求。
            total_ita_loss = text_feats.sum() * 0.0

            for modal_name, feats in batch_protos.items():
                labels = batch_labels[modal_name]
                if feats is None or feats.numel() == 0:
                    continue
                
                # ... (保持原本的 Mask 逻辑不变) ...
                if self.is_base_training:
                     ita_loss = self.compute_ita_loss(feats, labels, text_feats, modal_name)
                     ita_losses.append(ita_loss)
                else:
                    # Incremental 逻辑
                    mask = None
                    if getattr(self, 'novel_classes_tensor', None) is not None:
                        mask = torch.isin(labels, self.novel_classes_tensor)
                    elif getattr(self, 'new_class_start_id', None) is not None:
                        mask = labels >= self.new_class_start_id
                    
                    if mask is not None and mask.any():
                        ita_loss = self.compute_ita_loss(
                            feats[mask], labels[mask], text_feats, modal_name
                        )
                        ita_losses.append(ita_loss)
                    # 注意：如果 mask 全为 False，这里不要 append 0.0，直接跳过即可
                    # 因为我们已经有了 total_ita_loss 作为保底

            # 累加真实 Loss
            if ita_losses:
                total_ita_loss = total_ita_loss + 0.1 * sum(ita_losses)
            
            # 更新到 losses 字典
            losses.update({'ita_loss': total_ita_loss})

        # [新增] 计算 PTC Loss (语义锚定)
        # 建议权重: 0.1
        # 作用: 防止类中心漂移，利用 RGB/IFR 的 Text 作为 SAR 的负样本
        # if self.prototype_aligner is not None and self.use_text:
        #     loss_ptc = self.compute_proto_contrastive_loss()
        #     losses.update({'loss_ptc': 0.1 * loss_ptc})
            
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