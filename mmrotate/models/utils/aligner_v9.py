'''
完全修改了使用prototype的方式，不做不同模态prototype对齐了，转而是不同模态prototype做cross attention
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast
import os
import sys
import json

try:
    from .. import sar_clip
except ImportError as e:
    print(f"[Aligner] Failed to import sar_clip: {e}")

class PrototypeBank(nn.Module):
    def __init__(self, num_classes, dim, beta=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.beta = beta
        
        # 变化点：不再是一个 Tensor，而是一个字典来存不同 Stride 的 buffer
        # 由于 nn.Module 不能直接注册 dict 为 buffer，我们需要用 ModuleDict 或者手动管理 keys
        # 这里为了简单，我们用 nn.ParameterDict 或者普通的 dict (非持久化)，
        # 但为了能在 state_dict 里保存，建议使用 register_buffer 的命名技巧
        self.strides = [] # 记录不管哪个模态可能用到的 stride
        
        # 初始化为空，我们在 update 时动态或者在 init 时预定义 stride
        # 建议：预定义常用的 stride 以便注册 buffer
        self.supported_strides = [4, 8, 16, 32, 64, 128]
        
        for s in self.supported_strides:
            # 命名规则: protos_s8, protos_s16 ...
            self.register_buffer(f'protos_s{s}', torch.zeros(num_classes, dim))
            self.register_buffer(f'valid_s{s}', torch.zeros(num_classes, dtype=torch.bool))

    def get_proto(self, stride):
        """获取特定 stride 的 prototype"""
        if stride not in self.supported_strides:
            return None
        return getattr(self, f'protos_s{stride}')

    def get_valid(self, stride):
        """获取特定 stride 的 valid mask"""
        if stride not in self.supported_strides:
            return None
        return getattr(self, f'valid_s{stride}')

    @torch.no_grad()
    def update(self, class_ids: torch.LongTensor, protos_batch: torch.Tensor, stride: int):
        """
        更新特定 Stride 的 Prototype
        protos_batch: [B, Dim]
        class_ids: [B]
        """
        if stride not in self.supported_strides:
            # 如果遇到了未定义的 stride (比如 SAR 的 128)，可以选择忽略或打印警告
            # print(f"Warning: Stride {stride} not supported in Bank.")
            return

        target_protos = getattr(self, f'protos_s{stride}')
        target_valid = getattr(self, f'valid_s{stride}')

        for i, c in enumerate(class_ids.tolist()):
            if c >= self.num_classes or c < 0: continue 
            
            if target_valid[c]:
                # EMA 更新
                target_protos[c] = (1.0 - self.beta) * target_protos[c] + self.beta * protos_batch[i]
            else:
                # 初始化
                target_protos[c] = protos_batch[i]
                target_valid[c] = True
                

class TextPrototypeAlignerV9(nn.Module):
    def __init__(self,
                 modal_num_classes: dict, 
                 semantic_mapping: dict,  # {"ship": {"sar": 0, "rgb": 3}, ...}
                 
                 # 接收具体的类别列表
                 sar_category=None,
                 rgb_category=None,
                 ifr_category=None,
                 
                 modalities=('sar', 'rgb', 'ifr', 'text'),
                 align_dim=512, 
                 ema_beta=0.01,
                 vol_weight=0.5,
                 is_base_training=True,
                 
                 # SAR-CLIP Params
                 sar_clip_name='ViT-B-16',
                 sar_clip_path='',
                 sar_clip_cache='',
                 
                 # Text Params
                 prompt_file=None,
                 use_llm_prompt=False,
                 eps=1e-6,
                 **kwargs):
        
        super().__init__()
        
        if 'text' not in modalities:
            modalities = list(modalities) + ['text']
            
        self.modalities = list(modalities)
        self.align_dim = align_dim 
        self.vol_weight = vol_weight
        self.eps = eps
        
        self.modal_num_classes = modal_num_classes
        self.semantic_mapping = semantic_mapping
        
        self.category_lists = {
            'sar': sar_category if sar_category else [],
            'rgb': rgb_category if rgb_category else [],
            'ifr': ifr_category if ifr_category else []
        }
        
        # 🔥 [新增] 1. 构建反向查找表: SAR ID -> Concept Name
        # 这样在 compute_align_loss 时，能直接通过 SAR label 找到它属于哪个概念（如 "car"）
        self.sar_id_to_concept = {}
        for concept, map_data in self.semantic_mapping.items():
            sar_id = map_data.get('sar')
            if sar_id is not None:
                # 兼容 list 或 int
                if isinstance(sar_id, list): 
                    for sid in sar_id: self.sar_id_to_concept[sid] = concept
                else:
                    self.sar_id_to_concept[sar_id] = concept

        # 2. 从 Semantic Mapping 中提取核心概念 (用于 Text Bank)
        self.concept_list = list(self.semantic_mapping.keys())
        self.concept_to_id = {name: i for i, name in enumerate(self.concept_list)}

        # =====================================================================
        # [Part A] Init SAR-CLIP
        # =====================================================================
        print(f"[Aligner] Loading SAR-CLIP: {sar_clip_name}")
        full_model = sar_clip.create_model_with_args(sar_clip_name, pretrained=sar_clip_path)
        full_model.eval()
        
        self.clip_visual = full_model.visual
        for p in self.clip_visual.parameters(): p.requires_grad = False
        
        tokenizer = sar_clip.get_tokenizer(sar_clip_name, cache_dir=sar_clip_cache)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        temp_model = full_model.to(device)
        
        with torch.no_grad():
            dummy_emb = temp_model.encode_text(tokenizer(["test"]).to(device))
            self.clip_dim = dummy_emb.shape[-1]

        # =====================================================================
        # [Part B] Generate Text Features
        # =====================================================================
        prompts_dict = {}
        if use_llm_prompt and prompt_file and os.path.exists(prompt_file):
            print(f"[Aligner] Loading LLM prompts from {prompt_file}")
            with open(prompt_file, 'r') as f:
                prompts_dict = json.load(f)

        def get_embedding(text_list):
            if not text_list:
                return torch.zeros(0, self.clip_dim)
            embeddings = []
            with torch.no_grad():
                for cat_name in text_list:
                    if cat_name in prompts_dict:
                        p_content = prompts_dict[cat_name]
                        current_prompts = p_content if isinstance(p_content, list) else [p_content]
                    else:
                        current_prompts = [f"a SAR image of {cat_name}."] 
                    
                    tokens = tokenizer(current_prompts).to(device)
                    feats = temp_model.encode_text(tokens)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    embeddings.append(feats.mean(dim=0).cpu())
            return torch.stack(embeddings)

        visual_modals = [m for m in self.modalities if m != 'text']
        for m in visual_modals:
            cat_list = self.category_lists.get(m, [])
            num_cls = self.modal_num_classes.get(m, 0)
            
            if len(cat_list) == 0:
                self.register_buffer(f'text_feats_{m}', torch.zeros(num_cls, self.clip_dim))
                continue
            
            print(f"[Aligner] Generating text feats for {m} based on {len(cat_list)} names...")
            modal_emb = get_embedding(cat_list)
            
            final_tensor = torch.zeros(num_cls, self.clip_dim)
            copy_len = min(num_cls, len(modal_emb))
            final_tensor[:copy_len] = modal_emb[:copy_len]
            
            self.register_buffer(f'text_feats_{m}', final_tensor)

        if len(self.concept_list) > 0:
            print(f"[Aligner] Generating Global Text Bank for {len(self.concept_list)} concepts...")
            master_text_feats = get_embedding(self.concept_list)
            self.register_buffer('text_feats_global', master_text_feats)
        else:
            self.register_buffer('text_feats_global', torch.zeros(1, self.clip_dim))
        
        del full_model.transformer, full_model.token_embedding, full_model.ln_final
        torch.cuda.empty_cache()

        # =====================================================================
        # [Part C] Init Banks
        # =====================================================================
        self.banks = nn.ModuleDict()
        
        if 'text' in self.modalities:
            n_concepts = len(self.concept_list)
            self.banks['text'] = PrototypeBank(num_classes=n_concepts, dim=self.clip_dim, beta=ema_beta)

        for m in visual_modals:
            n_cls = self.modal_num_classes.get(m, 0)
            if n_cls > 0:
                self.banks[m] = PrototypeBank(num_classes=n_cls, dim=self.align_dim, beta=ema_beta)

        # 用于 Log 的辅助表
        self.local_id_to_concept = {m: {} for m in visual_modals}
        for m in visual_modals:
            cat_list = self.category_lists.get(m, [])
            for idx, name in enumerate(cat_list):
                self.local_id_to_concept[m][idx] = name
    
    # ---------------------------------------------------------------------
    # Helper Methods (🔥 新增)
    # ---------------------------------------------------------------------

    def _get_aggregated_proto(self, concept_name, modality):
        """
        [Helper] 根据 concept_name 和模态，获取(聚合)原型。
        例如: concept='car', modality='rgb', mapping=[0,1]
        则取出 RGB Bank 中 0 和 1 的向量，返回 mean(v0, v1)。
        """
        if modality not in self.banks:
            return None
            
        map_data = self.semantic_mapping.get(concept_name, {})
        target_ids = map_data.get(modality)
        
        if target_ids is None:
            return None
            
        if not isinstance(target_ids, (list, tuple)):
            target_ids = [target_ids]
            
        bank = self.banks[modality]
        valid_protos = []
        for tid in target_ids:
            if 0 <= tid < bank.num_classes and bank.valid[tid]:
                valid_protos.append(bank.protos[tid])
                
        if len(valid_protos) == 0:
            return None
            
        # 聚合: Mean & Normalize
        mean_proto = torch.stack(valid_protos).mean(dim=0)
        mean_proto = F.normalize(mean_proto, dim=0)
        return mean_proto

    def _batch_log_safe_det(self, batch_mat, eps=1e-6):
        """
        支持 Batch 的 Log Determinant 计算
        batch_mat: (N, 3, 3)
        """
        jitter = eps * torch.eye(batch_mat.size(1), device=batch_mat.device).unsqueeze(0)
        batch_mat = batch_mat + jitter
        _, logabsdet = torch.linalg.slogdet(batch_mat)
        return logabsdet

    # ---------------------------------------------------------------------
    # Core Methods
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def update_banks(self, batch_info: dict):
        """
        batch_info 结构预期:
        {
            'sar': {
                8: (feats_tensor, labels_tensor),
                16: (feats_tensor, labels_tensor),
                ...
            },
            'rgb': { ... }
        }
        """
        device = next(self.parameters()).device
        
        # 1. Update Text Bank (Global, 无 stride 概念，或者默认为 -1/通用)
        # Text Bank 通常不需要多尺度，因为它是一个高层的语义嵌入
        if hasattr(self, 'text_feats_global') and 'text' in self.banks:
            raw_text = self.text_feats_global.to(device)
            # Text Bank 我们只用 stride=0 或者特定标记来存储，这里假设 TextBank 还是旧结构或者只用 s0
            # 为了兼容上面的 PrototypeBank 修改，我们假设 Text Bank 也初始化了 strides，我们只用 s32 (高层语义)
            # 或者给 TextBank 单独写个逻辑，这里假设 TextBank 依然是旧的逻辑或者只更新 s32
            
            # 为了简单，Text Bank 可以不改，或者 Text Bank 把所有 stride 都填满一样的值
            pass # (Text 更新逻辑保持不变，或者只更新 Bank 中的一个默认 stride)

        # 2. Update Visual Banks (核心修改)
        for m in self.modalities:
            if m == 'text': continue
            
            # 获取该模态下的多尺度数据
            stride_data = batch_info.get(m, {}) # {8: (feats, labels), 16: ...}
            if not stride_data: continue

            if m not in self.banks: continue
            bank_module = self.banks[m]

            for stride, (feats, labels) in stride_data.items():
                if feats is None or labels is None or feats.numel() == 0: continue
                
                feats = feats.to(device)
                labels = labels.to(device)
                
                # 过滤非法 label
                valid_mask = (labels >= 0) & (labels < bank_module.num_classes)
                if valid_mask.any():
                    # 调用 Bank 的新 update 接口，传入 stride
                    bank_module.update(labels[valid_mask], feats[valid_mask], stride=stride)
    
    
