'''
对比v7修改了align loss中的计算方式，原来是不同模态的ema之间对比，现在是sar模态batch内样本和rgb、ifr、text的prototype做对比
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
        self.register_buffer('protos', torch.zeros(num_classes, dim))
        self.register_buffer('valid', torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, class_ids: torch.LongTensor, protos_batch: torch.Tensor):
        for i, c in enumerate(class_ids.tolist()):
            if c >= self.num_classes or c < 0: continue 
            if self.valid[c]:
                self.protos[c] = (1.0 - self.beta) * self.protos[c] + self.beta * protos_batch[i]
            else:
                self.protos[c] = protos_batch[i]
                self.valid[c] = True

class TextPrototypeAlignerV8(nn.Module):
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
        
        if self.align_dim != self.clip_dim:
            self.align_dim = self.clip_dim

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
            self.banks['text'] = PrototypeBank(num_classes=n_concepts, dim=self.align_dim, beta=ema_beta)

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
    def update_banks(self, batch_protos: dict, batch_labels: dict):
        device = next(self.parameters()).device
        
        # 1. Update Text Bank (使用 global concepts)
        if hasattr(self, 'text_feats_global') and 'text' in self.banks:
            raw_text = self.text_feats_global.to(device)
            # 这里的 text bank 存储的是 concept_list 的特征
            update_len = min(self.banks['text'].protos.size(0), raw_text.size(0))
            self.banks['text'].protos.data[:update_len].copy_(raw_text[:update_len])
            self.banks['text'].valid.data[:update_len].fill_(True)

        # 2. Update Visual Banks (使用 Local ID)
        for m in self.modalities:
            if m == 'text': continue
            feats = batch_protos.get(m, None)
            labels = batch_labels.get(m, None)
            if feats is None or labels is None or feats.numel() == 0: continue
            
            feats, labels = feats.to(device), labels.to(device)
            valid_mask = (labels >= 0) & (labels < self.banks[m].num_classes)
            if valid_mask.any():
                self.banks[m].update(labels[valid_mask], feats[valid_mask])

    # @torch.no_grad()
    # def update_banks(self, batch_protos: dict, batch_labels: dict):
    #     device = next(self.parameters()).device
    #     print(f"\n🔄 [Bank Update] Starting update_banks...")

    #     # 1. Update Text Bank
    #     if hasattr(self, 'text_feats_global') and 'text' in self.banks:
    #         # 简单打印 Text 状态
    #         raw_text = self.text_feats_global.to(device)
    #         update_len = min(self.banks['text'].protos.size(0), raw_text.size(0))
    #         self.banks['text'].protos.data[:update_len].copy_(raw_text[:update_len])
    #         self.banks['text'].valid.data[:update_len].fill_(True)
    #         # print(f"   ✅ [Text] Global text bank initialized ({update_len} concepts).")

    #     # 2. Update Visual Banks (核心排查区)
    #     for m in self.modalities:
    #         if m == 'text': continue
            
    #         # --- Check A: 输入是否存在 ---
    #         feats = batch_protos.get(m, None)
    #         labels = batch_labels.get(m, None)
            
    #         if feats is None:
    #             print(f"   ⚠️ [{m}] Skipped: No features in batch_protos.")
    #             continue
    #         if labels is None:
    #             print(f"   ⚠️ [{m}] Skipped: No labels in batch_labels.")
    #             continue
    #         if feats.numel() == 0:
    #             print(f"   ⚠️ [{m}] Skipped: Features tensor is empty.")
    #             continue
            
    #         feats, labels = feats.to(device), labels.to(device)
            
    #         # --- Check B: 标签是否合法 ---
    #         # 检查 label 是否在 [0, num_classes) 范围内
    #         valid_mask = (labels >= 0) & (labels < self.banks[m].num_classes)
    #         valid_count = valid_mask.sum().item()
            
    #         if valid_count > 0:
    #             # 真正执行更新
    #             updating_ids = labels[valid_mask].unique().tolist()
    #             print(f"   ✅ [{m}] Updating {valid_count} samples. Class IDs: {updating_ids}")
                
    #             self.banks[m].update(labels[valid_mask], feats[valid_mask])
    #         else:
    #             # 如果有数据但全部不合法 (例如 label全是 -1)
    #             print(f"   ❌ [{m}] Has inputs but NO valid labels! (Input labels: {labels.tolist()})")

    #         # --- Check C: 更新后 Bank 的健康状态 ---
    #         # 打印目前 Bank 里总共有多少个位置是 Valid 的
    #         total_valid = self.banks[m].valid.sum().item()
    #         total_slots = self.banks[m].num_classes
    #         print(f"   📊 [{m} Bank Status] Valid Slots: {total_valid} / {total_slots}")

    #     print("--------------------------------------------------")

    
    def compute_align_loss(self, batch_features: dict, batch_labels: dict):
        device = next(self.parameters()).device
        loss_vol = torch.tensor(0.0, device=device)
        loss_ot = torch.tensor(0.0, device=device)
        
        sar_feats = batch_features.get('sar')
        sar_lbls = batch_labels.get('sar')
        
        if sar_feats is None:
            # print("❌ [Debug] No SAR Features found!")
            return {'loss_align_vol': loss_vol, 'loss_align_ot': loss_ot}

        collected_groups = [] 
        
        # 遍历 Batch
        for i in range(sar_feats.size(0)):
            curr_feat = sar_feats[i] # [512]
            curr_lbl = sar_lbls[i].item()
            
            # A. Check Concept Mapping
            concept_name = self.sar_id_to_concept.get(curr_lbl)
            if concept_name is None: continue 

            # B. Check Text Bank
            text_proto = None
            if 'text' in self.banks:
                tid = self.concept_to_id.get(concept_name, -1)
                if tid != -1 and self.banks['text'].valid[tid]:
                    text_proto = self.banks['text'].protos[tid]
            
            if text_proto is None: continue

            # C. Check Visual Bank (聚合 RGB/IFR)
            rgb_proto = self._get_aggregated_proto(concept_name, 'rgb')
            ifr_proto = self._get_aggregated_proto(concept_name, 'ifr')
            
            final_vis_proto = None
            if rgb_proto is not None and ifr_proto is not None:
                final_vis_proto = (rgb_proto + ifr_proto) / 2.0
                final_vis_proto = F.normalize(final_vis_proto, dim=0)
            elif rgb_proto is not None:
                final_vis_proto = rgb_proto
            elif ifr_proto is not None:
                final_vis_proto = ifr_proto
            
            if final_vis_proto is None: continue

            # Success: 构建三元组
            # 🔥 [关键修复] 在这里直接归一化 SAR 特征！
            # 这样后面就不需要对 M 进行 inplace 修改了
            curr_feat_norm = F.normalize(curr_feat, dim=0) 
            
            group = torch.stack([curr_feat_norm, final_vis_proto, text_proto])
            collected_groups.append(group)

        # 批量计算 Loss
        if len(collected_groups) > 0:
            M = torch.stack(collected_groups, dim=0) # [Batch_Valid, 3, 512]
            
            # ❌ [删除这行报错代码] M[:, 0, :] = F.normalize(...) 
            # 因为 curr_feat_norm 已经是归一化的了，M 也就自动归一化了
            
            # 计算 Gram Matrix
            G = torch.bmm(M, M.transpose(1, 2))
            
            # 计算 Volume Loss
            log_vol = self._batch_log_safe_det(G, eps=self.eps)
            loss_vol = self.vol_weight * torch.exp(log_vol).mean()
            
            # 计算 OT Loss (Cosine)
            # M[:, 0] 是 SAR (已归一化), M[:, 2] 是 Text (已归一化)
            cos_sim = (M[:, 0] * M[:, 2]).sum(dim=-1)
            loss_ot = (1.0 - cos_sim).mean() * 0.1

        return {
            'loss_align_vol': loss_vol,
            'loss_align_ot': loss_ot
        }

    @torch.no_grad()
    def extract_teacher_features(self, img_patches):
        device = next(self.parameters()).device
        if img_patches.device != device:
            img_patches = img_patches.to(device)
        if next(self.clip_visual.parameters()).device != device:
            self.clip_visual = self.clip_visual.to(device)

        teacher_features = self.clip_visual(img_patches)
        teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)
        return teacher_features

    @staticmethod
    def _log_safe_det(mat, eps=1e-6):
        try:
            _, logabsdet = torch.linalg.slogdet(mat + eps * torch.eye(mat.size(0), device=mat.device))
        except:
            return torch.tensor(0.0, device=mat.device)
        return logabsdet

    def _class_volume_log(self, M):
        # 保留旧接口以防兼容性问题，但主要逻辑已移至 compute_align_loss
        G = M @ M.t()
        return 0.5 * self._log_safe_det(G, eps=self.eps)

    @autocast(enabled=False)
    def sinkhorn_loss(self, x, y, epsilon=0.1, niter=3):
        # 保留旧接口
        x, y = x.float(), y.float()
        if torch.isnan(x).any() or torch.isnan(y).any(): return x.sum() * 0.0
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        C = torch.clamp(1 - torch.mm(x_norm, y_norm.t()), min=0.0, max=2.0)
        n, m = x.shape[0], y.shape[0]
        mu = torch.ones(n, device=x.device) / n
        nu = torch.ones(m, device=x.device) / m
        K = torch.exp(-C / epsilon)
        u = torch.ones(n, device=x.device)
        for _ in range(niter):
            v = nu / (torch.matmul(K.t(), u) + 1e-8)
            u = mu / (torch.matmul(K, v) + 1e-8)
        return torch.sum(u.detach() * torch.matmul(K * C, v.detach()))
