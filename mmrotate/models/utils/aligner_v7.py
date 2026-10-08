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

class TextPrototypeAlignerV7(nn.Module):
    def __init__(self,
                 modal_num_classes: dict, 
                 semantic_mapping: dict,  # {"ship": {"sar": 0, "rgb": 3}, ...}
                 
                 # 🔥 [修改] 接收具体的类别列表
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
        
        # 1. 保存类别列表
        self.category_lists = {
            'sar': sar_category if sar_category else [],
            'rgb': rgb_category if rgb_category else [],
            'ifr': ifr_category if ifr_category else []
        }
        
        # 2. 从 Semantic Mapping 中提取核心概念 (用于 Text Bank 和 Loss)
        # concept_list 不再由外部传入，而是由 mapping 决定
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
        # [Part B] Generate Text Features (🔥 核心修改)
        # =====================================================================
        
        # 加载 Prompts
        prompts_dict = {}
        if use_llm_prompt and prompt_file and os.path.exists(prompt_file):
            print(f"[Aligner] Loading LLM prompts from {prompt_file}")
            with open(prompt_file, 'r') as f:
                prompts_dict = json.load(f)

        # 辅助函数：生成 Embedding
        def get_embedding(text_list):
            if not text_list:
                return torch.zeros(0, self.clip_dim)
            embeddings = []
            with torch.no_grad():
                for cat_name in text_list:
                    # 优先查表，否则默认
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

        # 1. 生成各模态特定的 text_feats_{m} (基于各自的 category_list)
        visual_modals = [m for m in self.modalities if m != 'text']
        
        for m in visual_modals:
            cat_list = self.category_lists.get(m, [])
            num_cls = self.modal_num_classes.get(m, 0)
            
            if len(cat_list) == 0:
                print(f"[Aligner] Warning: Category list for {m} is empty, filling with zeros.")
                self.register_buffer(f'text_feats_{m}', torch.zeros(num_cls, self.clip_dim))
                continue
            
            print(f"[Aligner] Generating text feats for {m} based on {len(cat_list)} names...")
            modal_emb = get_embedding(cat_list)
            
            # 安全填充：处理 cat_list 长度与 num_cls 不一致的情况
            final_tensor = torch.zeros(num_cls, self.clip_dim)
            copy_len = min(num_cls, len(modal_emb))
            final_tensor[:copy_len] = modal_emb[:copy_len]
            
            self.register_buffer(f'text_feats_{m}', final_tensor)

        # 2. 生成 Global Text Bank (基于 Semantic Mapping 的 Keys)
        # 这用于对齐 Loss，代表了 "Car", "Ship" 等抽象概念的特征
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
        
        # Text Bank 大小 = 概念数量
        if 'text' in self.modalities:
            n_concepts = len(self.concept_list)
            self.banks['text'] = PrototypeBank(num_classes=n_concepts, dim=self.align_dim, beta=ema_beta)

        # Visual Banks 大小 = 各模态类别数
        for m in visual_modals:
            n_cls = self.modal_num_classes.get(m, 0)
            if n_cls > 0:
                self.banks[m] = PrototypeBank(num_classes=n_cls, dim=self.align_dim, beta=ema_beta)

        # =====================================================================
        # [Part D] Local ID to Concept Name (用于 Log)
        # =====================================================================
        self.local_id_to_concept = {m: {} for m in visual_modals}
        for m in visual_modals:
            cat_list = self.category_lists.get(m, [])
            for idx, name in enumerate(cat_list):
                self.local_id_to_concept[m][idx] = name

    # ---------------------------------------------------------------------
    # Core Methods
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def update_banks(self, batch_protos: dict, batch_labels: dict):
        device = next(self.parameters()).device
        print(f"\n🔄 [Bank Update] Starting update_banks...")

        # 1. Update Text Bank
        if hasattr(self, 'text_feats_global') and 'text' in self.banks:
            # 简单打印 Text 状态
            raw_text = self.text_feats_global.to(device)
            update_len = min(self.banks['text'].protos.size(0), raw_text.size(0))
            self.banks['text'].protos.data[:update_len].copy_(raw_text[:update_len])
            self.banks['text'].valid.data[:update_len].fill_(True)
            # print(f"   ✅ [Text] Global text bank initialized ({update_len} concepts).")

        # 2. Update Visual Banks (核心排查区)
        for m in self.modalities:
            if m == 'text': continue
            
            # --- Check A: 输入是否存在 ---
            feats = batch_protos.get(m, None)
            labels = batch_labels.get(m, None)
            
            if feats is None:
                print(f"   ⚠️ [{m}] Skipped: No features in batch_protos.")
                continue
            if labels is None:
                print(f"   ⚠️ [{m}] Skipped: No labels in batch_labels.")
                continue
            if feats.numel() == 0:
                print(f"   ⚠️ [{m}] Skipped: Features tensor is empty.")
                continue
            
            feats, labels = feats.to(device), labels.to(device)
            
            # --- Check B: 标签是否合法 ---
            # 检查 label 是否在 [0, num_classes) 范围内
            valid_mask = (labels >= 0) & (labels < self.banks[m].num_classes)
            valid_count = valid_mask.sum().item()
            
            if valid_count > 0:
                # 真正执行更新
                updating_ids = labels[valid_mask].unique().tolist()
                print(f"   ✅ [{m}] Updating {valid_count} samples. Class IDs: {updating_ids}")
                
                self.banks[m].update(labels[valid_mask], feats[valid_mask])
            else:
                # 如果有数据但全部不合法 (例如 label全是 -1)
                print(f"   ❌ [{m}] Has inputs but NO valid labels! (Input labels: {labels.tolist()})")

            # --- Check C: 更新后 Bank 的健康状态 ---
            # 打印目前 Bank 里总共有多少个位置是 Valid 的
            total_valid = self.banks[m].valid.sum().item()
            total_slots = self.banks[m].num_classes
            print(f"   📊 [{m} Bank Status] Valid Slots: {total_valid} / {total_slots}")

        print("--------------------------------------------------")

    # @torch.no_grad()
    # def update_banks(self, batch_protos: dict, batch_labels: dict):
    #     device = next(self.parameters()).device
        
    #     # 1. Update Text Bank (使用 global concepts)
    #     if hasattr(self, 'text_feats_global') and 'text' in self.banks:
    #         raw_text = self.text_feats_global.to(device)
    #         # 这里的 text bank 存储的是 concept_list 的特征
    #         update_len = min(self.banks['text'].protos.size(0), raw_text.size(0))
    #         self.banks['text'].protos.data[:update_len].copy_(raw_text[:update_len])
    #         self.banks['text'].valid.data[:update_len].fill_(True)

    #     # 2. Update Visual Banks (使用 Local ID)
    #     for m in self.modalities:
    #         if m == 'text': continue
    #         feats = batch_protos.get(m, None)
    #         labels = batch_labels.get(m, None)
    #         if feats is None or labels is None or feats.numel() == 0: continue
            
    #         feats, labels = feats.to(device), labels.to(device)
    #         valid_mask = (labels >= 0) & (labels < self.banks[m].num_classes)
    #         if valid_mask.any():
    #             self.banks[m].update(labels[valid_mask], feats[valid_mask])

    def compute_align_loss(self, batch_protos: dict):
        """
        🔥 基于 semantic_mapping 计算对齐 Loss
        不依赖 concept_list 参数，而是直接从 semantic_mapping 的 Key 出发
        """
        device = next(self.parameters()).device
        
        total_vol = 0.0
        vol_cnt = 0
        loss_ot = torch.tensor(0.0, device=device)
        
        # 遍历映射表: concept_name -> {modality: [local_ids]}
        # 例如: "car" -> {"sar": 2, "rgb": [0,1]}
        for concept_name, mapping in self.semantic_mapping.items():
            
            # 1. 获取该概念对应的 Text Prototype
            # 我们去 concept_to_id 查表，看这个概念在 Text Bank 的第几行
            text_id = self.concept_to_id.get(concept_name, -1)
            text_proto = None
            
            if text_id != -1 and 'text' in self.banks and self.banks['text'].valid[text_id]:
                text_proto = self.banks['text'].protos[text_id]

            # 2. 获取该概念对应的 Visual Prototypes (聚合所有模态)
            rows = [] 
            vis_protos_for_ot = []
            
            for m, local_ids in mapping.items():
                if m not in self.banks: continue
                # 兼容 int 和 list
                ids_list = local_ids if isinstance(local_ids, (list, tuple)) else [local_ids]
                
                valid_local_protos = []
                for lid in ids_list:
                    # 检查 ID 是否越界且 Bank 中是否已存有有效特征
                    if lid < self.banks[m].num_classes and self.banks[m].valid[lid]:
                        valid_local_protos.append(self.banks[m].protos[lid])
                
                if len(valid_local_protos) > 0:
                    # 如果一个模态有多个对应 ID (如RGB的小车+大车)，取平均作为该模态对 "car" 的表示
                    avg_proto = torch.stack(valid_local_protos).mean(dim=0)
                    avg_proto = F.normalize(avg_proto, dim=0)
                    rows.append(avg_proto) # 加入体积计算
                    vis_protos_for_ot.append({'modality': m, 'proto': avg_proto}) # 加入 OT 计算

            # 3. Volume Loss (多模态一致性)
            if text_proto is not None:
                rows.append(text_proto)
            
            # 至少有2个点才能算体积 (e.g. SAR+Text, or RGB+Text)
            if len(rows) >= 2:
                M = torch.stack(rows)
                logvol = self._class_volume_log(M)
                total_vol += torch.exp(logvol)
                vol_cnt += 1

            # 4. OT Loss (Sinkhorn 距离)
            if text_proto is not None and len(vis_protos_for_ot) > 0:
                for item in vis_protos_for_ot:
                    m = item['modality']
                    v_p = item['proto']
                    # 计算视觉原型与文本原型的距离
                    dist = self.sinkhorn_loss(v_p.unsqueeze(0), text_proto.unsqueeze(0))
                    w = 1.0 if m == 'sar' else 0.5
                    loss_ot += w * dist

        avg_vol = total_vol / vol_cnt if vol_cnt > 0 else torch.tensor(1.0, device=device)
        loss_vol = self.vol_weight * avg_vol

        return {
            'loss_align_vol': loss_vol,
            'loss_align_ot': 0.1 * loss_ot
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
        G = M @ M.t()
        return 0.5 * self._log_safe_det(G, eps=self.eps)

    @autocast(enabled=False)
    def sinkhorn_loss(self, x, y, epsilon=0.1, niter=3):
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
