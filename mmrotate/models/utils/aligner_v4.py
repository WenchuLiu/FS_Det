# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import List, Dict, Union

def build_class_prompts_from_json(
    category_list: List[str],
    prompt_file: str = "prompts.json",
    use_templates: bool = True,
    use_llm: bool = True,
    combine_method: str = "mean"  # or "concat"
) -> List[str]:
    """
    为每个类别构建融合后的文本描述列表（每个类别一个字符串 or 多个字符串）。
    返回：List[List[str]] —— 每个类别对应多个 prompt 字符串
    """
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)

    all_class_prompts = []
    for cat in category_list:
        key = cat   # 直接用原始类名作为 key
        if key not in prompts:  
            all_class_prompts.append([f"a remote sensing image of {cat}"])
            continue
        prompt_pool = []
        if use_templates and "templates" in prompts[key]:
            prompt_pool.extend(prompts[key]["templates"])
        if use_llm and "llm" in prompts[key]:
            prompt_pool.extend(prompts[key]["llm"])

        if not prompt_pool:
            prompt_pool = [f"a remote sensing image of {cat}"]

        all_class_prompts.append(prompt_pool)
    return all_class_prompts


class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(in_dim // 2, out_dim)  # 自动设一个合适的中间维度
        self.down = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(hidden_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.up(self.act(self.down(x)))
        x = self.norm(x)
        return x


class PrototypeBank(nn.Module):
    """EMA prototype bank (registered buffers -> 随 model.to(device) 移动)."""
    def __init__(self, num_classes, dim, beta=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.beta = beta
        self.register_buffer('protos', torch.zeros(num_classes, dim))
        self.register_buffer('valid', torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, class_ids: torch.LongTensor, protos_batch: torch.Tensor):
        """
        class_ids: [K], protos_batch: [K, D] (per-class mean prototypes from this batch)
        """
        for i, c in enumerate(class_ids.tolist()):
            if self.valid[c]:
                self.protos[c] = (1.0 - self.beta) * self.protos[c] + self.beta * protos_batch[i]
            else:
                self.protos[c] = protos_batch[i]
                self.valid[c] = True

    def get(self, class_ids: torch.LongTensor):
        return self.protos[class_ids], self.valid[class_ids]

class PrototypeAlignerV4(nn.Module):
    """
    对齐器：对语义相同的类在多模态上的原型做 L2 + geometric-volume (log det(Gram)) 损失。
    使用方式:update_and_compute_loss(batch_protos, batch_labels) -> 返回 dict losses
    """
    def __init__(self,
                 num_classes=26,  # 🔥 直接指定总类别数
                 in_dims: dict = {
                        'sar': 256,
                        'rgb': 256,
                        'ifr': 256
                    },
                 semantic_mapping = None,
                 modalities=('sar', 'rgb', 'ifr'),
                 align_dim=256,
                 ema_beta=0.05,
                 vol_weight=0.5,
                 is_base_training=True,
                 eps=1e-6,):
        super().__init__()
        self.modalities = list(modalities)
        self.num_classes = num_classes
        self.align_dim = align_dim
        self.eps = eps
        self.vol_weight = vol_weight
        self.is_base_training = is_base_training
        # 设置默认的语义映射
        if semantic_mapping is None and is_base_training:
            semantic_mapping = {
                    "ship": {"sar": 0, "rgb": 9},
                    "car": {"sar": 2, "rgb": [6, 7], "ifr": [21, 22, 23, 24, 25]},
                    "tank": {"sar": 3, "rgb": 18},
                    "bridge": {"sar": 4, "rgb": 19},
                    "harbor": {"sar": 5, "rgb": 10},
                    "aircraft":{"sar": 1, "rgb": [8, 20]}
                }
        elif semantic_mapping is None and not is_base_training:
            semantic_mapping = {
                    "ship": {"sar": 0, "rgb": 9},
                    "car": {"sar": 2, "rgb": [6, 7], "ifr": [21, 22, 23, 24, 25]},
                    "tank": {"sar": 3, "rgb": 18},
                    "bridge": {"sar": 4, "rgb": 19},
                    "harbor": {"sar": 5, "rgb": 10},
                    "aircraft":{"sar": 1, "rgb": [8, 20]}
                }
        self.semantic_mapping = semantic_mapping
        # 投影层
        self.proj = nn.ModuleDict()
        for m in self.modalities:
            d = in_dims.get(m, align_dim)
            if d == align_dim:
                self.proj[m] = nn.Identity()
            else:
                self.proj[m] = Adapter(d, align_dim)

        # 微调时冻结 text 分支参数
        if not is_base_training:
            if 'text' in self.proj:
                text_proj = self.proj['text']
                if not isinstance(text_proj, nn.Identity):
                    for p in text_proj.parameters():
                        p.requires_grad = False

        # 🔥 每个模态 bank 全部设置成 num_classes=25
        self.banks = nn.ModuleDict({
            m: PrototypeBank(num_classes=num_classes, dim=align_dim, beta=ema_beta)
            for m in self.modalities
        })

    @staticmethod
    def _pool_classwise(feats: torch.Tensor, labels: torch.LongTensor):
        if feats.numel() == 0:
            return None, None
        unique = labels.unique()
        pro_list, cls_list = [], []
        for c in unique.tolist():
            mask = (labels == c)
            if mask.any():
                pro = feats[mask].mean(dim=0)
                pro_list.append(pro)
                cls_list.append(c)
        if len(pro_list) == 0:
            return None, None
        return torch.tensor(cls_list, dtype=torch.long, device=feats.device), torch.stack(pro_list, dim=0)

    @staticmethod
    def _log_safe_det(mat, eps=1e-6):
        try:
            sign, logabsdet = torch.linalg.slogdet(mat)
        except RuntimeError:
            mat = mat + eps * torch.eye(mat.size(0), device=mat.device)
            sign, logabsdet = torch.linalg.slogdet(mat)
        return logabsdet

    def _class_volume_log(self, M):
        K = M.size(0)
        G = M @ M.t()
        G = G + self.eps * torch.eye(K, device=G.device)
        logabsdet = self._log_safe_det(G, eps=self.eps)
        logvol = 0.5 * logabsdet
        return logvol

    def update_and_compute_loss(self, batch_protos: dict, batch_labels: dict):
        device = None
        # for m in self.modalities:
        #     feats = batch_protos.get(m, None)
        #     if feats is not None:
        #         print(f"{m} feats shape: {feats.shape}")

        # 更新 EMA bank
        for m in self.modalities:
            feats = batch_protos.get(m, None)
            labels = batch_labels.get(m, None)
            if feats is None or labels is None or feats.numel() == 0:
                continue
            if device is None:
                device = feats.device
            feats = feats.to(device)
            labels = labels.to(device)

            proj = self.proj[m](feats)
            self.banks[m].update(labels, proj.detach())

        # 计算跨模态对齐损失
        total_logvol = 0.0
        cnt = 0
        for sem_name, mapping in self.semantic_mapping.items():
            rows = []
            for m, idx in mapping.items():
                if isinstance(idx, list):
                    valid_protos = [self.banks[m].protos[i] for i in idx if self.banks[m].valid[i]]
                    if len(valid_protos) == 0:
                        continue
                    proto_mean = torch.stack(valid_protos, dim=0).mean(dim=0)
                    rows.append(proto_mean)
                else:
                    if self.banks[m].valid[idx]:
                        rows.append(self.banks[m].protos[idx])

            if len(rows) < 2:
                continue

            M = torch.stack(rows, dim=0)
            Mn = F.normalize(M, dim=-1)
            logvol = self._class_volume_log(Mn)

            total_logvol += logvol
            cnt += 1

        if cnt == 0:
            zero = torch.tensor(0.0, device=device)
            return {'loss_align': zero, 'loss_align_logvol': zero}

        avg_logvol = total_logvol / cnt
        loss = -self.vol_weight * avg_logvol
        return {
            'loss_align': loss,
            'loss_align_logvol': avg_logvol.detach()
        }
        
        
class TextPrototypeAlignerV4(PrototypeAlignerV4):
    """Text-specific prototype aligner that inherits from PrototypeAligner.
    Simplified version that only loads text embeddings at initialization.
    """
    def __init__(self,
                 num_classes=26,
                 in_dims={'sar': 256,'rgb': 256,
                          'ifr': 256,'text': 1024},
                 semantic_mapping=None,
                 modalities=('sar', 'rgb', 'ifr', 'text'),
                 align_dim=256,
                 ema_beta=0.05,
                 vol_weight=0.5,
                 is_base_training=True,
                 use_llm_prompt = True,
                 prompt_file=None,
                 category_list = None,
                 clip_model_name="lcybuaa/Git-RSCLIP",
                 eps=1e-6,):
        # 设置默认的输入维度
        if in_dims is None:
            in_dims = {
                'sar': 256,
                'rgb': 256,
                'ifr': 256,
                'text': 1024
            }
        
        # 设置默认的语义映射
        if semantic_mapping is None and is_base_training:
            semantic_mapping = {
                    "ship": {"sar": 0, "rgb": 9},
                    "aircraft":{"sar": 1, "rgb": [8, 20]},
                    "car": {"sar": 2, "rgb": [6, 7], "ifr": [21, 22, 23, 24, 25]},
                    "tank": {"sar": 3, "rgb": 18},
                    "bridge": {"sar": 4, "rgb": 19},
                    "harbor": {"sar": 5, "rgb": 10},
                }
        elif semantic_mapping is None and not is_base_training:
            semantic_mapping = {
                    "ship": {"sar": 0, "rgb": 9},
                    "aircraft":{"sar": 1, "rgb": [8, 20]},
                    "car": {"sar": 2, "rgb": [6, 7], "ifr": [21, 22, 23, 24, 25]},
                    "tank": {"sar": 3, "rgb": 18},
                    "bridge": {"sar": 4, "rgb": 19},
                    "harbor": {"sar": 5, "rgb": 10},
                }
        # 初始化父类
        super().__init__(num_classes=num_classes,
                         in_dims=in_dims,
                         semantic_mapping=semantic_mapping,
                         modalities=modalities,
                         align_dim=align_dim,
                         ema_beta=ema_beta,
                         vol_weight=vol_weight,
                         eps=eps)
        # 确保text在模态列表中
        if 'text' not in modalities:
            modalities = list(modalities) + ['text']
        
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        
        # -----------------------------------------------------------
        # [严谨的冻结逻辑]
        # -----------------------------------------------------------
        # 1. 打印确认 flag 是否真的是 False (排查 Config 写错的情况)
        print(f"[Aligner DEBUG] Freezing check: is_base_training = {is_base_training}")

        if not is_base_training:
            # 冻结 Text Projection
            if 'text' in self.proj:
                print("[Aligner] 🔒 Freezing Text Projection Adapter...")
                self.proj['text'].eval()  # 关掉 Dropout/BN
                for param in self.proj['text'].parameters():
                    param.requires_grad = False 
            
            # [关键修复] 冻结 Temperature
            # 之前的报错里明确提到了 prototype_aligner.temperature 没梯度
            # 因为它不属于 self.proj['text']，所以之前的循环没覆盖到它！
            if hasattr(self, 'temperature'):
                print("[Aligner] 🔒 Freezing Temperature Parameter...")
                self.temperature.requires_grad = False
        
        
        
        if use_llm_prompt:
            # 获取每个类别的多个 prompt（List[List[str]]）
            class_prompt_lists = build_class_prompts_from_json(
                category_list=category_list,
                prompt_file=prompt_file,
                use_templates=True,
                use_llm=True
            )
        else:
            # 仅使用简单模板（不读 JSON，不使用 LLM）
            class_prompt_lists = [[f"a remote sensing image of {cat}"] for cat in category_list]
        
        # 初始化CLIP模型并计算文本特征
        from transformers import AutoProcessor, AutoModel
        text_model = AutoModel.from_pretrained(clip_model_name)
        text_processor = AutoProcessor.from_pretrained(clip_model_name)
        text_model.eval()
        
        all_text_feats = []
        device = next(self.parameters()).device if next(self.parameters(), None) is not None else torch.device('cpu')

        with torch.no_grad():
            for prompt_list in class_prompt_lists:
                # Tokenize and encode all prompts for this class
                inputs = text_processor(
                    text=prompt_list,
                    padding="max_length",
                    return_tensors="pt"
                ).to(device)
                outputs = text_model.get_text_features(**inputs)  # [N, D]
                # Normalize each and take mean
                outputs = F.normalize(outputs, dim=-1)
                mean_feat = outputs.mean(dim=0)  # [D]
                mean_feat = F.normalize(mean_feat, dim=-1)  # 再归一化一次
                all_text_feats.append(mean_feat)
        
            self.text_feats = torch.stack(all_text_feats, dim=0)  # [num_classes, D]
            self.text_labels = torch.arange(self.num_classes).to(self.text_feats.device)
        
    # def update_and_compute_loss(self, batch_protos: dict, batch_labels: dict):
    #     """重写父类方法，特别处理文本模态"""
    #     device = None
    #     # 确定当前计算设备 (例如从输入的 batch_protos 中获取)
    #     for m in self.modalities:
    #         feats = batch_protos.get(m, None)
    #         if feats is not None and feats.numel() > 0:
    #             device = feats.device
    #             break
    #     if hasattr(self, 'text_feats'):
    #         self.text_feats = self.text_feats.to(device)
    #         self.text_labels = self.text_labels.to(device)
    #     # 更新EMA bank（跳过text模态）
    #     for m in self.modalities:
    #         # 跳过text模态，因为text特征是预计算的
    #         if m == 'text':
    #             text_protos=self.proj[m](self.text_feats)
    #             self.banks[m].update(self.text_labels, text_protos.detach())
    #             continue
                
    #         feats = batch_protos.get(m, None)
    #         labels = batch_labels.get(m, None)
    #         if feats is None or labels is None or feats.numel() == 0:
    #             continue
    #         if device is None:
    #             device = feats.device
    #         feats = feats.to(device)
    #         labels = labels.to(device)

    #         proj = self.proj[m](feats)
    #         self.banks[m].update(labels, proj.detach())
            
    #     # 确保device已设置
    #     if device is None:
    #         device = self.text_feats.device
            
    #     # 将text_prototype_feats移动到正确的设备
    #     if hasattr(self, 'text_feats'):
    #         self.text_feats = self.text_feats.to(device)

    #     # 计算跨模态对齐损失
    #     total_logvol = 0.0
    #     cnt = 0
    #     for sem_name, mapping in self.semantic_mapping.items():
    #         rows = []
    #         for m, idx in mapping.items():
    #             # 跳过不在模态列表中的模态
    #             if m not in self.modalities:
    #                 continue
    #             # 处理text模态
    #             if m == 'text' and hasattr(self, 'text_feats'):
    #                 idxs = idx if isinstance(idx, list) else [idx]
    #                 rows.extend(self.banks[m].protos[idxs].to(device))
    #                 continue
    #             # 处理其他模态
    #             if isinstance(idx, list):
    #                 valid_protos = [self.banks[m].protos[i] for i in idx if self.banks[m].valid[i]]
    #                 if len(valid_protos) == 0:
    #                     continue
    #                 proto_mean = torch.stack(valid_protos, dim=0).mean(dim=0)
    #                 rows.append(proto_mean)
    #             else:
    #                 if self.banks[m].valid[idx]:
    #                     rows.append(self.banks[m].protos[idx])
    #         if len(rows) < 2:
    #             continue
            
    #         M = F.normalize(torch.stack(rows), dim=-1)
    #         total_logvol += self._class_volume_log(M)
    #         cnt += 1

    #     if cnt == 0:
    #         zero = torch.tensor(0.0, device=device)
    #         return {'loss_align': zero,}

    #     avg_logvol = total_logvol / cnt
    #     loss = -self.vol_weight * avg_logvol
    #     return {
    #         'loss_align': loss,
    #     }
    
    def update_and_compute_loss(self, batch_protos: dict, batch_labels: dict):
        device = None
        
        # ========================================================
        # [第一步] 先确定当前的计算设备 (Device)
        # ========================================================
        # 遍历 batch_protos 里的任意一个非空 tensor 来获取 device
        for v in batch_protos.values():
            if v is not None and isinstance(v, torch.Tensor):
                device = v.device
                break
        
        # 如果 batch_protos 全空 (极端情况)，回退到模型自身的 device
        if device is None:
            if hasattr(self, 'text_feats'):
                device = self.text_feats.device
            else:
                # 如果连 text_feats 都没有，随便找个参数的 device
                device = next(self.parameters()).device

        # ========================================================
        # [第二步] 静态填充 Text Bank (不进行 EMA，只做 Copy)
        # ========================================================
        if 'text' in self.banks and hasattr(self, 'text_feats'):
            # 1. 确保在正确的设备上
            text_feats_curr = self.text_feats.to(device)
            
            # 2. (可选) 如果有投影层且不是 Identity，做一次投影
            if 'text' in self.proj and not isinstance(self.proj['text'], nn.Identity):
                # 既然是 Few-Shot 且你决定把 Text 当锚点，建议这里用 no_grad
                # 这样不会因为对齐操作反向传播去改变 MLP (除非你显式想微调它)
                with torch.no_grad():
                    text_feats_curr = self.proj['text'](text_feats_curr)
            
            # 3. [关键] 强制覆盖 Bank 的内容
            # 这一步保证了 Bank['text'] 里有值，且 Valid=True
            # 这样后面的 Loss 计算循环就能取到 Text 原型了
            self.banks['text'].protos.data.copy_(text_feats_curr)
            self.banks['text'].valid.data.fill_(True)

        # ========================================================
        # [第三步] 更新其他模态 (Visual) 的 EMA bank
        # ========================================================
        for m in self.modalities:
            # [关键] 遇到 text 直接跳过 (continue)，因为上面已经静态填充了
            if m == 'text':
                continue 
            
            feats = batch_protos.get(m, None)
            labels = batch_labels.get(m, None)
            
            # 检查数据是否有效
            if feats is None or labels is None or feats.numel() == 0:
                continue
            
            # 确保数据在正确的 device (虽然大部分情况已经是了)
            feats = feats.to(device)
            labels = labels.to(device)

            # 投影并更新 EMA
            proj = self.proj[m](feats)
            self.banks[m].update(labels, proj.detach())

        # ========================================================
        # [第四步] 计算跨模态对齐损失 (逻辑保持不变)
        # ========================================================
        total_logvol = 0.0
        cnt = 0
        for sem_name, mapping in self.semantic_mapping.items():
            rows = []
            for m, idx in mapping.items():
                if isinstance(idx, list):
                    valid_protos = [self.banks[m].protos[i] for i in idx if self.banks[m].valid[i]]
                    if len(valid_protos) == 0:
                        continue
                    proto_mean = torch.stack(valid_protos, dim=0).mean(dim=0)
                    rows.append(proto_mean)
                else:
                    if self.banks[m].valid[idx]:
                        rows.append(self.banks[m].protos[idx])

            # 只要凑齐 >= 2 个模态 (例如 SAR + Text)，就可以算 Loss
            if len(rows) < 2:
                continue

            M = torch.stack(rows, dim=0)
            Mn = F.normalize(M, dim=-1)
            logvol = self._class_volume_log(Mn)

            total_logvol += logvol
            cnt += 1

        if cnt == 0:
            zero = torch.tensor(0.0, device=device)
            return {'loss_align': zero}

        avg_logvol = total_logvol / cnt
        loss = -self.vol_weight * avg_logvol
        
        return {
            'loss_align': loss,
            # 'loss_align_logvol': avg_logvol.detach()
        }