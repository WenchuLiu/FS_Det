# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Base_Categories = [
    "ship","car","tank","bridge","harbor",
    "small-vehicle","large-vehicle","plane","Ship","Harbor",
    "tennis-court","soccer-ball-field","ground-track-field",
    "baseball-diamond","swimming-pool","roundabout","basketball-court",
    "storage-tank","Bridge","helicopter","CAR",
    "BUS","FERIGHT_CAR","TRUCK","VAN",
    # "aircraft"
]
All_Categories = [
    "ship","car","tank","bridge","harbor",
    "small-vehicle","large-vehicle","plane","Ship","Harbor",
    "tennis-court","soccer-ball-field","ground-track-field",
    "baseball-diamond","swimming-pool","roundabout","basketball-court",
    "storage-tank","Bridge","helicopter",
    "CAR","BUS","FERIGHT_CAR","TRUCK","VAN",
    "aircraft"
]

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

class PrototypeAlignerV2(nn.Module):
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
                    "ship": {"sar": 0, "rgb": 8},
                    "car": {"sar": 1, "rgb": [5, 6], "ifr": [20, 21, 22, 23, 24]},
                    "tank": {"sar": 2, "rgb": 17},
                    "bridge": {"sar": 3, "rgb": 18},
                    "harbor": {"sar": 4, "rgb": 9},
                }
        elif semantic_mapping is None and not is_base_training:
            semantic_mapping = {
                    "ship": {"sar": 0, "rgb": 8},
                    "car": {"sar": 1, "rgb": [5, 6], "ifr": [20, 21, 22, 23, 24]},
                    "tank": {"sar": 2, "rgb": 17},
                    "bridge": {"sar": 3, "rgb": 18},
                    "harbor": {"sar": 4, "rgb": 9},
                    "aircraft":{"sar": 25, "rgb": 7}
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
            if isinstance(self.proj['text'], nn.Identity): 
                pass
            else:
                for p in self.proj['text'].parameters():
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
        
        
class TextPrototypeAlignerV2(PrototypeAlignerV2):
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
                 eps=1e-6,
                 init_clip = True,
                 is_base_training=True,
                 clip_model_name="lcybuaa/Git-RSCLIP"):
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
                    "ship": {"sar": 0, "rgb": 8},
                    "car": {"sar": 1, "rgb": [5, 6], "ifr": [20, 21, 22, 23, 24]},
                    "tank": {"sar": 2, "rgb": 17},
                    "bridge": {"sar": 3, "rgb": 18},
                    "harbor": {"sar": 4, "rgb": 9},
                }
        elif semantic_mapping is None and not is_base_training:
            semantic_mapping = {
                    "ship": {"sar": 0, "rgb": 8},
                    "car": {"sar": 1, "rgb": [5, 6], "ifr": [20, 21, 22, 23, 24]},
                    "tank": {"sar": 2, "rgb": 17},
                    "bridge": {"sar": 3, "rgb": 18},
                    "harbor": {"sar": 4, "rgb": 9},
                    "aircraft":{"sar": 25, "rgb": 7}
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
        
        # ✅ 新增：从 JSON 加载融合 prompt
        prompt_file = "/liuwenchu/code/detection/category_descriptions.json"  # ← 改成你的实际路径！

        if is_base_training:
            category_list = Base_Categories
        else:
            category_list = All_Categories

        # 获取每个类别的多个 prompt（List[List[str]]）
        class_prompt_lists = build_class_prompts_from_json(
            category_list=category_list,
            prompt_file=prompt_file,
            use_templates=True,
            use_llm=True
        )
        
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
        # for m in self.modalities:
        #     feats = batch_protos.get(m, None)
        #     if feats is not None:
        #         print(f"{m} feats shape: {feats.shape}")

        # 更新 EMA bank
        for m in self.modalities:
            if m == 'text':
                break
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
            # return {'loss_align': zero, 'loss_align_logvol': zero}
            return {'loss_align': zero, }

        avg_logvol = total_logvol / cnt
        loss = -self.vol_weight * avg_logvol
        return {
            'loss_align': loss,
            # 'loss_align_logvol': avg_logvol.detach()
        }