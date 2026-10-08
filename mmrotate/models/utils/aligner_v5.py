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

class PrototypeAlignerV5(nn.Module):
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
        
        
class TextPrototypeAlignerV5(PrototypeAlignerV5):
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
        
    # 引入 amp 用于控制混合精度
    from torch.cuda.amp import autocast

    @autocast(enabled=False) # 🔥 强制此函数内部不使用 FP16
    def sinkhorn_loss(self, x, y, epsilon=0.1, niter=3):
        """
        数值稳定的 Sinkhorn 距离计算 (强制 FP32)
        x: [N, D] 视觉原型
        y: [M, D] 文本原型
        """
        # 1. 强制转换为 FP32 (float)
        # 即使输入是 FP16，这里也转回 FP32 进行计算，避免 exp/div 溢出
        x = x.float()
        y = y.float()

        # 0. 基础 NaN 检查
        if torch.isnan(x).any() or torch.isnan(y).any():
            # 为了防止训练崩溃，返回带梯度的 0
            return (x.sum() + y.sum()) * 0.0

        n = x.shape[0]
        m = y.shape[0]

        # 2. 计算 Cost Matrix (1 - Cosine)
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        
        # 这里的 mm 可能会产生略微大于 1 或小于 -1 的值 (FP 误差)，导致 clamp 前出问题
        cos_sim = torch.mm(x_norm, y_norm.t())
        C = 1 - cos_sim
        
        # 🔥 [关键] 限制 Cost 范围，防止 exp(-C/eps) 下溢
        # Cosine distance 理论最大是 2，但在 embedding 空间通常较小
        C = torch.clamp(C, min=0.0, max=2.0)

        # 3. 初始化对偶向量
        mu = torch.ones(n, device=x.device) / n
        nu = torch.ones(m, device=x.device) / m
        
        # Kernel matrix
        # 如果 epsilon 太小 (如 0.01)，在 FP32 下也会炸。建议 >= 0.1
        K = torch.exp(-C / epsilon)
        
        u = torch.ones(n, device=x.device)
        
        # 4. Sinkhorn 迭代 (带数值保护)
        for _ in range(niter):
            # v = nu / (K.T @ u)
            Kv_t_u = torch.matmul(K.t(), u)
            # 添加 1e-8 防止除零
            v = nu / (Kv_t_u + 1e-8) 
            
            # u = mu / (K @ v)
            Kv_v = torch.matmul(K, v)
            u = mu / (Kv_v + 1e-8)
            
        # 5. 计算最终 Cost (停止梯度技巧)
        u = u.detach() 
        v = v.detach()
        
        # dist = sum(P * C)
        dist = torch.sum(u * torch.matmul(K * C, v))
        
        return dist


    def update_and_compute_loss(self, batch_protos: dict, batch_labels: dict):
        device = None
        
        # [第一步] 确定设备
        for v in batch_protos.values():
            if v is not None and isinstance(v, torch.Tensor):
                device = v.device
                break
        
        if device is None:
            if hasattr(self, 'text_feats'):
                device = self.text_feats.device
            else:
                device = next(self.parameters()).device

        # ========================================================
        # [第二步] 准备 Text Feature (关键修复)
        # ========================================================
        # 预定义 text_feats_curr，用于后面的 Loss 计算 (必须保持梯度连通性)
        text_feats_curr = None
        
        if hasattr(self, 'text_feats'):
            # 1. 获取原始 CLIP 特征
            raw_text = self.text_feats.to(device)
            
            # 2. 通过 Adapter (投影层)
            if 'text' in self.proj and not isinstance(self.proj['text'], nn.Identity):
                if self.is_base_training:
                    # 🔥 [修复1] Base Training 时必须保留梯度！不要用 no_grad
                    text_feats_curr = self.proj['text'](raw_text)
                else:
                    # Few-shot 微调时，如果锁住了 adapter，则不需要梯度
                    with torch.no_grad():
                        text_feats_curr = self.proj['text'](raw_text)
            else:
                text_feats_curr = raw_text

            # 3. 更新 Bank (Bank 只存数值，不存梯度，所以要 detach)
            if 'text' in self.banks:
                self.banks['text'].protos.data.copy_(text_feats_curr.detach())
                self.banks['text'].valid.data.fill_(True)

        # 🔥 [修复2] 建立梯度挂钩 (Gradient Hook)
        # 这是一个数值为0的张量，但它连接了 Adapter 的权重。
        # 即使 Rank 2 后面啥 Loss 也没算，只要返回了这个值，DDP 就认为 Adapter 参与了计算。
        if text_feats_curr is not None and text_feats_curr.requires_grad:
            dummy_hook = text_feats_curr.sum() * 0.0
        else:
            dummy_hook = torch.tensor(0.0, device=device)

        # ========================================================
        # [第三步] 更新 Visual Banks (保持不变)
        # ========================================================
        for m in self.modalities:
            if m == 'text': continue
            
            feats = batch_protos.get(m, None)
            labels = batch_labels.get(m, None)
            
            if feats is None or labels is None or feats.numel() == 0:
                continue
            
            feats = feats.to(device)
            labels = labels.to(device)
            proj = self.proj[m](feats)
            self.banks[m].update(labels, proj.detach())

        # ========================================================
        # [第四步] 计算 Grammian Volume Loss (保持不变)
        # ========================================================
        total_logvol = 0.0
        cnt = 0
        if self.semantic_mapping:
            for sem_name, mapping in self.semantic_mapping.items():
                rows = []
                for m, idx in mapping.items():
                    # 注意：Bank 里拿出来的是 detach 的，没有梯度
                    # 如果你想让 Visual Adapter 也通过 Grammian Loss 更新，
                    # 这里的逻辑其实是不够的（目前只更新了 Bank EMA）。
                    # 但为了不破坏你原有逻辑，这里暂不改动 Visual 部分。
                    if isinstance(idx, list):
                        valid_protos = [self.banks[m].protos[i] for i in idx if self.banks[m].valid[i]]
                        if len(valid_protos) == 0: continue
                        proto_mean = torch.stack(valid_protos, dim=0).mean(dim=0)
                        rows.append(proto_mean)
                    else:
                        if self.banks[m].valid[idx]:
                            rows.append(self.banks[m].protos[idx])

                if len(rows) < 2: continue
                M = torch.stack(rows, dim=0)
                Mn = F.normalize(M, dim=-1)
                logvol = self._class_volume_log(Mn)
                total_logvol += logvol
                cnt += 1

        loss_align_vol = torch.tensor(0.0, device=device)
        if cnt > 0:
            avg_logvol = total_logvol / cnt
            loss_align_vol = -self.vol_weight * avg_logvol

        # ========================================================
        # [第五步] 计算 Sinkhorn OT Loss (关键修复)
        # ========================================================
        loss_ot = torch.tensor(0.0, device=device)
        
        # 确保我们有带梯度的 Text 特征
        if text_feats_curr is not None:
            
            # 使用 Bank 的 valid 状态来判断哪些类是有效的
            # (注意：Bank 本身的值没梯度，但 Valid 标记可以用)
            text_valid_mask = self.banks['text'].valid
            
            if text_valid_mask.any():
                target_modalities = [m for m in self.modalities if m != 'text']
                
                for m in target_modalities:
                    # 检查 Visual Bank 是否有数据
                    if m in self.banks and self.banks[m].valid.any():
                        vis_bank = self.banks[m]
                        
                        # 1. 找出共现类别 (即 Visual 和 Text 都存在的类别)
                        common_valid_mask = vis_bank.valid & text_valid_mask
                        
                        if common_valid_mask.sum() > 1:
                            # 2. 取 Visual 原型 (来自 Bank, Detached)
                            # 这里的目的是让 Visual 原型去靠近 Text 原型
                            vis_protos = vis_bank.protos[common_valid_mask]
                            
                            # 3. 🔥 [核心修复] 取 Text 原型 (来自 text_feats_curr)
                            # 必须从 text_feats_curr 中取值，因为它是 Adapter 的直接输出，有梯度！
                            # 绝对不能用 self.banks['text'].protos (那个是 detach 过的)
                            text_protos = text_feats_curr[common_valid_mask]
                            
                            # 4. 计算 OT 
                            ot_dist = self.sinkhorn_loss(vis_protos, text_protos, epsilon=0.1, niter=3)
                            
                            weight = 1.0 if m == 'sar' else 0.5 
                            loss_ot = loss_ot + weight * ot_dist

        # ========================================================
        # [第六步] 返回总 Loss (带钩子)
        # ========================================================
        
        # 将 dummy_hook 加到任意一个 loss 上
        # 这样 DDP 就能检测到 Adapter 的参数被“使用”了（虽然加的是 0）
        loss_ot = loss_ot + dummy_hook

        return {
            'loss_align_vol': loss_align_vol,
            'loss_align_ot': 0.1 * loss_ot
        }