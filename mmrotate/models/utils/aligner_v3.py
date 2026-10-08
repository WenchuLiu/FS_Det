# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from .. import sar_clip

import json
from typing import List, Dict, Union

try:
    from transformers import AutoModel, AutoProcessor
except ImportError:
    AutoModel = None
    AutoProcessor = None

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

class PrototypeAlignerV3(nn.Module):
    """
    对齐器：对语义相同的类使用非负 softplus(log-volume) 损失。
    使用方式:update_and_compute_loss(batch_protos, batch_labels) -> 返回 dict losses
    """
    def __init__(self,
                 num_classes=26,  # Total number of concepts
                 modal_num_classes=None,
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
        self.modal_num_classes = dict(modal_num_classes or {})
        self.align_dim = align_dim
        self.eps = eps
        self.vol_weight = float(vol_weight)
        if not np.isfinite(self.vol_weight) or self.vol_weight < 0:
            raise ValueError('vol_weight must be finite and nonnegative.')
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
            m: PrototypeBank(
                num_classes=self.modal_num_classes.get(m, num_classes),
                dim=align_dim, beta=ema_beta)
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
        # .float() alone does not stop autocast from computing Gram in FP16.
        # Keep the small diagonal regularizer and log-det arithmetic in FP32.
        with torch.cuda.amp.autocast(enabled=False):
            M = M.float()
            K = M.size(0)
            G = M @ M.t()
            G = G + self.eps * torch.eye(K, device=G.device)
            logabsdet = self._log_safe_det(G, eps=self.eps)
            return 0.5 * logabsdet

    def update_and_compute_loss(self, batch_protos: dict, batch_labels: dict):
        current = {}
        projected_batches = []
        for modality in self.modalities:
            if modality == 'text':
                continue
            feats = batch_protos.get(modality)
            labels = batch_labels.get(modality)
            if feats is None or labels is None or feats.numel() == 0:
                continue
            bank = self.banks[modality]
            labels = labels.to(feats.device).long()
            valid = (labels >= 0) & (labels < bank.num_classes)
            if not valid.any():
                continue
            projected = self.proj[modality](feats[valid]).float()
            projected_batches.append(projected)
            class_ids, means = self._pool_classwise(projected, labels[valid])
            # The first observation initializes a prototype directly. Later
            # observations enter the EMA with beta, including in backward.
            scales = torch.where(bank.valid[class_ids], bank.beta, 1.0)
            bank.update(class_ids, means.detach())
            current[modality] = dict(zip(class_ids.tolist(),
                zip(means.unbind(0), scales.unbind(0))))

        reference = next(iter(self.banks.values())).protos
        zero = sum((feats.sum() * 0 for feats in projected_batches), reference.sum() * 0)
        logvols = []
        for concept, mapping in self.semantic_mapping.items():
            rows = []
            for modality, indices in mapping.items():
                if modality not in self.banks:
                    continue
                bank = self.banks[modality]
                indices = indices if isinstance(indices, (list, tuple)) else [indices]
                prototypes = []
                for index in indices:
                    if not 0 <= index < bank.num_classes:
                        raise ValueError(f'{concept}: {modality} label {index} exceeds its class count.')
                    if not bank.valid[index]:
                        continue
                    proto = bank.protos[index].detach().clone()
                    batch_proto = current.get(modality, {}).get(index)
                    if batch_proto is not None:
                        # EMA value in forward; live batch mean in backward.
                        batch_mean, scale = batch_proto
                        proto = proto + scale * (batch_mean - batch_mean.detach())
                    prototypes.append(proto)
                if prototypes:
                    rows.append(torch.stack(prototypes).mean(dim=0))
            if len(rows) >= 2:
                logvols.append(self._class_volume_log(
                    F.normalize(torch.stack(rows).float(), dim=-1)))
        if not logvols:
            return {'loss_align': zero, 'align_logvol': zero.detach()}
        avg_logvol = torch.stack(logvols).mean()
        # softplus(log V) = log(1 + V) is nonnegative and strictly increasing
        # in volume. Minimizing it retains alignment rather than flipping the
        # direction with abs() or a minus sign. Keep raw log-volume diagnostic.
        return {'loss_align': self.vol_weight * F.softplus(avg_logvol),
                'align_logvol': avg_logvol.detach()}
        
        
class TextPrototypeAlignerV3(PrototypeAlignerV3):
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
                 use_llm_prompt = True,
                 prompt_file="/liuwenchu/code/detection/category_descriptions.json",
                 sar_clip_name='ViT-B-16',
                 sar_clip_path=None,
                 sar_clip_cache=None,
                 text_encoder_type='sarclip',
                 clip_model_name=None,
                 clip_processor_name=None,
                 modal_num_classes=None,
                 base_modal_categories=None,
                 all_modal_categories=None):
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
                         modal_num_classes=modal_num_classes,
                         in_dims=in_dims,
                         semantic_mapping=semantic_mapping,
                         modalities=modalities,
                         align_dim=align_dim,
                         ema_beta=ema_beta,
                         vol_weight=vol_weight,
                         is_base_training=is_base_training,
                         eps=eps)
        # 确保text在模态列表中
        if 'text' not in modalities:
            modalities = list(modalities) + ['text']
        
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        
        categories_by_modality = (
            base_modal_categories if is_base_training else all_modal_categories
        )
        if not categories_by_modality:
            raise ValueError(
                'TextPrototypeAlignerV3 requires modality-specific category lists; '
                'provide base_modal_categories and all_modal_categories.'
            )

        self.modal_categories = {
            m: tuple(categories)
            for m, categories in categories_by_modality.items()
            if m in self.modalities and m != 'text'
        }
        for m, categories in self.modal_categories.items():
            expected = self.modal_num_classes.get(m)
            if expected is not None and len(categories) != expected:
                raise ValueError(
                    f'{m} text category count ({len(categories)}) does not match '
                    f'its detector head ({expected}). Keep each modality category '
                    'list in the same local label order as its annotations.'
                )
        
        # Checkpoint expansion needs the trainable adapters and banks, but
        # regenerates text features from the real encoder at training startup.
        self.text_feats_by_modality = {}
        if not init_clip:
            return
        text_encoder_type = str(text_encoder_type).lower()
        local_sarclip = text_encoder_type in ('sarclip', 'sar_clip')
        if local_sarclip:
            # Derive defaults from this imported package, not config __file__:
            # MMCV 1.x evaluates configs from a temporary file path.
            model_config_dir = (
                Path(__file__).resolve().parents[1] / 'sar_clip' /
                'model_configs' / 'ViT-B-16')
            sar_clip_path = sar_clip_path or str(
                model_config_dir / 'vit_b_16_model.pth')
            sar_clip_cache = sar_clip_cache or str(model_config_dir)
            if not Path(sar_clip_path).is_file():
                raise FileNotFoundError(
                    f'SAR-CLIP checkpoint not found: {sar_clip_path}. '
                    'Place the ViT-B-16 weights in '
                    'mmrotate/models/sar_clip/model_configs/ViT-B-16/.')
            text_model = sar_clip.create_model_with_args(
                sar_clip_name, pretrained=sar_clip_path, device='cpu')
            text_model.eval()
            tokenizer = sar_clip.get_tokenizer(
                sar_clip_name, cache_dir=sar_clip_cache)
            text_processor = None
        elif text_encoder_type in (
                'huggingface', 'hf', 'rgbclip', 'rgb_clip'):
            if AutoModel is None or AutoProcessor is None:
                raise ImportError(
                    'Method3 Hugging Face text encoding requires transformers.')
            if not clip_model_name:
                raise ValueError(
                    'Set clip_model_name to a local Git-FSCLIP model directory.')
            local_model_dir = Path(clip_model_name).expanduser()
            if not local_model_dir.is_dir():
                raise FileNotFoundError(
                    f'Local Git-FSCLIP model directory not found: {local_model_dir}')
            processor_name = clip_processor_name or str(local_model_dir)
            if not Path(processor_name).expanduser().is_dir():
                raise FileNotFoundError(
                    f'Local Git-FSCLIP processor directory not found: '
                    f'{processor_name}')
            print(
                f'[Aligner] Loading local HuggingFace text encoder: '
                f'{local_model_dir}')
            text_model = AutoModel.from_pretrained(
                str(local_model_dir), local_files_only=True)
            text_processor = AutoProcessor.from_pretrained(
                str(Path(processor_name).expanduser()),
                local_files_only=True)
            text_model.eval()
            tokenizer = None
        else:
            raise ValueError(
                f'Unsupported method3 text_encoder_type={text_encoder_type!r}; '
                'expected sarclip or huggingface.')

        device = torch.device('cpu')
        self.text_feats_by_modality = {}
        with torch.no_grad():
            for modality, categories in self.modal_categories.items():
                if use_llm_prompt:
                    class_prompt_lists = build_class_prompts_from_json(
                        category_list=list(categories),
                        prompt_file=prompt_file,
                        use_templates=True,
                        use_llm=True,
                    )
                else:
                    class_prompt_lists = [
                        [f'a remote sensing image of {category}']
                        for category in categories
                    ]

                modal_text_feats = []
                for prompt_list in class_prompt_lists:
                    if local_sarclip:
                        tokens = tokenizer(prompt_list).to(device)
                        outputs = text_model.encode_text(tokens)
                    else:
                        inputs = text_processor(
                            text=prompt_list,
                            padding=True,
                            return_tensors='pt')
                        inputs = {
                            key: value.to(device) if torch.is_tensor(value)
                            else value
                            for key, value in inputs.items()
                        }
                        if not hasattr(text_model, 'get_text_features'):
                            raise AttributeError(
                                'The local Git-FSCLIP model does not expose '
                                'get_text_features().')
                        outputs = text_model.get_text_features(**inputs)
                    if not torch.is_tensor(outputs) or outputs.dim() != 2:
                        raise ValueError(
                            'Text encoder must return a [prompts, channels] '
                            f'tensor; got {type(outputs).__name__}.')
                    expected_text_dim = in_dims.get('text') if in_dims else None
                    if (expected_text_dim is not None and
                            outputs.size(-1) != expected_text_dim):
                        raise ValueError(
                            'Text encoder dimension does not match '
                            f'in_dims["text"]: got {outputs.size(-1)}, '
                            f'expected {expected_text_dim}.')
                    outputs = F.normalize(outputs, dim=-1)
                    mean_feat = F.normalize(outputs.mean(dim=0), dim=-1)
                    modal_text_feats.append(mean_feat)
                self.text_feats_by_modality[modality] = torch.stack(
                    modal_text_feats, dim=0
                )
        del text_model
        
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
        return super().update_and_compute_loss(batch_protos, batch_labels)
