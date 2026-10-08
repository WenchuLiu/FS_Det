'''
不用每个层维护一个prototype，而是一个类一个即可，不然太多了
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch.cuda.amp import autocast
import os
import json
from pathlib import Path

from .. import sar_clip

try:
    from transformers import AutoModel, AutoProcessor
except ImportError:
    AutoModel = None
    AutoProcessor = None

class PrototypeBank(nn.Module):
    def __init__(self, num_classes, dim, beta=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.beta = beta
        
        # 🔥 [修改] 不再区分 stride，只维护全局唯一的 prototypes
        self.register_buffer('protos', torch.zeros(num_classes, dim))
        self.register_buffer('valid', torch.zeros(num_classes, dtype=torch.bool))

    def get_proto(self):
        """直接获取全局 prototype"""
        return self.protos

    def get_valid(self):
        """直接获取全局 valid mask"""
        return self.valid

    @torch.no_grad()
    def update(self, class_ids: torch.LongTensor, protos_batch: torch.Tensor):
        """
        🔥 [修改] 移除 stride 参数
        """
        target_protos = self.protos
        target_valid = self.valid

        for i, c in enumerate(class_ids.tolist()):
            if c >= self.num_classes or c < 0: continue 
            
            if target_valid[c]:
                # EMA 更新
                target_protos[c] = (1.0 - self.beta) * target_protos[c] + self.beta * protos_batch[i]
            else:
                # 初始化
                target_protos[c] = protos_batch[i]
                target_valid[c] = True
                

class TextPrototypeAlignerV10(nn.Module):
    def __init__(self,
                 modal_num_classes: dict, 
                 semantic_mapping=None,  # optional in V16 backbone-text-only path
                 
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

                 # HuggingFace / RGB-CLIP Params
                 text_encoder_type='sarclip',
                 clip_model_name='lcybuaa/Git-RSCLIP',
                 clip_processor_name=None,
                 text_prompt_template=None,
                 
                 # Text Params
                 prompt_file=None,
                 use_llm_prompt=False,
                 eps=1e-6,
                 **kwargs):
        
        super().__init__()
        # Some downstream SAR-CLIP dependencies may configure the root logger,
        # which makes mmrotate training logs appear twice via propagation.
        logging.getLogger('mmrotate').propagate = False
        
        if 'text' not in modalities:
            modalities = list(modalities) + ['text']
            
        self.modalities = list(modalities)
        self.align_dim = align_dim 
        self.vol_weight = vol_weight
        self.eps = eps
        self.text_encoder_type = str(text_encoder_type).lower()
        
        self.modal_num_classes = modal_num_classes
        self.semantic_mapping = semantic_mapping or {}
        
        self.category_lists = {
            'sar': sar_category if sar_category else [],
            'rgb': rgb_category if rgb_category else [],
            'ifr': ifr_category if ifr_category else []
        }
        visual_modals = [m for m in self.modalities if m != 'text']
        for modality in visual_modals:
            category_names = self.category_lists.get(modality, [])
            expected_count = self.modal_num_classes.get(modality)
            if expected_count and not category_names:
                raise ValueError(
                    f'{modality} text alignment needs category names from the schema.')
            if expected_count is not None and len(category_names) != expected_count:
                raise ValueError(
                    f'{modality} category count ({len(category_names)}) does not match '
                    f'its detector head ({expected_count}); check fsdet_configs/schema.py.')
        
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

        clip_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hf_device = torch.device('cpu')
        full_model = None
        temp_model = None
        tokenizer = None
        text_model = None
        text_processor = None

        # =====================================================================
        # [Part A] Init text encoder backend
        # =====================================================================
        if self.text_encoder_type in ('sarclip', 'sar_clip'):
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

            print(f"[Aligner] Loading SAR-CLIP: {sar_clip_name} from {sar_clip_path}")
            full_model = sar_clip.create_model_with_args(
                sar_clip_name, pretrained=sar_clip_path, device='cpu')
            full_model.eval()

            tokenizer = sar_clip.get_tokenizer(sar_clip_name, cache_dir=sar_clip_cache)
            temp_model = full_model
            clip_device = torch.device('cpu')

            with torch.no_grad():
                dummy_emb = temp_model.encode_text(tokenizer(["test"]).to(clip_device))
                self.clip_dim = dummy_emb.shape[-1]
        elif self.text_encoder_type in ('huggingface', 'hf', 'rgbclip', 'rgb_clip'):
            if AutoModel is None or AutoProcessor is None:
                raise ImportError(
                    'text_encoder_type=huggingface requires transformers to be installed.')

            processor_name = clip_processor_name or clip_model_name
            print(f"[Aligner] Loading HuggingFace text encoder: {clip_model_name}")
            text_model = AutoModel.from_pretrained(clip_model_name)
            text_processor = AutoProcessor.from_pretrained(processor_name)
            text_model.eval()
            text_model = text_model.to(hf_device)

            with torch.no_grad():
                dummy_inputs = text_processor(
                    text=["test"],
                    padding="max_length",
                    return_tensors="pt")
                dummy_inputs = {
                    k: v.to(hf_device) if torch.is_tensor(v) else v
                    for k, v in dummy_inputs.items()
                }
                if not hasattr(text_model, 'get_text_features'):
                    raise AttributeError(
                        f'{clip_model_name} does not provide get_text_features().')
                dummy_emb = text_model.get_text_features(**dummy_inputs)
                self.clip_dim = dummy_emb.shape[-1]
        else:
            raise ValueError(
                f'Unsupported text_encoder_type: {text_encoder_type}. '
                'Expected sarclip or huggingface.')

        # =====================================================================
        # [Part B] Generate Text Features
        # =====================================================================
        prompts_dict = {}
        if use_llm_prompt and prompt_file and os.path.exists(prompt_file):
            print(f"[Aligner] Loading LLM prompts from {prompt_file}")
            with open(prompt_file, 'r') as f:
                prompts_dict = json.load(f)

        if text_prompt_template is None:
            if self.text_encoder_type in ('sarclip', 'sar_clip'):
                text_prompt_template = 'a SAR image of {}.'
            else:
                text_prompt_template = 'a remote sensing image of {}'

        def build_default_prompt(cat_name):
            if '{}' in text_prompt_template:
                return text_prompt_template.format(cat_name)
            return f'{text_prompt_template} {cat_name}'.strip()

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
                        current_prompts = [build_default_prompt(cat_name)]

                    if self.text_encoder_type in ('sarclip', 'sar_clip'):
                        tokens = tokenizer(current_prompts).to(clip_device)
                        feats = temp_model.encode_text(tokens)
                    else:
                        inputs = text_processor(
                            text=current_prompts,
                            padding="max_length",
                            return_tensors="pt")
                        inputs = {
                            k: v.to(hf_device) if torch.is_tensor(v) else v
                            for k, v in inputs.items()
                        }
                        feats = text_model.get_text_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    embeddings.append(feats.mean(dim=0).cpu())
            return torch.stack(embeddings)

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
        
        # The visual tower is not used by this aligner. Release the temporary
        # SAR-CLIP model after producing text prototypes instead of retaining
        # its large visual weights on the detector (and later on the GPU).
        if full_model is not None:
            del full_model
            temp_model = None
        if text_model is not None:
            text_model = text_model.to('cpu')
            del text_model
        torch.cuda.empty_cache()

        # =====================================================================
        # [Part C] Init Banks
        # =====================================================================
        self.banks = nn.ModuleDict()
        
        if 'text' in self.modalities and len(self.concept_list) > 0:
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
        batch_info 结构变更:
        {
            'sar': (all_feats, all_labels),
            'rgb': (all_feats, all_labels),
            ...
        }
        """
        device = next(self.parameters()).device
        
        # ... (Text 部分不变) ...

        # 2. Update Visual Banks
        for m in self.modalities:
            if m == 'text': continue
            
            # 🔥 [修改] 直接获取元组，不再是字典
            data_tuple = batch_info.get(m, None)
            if data_tuple is None: continue
            
            feats, labels = data_tuple
            
            if feats is None or labels is None or feats.numel() == 0: continue
            
            feats = feats.to(device)
            labels = labels.to(device)
            
            if m not in self.banks: continue
            bank_module = self.banks[m]
            
            # 过滤非法 label
            valid_mask = (labels >= 0) & (labels < bank_module.num_classes)
            if valid_mask.any():
                # 🔥 [修改] 调用 Bank update，不再传 stride
                bank_module.update(labels[valid_mask], feats[valid_mask])
    
    
