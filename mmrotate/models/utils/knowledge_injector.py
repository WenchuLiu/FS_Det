import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import roi_align
from mmdet.core import bbox2roi

class SemanticDecayCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4, beta_init=0.9):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 可学习的衰减系数 beta，初始化为 sigmoid(2.2) ≈ 0.9
        self.beta_logits = nn.Parameter(torch.tensor(2.2))

    def get_beta(self):
        return torch.sigmoid(self.beta_logits)

    def forward(self, x_sar, mixed_protos, sim_matrix):
        """
        x_sar: [B, L, C]
        mixed_protos: [B, N_total, C]
        sim_matrix: [B, 1, N_total] (Semantic Similarity)
        """
        B, L, C = x_sar.shape
        _, N, _ = mixed_protos.shape
        
        # 1. Q, K, V
        q = self.to_q(x_sar).reshape(B, L, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.to_k(mixed_protos).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.to_v(mixed_protos).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        # 2. Attention Logits
        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn_logits.softmax(dim=-1) # [B, Heads, L, N]
        
        # 3. Apply Semantic Decay
        # Distance = 1 - Similarity (范围 [0, 2])
        sem_dist = 1.0 - sim_matrix.unsqueeze(1) # 广播到 [B, 1, 1, N]
        
        beta = self.get_beta()
        # decay = beta ^ dist
        decay_mask = torch.exp(sem_dist * torch.log(beta + 1e-6))
        
        weighted_attn = attn_probs * decay_mask
        
        # 4. Aggregate
        out = (weighted_attn @ v).transpose(1, 2).reshape(B, L, C)
        return self.out_proj(out)

class TwoStageProtoInjector(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        # Stage 1: 原型间的交互 (带语义衰减)
        self.proto_attn = SemanticDecayCrossAttention(dim)
        self.proto_norm = nn.LayerNorm(dim)
        
        # Stage 2: 特征图与原型的交互 (标准 CA)
        self.feat_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.feat_norm = nn.LayerNorm(dim)
        
        self.inject_gamma = nn.Parameter(torch.ones(1)*0.1)

    def forward(self, sar_feat, aligner, stride):
        """
        sar_feat: SAR Feature Map [B, C, H, W]
        aligner: 包含 banks 和 text_feats 的对象
        stride: 当前处理的层级
        """
        B, C, H, W = sar_feat.shape
        device = sar_feat.device
        
        # =========================================================
        # [Preparation] 准备 Text 和 Visual Prototypes
        # =========================================================
        
        # 1. 获取 SAR 的 Visual Prototypes (Query) 和 Text (Query Anchor)
        # 注意：这里我们取当前 Stride 下 SAR 的所有有效原型
        sar_bank = aligner.banks['sar']
        sar_proto_vis = sar_bank.get_proto(stride)
        sar_valid_mask = sar_bank.get_valid(stride)
        
        # 获取有效的索引
        sar_valid_idx = torch.nonzero(sar_valid_mask).squeeze(1)
        if len(sar_valid_idx) == 0: return sar_feat # 如果当前没有 SAR 原型，直接跳过
        
        # 筛选出有效的 SAR Visual Proto [N_sar, C]
        q_proto_vis = sar_proto_vis[sar_valid_idx].unsqueeze(0).expand(B, -1, -1) # [B, N_sar, C]
        
        # 获取对应的 SAR Text Feats
        # 这里的 text_feats_sar 应该是预先存好的 Tensor [Total_SAR_Cls, C]
        if hasattr(aligner, 'text_feats_sar'):
            q_text = aligner.text_feats_sar[sar_valid_idx].unsqueeze(0).expand(B, -1, -1) # [B, N_sar, C]
        else:
            return sar_feat # 没有文本无法计算衰减
            
        # 2. 获取 RGB + IFR 的 Visual Prototypes (Key) 和 Text (Key Anchor)
        k_proto_list = []
        k_text_list = []
        
        for m in ['rgb', 'ifr']:
            if m in aligner.banks and hasattr(aligner, f'text_feats_{m}'):
                bank = aligner.banks[m]
                # 尝试获取当前 Stride，如果没有则回退到 32
                proto = bank.get_proto(stride)
                valid = bank.get_valid(stride)
                if proto is None: 
                    proto = bank.get_proto(32)
                    valid = bank.get_valid(32)
                
                # 筛选有效
                valid_idx = torch.nonzero(valid).squeeze(1)
                if len(valid_idx) > 0:
                    # Visual: [N_m, C]
                    k_proto_list.append(proto[valid_idx])
                    # Text: [N_m, C] (直接从 aligner 存好的 text_feats 里取)
                    text_feats = getattr(aligner, f'text_feats_{m}')
                    k_text_list.append(text_feats[valid_idx])
        
        if len(k_proto_list) == 0: return sar_feat
        
        # 拼接 RGB 和 IFR
        # Key Visual: [B, N_rgb+ifr, C]
        k_proto_vis = torch.cat(k_proto_list, dim=0).unsqueeze(0).expand(B, -1, -1).detach() 
        # Key Text: [B, N_rgb+ifr, C]
        k_text = torch.cat(k_text_list, dim=0).unsqueeze(0).expand(B, -1, -1)
        
        # =========================================================
        # [Stage 1] 计算语义相似度并做 Prototype CA
        # =========================================================
        
        # 计算 SAR Text 与 (RGB+IFR) Text 的相似度
        # Q: [B, N_sar, C], K: [B, N_k, C] -> Sim: [B, N_sar, N_k]
        q_text_norm = F.normalize(q_text, dim=-1)
        k_text_norm = F.normalize(k_text, dim=-1)
        sim_matrix = torch.bmm(q_text_norm, k_text_norm.transpose(1, 2))
        
        # 执行语义衰减 Attention
        # Output: [B, N_sar, C] -> 这是“增强后的 SAR 原型”
        # 它融合了 RGB/IFR 中与其语义相似的原型信息
        enhanced_sar_proto = self.proto_attn(q_proto_vis, k_proto_vis, sim_matrix)
        
        # 残差连接 + Norm (更新 SAR 原型表达)
        enhanced_sar_proto = self.proto_norm(q_proto_vis + enhanced_sar_proto)
        
        # =========================================================
        # [Stage 2] 特征图与增强原型的 CA
        # =========================================================
        
        # Query: Feature Map [B, H*W, C]
        feat_flat = sar_feat.flatten(2).transpose(1, 2)
        feat_query = self.feat_norm(feat_flat)
        
        # Key/Value: Enhanced SAR Prototypes [B, N_sar, C]
        # 特征图去查询这些增强后的原型
        feat_out, _ = self.feat_attn(feat_query, enhanced_sar_proto, enhanced_sar_proto)
        
        # Reshape & Injection
        feat_out = feat_out.transpose(1, 2).view(B, C, H, W)
        
        return sar_feat + self.inject_gamma * feat_out