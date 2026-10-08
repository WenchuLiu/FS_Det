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

    
class FrequencySpectralInjector(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.dim = dim
        
        # 1. Query 生成器: 将 SAR 的全局幅度谱映射为 Query
        self.to_q = nn.Linear(dim, dim)
        
        # 2. Key/Value 生成器: 将 RGB/IFR 原型映射为 K, V
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        # 3. 门控生成器 (Context -> Gate)
        # 包含 LayerNorm 以稳定训练，Sigmoid 将输出限制在 (0, 1)
        self.gate_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, dim),
            nn.Sigmoid()
        )
        
        # 4. 上下文规范化
        self.norm = nn.LayerNorm(dim)
        
        # 5. 可学习的注入强度 (初始化为0，形成 Identity Mapping)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, sar_feat, aligner, stride=None):
        """
        sar_feat: [B, C, H, W]
        aligner: TextPrototypeAlignerV9 实例
        """
        B, C, H, W = sar_feat.shape
        # 记录原始数据类型 (e.g., torch.float16)
        original_dtype = sar_feat.dtype 
        
        # =========================================================
        # 1. 准备 Key/Value (RGB & IFR Prototypes)
        # =========================================================
        memory_list = []
        for m in ['rgb', 'ifr']:
            if m in aligner.banks:
                bank = aligner.banks[m]
                proto = bank.get_proto()     
                valid = bank.get_valid()     
                
                valid_idx = torch.nonzero(valid).squeeze(1)
                if len(valid_idx) > 0:
                    memory_list.append(proto[valid_idx])
        
        if len(memory_list) == 0:
            return sar_feat
            
        memory_bank = torch.cat(memory_list, dim=0)
        
        # =========================================================
        # 2. 提取 SAR 频域 Query (🔥 关键修改：强制转 FP32)
        # =========================================================
        # cuFFT 在 FP16 下要求尺寸为 2 的幂次，转为 FP32 则无此限制
        sar_feat_fp32 = sar_feat.float() 

        # FFT 变换 (在 FP32 下进行)
        sar_fft = torch.fft.rfft2(sar_feat_fp32, norm='ortho')
        sar_amp = torch.abs(sar_fft)   # FP32
        sar_pha = torch.angle(sar_fft) # FP32 
        
        # 计算全局幅度统计量
        sar_amp_gap = sar_amp.mean(dim=[-2, -1]) # FP32
        
        # =========================================================
        # 3. 频域 Cross Attention
        # =========================================================
        # 为了进入 Linear 层，我们需要把数据转回原始精度 (FP16)
        # 因为 self.to_q 的权重是 FP16 的
        q_input = sar_amp_gap.to(original_dtype)
        
        # 生成 Query
        q = self.to_q(q_input).unsqueeze(1) 
        
        # Key/Value
        k = self.to_k(memory_bank).unsqueeze(0).expand(B, -1, -1) 
        v = self.to_v(memory_bank).unsqueeze(0).expand(B, -1, -1) 
        
        # Attention 计算 (FP16)
        attn_logits = torch.bmm(q, k.transpose(1, 2)) 
        attn_logits = attn_logits / (self.dim ** 0.5)
        attn_weights = F.softmax(attn_logits, dim=-1)
        
        # 获取上下文
        context = torch.bmm(attn_weights, v).squeeze(1) 
        context = self.norm(context + q_input) # Residual
        
        # =========================================================
        # 4. 幅度调制与重构 (Injection)
        # =========================================================
        # 生成门控 (FP16)
        gate = self.gate_net(context).view(B, C, 1, 1)
        
        # 🔥 计算增量 (混和精度计算)
        # Gate 和 Alpha 是 FP16，sar_amp 是 FP32
        # 我们将 Gate 转为 FP32 进行幅度调制，保证频域计算精度
        gate_fp32 = gate.float()
        alpha_fp32 = self.alpha.float()
        
        # 计算幅度增量 (FP32)
        sar_amp_delta = sar_amp * (alpha_fp32 * gate_fp32)
        
        # 结合相位 (FP32)
        sar_fft_delta = torch.polar(sar_amp_delta, sar_pha)
        
        # iFFT 回到空间域 (FP32)
        sar_feat_delta_fp32 = torch.fft.irfft2(sar_fft_delta, s=(H, W), norm='ortho')
        
        # 🔥 转回原始精度并残差连接
        sar_feat_delta = sar_feat_delta_fp32.to(original_dtype)
        
        return sar_feat + sar_feat_delta, attn_weights