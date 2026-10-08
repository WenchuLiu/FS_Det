from typing import Tuple, List, Dict, Literal

import torch
from torch import nn
from torch.distributions import Normal

from einops import rearrange, reduce

from ..utils import SparseDispatcher, cv_squared
from ...builder import GATE_FUNCS

from dataclasses import dataclass, field

@dataclass
class MMMoEArgs:
    type: str = "refinedgatev1"
    mid_scale: float = 1.0
    n_routed_experts: int = 64
    n_activated_experts: int = 6
    route_scale: float = 1.0
    gate_func_cfg: dict = field(default=dict)
    act_func: Literal["softmax", "sigmoid"] = "softmax"
    loss_coef_cfg: dict = field(default=dict)
    noisy_gating: bool = False
    noise_epsilon: float = 1e-2
    use_dispatch: bool = False
    dataset_names: List[str] = field(default_factory=list) 
    use_token: bool = False
    token_num: int = 16
    fusion_cfg: dict = field(default=dict)

class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x


class FusionModule(nn.Module):
    def __init__(self, dim, token_num, num_heads=4, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.token_num = token_num
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.dwc = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        nn.init.xavier_normal_(self.qkv.weight, gain=2 ** -2.5)
        nn.init.xavier_normal_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x, token, h, w):
        B, N, C = x.shape
        H = self.num_heads
        D = C // H

        qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D]

        token = token.unsqueeze(0).repeat(B, 1, 1).reshape(B, self.token_num, H, D).permute(0, 2, 1, 3)

        attn_token = self.softmax((token * self.scale) @ k.transpose(-2, -1))  # [B, H, T, N]
        attn_token = self.attn_drop(attn_token)
        fused_v = attn_token @ v  # [B, H, T, D]

        attn_q = self.softmax((q * self.scale) @ token.transpose(-2, -1))  # [B, H, N, T]
        attn_q = self.attn_drop(attn_q)
        x = attn_q @ fused_v  # [B, H, N, D]

        x = x.transpose(1, 2).reshape(B, N, C)

        v_conv = self.dwc(v.transpose(1, 2).reshape(B, N, D * H).reshape(B, h, w, C).permute(0, 3, 1, 2))
        x = x + v_conv.permute(0, 2, 3, 1).reshape(B, N, C)

        return self.proj_drop(self.proj(x))


class MMSplitGate(nn.Module):
    def __init__(self, 
                 in_channels: List[int], 
                 args: Dict, 
                 eps: float=1e-6):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        args = MMMoEArgs(**args)
        
        self.in_channels = in_channels
        self.n = args.n_routed_experts
        self.topk = args.n_activated_experts
        
        
        self.use_dispatch = args.use_dispatch
        self.dataset_names = args.dataset_names
        self.use_token = args.use_token
        
        self.route_scale = args.route_scale
        self.act_func = args.act_func

        
        args.gate_func_cfg["in_channels"] = self.in_channels
        args.gate_func_cfg["n_routed_experts"] = self.n
        self.share_weight = args.gate_func_cfg.pop("share_weight", False)
                
        if not self.share_weight:
            self.gate_func = nn.ModuleDict()
            for dataset_name in self.dataset_names:
                gate_func = GATE_FUNCS.build(args.gate_func_cfg)
                self.gate_func[dataset_name] = gate_func
        else:
            self.gate_func = GATE_FUNCS.build(args.gate_func_cfg)
        
        
        if self.use_token:
            self.fusion_module = nn.ModuleDict()
            for dataset_name in self.dataset_names:
                token = nn.Parameter(torch.randn((args.token_num, 
                                                  self.in_channels)),
                                    requires_grad=True).to(next(self.parameters()).device)
                setattr(self, f'token_{dataset_name}', token)
                for dataset_name in self.dataset_names:
                    fusion_cfg = args.fusion_cfg
                    fusion_cfg["dim"] = self.in_channels
                    fusion_cfg["token_num"] = args.token_num
                    fusion_module = FusionModule(**fusion_cfg)
                    self.fusion_module[dataset_name] = fusion_module
        
        self.noisy_gating = args.noisy_gating
        if self.noisy_gating:
            self.w_noise = nn.Parameter(torch.zeros(self.in_channels, 
                                                    self.n), 
                                        requires_grad=True)
            self.softplus = nn.Softplus()
            self.noise_epsilon = args.noise_epsilon
            
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.eps = eps
        
        self.router_loss_coef = args.loss_coef_cfg.get("router_loss_coef", 1.0)
        self.balance_loss_coef = args.loss_coef_cfg.get("balance_loss_coef", 1e-2)
        

    def _prob_in_top_k(self, 
                       clean_values,
                       noisy_values,
                       noise_stddev, 
                       noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten() # (bs x m)
        threshold_positions_if_in = torch.arange(batch) * m + self.topk # bs
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in.to(top_values_flat.device)), 1)

        if len(noisy_values.shape) == 3:
            threshold_if_in = threshold_if_in.unsqueeze(1)

        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out.to(top_values_flat.device)), 1)
        if len(noisy_values.shape) == 3:
            threshold_if_out = threshold_if_out.unsqueeze(1)

        # is each value currently in the top k.

        normal = Normal(self.mean.to(noise_stddev.device), self.std.to(noise_stddev.device))
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/ (noise_stddev + self.eps))
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/ (noise_stddev + self.eps))
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob
    
    
    def forward(self, 
                x: torch.Tensor,
                experts: nn.ModuleList,
                batch_indices: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        
        clean_logits = []

        for dataset_name in batch_indices:
            if not self.share_weight:
                gate_func = self.gate_func[dataset_name]
            else:
                gate_func = self.gate_func
            batch_index = batch_indices[dataset_name]
            x_single = x[batch_index]
            if self.use_token:
                token = getattr(self, f'token_{dataset_name}')
                fusion_module = self.fusion_module[dataset_name]
                x_single = fusion_module(x_single, token, H, W)                
            clean_logit_single = gate_func(x_single).reshape(-1, self.n)            
            clean_logits.append(clean_logit_single)
            
        clean_logits = torch.cat(clean_logits, dim=0)
        expert_inputs = x.reshape(-1, C)        
        
        if self.noisy_gating and self.training:
            raw_noise_stddev = expert_inputs @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon) * self.training)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        
        top_k1_logits, top_k1_indices = logits.topk(min(self.topk + 1, self.n), dim= -1)
        
        top_k_logits = top_k1_logits[..., :self.topk]
        top_k_indices = top_k1_indices[..., :self.topk]
        
        if self.act_func == "softmax":
            top_k_logits = top_k_logits.softmax(dim = -1)
        elif self.act_func == "sigmoid":
            top_k_logits = top_k_logits.sigmoid()
        else:
            raise ValueError(f"Unknown logit function: {self.act_func}")
        
        zeros = torch.zeros_like(logits, dtype=top_k_logits.dtype, requires_grad=True)
        top_k_logits_full = zeros.scatter(-1, top_k_indices, top_k_logits)        
        
        if self.noisy_gating and self.topk < self.n and self.training:
            load = (self._prob_in_top_k(clean_logits, 
                                        noisy_logits, 
                                        noise_stddev,
                                        top_k1_logits)).sum(0)
            balance_loss = cv_squared(load)
        else:
            density_1_proxy = reduce(logits.sigmoid(), '... n e -> ... e', 'mean')
            one_hot_gate_indices = nn.functional.one_hot(rearrange(top_k_indices, '... k -> k ...'), self.n).float()[0]
            density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')
            balance_loss = (density_1_proxy * density_1).mean() * float(self.n ** 2)
        
        top_k_logits = top_k_logits * self.route_scale
        top_k_logits_full = top_k_logits_full * self.route_scale
        
        router_loss = cv_squared(top_k_logits_full.sum(dim=0))
        gate_loss = router_loss * self.router_loss_coef +  balance_loss * self.balance_loss_coef
        
        if self.use_dispatch:
            dispatcher = SparseDispatcher(self.n, top_k_logits_full)
            dispatched_expert_input = dispatcher.dispatch(expert_inputs)
            top_k_logits_full = dispatcher.expert_to_gates()
            expert_outputs = [experts[i](dispatched_expert_input[i]).reshape(-1, C) for i in range(self.n)]
            expert_outputs = dispatcher.combine(expert_outputs)
        else:
            counts = torch.bincount(top_k_indices.flatten(), 
                                    minlength=self.n).tolist()
            expert_outputs = torch.zeros_like(expert_inputs)
            for i in range(self.n):
                if counts[i] == 0:
                    counts[i] = 1
                expert = experts[i]
                idx, top = torch.where(top_k_indices == i)
                expert_outputs[idx] += expert(expert_inputs[idx]) * top_k_logits[idx, top, None]
            expert_outputs = expert_outputs
        
        expert_outputs = expert_outputs.reshape(B, H, W, C)
        return gate_loss, expert_outputs