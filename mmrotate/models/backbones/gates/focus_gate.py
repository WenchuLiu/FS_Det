from typing import Tuple, List, Dict, Literal

from mmrotate.models.backbones.gates.gate_funcs import ModalityAwareLogit
import torch
from torch import nn
from torch.distributions import Normal

from einops import rearrange, reduce

from ..utils import SparseDispatcher, cv_squared
from ...builder import GATE_FUNCS

from dataclasses import dataclass, field

@dataclass
class MMMoEArgs:
    type: str = "focus_gate"
    mid_scale: float = 1.0
    n_routed_experts: int = 64
    n_activated_experts: int = 6
    n_preferred: int = 3
    route_scale: float = 1.0
    gate_func_cfg: dict = field(default=dict)
    act_func: Literal["softmax", "sigmoid"] = "softmax"
    loss_coef_cfg: dict = field(default=dict)
    specific_expert_indices: Dict[str, List[int]] = field(default_factory=dict)
    share_expert_indices: List[int] = field(default_factory=list)
    specific_mask_value: float = 1.0
    share_mask_value: float = 0.0
    sup_share_mask: bool = True
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



class FocusGate(nn.Module):
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
        self.n_preferred = args.n_preferred
        
        self.specific_expert_indices = args.specific_expert_indices
        self.modal_names = list(self.specific_expert_indices.keys())
        self.share_expert_indices = args.share_expert_indices
        self.specific_mask_value = args.specific_mask_value
        self.share_mask_value = args.share_mask_value
        self.sup_share_mask = args.sup_share_mask

        self.use_token = args.use_token
        
        self.route_scale = args.route_scale
        self.act_func = args.act_func

        
        args.gate_func_cfg["in_channels"] = self.in_channels
        args.gate_func_cfg["n_routed_experts"] = self.n
        args.gate_func_cfg["modal_names"] = self.modal_names
        self.gate_func = GATE_FUNCS.build(args.gate_func_cfg)
        
        if self.use_token:
            self.fusion_module = nn.ModuleDict()
            for modal_name in self.modal_names:
                token = nn.Parameter(torch.randn((args.token_num, 
                                                  self.in_channels)),
                                    requires_grad=True).to(next(self.parameters()).device)
                setattr(self, f'token_{modal_name}', token)
                fusion_cfg = args.fusion_cfg
                fusion_cfg["dim"] = self.in_channels
                fusion_cfg["token_num"] = args.token_num
                fusion_module = FusionModule(**fusion_cfg)
                self.fusion_module[modal_name] = fusion_module
        
        self.eps = eps
        self.router_loss_coef = args.loss_coef_cfg.get("router_loss_coef", 1.0)
        self.balance_loss_coef = args.loss_coef_cfg.get("balance_loss_coef", 1e-2)
        self.modal_pref_loss_coef = args.loss_coef_cfg.get("modal_pref_loss_coef", 1e-2)
        
        self.preference_mask = nn.Parameter(torch.zeros(len(self.modal_names), self.n))
        self._init_modality_mask()

    def _init_modality_mask(self):
        for i, name in enumerate(self.modal_names):
            specific_idx = self.specific_expert_indices[name]
            share_idx = self.share_expert_indices
            self.preference_mask.data[i] = 0
            self.preference_mask.data[i, specific_idx] = self.specific_mask_value
            self.preference_mask.data[i, share_idx] = self.share_mask_value
        
        # print(self.preference_mask)
    
    def forward(self, 
                features: torch.Tensor,
                experts: nn.ModuleList,
                batch_indices: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, C = features.shape
        features = features.reshape(B, -1, C)
        
        expert_outputs = []
        balance_losses = []
        router_losses = []
        modal_pref_losses = []
        
        for modal_name, batch_index in batch_indices.items():            
            single_feature = features[batch_index]
            single_feature_fusion = single_feature
            if self.use_token:
                token = getattr(self, f'token_{modal_name}')
                fusion_module = self.fusion_module[modal_name]
                single_feature_fusion = fusion_module(single_feature, token, H, W)                
            
            # Router matmul and probability arithmetic must not overflow or
            # underflow in FP16, especially log probabilities used by KL.
            with torch.cuda.amp.autocast(enabled=False):
                if isinstance(self.gate_func, ModalityAwareLogit):
                    logit_single = self.gate_func(single_feature_fusion.float(), modal_name).reshape(-1, self.n)
                else:
                    logit_single = self.gate_func(single_feature_fusion.float()).reshape(-1, self.n)
            
            modality_id = self.modal_names.index(modal_name)
            bias = self.preference_mask[modality_id]
            logit_single = bias + logit_single
            

            if self.act_func == "softmax":
                log_act_logit_single = torch.nn.functional.log_softmax(logit_single, dim=-1)
                act_logit_single = log_act_logit_single.exp()
            elif self.act_func == "sigmoid":
                log_act_logit_single = torch.nn.functional.logsigmoid(logit_single)
                act_logit_single = logit_single.sigmoid()
            else:
                raise ValueError(f"Unknown logit function: {self.act_func}")

            top_k1_logit, top_k1_indices = act_logit_single.topk(min(self.topk + 1, self.n), dim= -1)
            top_k_logit = top_k1_logit[..., :self.topk]
            top_k_indices = top_k1_indices[..., :self.topk]
            
            top_k_logit_full = torch.zeros_like(logit_single)
            top_k_logit_full.scatter_(-1, top_k_indices, top_k_logit)        
            
            density_1_proxy = reduce(logit_single.sigmoid(), '... n e -> ... e', 'mean')
            one_hot_gate_indices = nn.functional.one_hot(rearrange(top_k_indices, '... k -> k ...'), self.n).float()[0]
            density_1 = reduce(one_hot_gate_indices, '... n e -> ... e', 'mean')
            balance_loss = (density_1_proxy * density_1).mean() * float(self.n ** 2)
            
            top_k_logit = top_k_logit * self.route_scale
            top_k_logit_full = top_k_logit_full * self.route_scale
            router_loss = cv_squared(top_k_logit_full.sum(dim=0))
            

            preferred_mask = (self.preference_mask[modality_id] > 0).bool()  # [k]
            preferred_indices = torch.nonzero(preferred_mask, as_tuple=False).squeeze(-1)  # [n]
            
            if preferred_indices.numel() == 0:
                modality_pref_loss = torch.tensor(0.0, device=logit_single.device, requires_grad=True)
            else:
                preferred_logits = act_logit_single[:, preferred_indices]
                preferred_log_probs = log_act_logit_single[:, preferred_indices]
                target_dist = torch.full_like(preferred_logits, fill_value=1.0)
                if not self.sup_share_mask:
                    # Convert share_expert_indices to tensor for indexing
                    share_expert_tensor = torch.tensor(self.share_expert_indices, device=preferred_indices.device)
                    # Create mask to exclude shared experts from preferred_indices
                    mask = torch.ones(preferred_indices.shape[0], dtype=torch.bool, device=preferred_indices.device)
                    for i, idx in enumerate(preferred_indices):
                        if idx in share_expert_tensor:
                            mask[i] = False
                    # Apply mask to exclude shared experts
                    if mask.any():
                        target_dist = target_dist[:, mask]
                        preferred_logits = preferred_logits[:, mask]
                        preferred_log_probs = preferred_log_probs[:, mask]
                    else:
                        # If all preferred experts are shared experts, set loss to 0
                        modality_pref_loss = torch.tensor(0.0, device=logit_single.device, requires_grad=True)
                        modal_pref_losses.append(modality_pref_loss)
                        balance_losses.append(balance_loss)
                        router_losses.append(router_loss)
                        continue
                modality_pref_loss = torch.nn.functional.kl_div(
                    preferred_log_probs,
                    target_dist,
                    reduction='mean')
            
            modal_pref_losses.append(modality_pref_loss)
            balance_losses.append(balance_loss)
            router_losses.append(router_loss)

            single_feature = single_feature.reshape(-1, C)
            dispatcher = SparseDispatcher(self.n, top_k_logit_full)
            dispatched_expert_input = dispatcher.dispatch(single_feature)
            dispatcher.expert_to_gates()
            expert_output = [experts[i](dispatched_expert_input[i]).reshape(-1, C) for i in range(self.n)]
            expert_output = dispatcher.combine(expert_output)
            expert_outputs.append(expert_output.reshape(-1, H, W, C))
        
        expert_outputs = torch.cat(expert_outputs, dim=0)
        
        gate_loss = sum(router_losses)/len(router_losses) * self.router_loss_coef +  sum(balance_losses)/len(balance_losses) * self.balance_loss_coef + sum(modal_pref_losses)/len(modal_pref_losses) * self.modal_pref_loss_coef
        return gate_loss, expert_outputs
