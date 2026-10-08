from typing import Tuple, List, Dict


import torch
from torch import nn
from torch.distributions import Normal

from einops import rearrange, reduce

from ..utils import cv_squared, SparseDispatcher, MoEArgs
from ...builder import GATE_FUNCS


class OriginGate(nn.Module):
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
        args = MoEArgs(**args)
        self.in_channels = in_channels
        self.topk = args.n_activated_experts
        self.n = args.n_routed_experts
        self.use_dispatch = args.use_dispatch
        
        self.route_scale = args.route_scale
        
        args.gate_func_cfg["in_channels"] = self.in_channels
        args.gate_func_cfg["n_routed_experts"] = args.n_routed_experts
        
        self.gate_logit = GATE_FUNCS.build(args.gate_func_cfg)
        
        self.noisy_gating = args.noisy_gating
        self.act_func = args.act_func
        
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
    
    
    def forward(self, expert_input: torch.Tensor, experts: nn.ModuleList, batch_indices=None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, H, W, C = expert_input.shape        
        expert_input = expert_input.reshape(-1, C)
        
        clean_logits = self.gate_logit(expert_input)
        
        if self.noisy_gating and self.training:
            raw_noise_stddev = expert_input @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + self.noise_epsilon) * self.training)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
                    
        top_k1_logits, top_k1_indices = logits.topk(min(self.topk + 1, self.n), dim= -1)
        
        top_k_logits = top_k1_logits[..., :self.topk]
        top_k_indices = top_k1_indices[..., :self.topk]
        
        
        if self.act_func == "softmax":
            top_k_logits = top_k_logits.softmax(dim = -1, dtype=torch.float32).clamp(min=1e-9)
        elif self.act_func == "sigmoid":
            top_k_logits = top_k_logits.sigmoid()
        else:
            raise ValueError(f"Unknown logit function: {self.act_func}")
        
        
        zeros = torch.zeros_like(logits, requires_grad=True)
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
            dispatched_expert_input = dispatcher.dispatch(expert_input)
            top_k_logits_full = dispatcher.expert_to_gates()
            expert_outputs = [experts[i](dispatched_expert_input[i]).reshape(-1, C) for i in range(self.n)]
            expert_outputs = dispatcher.combine(expert_outputs)
        else:
            counts = torch.bincount(top_k_indices.flatten(), 
                                    minlength=self.n).tolist()
            expert_outputs = torch.zeros_like(expert_input)
            for i in range(self.n):
                if counts[i] == 0:
                    counts[i] = 1
                expert = experts[i]
                idx, top = torch.where(top_k_indices == i)
                expert_outputs[idx] += expert(expert_input[idx]) * top_k_logits[idx, top, None]
            expert_outputs = expert_outputs
        
        expert_outputs = expert_outputs.reshape(B, H, W, C)
        return gate_loss, expert_outputs