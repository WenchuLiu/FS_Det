from typing import Tuple, List, Dict, Literal

import torch
from torch import nn
from torch.distributions import Normal

from einops import rearrange, reduce

from ..utils import cv_squared, SparseDispatcher, MoEArgs
from ...builder import GATE_FUNCS

from dataclasses import dataclass, field

@dataclass
class MMMoEArgs:
    type: str = "refinedgatev1"
    mid_scale: float = 1.0
    n_routed_experts: int = 8



class ShareGate(nn.Module):
    def __init__(self, 
                 in_channels: List[int], 
                 args: Dict):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        args = MoEArgs(**args)
        self.in_channels = in_channels
        self.n = args.n_routed_experts
    
    def forward(self, expert_input: torch.Tensor, experts: nn.ModuleList, batch_indices=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, C = expert_input.shape        
        expert_input = expert_input.reshape(-1, C)
        expert_outputs = torch.zeros_like(expert_input)
        for i in range(self.n):
            expert = experts[i]            
            expert_outputs += expert(expert_input)
        expert_outputs = expert_outputs / self.n
        expert_outputs = expert_outputs.reshape(B, H, W, C)
        return expert_outputs.sum() * 0, expert_outputs