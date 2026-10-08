import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple

from mmrotate.models.builder import GATE_FUNCS


@GATE_FUNCS.register_module()
class ModalityAwareLogit(nn.Module):
    def __init__(self, 
                 in_channels, 
                 modal_names,
                 n_routed_experts):
        super().__init__()
        self.modal_names = modal_names
        self.modality_embeddings = nn.Embedding(len(modal_names), in_channels)
        self.shared_fc = nn.Linear(in_channels, n_routed_experts)

    def forward(self, x: torch.Tensor, dataset_name: str):
        modality_ids = self.modal_names.index(dataset_name)
        modality_ids = torch.tensor(modality_ids, dtype=torch.long, device=x.device)
        mod_embed = self.modality_embeddings(modality_ids)  # [B, D]
        gated_input = x + mod_embed
        return self.shared_fc(gated_input)

@GATE_FUNCS.register_module()
class LinearLogit(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 n_routed_experts, 
                 bias = False):
        super(LinearLogit, self).__init__()
        self.linear = nn.Linear(in_channels, 
                                n_routed_experts, 
                                bias = bias)        
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.linear(x)
        return logits

@GATE_FUNCS.register_module()
class CosineLogit(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 n_routed_experts, 
                 init_t = 0.5):
        super(CosineLogit, self).__init__()
        proj_dim = min(in_channels//2, 256)
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = torch.nn.Linear(in_channels, proj_dim)
        self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, n_routed_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x):
        cosine_projector = self.cosine_projector
        sim_matrix = self.sim_matrix
        logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
                              F.normalize(sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits
