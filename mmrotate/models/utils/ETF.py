# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ETF_mapping={
    
}

import torch
import torch.nn as nn
import torch.nn.functional as F

class ETFAuxiliaryClassifier(nn.Module):
    def __init__(self, feat_dim, num_classes, scale=10.0):
        super(ETFAuxiliaryClassifier, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.scale = scale

        # 构造 ETF prototype 矩阵 (D, C)
        with torch.no_grad():
            etf_proto = self._build_etf(num_classes, feat_dim)
        self.register_buffer("W", etf_proto)  # 固定参数，不训练

    def _build_etf(self, num_classes, feat_dim):
        I = torch.eye(num_classes)
        ones = torch.ones(num_classes, num_classes) / num_classes
        M = I - ones  # (C, C)

        # 在类别空间构造 ETF (C, C)
        U, S, Vh = torch.linalg.svd(M)
        proto_c = U[:, :num_classes].T  # (C, C)
        proto_c = F.normalize(proto_c, dim=0) * (num_classes / (num_classes - 1))**0.5

        # 如果特征维度和类别数不一致，用随机正交矩阵投影到 feat_dim
        if feat_dim != num_classes:
            rand_proj = torch.randn(feat_dim, num_classes, device=proto_c.device)
            rand_proj, _ = torch.linalg.qr(rand_proj)   # 保证正交
            proto = torch.matmul(rand_proj, proto_c)   # (D, C)
        else:
            proto = proto_c

        return proto  # (D, C)

    def forward(self, x, labels, base_training=False):
        """
        x: (N, D)
        labels: (N,) 或 (N,1)
        """
        if labels.dim() > 1:
            labels = labels.view(-1)

        x = F.normalize(x, dim=1)  # (N, D)
        if base_training:
            logits = torch.matmul(x, self.W[:, :self.num_classes-1]) * self.scale  # (N, C)
        else:
            logits = torch.matmul(x, self.W) * self.scale  # (N, C)
        
        loss = F.cross_entropy(logits, labels)
        return loss

    