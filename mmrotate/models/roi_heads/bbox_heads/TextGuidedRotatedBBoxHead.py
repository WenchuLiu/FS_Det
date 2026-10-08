import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.utils import build_linear_layer
from ...builder import ROTATED_HEADS
from .convfc_rbbox_head import RotatedConvFCBBoxHead

@ROTATED_HEADS.register_module()
class TextGuidedRotatedShared2FCBBoxHead(RotatedConvFCBBoxHead):
    def __init__(self, 
                 fc_out_channels=1024, 
                 text_dim=512, 
                 logit_scale_init_value=4.605, # ln(100) ≈ 4.6
                 *args, 
                 **kwargs):
        
        # 强制设置结构参数
        kwargs['num_shared_convs'] = 0
        kwargs['num_shared_fcs'] = 2
        kwargs['num_cls_convs'] = 0
        kwargs['num_cls_fcs'] = 0
        kwargs['num_reg_convs'] = 0
        kwargs['num_reg_fcs'] = 0
        
        super(TextGuidedRotatedShared2FCBBoxHead, self).__init__(
            fc_out_channels=fc_out_channels, 
            *args, 
            **kwargs)
        
        self.text_dim = text_dim

        # 🔥 [关键修复] 手动删除父类创建的 fc_cls
        # 父类 RotatedConvFCBBoxHead 可能会根据 num_classes 自动创建 self.fc_cls
        if hasattr(self, 'fc_cls'):
            del self.fc_cls
        # 3. 🔥 [修复 DDP 报错] 删除未使用的 fc_cls
        if hasattr(self, 'fc_cls'):
            del self.fc_cls 

        # 4. 🔥 [修复 Init 报错] 清洗 init_cfg
        # 父类默认的 init_cfg 里包含了对 fc_cls 的初始化指令，必须将其移除
        if self.init_cfg and isinstance(self.init_cfg, list):
            new_init_cfg = []
            for cfg in self.init_cfg:
                # 检查当前配置是否针对 fc_cls
                is_fc_cls_config = False
                if isinstance(cfg, dict) and 'override' in cfg:
                    # override 可能是 dict 也可能是 list
                    overrides = cfg['override']
                    if not isinstance(overrides, list):
                        overrides = [overrides]
                    
                    # 遍历 override 查看是否包含 name='fc_cls'
                    for o in overrides:
                        if o.get('name') == 'fc_cls':
                            is_fc_cls_config = True
                            break
                
                # 如果不是针对 fc_cls 的配置，则保留
                if not is_fc_cls_config:
                    new_init_cfg.append(cfg)
            
            self.init_cfg = new_init_cfg
        
        # 1. 对齐层 (Visual -> Text space)
        # 将 ROI 特征 (1024) 映射到 Text 特征维度 (512)
        self.fc_align = build_linear_layer(
            self.cls_predictor_cfg,
            in_features=self.cls_last_dim,
            out_features=self.text_dim
        )
        
        # 2. 🔥 [新增] 背景类 Embedding (Learnable)
        # 假设 text_embeds 只包含 N 个前景类，我们需要第 N+1 个向量表示背景
        self.bg_embedding = nn.Parameter(torch.randn(1, self.text_dim))
        nn.init.normal_(self.bg_embedding, std=0.01)

        # 3. 🔥 [新增] 温度系数 (Logit Scale)
        # 计算 Cosine Similarity 后数值在 [-1, 1]，通过 scale 放大以便 Softmax 收敛
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        # 初始化新增层
        if self.init_cfg is None:
            nn.init.xavier_uniform_(self.fc_align.weight)
            nn.init.constant_(self.fc_align.bias, 0)

    @auto_fp16()
    def forward(self, x, text_embeds):
        """
        Args:
            x: RoI Features [N, C, H, W] or [N, C]
            text_embeds: [Num_Classes, Text_Dim]
        """
        # ===== 1. Shared Part =====
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # ===== 2. Regression Branch (Standard) =====
        bbox_pred = self.fc_reg(x) if self.with_reg else None

        # ===== 3. Classification Branch (Text-Guided) =====
        # A. 映射视觉特征
        visual_feat = self.fc_align(x) # [N_roi, text_dim]
        visual_feat = F.normalize(visual_feat, dim=-1) # 归一化

        # B. 准备分类器权重 (Text Embeds + BG Embed)
        # text_embeds shape: [num_classes, text_dim]
        text_feat = F.normalize(text_embeds, dim=-1)
        bg_feat = F.normalize(self.bg_embedding, dim=-1)
        
        # 拼接前景和背景: [num_classes + 1, text_dim]
        classifier_weights = torch.cat([text_feat, bg_feat], dim=0)

        # C. 计算 Cosine Similarity 作为 Logits
        # [N_roi, text_dim] @ [text_dim, num_classes+1] -> [N_roi, num_classes+1]
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100) # 防止溢出
        cls_score = logit_scale * (visual_feat @ classifier_weights.t())

        return cls_score, bbox_pred, visual_feat