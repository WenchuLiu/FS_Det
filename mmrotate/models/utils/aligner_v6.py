import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, sys
from torch.cuda.amp import autocast

# -------------------------------------------------------------------------
# 1. 尝试导入 SAR-CLIP
# -------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.dirname(current_dir)
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

try:
    from .. import sar_clip
    # print(f"[Aligner] Successfully imported sar_clip from {sar_clip.__file__}")
except ImportError as e:
    print(f"[Aligner] Failed to import sar_clip: {e}")

# -------------------------------------------------------------------------
# 2. 辅助模块
# -------------------------------------------------------------------------
class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512):
        super().__init__()
        if hidden_dim is None: hidden_dim = max(in_dim // 2, out_dim)
        self.down = nn.Linear(in_dim, hidden_dim, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Linear(hidden_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        x = self.up(self.act(self.down(x)))
        x = self.norm(x)
        return x

class PrototypeBank(nn.Module):
    def __init__(self, num_classes, dim, beta=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.beta = beta
        self.register_buffer('protos', torch.zeros(num_classes, dim))
        self.register_buffer('valid', torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, class_ids: torch.LongTensor, protos_batch: torch.Tensor):
        for i, c in enumerate(class_ids.tolist()):
            if self.valid[c]:
                self.protos[c] = (1.0 - self.beta) * self.protos[c] + self.beta * protos_batch[i]
            else:
                self.protos[c] = protos_batch[i]
                self.valid[c] = True

# -------------------------------------------------------------------------
# 3. TextPrototypeAlignerV6 (功能分离版)
# -------------------------------------------------------------------------
class TextPrototypeAlignerV6(nn.Module):
    def __init__(self,
                 num_classes=26,
                 in_dims=None,
                 semantic_mapping=None,
                 modalities=('sar', 'rgb', 'ifr', 'text'),
                 align_dim=256,
                 ema_beta=0.05,
                 vol_weight=0.5,
                 is_base_training=True,
                 # SAR-CLIP 参数
                 sar_clip_name='ViT-B/16',
                 sar_clip_path=r'./sar_clip/model_configs/ViT-B-16.pt',
                 sar_clip_cache=r"./sar_clip/model_configs",
                 sar_all_category=['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'],
                 # ITA Loss 映射
                 modal_class_mapping=None, 
                 # 其他参数
                 category_list=None,
                 fpn_channels=256,
                 eps=1e-6,
                 **kwargs):
        
        super().__init__()
        
        # 参数初始化
        if in_dims is None:
            # ⚠️ 注意: text 的维度稍后会被 self.clip_dim 覆盖
            in_dims = {'sar': 256, 'rgb': 256, 'ifr': 256, 'text': 768}
        
        if 'text' not in modalities:
            modalities = list(modalities) + ['text']

        self.num_classes = num_classes
        # self.in_dims = in_dims # 暂存，后面更新
        self.modalities = list(modalities)
        self.align_dim = align_dim
        self.vol_weight = vol_weight
        self.is_base_training = is_base_training
        self.eps = eps
        self.semantic_mapping = semantic_mapping or {}

        self.temperature = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        if not is_base_training:
            self.temperature.requires_grad = False

        # =====================================================================
        # [Part A] 初始化 SAR-CLIP 模型
        # =====================================================================
        print(f"[Aligner] Loading SAR-CLIP: {sar_clip_name} from {sar_clip_path}")
        full_model = sar_clip.create_model_with_args(sar_clip_name, pretrained=sar_clip_path)
        full_model.eval()
        
        self.clip_visual = full_model.visual
        for p in self.clip_visual.parameters(): p.requires_grad = False
        
        # 🔥 获取 CLIP 真实维度 (解决 768 vs 512 报错)
        tokenizer = sar_clip.get_tokenizer(sar_clip_name, cache_dir=sar_clip_cache)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        temp_model = full_model.to(device)
        with torch.no_grad():
            dummy_emb = temp_model.encode_text(tokenizer(["test"]).to(device))
            self.clip_dim = dummy_emb.shape[-1]
            print(f"[Aligner] Detected SAR-CLIP embedding dim: {self.clip_dim}")
        
        # 🔥 强制更新输入维度
        in_dims['text'] = self.clip_dim
        self.in_dims = in_dims

        # 1. 预计算 SAR 特有 Text Features (6类, 用于 fuse)
        sar_feats_list = []
        with torch.no_grad():
            for cat in sar_all_category:
                prompts = [f"a SAR image of the {cat}.", f"synthetic aperture radar image of {cat}."]
                tokens = tokenizer(prompts).to(device)
                feats = temp_model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                sar_feats_list.append(feats.mean(dim=0))
        sar_text_feats = torch.stack(sar_feats_list)
        sar_text_feats = sar_text_feats / sar_text_feats.norm(dim=-1, keepdim=True)
        self.register_buffer('sar_text_feats', sar_text_feats.cpu()) 
        self.num_sar_fuse_classes = len(sar_all_category)

        # 2. 预计算 全局 Text Features (26类, 用于 Loss)
        if category_list is not None:
            global_feats_list = []
            with torch.no_grad():
                for cat in category_list:
                    prompts = [f"a remote sensing image of {cat}."]
                    tokens = tokenizer(prompts).to(device)
                    feats = temp_model.encode_text(tokens)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    global_feats_list.append(feats.mean(dim=0))
            text_feats = torch.stack(global_feats_list)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            self.register_buffer('text_feats', text_feats.cpu())
        else:
            self.register_buffer('text_feats', torch.zeros(num_classes, self.clip_dim))

        del full_model.transformer, full_model.token_embedding, full_model.ln_final
        torch.cuda.empty_cache()

        # =====================================================================
        # [Part B] 初始化融合模块
        # =====================================================================
        if hasattr(self.clip_visual, 'attnpool'):
            vis_dim = self.clip_visual.attnpool.k_proj.in_features
        elif hasattr(self.clip_visual, 'width'):
            vis_dim = self.clip_visual.width
        else:
            vis_dim = 768

        self.vis_project = nn.Conv2d(vis_dim, self.clip_dim, kernel_size=1, bias=False)
        nn.init.normal_(self.vis_project.weight, std=0.001)
        def make_grads_contiguous(grad):
            return grad.contiguous()
        for param in self.vis_project.parameters():
            if param.requires_grad:
                param.register_hook(make_grads_contiguous)

        self.in_channels_fusion = fpn_channels + self.num_sar_fuse_classes
        
        # 仅使用一个 1x1 卷积将通道数从 (256+6) 降回 256
        self.fusion_conv = nn.Conv2d(self.in_channels_fusion, fpn_channels, kernel_size=1)
        def make_grads_contiguous(grad):
            return grad.contiguous()
        for param in self.fusion_conv.parameters():
            if param.requires_grad:
                param.register_hook(make_grads_contiguous)

        with torch.no_grad():
            # 这里为了稳妥，建议使用极小高斯分布，让网络自己学
            nn.init.normal_(self.fusion_conv.weight, std=0.001)
            nn.init.constant_(self.fusion_conv.bias, 0)

        # =====================================================================
        # [Part C] 初始化对齐模块
        # =====================================================================
        # 此时 in_dims['text'] 已被修正为 512
        self.proj = nn.ModuleDict()
        for m in self.modalities:
            d = in_dims.get(m, align_dim)
            if d == align_dim:
                self.proj[m] = nn.Identity()
            else:
                self.proj[m] = Adapter(d, align_dim)
        
        if not is_base_training and 'text' in self.proj:
            if not isinstance(self.proj['text'], nn.Identity):
                self.proj['text'].eval()
                for p in self.proj['text'].parameters(): p.requires_grad = False

        self.banks = nn.ModuleDict({
            m: PrototypeBank(num_classes=num_classes, dim=align_dim, beta=ema_beta)
            for m in self.modalities
        })

        # =====================================================================
        # [Part D] ITA Loss 映射表初始化
        # =====================================================================
        self.modal_classes = {}     
        self.modal_class_maps = {}  
        
        if modal_class_mapping:
            for modal, global_ids in modal_class_mapping.items():
                if isinstance(global_ids, list):
                    cls_tensor = torch.tensor(global_ids, dtype=torch.long)
                else:
                    cls_tensor = global_ids
                
                self.register_buffer(f'modal_classes_{modal}', cls_tensor)
                self.modal_classes[modal] = f'modal_classes_{modal}'
                
                mapping_tensor = torch.full((self.num_classes,), -1, dtype=torch.long)
                if cls_tensor.numel() > 0:
                    valid_cls = cls_tensor[cls_tensor < self.num_classes]
                    mapping_tensor[valid_cls] = torch.arange(len(valid_cls), dtype=torch.long)
                
                self.register_buffer(f'modal_map_{modal}', mapping_tensor)
                self.modal_class_maps[modal] = f'modal_map_{modal}'

    # ---------------------------------------------------------------------
    # 辅助函数
    # ---------------------------------------------------------------------
    def _extract_clip_resnet(self, x):
        x = self.clip_visual.stem(x)
        x = self.clip_visual.layer1(x)
        x = self.clip_visual.layer2(x)
        x = self.clip_visual.layer3(x)
        return self.clip_visual.layer4(x)

    def _extract_clip_patches(self, x):
        visual = self.clip_visual
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls_token = visual.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token.to(x.dtype), x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)
        x = visual.transformer(x)
        if visual.attn_pool is None: x = visual.ln_post(x)
        return x[:, 1:, :]

    @staticmethod
    def _log_safe_det(mat, eps=1e-6):
        try:
            _, logabsdet = torch.linalg.slogdet(mat + eps * torch.eye(mat.size(0), device=mat.device))
        except:
            return torch.tensor(0.0, device=mat.device)
        return logabsdet

    def _class_volume_log(self, M):
        K = M.size(0)
        G = M @ M.t()
        return 0.5 * self._log_safe_det(G, eps=self.eps)

    @autocast(enabled=False)
    def sinkhorn_loss(self, x, y, epsilon=0.1, niter=3):
        x, y = x.float(), y.float()
        if torch.isnan(x).any() or torch.isnan(y).any(): return x.sum() * 0.0
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        C = torch.clamp(1 - torch.mm(x_norm, y_norm.t()), min=0.0, max=2.0)
        n, m = x.shape[0], y.shape[0]
        mu = torch.ones(n, device=x.device) / n
        nu = torch.ones(m, device=x.device) / m
        K = torch.exp(-C / epsilon)
        u = torch.ones(n, device=x.device)
        for _ in range(niter):
            v = nu / (torch.matmul(K.t(), u) + 1e-8)
            u = mu / (torch.matmul(K, v) + 1e-8)
        return torch.sum(u.detach() * torch.matmul(K * C, v.detach()))

    # ---------------------------------------------------------------------
    # 函数 1: Fuse SAR Features (Detection Backbone/Neck)
    # ---------------------------------------------------------------------
    def fuse_sar_features(self, sar_imgs_normalized, fpn_feats):
        device = sar_imgs_normalized.device
        if next(self.clip_visual.parameters()).device != device:
            self.clip_visual = self.clip_visual.to(device)
        
        mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1,3,1,1)
        std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1,3,1,1)
        sar_imgs = (sar_imgs_normalized * std + mean)
        sar_imgs = F.interpolate(sar_imgs, size=(224, 224), mode='bicubic', align_corners=False)
        sar_imgs = sar_imgs.clamp(0, 255) / 127.5 - 1.0
        if sar_imgs.size(1) == 1: sar_imgs = sar_imgs.repeat(1,3,1,1)

        with torch.no_grad():
            if hasattr(self.clip_visual, 'layer4'):
                feat_map = self._extract_clip_resnet(sar_imgs)
            else:
                patches = self._extract_clip_patches(sar_imgs)
                B, N, C = patches.shape
                grid = int(N**0.5)
                feat_map = patches.permute(0, 2, 1).reshape(B, C, grid, grid).contiguous()

        clip_map = self.vis_project(feat_map)
        clip_map_norm = clip_map / (clip_map.norm(dim=1, keepdim=True) + 1e-6)
        
        prior_kernels = self.sar_text_feats.to(device).view(self.num_sar_fuse_classes, self.clip_dim, 1, 1)
        raw_sim = F.conv2d(clip_map_norm, prior_kernels)
        
        beta, gamma = 1.8, 3.0
        mean_sim = raw_sim.mean(dim=(2,3), keepdim=True)
        prior_map = F.relu(gamma * (raw_sim - beta * mean_sim))
        
        B, N_k, H, W = prior_map.shape
        flat = prior_map.view(B, N_k, -1)
        prior_map = (flat - flat.min(2, keepdim=True)[0]) / (flat.max(2, keepdim=True)[0] - flat.min(2, keepdim=True)[0] + 1e-6)
        prior_map = prior_map.view(B, N_k, H, W)

        new_fpn_feats = []
        for feat in fpn_feats:
            prior_up = F.interpolate(prior_map, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            concat = torch.cat([feat, prior_up], dim=1)
            new_fpn_feats.append(self.fusion_conv(concat)+feat)
            
        return tuple(new_fpn_feats)
    
    # # ---------------------------------------------------------------------
    # # 🔥 修改版: Fuse SAR Features (带可视化调试功能)
    # # ---------------------------------------------------------------------
    # def fuse_sar_features(self, sar_imgs_normalized, fpn_feats, 
    #                       gt_bboxes=None, img_metas=None):
    #     device = sar_imgs_normalized.device
    #     if next(self.clip_visual.parameters()).device != device:
    #         self.clip_visual = self.clip_visual.to(device)
        
    #     # 1. 图像还原 (用于可视化)
    #     mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1,3,1,1)
    #     std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1,3,1,1)
    #     sar_imgs_restored = sar_imgs_normalized * std + mean
        
    #     # CLIP 预处理
    #     sar_imgs_clip = F.interpolate(sar_imgs_restored, size=(224, 224), mode='bicubic', align_corners=False)
    #     sar_imgs_clip = sar_imgs_clip.clamp(0, 255) / 127.5 - 1.0
    #     if sar_imgs_clip.size(1) == 1: sar_imgs_clip = sar_imgs_clip.repeat(1,3,1,1)

    #     # 2. 提取原始视觉特征 (Raw ViT Feature)
    #     with torch.no_grad():
    #         if hasattr(self.clip_visual, 'layer4'):
    #             feat_map = self._extract_clip_resnet(sar_imgs_clip)
    #         else:
    #             patches = self._extract_clip_patches(sar_imgs_clip)
    #             B, N, C = patches.shape
    #             grid = int(N**0.5)
    #             # [B, 768, 14, 14]
    #             feat_map = patches.permute(0, 2, 1).reshape(B, C, grid, grid).contiguous()

    #     # 3. 投影 (Projected Feature) -> [B, 512, 14, 14]
    #     clip_map = self.vis_project(feat_map)
    #     clip_map_norm = clip_map / (clip_map.norm(dim=1, keepdim=True) + 1e-6)
        
    #     # 4. 生成 Prior Map
    #     prior_kernels = self.sar_text_feats.to(device).view(self.num_sar_fuse_classes, self.clip_dim, 1, 1)
    #     raw_sim = F.conv2d(clip_map_norm, prior_kernels)
        
    #     beta, gamma = 1.2, 3.0
    #     mean_sim = raw_sim.mean(dim=(2,3), keepdim=True)
    #     prior_map = F.relu(gamma * (raw_sim - beta * mean_sim))
        
    #     B, N_k, H, W = prior_map.shape
    #     flat = prior_map.view(B, N_k, -1)
    #     prior_map_norm = (flat - flat.min(2, keepdim=True)[0]) / (flat.max(2, keepdim=True)[0] - flat.min(2, keepdim=True)[0] + 1e-6)
    #     prior_map_norm = prior_map_norm.view(B, N_k, H, W)

    #     # =================================================================
    #     # 🕵️‍♂️ 增强版可视化 (Raw vs Projected vs Prior)
    #     # =================================================================
    #     if gt_bboxes is not None:
    #         try:
    #             import cv2
    #             import os
    #             import sys
    #             save_dir = "debug_vis_sar_clip"
    #             os.makedirs(save_dir, exist_ok=True)
                
    #             idx = 0
                
    #             # A. 原图 + GT
    #             img_vis = sar_imgs_restored[idx].permute(1, 2, 0).detach().cpu().numpy()
    #             img_vis = img_vis[:, :, ::-1].copy()
    #             img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)
    #             img_h, img_w = img_vis.shape[:2]
                
    #             img_gt = img_vis.copy()
    #             if idx < len(gt_bboxes):
    #                 bboxes = gt_bboxes[idx]
    #                 if isinstance(bboxes, torch.Tensor): bboxes = bboxes.detach().cpu().numpy()
    #                 for bbox in bboxes:
    #                     if len(bbox) == 5:
    #                         xc, yc, w, h, ag = bbox
    #                         rect = ((xc, yc), (w, h), ag * 180 / np.pi)
    #                         box = cv2.boxPoints(rect); box = np.int0(box)
    #                         cv2.drawContours(img_gt, [box], 0, (0, 255, 0), 2)
    #                     else:
    #                         x1, y1, x2, y2 = bbox[:4].astype(int)
    #                         cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #             # 🔥 B. 原始特征图 (Raw ViT Output) - 768维
    #             # 取 Channel Mean，看网络最原始的注意力
    #             feat_raw_vis = feat_map[idx].mean(dim=0).detach().cpu().numpy()
    #             feat_raw_vis = (feat_raw_vis - feat_raw_vis.min()) / (feat_raw_vis.max() - feat_raw_vis.min() + 1e-6)
    #             feat_raw_vis = cv2.resize((feat_raw_vis * 255).astype(np.uint8), (img_w, img_h))
    #             heatmap_raw = cv2.applyColorMap(feat_raw_vis, cv2.COLORMAP_JET)
                
    #             # 🔥 C. 投影后特征图 (Projected Feature) - 512维
    #             # 如果做了 Zero-Init，刚开始这里应该是全蓝/全黑
    #             feat_proj_vis = clip_map_norm[idx].mean(dim=0).detach().cpu().numpy()
    #             feat_proj_vis = (feat_proj_vis - feat_proj_vis.min()) / (feat_proj_vis.max() - feat_proj_vis.min() + 1e-6)
    #             feat_proj_vis = cv2.resize((feat_proj_vis * 255).astype(np.uint8), (img_w, img_h))
    #             heatmap_proj = cv2.applyColorMap(feat_proj_vis, cv2.COLORMAP_JET)

    #             # D. 最终 Prior Map (Class Activation)
    #             prior_vis_raw = prior_map_norm[idx].max(dim=0)[0].detach().cpu().numpy()
    #             prior_vis = (prior_vis_raw * 255).astype(np.uint8)
    #             prior_vis = cv2.resize(prior_vis, (img_w, img_h))
    #             heatmap_prior = cv2.applyColorMap(prior_vis, cv2.COLORMAP_JET)
                
    #             # 叠加
    #             overlay_raw = cv2.addWeighted(img_vis, 0.5, heatmap_raw, 0.5, 0)
    #             overlay_proj = cv2.addWeighted(img_vis, 0.5, heatmap_proj, 0.5, 0)
    #             overlay_prior = cv2.addWeighted(img_vis, 0.5, heatmap_prior, 0.5, 0)

    #             # 4图并排: GT | Raw Feat | Proj Feat | Prior
    #             concat_img = np.hstack([img_gt, overlay_raw, overlay_proj, overlay_prior])
                
    #             fname = "vis_debug.jpg"
    #             cv2.imwrite(os.path.join(save_dir, fname), concat_img)
    #             print(f"[Aligner] Saved debug visualiation: [GT | RawViT | Projected | PriorMap] to {os.path.join(save_dir, fname)}")
                
    #             print("🛑 [Debug] Visualization done. Stopping.")
    #             sys.exit(0)

    #         except Exception as e:
    #             if isinstance(e, SystemExit): raise e
    #             print(f"[Aligner] Visualization failed: {e}")
    #     # =================================================================

    #     # 5. Fusion ... (保持不变)
    #     new_fpn_feats = []
    #     for feat in fpn_feats:
    #         prior_up = F.interpolate(prior_map_norm, size=feat.shape[-2:], mode='bilinear', align_corners=False)
    #         concat = torch.cat([feat, prior_up], dim=1)
    #         new_fpn_feats.append(self.fusion_conv(concat)+feat)
            
    #     return tuple(new_fpn_feats)
    # ---------------------------------------------------------------------
    # 函数 2: Update Banks (训练时调用，无返回值)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def update_banks(self, batch_protos: dict, batch_labels: dict):
        device = None
        for v in batch_protos.values():
            if isinstance(v, torch.Tensor): device = v.device; break
        if device is None: device = next(self.parameters()).device

        # 1. Update Text Bank (安全更新)
        if hasattr(self, 'text_feats') and 'text' in self.banks:
            raw_text = self.text_feats.to(device)
            # 使用投影后的特征更新 (no_grad)
            if 'text' in self.proj and not isinstance(self.proj['text'], nn.Identity):
                text_proj = self.proj['text'](raw_text)
            else:
                text_proj = raw_text
            
            # 维度截断，防止 mismatch
            update_len = min(self.banks['text'].protos.size(0), text_proj.size(0))
            self.banks['text'].protos.data[:update_len].copy_(text_proj[:update_len])
            self.banks['text'].valid.data[:update_len].fill_(True)

        # 2. Update Visual Banks (带边界检查)
        for m in self.modalities:
            if m == 'text': continue
            
            feats = batch_protos.get(m, None)
            labels = batch_labels.get(m, None)
            
            if feats is None or labels is None or feats.numel() == 0: continue
            
            feats, labels = feats.to(device), labels.to(device)
            
            if m in self.proj and not isinstance(self.proj[m], nn.Identity):
                 if next(self.proj[m].parameters()).device != device:
                    self.proj[m] = self.proj[m].to(device)
            proj = self.proj[m](feats)
            
            # 🔥 严格过滤非法标签，防止 CUDA Crash
            valid_mask = (labels >= 0) & (labels < self.num_classes)
            if valid_mask.any():
                self.banks[m].update(labels[valid_mask], proj[valid_mask])

    # ---------------------------------------------------------------------
    # 函数 3: Compute Align Loss (Volume + Sinkhorn)
    # ---------------------------------------------------------------------
    def compute_align_loss(self, batch_protos: dict):
        device = None
        for v in batch_protos.values():
            if isinstance(v, torch.Tensor): device = v.device; break
        if device is None: device = next(self.parameters()).device

        # 1. 获取当前 Text Features (带梯度)
        text_feats_curr = None
        if hasattr(self, 'text_feats'):
            raw_text = self.text_feats.to(device)
            if 'text' in self.proj and not isinstance(self.proj['text'], nn.Identity):
                if self.is_base_training:
                    text_feats_curr = self.proj['text'](raw_text)
                else:
                    with torch.no_grad():
                        text_feats_curr = self.proj['text'](raw_text)
            else:
                text_feats_curr = raw_text

        # 2. 梯度钩子
        if text_feats_curr is not None and text_feats_curr.requires_grad:
            dummy_hook = text_feats_curr.sum() * 0.0 
        else:
            dummy_hook = torch.tensor(0.0, device=device)

        # 3. Volume Loss (Geometry)
        total_logvol = 0.0
        cnt = 0
        if self.semantic_mapping:
            for _, mapping in self.semantic_mapping.items():
                rows = []
                for m, idx in mapping.items():
                    indices = idx if isinstance(idx, list) else [idx]
                    valid_p = []
                    for i in indices:
                        if i < self.num_classes and self.banks[m].valid[i]:
                            valid_p.append(self.banks[m].protos[i])
                    if not valid_p: continue
                    rows.append(torch.stack(valid_p).mean(0))
                
                if len(rows) >= 2:
                    M = F.normalize(torch.stack(rows), dim=-1)
                    logvol = self._class_volume_log(M)
                    total_logvol += logvol
                    cnt += 1
        
        loss_vol = -self.vol_weight * (total_logvol / cnt) if cnt > 0 else torch.tensor(0.0, device=device)

        # 4. Sinkhorn OT Loss (Bank vs Text)
        loss_ot = torch.tensor(0.0, device=device)
        if text_feats_curr is not None and self.banks['text'].valid.any():
            text_valid = self.banks['text'].valid
            max_len = min(text_valid.size(0), text_feats_curr.size(0))
            
            for m in [x for x in self.modalities if x != 'text']:
                if m in self.banks and self.banks[m].valid.any():
                    vis_valid = self.banks[m].valid
                    # 取交集 mask
                    curr_len = min(max_len, vis_valid.size(0))
                    common = vis_valid[:curr_len] & text_valid[:curr_len]
                    
                    if common.sum() > 1:
                        vis_p = self.banks[m].protos[:curr_len][common]
                        # 🔥 使用带梯度的 text_feats_curr
                        txt_p = text_feats_curr[:curr_len][common]
                        
                        dist = self.sinkhorn_loss(vis_p, txt_p)
                        w = 1.0 if m == 'sar' else 0.5
                        loss_ot += w * dist

        loss_ot += dummy_hook

        return {
            'loss_align_vol': loss_vol,
            'loss_align_ot': 0.1 * loss_ot
        }

    # ---------------------------------------------------------------------
    # 🔥 [DDP 修复版] 函数 4: Compute ITA Loss (内部自闭环修复)
    # ---------------------------------------------------------------------
    def compute_ita_loss(self, feats, labels, modal_name):
        device = feats.device

        # =========================================================
        # 1. 动态生成 Text Features (保持原逻辑)
        # =========================================================
        # 我们必须确保这一步先执行，这样 proj['text'] 才能进入计算图
        if hasattr(self, 'text_feats'):
            raw_text = self.text_feats.to(device)
            if 'text' in self.proj and not isinstance(self.proj['text'], nn.Identity):
                if self.is_base_training:
                    text_feats = self.proj['text'](raw_text)
                else:
                    with torch.no_grad():
                        text_feats = self.proj['text'](raw_text)
            else:
                text_feats = raw_text
        else:
            # 如果连 text_feats 都没有，返回带梯度的 0 也没用，因为没法指向 proj
            return torch.tensor(0.0, device=device, requires_grad=True)

        # =========================================================
        # 🔥 [关键修复] 创建 Dummy Loss 锚点
        # =========================================================
        # 只要 text_feats 需要梯度，这个 dummy_loss 就会连接到 self.proj['text']
        # 它的值为 0，不影响最终 loss 大小，但能骗过 DDP 检查
        if text_feats.requires_grad:
            dummy_loss = text_feats.sum() * 0.0
        else:
            dummy_loss = torch.tensor(0.0, device=device)

        # =========================================================
        # 2. 各种检查 (所有 Early Return 都要带上 dummy_loss)
        # =========================================================
        
        # 检查映射表
        if modal_name not in self.modal_classes:
            # ❌ 原代码: return 0.0
            # ✅ 新代码: 返回 dummy_loss，证明 text 参数被用过
            return dummy_loss 

        modal_classes = getattr(self, self.modal_classes[modal_name])
        class_map = getattr(self, self.modal_class_maps[modal_name])

        # 3. 过滤非法 Label
        valid_mask = (labels >= 0) & (labels < len(class_map))
        if not valid_mask.all():
            feats = feats[valid_mask]
            labels = labels[valid_mask]
        
        if feats.numel() == 0: 
            # ❌ 原代码: return feats.sum() * 0.0
            # ✅ 新代码: 同时连接 视觉分支(feats) 和 文本分支(dummy)
            return feats.sum() * 0.0 + dummy_loss

        # 4. 映射 Label
        modal_labels = class_map[labels]
        valid_map_mask = modal_labels >= 0
        if not valid_map_mask.all():
            feats = feats[valid_map_mask]
            modal_labels = modal_labels[valid_map_mask]
            
        if feats.numel() == 0: 
            # 同上，防止映射后变空
            return feats.sum() * 0.0 + dummy_loss

        # =========================================================
        # 5. 正常计算 Loss
        # =========================================================
        # 取出对应的 Text Feats
        target_text_feats = text_feats[modal_classes]

        # 归一化
        feats = F.normalize(feats, dim=-1)
        target_text_feats = F.normalize(target_text_feats, dim=-1)
        logit_scale = self.temperature.exp().clamp(max=100.0)

        logits = torch.matmul(feats, target_text_feats.T) * logit_scale
        loss = F.cross_entropy(logits, modal_labels)

        # 虽然正常计算的 loss 已经包含了 text_feats 的梯度，
        # 但加上 dummy_loss (值为0) 是最安全的，能保证代码逻辑统一
        return loss + dummy_loss
