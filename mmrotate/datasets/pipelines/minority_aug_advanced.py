"""
高级少样本类别增广操作
包含 CopyPaste, Mosaic, MixUp 等需要多图像的增广
"""

import copy
import cv2
import mmcv
import numpy as np
from numpy import random as np_random

from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class MinorityCopyPaste:
    """
    CopyPaste增广：从其他图像复制少样本类别的目标到当前图像
    
    特别针对少样本类别，增加其训练样本多样性
    支持水平框和旋转框
    
    Args:
        minority_classes (list): 少样本类别ID列表
        min_prob (float): 对少样本类别应用的概率
        max_paste_objs (int): 最多粘贴的目标数量
        paste_prob_per_obj (float): 每个少样本目标被粘贴的概率
        iou_thresh (float): IoU阈值，避免粘贴的目标重叠过多
        scale_range (tuple): 粘贴目标的缩放范围
    """
    
    def __init__(self,
                 minority_classes=None,
                 min_prob=0.5,
                 max_paste_objs=5,
                 paste_prob_per_obj=0.8,
                 iou_thresh=0.3,
                 scale_range=(0.8, 1.2)):
        self.minority_classes = minority_classes or []
        self.min_prob = min_prob
        self.max_paste_objs = max_paste_objs
        self.paste_prob_per_obj = paste_prob_per_obj
        self.iou_thresh = iou_thresh
        self.scale_range = scale_range
        
        # 用于缓存少样本类别的目标
        self.minority_cache = []
        self.cache_size = 100
        
    def __call__(self, results):
        # 检查是否包含少样本类别
        has_minority = False
        if 'gt_labels' in results and self.minority_classes:
            labels = results['gt_labels']
            has_minority = any(label in self.minority_classes for label in labels)
        
        # 缓存当前图像中的少样本类别目标
        self._cache_minority_objects(results)
        
        # 决定是否应用CopyPaste
        if np_random.rand() > self.min_prob or len(self.minority_cache) == 0:
            return results
        
        # 执行CopyPaste
        return self._copy_paste(results)
    
    def _cache_minority_objects(self, results):
        """缓存少样本类别的目标"""
        if 'gt_labels' not in results or 'gt_bboxes' not in results:
            return
        
        labels = results['gt_labels']
        bboxes = results['gt_bboxes']
        img = results['img']
        
        for idx, label in enumerate(labels):
            if label in self.minority_classes:
                # 提取目标的图像patch和bbox
                bbox = bboxes[idx]
                obj_info = {
                    'label': label,
                    'bbox': bbox.copy(),
                    'img': img.copy(),
                    'img_shape': results['img_shape']
                }
                
                self.minority_cache.append(obj_info)
                
                # 限制缓存大小
                if len(self.minority_cache) > self.cache_size:
                    self.minority_cache.pop(0)
    
    def _copy_paste(self, results):
        """执行CopyPaste操作"""
        if len(self.minority_cache) == 0:
            return results
        
        img = results['img']
        h, w = img.shape[:2]
        bboxes = results.get('gt_bboxes', np.array([])).copy()
        labels = results.get('gt_labels', np.array([])).copy()
        
        # 随机选择要粘贴的目标数量
        num_paste = min(
            np_random.randint(1, self.max_paste_objs + 1),
            len(self.minority_cache)
        )
        
        new_bboxes = []
        new_labels = []
        
        for _ in range(num_paste):
            if np_random.rand() > self.paste_prob_per_obj:
                continue
            
            # 随机选择一个缓存的少样本目标
            obj_info = self.minority_cache[np_random.randint(len(self.minority_cache))]
            
            # 提取目标patch
            paste_result = self._extract_and_paste_object(
                img, obj_info, bboxes, h, w
            )
            
            if paste_result is not None:
                img, new_bbox = paste_result
                new_bboxes.append(new_bbox)
                new_labels.append(obj_info['label'])
        
        # 更新results
        if len(new_bboxes) > 0:
            results['img'] = img
            if len(bboxes) > 0:
                results['gt_bboxes'] = np.vstack([bboxes, np.array(new_bboxes)])
                results['gt_labels'] = np.concatenate([labels, np.array(new_labels)])
            else:
                results['gt_bboxes'] = np.array(new_bboxes)
                results['gt_labels'] = np.array(new_labels)
        
        return results
    
    def _extract_and_paste_object(self, img, obj_info, existing_bboxes, h, w):
        """
        提取目标并粘贴到新位置
        支持水平框和旋转框
        """
        src_img = obj_info['img']
        src_bbox = obj_info['bbox']
        
        is_hbb = src_bbox.shape[0] == 4  # 是否为水平框
        
        # 根据bbox格式提取目标区域
        if is_hbb:
            # 水平框 [x1, y1, x2, y2]
            x1, y1, x2, y2 = np.clip(src_bbox.astype(int), 0, [src_img.shape[1], src_img.shape[0], src_img.shape[1], src_img.shape[0]])
            obj_patch = src_img[y1:y2, x1:x2].copy()
            obj_h, obj_w = y2 - y1, x2 - x1
            original_angle = 0
        else:
            # 旋转框 [cx, cy, w, h, angle]
            cx, cy, box_w, box_h = src_bbox[:4]
            original_angle = src_bbox[4] if len(src_bbox) > 4 else 0
            
            # 使用getRotationMatrix2D提取旋转目标（保持角度信息）
            # 但为了简化，我们提取外接矩形，保留角度信息用于后续粘贴
            half_w, half_h = int(box_w // 2), int(box_h // 2)
            x1 = max(0, int(cx - half_w))
            y1 = max(0, int(cy - half_h))
            x2 = min(src_img.shape[1], int(cx + half_w))
            y2 = min(src_img.shape[0], int(cy + half_h))
            
            obj_patch = src_img[y1:y2, x1:x2].copy()
            obj_h, obj_w = y2 - y1, x2 - x1
        
        if obj_h < 5 or obj_w < 5:  # 目标太小
            return None
        
        # 随机缩放
        scale = np_random.uniform(*self.scale_range)
        new_h, new_w = int(obj_h * scale), int(obj_w * scale)
        
        if new_h > h * 0.5 or new_w > w * 0.5:  # 目标太大
            return None
        
        if new_h < 5 or new_w < 5:  # 缩放后太小
            return None
        
        obj_patch = cv2.resize(obj_patch, (new_w, new_h))
        
        # 随机选择粘贴位置（避免重叠）
        max_attempts = 50
        for _ in range(max_attempts):
            paste_x = np_random.randint(0, max(1, w - new_w))
            paste_y = np_random.randint(0, max(1, h - new_h))
            
            # 创建新bbox（根据原始格式）
            if is_hbb:
                # 水平框: [x1, y1, x2, y2]
                new_bbox = np.array([
                    paste_x, 
                    paste_y, 
                    paste_x + new_w, 
                    paste_y + new_h
                ], dtype=np.float32)
            else:
                # 旋转框: [cx, cy, w, h, angle]
                # 保持原始角度，只改变位置和尺寸
                new_bbox = np.array([
                    paste_x + new_w / 2.0,      # 新中心x
                    paste_y + new_h / 2.0,      # 新中心y
                    new_w,                       # 缩放后的宽度
                    new_h,                       # 缩放后的高度
                    original_angle               # 保持原始角度
                ], dtype=np.float32)
            
            # 检查是否重叠
            if len(existing_bboxes) == 0 or not self._has_high_overlap(new_bbox, existing_bboxes):
                # 粘贴目标（使用alpha混合以避免硬边缘）
                try:
                    # 简单粘贴（可以改进为alpha混合）
                    img[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = obj_patch
                except:
                    continue
                    
                return img, new_bbox
        
        return None
    
    def _has_high_overlap(self, new_bbox, existing_bboxes):
        """检查是否与现有bbox重叠过多"""
        # 简化的IoU检查（只检查中心点区域）
        if new_bbox.shape[0] == 4:
            cx = (new_bbox[0] + new_bbox[2]) / 2
            cy = (new_bbox[1] + new_bbox[3]) / 2
        else:
            cx, cy = new_bbox[0], new_bbox[1]
        
        for bbox in existing_bboxes:
            if bbox.shape[0] == 4:
                bx1, by1, bx2, by2 = bbox
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    return True
            else:
                bcx, bcy, bw, bh = bbox[:4]
                if abs(cx - bcx) < bw / 2 and abs(cy - bcy) < bh / 2:
                    return True
        
        return False
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(minority_classes={self.minority_classes}, '
        repr_str += f'min_prob={self.min_prob})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class MinorityMosaic:
    """
    Mosaic增广：优先使用包含少样本类别的图像
    
    将4张图像拼接成1张，如果图像包含少样本类别，提高其被选中的概率
    支持水平框和旋转框
    
    Args:
        minority_classes (list): 少样本类别ID列表
        img_scale (tuple): 输出图像尺寸 (h, w)
        center_ratio_range (tuple): 中心点位置范围
        min_bbox_size (int): 最小bbox尺寸
        min_prob (float): 对少样本类别应用的概率
        minority_boost (float): 包含少样本类别的图像被选中的概率提升
        pad_val (int): 填充值
    """
    
    def __init__(self,
                 minority_classes=None,
                 img_scale=(1024, 1024),
                 center_ratio_range=(0.5, 1.5),
                 min_bbox_size=5,
                 min_prob=0.3,
                 minority_boost=0.4,
                 pad_val=114):
        self.minority_classes = minority_classes or []
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.min_bbox_size = min_bbox_size
        self.min_prob = min_prob
        self.minority_boost = minority_boost
        self.pad_val = pad_val
        
        # 缓存包含少样本类别的图像
        self.minority_image_cache = []
        self.cache_size = 50
    
    def __call__(self, results):
        # 缓存当前图像
        self._cache_image_if_minority(results)
        
        # 检查是否应用Mosaic
        has_minority = self._has_minority(results)
        apply_prob = self.min_prob + (self.minority_boost if has_minority else 0)
        
        if np_random.rand() > apply_prob or len(self.minority_image_cache) < 3:
            return results
        
        # 执行Mosaic
        return self._mosaic(results)
    
    def _has_minority(self, results):
        """检查是否包含少样本类别"""
        if 'gt_labels' in results and self.minority_classes:
            labels = results['gt_labels']
            return any(label in self.minority_classes for label in labels)
        return False
    
    def _cache_image_if_minority(self, results):
        """缓存包含少样本类别的图像"""
        if not self._has_minority(results):
            return
        
        # 深拷贝results
        cached_results = {
            'img': results['img'].copy(),
            'img_shape': results['img_shape'],
            'gt_bboxes': results.get('gt_bboxes', np.array([])).copy(),
            'gt_labels': results.get('gt_labels', np.array([])).copy(),
        }
        
        self.minority_image_cache.append(cached_results)
        
        # 限制缓存大小
        if len(self.minority_image_cache) > self.cache_size:
            self.minority_image_cache.pop(0)
    
    def _mosaic(self, results):
        """执行Mosaic拼接"""
        # 确保有足够的缓存图像
        if len(self.minority_image_cache) < 3:
            return results
        
        # 随机选择另外3张图像
        indices = np_random.choice(len(self.minority_image_cache), 3, replace=False)
        mosaic_images = [results] + [self.minority_image_cache[i] for i in indices]
        
        # 创建mosaic画布
        h, w = self.img_scale
        mosaic_img = np.full((h, w, 3), self.pad_val, dtype=np.uint8)
        
        # 计算中心点
        center_x = int(np_random.uniform(*self.center_ratio_range) * w / 2)
        center_y = int(np_random.uniform(*self.center_ratio_range) * h / 2)
        
        # 放置位置：左上、右上、左下、右下
        placements = [
            (0, 0, center_x, center_y),           # 左上
            (center_x, 0, w, center_y),           # 右上
            (0, center_y, center_x, h),           # 左下
            (center_x, center_y, w, h),           # 右下
        ]
        
        all_bboxes = []
        all_labels = []
        
        for img_data, (x1, y1, x2, y2) in zip(mosaic_images, placements):
            img = img_data['img']
            img_h, img_w = img.shape[:2]
            
            # 计算放置区域尺寸
            place_h, place_w = y2 - y1, x2 - x1
            
            # Resize图像以适应放置区域
            if img_h > 0 and img_w > 0:
                scale = min(place_h / img_h, place_w / img_w)
                new_h, new_w = int(img_h * scale), int(img_w * scale)
                img_resized = cv2.resize(img, (new_w, new_h))
                
                # 放置图像
                mosaic_img[y1:y1+new_h, x1:x1+new_w] = img_resized
                
                # 调整bbox
                if 'gt_bboxes' in img_data and len(img_data['gt_bboxes']) > 0:
                    bboxes = img_data['gt_bboxes'].copy()
                    labels = img_data['gt_labels'].copy()
                    
                    # 缩放和平移bbox（根据格式）
                    if bboxes.shape[-1] == 4:
                        # 水平框 [x1, y1, x2, y2]
                        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + x1  # x坐标
                        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + y1  # y坐标
                    elif bboxes.shape[-1] >= 5:
                        # 旋转框 [cx, cy, w, h, angle, ...]
                        bboxes[:, 0] = bboxes[:, 0] * scale + x1    # 中心x
                        bboxes[:, 1] = bboxes[:, 1] * scale + y1    # 中心y
                        bboxes[:, 2] = bboxes[:, 2] * scale         # 宽度
                        bboxes[:, 3] = bboxes[:, 3] * scale         # 高度
                        # angle保持不变 (bboxes[:, 4])
                    
                    # 过滤掉超出画布的bbox
                    valid_mask = np.ones(len(bboxes), dtype=bool)
                    if bboxes.shape[-1] == 4:
                        # 水平框：检查是否在画布内
                        valid_mask = (bboxes[:, 0] < w) & (bboxes[:, 1] < h) & \
                                   (bboxes[:, 2] > 0) & (bboxes[:, 3] > 0)
                    else:
                        # 旋转框：检查中心点是否在画布内
                        valid_mask = (bboxes[:, 0] >= 0) & (bboxes[:, 0] < w) & \
                                   (bboxes[:, 1] >= 0) & (bboxes[:, 1] < h)
                    
                    if valid_mask.sum() > 0:
                        all_bboxes.append(bboxes[valid_mask])
                        all_labels.append(labels[valid_mask])
        
        # 合并所有bbox
        if len(all_bboxes) > 0:
            results['gt_bboxes'] = np.vstack(all_bboxes)
            results['gt_labels'] = np.concatenate(all_labels)
        else:
            results['gt_bboxes'] = np.array([])
            results['gt_labels'] = np.array([])
        
        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'min_prob={self.min_prob})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class MinorityMixUp:
    """
    MixUp增广：混合包含少样本类别的两张图像
    
    Args:
        minority_classes (list): 少样本类别ID列表
        min_prob (float): 对少样本类别应用的概率
        alpha (float): Beta分布参数
        beta (float): Beta分布参数
    """
    
    def __init__(self,
                 minority_classes=None,
                 min_prob=0.3,
                 alpha=1.5,
                 beta=1.5):
        self.minority_classes = minority_classes or []
        self.min_prob = min_prob
        self.alpha = alpha
        self.beta = beta
        
        # 缓存包含少样本类别的图像
        self.minority_image_cache = []
        self.cache_size = 50
    
    def __call__(self, results):
        # 缓存当前图像
        if self._has_minority(results):
            self._cache_image(results)
        
        # 决定是否应用MixUp
        has_minority = self._has_minority(results)
        if np_random.rand() > self.min_prob or not has_minority or len(self.minority_image_cache) < 2:
            return results
        
        # 执行MixUp
        return self._mixup(results)
    
    def _has_minority(self, results):
        """检查是否包含少样本类别"""
        if 'gt_labels' in results and self.minority_classes:
            labels = results['gt_labels']
            return any(label in self.minority_classes for label in labels)
        return False
    
    def _cache_image(self, results):
        """缓存图像"""
        cached_results = {
            'img': results['img'].copy(),
            'img_shape': results['img_shape'],
            'gt_bboxes': results.get('gt_bboxes', np.array([])).copy(),
            'gt_labels': results.get('gt_labels', np.array([])).copy(),
        }
        
        self.minority_image_cache.append(cached_results)
        
        if len(self.minority_image_cache) > self.cache_size:
            self.minority_image_cache.pop(0)
    
    def _mixup(self, results):
        """
        执行MixUp - 混合两张图像及其bbox
        支持水平框和旋转框
        """
        # 随机选择另一张图像
        mix_img_data = self.minority_image_cache[np_random.randint(len(self.minority_image_cache))]
        
        # 生成混合系数
        lam = np.random.beta(self.alpha, self.beta)
        
        # 确保两张图像尺寸相同
        img1 = results['img']
        img2 = mix_img_data['img']
        
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # 同时调整img2的bbox
            if 'gt_bboxes' in mix_img_data and len(mix_img_data['gt_bboxes']) > 0:
                scale_x = img1.shape[1] / mix_img_data['img_shape'][1]
                scale_y = img1.shape[0] / mix_img_data['img_shape'][0]
                
                bboxes2 = mix_img_data['gt_bboxes'].copy()
                if bboxes2.shape[-1] == 4:
                    # 水平框 [x1, y1, x2, y2]
                    bboxes2[:, [0, 2]] *= scale_x  # x坐标
                    bboxes2[:, [1, 3]] *= scale_y  # y坐标
                elif bboxes2.shape[-1] >= 5:
                    # 旋转框 [cx, cy, w, h, angle]
                    bboxes2[:, 0] *= scale_x       # 中心x
                    bboxes2[:, 1] *= scale_y       # 中心y
                    bboxes2[:, 2] *= scale_x       # 宽度
                    bboxes2[:, 3] *= scale_y       # 高度
                    # angle保持不变
                
                mix_img_data['gt_bboxes'] = bboxes2
        
        # 混合图像
        mixed_img = (img1 * lam + img2 * (1 - lam)).astype(np.uint8)
        
        # 合并bbox和labels
        bboxes1 = results.get('gt_bboxes', np.array([]))
        labels1 = results.get('gt_labels', np.array([]))
        bboxes2 = mix_img_data.get('gt_bboxes', np.array([]))
        labels2 = mix_img_data.get('gt_labels', np.array([]))
        
        # 合并bbox（确保格式兼容）
        all_bboxes = []
        all_labels = []
        
        if len(bboxes1) > 0:
            all_bboxes.append(bboxes1)
            all_labels.append(labels1)
        
        if len(bboxes2) > 0:
            # 检查bbox格式是否兼容
            if len(bboxes1) == 0 or bboxes1.shape[-1] == bboxes2.shape[-1]:
                all_bboxes.append(bboxes2)
                all_labels.append(labels2)
            # 如果格式不兼容，跳过bboxes2（避免混合水平框和旋转框）
        
        if len(all_bboxes) > 0:
            results['gt_bboxes'] = np.vstack(all_bboxes)
            results['gt_labels'] = np.concatenate(all_labels)
        else:
            results['gt_bboxes'] = np.array([])
            results['gt_labels'] = np.array([])
        
        results['img'] = mixed_img
        results['img_shape'] = mixed_img.shape
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_prob={self.min_prob}, '
        repr_str += f'alpha={self.alpha}, beta={self.beta})'
        return repr_str

