"""
针对少样本类别的数据增广操作
基于现有的mmdet和mmrotate实现
"""

import copy
import cv2
import mmcv
import numpy as np
import random
from numpy import random as np_random

from mmdet.datasets.pipelines import PhotoMetricDistortion
from ..builder import ROTATED_PIPELINES


@ROTATED_PIPELINES.register_module()
class ClassBalancedRandomFlip:
    """
    对包含少样本类别的样本提高翻转概率
    
    Args:
        flip_ratio (float): 基础翻转概率
        minority_classes (list): 少样本类别ID列表
        minority_boost (float): 对少样本类别增加的翻转概率
        direction (str): 翻转方向 'horizontal', 'vertical', 'diagonal'
    """
    
    def __init__(self, 
                 flip_ratio=0.5, 
                 minority_classes=None,
                 minority_boost=0.3,
                 direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.minority_classes = minority_classes or []
        self.minority_boost = minority_boost
        self.direction = direction
        
    def __call__(self, results):
        # 检查是否包含少样本类别
        has_minority = False
        if 'gt_labels' in results and self.minority_classes:
            labels = results['gt_labels']
            has_minority = any(label in self.minority_classes for label in labels)
        
        # 动态调整翻转概率
        current_flip_ratio = self.flip_ratio
        if has_minority:
            current_flip_ratio = min(1.0, self.flip_ratio + self.minority_boost)
        
        # 执行翻转
        if np_random.rand() < current_flip_ratio:
            results['flip'] = True
            results['flip_direction'] = self.direction
            
            # 翻转图像
            for key in results.get('img_fields', ['img']):
                results[key] = mmcv.imflip(
                    results[key], direction=self.direction)
            
            # 翻转bbox
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key], 
                                               results['img_shape'],
                                               self.direction)
        else:
            results['flip'] = False
            results['flip_direction'] = None
            
        return results
    
    def bbox_flip(self, bboxes, img_shape, direction):
        """
        翻转bbox，支持水平框（4参数）和旋转框（5参数）
        
        水平框 (HBB): [x1, y1, x2, y2]
        旋转框 (OBB): [cx, cy, w, h, angle] 或其变体
        """
        if len(bboxes) == 0:
            return bboxes
            
        flipped = bboxes.copy()
        h, w = img_shape[:2]
        
        # 检测bbox格式
        if bboxes.shape[-1] == 4:
            # 水平框 [x1, y1, x2, y2]
            if direction == 'horizontal':
                flipped[:, 0] = w - bboxes[:, 2]
                flipped[:, 2] = w - bboxes[:, 0]
            elif direction == 'vertical':
                flipped[:, 1] = h - bboxes[:, 3]
                flipped[:, 3] = h - bboxes[:, 1]
            elif direction == 'diagonal':
                flipped[:, 0] = w - bboxes[:, 2]
                flipped[:, 2] = w - bboxes[:, 0]
                flipped[:, 1] = h - bboxes[:, 3]
                flipped[:, 3] = h - bboxes[:, 1]
        elif bboxes.shape[-1] >= 5:
            # 旋转框 [cx, cy, w, h, angle, ...]
            if direction == 'horizontal':
                flipped[:, 0] = w - bboxes[:, 0] - 1
            elif direction == 'vertical':
                flipped[:, 1] = h - bboxes[:, 1] - 1
            elif direction == 'diagonal':
                flipped[:, 0] = w - bboxes[:, 0] - 1
                flipped[:, 1] = h - bboxes[:, 1] - 1
        else:
            raise ValueError(f"Unsupported bbox format with {bboxes.shape[-1]} parameters")
            
        return flipped
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(flip_ratio={self.flip_ratio}, '
        repr_str += f'minority_classes={self.minority_classes}, '
        repr_str += f'minority_boost={self.minority_boost})'
        return repr_str


@ROTATED_PIPELINES.register_module()
class ClassBalancedRotate:
    """
    对包含少样本类别的样本提高旋转概率
    
    Args:
        rotate_ratio (float): 基础旋转概率
        minority_classes (list): 少样本类别ID列表
        minority_boost (float): 对少样本类别增加的旋转概率
        angles_range (int): 旋转角度范围
    """
    
    def __init__(self,
                 rotate_ratio=0.5,
                 minority_classes=None,
                 minority_boost=0.3,
                 angles_range=180):
        self.rotate_ratio = rotate_ratio
        self.minority_classes = minority_classes or []
        self.minority_boost = minority_boost
        self.angles_range = angles_range
        
    def __call__(self, results):
        # 检查是否包含少样本类别
        has_minority = False
        if 'gt_labels' in results and self.minority_classes:
            labels = results['gt_labels']
            has_minority = any(label in self.minority_classes for label in labels)
        
        # 动态调整旋转概率
        current_rotate_ratio = self.rotate_ratio
        if has_minority:
            current_rotate_ratio = min(1.0, self.rotate_ratio + self.minority_boost)
        
        # 执行旋转
        if np_random.rand() < current_rotate_ratio:
            angle = np_random.uniform(-self.angles_range, self.angles_range)
            results['rotate_angle'] = angle
        else:
            results['rotate_angle'] = 0
            
        return results


@ROTATED_PIPELINES.register_module()
class ClassBalancedPhotoMetricDistortion(PhotoMetricDistortion):
    """
    对包含少样本类别的样本应用更强的色彩增强
    
    Args:
        minority_classes (list): 少样本类别ID列表
        brightness_delta (int): 亮度变化范围
        contrast_range (tuple): 对比度范围
        saturation_range (tuple): 饱和度范围
        hue_delta (int): 色调变化范围
        minority_strength (float): 对少样本类别的增强强度倍数
    """
    
    def __init__(self,
                 minority_classes=None,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 minority_strength=1.5):
        super().__init__(
            brightness_delta=brightness_delta,
            contrast_range=contrast_range,
            saturation_range=saturation_range,
            hue_delta=hue_delta
        )
        self.minority_classes = minority_classes or []
        self.minority_strength = minority_strength
        self.base_brightness_delta = brightness_delta
        self.base_contrast_range = contrast_range
        self.base_saturation_range = saturation_range
        self.base_hue_delta = hue_delta
        
    def __call__(self, results):
        # 检查是否包含少样本类别
        has_minority = False
        if 'gt_labels' in results and self.minority_classes:
            labels = results['gt_labels']
            has_minority = any(label in self.minority_classes for label in labels)
        
        # 对少样本类别使用更强的增强
        if has_minority:
            self.brightness_delta = int(self.base_brightness_delta * self.minority_strength)
            self.contrast_lower, self.contrast_upper = (
                0.5 / self.minority_strength,
                0.5 + (self.base_contrast_range[1] - 0.5) * self.minority_strength
            )
            self.saturation_lower, self.saturation_upper = (
                0.5 / self.minority_strength,
                0.5 + (self.base_saturation_range[1] - 0.5) * self.minority_strength
            )
            self.hue_delta = int(self.base_hue_delta * self.minority_strength)
        else:
            # 恢复基础参数
            self.brightness_delta = self.base_brightness_delta
            self.contrast_lower, self.contrast_upper = self.base_contrast_range
            self.saturation_lower, self.saturation_upper = self.base_saturation_range
            self.hue_delta = self.base_hue_delta
        
        # 调用父类方法执行增强
        return super().__call__(results)


@ROTATED_PIPELINES.register_module()
class RandomCropWithMinority:
    """
    随机裁剪，优先保留少样本类别
    
    Args:
        crop_size (tuple): 裁剪尺寸 (h, w)
        minority_classes (list): 少样本类别ID列表
        minority_prob (float): 对包含少样本类别的样本应用裁剪的概率
        allow_negative_crop (bool): 是否允许裁剪后没有bbox
    """
    
    def __init__(self,
                 crop_size=(600, 600),
                 minority_classes=None,
                 minority_prob=0.5,
                 allow_negative_crop=False):
        self.crop_size = crop_size
        self.minority_classes = minority_classes or []
        self.minority_prob = minority_prob
        self.allow_negative_crop = allow_negative_crop
        
    def __call__(self, results):
        # 检查是否包含少样本类别
        has_minority = False
        minority_indices = []
        if 'gt_labels' in results and self.minority_classes:
            labels = results['gt_labels']
            for idx, label in enumerate(labels):
                if label in self.minority_classes:
                    has_minority = True
                    minority_indices.append(idx)
        
        # 只对包含少样本类别的样本进行裁剪
        if not has_minority or np_random.rand() > self.minority_prob:
            return results
        
        img = results['img']
        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size
        
        # 如果图像小于裁剪尺寸，不裁剪
        if h <= crop_h or w <= crop_w:
            return results
        
        # 尝试找到包含少样本类别目标的裁剪区域
        max_attempts = 10
        for _ in range(max_attempts):
            # 如果有少样本目标，优先在其附近裁剪
            if minority_indices and 'gt_bboxes' in results:
                # 随机选择一个少样本目标
                target_idx = np_random.choice(minority_indices)
                bbox = results['gt_bboxes'][target_idx]
                
                # 获取bbox中心点（根据格式不同）
                if bbox.shape[0] == 4:
                    # 水平框 [x1, y1, x2, y2]
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                else:
                    # 旋转框 [cx, cy, w, h, angle]
                    center_x, center_y = bbox[0], bbox[1]
                
                # 计算裁剪起点，确保目标在裁剪区域内
                x_offset = int(max(0, min(center_x - crop_w // 2, w - crop_w)))
                y_offset = int(max(0, min(center_y - crop_h // 2, h - crop_h)))
            else:
                # 随机裁剪
                x_offset = np_random.randint(0, w - crop_w + 1)
                y_offset = np_random.randint(0, h - crop_h + 1)
            
            # 执行裁剪
            cropped_results = self._crop(results, x_offset, y_offset, crop_w, crop_h)
            
            # 检查是否保留了足够的bbox
            if 'gt_bboxes' in cropped_results and len(cropped_results['gt_bboxes']) > 0:
                return cropped_results
            elif self.allow_negative_crop:
                return cropped_results
        
        # 如果所有尝试都失败，返回原图
        return results
    
    def _crop(self, results, x_offset, y_offset, crop_w, crop_h):
        """
        执行裁剪操作，支持水平框和旋转框
        
        水平框 (HBB): [x1, y1, x2, y2]
        旋转框 (OBB): [cx, cy, w, h, angle]
        """
        results = copy.deepcopy(results)
        
        # 裁剪图像
        img = results['img']
        cropped_img = img[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]
        results['img'] = cropped_img
        results['img_shape'] = cropped_img.shape
        
        # 调整bbox
        if 'gt_bboxes' in results and len(results['gt_bboxes']) > 0:
            bboxes = results['gt_bboxes'].copy()
            labels = results.get('gt_labels', None)
            
            if bboxes.shape[-1] == 4:
                # 水平框 [x1, y1, x2, y2]
                # 平移bbox坐标
                bboxes[:, [0, 2]] -= x_offset  # x1, x2
                bboxes[:, [1, 3]] -= y_offset  # y1, y2
                
                # 裁剪bbox到图像边界
                bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, crop_w)
                bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, crop_h)
                
                # 过滤掉太小的bbox
                valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
                
            elif bboxes.shape[-1] >= 5:
                # 旋转框 [cx, cy, w, h, angle, ...]
                # 平移中心点
                bboxes[:, 0] -= x_offset
                bboxes[:, 1] -= y_offset
                
                # 检查中心点是否在裁剪区域内
                # 对于旋转框，只要中心点在区域内就保留（简化处理）
                valid_inds = (bboxes[:, 0] >= 0) & (bboxes[:, 0] < crop_w) & \
                             (bboxes[:, 1] >= 0) & (bboxes[:, 1] < crop_h)
            else:
                raise ValueError(f"Unsupported bbox format with {bboxes.shape[-1]} parameters")
            
            results['gt_bboxes'] = bboxes[valid_inds]
            if labels is not None:
                results['gt_labels'] = labels[valid_inds]
        
        return results


@ROTATED_PIPELINES.register_module()
class IncreasedMosaicForMinority:
    """
    对包含少样本类别的样本有更高概率应用Mosaic增广
    这是一个wrapper，配合RMosaic使用
    
    Args:
        minority_classes (list): 少样本类别ID列表
        base_prob (float): 基础应用概率
        minority_prob (float): 对少样本类别的应用概率
    """
    
    def __init__(self,
                 minority_classes=None,
                 base_prob=0.3,
                 minority_prob=0.7):
        self.minority_classes = minority_classes or []
        self.base_prob = base_prob
        self.minority_prob = minority_prob
        
    def __call__(self, results):
        # 检查是否包含少样本类别
        has_minority = False
        if 'gt_labels' in results and self.minority_classes:
            labels = results['gt_labels']
            has_minority = any(label in self.minority_classes for label in labels)
        
        # 设置标记，让后续的Mosaic根据这个标记决定是否应用
        current_prob = self.minority_prob if has_minority else self.base_prob
        results['mosaic_prob'] = current_prob
        results['apply_mosaic'] = np_random.rand() < current_prob
        
        return results
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(base_prob={self.base_prob}, '
        repr_str += f'minority_prob={self.minority_prob}, '
        repr_str += f'minority_classes={self.minority_classes})'
        return repr_str

