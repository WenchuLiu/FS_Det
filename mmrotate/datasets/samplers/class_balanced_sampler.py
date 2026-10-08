"""
类别平衡采样器
对少样本类别进行过采样
"""

import math
import numpy as np
import torch
from torch.utils.data import Sampler
from mmdet.datasets import DATASETS
from typing import Iterator, List, Optional


class ClassBalancedSampler(Sampler):
    """
    类别平衡采样器，对少样本类别进行重复采样
    
    参考论文: 
    - LVIS: A Dataset for Large Vocabulary Instance Segmentation (https://arxiv.org/abs/1908.03195)
    - Class-Balanced Loss Based on Effective Number of Samples (https://arxiv.org/abs/1901.05555)
    
    Args:
        dataset: 数据集对象
        minority_classes (list): 少样本类别ID列表
        oversample_thr (float): 类别频率阈值，低于此值的类别会被过采样
        oversample_ratio (float): 过采样比例，1.0表示不过采样
        shuffle (bool): 是否打乱顺序
        seed (int): 随机种子
    
    Example:
        >>> minority_classes = [20, 21, 22, 23, 24, 25]
        >>> sampler = ClassBalancedSampler(
        >>>     dataset=train_dataset,
        >>>     minority_classes=minority_classes,
        >>>     oversample_thr=0.001,
        >>>     oversample_ratio=2.0,
        >>>     shuffle=True
        >>> )
        >>> dataloader = DataLoader(dataset, sampler=sampler, ...)
    """
    
    def __init__(self,
                 dataset,
                 minority_classes: Optional[List[int]] = None,
                 oversample_thr: float = 0.0,
                 oversample_ratio: float = 2.0,
                 shuffle: bool = True,
                 seed: int = 0):
        self.dataset = dataset
        self.minority_classes = minority_classes or []
        self.oversample_thr = oversample_thr
        self.oversample_ratio = oversample_ratio
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # 统计每个样本包含的类别
        self.sample_indices = self._get_sample_indices()
        
    def _get_sample_indices(self):
        """
        获取采样索引，对包含少样本类别的样本进行重复采样
        """
        # 统计每个类别的样本数
        class_sample_count = {}
        sample_to_classes = {}  # 记录每个样本包含哪些类别
        
        for idx in range(len(self.dataset)):
            # 获取该样本的类别
            data_info = self.dataset.get_ann_info(idx)
            labels = data_info.get('labels', [])
            
            sample_to_classes[idx] = set(labels)
            
            for label in labels:
                if label not in class_sample_count:
                    class_sample_count[label] = 0
                class_sample_count[label] += 1
        
        # 计算每个类别的频率
        total_samples = len(self.dataset)
        class_freq = {k: v / total_samples for k, v in class_sample_count.items()}
        
        # 确定哪些是少样本类别
        if not self.minority_classes:
            # 如果没有指定，自动识别低于阈值的类别
            self.minority_classes = [
                cls_id for cls_id, freq in class_freq.items() 
                if freq < self.oversample_thr
            ]
        
        print(f"ClassBalancedSampler: 识别到 {len(self.minority_classes)} 个少样本类别")
        print(f"少样本类别: {self.minority_classes}")
        
        # 构建采样索引
        indices = []
        for idx in range(len(self.dataset)):
            # 基础采样：每个样本至少采样一次
            indices.append(idx)
            
            # 检查是否包含少样本类别
            sample_classes = sample_to_classes[idx]
            has_minority = any(cls in self.minority_classes for cls in sample_classes)
            
            if has_minority:
                # 对包含少样本类别的样本进行额外采样
                repeat_times = int(self.oversample_ratio - 1)  # -1因为已经采样了一次
                for _ in range(repeat_times):
                    indices.append(idx)
        
        print(f"ClassBalancedSampler: 原始样本数 {total_samples}, 过采样后 {len(indices)}")
        
        return indices
    
    def __iter__(self) -> Iterator[int]:
        """返回采样索引的迭代器"""
        # 设置随机种子
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.sample_indices), generator=g).tolist()
            return iter([self.sample_indices[i] for i in indices])
        else:
            return iter(self.sample_indices)
    
    def __len__(self) -> int:
        """返回采样后的样本总数"""
        return len(self.sample_indices)
    
    def set_epoch(self, epoch: int):
        """设置epoch，用于分布式训练"""
        self.epoch = epoch


class RepeatFactorSampler(Sampler):
    """
    重复因子采样器（Repeat Factor Sampling）
    
    基于LVIS论文的采样策略，根据类别频率计算重复因子
    
    Args:
        dataset: 数据集对象
        repeat_thr (float): 重复因子阈值 t，论文中推荐0.001
        shuffle (bool): 是否打乱
        seed (int): 随机种子
    
    Reference:
        LVIS: A Dataset for Large Vocabulary Instance Segmentation
        https://arxiv.org/abs/1908.03195
    """
    
    def __init__(self,
                 dataset,
                 repeat_thr: float = 0.001,
                 shuffle: bool = True,
                 seed: int = 0):
        self.dataset = dataset
        self.repeat_thr = repeat_thr
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # 计算重复因子
        self.repeat_factors = self._get_repeat_factors()
        
    def _get_repeat_factors(self):
        """
        计算每个样本的重复因子
        
        公式: r_j = max(1, sqrt(t / f_j))
        其中 f_j 是样本j中最稀有类别的频率
        """
        # 统计类别频率
        category_freq = {}
        num_images = len(self.dataset)
        
        for idx in range(num_images):
            data_info = self.dataset.get_ann_info(idx)
            labels = data_info.get('labels', [])
            
            for label in labels:
                category_freq[label] = category_freq.get(label, 0) + 1
        
        # 转换为频率
        for k in category_freq:
            category_freq[k] /= num_images
        
        # 计算每个样本的重复因子
        repeat_factors = []
        for idx in range(num_images):
            data_info = self.dataset.get_ann_info(idx)
            labels = data_info.get('labels', [])
            
            if len(labels) == 0:
                repeat_factor = 1.0
            else:
                # 找到最稀有的类别
                min_freq = min(category_freq.get(label, 1.0) for label in labels)
                # 计算重复因子: r = max(1, sqrt(t / f))
                repeat_factor = max(1.0, math.sqrt(self.repeat_thr / min_freq))
            
            repeat_factors.append(repeat_factor)
        
        return repeat_factors
    
    def _get_sampling_indices(self):
        """根据重复因子生成采样索引"""
        indices = []
        for idx, repeat_factor in enumerate(self.repeat_factors):
            # 整数部分：必定重复的次数
            repeat_times = int(repeat_factor)
            indices.extend([idx] * repeat_times)
            
            # 小数部分：以概率重复
            fractional_part = repeat_factor - repeat_times
            if np.random.rand() < fractional_part:
                indices.append(idx)
        
        return indices
    
    def __iter__(self) -> Iterator[int]:
        """返回采样索引的迭代器"""
        # 设置随机种子
        np.random.seed(self.seed + self.epoch)
        
        # 生成采样索引
        indices = self._get_sampling_indices()
        
        # 打乱
        if self.shuffle:
            np.random.shuffle(indices)
        
        return iter(indices)
    
    def __len__(self) -> int:
        """返回期望的样本数"""
        return int(sum(self.repeat_factors))
    
    def set_epoch(self, epoch: int):
        """设置epoch"""
        self.epoch = epoch

