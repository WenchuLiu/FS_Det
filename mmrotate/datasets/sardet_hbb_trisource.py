from .ordered_coco import OrderedCocoLabels, ordered_category_ids
import os

from mmdet.datasets import CocoDataset 

from .builder import ROTATED_DATASETS

from pycocotools.coco import COCO
import mmcv
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import numpy as np
from typing import Set, Dict, List, Tuple, Optional
from mmcv.utils import print_log
import torch
from mmcv.ops import nms
import contextlib   # 用于 contextlib.redirect_stdout (重定向输出流)
import io


SARDET100K_ALL_CLASSES = (
    'ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'
)

SARDET100K_SPLIT_NOVEL_CLASSES = {
    1: ('aircraft',),
    2: ('ship',),
}


def get_sardet100k_split_classes(split_id, use_base_only=False, include_novel=True):
    if split_id not in SARDET100K_SPLIT_NOVEL_CLASSES:
        raise KeyError(
            f'Invalid SARDet100K split_id={split_id}. '
            f'Available splits: {tuple(sorted(SARDET100K_SPLIT_NOVEL_CLASSES))}')

    novel_classes = SARDET100K_SPLIT_NOVEL_CLASSES[split_id]
    base_classes = tuple(
        class_name for class_name in SARDET100K_ALL_CLASSES
        if class_name not in novel_classes)

    if use_base_only:
        return base_classes
    if include_novel:
        return base_classes + novel_classes
    return SARDET100K_ALL_CLASSES


@ROTATED_DATASETS.register_module()
class SARDet_hbb_trisource(OrderedCocoLabels, CocoDataset):
    """SAR Detection dataset with horizontal bounding boxes from three sources.
    
    This dataset supports custom evaluation metrics including detection rate,
    recognition rate, and false alarm rate.
    """

    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN')
    

    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (255, 193, 193), (0, 51, 153)]
     
    def __init__(self, 
                 *args, 
                 other_iou_threshold: float = 0.3, 
                 other_score_threshold: float = 0.3, 
                 other_nms_threshold: float = 0.6, 
                 **kwargs):
        """Initialize the dataset.
        
        Args:
            other_iou_threshold: IoU threshold for custom evaluation metrics.
            other_score_threshold: Score threshold for filtering predictions.
            other_nms_threshold: IoU threshold for NMS.
            *args, **kwargs: Arguments passed to parent class.
        """
        super().__init__(*args, **kwargs)
        self.other_iou_threshold = other_iou_threshold
        self.other_score_threshold = other_score_threshold
        self.other_nms_threshold = other_nms_threshold

    def _convert_bbox_format(self, bboxes: List) -> List:
        """Convert bbox format from [x, y, w, h] to [x1, y1, x2, y2].
        
        Args:
            bboxes: List of bounding boxes in [x, y, w, h] format.
            
        Returns:
            List of bounding boxes in [x1, y1, x2, y2] format.
        """
        converted_boxes = []
        for bbox in bboxes:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            converted_boxes.append([x1, y1, x2, y2])
        return converted_boxes

    def _calculate_iou_matrix(self, pred_boxes: List, gt_boxes: List) -> np.ndarray:
        """Calculate IoU matrix between predictions and ground truth boxes.
        
        Args:
            pred_boxes: List of prediction bounding boxes in [x, y, w, h] format.
            gt_boxes: List of ground truth bounding boxes in [x, y, w, h] format.
            
        Returns:
            IoU matrix of shape (num_predictions, num_ground_truth).
        """
        if not pred_boxes or not gt_boxes:
            return np.array([])
        
        # Convert bbox format from [x, y, w, h] to [x1, y1, x2, y2]
        pred_boxes_xyxy = self._convert_bbox_format(pred_boxes)
        gt_boxes_xyxy = self._convert_bbox_format(gt_boxes)
        
        # Convert to numpy arrays for vectorized computation
        pred_boxes_array = np.array(pred_boxes_xyxy)
        gt_boxes_array = np.array(gt_boxes_xyxy)
        
        # Ensure arrays are 2D
        if pred_boxes_array.ndim == 1:
            pred_boxes_array = pred_boxes_array.reshape(1, -1)
        if gt_boxes_array.ndim == 1:
            gt_boxes_array = gt_boxes_array.reshape(1, -1)
        
        # Calculate IoU matrix using bbox_overlaps
        iou_matrix = bbox_overlaps(pred_boxes_array, gt_boxes_array)
        return iou_matrix

    def _evaluate_single_image(self, pred_anns: List, gt_anns: List) -> Tuple[int, int]:
        """Evaluate detection and classification accuracy for a single image.
        
        Args:
            pred_anns: List of prediction annotations.
            gt_anns: List of ground truth annotations.
            
        Returns:
            Tuple of (correct_detections, correct_classifications).
        """
        if not pred_anns or not gt_anns:
            return 0, 0
        
        # Sort predictions by confidence score
        pred_anns = sorted(pred_anns, key=lambda x: x["score"], reverse=True)
        
        # Extract bounding boxes
        pred_boxes = [ann["bbox"] for ann in pred_anns]
        gt_boxes = [ann["bbox"] for ann in gt_anns]
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(pred_boxes, gt_boxes)
        
        # Handle empty IoU matrix case
        if iou_matrix.size == 0:
            return 0, 0
        
        # Track matched ground truth boxes
        gt_matched = [False] * len(gt_anns)
        correct_detections = 0
        correct_classifications = 0
        
        # Match predictions to ground truth using greedy algorithm
        for pred_idx, pred_ann in enumerate(pred_anns):
            # Find best matching ground truth that hasn't been matched yet
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_ann in enumerate(gt_anns):
                if not gt_matched[gt_idx]:
                    iou = iou_matrix[pred_idx, gt_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            # Check if IoU meets threshold and mark as matched
            if best_iou >= self.other_iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                correct_detections += 1
                
                # Check classification accuracy
                if pred_ann["category_id"] == gt_anns[best_gt_idx]["category_id"]:
                    correct_classifications += 1
        
        return correct_detections, correct_classifications

    def _apply_score_filter_and_nms(self, pred_anns: List) -> List:
        """Apply score threshold filtering and NMS to predictions.
        
        Args:
            pred_anns: List of prediction annotations.
            
        Returns:
            Filtered and NMS-processed prediction annotations.
        """
        if not pred_anns:
            return []
        
        # Filter by score threshold
        filtered_anns = [ann for ann in pred_anns if ann["score"] >= self.other_score_threshold]
        
        if not filtered_anns:
            return []
        
        # Group by category_id for per-class NMS
        category_groups = {}
        for ann in filtered_anns:
            cat_id = ann["category_id"]
            if cat_id not in category_groups:
                category_groups[cat_id] = []
            category_groups[cat_id].append(ann)
        
        # Apply NMS for each category
        nms_results = []
        for cat_id, cat_anns in category_groups.items():
            if len(cat_anns) == 1:
                nms_results.append(cat_anns[0])
                continue
            
            # Convert to tensor format for NMS
            bboxes = []
            scores = []
            for ann in cat_anns:
                bbox = ann["bbox"]  # [x, y, w, h]
                # Convert to [x1, y1, x2, y2] format for NMS
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
                bboxes.append([x1, y1, x2, y2])
                scores.append(ann["score"])
            
            bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            
            # Apply NMS
            _, keep_indices = nms(bboxes_tensor, scores_tensor, self.other_nms_threshold)
            
            # Convert tensor indices to Python indices and keep the filtered annotations
            if isinstance(keep_indices, torch.Tensor):
                keep_indices = keep_indices.cpu().numpy()
            for idx in keep_indices:
                nms_results.append(cat_anns[idx])
        
        return nms_results

    def _evaluate_group_metrics(self, 
                                pred_anns: List, 
                                gt_anns: List, 
                                target_cat_ids: Set[int]) -> Tuple[int, int, int, int]:
        """
        Evaluate detection and classification metrics for a specific group of categories.

        Args:
            pred_anns: List of prediction annotations (after NMS & score filtering).
            gt_anns: List of ground truth annotations.
            target_cat_ids: Set of category IDs to consider.

        Returns:S
            Tuple of:
                correct_detections,
                correct_classifications,
                total_gt_in_group,
                total_pred_in_group
        """
        if not pred_anns and not gt_anns:
            return 0, 0, 0, 0

        # Filter GT and predictions to only include target categories
        filtered_gt = [ann for ann in gt_anns if ann["category_id"] in target_cat_ids]
        filtered_pred = [ann for ann in pred_anns if ann["category_id"] in target_cat_ids]

        if not filtered_gt and not filtered_pred:
            return 0, 0, 0, 0

        # Sort predictions by score (already filtered, but safe to sort again)
        filtered_pred = sorted(filtered_pred, key=lambda x: x["score"], reverse=True)

        pred_boxes = [ann["bbox"] for ann in filtered_pred]
        gt_boxes = [ann["bbox"] for ann in filtered_gt]

        iou_matrix = self._calculate_iou_matrix(pred_boxes, gt_boxes)
        if iou_matrix.size == 0:
            return 0, 0, len(filtered_gt), len(filtered_pred)

        gt_matched = [False] * len(filtered_gt)
        correct_detections = 0
        correct_classifications = 0

        for pred_idx, pred_ann in enumerate(filtered_pred):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_ann in enumerate(filtered_gt):
                if not gt_matched[gt_idx]:
                    iou = iou_matrix[pred_idx, gt_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_iou >= self.other_iou_threshold and best_gt_idx >= 0:
                gt_matched[best_gt_idx] = True
                correct_detections += 1
                if pred_ann["category_id"] == filtered_gt[best_gt_idx]["category_id"]:
                    correct_classifications += 1

        return correct_detections, correct_classifications, len(filtered_gt), len(filtered_pred)
    
    def other_evaluate(self, result_files: Dict, coco_gt: COCO, logger: Optional[object] = None) -> Dict[str, float]:
          """Evaluate custom metrics: detection rate, recognition rate, and false alarm rate.

          Args:
               result_files: Dictionary containing result file paths.
               coco_gt: COCO ground truth object.
               
          Returns:
               Dictionary containing evaluation metrics.
          """
          metric = 'bbox'
          if metric not in result_files:
               raise KeyError(f"Result file for metric '{metric}' not found")

          # Load predictions
          predictions = mmcv.load(result_files[metric])
          # COCO.loadRes indexes the first prediction and rejects an empty list.
          coco_det = coco_gt.loadRes(predictions) if predictions else None
          img_ids = coco_gt.getImgIds()

          # Initialize counters
          total_correct_detections = 0
          total_correct_classifications = 0
          total_actual_objects = 0
          total_detected_objects = 0  # Only count predictions that meet IoU threshold

          # Process each image
          for img_id in img_ids:
               # Get annotations for current image
               gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
               pred_anns = (coco_det.loadAnns(coco_det.getAnnIds(imgIds=img_id))
                            if coco_det is not None else [])
               
               # Apply score filtering and NMS
               filtered_pred_anns = self._apply_score_filter_and_nms(pred_anns)
               
               # Evaluate single image with filtered predictions
               correct_detections, correct_classifications = self._evaluate_single_image(filtered_pred_anns, gt_anns)
               
               # Update totals
               total_correct_detections += correct_detections
               total_correct_classifications += correct_classifications
               total_actual_objects += len(gt_anns)
               total_detected_objects += correct_detections  # Only count successful detections

          # Calculate metrics
          detection_rate = (total_correct_detections / total_actual_objects 
                         if total_actual_objects > 0 else 0.0)

          recognition_rate = (total_correct_classifications / total_actual_objects 
                         if total_correct_detections > 0 else 0.0)

          # False alarm rate should be calculated based on all filtered predictions vs correct detections
          total_predictions = sum(len(self._apply_score_filter_and_nms(
              coco_det.loadAnns(coco_det.getAnnIds(imgIds=img_id))
              if coco_det is not None else []))
              for img_id in img_ids)
          false_alarm_rate = ((total_predictions - total_correct_detections) / 
                              total_predictions if total_predictions > 0 else 0.0)

          print_log(f"Detection Rate: {detection_rate:.4f}", logger=logger)
          print_log(f"Recognition Rate: {recognition_rate:.4f}", logger=logger)
          print_log(f"False Alarm Rate: {false_alarm_rate:.4f}", logger=logger)
          print_log(f"Total Predictions (after filtering): {total_predictions}, Correct Detections: {total_correct_detections}", logger=logger)
          print_log(f"Score threshold: {self.other_score_threshold}, NMS threshold: {self.other_nms_threshold}", logger=logger)

          return {
               "detection_rate": detection_rate,
               "recognition_rate": recognition_rate,
               "false_alarm_rate": false_alarm_rate
          }

    def evaluate(self,
                results: List,
                metric: str = 'bbox',
                logger: Optional[object] = None,
                jsonfile_prefix: Optional[str] = None,
                classwise: bool = False,
                proposal_nums: Tuple[int, ...] = (100, 300, 1000),
                iou_thrs: Optional[List[float]] = None,
                metric_items: Optional[List[str]] = None) -> Dict:
        """Evaluate detection results.
        
        Args:
            results: Detection results.
            metric: Evaluation metric.
            logger: Logger object.
            jsonfile_prefix: Prefix for result files.
            classwise: Whether to evaluate class-wise.
            proposal_nums: Number of proposals to evaluate.
            iou_thrs: IoU thresholds.
            metric_items: Metric items to evaluate.
            
        Returns:
            Evaluation results dictionary.
        """
        # Validate metrics
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        # Get COCO ground truth
        coco_gt: COCO = self.coco
        self.cat_ids = ordered_category_ids(coco_gt, self.CLASSES)

        # Format results
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        
        # Standard evaluation
        eval_results = self.evaluate_det_segm(
            results, result_files, coco_gt, metrics, logger, classwise,
            proposal_nums, iou_thrs, metric_items
        )

        # Custom evaluation
        other_eval_results = self.other_evaluate(result_files, coco_gt, logger)
        
        # Merge results
        eval_results.update(other_eval_results)
        
        # Cleanup temporary directory
        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results

@ROTATED_DATASETS.register_module()
class SARDet_hbb_novel_trisource(SARDet_hbb_trisource):
    """SAR Detection dataset with horizontal bounding boxes from three sources.
    
    This dataset supports custom evaluation metrics including detection rate,
    recognition rate, and false alarm rate.
    """
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN', 
         )

    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (255, 193, 193)]
    
    NOVEL_CLASSES = SARDET100K_SPLIT_NOVEL_CLASSES[1]
    BASE_CLASSES = get_sardet100k_split_classes(1, use_base_only=True)

    def __init__(self, *args, split_id=None, **kwargs):
        ann_file = kwargs.get('ann_file')
        self.split_id = self._resolve_split_id(split_id=split_id, ann_file=ann_file)
        self.NOVEL_CLASSES = SARDET100K_SPLIT_NOVEL_CLASSES[self.split_id]
        self.BASE_CLASSES = get_sardet100k_split_classes(
            self.split_id, use_base_only=True)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _resolve_split_id(split_id=None, ann_file=None):
        if split_id is not None:
            return int(split_id)

        if isinstance(ann_file, str):
            if 'split1' in ann_file:
                return 1
            if 'split2' in ann_file:
                return 2

        env_split = os.environ.get('SPLIT_NUM')
        if env_split is not None:
            env_split = int(env_split)
            if env_split in SARDET100K_SPLIT_NOVEL_CLASSES:
                return env_split

        return 1
    
    def other_evaluate(self, result_files: Dict, coco_gt: COCO, logger: Optional[object] = None) -> Dict[str, float]:
        """Evaluate custom metrics: detection rate, recognition rate, and false alarm rate,
        for overall, base, and novel categories.
        """
        metric = 'bbox'
        if metric not in result_files:
            raise KeyError(f"Result file for metric '{metric}' not found")

        predictions = mmcv.load(result_files[metric])
        coco_det = coco_gt.loadRes(predictions)
        img_ids = coco_gt.getImgIds()

        # Get category IDs for base and novel
        novel_ids = set(coco_gt.get_cat_ids(cat_names=self.NOVEL_CLASSES)) if self.NOVEL_CLASSES else set()
        base_ids = set(coco_gt.get_cat_ids(cat_names=self.BASE_CLASSES)) if self.BASE_CLASSES else set()
        all_ids = set(coco_gt.get_cat_ids())

        # Initialize accumulators
        metrics = {
            'overall': {'cd': 0, 'cc': 0, 'gt': 0, 'pred': 0},
            'base': {'cd': 0, 'cc': 0, 'gt': 0, 'pred': 0},
            'novel': {'cd': 0, 'cc': 0, 'gt': 0, 'pred': 0},
        }

        for img_id in img_ids:
            gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
            pred_anns = coco_det.loadAnns(coco_det.getAnnIds(imgIds=img_id))
            filtered_pred = self._apply_score_filter_and_nms(pred_anns)

            # Overall
            cd, cc, gt_cnt, pred_cnt = self._evaluate_group_metrics(filtered_pred, gt_anns, all_ids)
            metrics['overall']['cd'] += cd
            metrics['overall']['cc'] += cc
            metrics['overall']['gt'] += gt_cnt
            metrics['overall']['pred'] += pred_cnt

            # Base
            if base_ids:
                cd_b, cc_b, gt_b, pred_b = self._evaluate_group_metrics(filtered_pred, gt_anns, base_ids)
                metrics['base']['cd'] += cd_b
                metrics['base']['cc'] += cc_b
                metrics['base']['gt'] += gt_b
                metrics['base']['pred'] += pred_b

            # Novel
            if novel_ids:
                cd_n, cc_n, gt_n, pred_n = self._evaluate_group_metrics(filtered_pred, gt_anns, novel_ids)
                metrics['novel']['cd'] += cd_n
                metrics['novel']['cc'] += cc_n
                metrics['novel']['gt'] += gt_n
                metrics['novel']['pred'] += pred_n

        results = {}

        for group_name, stat in metrics.items():
            if stat['gt'] == 0 and stat['pred'] == 0:
                continue  # Skip if no instances

            detection_rate = stat['cd'] / stat['gt'] if stat['gt'] > 0 else 0.0
            recognition_rate = stat['cc'] / stat['gt'] if stat['gt'] > 0 else 0.0
            false_alarm_rate = (stat['pred'] - stat['cd']) / stat['pred'] if stat['pred'] > 0 else 0.0

            prefix = f"{group_name}_" if group_name != 'overall' else ""
            results[f"{prefix}detection_rate"] = detection_rate
            results[f"{prefix}recognition_rate"] = recognition_rate
            results[f"{prefix}false_alarm_rate"] = false_alarm_rate

            # Optional: log
            print_log(f"[{group_name.capitalize()}] Detection Rate: {detection_rate:.4f}", logger=logger)
            print_log(f"[{group_name.capitalize()}] Recognition Rate: {recognition_rate:.4f}", logger=logger)
            print_log(f"[{group_name.capitalize()}] False Alarm Rate: {false_alarm_rate:.4f}", logger=logger)

        # Also log thresholds
        print_log(f"Score threshold: {self.other_score_threshold}, NMS threshold: {self.other_nms_threshold}", logger=logger)

        return results

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):
        # ---------------------- 保留原逻辑 ----------------------
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for m in metrics:
            if m not in allowed_metrics:
                raise KeyError(f'metric {m} is not supported')

        coco_gt: COCO = self.coco
        self.cat_ids = ordered_category_ids(coco_gt, self.CLASSES)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm(
            results, result_files, coco_gt, metrics, logger, classwise,
            proposal_nums, iou_thrs, metric_items
        )

        # ---------------------- 新增：计算 base/novel AP ----------------------
        if 'bbox' in metrics:
            coco_dt = coco_gt.loadRes(result_files['bbox'])
            # 获取子类别的 ID 列表
            base_class_names = getattr(self, 'BASE_CLASSES', [])
            novel_class_names = getattr(self, 'NOVEL_CLASSES', [])

            base_ids = coco_gt.get_cat_ids(cat_names=base_class_names)
            novel_ids = coco_gt.get_cat_ids(cat_names=novel_class_names)

            def print_coco_table_for_subset(cat_ids, title):
                """辅助函数：针对特定类别子集运行 COCOeval 并捕获打印"""
                if not cat_ids:
                    return
                # 初始化 COCOeval
                coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
                coco_eval.params.catIds = list(cat_ids) 
                coco_eval.params.useCats = 1
                # 运行评估
                coco_eval.evaluate()
                coco_eval.accumulate()
                # --- 关键修改：捕获 stdout ---
                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    coco_eval.summarize()
                # 获取捕获到的表格字符串
                summary_table = redirect_string.getvalue()

                # 使用 mmcv 的 logger 打印，这样就能同时进控制台和日志文件了
                header = f'\n{"="*20} {title} Results {"="*20}\n'
                print_log(header + summary_table, logger=logger)
                # ---------------------------

                # 将关键指标存入 eval_results
                stats = coco_eval.stats
                eval_results[f'{title.lower()}_mAP'] = float(stats[0])
                eval_results[f'{title.lower()}_AP50'] = float(stats[1])
                eval_results[f'{title.lower()}_AP75'] = float(stats[2])

            if novel_ids:
                print_coco_table_for_subset(novel_ids, 'novel')
            if base_ids:
                print_coco_table_for_subset(base_ids, 'base')

            print_log(
                f'SARDet100K split {self.split_id} eval: '
                f'base_classes={self.BASE_CLASSES}, '
                f'novel_classes={self.NOVEL_CLASSES}',
                logger=logger)

        # ---------------------- 保留自定义指标 ----------------------
        other_eval_results = self.other_evaluate(result_files, coco_gt, logger)
        eval_results.update(other_eval_results)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results

@ROTATED_DATASETS.register_module()
class SARDet_hbb_novel_fine_aircraft_trisource(SARDet_hbb_novel_trisource):
    """SAR Detection dataset with horizontal bounding boxes from three sources.
    
    This dataset supports custom evaluation metrics including detection rate,
    recognition rate, and false alarm rate.
    """
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
         'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
         'Boeing737', 'Boeing787', 'A330', 'ARJ21', 'other')
        
    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (255, 193, 193), (0, 51, 153), (0, 51, 153), (0, 51, 153),
               (0, 51, 153), (0, 51, 153), (0, 51, 153)]
    
    # ====================================================================
    # TODO: 请根据你的实验需求，正确地将上面的类别名称填充到这两个列表中
    # 注意: 这里的类别名称必须与 CLASSES 元组中的完全一致
    NOVEL_CLASSES = ('Boeing737', 'Boeing787', 'A330', 'ARJ21', 'other')
    BASE_CLASSES = ('ship', 'car', 'tank', 'bridge', 'harbor')
    # ====================================================================

    def __init__(self, *args, split_id=None, **kwargs):
        super().__init__(*args, split_id=split_id, **kwargs)
        # Override parent's NOVEL_CLASSES/BASE_CLASSES with fine-grained aircraft splits
        self.NOVEL_CLASSES = ('Boeing737', 'Boeing787', 'A330', 'ARJ21', 'other')
        self.BASE_CLASSES = ('ship', 'car', 'tank', 'bridge', 'harbor')
    


@ROTATED_DATASETS.register_module()
class SARDet_hbb_novel_srsdd_trisource(SARDet_hbb_trisource):
    """SAR Detection dataset with horizontal bounding boxes from three sources.
    
    This dataset supports custom evaluation metrics including detection rate,
    recognition rate, and false alarm rate.
    """
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
         'ore-oil', 'Container', 'Fishing', 'LawEnforce', 'Dredger', 'Cell-Container')
                  
    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (255, 193, 193), (0, 51, 153), (0, 51, 153), (0, 51, 153),
               (0, 51, 153), (0, 51, 153), (0, 51, 153)]

    # ====================================================================
    # TODO: 请根据你的实验需求，正确地将上面的类别名称填充到这两个列表中
    # 注意: 这里的类别名称必须与 CLASSES 元组中的完全一致
    NOVEL_CLASSES = ('ore-oil', 'Container', 'Fishing', 'LawEnforce', 'Dredger', 'Cell-Container')
    BASE_CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', )
    # ====================================================================

@ROTATED_DATASETS.register_module()
class SARDet_hbb_base_trisource(SARDet_hbb_trisource):
    """SAR Detection dataset with horizontal bounding boxes from three sources.
    
    This dataset supports custom evaluation metrics including detection rate,
    recognition rate, and false alarm rate.
    """
    CLASSES = ('ship', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN')

    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (255, 193, 193)]
    
    

@ROTATED_DATASETS.register_module()
class SARDet_base(SARDet_hbb_trisource):
    """SAR Detection dataset with horizontal bounding boxes from three sources.
    
    This dataset supports custom evaluation metrics including detection rate,
    recognition rate, and false alarm rate.
    """
    CLASSES = ('ship', 'car', 'tank', 'bridge', 'harbor')

    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (255, 193, 193)]


@ROTATED_DATASETS.register_module()
class SARDet_base_split2(SARDet_hbb_trisource):
    """SAR base dataset variant for split2."""

    CLASSES = ('aircraft','car', 'tank', 'bridge', 'harbor')

    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220),
               (220, 120, 0)]
    
@ROTATED_DATASETS.register_module()
class SARDet_ALL(SARDet_hbb_trisource):
    """SAR Detection dataset with horizontal bounding boxes from three sources.
    
    This dataset supports custom evaluation metrics including detection rate,
    recognition rate, and false alarm rate.
    """
    CLASSES = ('ship', 'aircraft','car', 'tank', 'bridge', 'harbor')

    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (255, 193, 193)]
    
    
@ROTATED_DATASETS.register_module()
class SARDet_novel(SARDet_hbb_novel_trisource):
    """SAR Detection dataset with horizontal bounding boxes from three sources.
    
    This dataset supports custom evaluation metrics including detection rate,
    recognition rate, and false alarm rate.
    """
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor')

    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (255, 193, 193)]
