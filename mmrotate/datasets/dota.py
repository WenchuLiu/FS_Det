# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial
import pickle
import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated, nms
from mmdet.datasets.custom import CustomDataset 
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from mmcv.runner import get_dist_info
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmcv.utils import print_log
from typing import Dict, List, Tuple, Optional
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class DOTADataset(CustomDataset):
    """DOTA dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
               (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
               (255, 255, 0), (147, 116, 116), (0, 0, 255)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 cache_annotations=False,
                 cache_filtered=False,
                 other_iou_threshold=0.3,
                 other_score_threshold=0.3,
                 other_nms_threshold=0.3,
                 filter_unknown_classes=False,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty
        self.cache_annotations = cache_annotations
        self.cache_filtered = cache_filtered
        self.other_iou_threshold = other_iou_threshold
        self.other_score_threshold = other_score_threshold
        self.other_nms_threshold = other_nms_threshold
        self.filter_unknown_classes = filter_unknown_classes

        super(DOTADataset, self).__init__(ann_file, pipeline, **kwargs)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        if self.cache_annotations:
            if osp.exists(self.cache_annotations):
                with open(self.cache_annotations, 'rb') as f:
                    data_infos = pickle.load(f)
                return data_infos
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        # Some DOTA splits capitalize Ship/Harbor/Bridge. Accept a case
        # variant only when it maps to exactly one configured class.
        folded_cls_map = {}
        ambiguous_names = set()
        for name, label in cls_map.items():
            folded = name.casefold()
            if folded in folded_cls_map:
                ambiguous_names.add(folded)
            else:
                folded_cls_map[folded] = label
        for folded in ambiguous_names:
            folded_cls_map.pop(folded)
        ann_files = glob.glob(ann_folder + '/*.txt')
        image_root = getattr(self, 'img_prefix', None) or ann_folder
        image_paths = []
        for extension in ('png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'):
            image_paths.extend(glob.glob(osp.join(image_root, '*.' + extension)))
            image_paths.extend(glob.glob(osp.join(image_root, '*.' + extension.upper())))
        image_by_stem = {osp.splitext(osp.basename(path))[0]:
                         osp.basename(path) for path in image_paths}

        def image_filename(image_id):
            return image_by_stem.get(image_id, image_id + '.png')

        data_infos = []
        if not ann_files:  # test phase
            for image_path in image_paths:
                data_info = {}
                img_name = osp.basename(image_path)
                img_id = osp.splitext(img_name)[0]
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = image_filename(img_id)
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        cls_name = bbox_info[8]
                        difficulty = int(bbox_info[9])
                        label = cls_map.get(cls_name)
                        if label is None:
                            label = folded_cls_map.get(cls_name.casefold())
                        if label is None:
                            if self.filter_unknown_classes:
                                continue
                            raise KeyError(
                                f'Unknown category {cls_name!r} in {ann_file}; '
                                'add it to classes or set '
                                'filter_unknown_classes=True.')
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        rank, world_size = get_dist_info()
        if self.cache_annotations and rank==0:
            directory = osp.dirname(self.cache_annotations)
            if not osp.exists(directory):
                os.makedirs(directory)
            with open(self.cache_annotations, 'wb') as f:
                pickle.dump(data_infos, f)
        return data_infos

    def _filter_imgs(self):
        """Filter images without ground truths."""
        if self.cache_filtered:
            if osp.exists(self.cache_filtered):
                with open(self.cache_filtered, 'rb') as f:
                    valid_inds = pickle.load(f)
                return valid_inds
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if (not self.filter_empty_gt
                    or data_info['ann']['labels'].size > 0):
                valid_inds.append(i) 
        rank, world_size = get_dist_info() 
        if self.cache_filtered and rank==0:
            directory = osp.dirname(self.cache_filtered)
            if not osp.exists(directory):
                os.makedirs(directory)
            with open(self.cache_filtered, 'wb') as f:
                pickle.dump(valid_inds, f)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def _convert_rotated_bbox_to_horizontal(self, rotated_bboxes: List) -> List:
        """Convert rotated bounding boxes to horizontal bounding boxes.
        
        Args:
            rotated_bboxes: List of rotated bounding boxes in [x, y, w, h, a] format.
            
        Returns:
            List of horizontal bounding boxes in [x1, y1, x2, y2] format.
        """
        horizontal_boxes = []
        for bbox in rotated_bboxes:
            x, y, w, h, a = bbox
            # Convert to polygon first
            cos_a = np.cos(a)
            sin_a = np.sin(a)
            
            # Calculate the four corners of the rotated box
            corners = np.array([
                [-w/2, -h/2],
                [w/2, -h/2],
                [w/2, h/2],
                [-w/2, h/2]
            ])
            
            # Apply rotation
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])
            rotated_corners = corners @ rotation_matrix.T
            
            # Translate to center position
            rotated_corners[:, 0] += x
            rotated_corners[:, 1] += y
            
            # Get bounding box of the rotated rectangle
            x1 = np.min(rotated_corners[:, 0])
            y1 = np.min(rotated_corners[:, 1])
            x2 = np.max(rotated_corners[:, 0])
            y2 = np.max(rotated_corners[:, 1])
            
            horizontal_boxes.append([x1, y1, x2, y2])
        
        return horizontal_boxes

    def _calculate_iou_matrix(self, pred_boxes: List, gt_boxes: List) -> np.ndarray:
        """Calculate IoU matrix between predictions and ground truth boxes.
        
        Args:
            pred_boxes: List of prediction bounding boxes in [x, y, w, h, a] format.
            gt_boxes: List of ground truth bounding boxes in [x, y, w, h, a] format.
            
        Returns:
            IoU matrix of shape (num_predictions, num_ground_truth).
        """
        if not pred_boxes or not gt_boxes:
            return np.array([])
        
        # Convert rotated boxes to horizontal boxes
        pred_boxes_horizontal = self._convert_rotated_bbox_to_horizontal(pred_boxes)
        gt_boxes_horizontal = self._convert_rotated_bbox_to_horizontal(gt_boxes)
        
        # Convert to numpy arrays for vectorized computation
        pred_boxes_array = np.array(pred_boxes_horizontal)
        gt_boxes_array = np.array(gt_boxes_horizontal)
        
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
            
            # Convert rotated boxes to horizontal boxes for NMS
            rotated_bboxes = [ann["bbox"] for ann in cat_anns]  # [x, y, w, h, a]
            horizontal_bboxes = self._convert_rotated_bbox_to_horizontal(rotated_bboxes)
            scores = [ann["score"] for ann in cat_anns]
            
            bboxes_tensor = torch.tensor(horizontal_bboxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            
            # Apply horizontal NMS
            _, keep_indices = nms(bboxes_tensor, scores_tensor, self.other_nms_threshold)
            
            # Convert tensor indices to Python indices and keep the filtered annotations
            if isinstance(keep_indices, torch.Tensor):
                keep_indices = keep_indices.cpu().numpy()
            for idx in keep_indices:
                nms_results.append(cat_anns[idx])
        
        return nms_results

    def other_evaluate(self, results: List, logger: Optional[object] = None) -> Dict[str, float]:
        """Evaluate custom metrics: detection rate, recognition rate, and false alarm rate.

        Args:
            results: List of detection results.
            logger: Logger object.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        # Initialize counters
        total_correct_detections = 0
        total_correct_classifications = 0
        total_actual_objects = 0
        total_detected_objects = 0

        # Process each image
        for idx in range(len(self)):
            # Get ground truth annotations for current image
            gt_info = self.get_ann_info(idx)
            gt_bboxes = gt_info['bboxes']  # [x, y, w, h, a] format
            gt_labels = gt_info['labels']
            
            # Get predictions for current image
            pred_result = results[idx]
            
            # Convert ground truth to annotation format
            gt_anns = []
            for i, (bbox, label) in enumerate(zip(gt_bboxes, gt_labels)):
                gt_anns.append({
                    "bbox": bbox,  # [x, y, w, h, a] format
                    "category_id": label,
                    "score": 1.0
                })
            
            # Convert predictions to annotation format
            pred_anns = []
            for class_id, class_preds in enumerate(pred_result):
                if len(class_preds) > 0:
                    for pred in class_preds:
                        bbox = pred[:5]  # [x, y, w, h, a]
                        score = pred[5] if len(pred) > 5 else 0.5
                        pred_anns.append({
                            "bbox": bbox,
                            "category_id": class_id,
                            "score": score
                        })
            
            # Apply score filtering and NMS
            filtered_pred_anns = self._apply_score_filter_and_nms(pred_anns)
            
            # Evaluate single image with filtered predictions
            correct_detections, correct_classifications = self._evaluate_single_image(filtered_pred_anns, gt_anns)
            
            # Update totals
            total_correct_detections += correct_detections
            total_correct_classifications += correct_classifications
            total_actual_objects += len(gt_anns)
            total_detected_objects += correct_detections

        # Calculate metrics
        detection_rate = (total_correct_detections / total_actual_objects 
                       if total_actual_objects > 0 else 0.0)

        recognition_rate = (total_correct_classifications / total_actual_objects 
                       if total_actual_objects > 0 else 0.0)

        # Calculate total predictions after filtering
        total_predictions = 0
        for idx in range(len(self)):
            pred_result = results[idx]
            pred_anns = []
            for class_id, class_preds in enumerate(pred_result):
                if len(class_preds) > 0:
                    for pred in class_preds:
                        bbox = pred[:5]  # [x, y, w, h, a]
                        score = pred[5] if len(pred) > 5 else 0.5
                        pred_anns.append({
                            "bbox": bbox,
                            "category_id": class_id,
                            "score": score
                        })
            filtered_pred_anns = self._apply_score_filter_and_nms(pred_anns)
            total_predictions += len(filtered_pred_anns)

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
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=[0.5],
                 scale_ranges=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = min(nproc, os.cpu_count())
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        if len(results) != len(self):
            raise ValueError('DOTA prediction count must match dataset length.')
        for index, image_results in enumerate(results):
            if len(image_results) != len(self.CLASSES):
                raise ValueError(
                    f'DOTA image {index}: model returned {len(image_results)} '
                    f'classes but dataset has {len(self.CLASSES)}. '
                    'Use the base class table for a base checkpoint.')
            for boxes in image_results:
                if boxes.ndim != 2 or boxes.shape[1] != 6:
                    raise ValueError('DOTA detections must have shape (N, 6): '
                                     'cx, cy, w, h, angle, score.')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}
        if metric == 'mAP':
            # assert isinstance(iou_thr, float)
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                # iou_thr=iou_thr,
                dataset=self.CLASSES,
                logger=logger,
                nproc=nproc)
            eval_results['mAP'] = mean_ap
        else:
            raise NotImplementedError

        # Add custom evaluation metrics
        other_eval_results = self.other_evaluate(results, logger)
        eval_results.update(other_eval_results)

        return eval_results

    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)

    def _results2submission(self, id_list, dets_list, out_folder=None):
        """Generate the submission of full images.

        Args:
            id_list (list): Id of images.
            dets_list (list): Detection results of per class.
            out_folder (str, optional): Folder of submission.
        """
        if osp.exists(out_folder):
            raise ValueError(f'The out_folder should be a non-exist path, '
                             f'but {out_folder} is existing')
        os.makedirs(out_folder)

        files = [
            osp.join(out_folder, 'Task1_' + cls + '.txt')
            for cls in self.CLASSES
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                bboxes = obb2poly_np(dets, self.version)
                for bbox in bboxes:
                    txt_element = [img_id, str(bbox[-1])
                                   ] + [f'{p:.2f}' for p in bbox[:-1]]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        target_name = osp.split(out_folder)[-1]
        with zipfile.ZipFile(
                osp.join(out_folder, target_name + '.zip'), 'w',
                zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return files

    def format_results(self, results, submission_dir=None, nproc=4, **kwargs):
        """Format the results to submission text (standard format for DOTA
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            submission_dir (str, optional): The folder that contains submission
                files. If not specified, a temp folder will be created.
                Default: None.
            nproc (int, optional): number of process.

        Returns:
            tuple:

                - result_files (dict): a dict containing the json filepaths
                - tmp_dir (str): the temporal directory created for saving \
                    json files when submission_dir is not specified.
        """
        nproc = min(nproc, os.cpu_count())
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            f'The length of results is not equal to '
            f'the dataset len: {len(results)} != {len(self)}')
        if submission_dir is None:
            submission_dir = tempfile.TemporaryDirectory()
        else:
            tmp_dir = None

        print('\nMerging patch bboxes into full image!!!')
        start_time = time.time()
        id_list, dets_list = self.merge_det(results, nproc)
        stop_time = time.time()
        print(f'Used time: {(stop_time - start_time):.1f} s')

        result_files = self._results2submission(id_list, dets_list,
                                                submission_dir)

        return result_files, tmp_dir


def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results
