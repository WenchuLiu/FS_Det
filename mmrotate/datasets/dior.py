import glob
import os
import os.path as osp
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from mmcv.runner import get_dist_info
from mmcv.utils import print_log
from mmdet.core import eval_map
from mmdet.datasets.custom import CustomDataset

from .builder import ROTATED_DATASETS


DIOR_ALL_CLASSES = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'Bridge',
    'chimney', 'dam', 'Expressway-Service-area',
    'Expressway-toll-station', 'golffield', 'groundtrackfield', 'Harbor',
    'overpass', 'Ship', 'stadium', 'storagetank', 'tenniscourt',
    'trainstation', 'vehicle', 'windmill')

DIOR_SPLIT_NOVEL_CLASSES = {
    1: ('baseballfield', 'basketballcourt', 'Bridge', 'chimney', 'Ship'),
    2: ('airplane', 'airport', 'Expressway-toll-station', 'Harbor', 'groundtrackfield'),
    3: ('dam', 'golffield', 'storagetank', 'tenniscourt', 'vehicle'),
    4: ('Expressway-Service-area', 'overpass', 'stadium', 'trainstation', 'windmill'),
}

DIOR_PALETTE = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 192),
    (250, 170, 30), (100, 170, 30), (220, 220, 0), (175, 116, 175),
    (250, 0, 30), (165, 42, 42), (0, 226, 252), (182, 182, 255),
    (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255)
]


def get_dior_split_classes(split_id, use_base_only=False, include_novel=True):
    if split_id not in DIOR_SPLIT_NOVEL_CLASSES:
        raise KeyError(
            f'Invalid DIOR split_id={split_id}. '
            f'Available splits: {tuple(sorted(DIOR_SPLIT_NOVEL_CLASSES))}')

    novel_classes = DIOR_SPLIT_NOVEL_CLASSES[split_id]
    base_classes = tuple(
        class_name for class_name in DIOR_ALL_CLASSES
        if class_name not in novel_classes)

    if use_base_only:
        return base_classes
    if include_novel:
        return base_classes + novel_classes
    return DIOR_ALL_CLASSES


@ROTATED_DATASETS.register_module()
class DIORDataset(CustomDataset):
    """DIOR horizontal-bbox dataset.

    Supported ``ann_file`` forms:
    1. XML directory: ``.../Annotations/``
    2. Single split txt: one image id per line
    3. Few-shot split directory: multiple ``*.txt`` files, each listing image ids

    When ``ann_file`` is a txt file or a directory of txt files, ``ann_subdir``
    should point to the directory that stores the DIOR XML annotations.
    """

    CLASSES = DIOR_ALL_CLASSES
    PALETTE = DIOR_PALETTE

    def __init__(self,
                 ann_file,
                 pipeline,
                 ann_subdir=None,
                 img_suffix='.jpg',
                 difficulty=0,
                 cache_annotations=False,
                 cache_filtered=False,
                 exclude_image_classes=None,
                 **kwargs):
        self.ann_subdir = ann_subdir
        self.img_suffix = img_suffix
        self.difficulty = difficulty
        self.cache_annotations = cache_annotations
        self.cache_filtered = cache_filtered
        self.exclude_image_classes = tuple(exclude_image_classes or ())
        self.exclude_image_class_keys = {
            self._canonicalize_class_name(class_name)
            for class_name in self.exclude_image_classes
        }
        super().__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        if self.cache_annotations and osp.exists(self.cache_annotations):
            with open(self.cache_annotations, 'rb') as f:
                data_infos = pickle.load(f)
            self.img_ids = [osp.splitext(x['filename'])[0] for x in data_infos]
            return data_infos

        xml_paths, img_ids = self._resolve_annotation_sources(ann_file)
        cls_map = self._build_class_map()
        data_infos = []

        for xml_path, img_id in zip(xml_paths, img_ids):
            if not osp.exists(xml_path):
                print_log(f'Skip missing DIOR annotation: {xml_path}')
                continue

            data_info = self._parse_xml(xml_path, img_id, cls_map)
            if data_info is None:
                continue
            data_infos.append(data_info)

        self.img_ids = [osp.splitext(x['filename'])[0] for x in data_infos]
        rank, _ = get_dist_info()
        if self.cache_annotations and rank == 0:
            directory = osp.dirname(self.cache_annotations)
            if directory and not osp.exists(directory):
                os.makedirs(directory)
            with open(self.cache_annotations, 'wb') as f:
                pickle.dump(data_infos, f)
        return data_infos

    def _filter_imgs(self):
        if self.cache_filtered and osp.exists(self.cache_filtered):
            with open(self.cache_filtered, 'rb') as f:
                return pickle.load(f)

        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if not self.filter_empty_gt or data_info['ann']['labels'].size > 0:
                valid_inds.append(i)

        rank, _ = get_dist_info()
        if self.cache_filtered and rank == 0:
            directory = osp.dirname(self.cache_filtered)
            if directory and not osp.exists(directory):
                os.makedirs(directory)
            with open(self.cache_filtered, 'wb') as f:
                pickle.dump(valid_inds, f)
        return valid_inds

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i, data_info in enumerate(self.data_infos):
            width = data_info.get('width', 0)
            height = data_info.get('height', 0)
            if width > height:
                self.flag[i] = 1

    def _resolve_annotation_sources(self, ann_file):
        ann_file = osp.abspath(ann_file)
        if osp.isdir(ann_file):
            xml_paths = sorted(glob.glob(osp.join(ann_file, '*.xml')))
            if xml_paths:
                img_ids = [osp.splitext(osp.basename(p))[0] for p in xml_paths]
                return xml_paths, img_ids

            split_files = sorted(glob.glob(osp.join(ann_file, '*.txt')))
            if split_files:
                img_ids = self._load_img_ids_from_split_files(split_files)
                ann_dir = self._infer_ann_dir(ann_file)
                xml_paths = [osp.join(ann_dir, f'{img_id}.xml') for img_id in img_ids]
                return xml_paths, img_ids

            raise FileNotFoundError(
                f'No xml/txt annotations found in DIOR ann_file directory: {ann_file}')

        if ann_file.endswith('.txt'):
            img_ids = self._load_img_ids_from_split_files([ann_file])
            ann_dir = self._infer_ann_dir(ann_file)
            xml_paths = [osp.join(ann_dir, f'{img_id}.xml') for img_id in img_ids]
            return xml_paths, img_ids

        raise ValueError(
            'DIORDataset ann_file must be an xml directory, a split txt, '
            f'or a directory of split txt files, but got: {ann_file}')

    def _load_img_ids_from_split_files(self, split_files):
        img_ids = []
        seen = set()
        for split_file in split_files:
            with open(split_file, 'r') as f:
                for line in f:
                    img_id = line.strip()
                    if not img_id or img_id in seen:
                        continue
                    seen.add(img_id)
                    img_ids.append(img_id)
        return img_ids

    def _infer_ann_dir(self, ann_file):
        if self.ann_subdir is not None:
            return osp.abspath(self.ann_subdir)

        if osp.isdir(ann_file):
            base_dir = ann_file
        else:
            base_dir = osp.dirname(ann_file)

        candidates = [
            osp.join(base_dir, 'Annotations', 'Horizontal Bounding Boxes'),
            osp.join(base_dir, 'Annotations'),
            osp.join(osp.dirname(base_dir), 'Annotations', 'Horizontal Bounding Boxes'),
            osp.join(osp.dirname(base_dir), 'Annotations'),
            osp.join(osp.dirname(osp.dirname(base_dir)), 'Annotations', 'Horizontal Bounding Boxes'),
            osp.join(osp.dirname(osp.dirname(base_dir)), 'Annotations'),
        ]
        for candidate in candidates:
            if osp.isdir(candidate):
                return osp.abspath(candidate)

        raise FileNotFoundError(
            'Cannot infer DIOR annotation xml directory. '
            'Please set ann_subdir explicitly.')

    def _build_class_map(self):
        cls_map = {}
        for idx, class_name in enumerate(self.CLASSES):
            cls_map[self._canonicalize_class_name(class_name)] = idx
        return cls_map

    def _canonicalize_class_name(self, class_name):
        return ''.join(ch for ch in class_name.lower() if ch.isalnum())

    def _should_exclude_image(self, root):
        if not self.exclude_image_class_keys:
            return False

        for obj in root.findall('object'):
            class_name = obj.findtext('name', default='').strip()
            class_key = self._canonicalize_class_name(class_name)
            if class_key in self.exclude_image_class_keys:
                return True
        return False

    def _parse_xml(self, xml_path, img_id, cls_map):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        if self._should_exclude_image(root):
            return None

        size = root.find('size')
        width = int(float(size.findtext('width', default='0'))) if size is not None else 0
        height = int(float(size.findtext('height', default='0'))) if size is not None else 0
        filename = root.findtext('filename', default=f'{img_id}{self.img_suffix}')
        if osp.splitext(filename)[1] == '':
            filename = f'{filename}{self.img_suffix}'

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []

        for obj in root.findall('object'):
            class_name = obj.findtext('name', default='').strip()
            class_key = self._canonicalize_class_name(class_name)
            if class_key not in cls_map:
                continue

            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            xmin = float(bndbox.findtext('xmin', default='0'))
            ymin = float(bndbox.findtext('ymin', default='0'))
            xmax = float(bndbox.findtext('xmax', default='0'))
            ymax = float(bndbox.findtext('ymax', default='0'))

            if xmax <= xmin or ymax <= ymin:
                continue

            bbox = [xmin, ymin, xmax, ymax]
            label = cls_map[class_key]
            difficult = int(float(obj.findtext('difficult', default='0')))

            if difficult > self.difficulty:
                gt_bboxes_ignore.append(bbox)
                gt_labels_ignore.append(label)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(label)

        if not gt_bboxes and self.filter_empty_gt:
            return None

        ann = dict(
            bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(gt_labels, dtype=np.int64),
            bboxes_ignore=np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
            labels_ignore=np.array(gt_labels_ignore, dtype=np.int64),
        )

        return dict(filename=filename, width=width, height=height, ann=ann)

    def _compact_results_for_eval(self, results, logger=None):
        expected_num_classes = len(self.CLASSES)
        if not results:
            return results

        actual_num_classes = len(results[0])
        if actual_num_classes == expected_num_classes:
            return results

        print_log(
            'DIOR eval got '
            f'{actual_num_classes} result classes but dataset has '
            f'{expected_num_classes}; '
            f'keep the first {expected_num_classes} classes for evaluation.',
            logger=logger)

        compacted_results = []
        for img_res in results:
            img_res = list(img_res)
            if len(img_res) >= expected_num_classes:
                compacted_results.append(img_res[:expected_num_classes])
                continue

            compacted_results.append(
                img_res + [np.zeros((0, 5), dtype=np.float32)
                           for _ in range(expected_num_classes - len(img_res))])
        return compacted_results

    def _get_split_eval_class_indices(self):
        split_id = getattr(self, 'split_id', None)
        if split_id not in DIOR_SPLIT_NOVEL_CLASSES:
            return None, None

        novel_class_keys = {
            self._canonicalize_class_name(class_name)
            for class_name in DIOR_SPLIT_NOVEL_CLASSES[split_id]
        }
        base_inds = []
        novel_inds = []
        for idx, class_name in enumerate(self.CLASSES):
            class_key = self._canonicalize_class_name(class_name)
            if class_key in novel_class_keys:
                novel_inds.append(idx)
            else:
                base_inds.append(idx)
        return base_inds, novel_inds

    def _compute_group_map(self, eval_results, class_indices):
        if not class_indices:
            return None

        aps = []
        for idx in class_indices:
            cls_result = eval_results[idx]
            cls_num_gts = np.asarray(cls_result['num_gts'])
            cls_ap = np.asarray(cls_result['ap'])
            if cls_num_gts.ndim == 0:
                if cls_num_gts.item() > 0:
                    aps.append(float(cls_ap.item()))
                continue

            valid = cls_num_gts > 0
            if np.any(valid):
                aps.append(float(cls_ap[valid].mean()))

        return float(np.mean(aps)) if aps else 0.0

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        if metric != 'mAP':
            raise KeyError(f'metric {metric} is not supported')

        results = self._compact_results_for_eval(results, logger=logger)
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        mean_ap, eval_results = eval_map(
            results,
            annotations,
            scale_ranges=scale_ranges,
            iou_thr=iou_thr,
            dataset=self.CLASSES,
            logger=logger)

        eval_dict = {'mAP': mean_ap}
        base_inds, novel_inds = self._get_split_eval_class_indices()
        base_map = self._compute_group_map(eval_results, base_inds)
        novel_map = self._compute_group_map(eval_results, novel_inds)

        if base_map is not None:
            eval_dict['base_mAP'] = base_map
        if novel_map is not None:
            eval_dict['novel_mAP'] = novel_map

        if base_map is not None or novel_map is not None:
            split_id = getattr(self, 'split_id', 'unknown')
            summary = [f'DIOR split {split_id} eval']
            if base_map is not None:
                summary.append(f'base_mAP={base_map:.4f}')
            if novel_map is not None:
                summary.append(f'novel_mAP={novel_map:.4f}')
            print_log(', '.join(summary), logger=logger)

        return eval_dict


@ROTATED_DATASETS.register_module()
class DIORBaseDataset(DIORDataset):
    """DIOR dataset with split-aware base classes kept and remapped."""

    PALETTE = DIOR_PALETTE

    def __init__(self, *args, split_id=1, **kwargs):
        self.split_id = int(split_id)
        self.CLASSES = get_dior_split_classes(self.split_id, use_base_only=True)
        exclude_image_classes = kwargs.pop('exclude_image_classes', None)
        if exclude_image_classes is None and kwargs.get('test_mode', False):
            exclude_image_classes = DIOR_SPLIT_NOVEL_CLASSES[self.split_id]
        super().__init__(
            *args,
            exclude_image_classes=exclude_image_classes,
            **kwargs)


@ROTATED_DATASETS.register_module()
class DIORNovelDataset(DIORDataset):
    """DIOR few-shot dataset with split-aware class order.

    Class order is always: base classes first, then the novel classes for
    the selected split.
    """

    PALETTE = DIOR_PALETTE

    def __init__(self, *args, split_id=1, **kwargs):
        self.split_id = int(split_id)
        self.CLASSES = get_dior_split_classes(self.split_id, use_base_only=False, include_novel=True)
        super().__init__(*args, **kwargs)
