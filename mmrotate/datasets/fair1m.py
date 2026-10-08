# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import pickle
import re
import xml.etree.ElementTree as ET

import numpy as np
from mmcv.runner import get_dist_info
from mmcv.utils import print_log
from mmdet.datasets.api_wrappers import COCO

from mmrotate.core import poly2obb_np
from .builder import ROTATED_DATASETS
from .dota import DOTADataset


FAIR1M2_FINE_CLASSES = (
    'Boeing737', 'Boeing777', 'Boeing747', 'Boeing787', 'A321', 'A220',
    'A330', 'A350', 'C919', 'ARJ21', 'other-airplane', 'Passenger Ship',
    'Motorboat', 'Fishing Boat', 'Tugboat', 'Engineering Ship',
    'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship', 'other-ship',
    'Small Car', 'Bus', 'Cargo Truck', 'Dump Truck', 'Van', 'Trailer',
    'Tractor', 'Truck Tractor', 'Excavator', 'other-vehicle',
    'Baseball Field', 'Basketball Court', 'Football Field', 'Tennis Court',
    'Roundabout', 'Intersection', 'Bridge')

FAIR1M2_FINE_PALETTE = [
    (165, 42, 42), (165, 42, 42), (165, 42, 42), (165, 42, 42),
    (165, 42, 42), (165, 42, 42), (165, 42, 42), (165, 42, 42),
    (165, 42, 42), (165, 42, 42), (165, 42, 42), (189, 183, 107),
    (189, 183, 107), (189, 183, 107), (189, 183, 107), (189, 183, 107),
    (189, 183, 107), (189, 183, 107), (189, 183, 107), (189, 183, 107),
    (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0),
    (255, 0, 0), (138, 43, 226), (255, 128, 0), (255, 0, 255),
    (0, 255, 255), (255, 193, 193), (0, 51, 153)
]

FAIR1M2_IMAGE_SUFFIXES = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')


@ROTATED_DATASETS.register_module()
class FAIR1M2Dataset(DOTADataset):
    """FAIR1M-2.0 fine-grained dataset loader.

    This dataset reads the official FAIR1M-2.0 XML annotations directly and
    converts quadrilateral boxes into rotated boxes in the same annotation
    style used by ``DOTADataset``.

    Notes:
        - Default classes are the 37 fine-grained FAIR1M categories.
        - Category subsets are supported through the standard ``classes=...``
          dataset argument from MMDetection/MMRotate.
        - Subset names can use FAIR's original spacing or the more config-
          friendly ``_``/``-`` variants, for example ``Passenger Ship`` and
          ``Passenger_Ship`` are treated as the same class.
    """

    CLASSES = FAIR1M2_FINE_CLASSES
    PALETTE = FAIR1M2_FINE_PALETTE

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_suffix='.tif',
                 version='oc',
                 difficulty=100,
                 cache_annotations=False,
                 cache_filtered=False,
                 other_iou_threshold=0.3,
                 other_score_threshold=0.3,
                 other_nms_threshold=0.3,
                 **kwargs):
        self.img_suffix = img_suffix
        super().__init__(
            ann_file,
            pipeline,
            version=version,
            difficulty=difficulty,
            cache_annotations=cache_annotations,
            cache_filtered=cache_filtered,
            other_iou_threshold=other_iou_threshold,
            other_score_threshold=other_score_threshold,
            other_nms_threshold=other_nms_threshold,
            **kwargs)
        self.PALETTE = self._get_active_palette()

    def load_annotations(self, ann_file):
        if self.cache_annotations and osp.exists(self.cache_annotations):
            with open(self.cache_annotations, 'rb') as f:
                data_infos = pickle.load(f)
            self.img_ids = [osp.splitext(x['filename'])[0] for x in data_infos]
            return data_infos

        cls_map = self._build_class_map()
        xml_files = self._resolve_xml_files(ann_file)
        data_infos = []

        if xml_files:
            for xml_file in xml_files:
                data_info = self._parse_xml_file(xml_file, cls_map)
                if data_info is None:
                    continue
                data_infos.append(data_info)
        else:
            image_files = self._resolve_image_files(ann_file)
            for image_file in image_files:
                filename = osp.basename(image_file)
                data_infos.append(
                    dict(
                        filename=filename,
                        ann=self._build_empty_ann(),
                    ))

        self.img_ids = [osp.splitext(x['filename'])[0] for x in data_infos]
        rank, _ = get_dist_info()
        if self.cache_annotations and rank == 0:
            directory = osp.dirname(self.cache_annotations)
            if directory and not osp.exists(directory):
                os.makedirs(directory)
            with open(self.cache_annotations, 'wb') as f:
                pickle.dump(data_infos, f)
        return data_infos

    @classmethod
    def _canonicalize_class_name(cls, class_name):
        canonical_name = class_name.strip().lower()
        canonical_name = canonical_name.replace('_', ' ').replace('-', ' ')
        canonical_name = re.sub(r'\s+', ' ', canonical_name)
        canonical_name = canonical_name.replace(' ', '')
        return canonical_name

    def _build_class_map(self):
        official_lookup = {
            self._canonicalize_class_name(class_name): class_name
            for class_name in FAIR1M2_FINE_CLASSES
        }
        cls_map = {}
        unknown_classes = []
        for label, class_name in enumerate(self.CLASSES):
            class_key = self._canonicalize_class_name(class_name)
            if class_key not in official_lookup:
                unknown_classes.append(class_name)
                continue
            cls_map[class_key] = label

        if unknown_classes:
            raise KeyError(
                'FAIR1M2Dataset received unknown classes: '
                f'{unknown_classes}. Supported classes are: '
                f'{FAIR1M2_FINE_CLASSES}')
        return cls_map

    def _get_active_palette(self):
        palette_lookup = {
            self._canonicalize_class_name(class_name): FAIR1M2_FINE_PALETTE[idx]
            for idx, class_name in enumerate(FAIR1M2_FINE_CLASSES)
        }
        return [
            palette_lookup.get(
                self._canonicalize_class_name(class_name), (0, 0, 255))
            for class_name in self.CLASSES
        ]

    def _resolve_xml_files(self, ann_file):
        ann_file = osp.abspath(ann_file)
        if osp.isfile(ann_file) and ann_file.lower().endswith('.xml'):
            return [ann_file]
        if osp.isdir(ann_file):
            return sorted(glob.glob(osp.join(ann_file, '*.xml')))
        return []

    def _resolve_image_files(self, ann_file):
        search_dirs = []
        ann_file = osp.abspath(ann_file)
        if osp.isdir(ann_file):
            search_dirs.append(ann_file)
        img_prefix = getattr(self, 'img_prefix', None)
        if img_prefix and osp.isdir(img_prefix):
            search_dirs.append(osp.abspath(img_prefix))

        for search_dir in search_dirs:
            image_files = []
            for suffix in FAIR1M2_IMAGE_SUFFIXES:
                image_files.extend(
                    glob.glob(osp.join(search_dir, f'*{suffix}')))
            if image_files:
                return sorted(set(image_files))

        raise FileNotFoundError(
            'FAIR1M2Dataset could not find XML annotations or images. '
            f'Checked ann_file={ann_file} and img_prefix={img_prefix}.')

    def _parse_xml_file(self, xml_file, cls_map):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError as exc:
            print_log(f'Skip malformed FAIR1M xml {xml_file}: {exc}')
            return None

        root = tree.getroot()
        img_id = osp.splitext(osp.basename(xml_file))[0]
        filename = root.findtext('./source/filename')
        filename = osp.basename(filename.strip()) if filename else (
            f'{img_id}{self.img_suffix}')

        data_info = dict(
            filename=filename,
            width=self._safe_int(root.findtext('./size/width')),
            height=self._safe_int(root.findtext('./size/height')),
            ann=self._build_empty_ann(),
        )

        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_polygons_ignore = []

        objects = root.findall('./objects/object')
        for obj in objects:
            cls_name = obj.findtext('./possibleresult/name', default='')
            label = cls_map.get(self._canonicalize_class_name(cls_name))
            if label is None:
                continue

            polygon = self._parse_object_polygon(obj)
            if polygon is None:
                continue

            try:
                x, y, w, h, a = poly2obb_np(polygon, self.version)
            except Exception:
                continue

            gt_bboxes.append([x, y, w, h, a])
            gt_labels.append(label)
            gt_polygons.append(polygon)

        if gt_bboxes:
            data_info['ann']['bboxes'] = np.array(gt_bboxes, dtype=np.float32)
            data_info['ann']['labels'] = np.array(gt_labels, dtype=np.int64)
            data_info['ann']['polygons'] = np.array(
                gt_polygons, dtype=np.float32)
        else:
            data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
            data_info['ann']['labels'] = np.array([], dtype=np.int64)
            data_info['ann']['polygons'] = np.zeros((0, 8), dtype=np.float32)

        data_info['ann']['bboxes_ignore'] = np.array(
            gt_bboxes_ignore, dtype=np.float32).reshape(-1, 5)
        data_info['ann']['labels_ignore'] = np.array(
            gt_labels_ignore, dtype=np.int64)
        data_info['ann']['polygons_ignore'] = np.array(
            gt_polygons_ignore, dtype=np.float32).reshape(-1, 8)

        return data_info

    def _parse_object_polygon(self, obj):
        points = []
        for point in obj.findall('./points/point'):
            if point.text is None:
                continue
            coords = [x for x in re.split(r'[\s,]+', point.text.strip()) if x]
            if len(coords) < 2:
                continue
            try:
                points.append([float(coords[0]), float(coords[1])])
            except ValueError:
                return None

        if len(points) >= 5 and np.allclose(points[0], points[-1]):
            points = points[:-1]
        if len(points) < 4:
            return None

        polygon = np.asarray(points[:4], dtype=np.float32).reshape(-1)
        if polygon.shape[0] != 8:
            return None
        return polygon

    @staticmethod
    def _build_empty_ann():
        return dict(
            bboxes=np.zeros((0, 5), dtype=np.float32),
            labels=np.array([], dtype=np.int64),
            polygons=np.zeros((0, 8), dtype=np.float32),
            bboxes_ignore=np.zeros((0, 5), dtype=np.float32),
            labels_ignore=np.array([], dtype=np.int64),
            polygons_ignore=np.zeros((0, 8), dtype=np.float32),
        )

    @staticmethod
    def _safe_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


@ROTATED_DATASETS.register_module()
class FAIR1M2CocoDataset(FAIR1M2Dataset):
    """FAIR1M-2.0 dataset loader for COCO-style JSON annotations.

    The JSON annotations may store oriented boxes in ``obb`` or polygon data in
    ``segmentation``. This class converts them into the same rotated bbox
    format used by :class:`FAIR1M2Dataset`, so existing MMRotate pipelines can
    stay unchanged.
    """

    def load_annotations(self, ann_file):
        if self.cache_annotations and osp.exists(self.cache_annotations):
            with open(self.cache_annotations, 'rb') as f:
                data_infos = pickle.load(f)
            self.img_ids = [x['id'] for x in data_infos if 'id' in x]
            return data_infos

        self.coco = COCO(ann_file)
        cat_mappings = self._build_json_category_mappings()
        self.cat_ids = [cat_id for cat_id, _ in cat_mappings]
        self.cat2label = {
            cat_id: label for label, (cat_id, _) in enumerate(cat_mappings)
        }

        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for img_id in self.img_ids:
            img_info = self.coco.load_imgs([img_id])[0]
            img_info['filename'] = img_info['file_name']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            parsed_ann = self._parse_coco_ann_info(img_info, ann_info)
            img_info['ann'] = parsed_ann
            data_infos.append(img_info)

        assert len(set(total_ann_ids)) == len(total_ann_ids), (
            f"Annotation ids in '{ann_file}' are not unique!")

        rank, _ = get_dist_info()
        if self.cache_annotations and rank == 0:
            directory = osp.dirname(self.cache_annotations)
            if directory and not osp.exists(directory):
                os.makedirs(directory)
            with open(self.cache_annotations, 'wb') as f:
                pickle.dump(data_infos, f)
        return data_infos

    def _build_json_category_mappings(self):
        json_cats = self.coco.loadCats(self.coco.get_cat_ids())
        json_name_to_id = {}
        for cat in json_cats:
            cat_name = cat.get('name', '')
            json_name_to_id[self._canonicalize_class_name(cat_name)] = (
                cat['id'], cat_name)

        cat_mappings = []
        unknown_classes = []
        for class_name in self.CLASSES:
            class_key = self._canonicalize_class_name(class_name)
            if class_key not in json_name_to_id:
                unknown_classes.append(class_name)
                continue
            cat_mappings.append(json_name_to_id[class_key])

        if unknown_classes:
            available_names = [cat.get('name', '') for cat in json_cats]
            raise KeyError(
                'FAIR1M2CocoDataset received classes that do not exist in '
                f'{self.ann_file}: {unknown_classes}. '
                f'Available JSON categories are: {available_names}')
        return cat_mappings

    def _parse_coco_ann_info(self, img_info, ann_info):
        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_bboxes_ignore = []
        gt_labels_ignore = []
        gt_polygons_ignore = []

        for ann in ann_info:
            if ann.get('ignore', False):
                continue

            bbox = ann.get('bbox', None)
            if bbox is None or len(bbox) < 4:
                continue
            x1, y1, w, h = bbox[:4]
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann.get('area', 0) <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat2label:
                continue

            polygon = self._extract_coco_polygon(ann)
            if polygon is None:
                continue

            try:
                x, y, bw, bh, a = poly2obb_np(polygon, self.version)
            except Exception:
                continue

            label = self.cat2label[ann['category_id']]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append([x, y, bw, bh, a])
                gt_labels_ignore.append(label)
                gt_polygons_ignore.append(polygon)
            else:
                gt_bboxes.append([x, y, bw, bh, a])
                gt_labels.append(label)
                gt_polygons.append(polygon)

        ann_dict = self._build_empty_ann()
        if gt_bboxes:
            ann_dict['bboxes'] = np.array(gt_bboxes, dtype=np.float32)
            ann_dict['labels'] = np.array(gt_labels, dtype=np.int64)
            ann_dict['polygons'] = np.array(gt_polygons, dtype=np.float32)
        if gt_bboxes_ignore:
            ann_dict['bboxes_ignore'] = np.array(
                gt_bboxes_ignore, dtype=np.float32)
            ann_dict['labels_ignore'] = np.array(
                gt_labels_ignore, dtype=np.int64)
            ann_dict['polygons_ignore'] = np.array(
                gt_polygons_ignore, dtype=np.float32)
        return ann_dict

    def _extract_coco_polygon(self, ann):
        polygon = ann.get('obb', None)
        if polygon is not None:
            polygon = np.asarray(polygon, dtype=np.float32).reshape(-1)
            if polygon.shape[0] >= 8:
                return polygon[:8]

        segmentation = ann.get('segmentation', None)
        if isinstance(segmentation, list) and segmentation:
            polygon = np.asarray(segmentation[0], dtype=np.float32).reshape(-1)
            if polygon.shape[0] >= 8:
                return polygon[:8]

        bbox = ann.get('bbox', None)
        if bbox is not None and len(bbox) >= 4:
            x1, y1, w, h = bbox[:4]
            return np.array(
                [x1, y1, x1 + w, y1, x1 + w, y1 + h, x1, y1 + h],
                dtype=np.float32)
        return None


@ROTATED_DATASETS.register_module()
class FAIR1MtrisourceDataset(FAIR1M2CocoDataset):
    """FAIR1M-2.0 dataset loader for COCO-style JSON annotations.

    The JSON annotations may store oriented boxes in ``obb`` or polygon data in
    ``segmentation``. This class converts them into the same rotated bbox
    format used by :class:`FAIR1M2Dataset`, so existing MMRotate pipelines can
    stay unchanged.
    """
    CLASSES = ("A220", "A320orA321", "A330", "ARJ21", "Boeing 737",
               "Boeing 787", "Other")
    PALETTE = FAIR1M2_FINE_PALETTE

    # Map FAIR1M official categories into the trisource RGB taxonomy.
    # This supports both the old 7-class setup and the new 33-class setup used
    # by local_configs4 mixed FAIR jsons.
    CATEGORY_ALIAS_TO_TRISOURCE = {
        'a220': 'A220',
        'a321': 'A320orA321',
        'a320ora321': 'A320orA321',
        'a330': 'A330',
        'arj21': 'ARJ21',
        'boeing737': 'Boeing 737',
        'boeing787': 'Boeing 787',
        'a350': 'Other',
        'boeing747': 'Other',
        'boeing777': 'Other',
        'c919': 'Other',
        'otherairplane': 'Other',
    }

    def _build_trisource_cat2label(self):
        json_cats = self.coco.loadCats(self.coco.get_cat_ids())
        target_name_to_label = {
            self._canonicalize_class_name(class_name): label
            for label, class_name in enumerate(self.CLASSES)
        }

        other_label = target_name_to_label.get(
            self._canonicalize_class_name('Other'))

        cat_ids = []
        cat2label = {}
        for cat in json_cats:
            cat_name = cat.get('name', '')
            cat_key = self._canonicalize_class_name(cat_name)
            if cat_key in target_name_to_label:
                mapped_key = cat_key
            else:
                mapped_class = self.CATEGORY_ALIAS_TO_TRISOURCE.get(cat_key, None)
                if mapped_class is None:
                    if other_label is None:
                        continue
                    mapped_key = self._canonicalize_class_name('Other')
                else:
                    mapped_key = self._canonicalize_class_name(mapped_class)
                    if mapped_key not in target_name_to_label:
                        if other_label is None:
                            continue
                        mapped_key = self._canonicalize_class_name('Other')
            cat_ids.append(cat['id'])
            cat2label[cat['id']] = target_name_to_label[mapped_key]

        return cat_ids, cat2label

    def load_annotations(self, ann_file):
        if self.cache_annotations and osp.exists(self.cache_annotations):
            with open(self.cache_annotations, 'rb') as f:
                data_infos = pickle.load(f)
            self.img_ids = [x['id'] for x in data_infos if 'id' in x]
            return data_infos

        self.coco = COCO(ann_file)
        self.cat_ids, self.cat2label = self._build_trisource_cat2label()

        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for img_id in self.img_ids:
            img_info = self.coco.load_imgs([img_id])[0]
            img_info['filename'] = img_info['file_name']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)
            parsed_ann = self._parse_coco_ann_info(img_info, ann_info)
            img_info['ann'] = parsed_ann
            data_infos.append(img_info)

        assert len(set(total_ann_ids)) == len(total_ann_ids), (
            f"Annotation ids in '{ann_file}' are not unique!")

        rank, _ = get_dist_info()
        if self.cache_annotations and rank == 0:
            directory = osp.dirname(self.cache_annotations)
            if directory and not osp.exists(directory):
                os.makedirs(directory)
            with open(self.cache_annotations, 'wb') as f:
                pickle.dump(data_infos, f)
        return data_infos
