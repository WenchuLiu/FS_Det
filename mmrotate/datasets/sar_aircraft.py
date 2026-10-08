import contextlib
import io
import logging
import os
import re
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet.datasets import CocoDataset
from mmdet.datasets.api_wrappers import COCOeval
from terminaltables import AsciiTable

from .builder import ROTATED_DATASETS


SAR_AIRCRAFT_CLASSES = (
    "Boeing737",
    "Boeing787",
    "A330",
    "A220",
    "A320orA321",
    "ARJ21",
    "other",
)

SAR_AIRCRAFT_PALETTE = [
    (0, 51, 153),
    (0, 102, 204),
    (51, 153, 255),
    (0, 204, 153),
    (255, 153, 51),
    (255, 102, 102),
    (153, 153, 153),
]

SAR_AIRCRAFT_SPLIT_GROUPS = {
    1: dict(
        base=("Boeing787", "A330", "A220", "A320orA321", "other"),
        novel=("Boeing737", "ARJ21"),
    ),
    2: dict(
        base=("Boeing737", "A330", "A220", "A320orA321", "other"),
        novel=("Boeing787", "ARJ21"),
    ),
}


@ROTATED_DATASETS.register_module()
class SARAircraftDataset(CocoDataset):
    """COCO-style horizontal-bbox dataset for SAR-Aircraft-1.0."""

    CLASSES = SAR_AIRCRAFT_CLASSES
    PALETTE = SAR_AIRCRAFT_PALETTE

    @staticmethod
    def _format_metric(value):
        return "nan" if np.isnan(value) else f"{value:.3f}"

    @staticmethod
    def _mean_valid(values):
        valid_values = [value for value in values if not np.isnan(value)]
        if not valid_values:
            return float("nan")
        return float(np.mean(valid_values))

    def _infer_split_id(self):
        search_targets = []
        ann_file = getattr(self, "ann_file", None)
        if isinstance(ann_file, str):
            search_targets.append(ann_file)

        split_env = os.environ.get("SAR_AIRCRAFT_SPLIT")
        if split_env:
            search_targets.append(split_env)

        for target in search_targets:
            match = re.search(r"split(\d+)", target)
            if match:
                return int(match.group(1))
        return None

    def _resolve_eval_class_names(self):
        classes = getattr(self, "CLASSES", None)
        if classes is not None and len(classes) == len(self.cat_ids):
            return [str(class_name) for class_name in classes]
        return [self.coco.loadCats(cat_id)[0]["name"] for cat_id in self.cat_ids]

    def _collect_per_class_metrics(self, coco_eval):
        precisions = coco_eval.eval["precision"]
        assert len(self.cat_ids) == precisions.shape[2]

        class_names = self._resolve_eval_class_names()
        ap50_indices = np.where(np.isclose(coco_eval.params.iouThrs, 0.5))[0]
        ap50_index = int(ap50_indices[0]) if ap50_indices.size else None

        per_class_metrics = []
        for class_index, class_name in enumerate(class_names):
            precision = precisions[:, :, class_index, 0, -1]
            precision = precision[precision > -1]
            ap = float(np.mean(precision)) if precision.size else float("nan")

            ap50 = float("nan")
            if ap50_index is not None:
                precision50 = precisions[ap50_index, :, class_index, 0, -1]
                precision50 = precision50[precision50 > -1]
                if precision50.size:
                    ap50 = float(np.mean(precision50))

            per_class_metrics.append(
                dict(name=class_name, ap=ap, ap50=ap50)
            )

        return per_class_metrics

    def _log_metric_table(self, title, rows, logger=None, include_map=False):
        table_data = [["category", "AP", "AP50"]]
        for row in rows:
            table_data.append(
                [
                    row["name"],
                    self._format_metric(row["ap"]),
                    self._format_metric(row["ap50"]),
                ]
            )

        if include_map:
            table_data.append(
                [
                    "mAP",
                    self._format_metric(self._mean_valid([row["ap"] for row in rows])),
                    self._format_metric(
                        self._mean_valid([row["ap50"] for row in rows])
                    ),
                ]
            )

        print_log(f"\n{title}", logger=logger)
        print_log("\n" + AsciiTable(table_data).table, logger=logger)

    def _collect_group_metrics(self, per_class_metrics):
        split_id = self._infer_split_id()
        split_groups = SAR_AIRCRAFT_SPLIT_GROUPS.get(split_id)
        if split_groups is None:
            return None, OrderedDict()

        metric_by_name = {
            metric["name"]: metric for metric in per_class_metrics
        }
        group_metrics = OrderedDict()
        for group_name, class_names in split_groups.items():
            rows = [
                metric_by_name[class_name]
                for class_name in class_names
                if class_name in metric_by_name
            ]
            if not rows:
                continue

            group_metrics[group_name] = dict(
                rows=rows,
                mAP=self._mean_valid([row["ap"] for row in rows]),
                mAP_50=self._mean_valid([row["ap50"] for row in rows]),
            )

        return split_id, group_metrics

    def evaluate_det_segm(
        self,
        results,
        result_files,
        coco_gt,
        metrics,
        logger=None,
        classwise=False,
        proposal_nums=(100, 300, 1000),
        iou_thrs=None,
        metric_items=None,
    ):
        """COCO evaluation with split-aware base/novel reporting."""
        if iou_thrs is None:
            iou_thrs = np.linspace(
                0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
            )
        if metric_items is not None and not isinstance(metric_items, list):
            metric_items = [metric_items]

        eval_results = OrderedDict()
        for metric in metrics:
            msg = f"Evaluating {metric}..."
            if logger is None:
                msg = "\n" + msg
            print_log(msg, logger=logger)

            if metric == "proposal_fast":
                if isinstance(results[0], tuple):
                    raise KeyError(
                        "proposal_fast is not supported for instance segmentation result."
                    )
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger="silent"
                )
                log_msg = []
                for index, num in enumerate(proposal_nums):
                    eval_results[f"AR@{num}"] = ar[index]
                    log_msg.append(f"\nAR@{num}\t{ar[index]:.4f}")
                print_log("".join(log_msg), logger=logger)
                continue

            iou_type = "bbox" if metric == "proposal" else metric
            if metric not in result_files:
                raise KeyError(f"{metric} is not in results")

            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == "segm":
                    for prediction in predictions:
                        prediction.pop("bbox")
                    warnings.simplefilter("once")
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        "of small/medium/large instances since v2.12.0. This "
                        "does not change the overall mAP calculation.",
                        UserWarning,
                    )
                coco_det = coco_gt.loadRes(predictions)
            except IndexError:
                print_log(
                    "The testing results of the whole dataset is empty.",
                    logger=logger,
                    level=logging.ERROR,
                )
                break

            coco_eval = COCOeval(coco_gt, coco_det, iou_type)
            coco_eval.params.catIds = self.cat_ids
            coco_eval.params.imgIds = self.img_ids
            coco_eval.params.maxDets = list(proposal_nums)
            coco_eval.params.iouThrs = iou_thrs

            coco_metric_names = {
                "mAP": 0,
                "mAP_50": 1,
                "mAP_75": 2,
                "mAP_s": 3,
                "mAP_m": 4,
                "mAP_l": 5,
                "AR@100": 6,
                "AR@300": 7,
                "AR@1000": 8,
                "AR_s@1000": 9,
                "AR_m@1000": 10,
                "AR_l@1000": 11,
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f"metric item {metric_item} is not supported"
                        )

            local_metric_items = metric_items

            if metric == "proposal":
                coco_eval.params.useCats = 0
                coco_eval.evaluate()
                coco_eval.accumulate()

                redirect_string = io.StringIO()
                with contextlib.redirect_stdout(redirect_string):
                    coco_eval.summarize()
                print_log("\n" + redirect_string.getvalue(), logger=logger)

                if local_metric_items is None:
                    local_metric_items = [
                        "AR@100",
                        "AR@300",
                        "AR@1000",
                        "AR_s@1000",
                        "AR_m@1000",
                        "AR_l@1000",
                    ]

                for item in local_metric_items:
                    eval_results[item] = float(
                        f"{coco_eval.stats[coco_metric_names[item]]:.4f}"
                    )
                continue

            coco_eval.evaluate()
            coco_eval.accumulate()

            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                coco_eval.summarize()
            print_log("\n" + redirect_string.getvalue(), logger=logger)

            per_class_metrics = self._collect_per_class_metrics(coco_eval)
            if classwise:
                self._log_metric_table(
                    f"{metric} per-class AP/AP50",
                    per_class_metrics,
                    logger=logger,
                    include_map=False,
                )

            split_id, group_metrics = self._collect_group_metrics(per_class_metrics)
            if split_id is not None and group_metrics:
                summary_rows = []
                for group_name, group_info in group_metrics.items():
                    summary_rows.append(
                        dict(
                            name=group_name,
                            ap=group_info["mAP"],
                            ap50=group_info["mAP_50"],
                        )
                    )
                    eval_results[f"{metric}_{group_name}_mAP"] = float(
                        f"{group_info['mAP']:.4f}"
                    )
                    eval_results[f"{metric}_{group_name}_mAP_50"] = float(
                        f"{group_info['mAP_50']:.4f}"
                    )

                self._log_metric_table(
                    f"{metric} split{split_id} base/novel summary",
                    summary_rows,
                    logger=logger,
                    include_map=False,
                )
                for group_name, group_info in group_metrics.items():
                    self._log_metric_table(
                        f"{metric} split{split_id} {group_name} classes",
                        group_info["rows"],
                        logger=logger,
                        include_map=True,
                    )

            if local_metric_items is None:
                local_metric_items = [
                    "mAP",
                    "mAP_50",
                    "mAP_75",
                    "mAP_s",
                    "mAP_m",
                    "mAP_l",
                ]

            for metric_item in local_metric_items:
                key = f"{metric}_{metric_item}"
                eval_results[key] = float(
                    f"{coco_eval.stats[coco_metric_names[metric_item]]:.4f}"
                )
            ap = coco_eval.stats[:6]
            eval_results[f"{metric}_mAP_copypaste"] = (
                f"{ap[0]:.4f} {ap[1]:.4f} {ap[2]:.4f} "
                f"{ap[3]:.4f} {ap[4]:.4f} {ap[5]:.4f}"
            )

        return eval_results
