from .dota import *
from .builder import ROTATED_DATASETS
from .sardet_hbb_trisource import SARDet_hbb_trisource
from .sardet_dota_ifred import SARDetDotaIFRedDataset


@ROTATED_DATASETS.register_module()
class SARDetDotaIFRedNovelRGBDataset(SARDetDotaIFRedDataset):
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN', )

    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (255, 193, 193)]

    # =================== 请按实验需求填写 ===================
    BASE_CLASSES = ('small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter',)      # 你定义的旧类
    NOVEL_CLASSES = ('CAR', 'BUS',)   # 你定义的新类
    # =====================================================

    def _get_class_ids(self, class_names):
        """Convert class names to indices using self.CLASSES."""
        name_to_id = {name: i for i, name in enumerate(self.CLASSES)}
        return [name_to_id.get(name, -1) for name in class_names]  # Use get() to return -1 if name not found

    def _filter_results_by_class_ids(self, results, class_ids):
        """Return results in compact format: [img][local_class_id] = detections"""
        if not class_ids:
            return [[] for _ in results]
        
        filtered = []
        for img_res in results:
            # For each class_id, check if the class has results and extract them
            new_img_res = []
            for cls_id in class_ids:
                if cls_id < len(img_res):  # Check if the class ID exists in img_res
                    new_img_res.append(img_res[cls_id])
                else:
                    new_img_res.append([])  # If no result for that class, append empty list
            filtered.append(new_img_res)
        
        return filtered

    def _filter_annotations_by_class_ids(self, annotations, class_ids):
        if not class_ids:
            return [{'bboxes': np.zeros((0, 5)), 'labels': np.array([]), 'polygons': np.zeros((0, 8))} for _ in annotations]
        
        class_id_to_local = {global_id: local_id for local_id, global_id in enumerate(class_ids)}
        class_id_set = set(class_ids)
        filtered = []
        for ann in annotations:
            keep = np.array([label in class_id_set for label in ann['labels']], dtype=bool)
            if keep.any():
                global_labels = ann['labels'][keep]
                local_labels = np.array([class_id_to_local[label] for label in global_labels], dtype=np.int64)
                bboxes = ann['bboxes'][keep]
                polygons = ann['polygons'][keep] if 'polygons' in ann else np.zeros((0, 8), dtype=np.float32)
            else:
                bboxes = np.zeros((0, 5), dtype=np.float32)
                local_labels = np.array([], dtype=np.int64)
                polygons = np.zeros((0, 8), dtype=np.float32)

            new_ann = {
                'bboxes': bboxes,
                'labels': local_labels,  # 👈 关键：转为局部 ID
                'polygons': polygons,
                'bboxes_ignore': ann.get('bboxes_ignore', np.zeros((0, 5), dtype=np.float32)),
                'labels_ignore': ann.get('labels_ignore', np.array([], dtype=np.int64)),
            }
            filtered.append(new_ann)
        return filtered

    def _evaluate_group_custom_metrics(self, results, annotations, class_ids, group_name, logger=None):
        if not class_ids:
            return {}
        total_cd = total_cc = total_gt = total_pred = 0

        for idx in range(len(results)):
            gt_info = annotations[idx]
            gt_bboxes = gt_info['bboxes']
            gt_labels = gt_info['labels']

            pred_result = results[idx]

            # GT annotations in this group
            gt_anns = []
            for bbox, label in zip(gt_bboxes, gt_labels):
                if label in class_ids:
                    gt_anns.append({"bbox": bbox, "category_id": int(label), "score": 1.0})

            # Prediction annotations in this group
            pred_anns = []
            for class_id, class_preds in enumerate(pred_result):
                if class_id in class_ids and len(class_preds) > 0:
                    for pred in class_preds:
                        bbox = pred[:5]
                        score = pred[5] if len(pred) > 5 else 0.5
                        pred_anns.append({"bbox": bbox, "category_id": class_id, "score": score})

            filtered_pred_anns = self._apply_score_filter_and_nms(pred_anns)
            cd, cc = self._evaluate_single_image(filtered_pred_anns, gt_anns)

            total_cd += cd
            total_cc += cc
            total_gt += len(gt_anns)
            total_pred += len(filtered_pred_anns)

        detection_rate = total_cd / total_gt if total_gt > 0 else 0.0
        recognition_rate = total_cc / total_gt if total_gt > 0 else 0.0
        false_alarm_rate = (total_pred - total_cd) / total_pred if total_pred > 0 else 0.0

        print_log(f"[{group_name}] Detection Rate: {detection_rate:.4f}", logger=logger)
        print_log(f"[{group_name}] Recognition Rate: {recognition_rate:.4f}", logger=logger)
        print_log(f"[{group_name}] False Alarm Rate: {false_alarm_rate:.4f}", logger=logger)

        return {
            f"{group_name}_detection_rate": detection_rate,
            f"{group_name}_recognition_rate": recognition_rate,
            f"{group_name}_false_alarm_rate": false_alarm_rate
        }

    def evaluate(self,
                results,
                metric='mAP',
                logger=None,
                proposal_nums=(100, 300, 1000),
                iou_thr=[0.5],
                scale_ranges=None,
                nproc=4):
        if not isinstance(metric, str):
            metric = metric[0]
        if metric != 'mAP':
            raise NotImplementedError("Only mAP is supported.")

        # 获取所有类别的 class IDs
        all_ids = [i for i in range(len(self.CLASSES))]

        # 获取 base 类和 novel 类的有效类 ID
        base_ids = [i for i in self._get_class_ids(self.BASE_CLASSES) if i != -1]
        novel_ids = [i for i in self._get_class_ids(self.NOVEL_CLASSES) if i != -1]

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = {}

        # --- Overall Evaluation (所有类) ---
        all_res = self._filter_results_by_class_ids(results, all_ids)
        all_ann = self._filter_annotations_by_class_ids(annotations, all_ids)
        
        print("Evaluating All Classes:", [self.CLASSES[i] for i in all_ids])  # 调试输出
        all_mAP, _ = eval_rbbox_map(
            all_res, all_ann,
            dataset=[self.CLASSES[i] for i in all_ids],
            logger=logger, nproc=nproc
        )
        eval_results['overall_mAP'] = all_mAP
        eval_results.update(self._evaluate_group_custom_metrics(all_res, all_ann, all_ids, 'overall', logger))

        # # --- Base Evaluation (旧类) ---
        # if base_ids:
        #     base_res = self._filter_results_by_class_ids(results, base_ids)
        #     base_ann = self._filter_annotations_by_class_ids(annotations, base_ids)
            
        #     print("Evaluating Base Classes:", [self.CLASSES[i] for i in base_ids])  # 调试输出
        #     base_mAP, _ = eval_rbbox_map(
        #         base_res, base_ann,
        #         dataset=[self.CLASSES[i] for i in base_ids],
        #         logger=logger, nproc=nproc
        #     )
        #     eval_results['base_mAP'] = base_mAP
        #     eval_results.update(self._evaluate_group_custom_metrics(base_res, base_ann, base_ids, 'base', logger))

        # # --- Novel Evaluation (新类) ---
        # if novel_ids:
        #     novel_res = self._filter_results_by_class_ids(results, novel_ids)
        #     novel_ann = self._filter_annotations_by_class_ids(annotations, novel_ids)
            
        #     print("Evaluating Novel Classes:", [self.CLASSES[i] for i in novel_ids])  # 调试输出
        #     novel_mAP, _ = eval_rbbox_map(
        #         novel_res, novel_ann,
        #         dataset=[self.CLASSES[i] for i in novel_ids],
        #         logger=logger, nproc=nproc
        #     )
        #     eval_results['novel_mAP'] = novel_mAP
        #     eval_results.update(self._evaluate_group_custom_metrics(novel_res, novel_ann, novel_ids, 'novel', logger))

        # --- Summary ---
        print_log("=" * 50, logger=logger)
        
        # 打印整体结果
        print_log(f"Overall mAP: {eval_results.get('overall_mAP', 0):.4f}", logger=logger)
        if 'overall_detection_rate' in eval_results:
            print_log(f"Overall Detection Rate: {eval_results['overall_detection_rate']:.4f}", logger=logger)
            print_log(f"Overall Recognition Rate: {eval_results['overall_recognition_rate']:.4f}", logger=logger)
            print_log(f"Overall False Alarm Rate: {eval_results['overall_false_alarm_rate']:.4f}", logger=logger)

        # # 打印 Base 类的结果
        # if base_ids:
        #     print_log(f"Base Classes: {self.BASE_CLASSES}", logger=logger)
        #     print_log(f"Base mAP: {eval_results.get('base_mAP', 0):.4f}", logger=logger)
        #     if 'base_detection_rate' in eval_results:
        #         print_log(f"Base Detection Rate: {eval_results['base_detection_rate']:.4f}", logger=logger)
        #         print_log(f"Base Recognition Rate: {eval_results['base_recognition_rate']:.4f}", logger=logger)
        #         print_log(f"Base False Alarm Rate: {eval_results['base_false_alarm_rate']:.4f}", logger=logger)

        # # 打印 Novel 类的结果
        # if novel_ids:
        #     print_log(f"Novel Classes: {self.NOVEL_CLASSES}", logger=logger)
        #     print_log(f"Novel mAP: {eval_results.get('novel_mAP', 0):.4f}", logger=logger)
        #     if 'novel_detection_rate' in eval_results:
        #         print_log(f"Novel Detection Rate: {eval_results['novel_detection_rate']:.4f}", logger=logger)
        #         print_log(f"Novel Recognition Rate: {eval_results['novel_recognition_rate']:.4f}", logger=logger)
        #         print_log(f"Novel False Alarm Rate: {eval_results['novel_false_alarm_rate']:.4f}", logger=logger)

        print_log(f"Score threshold: {self.other_score_threshold}, NMS threshold: {self.other_nms_threshold}", logger=logger)
        print_log("=" * 50, logger=logger)

        return eval_results





@ROTATED_DATASETS.register_module()
class SARDetDotaIFRedProj_RGB_Aircraft(SARDetDotaIFRedNovelRGBDataset):
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
               'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
               'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
               'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
               'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
               'other-aircraft', 'fighter-aircraft', 'tanker-aircraft', 'transport-aircraft',
               'anti-submarine-warfare-aircraft', 'helicopter-aircraft', 'reconnaissance-aircraft',
               'trainer-aircraft', 'airborne-early-warning-aircraft','bomber-aircraft',
               'search-and-rescue-aircraft', )
        
    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100),
               (0, 51, 153), (220, 220, 60),]
    
    # ==================== 配置你的 base / novel 类别 ====================
    BASE_CLASSES = ('small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
               'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
               'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', )
    NOVEL_CLASSES = ('other-aircraft', 'fighter-aircraft', 'tanker-aircraft', 'transport-aircraft',
               'anti-submarine-warfare-aircraft', 'helicopter-aircraft', 'reconnaissance-aircraft',
               'trainer-aircraft', 'airborne-early-warning-aircraft','bomber-aircraft',
               'search-and-rescue-aircraft',)
    # ===================================================================


@ROTATED_DATASETS.register_module()
class SARDetDotaIFRedProj_RGB_Ship(SARDetDotaIFRedNovelRGBDataset):
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
         'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
         'Military-ship',)
        
    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (255, 193, 193), (0, 51, 153), (0, 51, 153), (0, 51, 153),
               (0, 51, 153), (0, 51, 153), (0, 51, 153), (0, 51, 153)]
    
    # ==================== 配置你的 base / novel 类别 ====================
    BASE_CLASSES = ('small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
               'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
               'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', )
    NOVEL_CLASSES = ('Military-Ship',)
    # ===================================================================

@ROTATED_DATASETS.register_module()
class SARDetDotaIFRedProj_RGB_Vehicle(SARDetDotaIFRedNovelRGBDataset):
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
         'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
         'CAR_RGB',  'BUS_RGB',  'FREIGHT', 'FREIGHT-MILITARY', 'TRUCK_RGB', 'HMS_RGB',)
    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
               (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
               (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
               (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
               (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
               (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), ]
    # ==================== 配置你的 base / novel 类别 ====================
    BASE_CLASSES = ('small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
               'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
               'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter',)
    NOVEL_CLASSES = ('CAR_RGB', 'HMS_RGB', 'FREIGHT', 'FREIGHT-MILITARY', 'TRUCK_RGB', 'BUS_RGB')
    # ===================================================================

@ROTATED_DATASETS.register_module()
class Proj_rgb_aircraft_trisource(SARDet_hbb_trisource):
    """SAR Detection dataset with horizontal bounding boxes from three sources.
    
    This dataset supports custom evaluation metrics including detection rate,
    recognition rate, and false alarm rate.
    """
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
               'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
               'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
               'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
               'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
               'other-aircraft', 'fighter-aircraft', 'tanker-aircraft', 'transport-aircraft',
               'anti-submarine-warfare-aircraft', 'helicopter-aircraft', 'reconnaissance-aircraft',
               'trainer-aircraft', 'airborne-early-warning-aircraft','bomber-aircraft',
               'search-and-rescue-aircraft', )
    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
            (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
            (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
            (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
            (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
            (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
            (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100),
            (0, 51, 153), (220, 220, 60),]

@ROTATED_DATASETS.register_module()
class Proj_rgb_ship_trisource(SARDet_hbb_trisource):
    """SAR Detection dataset with horizontal bounding boxes from three sources.
    
    This dataset supports custom evaluation metrics including detection rate,
    recognition rate, and false alarm rate.
    """
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
         'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
         'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
         'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
         'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
         'Military-ship',)
        
    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
                (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
                (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
                (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
                (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
                (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
                (255, 193, 193), (0, 51, 153), (0, 51, 153), (0, 51, 153),
                (0, 51, 153), (0, 51, 153), (0, 51, 153), (0, 51, 153)]


@ROTATED_DATASETS.register_module()
class Proj_rgb_vehicle_trisource(SARDet_hbb_trisource):
    """SAR Detection dataset with horizontal bounding boxes from three sources.
    
    This dataset supports custom evaluation metrics including detection rate,
    recognition rate, and false alarm rate.
    """
    CLASSES = ('ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor', 
        'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor', 'tennis-court',
        'soccer-ball-field', 'ground-track-field', 'baseball-diamond', 'swimming-pool', 
        'roundabout', 'basketball-court', 'storage-tank', 'Bridge', 'helicopter', 
        'CAR', 'BUS', 'FERIGHT_CAR', 'TRUCK', 'VAN',
        'CAR_RGB', 'HMS_RGB', 'FREIGHT', 'FREIGHT-MILITARY', 'TRUCK_RGB', 
        'BUS_RGB')
        
    PALETTE = [(220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220), 
                (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32), 
                (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), 
                (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
                (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
                (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), 
                (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30), 
                (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), 
                (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255), ]

