"""Experiment parameters; common heads and datasets are inherited from _base_."""
import os
from fsdet_configs.schema import *

_base_ = [
    '../../_base_/models/fsdet_tiny_finetune_sar_' + sar_model_format + '.py',
    '../../_base_/models/rsclip_paths.py',
    '../../_base_/datasets/fsdet_100shot_sar.py',
    '../../_base_/schedules/fsdet_finetune_100_sar.py',
    '../../_base_/default_runtime.py'
]
angle_version = 'le90'
# Custom detection/recognition/false-alarm metrics; edit per experiment.
# These are dataset thresholds, separate from model score/NMS and mAP IoU.
other_score_threshold = 0.3
other_iou_threshold = 0.3
other_nms_threshold = dict(hbb_json=0.6, dota=0.3)[sar_ann_format]
data = dict(
    val=dict(
        other_score_threshold=other_score_threshold,
        other_iou_threshold=other_iou_threshold,
        other_nms_threshold=other_nms_threshold),
    test=dict(
        other_score_threshold=other_score_threshold,
        other_iou_threshold=other_iou_threshold,
        other_nms_threshold=other_nms_threshold))
clip_model_path = os.path.abspath(
    os.path.expanduser(os.environ.get('FSDET_METHOD3_CLIP_MODEL_PATH', '')))
clip_processor_path = os.path.abspath(
    os.path.expanduser(
        os.environ.get('FSDET_METHOD3_CLIP_PROCESSOR_PATH')
        or clip_model_path))
align_cfg = dict(
    is_base_training=False,
    init_clip=True,
    novel_class_ids=novel_sar_label_ids,
    semantic_mapping=dict(),
    modalities=['sar', 'text'],
    in_dims=dict(sar=256, text=512),
    text_encoder_type=os.environ.get('FSDET_METHOD3_TEXT_ENCODER',
                                    'huggingface'),
    clip_model_name=clip_model_path,
    clip_processor_name=clip_processor_path,
    base_modal_categories=dict(sar=sar_base_text_classes),
    all_modal_categories=dict(sar=sar_all_text_classes))
model = dict(
    type='TriSourceAlignDetectorV6',
    backbone=dict(
        type='ConvNeXtMoE',
        moe_indices=[[], [], [0, 2, 4, 6, 8], [0, 2]],
        arch='tiny',
        drop_path_rate=0.1,
        moe_args=[
            dict(
                type='focusgate',
                mid_scale=1,
                n_routed_experts=8,
                n_activated_experts=3,
                n_preferred=4,
                route_scale=1.0,
                use_token=False,
                token_num=4,
                gate_func_cfg=dict(type='ModalityAwareLogit'),
                fusion_cfg=dict(num_heads=8),
                loss_coef_cfg=dict(
                    balance_loss_coef=0.01,
                    router_loss_coef=1.0,
                    modal_pref_loss_coef=0.01),
                specific_expert_indices=dict(
                    sar=[0, 1, 6, 7], rgb=[2, 3, 6, 7], ifr=[4, 5, 6, 7]),
                share_expert_indices=[],
                specific_mask_value=1.0,
                share_mask_value=0,
                sup_share_mask=True,
                act_func='softmax')
        ],
        init_cfg=dict(
            type='Pretrained',
            reformat=True,
            prefix='backbone',
            checkpoint='pretrained/convnext_tiny_22k_224_mmpretrain.pth')),
    neck=dict(
        type='MultitaskFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        extra_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    rgb_rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version='le90',
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range='le90',
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    rgb_roi_head=dict(
        type='OrientedStandardRoIAlignHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=rgb_num_classes,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range='le90',
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    ifr_rpn_head=None,
    ifr_roi_head=None,
    rgb_train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                gpu_assign_thr=600,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                gpu_assign_thr=600,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    rgb_test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)),
    ifr_train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                gpu_assign_thr=600,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                gpu_assign_thr=600,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    ifr_test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000)),
    align_cfg=align_cfg,
    sar_bbox_head={{_base_.sar_v6_bbox_head}},
    sar_roi_head={{_base_.sar_align_roi_head}})
load_from = 'work_dirs/method3/tiny/base/expanded_for_ft.pth'
