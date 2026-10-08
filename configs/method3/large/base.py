"""Experiment parameters; common heads and datasets are inherited from _base_."""
import os
from fsdet_configs.schema import *

_base_ = [
    '../../_base_/models/fsdet_large_base_sar_' + sar_model_format + '.py',
    '../../_base_/models/rsclip_paths.py',
    '../../_base_/datasets/fsdet_base_two_modal.py',
    '../../_base_/schedules/fsdet_schedule.py',
    '../../_base_/default_runtime.py'
]
angle_version = 'le90'
clip_model_path = os.path.abspath(
    os.path.expanduser(os.environ.get('FSDET_METHOD3_CLIP_MODEL_PATH', '')))
clip_processor_path = os.path.abspath(
    os.path.expanduser(
        os.environ.get('FSDET_METHOD3_CLIP_PROCESSOR_PATH')
        or clip_model_path))
align_cfg = dict(
    in_dims=dict(sar=768, rgb=768, text=512),
    modalities=['sar', 'rgb', 'text'],
    semantic_mapping=semantic_mapping,
    align_dim=768,
    ema_beta=0.05,
    vol_weight=0.5,
    init_clip=True,
    is_base_training=True,
    use_llm_prompts=False,
    text_encoder_type=os.environ.get('FSDET_METHOD3_TEXT_ENCODER',
                                    'huggingface'),
    clip_model_name=clip_model_path,
    clip_processor_name=clip_processor_path,
    prompt_file='LLM_prompts/fsdet_category_descriptions.json',
    base_modal_categories=base_modal_categories,
    all_modal_categories=all_modal_categories)
model = dict(
    type='TriSourceAlignDetectorV6',
    backbone=dict(
        type='ConvNeXtMoE',
        moe_indices=[[], [], [2, 6, 10, 14, 18, 22, 26], [0, 2]],
        arch='large',
        drop_path_rate=0.1,
        moe_args=[
            dict(
                type='focusgate',
                mid_scale=1,
                n_routed_experts=12,
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
                    sar=[0, 1, 2, 3, 8, 9, 10, 11],
                    rgb=[4, 5, 6, 7, 8, 9, 10, 11],
                    ifr=[8, 9, 10, 11]),
                share_expert_indices=[],
                specific_mask_value=1.0,
                share_mask_value=0,
                sup_share_mask=True,
                act_func='softmax')
        ]),
    neck=dict(
        type='MultitaskFPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=768,
        extra_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    rgb_rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=768,
        feat_channels=768,
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
            out_channels=768,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=768,
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
    train_modalities=['sar', 'rgb'],
    sar_bbox_head={{_base_.sar_v6_bbox_head}},
    sar_roi_head={{_base_.sar_align_roi_head}})
load_from = None
