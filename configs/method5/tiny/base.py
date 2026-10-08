"""Experiment parameters; common heads and datasets are inherited from _base_."""
import os
from fsdet_configs.schema import *

_base_ = [
    '../../_base_/models/fsdet_tiny_base_sar_' + sar_model_format + '.py',
    '../../_base_/datasets/fsdet_base_two_modal.py',
    '../../_base_/schedules/fsdet_schedule_v16.py',
    '../../_base_/default_runtime.py'
]
angle_version = 'le90'
sar_clip_model_dir = os.path.abspath(
    os.path.expanduser(
        os.environ.get('FSDET_SARCLIP_DIR',
                       'mmrotate/models/sar_clip/model_configs/ViT-B-16')))
align_cfg = dict(
    enable_backbone_align_inputs=True,
    auto_backbone_roi_from_gt=True,
    auto_backbone_text_from_aligner=True,
    backbone_roi_pool_size=7,
    backbone_roi_max_samples=128,
    in_dims=dict(sar=256, rgb=256, text=512),
    use_clip_fusion=True,
    text_encoder_type='sarclip',
    sar_clip_name='ViT-B-16',
    sar_clip_path=os.path.join(sar_clip_model_dir, 'vit_b_16_model.pth'),
    sar_clip_cache=sar_clip_model_dir,
    init_clip=True,
    is_base_training=True,
    ema_beta=0.001,
    vol_weight=0.5,
    use_llm_prompt=False,
    prompt_file='LLM_prompts/SARDet_SRSDD_category_descriptions.json',
    sar_category=list(sar_base_text_classes),
    rgb_category=list(rgb_text_classes),
    semantic_mapping=semantic_mapping,
    new_modality='sar')
routing_transfer_cfg = dict(
    base_classes=list(range(sar_base_num_classes)),
    novel_classes=list(sar_novel_text_classes),
    sar_class_names=list(sar_base_text_classes),
    prior_mode='semantic_topk',
    topk=3,
    temperature=0.07,
    loss_weight=0.1,
    collect_base_signatures=True,
    enable_transfer=False)
model = dict(
    type='TriSourceDetectorV16Ours2',
    train_modalities=['sar', 'rgb'],
    backbone=dict(
        type='ConvNeXtMoEV2',
        moe_indices=[[], [], [0, 2, 4, 6, 8], [0, 2]],
        arch='tiny',
        drop_path_rate=0.1,
        routing_fusion_cfg=dict(
            dataset_names=['sar', 'rgb'],
            shared_dim=64,
            roi_dim=256,
            text_dim=512,
            fuse_mode='concat',
            route_alpha=0.5,
            learnable_alpha=True,
            align_loss_weight=1.0,
            align_loss_type='contrastive',
            stop_align_grad_to_backbone=False,
            align_backbone_grad_scale=0.5,
            align_loss_temperature=0.07,
            align_loss_label_smoothing=0.05),
        moe_args=[
            dict(
                type='fusionorigingate',
                mid_scale=1,
                n_routed_experts=8,
                n_activated_experts=2,
                route_scale=1.0,
                gate_func_cfg=dict(type='CosineLogit'),
                loss_coef_cfg=dict(
                    balance_loss_coef=0.01, router_loss_coef=1.0),
                act_func='softmax',
                noisy_gating=True,
                use_dispatch=True)
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
        type='OrientedStandardRoIHead',
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
    routing_transfer_cfg=routing_transfer_cfg)
seed = 0
load_from = None
