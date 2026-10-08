"""Shared tiny base SAR dota model; method-specific head types are overridden locally."""
import os
import fsdet_configs.schema as _schema

angle_version = 'le90'
sar_num_classes = _schema.sar_base_num_classes
sar_bbox_head = None
sar_rpn_head = dict(
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
    loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
    loss_bbox=dict(
        type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0))
sar_roi_head = dict(
    type='OrientedStandardRoIHead',
    bbox_roi_extractor=dict(
        type='RotatedSingleRoIExtractor',
        roi_layer=dict(
            type='RoIAlignRotated', out_size=7, sample_num=2, clockwise=True),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='RotatedShared2FCBBoxHead',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=sar_num_classes,
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
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
sar_train_cfg = dict(
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
        debug=False))
sar_test_cfg = dict(
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
        max_per_img=2000))
model = dict(
    sar_bbox_head=sar_bbox_head,
    sar_rpn_head=sar_rpn_head,
    sar_roi_head=sar_roi_head,
    sar_train_cfg=sar_train_cfg,
    sar_test_cfg=sar_test_cfg)
del angle_version, sar_bbox_head, sar_num_classes, sar_roi_head, sar_rpn_head, sar_test_cfg, sar_train_cfg
sar_v6_bbox_head = None
sar_align_roi_head = dict(type='OrientedStandardRoIAlignHead')
