"""Shared tiny base SAR gfl model; method-specific head types are overridden locally."""
import os
import fsdet_configs.schema as _schema

angle_version = 'le90'
sar_num_classes = _schema.sar_base_num_classes
sar_bbox_head = dict(
    type='GFLHead',
    num_classes=sar_num_classes,
    in_channels=256,
    stacked_convs=4,
    feat_channels=256,
    anchor_generator=dict(
        type='AnchorGenerator',
        ratios=[1.0],
        octave_base_scale=8,
        scales_per_octave=1,
        strides=[8, 16, 32, 64, 128]),
    loss_cls=dict(
        type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
    loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
    reg_max=16,
    loss_bbox=dict(type='GIoULoss', loss_weight=2.0))
sar_rpn_head = None
sar_roi_head = None
sar_train_cfg = dict(
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
sar_test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)
model = dict(
    sar_bbox_head=sar_bbox_head,
    sar_rpn_head=sar_rpn_head,
    sar_roi_head=sar_roi_head,
    sar_train_cfg=sar_train_cfg,
    sar_test_cfg=sar_test_cfg)
del angle_version, sar_bbox_head, sar_num_classes, sar_roi_head, sar_rpn_head, sar_test_cfg, sar_train_cfg
sar_v6_bbox_head = dict(type='GFLAlignHead')
sar_align_roi_head = None
