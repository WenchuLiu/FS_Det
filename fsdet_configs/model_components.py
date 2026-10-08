"""Reusable detector component configs for the supported FS_Det heads."""

import copy


def sar_two_stage_components(num_classes):
    """Return a horizontal-box RPN + RoI head for the SAR branch."""
    return dict(
        sar_bbox_head=None,
        sar_rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
        sar_roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[8, 16, 32, 64]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
        sar_train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
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
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        sar_test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)))


def sar_rotated_two_stage_components(num_classes, angle_version='le90'):
    """Return an oriented RPN + RoI head for SAR DOTA polygon annotations."""
    return dict(
        sar_bbox_head=None,
        sar_rpn_head=dict(
            type='OrientedRPNHead',
            in_channels=256,
            feat_channels=256,
            version=angle_version,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='MidpointOffsetCoder',
                angle_range=angle_version,
                target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(
                type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
        sar_roi_head=dict(
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
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHAOBBoxCoder',
                    angle_range=angle_version,
                    norm_factor=None,
                    edge_swap=True,
                    proj_xy=True,
                    target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                    target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
        sar_train_cfg=dict(
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
        sar_test_cfg=dict(
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
                max_per_img=2000)))


def sar_two_stage_components_for_mode(mode, num_classes, angle_version='le90'):
    if mode == 'two_stage':
        return sar_two_stage_components(num_classes)
    if mode == 'rotated_two_stage':
        return sar_rotated_two_stage_components(num_classes, angle_version)
    raise ValueError(f'Unsupported SAR two-stage head mode: {mode!r}.')


def finetune_model_overrides(modality,
                             sar_head_mode,
                             sar_num_classes,
                             rgb_num_classes,
                             sar_align_cfg=None,
                             routing_transfer_cfg=None):
    """Build the small model override shared by all 18/60/100-shot configs."""
    if modality == 'sar':
        if sar_head_mode in ('two_stage', 'rotated_two_stage'):
            overrides = dict(
                train_modalities=['sar'],
                sar_roi_head=dict(bbox_head=dict(num_classes=sar_num_classes)))
        else:
            overrides = dict(
                train_modalities=['sar'],
                sar_bbox_head=dict(num_classes=sar_num_classes))
        # Fine-tuning receives only the selected sensor stream. Disable the
        # base cross-modal aligner unless the method supplies a SAR/text config.
        overrides['align_cfg'] = copy.deepcopy(sar_align_cfg)
        if routing_transfer_cfg is not None:
            overrides['routing_transfer_cfg'] = copy.deepcopy(
                routing_transfer_cfg)
        return overrides

    if modality != 'rgb':
        raise ValueError('Fine-tuning modality must be "sar" or "rgb".')

    overrides = dict(
        train_modalities=['rgb'],
        align_cfg=None,
        rgb_roi_head=dict(bbox_head=dict(num_classes=rgb_num_classes)))
    if routing_transfer_cfg is not None:
        disabled_routing_cfg = copy.deepcopy(routing_transfer_cfg)
        disabled_routing_cfg['enable_transfer'] = False
        disabled_routing_cfg['collect_base_signatures'] = False
        overrides['routing_transfer_cfg'] = disabled_routing_cfg
    return overrides
