"""Model overrides shared by the five ConvNeXt MoE Large configurations."""

import os


MOE_INDICES = [[], [], [2, 6, 10, 14, 18, 22, 26], [0, 2]]
BACKBONE_CHANNELS = [192, 384, 768, 1536]
FPN_CHANNELS = 768


def large_model_overrides(sar_head_mode, v2=False):
    """Adapt backbone, FPN, detector heads and alignment feature widths."""
    backbone = dict(
        arch='large',
        moe_indices=MOE_INDICES,
        # The bundled Tiny checkpoint cannot initialize a Large backbone.
        init_cfg=None,
    )
    checkpoint = os.environ.get('FSDET_CONVNEXT_LARGE_CKPT')
    if checkpoint:
        backbone['init_cfg'] = dict(
            type='Pretrained', reformat=True, prefix='backbone',
            checkpoint=checkpoint)
    if v2:
        backbone['routing_fusion_cfg'] = dict(roi_dim=FPN_CHANNELS)
        backbone['moe_args'] = [dict(
            type='fusionorigingate', mid_scale=1,
            n_routed_experts=12, n_activated_experts=3,
            route_scale=1.0, gate_func_cfg=dict(type='CosineLogit'),
            loss_coef_cfg=dict(balance_loss_coef=1e-2,
                               router_loss_coef=1.0),
            act_func='softmax', noisy_gating=True, use_dispatch=True)]
    else:
        # Preserve the SAR/RGB preference structure while covering 12 experts.
        backbone['moe_args'] = [dict(
            type='focusgate', mid_scale=1,
            n_routed_experts=12, n_activated_experts=3, n_preferred=4,
            route_scale=1.0, use_token=False, token_num=4,
            gate_func_cfg=dict(type='ModalityAwareLogit'),
            fusion_cfg=dict(num_heads=8),
            loss_coef_cfg=dict(balance_loss_coef=1e-2,
                               router_loss_coef=1.0,
                               modal_pref_loss_coef=1e-2),
            specific_expert_indices=dict(
                sar=[0, 1, 2, 3, 8, 9, 10, 11],
                rgb=[4, 5, 6, 7, 8, 9, 10, 11],
                ifr=[8, 9, 10, 11]),
            share_expert_indices=[], specific_mask_value=1.0,
            share_mask_value=0, sup_share_mask=True,
            act_func='softmax')]

    model = dict(
        backbone=backbone,
        neck=dict(in_channels=BACKBONE_CHANNELS, out_channels=FPN_CHANNELS),
        sar_bbox_head=dict(in_channels=FPN_CHANNELS,
                           feat_channels=FPN_CHANNELS),
        rgb_rpn_head=dict(in_channels=FPN_CHANNELS,
                          feat_channels=FPN_CHANNELS),
        rgb_roi_head=dict(
            bbox_roi_extractor=dict(out_channels=FPN_CHANNELS),
            bbox_head=dict(in_channels=FPN_CHANNELS)),
        align_cfg=dict(in_dims=dict(sar=FPN_CHANNELS, rgb=FPN_CHANNELS)),
    )
    if sar_head_mode != 'gfl':
        model['sar_bbox_head'] = None
        model['sar_rpn_head'] = dict(
            in_channels=FPN_CHANNELS, feat_channels=FPN_CHANNELS)
        model['sar_roi_head'] = dict(
            bbox_roi_extractor=dict(out_channels=FPN_CHANNELS),
            bbox_head=dict(in_channels=FPN_CHANNELS))
    return model
