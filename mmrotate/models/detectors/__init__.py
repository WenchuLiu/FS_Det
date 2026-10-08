# Copyright (c) OpenMMLab. All rights reserved.
from .base import RotatedBaseDetector
from .gfl_moe import GFLMoE
from .gliding_vertex import GlidingVertex
from .r3det import R3Det
from .redet import ReDet
from .roi_transformer import RoITransformer
from .rotate_faster_rcnn import RotatedFasterRCNN
from .rotated_fcos import RotatedFCOS
from .rotated_reppoints import RotatedRepPoints
from .rotated_retinanet import RotatedRetinaNet
from .s2anet import S2ANet
from .single_stage import RotatedSingleStageDetector
from .two_stage import RotatedTwoStageDetector
from .trisource_H1stage_R2stage_detector import TriSourceDetector
from .trisource_H1stage_R2stage_rgb_gfl_detector import TriSourceRGBGFLDetector
from .trisource_H1stage_R2stage_detector_kd import TriSourceKDDetector

from .trisource_H2stage_R2stage_detector import TriSourceTwoTwoDetector
from .trisource_H2stage_R1stage_detector import TriSourceTwoOneDetector
from .trisource_H1stage_R1stage_detector import TriSourceOneOneDetector

from .pre_trisource_align_detector_v1 import TriSourceAlignDetector
from .pre_trisource_align_detector_v2 import TriSourceAlignDetectorV2
from .pre_trisource_align_detector_v3 import TriSourceAlignDetectorV3
from .pre_trisource_align_detector_v4 import TriSourceAlignDetectorV4
from .pre_trisource_align_detector_v5 import TriSourceAlignDetectorV5
from .pre_trisource_align_detector_v6 import TriSourceAlignDetectorV6
from .pre_trisource_align_detector_v3_1 import TriSourceAlignDetectorV3_1
from .pre_trisource_align_detector_v7 import TriSourceAlignDetectorV7
from .pre_trisource_align_detector_v8 import TriSourceAlignDetectorV8
from .pre_trisource_align_detector_v9 import TriSourceAlignDetectorV9
from .pre_trisource_align_detector_v10 import TriSourceAlignDetectorV10
from .pre_trisource_align_detector_v11 import TriSourceAlignDetectorV11
from .pre_trisource_align_detector_v12 import TriSourceAlignDetectorV12
from .pre_trisource_align_detector_v13 import TriSourceAlignDetectorV13
from .pre_trisource_align_detector_v14 import TriSourceAlignDetectorV14
from .pre_trisource_align_detector_v15 import TriSourceAlignDetectorV15
from .pre_trisource_align_detector_v16 import TriSourceDetectorV16
from .pre_trisource_align_detector_ours2 import TriSourceDetectorV16Ours2
from .pre_trisource_align_detector_v16_rgb_gfl import TriSourceDetectorV16RGBGFL

__all__ = [
    'RotatedRetinaNet', 'RotatedFasterRCNN',  'RoITransformer',
    'GlidingVertex', 'GFLMoE', 'ReDet', 'R3Det', 'S2ANet', 'RotatedRepPoints',
    'RotatedBaseDetector', 'RotatedTwoStageDetector',
    'RotatedSingleStageDetector', 'RotatedFCOS','TriSourceDetector',
    'TriSourceRGBGFLDetector',
    'TriSourceTwoTwoDetector','TriSourceTwoOneDetector','TriSourceOneOneDetector', 
    'TriSourceKDDetector', 'TriSourceAlignDetector','TriSourceAlignDetectorV2',
    'TriSourceAlignDetectorV3', 'TriSourceAlignDetectorV4', 'TriSourceAlignDetectorV5', 
    'TriSourceAlignDetectorV6', 'TriSourceAlignDetectorV3_1',
    'TriSourceAlignDetectorV7','TriSourceAlignDetectorV8','TriSourceAlignDetectorV9',
    'TriSourceAlignDetectorV10', 'TriSourceAlignDetectorV11', 'TriSourceAlignDetectorV12',
    'TriSourceAlignDetectorV13', 'TriSourceAlignDetectorV14','TriSourceAlignDetectorV15',
    'TriSourceDetectorV16', 'TriSourceDetectorV16Ours2',
    'TriSourceDetectorV16RGBGFL'
]
