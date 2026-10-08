# Copyright (c) OpenMMLab. All rights reserved.
from .enn import (build_enn_divide_feature, build_enn_feature,
                  build_enn_norm_layer, build_enn_trivial_feature, ennAvgPool,
                  ennConv, ennInterpolate, ennMaxPool, ennReLU, ennTrivialConv)
from .orconv import ORConv2d
from .ripool import RotationInvariantPooling
from .aligner import PrototypeAligner, TextPrototypeAligner
from .ETF import ETFAuxiliaryClassifier
from .aligner_v2 import PrototypeAlignerV2, TextPrototypeAlignerV2
from .aligner_v3 import PrototypeAlignerV3, TextPrototypeAlignerV3
from .alignerv1_1 import PrototypeAlignerV1_1, TextPrototypeAlignerV1_1
from .aligner_v4 import PrototypeAlignerV4, TextPrototypeAlignerV4
from .aligner_v5 import PrototypeAlignerV5, TextPrototypeAlignerV5
from .aligner_v6 import TextPrototypeAlignerV6
from .aligner_v7 import TextPrototypeAlignerV7
from .aligner_v8 import TextPrototypeAlignerV8
from .aligner_v9 import TextPrototypeAlignerV9
from .aligner_v10 import TextPrototypeAlignerV10
from .knowledge_injector import TwoStageProtoInjector
from .knowledge_injector_v2 import FrequencySpectralInjector

__all__ = [
    'ORConv2d', 'RotationInvariantPooling', 'ennConv', 'ennReLU', 'ennAvgPool',
    'ennMaxPool', 'ennInterpolate', 'build_enn_divide_feature',
    'build_enn_feature', 'build_enn_norm_layer', 'build_enn_trivial_feature',
    'ennTrivialConv', 'PrototypeAligner', 'TextPrototypeAligner', 'ETFAuxiliaryClassifier',
    'PrototypeAlignerV2', 'TextPrototypeAlignerV2', 'PrototypeAlignerV3', 'TextPrototypeAlignerV3',
    'PrototypeAlignerV1_1', 'TextPrototypeAlignerV1_1','PrototypeAlignerV4', 'TextPrototypeAlignerV4',
    'PrototypeAlignerV5', 'TextPrototypeAlignerV5','TextPrototypeAlignerV6','TextPrototypeAlignerV7','TextPrototypeAlignerV8',
    'TextPrototypeAlignerV9', 'TwoStageProtoInjector', 'FrequencySpectralInjector', 'TextPrototypeAlignerV10'
]
