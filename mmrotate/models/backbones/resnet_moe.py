from mmdet.models.backbones.resnet import ResNet

from .gates import MMSplitGate, OriginGate, FocusGate, ShareGate
from .utils import build_LayerNorm2d_layer


GATES = dict(
    mmsplitgate=MMSplitGate,
    focusgate=FocusGate,
    origingate=OriginGate,
    sharegate=ShareGate
)
