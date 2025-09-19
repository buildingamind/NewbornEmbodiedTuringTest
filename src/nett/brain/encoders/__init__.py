"""Simplifies imports for encoders"""

from ..utils.rllib_compat import BaseFeaturesExtractor, NatureCNN

from .resnet18 import Resnet18CNN
from .resnet10 import Resnet10CNN
from .dinov1 import DinoV1
from .dinov2 import DinoV2
from .sam import SegmentAnything
from .simplevit import SimpleViT
from .vit import ViT
from .cnnlstm import CNNLSTM
from .frozensimclr import FrozenSimCLR
from .multiinput import MultiInputEncoder
