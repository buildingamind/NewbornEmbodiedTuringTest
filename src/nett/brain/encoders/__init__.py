"""Simplifies imports for encoders"""

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

from .resnet18 import Resnet18CNN
from .resnet10 import Resnet10CNN
from .dinov1 import DinoV1
from .dinov2 import DinoV2
from .sam import SegmentAnything
from .vit import ViT
from .cnnlstm import CNNLSTM
from .sam import SegmentAnything
from .frozensimclr import FrozenSimCLR
