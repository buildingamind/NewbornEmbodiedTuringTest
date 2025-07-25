"""Simplifies imports for encoders"""

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN

from .resnet18 import Resnet18CNN
from .resnet10 import Resnet10CNN
from .dinov1 import DinoV1
from .dinov2 import DinoV2
from .sam import SegmentAnything
from .simplevit import SimpleViT
from .vit import ViT
from .frozensimclr import FrozenSimCLR
from .multiinput import MultiInputEncoder
from .dreamerv3 import DreamerV3