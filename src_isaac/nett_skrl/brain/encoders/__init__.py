"""skrl-compatible image encoders for the Isaac backend."""

from __future__ import annotations

from .base import NETTFeatureExtractor
from .compact_3dcnn import Compact3DCNN
from .compact_3dcnn_pretrained import Compact3DCNNPretrained
from .compact_cnn import CompactCNN
from .compact_cnn_pretrained import CompactCNNPretrained
from .compact_vivit import CompactViViT
from .compact_vit import CompactViT
from .guess_what_moves import GuessWhatMoves
from .hwc_feature_extractor import HWCFeatureExtractor
from .nature_cnn import NatureCNN
from .resnet10_cnn import Resnet10CNN
from .resnet18_cnn import Resnet18CNN
from .simclr_cltt import SimCLRCLTT
from .small_cnn import SmallCNN

__all__ = [
    "Compact3DCNN",
    "CompactCNN",
    "CompactCNNPretrained",
    "Compact3DCNNPretrained",
    "CompactViT",
    "CompactViViT",
    "GuessWhatMoves",
    "HWCFeatureExtractor",
    "NETTFeatureExtractor",
    "NatureCNN",
    "Resnet10CNN",
    "Resnet18CNN",
    "SimCLRCLTT",
    "SmallCNN",
]
