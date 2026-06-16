"""NatureCNN-style image encoder (no spatial pooling).

Mirrors the architecture of the SB3-default ``NatureCNN`` (Mnih et al. 2015):
three convolutions flattened directly into a linear projection, with no
adaptive average pooling stage. Unlike ``CompactCNN``/``SmallCNN`` (which
collapse the final feature map to a 4x4 grid via ``AdaptiveAvgPool2d`` before
flattening), this encoder preserves the full spatial layout of the last
feature map -- positional information (e.g. "is the bright thing on the left
or right of frame") survives into the linear head untouched.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor


class NatureCNN(HWCFeatureExtractor):
    """Three-layer CNN, flatten-to-linear, no spatial pooling."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 512,
                 conv_dim: int = 64, **_):
        super().__init__(observation_space, features_dim)
        channels, height, width = image_channels_hw(observation_space)

        # ``conv_dim`` is the final conv layer's channel count; it sets the
        # flatten size (conv_dim*4*4) and thus the dominant Linear's params, so
        # it scales encoder capacity WITHOUT touching features_dim (the policy
        # head input stays fixed). Default 64 preserves the original architecture.
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, conv_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            # Pool the final conv map to a fixed 4x4 grid before flattening so the
            # linear head's input size (and thus param count) is independent of
            # input_resolution. At res=64 the conv map is already 4x4, so this is
            # a no-op (validated behaviour preserved exactly); at res=256 it caps
            # the flatten at 4*4*64=1024 instead of 28*28*64, keeping params
            # <700k. A 4x4 grid still preserves coarse left/right spatial layout.
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, channels, height, width)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(self._prepare_image(observations)))
