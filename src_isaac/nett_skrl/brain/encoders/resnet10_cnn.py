"""ResNet-10-style image encoder."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from ...observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor
from .utils.residual_block import ResidualBlock


class Resnet10CNN(HWCFeatureExtractor):
    """Efficient ResNet-style feature extractor matching the legacy medium role."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        channels, _, _ = image_channels_hw(observation_space)
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(self._prepare_image(observations))
