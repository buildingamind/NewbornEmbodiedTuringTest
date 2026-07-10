"""Compact CNN image encoder."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor
from .utils.pool import DeterministicAvgPool2d


class SmallCNN(HWCFeatureExtractor):
    """Compact CNN encoder for Isaac image observations."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        channels, height, width = image_channels_hw(observation_space)
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            DeterministicAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, channels, height, width)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(self._prepare_image(observations)))
