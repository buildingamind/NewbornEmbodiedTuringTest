"""Compact CNN image encoder (~187 K encoder parameters).

Three convolutional layers with an adaptive 4×4 spatial pool followed by a
single linear projection. Designed to stay well under 600 K total model
parameters when combined with PPO actor/critic MLP heads.

Architecture mirrors the empirically validated ``SmallCNN`` from the NETT
skrl stack (same 4×4 pool, same activation pattern) while keeping
``features_dim`` at 128 so the downstream MLP trunks remain small.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor
from .utils.pool import DeterministicAvgPool2d


class CompactCNN(HWCFeatureExtractor):
    """Lightweight 3-layer CNN encoder for NETT visual RL.

    Parameter budget (64×64 RGB, features_dim=128):
        Conv layers  :  ~56 K
        Linear head  : ~131 K
        Total encoder: ~187 K
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 128, **_):
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
