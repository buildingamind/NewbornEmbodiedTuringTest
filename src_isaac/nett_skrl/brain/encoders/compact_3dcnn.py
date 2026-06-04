"""Compact 3D CNN encoder for 2-frame stacked observations (~188 K parameters).

Expects a 2-frame FrameStack body wrapper so the observation arrives as
(C*2, H, W) in CHW format (6 channels for RGB). Internally reshapes to
(B, C, T, H, W) = (B, 3, 2, H, W), applies one Conv3d layer to capture
spatio-temporal patterns, then continues with standard 2D convolutions.

Architecture follows Ji et al. (2013) "3D Convolutional Neural Networks for
Human Action Recognition" adapted for the compact-model setting.

Parameter budget (64×64 RGB 2-frame input, features_dim=128):
    Conv3d       :   ~2 K
    2×Conv2d     :  ~55 K
    Linear head  : ~131 K
    Total encoder: ~188 K
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor


class Compact3DCNN(HWCFeatureExtractor):
    """3D CNN encoder that exploits temporal information from 2 stacked frames.

    The first Conv3d layer collapses the temporal dimension (kernel depth = num_frames),
    after which the network continues as a standard 2D CNN. This design is
    parameter-efficient: the 3D conv adds only ~1.7 K parameters over a 2D
    equivalent while capturing short-range motion cues.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 128,
        num_frames: int = 2,
        **_,
    ) -> None:
        super().__init__(observation_space, features_dim)
        total_channels, height, width = image_channels_hw(observation_space)
        self.num_frames = int(num_frames)
        self.base_channels = total_channels // self.num_frames  # 3 for RGB

        # Temporal-spatial convolution: T×H×W kernel collapses time to 1
        self.conv3d = nn.Conv3d(
            self.base_channels, 32,
            kernel_size=(self.num_frames, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

        # 2-D spatial stages
        self.cnn2d = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Compute flat size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, self.base_channels, self.num_frames, height, width)
            n_flat = self.cnn2d(self.conv3d(dummy).squeeze(2)).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self._prepare_image(observations)  # (B, C*T, H, W)
        B, CT, H, W = x.shape
        # Reshape to (B, C, T, H, W) for Conv3d
        x = x.view(B, self.base_channels, self.num_frames, H, W)
        x = self.conv3d(x).squeeze(2)          # (B, 32, H', W')
        x = self.cnn2d(x)
        return self.linear(x)
