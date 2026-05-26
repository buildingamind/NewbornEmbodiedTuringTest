"""skrl-compatible image encoders for the Isaac backend."""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from ..observation import image_channels_hw, prepare_image_tensor


class NETTFeatureExtractor(nn.Module):
    """Base feature extractor contract for NETT skrl models.

    Implementations must expose ``features_dim`` and accept Isaac camera
    observations as HWC, CHW, or flattened tensors.
    """

    features_dim: int

    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__()
        self.observation_space = observation_space
        self.features_dim = int(features_dim)


class HWCFeatureExtractor(NETTFeatureExtractor):
    """Base extractor that accepts flat, HWC, or CHW image tensors."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

    def _prepare_image(self, observations: torch.Tensor) -> torch.Tensor:
        return prepare_image_tensor(observations, self.observation_space)


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
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, channels, height, width)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(self._prepare_image(observations)))


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class Resnet10CNN(HWCFeatureExtractor):
    """Efficient ResNet-style feature extractor matching the legacy medium role."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        channels, _, _ = image_channels_hw(observation_space)
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            _ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            _ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(self._prepare_image(observations))


class Resnet18CNN(HWCFeatureExtractor):
    """Deeper ResNet-style feature extractor matching the legacy large role."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        channels, _, _ = image_channels_hw(observation_space)
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            _ResidualBlock(32),
            _ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            _ResidualBlock(64),
            _ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            _ResidualBlock(128),
            _ResidualBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(self._prepare_image(observations))


__all__ = [
    "HWCFeatureExtractor",
    "NETTFeatureExtractor",
    "Resnet10CNN",
    "Resnet18CNN",
    "SmallCNN",
]
