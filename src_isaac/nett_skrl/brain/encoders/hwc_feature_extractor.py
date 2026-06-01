"""Base image feature extractor for HWC, CHW, and flattened tensors."""

from __future__ import annotations

import gymnasium as gym
import torch

from ...observation import prepare_image_tensor
from .base import NETTFeatureExtractor


class HWCFeatureExtractor(NETTFeatureExtractor):
    """Base extractor that accepts flat, HWC, or CHW image tensors."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

    def _prepare_image(self, observations: torch.Tensor) -> torch.Tensor:
        return prepare_image_tensor(observations, self.observation_space)
