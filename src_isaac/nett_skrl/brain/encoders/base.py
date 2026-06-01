"""Base feature extractor contract for NETT skrl models."""

from __future__ import annotations

import gymnasium as gym
import torch.nn as nn


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
