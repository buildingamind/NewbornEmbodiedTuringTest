"""
Frozen SimCLR encoder for stable-baselines3

This module provides a feature extractor based on the SimCLR model. It takes in observations from an environment and extracts features using the SimCLR model.
"""

import torch as th
import gymnasium as gym

from ..utils.rllib_compat import BaseFeaturesExtractor
from .disembodied_models.simclr import SimCLR

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FrozenSimCLR(BaseFeaturesExtractor):
    """

    Frozen SimCLR encoder for stable-baselines3

    Args:
        observation_space (gym.spaces.Box): Observation space
        features_dim (int, optional): Output dimension of features extractor. Defaults to 512.
        checkpoint_path (str, optional): Path to the SimCLR checkpoint. Defaults to "simclr".
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        checkpoint_path: str = "simclr",
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]
        # logger.info("FrozenSimCLR Encoder: ")
        # logger.info(checkpoint_path)
        self.model = SimCLR.load_from_checkpoint(checkpoint_path)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass in the network

        Args:
            observations (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        return self.model(observations)
