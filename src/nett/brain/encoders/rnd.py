#!/usr/bin/env python3
"""RND encoder module"""
from typing import Tuple
import torch as th
from torch import nn

class RND(nn.Module):
    """Encoder for encoding observations.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): The dimension of actions.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """

    def __init__(self, obs_shape: Tuple, action_dim: int, latent_dim: int) -> None: # pylint: disable=unused-argument
        """
        Initialize the RND encoder.

        Args:
            obs_shape (Tuple): The data shape of observations.
            action_dim (int): The dimension of actions.
            latent_dim (int): The dimension of encoding vectors.
        """
        super().__init__()

        ## visual
        self.main = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size = 3, stride = 2, padding = 1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten()
        )

        with th.no_grad():
            sample = th.ones(size = tuple(obs_shape))
            n_flatten = self.main(sample.unsqueeze(0)).shape[1]

        self.linear = nn.Linear(n_flatten, latent_dim)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors

        Args:
            obs (th.Tensor): Observations

        Returns:
            Encoding tensors
        """
        return self.linear(self.main(obs))
