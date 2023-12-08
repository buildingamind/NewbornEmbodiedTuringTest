#!/usr/bin/env python3
import pdb
import gym

import os
import torch as th
import torch.nn as nn
import torchvision
import timm
from torchvision.transforms import Compose
from torchvision.transforms import Resize, CenterCrop, Normalize, InterpolationMode
# from networks.disembodied_models.models.simclr import SimCLR

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class FrozenSimCLR(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384):
        super(FrozenSimCLR, self).__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]
        n_input_channels = observation_space.shape[0]
        
        print("N_input_channels", n_input_channels)
        checkpoint_path = os.path.join(os.getcwd(), "data/checkpoints")
        self.model = SimCLR.load_from_checkpoint(os.path.join(checkpoint_path, "sim_clr/epoch=98-step=15542.ckpt"))
    
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.model(observations)