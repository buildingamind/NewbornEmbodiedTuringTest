#!/usr/bin/env python3
import pdb
import gym

import os
import torch as th
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose
from torchvision.transforms import Resize, CenterCrop, Normalize, InterpolationMode
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from nett.brain.encoders.disembodied_models.simclr import SimCLR

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FrozenSimCLR(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box,features_dim: int = 512, checkpoint_path: str = "simclr")->None:
        super(FrozenSimCLR, self).__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]
        logger.info("FrozenSimCLR Encoder: ")
        logger.info(checkpoint_path)
        self.model = SimCLR.load_from_checkpoint(checkpoint_path)
        
        
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.model(observations)
    
    