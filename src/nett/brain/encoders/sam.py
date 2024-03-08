#!/usr/bin/env python3
"""
This module contains the implementation of the SegmentAnything class, which is a custom feature extractor
for image observations in a gym environment. It uses a pre-trained model from the timm library to extract
features from the input images.
"""
import gym

import torch as th
import timm
from torchvision.transforms import Compose
from torchvision.transforms import Resize, CenterCrop, Normalize, InterpolationMode

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SegmentAnything(BaseFeaturesExtractor):
    """
    Custom feature extractor for image observations in a gym environment.

    Args:
        observation_space (gym.spaces.Box): The observation space of the environment.
        features_dim (int, optional): Number of features extracted. This corresponds to the number of units for the last layer. Defaults to 384.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384):
        super(SegmentAnything, self).__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]
        self.transforms = Compose([Resize(size=256,
                                          interpolation=InterpolationMode.BICUBIC,
                                          max_size=None,
                                          antialias=True),
                                   CenterCrop(size=(224, 224)),
                                   Normalize(mean=th.tensor([0.485, 0.456, 0.406]),
                                             std=th.tensor([0.229, 0.224, 0.225]))])

        n_input_channels = observation_space.shape[0]
        print("N_input_channels", n_input_channels)

        self.model = timm.create_model("samvit_base_patch16.sa1b", pretrained=True,
                                       num_classes=0)  # remove classifier th.nn.Linear)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass of the feature extractor.
        """
        # Cut off image
        # reshape to from vector to W*H
        # gray to color transform
        # application of ResNet
        # Concat features to the rest of observation vector
        # return
        return self.model(self.transforms(observations))
