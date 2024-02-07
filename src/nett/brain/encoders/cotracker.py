#!/usr/bin/env python3
"""CoTracker feature extractor"""
# import pdb
import gym
import torch as th
from torchvision.transforms import Compose
from torchvision.transforms import Resize, CenterCrop, Normalize, InterpolationMode
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from collections import OrderedDict

class CoTracker(BaseFeaturesExtractor):
    """
    CoTracker is a feature extractor that uses the Co-Tracker model to extract features from observations.

    :param observation_space: (gym.Space) The observation space of the environment.
    :param features_dim: (int) Number of features extracted. This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384):
        super(CoTracker, self).__init__(observation_space, features_dim)
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

        model = th.hub.load("facebookresearch/co-tracker", "cotracker_w8")
        modules = []
        for name, layer in model.named_modules():
            if name == "model.fnet":
                modules.append(layer)

        self.cnn = th.nn.Sequential(*modules)
        self.linear = th.nn.Sequential(th.nn.Linear(128*16*16, 512),
                                            th.nn.ReLU())

        #dummy_x = th.zeros((1, 1, 3, 64, 64))
        #conv_out_size = (self.model(dummy_x))
        #pdb.set_trace()
        ## remove classifier
        #self.model = th.nn.Sequential(OrderedDict([*(list(model.named_children())[:-1])]))



    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass of the CoTracker feature extractor.

        :param observations: (th.Tensor) The input observations.
        :return: (th.Tensor) The extracted features.
        """
        # Cut off image
        # reshape to from vector to W*H
        # gray to color transform
        # application of ResNet
        # Concat features to the rest of observation vector
        # return
        x = self.cnn(observations)
        x = x.view(x.size(0), -1)
        return self.linear(x)
