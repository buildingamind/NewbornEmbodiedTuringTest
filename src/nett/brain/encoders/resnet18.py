"""
This module defines the Resnet18CNN class, which is a custom feature extractor
based on the ResNet-18 architecture. It is used for encoding observations in
the Newborn Embodied Turing Test project.

The Resnet18CNN class inherits from the BaseFeaturesExtractor class provided
by the stable_baselines3 library. It takes an observation space and the desired
number of features as input and extracts features using the ResNet-18 model.

The ResNet-18 architecture consists of several residual blocks, each containing
two convolutional layers and a skip connection. The final features are obtained
by applying a linear layer to the output of the last residual block.

The ResBlock class and the ResNet_18 class are helper classes used by the
Resnet18CNN class to define the residual blocks and the overall ResNet-18
architecture, respectively.

Example usage:

    observation_space = gymnasium.spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
    features_dim = 256
    encoder = Resnet18CNN(observation_space, features_dim)
    features = encoder(observation)

"""

#!/usr/bin/env python3

# import pdb
import gymnasium

import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging

logger = logging.getLogger(__name__)

class Resnet18CNN(BaseFeaturesExtractor):
    """
    Custom feature extractor based on the ResNet-18 architecture.

    Args:
        observation_space (gymnasium.Space): The observation space of the environment.
        features_dim (int): Number of features to be extracted.
    """

    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 256) -> None:
        super(Resnet18CNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        ## pretrain set false;
        #self.cnn = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        n_input_channels = observation_space.shape[0]
        logger.info("Resnet18CNN Encoder: ")
        self.cnn = ResNet_18(n_input_channels, features_dim)
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass of the feature extractor.

        Args:
            observations (torch.Tensor): The input observations.

        Returns:
            torch.Tensor: The extracted features.
        """
        # Cut off image
        # reshape to from vector to W*H
        # gray to color transform
        # application of ResNet
        # Concat features to the rest of observation vector
        # return
        return self.linear(self.cnn(observations))

## reference - online
class ResBlock(nn.Module):
    """
    Residual block used in the ResNet-18 architecture.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        identity_downsample (nn.Sequential): Downsample layer for the identity.
        stride (int): Stride of the convolutional layers.
    """

    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1) -> None:
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet_18(nn.Module):
    """
    ResNet-18 architecture used in the Resnet18CNN class.

    Args:
        image_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
    """

    def __init__(self, image_channels, num_classes) -> None:
        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride) -> nn.Sequential:
        """
        Helper function to create a residual layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolutional layers.

        Returns:
            nn.Sequential: The residual layer.
        """
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            ResBlock(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            ResBlock(out_channels, out_channels)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Forward pass of the ResNet-18 architecture.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels) -> nn.Sequential:
        """
        Helper function to create an identity downsample layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.

        Returns:
            nn.Sequential: The identity downsample layer.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
