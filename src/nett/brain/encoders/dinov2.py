"""DinoV2 feature extractor

This module provides a feature extractor based on the DINOv2 model. It takes in observations from an environment and extracts features using the DINOv2 model.

Example:
    >>> observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
    >>> features_dim = 384
    >>> extractor = DinoV2(observation_space, features_dim)
    >>> observations = torch.randn(1, 3, 84, 84)
    >>> features = extractor.forward(observations)

Attributes:
    n_input_channels (int): The number of input channels in the observation space.
    transforms (torchvision.transforms.Compose): A series of image transformations applied to the observations.
    model (torch.nn.Module): The DINOv2 model used for feature extraction.

Methods:
    __init__(observation_space, features_dim): Initializes the DinoV2 feature extractor.
    forward(observations): Performs a forward pass of the DinoV2 feature extractor.

"""

import gym
import torch

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DinoV2(BaseFeaturesExtractor):
    """
    DinoV2 is a feature extractor based on the DINOv2 model.

    :param observation_space: The observation space of the environment. 
    :type observation_space: gym.spaces.Box
    :param features_dim: Number of features extracted. This corresponds to the number of units for the last layer. Defaults to 384.
    :type features_dim: int, optional
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384):
        super(DinoV2, self).__init__(observation_space, features_dim)
        """Constructor method
        """
        self.n_input_channels = observation_space.shape[0]
        self.transforms = Compose([Resize(size=256,
                                          interpolation=InterpolationMode.BICUBIC,
                                          max_size=None,
                                          antialias=True),
                                   CenterCrop(size=(224, 224)),
                                   Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                             std=torch.tensor([0.229, 0.224, 0.225]))])
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DinoV2 feature extractor.

        :param observations: (torch.Tensor) The input observations.
        :return: (torch.Tensor) The extracted features.
        """
        return self.model(self.transforms(observations))
