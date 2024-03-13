"""DINO (Emerging Properties in Self-Supervised Vision Transformers) model"""
import gym
import torch
import timm

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DinoV1(BaseFeaturesExtractor):
    """
    Initialize DinoV1 feature extractor.

    :param observation_space: The observation space of the environment.
    :type observation_space: gym.spaces.Box
    :param features_dim: Number of features extracted. This corresponds to the number of units for the last layer.
    :type features_dim: int
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384):
        """Constructor method
        """
        super(DinoV1, self).__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]
        self.transforms = Compose([Resize(size=248,
                                          interpolation=InterpolationMode.BICUBIC,
                                          max_size=None,
                                          antialias=True),
                                   CenterCrop(size=(224, 224)),
                                   Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                                             std=torch.tensor([0.2290, 0.2240, 0.2250]))])
        self.model = timm.create_model('vit_small_patch8_224.dino',
                                       in_chans=self.n_input_channels,
                                       num_classes=0,
                                       pretrained=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DinoV1 model.

        :param observations: The input observations.
        :type observations: torch.Tensor
        :return: The extracted features.
        :rtype: torch.Tensor
        """
        return self.model(self.transforms(observations))
