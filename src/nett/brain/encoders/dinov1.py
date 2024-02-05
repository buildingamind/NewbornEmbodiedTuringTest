import gym
import torch
import timm

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DinoV1(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384):
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
        return self.model(self.transforms(observations))
