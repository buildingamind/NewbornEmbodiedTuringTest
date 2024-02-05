import gym
import torch

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DinoV2(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384):
        super(DinoV2, self).__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]
        self.transforms = Compose([Resize(size=256,
                                          interpolation=InterpolationMode.BICUBIC,
                                          max_size=None,
                                          antialias=True),
                                   CenterCrop(size=(224, 224)),
                                   Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),
                                             std=torch.tensor([0.229, 0.224, 0.225]))])
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(self.transforms(observations))