import gym
import torch
import timm

from torchvision.transforms import Compose
from torchvision.transforms import Resize, CenterCrop, Normalize, InterpolationMode
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ViT(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384):
        """
        Initializes the ViT (Vision Transformer) encoder.

        Args:
            observation_space (gym.spaces.Box): The observation space of the environment.
            features_dim (int, optional): The dimension of the extracted features. Defaults to 384.
        """
        super(ViT, self).__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]
        self.transforms = Compose([Resize(size=248,
                                          interpolation=InterpolationMode.BICUBIC,
                                          max_size=None,
                                          antialias=True),
                                   CenterCrop(size=(224, 224)),
                                   Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]),
                                             std=torch.tensor([0.2290, 0.2240, 0.2250]))])

        self.model = timm.create_model("vit_small_patch8_224.dino",
                                       in_chans=self.n_input_channels,
                                       num_classes=0,
                                       pretrained=False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ViT encoder.

        Args:
            observations (torch.Tensor): The input observations.

        Returns:
            torch.Tensor: The extracted features.
        """
        return self.model(self.transforms(observations))
