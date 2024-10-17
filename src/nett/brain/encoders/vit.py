
import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .disembodied_models.vit_contrastive import LitClassifier, ViTConfigExtended, Backbone

class ViT(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 features_dim: int = 512) -> None:
        #TODO: Line up this with num_classes
        super().__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]

        #self.model = LitClassifier.load_from_checkpoint(p)
        #self.model.fc = nn.Identity()
        configuration = ViTConfigExtended()
        configuration.image_size = 64 # 64x64 image #change for binocular
        configuration.patch_size = 8 # default to 8, but allow to change
        configuration.num_hidden_layers = 3 # default
        configuration.num_attention_heads = 3 # default
        # print configuration parameters of ViT
        print('image_size - ', configuration.image_size)
        print('patch_size - ', configuration.patch_size)
        print('num_classes - ', configuration.num_classes)
        print('hidden_size - ', configuration.hidden_size)
        print('intermediate_size - ', configuration.intermediate_size)
        print('num_hidden_layers - ', configuration.num_hidden_layers)
        print('num_attention_heads - ', configuration.num_attention_heads)
        backbone = Backbone('vit', configuration)
        
        self.model = LitClassifier(backbone).backbone
        self.model.fc = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.model(observations)