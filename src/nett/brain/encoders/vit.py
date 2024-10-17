
import gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .disembodied_models.vit_contrastive import LitClassifier, VisionTransformer

class ViT(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, 
                 observation_space: gym.spaces.Box,
                 learning_rate: float = 1e-3,
                 features_dim: int = 512,
                 patch_size: int = 8,
                 num_hidden_layers: int = 3,
                 num_attention_heads: int = 3,
                 intermediate_size: int = 3072,
                 hidden_size: int = 768,
                 hidden_dropout_prob=0.0,
                 attention_probs_dropout_prob=0.0,
                ) -> None:
        #TODO: Line up this with num_classes
        super().__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]

        configuration = dict(
            image_size = observation_space.shape[1],
            patch_size = patch_size,
            num_classes = features_dim,
            dim = hidden_size,
            depth = num_hidden_layers,
            heads = num_attention_heads,
            mlp_dim = intermediate_size,
            channels = self.n_input_channels,
            dropout = hidden_dropout_prob,
            emb_dropout = attention_probs_dropout_prob
        )

        backbone = VisionTransformer(configuration)
        
        self.model = LitClassifier(backbone, learning_rate=learning_rate).backbone
        self.model.fc = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.model(observations)