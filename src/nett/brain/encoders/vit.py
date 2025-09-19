"""ViT (Vision Transformer) encoder"""

import gymnasium as gym
import torch as th
from torch import nn

from ..utils.rllib_compat import BaseFeaturesExtractor

from .disembodied_models.vit_contrastive import LitClassifier


class ViT(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        learning_rate: float = 3e-4,
        features_dim: int = 512,
        patch_size: int = 4,  # 8,
        depth: int = 3,
        heads: int = 3,
        intermediate_size: int = 128,  # 3072,
        hidden_size: int = 64,  # 768,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    ) -> None:
        # TODO: Line up this with num_classes
        super().__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]

        configuration = dict(
            image_size=observation_space.shape[1],
            patch_size=patch_size,
            num_classes=features_dim,
            dim=hidden_size,
            depth=depth,
            heads=heads,
            mlp_dim=intermediate_size,
            channels=self.n_input_channels,
        )

        if hidden_dropout_prob > 0.0 or attention_probs_dropout_prob > 0.0:
            configuration["dropout"] = hidden_dropout_prob
            configuration["emb_dropout"] = attention_probs_dropout_prob

        self.model = LitClassifier(configuration, learning_rate=learning_rate)
        self.model.fc = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.model(observations)
