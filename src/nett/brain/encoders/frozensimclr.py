"""
Frozen SimCLR encoder for stable-baselines3

This module provides a feature extractor based on the SimCLR model. It takes in observations from an environment and extracts features using the SimCLR model.
"""
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from nett.brain.encoders.disembodied_models.simclr import SimCLR

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FrozenSimCLR(BaseFeaturesExtractor):
    """

    Frozen SimCLR encoder for stable-baselines3

    Args:
        observation_space (gym.spaces.Box): Observation space
        features_dim (int, optional): Output dimension of features extractor. Defaults to 512.
        checkpoint_path (str, optional): Path to the SimCLR checkpoint. Defaults to "simclr".
    """

    def __init__(self, observation_space: "gym.spaces.Box", features_dim: int = 512, checkpoint_path: str = "simclr") -> None:
        super(FrozenSimCLR, self).__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]
        logger.info("FrozenSimCLR Encoder: ")
        logger.info(checkpoint_path)
        self.model = SimCLR.load_from_checkpoint(checkpoint_path)

    def forward(self, observations: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass in the network
        
        Args:
            observations (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: output tensor
        """
        return self.model(observations)
