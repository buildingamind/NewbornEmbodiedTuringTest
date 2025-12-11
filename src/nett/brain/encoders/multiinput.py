"""
Multi-input encoder for MultiInputPolicy.

This module provides the MultiInputEncoder class, a custom feature extractor for
handling Dict observation spaces in reinforcement learning environments. It processes
different observation types (images and vectors) through specialized submodules and
combines their outputs into a unified feature representation.

The encoder automatically:
- Detects image spaces and applies CNN-based feature extraction
- Flattens vector spaces for direct concatenation
- Balances feature dimensions across different input types
- Validates that the specified feature dimension is sufficient for all inputs

Classes:
    MultiInputEncoder: Combined features extractor for Dict observation spaces.

Example:
    >>> import gymnasium as gym
    >>> from nett.brain.encoders import MultiInputEncoder, Resnet18CNN
    >>> 
    >>> observation_space = gym.spaces.Dict({
    ...     'image': gym.spaces.Box(low=0, high=255, shape=(3, 84, 84)),
    ...     'vector': gym.spaces.Box(low=-1, high=1, shape=(10,))
    ... })
    >>> 
    >>> encoder = MultiInputEncoder(
    ...     observation_space=observation_space,
    ...     extractor_class=Resnet18CNN,
    ...     features_dim=256
    ... )
"""

from typing import Type
import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space

class MultiInputEncoder(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.

    This encoder processes multiple observation types simultaneously by building
    separate feature extractors for each key in the observation space. Image inputs
    are processed through CNN-based extractors (specified by extractor_class), while
    vector inputs are flattened. The output features from all submodules are
    concatenated to form the final feature representation.

    The encoder intelligently allocates feature dimensions:
    - Vector spaces preserve their full dimensionality through flattening
    - Remaining features are evenly distributed among image spaces
    - Feature dimension must be large enough to accommodate all inputs

    Args:
        observation_space: Dict space containing multiple observation types.
            Each value can be either an image space (3D) or vector space (1D/2D).
        extractor_class: CNN feature extractor class to use for image observations.
            Must inherit from BaseFeaturesExtractor (e.g., Resnet18CNN, ViT).
        features_dim: Total number of output features. Must be sufficient to
            accommodate vector spaces and at least 1 feature per image space.
            Defaults to 512.
        **extractor_args: Additional keyword arguments passed to the CNN extractor
            class for image observations.

    Raises:
        ValueError: If features_dim is too small for the vector spaces alone.
        ValueError: If remaining features after vector allocation are insufficient
            for the number of image spaces.

    Attributes:
        extractors: ModuleDict containing feature extractors for each observation key.
            Image spaces map to CNN extractors, vector spaces map to Flatten layers.

    Example:
        >>> import gymnasium as gym
        >>> from nett.brain.encoders import MultiInputEncoder, Resnet18CNN
        >>> 
        >>> obs_space = gym.spaces.Dict({
        ...     'camera': gym.spaces.Box(low=0, high=255, shape=(3, 84, 84)),
        ...     'lidar': gym.spaces.Box(low=0, high=255, shape=(1, 32, 32)),
        ...     'velocity': gym.spaces.Box(low=-1, high=1, shape=(2,))
        ... })
        >>> 
        >>> encoder = MultiInputEncoder(
        ...     observation_space=obs_space,
        ...     extractor_class=Resnet18CNN,
        ...     features_dim=256,
        ...     use_batchnorm=True
        ... )
        >>> # Output: 2 (velocity) + 127*2 (camera + lidar) = 256 features
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        extractor_class: Type[BaseFeaturesExtractor],
        features_dim: int = 512,
        **extractor_args,
    ) -> None:
        """
        Initialize the MultiInputEncoder.

        Args:
            observation_space: Dict space containing multiple observation types.
            extractor_class: CNN feature extractor class for image observations.
            features_dim: Total number of output features. Defaults to 512.
            **extractor_args: Additional arguments for the CNN extractor class.

        Raises:
            ValueError: If features_dim is insufficient for the observation space.
        """
        n_image_spaces = 0
        vector_features = 0

        # Count image spaces and calculate total vector features
        for subspace in observation_space.spaces.values():
            if is_image_space(subspace):
                n_image_spaces += 1
            else:
                vector_features += get_flattened_obs_dim(subspace)

        # Validate that features_dim is sufficient for vector spaces
        if vector_features > features_dim:
            raise ValueError(
                f"Features dim ({features_dim}) is too small for the number of vector spaces. Vector spaces alone output {vector_features} features. Adjust value of embedding dimensions to be at least {vector_features}. Any remaining features will be used for image spaces."
            )

        # Calculate number of features available for image spaces
        image_features = features_dim - vector_features

        # Validate that remaining features are sufficient for image spaces
        if image_features < n_image_spaces:
            raise ValueError(
                f"Features dim ({features_dim}) is too small for the number of image spaces. After vector spaces are accounted for, only {image_features} features remain for the {n_image_spaces} image spaces. Adjust value of embedding dimensions to account for this."
            )

        # Calculate the number of features to actually be used
        used_features = vector_features

        if n_image_spaces > 0:
            # Remove any features that cannot be evenly divided among image spaces
            image_features -= image_features % n_image_spaces
            # Add image features to used features
            used_features += image_features
            # Calculate the number of features per image space
            features_per_image = int(image_features / n_image_spaces)

        # Initialize the base features extractor
        super().__init__(observation_space, features_dim=used_features)

        extractors: dict[str, nn.Module] = {}

        # Build feature extractors for each observation key
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                # Create CNN extractor for image observations
                extractors[key] = extractor_class(
                    observation_space=subspace,
                    features_dim=features_per_image,
                    **extractor_args,
                )
            else:
                # Use flatten layer for vector observations
                extractors[key] = nn.Flatten()

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: TensorDict) -> th.Tensor:
        """
        Extract features from multi-input observations.

        Processes each observation through its corresponding feature extractor
        and concatenates the results into a single feature tensor.

        Args:
            observations: Dictionary mapping observation keys to tensors.
                Each key must match a key in the observation_space Dict.
                Tensors should have shape (batch_size, *obs_shape).

        Returns:
            Concatenated feature tensor of shape (batch_size, features_dim),
            where features_dim is the total number of output features.

        Example:
            >>> observations = {
            ...     'image': torch.randn(32, 3, 84, 84),
            ...     'vector': torch.randn(32, 10)
            ... }
            >>> features = encoder.forward(observations)
            >>> features.shape
            torch.Size([32, 256])
        """
        encoded_tensor_list = []

        # Extract features from each observation
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        
        # Concatenate all features along the feature dimension
        return th.cat(encoded_tensor_list, dim=1)
