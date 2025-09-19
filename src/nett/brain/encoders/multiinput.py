"""Multi-input encoder for MultiInputPolicy"""

from typing import Type
import gymnasium as gym
import torch as th
from torch import nn

from ..utils.rllib_compat import BaseFeaturesExtractor
from typing import Dict as TensorDict
# from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from ..utils.rllib_compat import get_flattened_obs_dim, is_image_space

class MultiInputEncoder(BaseFeaturesExtractor):
    """
    TODO: Update docstring
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        extractor_class: Type[BaseFeaturesExtractor],
        features_dim: int = 512,
        **extractor_args,
    ) -> None:
        n_image_spaces = 0
        vector_features = 0

        for subspace in observation_space.spaces.values():
            if is_image_space(subspace):
                n_image_spaces += 1
            else:
                vector_features += get_flattened_obs_dim(subspace)

        if vector_features > features_dim:
            raise ValueError(
                f"Features dim ({features_dim}) is too small for the number of vector spaces. Vector spaces alone output {vector_features} features. Adjust value of embedding dimensions to be at least {vector_features}. Any remaining features will be used for image spaces."
            )

        # number of features available for image spaces
        image_features = features_dim - vector_features

        if image_features < n_image_spaces:
            raise ValueError(
                f"Features dim ({features_dim}) is too small for the number of image spaces. After vector spaces are accounted for, only {image_features} features remain for the {n_image_spaces} image spaces. Adjust value of embedding dimensions to account for this."
            )

        # calculate the number of features to actually be used
        used_features = vector_features

        if n_image_spaces > 0:
            # remove any features that cannot be evenly divided among image spaces
            image_features -= image_features % n_image_spaces
            # add image features to used features
            used_features += image_features
            # calculate the number of features per image space
            features_per_image = int(image_features / n_image_spaces)

        # initialize the base features extractor
        super().__init__(observation_space, features_dim=used_features)

        extractors: dict[str, nn.Module] = {}

        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = extractor_class(
                    observation_space=subspace,
                    features_dim=features_per_image,
                    **extractor_args,
                )
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
