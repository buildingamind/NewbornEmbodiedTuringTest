from types import ModuleType
from typing import Optional, Callable

import stable_baselines3
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.ppo.ppo import PPO

import sb3_contrib
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO

from rllte.common.prototype import BaseReward
import rllte.xplore.reward as rl_rewards

from .. import encoders

# custom reward override
from .. import rewards


def _getMapping(
    sources: list[ModuleType], override: dict[str, type]
) -> dict[str, type]:
    # creates a dictionary relating input strings to classes
    mapping: dict[str, type] = {}
    for source in sources:
        for key in dir(source):
            if key[0].isupper():
                mapping[key] = getattr(source, key)

    for key, value in override.items():
        mapping[key] = value

    return mapping


def _getValidator(
    label: str, baseclass: type, mapping: dict[str, type]
) -> Callable[[str | type], type]:
    # validate the value of the input to ensure it is a valid option

    def validate(input: str | type):
        if type(input) == str:
            if input in mapping:
                return mapping[input]
            else:
                raise KeyError(
                    f"If string, {label} should be one of: {mapping.keys()}. Provided {label} {input} is not one of them."
                )
        elif isinstance(input, type) and issubclass(input, baseclass):
            return input
        else:
            raise TypeError(
                f"{label} should only be either a string or a subclass of {baseclass.__name__}"
            )

    return validate


# grabs all algorithms from stable_baselines3 and sb3-contrib
algorithm_mapping: dict[str, type[BaseAlgorithm]] = _getMapping(
    [stable_baselines3, sb3_contrib], {}
)  # keys = ['A2C', 'DDPG', 'DQN', 'HER', 'HerReplayBuffer', 'PPO', 'SAC', 'TD3', 'ARS', 'MaskablePPO', 'QRDQN', 'RecurrentPPO', 'TQC', 'TRPO']

# grabs all encoders from encoders and adds custom encoders at the end
encoder_mapping: dict[str, type[BaseFeaturesExtractor]] = _getMapping(
    [encoders],
    {
        "small": NatureCNN,
        "medium": encoders.Resnet10CNN,
        "large": encoders.Resnet18CNN,
    },
)  # keys = ['CNNLSTM', 'DinoV1', 'DinoV2', 'FrozenSimCLR', 'Resnet10CNN', 'Resnet18CNN', 'SegmentAnything', 'SimpleViT', 'ViT', 'small', 'medium', 'large']

# grabs all rewards from rllte.xplore.reward and overrrides/adds custom rewards at the end
reward_mapping: dict[str, Optional[type[BaseReward]]] = _getMapping(
    [rl_rewards, rewards],
    {
        "unsupervised": None,
        "closeness": None,
        "completeness": None,
        "closeness,completeness": None,
    },
)  # keys = ['disagreement', 'e3b', 'fabric', 'icm', 'ngu', 'pseudocounts', 're3', 'ride', 'rnd', 'unsupervised', 'closeness', 'completeness', 'closeness,completeness']

# grabs all encoders from ppo and recurrentPPO, which covers nearly all algorithms in SB3 and SB3-contrib
policy_mapping: dict[str, type[BasePolicy]] = (
    RecurrentPPO.policy_aliases | PPO.policy_aliases
)  # keys = ['CnnLstmPolicy', 'CnnPolicy', 'MlpPolicy', 'MlpLstmPolicy', 'MultiInputLstmPolicy', 'MultiInputPolicy']

# list valid options
algorithms_list: list[str] = list(algorithm_mapping.keys())
encoders_list: list[str] = list(encoder_mapping.keys())
policies_list: list[str] = list(policy_mapping.keys())
rewards_list: list[str] = list(reward_mapping.keys())

# validators
validate_algorithm = _getValidator("algorithm", BaseAlgorithm, algorithm_mapping)
validate_encoder = _getValidator("encoder", BaseFeaturesExtractor, encoder_mapping)
validate_policy = _getValidator("policy", BasePolicy, policy_mapping)
validate_reward = _getValidator("reward", BaseReward, reward_mapping)
