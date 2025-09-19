from types import ModuleType
from typing import Optional, Callable

from .rllib_compat import BaseAlgorithm, BaseFeaturesExtractor, BasePolicy, NatureCNN
from .rllib_compat import PPO, SAC, DQN

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
        elif issubclass(input, baseclass):
            return input
        else:
            raise TypeError(
                "Reward should only be either a string or a subclass of rllte BaseReward"
            )

    return validate


# grabs all algorithms from rllib_compat module
algorithm_mapping: dict[str, type[BaseAlgorithm]] = {
    'PPO': PPO,
    'SAC': SAC, 
    'DQN': DQN,
}  # RLlib compatible algorithms

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

# grabs all encoders from ppo - RLlib compatible policies
policy_mapping: dict[str, type[BasePolicy]] = {
    'CnnPolicy': BasePolicy,
    'MlpPolicy': BasePolicy,
    'MultiInputPolicy': BasePolicy,
    'CnnLstmPolicy': BasePolicy,
    'MlpLstmPolicy': BasePolicy,
    'MultiInputLstmPolicy': BasePolicy,
}  # RLlib compatible policies

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
