"""skrl algorithm/encoder/reward registry.

The Isaac backend stays self-contained instead of importing the legacy
``nett`` package. Pulling from legacy modules would drag in timm, lightning,
rllte, and Unity-era assumptions that the Isaac Lab/skrl stack does not need.
This registry exposes the functional names that are available natively here,
plus extension hooks for callers with project-specific encoders or intrinsic
rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from skrl.agents.torch.a2c import A2C
from skrl.agents.torch.base import Agent as BaseAgent
from skrl.agents.torch.cem import CEM
from skrl.agents.torch.a2c import A2C_CFG
from skrl.agents.torch.cem import CEM_CFG
from skrl.agents.torch.ddpg import DDPG
from skrl.agents.torch.ddpg import DDPG_CFG
from skrl.agents.torch.ppo import PPO
from skrl.agents.torch.ppo import PPO_CFG
from skrl.agents.torch.rpo import RPO
from skrl.agents.torch.rpo import RPO_CFG
from skrl.agents.torch.sac import SAC
from skrl.agents.torch.sac import SAC_CFG
from skrl.agents.torch.td3 import TD3
from skrl.agents.torch.td3 import TD3_CFG
from skrl.agents.torch.trpo import TRPO
from skrl.agents.torch.trpo import TRPO_CFG
from typing import Callable
from . import encoders
from .encoders import NETTFeatureExtractor
from .rewards import (
    CLTTReward,
    Disagreement,
    E3B,
    Fabric,
    ICM,
    NGU,
    PseudoCounts,
    RE3,
    RIDE,
    RND,
)


# --- Algorithm mapping (skrl) -----------------------------------------------

AlgorithmFamily = Literal["on_policy", "cross_entropy", "continuous_off_policy"]
ActorType = Literal["gaussian", "deterministic"]
CriticType = Literal["value", "q"]


@dataclass(frozen=True)
class AlgorithmSpec:
    cls: type[BaseAgent]
    cfg_cls: type
    family: AlgorithmFamily
    model_keys: tuple[str, ...]
    actor_type: ActorType
    critic_type: CriticType

algorithm_mapping: dict[str, type[BaseAgent]] = {
    "A2C": A2C, "CEM": CEM, "DDPG": DDPG,
    "PPO": PPO, "RPO": RPO, "SAC": SAC, "TD3": TD3,
    "TRPO": TRPO,
}

algorithm_specs: dict[type[BaseAgent], AlgorithmSpec] = {
    A2C: AlgorithmSpec(A2C, A2C_CFG, "on_policy", ("policy", "value"), "gaussian", "value"),
    CEM: AlgorithmSpec(CEM, CEM_CFG, "cross_entropy", ("policy", "value"), "gaussian", "value"),
    DDPG: AlgorithmSpec(
        DDPG,
        DDPG_CFG,
        "continuous_off_policy",
        ("policy", "target_policy", "critic", "target_critic"),
        "deterministic",
        "q",
    ),
    PPO: AlgorithmSpec(PPO, PPO_CFG, "on_policy", ("policy", "value"), "gaussian", "value"),
    RPO: AlgorithmSpec(RPO, RPO_CFG, "on_policy", ("policy", "value"), "gaussian", "value"),
    SAC: AlgorithmSpec(
        SAC,
        SAC_CFG,
        "continuous_off_policy",
        ("policy", "critic_1", "critic_2", "target_critic_1", "target_critic_2"),
        "gaussian",
        "q",
    ),
    TD3: AlgorithmSpec(
        TD3,
        TD3_CFG,
        "continuous_off_policy",
        (
            "policy",
            "target_policy",
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
        ),
        "deterministic",
        "q",
    ),
    TRPO: AlgorithmSpec(TRPO, TRPO_CFG, "on_policy", ("policy", "value"), "gaussian", "value"),
}


def algorithm_spec(algorithm_cls: type[BaseAgent]) -> AlgorithmSpec:
    return algorithm_specs[algorithm_cls]


# --- Encoder mapping --------------------------------------------------------
# Default set ships with the package; callers with the legacy stack
# (timm/lightning/etc.) install can register more via `register_encoder`.
encoder_mapping: dict[str, type[NETTFeatureExtractor]] = {
    "small": encoders.SmallCNN,
    "medium": encoders.Resnet10CNN,
    "large": encoders.Resnet18CNN,
    "nature_cnn": encoders.NatureCNN,
    "Resnet10CNN": encoders.Resnet10CNN,
    "Resnet18CNN": encoders.Resnet18CNN,
    # Compact model suite (< 600 K total parameters with PPO heads)
    "compact_cnn": encoders.CompactCNN,
    "compact_cnn_pretrained": encoders.CompactCNNPretrained,
    "compact_3dcnn_pretrained": encoders.Compact3DCNNPretrained,
    "compact_vit": encoders.CompactViT,
    "compact_3dcnn": encoders.Compact3DCNN,
    "compact_vivit": encoders.CompactViViT,
    "simclr_cltt": encoders.SimCLRCLTT,
    "guess_what_moves": encoders.GuessWhatMoves,
}


def _validate_encoder_cls(cls: type) -> type[NETTFeatureExtractor]:
    if not isinstance(cls, type) or not issubclass(cls, NETTFeatureExtractor):
        raise TypeError(
            "encoder must be a NETTFeatureExtractor subclass; "
            "SB3 BaseFeaturesExtractor classes are not supported in src_isaac."
        )
    return cls


def register_encoder(name: str, cls: type[NETTFeatureExtractor]) -> None:
    """Plug an extra NETT feature extractor class into the registry."""
    cls = _validate_encoder_cls(cls)
    encoder_mapping[name] = cls
    if name not in encoders_list:
        encoders_list.append(name)


# --- Reward mapping ---------------------------------------------------------
# NETT-environment reward names are handled inside `NETTEnv`, so they map to
# None. Intrinsic reward names map to skrl-compatible Python classes.

reward_mapping: dict[str, type | None] = {
    "unsupervised": None,
    "closeness": None,
    "completeness": None,
    "closeness,completeness": None,
    "CLTT": CLTTReward,
    "CLTTReward": CLTTReward,
    "E3B": E3B,
    "ICM": ICM,
    "NGU": NGU,
    "PseudoCounts": PseudoCounts,
    "RIDE": RIDE,
    "Disagreement": Disagreement,
    "Fabric": Fabric,
    "RE3": RE3,
    "RND": RND,
}


def register_reward(name: str, cls: type | None) -> None:
    """Plug an extra intrinsic-reward class into the registry."""
    reward_mapping[name] = cls
    if name not in rewards_list:
        rewards_list.append(name)


# --- String-form lists ------------------------------------------------------

algorithms_list: list[str] = list(algorithm_mapping)
encoders_list: list[str] = list(encoder_mapping)
rewards_list: list[str] = list(reward_mapping)


# --- Validators -------------------------------------------------------------


def _get_validator(
    label: str,
    baseclass: type,
    mapping: dict[str, type],
) -> Callable[[str | type | None], type | None]:
    """Return a validator that resolves string names through ``mapping``."""

    def validate(value):
        if value is None:
            return None
        if isinstance(value, str):
            if value not in mapping:
                raise KeyError(
                    f"{label} should be one of {sorted(mapping)}; got {value!r}."
                )
            return mapping[value]
        if callable(value):
            return value
        raise TypeError(
            f"{label} should be a string or a subclass of {baseclass.__name__}."
        )

    return validate


validate_algorithm = _get_validator("algorithm", BaseAgent, algorithm_mapping)
validate_reward = _get_validator("reward", object, reward_mapping)


def validate_encoder(value: str | type[NETTFeatureExtractor] | None):
    if value is None:
        return None
    if isinstance(value, str):
        if value not in encoder_mapping:
            raise KeyError(
                f"encoder should be one of {sorted(encoder_mapping)}; got {value!r}. "
                "Custom or heavy encoders must be registered with register_encoder(name, cls)."
            )
        return encoder_mapping[value]
    return _validate_encoder_cls(value)
