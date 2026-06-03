"""Configuration for NETT skrl model heads."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class ModelCfg:
    """Configuration for NETT actor/critic MLP heads."""

    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "elu"
    initial_log_std: float = 0.0
    clip_actions: bool = True
    value_bound: float | None = 10.0
    orthogonal_init: bool = True
    hidden_gain: float = math.sqrt(2.0)
    output_gain: float = 0.01
    shared_encoder: bool = False


def model_cfg_from(value: dict[str, Any] | None = None) -> ModelCfg:
    data = dict(value or {})
    return ModelCfg(**data)
