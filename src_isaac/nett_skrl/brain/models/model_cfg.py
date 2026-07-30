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
    # Upper bound on the policy log-std (caps exploration std at exp(max_log_std)).
    # None keeps skrl's default of 2.0 (std up to e^2 ~ 7.39, effectively random).
    # Lowering it (e.g. ~0.92 -> std cap 2.5) prevents the entropy bonus from
    # inflating std without bound on hard/long runs (the num_envs=16 side-lock
    # remediation). See workspace notes on entropy/sigma-runaway.
    max_log_std: float | None = None
    clip_actions: bool = True
    value_bound: float | None = 10.0
    orthogonal_init: bool = True
    hidden_gain: float = math.sqrt(2.0)
    output_gain: float = 0.01
    shared_encoder: bool = False
    # Stochastic-actor distribution head. THE NETT DEFAULT IS THE JOINT
    # ``MultivariateGaussianActor``: the wheeled agent's two commands are
    # correlated (they mix into one differential drive), and a diagonal Gaussian
    # cannot represent that correlation. Set ``"gaussian"`` -- or explicit
    # ``None``, kept as the legacy escape hatch -- for the independent
    # per-component actor. Only affects stochastic on-policy actors
    # (``actor_type == "gaussian"``); ignored for deterministic actors.
    # See workspace/notes/10_wheeled_physics.md.
    # ⚠ FLIPPED FROM ``None`` 2026-07-30. blueprint.md's header had claimed
    # multivariate-Gaussian was the out-of-the-box default since well before that,
    # while the code shipped diagonal -- so runs made on defaults did NOT match the
    # documented configuration. Anything trained on the old default is not
    # comparable to anything trained after it; see the DEFAULT-FLIP LEDGER.
    actor_distribution: str | None = "multivariate_gaussian"


def model_cfg_from(value: dict[str, Any] | None = None) -> ModelCfg:
    data = dict(value or {})
    return ModelCfg(**data)
