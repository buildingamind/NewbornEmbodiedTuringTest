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
    # ⚠⚠ FLIPPED BACK TO DIAGONAL ("gaussian") 2026-07-31, ON MEASUREMENT.
    # The 2026-07-30 flip to multivariate rested on the construction argument above
    # plus matching the validated recipe -- never on a measured win. Three comparisons
    # have now run, all favouring the DIAGONAL head:
    #   * 2026-07-30 A/B, n=3, actor isolated (both arms [left,right]):
    #     diagonal 3/3 vs multivariate 2/3;
    #   * B5 500 ep (chunked, artifact-affected): diag+[turn,move] 4/4 vs mvg+[L,R] 2/4;
    #   * B5 384 ep, CONTINUOUS training: diag+[turn,move] 4/4 (mean 0.963) vs
    #     mvg+[L,R] 2/4 (mean 0.797).
    # ⚠ NONE of these is significant on its own (p ~ 0.21), and the advantage is
    # BUDGET-DEPENDENT: at 192 episodes the arms are indistinguishable (2/4 vs 2/4,
    # means 0.754 vs 0.757, p=0.76). What differs is RECOVERY -- diagonal's weak seeds
    # climb out between 192 and 384 ep (0.50->0.87, 0.52->0.98) while multivariate's
    # stay stuck (0.50->0.55, 0.52->0.64). So this is "recovers from early collapse",
    # NOT "learns faster".
    # ⚠ Pairs with the wheeled action default, which flipped to [turn, move] in the
    # same change: diagonal x [left,right] is historically POOR, so the two axes
    # INTERACT and were flipped as a package. Only the combined arm is evidenced.
    # Anything trained on a different default is not comparable; see the
    # DEFAULT-FLIP LEDGER in blueprint.md.
    actor_distribution: str | None = "gaussian"


def model_cfg_from(value: dict[str, Any] | None = None) -> ModelCfg:
    data = dict(value or {})
    return ModelCfg(**data)
