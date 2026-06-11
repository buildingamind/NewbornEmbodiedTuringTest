"""Task plumbing — one task per condition (vs the legacy per-brain-x-condition).

In the Isaac Lab port, N brains share one process (and one vectorized env)
per imprint condition. ``TaskConfig`` no longer carries a single ``brain_id``;
it carries ``num_brains`` and ``num_envs`` so the Brain layer can allocate N
agents over contiguous env scopes.
"""

from __future__ import annotations

import random
import hashlib
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class TaskConfig:
    """Per-condition task config (replaces per-brain config in the legacy package).

    ``current_mode`` is ``None`` on the base config built per-condition and is
    set per-mode via :meth:`for_mode`, which returns a new config produced
    through :func:`dataclasses.replace`.
    """

    condition: str
    output_dir: Path
    modes: list[str]
    episodes: dict[str, int] | None = None
    memory: Optional[float] = None
    num_brains: int = 1
    num_envs: int | None = None
    brain_id_offset: int = 0
    eval_freq: int | None = None
    max_parallel_envs: int | None = None
    train_timesteps: int | None = None
    train_global_step: int | None = None
    train_start_step: int | None = None
    eval_step: int | None = None
    eval_metrics_only: bool = False
    device: int | None = None
    current_mode: str | None = None
    dry_run: bool = False
    seed: int = field(init=False)
    name: str = field(init=False)
    path: Path = field(init=False)
    logger: logging.Logger = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        output_dir = Path(self.output_dir)
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "modes", list(self.modes))
        object.__setattr__(self, "episodes", dict(self.episodes or {}))
        object.__setattr__(self, "num_envs", int(self.num_envs or self.num_brains))
        # Seed is derived from the condition name so a condition is reproducible.
        # brain_id_offset perturbs it so repeat runs of the same condition can
        # sample different seeds (offset=0 preserves the original seed).
        digest = hashlib.sha256(self.condition.encode("utf-8")).digest()
        object.__setattr__(
            self,
            "seed",
            (int.from_bytes(digest[:8], "big") * 7919 + int(self.brain_id_offset) * 2_654_435_761)
            % (2**31 - 1),
        )
        object.__setattr__(self, "name", output_dir.stem)
        object.__setattr__(self, "path", output_dir / self.condition)
        object.__setattr__(self, "logger", logging.getLogger(f"{self.name}-{self.condition}"))

    def for_mode(self, mode: str, **overrides) -> "TaskConfig":
        return replace(self, current_mode=mode, **overrides)


@dataclass(frozen=True)
class Agent:
    """Co-located brain/body/env triplet for run_task."""
    brain: object
    body: object
    env: object


class Task:
    """A single task = one condition × N brains, executed in one Isaac Sim process."""

    def __init__(
        self,
        brain,
        body,
        env,
        condition: str,
        output_dir: Path,
        modes: list[str],
        episodes: dict[str, int] | None = None,
        memory: Optional[float] = None,
        num_brains: int = 1,
        num_envs: int | None = None,
        brain_id_offset: int = 0,
        eval_freq: int | None = None,
        max_parallel_envs: int | None = None,
    ) -> None:
        self.config = TaskConfig(
            condition, output_dir, modes, episodes, memory,
            num_brains=num_brains, num_envs=num_envs,
            brain_id_offset=brain_id_offset,
            eval_freq=eval_freq,
            max_parallel_envs=max_parallel_envs,
        )
        self.agent = Agent(brain, body, env)

    def set_device(self, device: int) -> None:
        self.config = replace(self.config, device=device)

    def set_dry_run(self, dry_run: bool = True) -> None:
        self.config = replace(self.config, dry_run=dry_run)


def set_seeds(seed: int) -> None:
    """Reproducibility — same call site as legacy, minus cv2 (deferred until needed)."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

