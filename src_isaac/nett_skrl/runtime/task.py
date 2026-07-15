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
    # Test-phase env ceiling, resolved separately from train's: test replays a
    # fixed schedule and learns nothing, so it can run far wider than training.
    max_test_envs: int | None = None
    train_timesteps: int | None = None
    train_global_step: int | None = None
    train_start_step: int | None = None
    eval_step: int | None = None
    eval_metrics_only: bool = False
    device: int | None = None
    current_mode: str | None = None
    dry_run: bool = False
    # Absolute cap on a DRY-RUN subprocess, in seconds. Only dry runs set it: a
    # probe knows its own budget (boot + one rollout, or a few eval steps), whereas
    # a real run's duration is unbounded by design. Without it an over-size probe
    # can wedge forever -- a 484-env probe hit a Vulkan OOM and then hung, holding
    # 24GB, because crash_guard's bounded exit keys on DEVICE_LOST and an
    # allocation OOM is not one.
    dry_run_timeout: float | None = None
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
        max_test_envs: int | None = None,
    ) -> None:
        self.config = TaskConfig(
            condition, output_dir, modes, episodes, memory,
            num_brains=num_brains, num_envs=num_envs,
            brain_id_offset=brain_id_offset,
            eval_freq=eval_freq,
            max_parallel_envs=max_parallel_envs,
            max_test_envs=max_test_envs,
        )
        self.agent = Agent(brain, body, env)

    def set_device(self, device: int) -> None:
        self.config = replace(self.config, device=device)

    def set_dry_run(self, dry_run: bool = True, timeout: float | None = None) -> None:
        self.config = replace(self.config, dry_run=dry_run, dry_run_timeout=timeout)


def set_seeds(seed: int) -> None:
    """Seed every RNG **and** force deterministic CUDA execution.

    Same call site as legacy (parent ``run_task`` + child ``_run_single_mode``),
    run before any Isaac/CUDA import so the env vars below take effect before the
    CUDA context is created. Seeding alone is not enough for cross-run /
    cross-machine reproducibility: cuDNN's autotuner and nondeterministic CUDA
    kernels must also be pinned, otherwise same-seed runs diverge (especially on
    different GPUs). See workspace/notes/09_reproducibility.md (G1, G3).
    """
    import os

    seed = int(seed)
    # Env vars first — must precede CUDA context creation.
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic cuBLAS GEMMs (see CUDA cuBLAS reproducibility docs).
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Seed skrl's PRNG keys (config.torch.key etc.) from the same seed so the
    # agent + the env-reset seed read in body/skrl_adapter.py are condition-derived
    # and reproducible. Guarded so set_seeds stays usable without skrl installed.
    # NOTE: skrl.set_seed(deterministic=True) calls torch.use_deterministic_
    # algorithms(True) in *strict* mode; we re-assert warn_only below so it does
    # not override our policy.
    try:
        from skrl.utils import set_seed as _skrl_set_seed

        _skrl_set_seed(seed, deterministic=True)
    except Exception:  # pragma: no cover - skrl optional / version drift
        pass

    # Assert our determinism policy LAST so it wins over skrl's strict toggle.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # warn_only: some ops (e.g. certain scatter/pool/index kernels Isaac's
    # renderer/camera path may invoke on CUDA) lack a deterministic
    # implementation. warn_only makes them WARN instead of raising and killing a
    # long training run, while every op that *can* be deterministic still is.
    torch.use_deterministic_algorithms(True, warn_only=True)

    # TF32 tensor-core matmul/conv (NETT_TF32). DEFAULT ON. ~1.44x on any fp32
    # matmul/conv, REPLAY-DETERMINISTIC (measured run-to-run grad diff = 0). Under
    # the bf16 encoder default it is largely redundant (bf16 already runs the
    # encoder in tensor-core precision; TF32 then only covers the tiny fp32 heads +
    # any fp32 matmul in the value/GAE path -- measured +/-0.2% on top of bf16), but
    # it is harmless and covers the fp32 path when NETT_AMP=off. Shifts values
    # ~1e-3 vs strict fp32 (one-time change, replay preserved). Opt OUT with
    # NETT_TF32=0.
    if os.environ.get("NETT_TF32", "1") != "0":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def recording_phase_map(recording, kind: str) -> dict:
    """Per-phase episode mapping for recording camera ``kind``.

    The recording config has shape ``{<camera>: {<phase>: selector}}``; a camera
    is "enabled" purely by having a non-empty per-phase mapping (there is no
    separate boolean flag). Returns ``{}`` when the camera is absent. Shared by
    ``environment`` (phase-membership check) and ``runtime.task_runner``
    (non-empty / enabled check) so both read the same predicate.
    """
    return (recording or {}).get(kind, {}) or {}

