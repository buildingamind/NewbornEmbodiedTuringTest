"""Isaac Lab simulation backend for NETT.

Replaces the Unity ML-Agents `Environment` class. Constructs a `NETTEnvCfg`
from user config, launches Isaac Sim via `AppLauncher`, instantiates the
`NETTEnv` DirectRLEnv subclass (from the `nett_isaac` package shipped in
`isaac_lab/`), and returns it ready for skrl wrapping.

Key differences from the legacy Unity path:
    - No Unity executable; design sheet + media root drive the experiment.
    - No multi-port worker contention; one Isaac Sim process per condition.
    - Vectorization handled by `cfg.scene.num_envs` inside Isaac Lab, not by
      gym `SubprocVecEnv`. `num_envs` is set equal to `num_brains` so the N
      brains share a single vectorized env (each env-slice == one brain).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..runtime.task import TaskConfig

logger = logging.getLogger("nett.environment")


class Environment:
    """Isaac Lab environment loader.

    Args:
        design_sheet: Path to the CSV design sheet (replaces Unity AssetBundles).
        media_root: Directory containing video/PNG stimulus assets referenced
            by the design sheet.
        conditions: Subset of imprint conditions to run. ``None`` runs all
            conditions found in the sheet (Train-phase rows).
        headless: Run Isaac Sim without a GUI window. Defaults to ``True``.
        binocular_vision: Two side-cameras (forwarded into ``NETTEnvCfg``).
        input_resolution: Per-eye square resolution.
        episode_steps: Steps per episode (matches Unity ``--episode-steps``).
        reward_types: Tuple of reward names accepted by ``NETTEnv``
            (``"closeness"``, ``"completeness"``, or both).
    """

    def __init__(
        self,
        design_sheet: str | Path,
        media_root: str | Path,
        conditions: Optional[list[str]] = None,
        *,
        headless: bool = True,
        binocular_vision: bool = True,
        input_resolution: int = 64,
        episode_steps: int = 200,
        reward_types: tuple[str, ...] = (),
        record_mode: str = "tSNE",
        recording: dict | None = None,
        random_first_frame: bool = False,
        switch_steps: int = 0,
        decision_period: int = 1,
    ):
        from .design import get_experiment_design, validate_conditions

        self.design_sheet = Path(design_sheet)
        self.media_root = Path(media_root)
        if not self.media_root.exists():
            raise FileNotFoundError(f"Media root not found: {self.media_root}")
        self.asset_root = _infer_asset_root_from_paths(self.design_sheet, self.media_root)

        # design.get_experiment_design raises if the sheet doesn't exist.
        valid = get_experiment_design(self.design_sheet)
        self.conditions = validate_conditions(valid, conditions)

        # Mirror legacy `iterations_per_test_episode` so the Brain class's
        # eval-iteration math keeps working. Empty for conditions with no
        # test rows.
        self.iterations_per_test_episode: dict[str, int] = {
            c: valid[c] for c in self.conditions
        }

        self.headless = headless
        self.binocular_vision = binocular_vision
        self.input_resolution = input_resolution
        self.episode_steps = episode_steps
        self.reward_types = reward_types
        self.record_mode = record_mode
        self.recording = recording or {}
        self.random_first_frame = bool(random_first_frame)
        self.switch_steps = int(switch_steps or 0)
        self.decision_period = int(decision_period or 1)
        self.num_brains = 1  # overridden by adjust_to_agent()

        self._sim_app = None  # populated lazily on first load()

    def adjust_to_agent(self, num_brains: int, **kwargs) -> None:
        """Bind agent-side knobs before `load()`.

        Stores num_brains so `load()` can set ``cfg.scene.num_envs``. Extra
        kwargs (``input_resolution``, ``binocular_vision``, ``reward_types``,
        ``episode_steps``) override constructor defaults.
        """
        self.num_brains = num_brains
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                logger.warning("adjust_to_agent: unknown kwarg %r", k)

    def load(self, config: TaskConfig, seed: Optional[int] = None):
        """Launch Isaac Sim and instantiate ``NETTEnv``.

        Returns the raw `NETTEnv`; the caller (Brain.train) is responsible for
        wrapping it with ``skrl.envs.wrappers.torch.wrap_env``.
        """
        from isaaclab.app import AppLauncher

        if self._sim_app is None:
            self._sim_app = AppLauncher(
                headless=self.headless,
                enable_cameras=True,
                device=f"cuda:{getattr(config, 'device', 0)}",
            ).app

        from nett_isaac.nett_env import NETTEnv
        from nett_isaac.nett_env_cfg import NETTEnvCfg

        cfg = NETTEnvCfg()
        self._configure_cfg(cfg, config, seed)
        return NETTEnv(cfg)

    def _configure_cfg(self, cfg, config: TaskConfig, seed: Optional[int] = None) -> None:
        """Apply NETT-skrl settings to a fresh ``NETTEnvCfg`` and refresh derived fields."""
        cfg.scene.num_envs = self.num_brains
        cfg.phase = config.current_mode
        cfg.imprint_condition = config.condition
        if hasattr(cfg, "asset_root") and self.asset_root is not None:
            cfg.asset_root = str(self.asset_root)
        cfg.design_sheet = str(self.design_sheet)
        cfg.media_root = str(self.media_root)
        cfg.episode_steps = self.episode_steps
        cfg.observation.binocular = self.binocular_vision
        cfg.observation.input_resolution = self.input_resolution
        cfg.reward_types = tuple(self.reward_types)
        cfg.record_mode = self.record_mode
        cfg.screens.random_first_frame = self.random_first_frame
        cfg.screens.switch_steps = self.switch_steps
        cfg.screens.decision_period = self.decision_period
        if hasattr(cfg, "seed"):
            cfg.seed = config.seed if seed is None else seed
        if hasattr(cfg, "validation_mode") and config.dry_run:
            cfg.validation_mode = True

        # In a dry-run we don't want any on-disk artifacts (CSV log, profile,
        # recordings): the only output that matters is the parent reading the
        # post-train free VRAM via ``mem.txt``.
        if not config.dry_run:
            log_path = config.path / "logs"
            log_path.mkdir(exist_ok=True, parents=True)
            cfg.log_path = str(
                log_path / f"{config.current_mode}_{config.condition}_{seed or 0}.csv"
            )
            if hasattr(cfg, "profile_path"):
                cfg.profile_path = str(
                    log_path / f"profile_{config.current_mode}_{config.condition}_{seed or 0}.json"
                )

            egocentric_episodes = self._recording_episodes("egocentric", config)
            if egocentric_episodes:
                recording_path = config.path / "recordings" / "egocentric" / config.current_mode
                recording_path.mkdir(exist_ok=True, parents=True)
                cfg.record_path = str(recording_path)
                cfg.record_episodes = egocentric_episodes
                cfg.egocentric_record_episodes = egocentric_episodes
            chamber_episodes = self._recording_episodes("chamber", config)
            if chamber_episodes:
                chamber_path = config.path / "recordings" / "chamber" / config.current_mode
                chamber_path.mkdir(exist_ok=True, parents=True)
                setattr(cfg, "chamber_record_path", str(chamber_path))
                setattr(cfg, "chamber_record_episodes", chamber_episodes)

        cfg.__post_init__()

    def _recording_episodes(self, kind: str, config: TaskConfig) -> tuple[int, ...]:
        """Resolve episode indices to record for camera ``kind`` in this phase.

        Reads the per-camera section from ``self.recording`` (shape:
        ``{<camera>: {<phase>: selector}}``). A camera is "enabled" by virtue
        of *having an entry for the current phase* — there's no separate
        boolean. Missing camera, missing phase, or eval-metrics-only mode
        each return an empty tuple.
        """
        if getattr(config, "eval_metrics_only", False):
            return ()
        section = (self.recording or {}).get(kind, {}) or {}
        if config.current_mode not in section:
            return ()
        total_episodes = int(getattr(config, "episodes", {}).get(config.current_mode, 1))
        return parse_episode_selector(section[config.current_mode], total_episodes)

    def close(self) -> None:
        if self._sim_app is not None:
            self._sim_app.close()
            self._sim_app = None


def parse_episode_selector(spec, total_episodes: int) -> tuple[int, ...]:
    """Return concrete zero-based episode indices selected by ``spec``."""
    total_episodes = int(total_episodes)
    if total_episodes < 0:
        raise ValueError(f"total_episodes must be non-negative; got {total_episodes}.")
    universe = range(total_episodes)
    if spec is None:
        return tuple(universe)
    if isinstance(spec, int):
        if spec < 0:
            raise ValueError(f"Invalid recording episodes selector: {spec!r}")
        return (spec,) if spec < total_episodes else ()
    if isinstance(spec, (list, tuple)):
        selected: list[int] = []
        for item in spec:
            if not isinstance(item, int) or item < 0:
                raise ValueError(f"Invalid recording episodes selector: {spec!r}")
            if item < total_episodes and item not in selected:
                selected.append(item)
        return tuple(selected)
    if isinstance(spec, str):
        parts = spec.split(":")
        if len(parts) == 1:
            value = int(parts[0])
            if value < 0:
                raise ValueError(f"Invalid recording episodes selector: {spec!r}")
            return (value,) if value < total_episodes else ()
        if len(parts) in {2, 3}:
            start = int(parts[0] or 0)
            stop = int(parts[1]) if parts[1] else total_episodes
            step = int(parts[2] or 1) if len(parts) == 3 else 1
            if step == 0:
                raise ValueError(f"Invalid recording episodes selector: {spec!r}")
            return tuple(universe[slice(start, stop, step)])
    raise ValueError(f"Invalid recording episodes selector: {spec!r}")


def _infer_asset_root_from_paths(*paths: str | Path) -> Path | None:
    """Infer a private Isaac asset root from user-provided asset paths."""
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        candidates = [p] if p.is_dir() else []
        candidates.extend(p.parents)
        for candidate in candidates:
            if candidate.name != "assets":
                continue
            if (
                (candidate / "chick" / "chick.usd").exists()
                and (candidate / "chamber" / "chamber.usd").exists()
                and (candidate / "design_sheets").is_dir()
            ):
                return candidate
    return None
