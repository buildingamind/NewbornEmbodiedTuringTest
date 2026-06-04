"""Isaac Lab simulation backend for NETT.

Replaces the Unity ML-Agents `Environment` class. Constructs a `NETTEnvCfg`
from user config, launches Isaac Sim via `AppLauncher`, instantiates the
`NETTEnv` DirectRLEnv subclass (from the `nett_isaac` package shipped in
`isaac_lab/`), and returns it ready for skrl wrapping.

Key differences from the legacy Unity path:
    - No Unity executable; design sheet + media root drive the experiment.
    - No multi-port worker contention; one Isaac Sim process per condition.
    - Vectorization handled by `cfg.scene.num_envs` inside Isaac Lab, not by
      gym `SubprocVecEnv`. `num_envs` may be larger than `num_brains` so each
      brain can own a contiguous parallel env scope.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..runtime.task import TaskConfig
from .design import get_experiment_design, validate_conditions

logger = logging.getLogger("nett.environment")

_RECORDING_TARGETS = ("egocentric", "chamber")

_ENV_CFG_FIELDS = (
    ("scene.num_envs", "num_envs", int),
    ("design_sheet", "design_sheet", str),
    ("media_root", "media_root", str),
    ("episode_steps", "episode_steps", int),
    ("observation.binocular", "binocular_vision", bool),
    ("observation.input_resolution", "input_resolution", int),
    ("observation.fov", "camera_fov", float),
    ("reward_types", "reward_types", tuple),
    ("record_mode", "record_mode", None),
    ("motor.enable_neck_flexion", "enable_neck_flexion", bool),
    ("motor.enable_lateral_bending", "enable_lateral_bending", bool),
    ("screens.random_first_frame", "random_first_frame", bool),
    ("screens.switch_steps", "switch_steps", int),
    ("screens.decision_period", "decision_period", int),
)


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
        enable_neck_flexion: bool = False,
        enable_lateral_bending: bool = False,
        render_mode: str = "RealTimeRenderer",
        camera_fov: float = 120.0,
    ):
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
        self.enable_neck_flexion = bool(enable_neck_flexion)
        self.enable_lateral_bending = bool(enable_lateral_bending)
        self.render_mode = render_mode
        self.camera_fov = float(camera_fov)
        self.num_brains = 1  # overridden by adjust_to_agent()
        self.num_envs = 1  # total vectorized Isaac env rows

        self._sim_app = None  # populated lazily on first load()

    def adjust_to_agent(self, num_brains: int, num_envs: int | None = None, **kwargs) -> None:
        """Bind agent-side knobs before `load()`.

        Stores num_brains and total num_envs so `load()` can set
        ``cfg.scene.num_envs``. Extra kwargs (``input_resolution``,
        ``binocular_vision``, ``reward_types``, ``episode_steps``) override
        constructor defaults.
        """
        self.num_brains = int(num_brains)
        self.num_envs = int(num_envs if num_envs is not None else num_brains)
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                logger.warning("adjust_to_agent: unknown kwarg %r", k)

    def load(self, config: TaskConfig, seed: Optional[int] = None):
        """Launch Isaac Sim and instantiate ``NETTEnv``.

        Returns the raw `NETTEnv`; the Brain layer wraps it with
        ``body.skrl_adapter.IsaacEnvWrapper`` after body-side observation wrappers are
        applied.
        """
        from isaaclab.app import AppLauncher

        if self._sim_app is None:
            if self.render_mode:
                import sys
                flag = f"--/rtx/rendermode={self.render_mode}"
                if flag not in sys.argv:
                    sys.argv.append(flag)
            self._sim_app = AppLauncher(
                headless=self.headless,
                enable_cameras=True,
                device=f"cuda:{getattr(config, 'device', 0)}",
            ).app

        from nett_isaac.nett_env import NETTEnv
        from nett_isaac.nett_env_cfg import NETTEnvCfg

        cfg_kwargs = {}
        if self.asset_root is not None:
            cfg_kwargs["asset_root"] = str(self.asset_root)
        cfg = NETTEnvCfg(**cfg_kwargs)
        self._configure_cfg(cfg, config, seed)
        return NETTEnv(cfg)

    def _configure_cfg(self, cfg, config: TaskConfig, seed: Optional[int] = None) -> None:
        """Apply NETT-skrl settings to a fresh ``NETTEnvCfg`` and refresh derived fields."""
        self._apply_sim_cfg(cfg, config, seed)
        if not config.dry_run:
            self._configure_artifacts(cfg, config, seed)
        cfg.__post_init__()

    def _apply_sim_cfg(self, cfg, config: TaskConfig, seed: Optional[int] = None) -> None:
        cfg.phase = config.current_mode
        cfg.imprint_condition = config.condition
        self._copy_env_cfg_fields(cfg)
        if self.asset_root is not None:
            _set_if_present(cfg, "asset_root", str(self.asset_root))
        _set_if_present(cfg, "seed", config.seed if seed is None else seed)
        if config.dry_run:
            _set_if_present(cfg, "validation_mode", True)

    def _copy_env_cfg_fields(self, cfg) -> None:
        for cfg_path, env_attr, transform in _ENV_CFG_FIELDS:
            value = getattr(self, env_attr)
            if transform is not None:
                value = transform(value)
            _set_attr_path(cfg, cfg_path, value)

    def _configure_artifacts(self, cfg, config: TaskConfig, seed: Optional[int] = None) -> None:
        log_path = config.path / "logs"
        log_path.mkdir(exist_ok=True, parents=True)
        # Append eval_step to the filename for mid-training eval subprocesses so
        # each eval's CSV is uniquely named and parseable per-checkpoint by the
        # W&B bar-chart logger.
        eval_step = getattr(config, "eval_step", None)
        eval_suffix = f"_{eval_step}" if eval_step is not None else ""
        suffix = f"{config.current_mode}_{config.condition}_{seed or 0}{eval_suffix}"
        cfg.log_path = str(log_path / f"{suffix}.csv")
        _set_if_present(cfg, "profile_path", str(log_path / f"profile_{suffix}.json"))

        for kind in _RECORDING_TARGETS:
            self._configure_recording_target(cfg, config, kind)

    def _configure_recording_target(self, cfg, config: TaskConfig, kind: str) -> None:
        episodes = self._recording_episodes(kind, config)
        if not episodes:
            return
        recording_path = config.path / "recordings" / kind / config.current_mode
        recording_path.mkdir(exist_ok=True, parents=True)
        setattr(cfg, f"{kind}_record_path", str(recording_path))
        setattr(cfg, f"{kind}_record_episodes", episodes)
        if kind == "egocentric":
            self._set_egocentric_legacy_recording_attrs(cfg, recording_path, episodes)

    def _set_egocentric_legacy_recording_attrs(
        self, cfg, recording_path: Path, episodes: tuple[int, ...]
    ) -> None:
        """Bridge nett_isaac's generic egocentric recording field names."""
        _set_if_present(cfg, "record_path", str(recording_path))
        _set_if_present(cfg, "record_episodes", episodes)

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


def _set_if_present(obj, name: str, value) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _set_attr_path(obj, dotted_name: str, value) -> None:
    parts = dotted_name.split(".")
    target = obj
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def _infer_asset_root_from_paths(*paths: str | Path) -> Path | None:
    """Infer a private Isaac asset root from user-provided asset paths."""
    required = (
        Path("chick/robot_chick2.usd"),
        Path("chamber/chamber2.usd"),
        Path("design_sheets/example_design.csv"),
    )
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        candidates = [p] if p.is_dir() else []
        candidates.extend(p.parents)
        for candidate in candidates:
            if candidate.name != "assets":
                continue
            if candidate.is_dir() and all((candidate / rel).exists() for rel in required):
                return candidate
    return None
