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
import os
from pathlib import Path
from typing import Optional

from ..runtime.cpu_budget import cell_cpu_threads, kit_thread_args
from ..runtime.texture_defaults import kit_texture_args
from ..runtime.device import torch_device_index, torch_device_str
from ..runtime.task import TaskConfig, recording_phase_map
from .design import get_experiment_design, validate_conditions
from .physx_strategy import (
    probe_free_vram_bytes,
    select_physx_strategy,
    usable_cpu_threads,
)

logger = logging.getLogger("nett.environment")

_RECORDING_TARGETS = ("egocentric", "chamber")

_ENV_CFG_FIELDS = (
    ("scene.num_envs", "num_envs", int),
    ("design_sheet", "design_sheet", str),
    ("media_root", "media_root", str),
    ("episode_steps", "episode_steps", int),
    ("observation.input_resolution", "input_resolution", int),
    ("observation.fov", "camera_fov", float),
    ("reward_types", "reward_types", tuple),
    ("record_mode", "record_mode", None),
    ("motor.enable_neck_flexion", "enable_neck_flexion", bool),
    ("motor.enable_lateral_bending", "enable_lateral_bending", bool),
    ("motor.locomotion", "locomotion", str),
    ("screens.random_first_frame", "random_first_frame", bool),
    ("screens.decision_period", "decision_period", int),
    ("tracemalloc_interval", "tracemalloc_interval", int),
    ("train_step_logging", "train_step_logging", bool),
)


def _applauncher_device(config) -> str:
    """Device string for ``AppLauncher``, preserving the unassigned-device behaviour.

    ⚠ WHEN ``config.device`` IS None THIS DELIBERATELY EMITS THE INVALID STRING
    ``"cuda:None"``, which is what this call site has always produced. Do not "clean it
    up" -- doing so cost a full e2e suite (2026-07-28) and the reason is a latent defect
    elsewhere:

    ``nett.py`` runs ``validate_tasklist`` (line ~359) BEFORE any ``set_device`` (~782/842),
    so ``config.device`` is None during validation. Validation executes in a ProcessPool
    worker created by **fork**, and it calls ``Environment.load`` -> ``AppLauncher``. The
    invalid device string makes AppLauncher fail there, validation is skipped ("skip if the
    env claims it cannot dry-run"), and no Kit ever boots in that forked worker.

    Emit a VALID device instead and Kit really does boot inside the forked worker, where it
    parses the PARENT's argv -- under pytest that is ``-m pytest``:
        [Error] [omni.kit.app.plugin] Ill formed parameter: -m
        Fatal Python error: Segmentation fault
    which breaks the pool (BrokenProcessPool) and fails every task. Measured: 20 passed
    before, 5 passed / 11 failed / 4 errors after, in 36s.

    So an accident is currently load-bearing. THE REAL FIX is for validation not to boot
    Kit in a forked worker at all (fork + CUDA is unsafe regardless of pytest); until that
    is addressed deliberately, this preserves the status quo instead of silently enabling
    a Kit boot nobody asked for. Once a device IS assigned, the index is translated
    properly -- see runtime/device.py.
    """
    configured = getattr(config, "device", None)
    if configured is None:
        return f"cuda:{configured}"
    return torch_device_str(configured)


class Environment:
    """Isaac Lab environment loader.

    Args:
        design_sheet: Path to the CSV design sheet (replaces Unity AssetBundles).
        media_root: Directory containing video/PNG stimulus assets referenced
            by the design sheet.
        conditions: Subset of imprint conditions to run. ``None`` runs all
            conditions found in the sheet (Train-phase rows).
        headless: Run Isaac Sim without a GUI window. Defaults to ``True``.
        input_resolution: Per-eye square resolution.
        episode_steps: Steps per episode (matches Unity ``--episode-steps``).
        reward_types: Tuple of reward names accepted by ``NETTEnv``
            (``"closeness"``, ``"completeness"``, or both).
    """

    def __init__(
        self,
        design_sheet: str | Path | None = None,
        media_root: str | Path | None = None,
        conditions: Optional[list[str]] = None,
        *,
        experiment: str | Path | None = None,
        headless: bool = True,
        input_resolution: int = 64,
        episode_steps: int = 200,
        reward_types: tuple[str, ...] = (),
        record_mode: str = "tSNE",
        recording: dict | None = None,
        random_first_frame: bool = False,
        decision_period: int = 1,
        enable_neck_flexion: bool = False,
        enable_lateral_bending: bool = False,
        # Default WHEELED, matching schema.json's "default": "wheeled" and the
        # validated operating point (blueprint 04 / rewards.py: the ~0.95-rest binding
        # validation is wheeled). jsonschema does NOT apply schema defaults, so this
        # Python default is what actually governs an unset config — it used to be
        # "kinematic", silently running the UNVALIDATED mode and costing a debug cycle.
        locomotion: str = "wheeled",
        render_mode: str = "RealTimeRenderer",
        tracemalloc_interval: int = 0,
        camera_fov: float = 120.0,
        train_phase: str = "train",
        train_step_logging: bool = False,
    ):
        # A self-contained experiment bundle (dir or .zip with a design CSV +
        # videos) supplies both design_sheet and media_root. See
        # nett_isaac.utils.experiment_bundle.
        self.experiment = str(experiment) if experiment else None
        if experiment is not None:
            from nett_isaac.utils.experiment_bundle import resolve_experiment

            design_sheet, media_root = resolve_experiment(experiment)
        if design_sheet is None or media_root is None:
            raise ValueError(
                "Environment needs either `experiment` (a bundle dir/.zip) or "
                "both `design_sheet` and `media_root`."
            )
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
        self.input_resolution = input_resolution
        self.episode_steps = episode_steps
        self.reward_types = reward_types
        self.record_mode = record_mode
        self.recording = recording or {}
        self.random_first_frame = bool(random_first_frame)
        self.decision_period = int(decision_period or 1)
        self.enable_neck_flexion = bool(enable_neck_flexion)
        self.enable_lateral_bending = bool(enable_lateral_bending)
        if locomotion not in ("kinematic", "wheeled"):
            raise ValueError(
                f"locomotion must be 'kinematic' or 'wheeled', got {locomotion!r}"
            )
        self.locomotion = locomotion
        # NOTE: there is no lighting_mode. Repo A ships exactly ONE chamber
        # (assets/chamber/chamber.usdc), statically baked, monitors measured at
        # 250 cd/m^2 (the Acer V193W EJb panels of the original experiment). The former
        # emissive/rectlight/analytic axis is gone: only the baked build is realistic,
        # temporally static and bit-reproducible at once. ⚠ 250, NOT the 300 this
        # comment used to claim -- that target predates the 2026-07-27 recalibration,
        # which found the monitor measurement had been pegged to a mid-grey seed
        # texture rather than a full-white screen. See isaac/CHAMBER_LIGHTING_STATE.md.
        self.render_mode = render_mode
        self.tracemalloc_interval = int(tracemalloc_interval or 0)
        self.camera_fov = float(camera_fov)
        self.train_phase: str = train_phase
        # Per-step train-phase CSV logging. Default OFF: it's a per-step GPU->CPU
        # handover the rest analysis never reads (analyze uses test CSVs +
        # tfevents). Set True to restore the diagnostic train trajectory CSV.
        self.train_step_logging: bool = bool(train_step_logging)
        self.num_brains = 1  # overridden by adjust_to_agent()
        self.num_envs = 1  # total vectorized Isaac env rows

        self._sim_app = None  # populated lazily on first load()

    def adjust_to_agent(self, num_brains: int, num_envs: int | None = None, **kwargs) -> None:
        """Bind agent-side knobs before `load()`.

        Stores num_brains and total num_envs so `load()` can set
        ``cfg.scene.num_envs``. Extra kwargs (``input_resolution``,
        ``reward_types``, ``episode_steps``) override constructor defaults.
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
            # Size Kit's carb.tasking pool to THIS cell's CPU budget; left at
            # Isaac's default every cell would claim 32 threads no matter how
            # many cells share the host. See runtime/cpu_budget.py.
            self._sim_app = AppLauncher(
                headless=self.headless,
                enable_cameras=True,
                device=_applauncher_device(config),
                kit_args=kit_thread_args(
                    cell_cpu_threads(),
                    # Texture-residency defaults (loader threads + on-disk texture
                    # cache): +2.8% throughput vs an unmodified control, flicker-free,
                    # stimulus unchanged. NOT a memory win -- see
                    # runtime/texture_defaults.py for the measurements, the retracted
                    # memory claim, and the NETT_TEXTURE_DEFAULTS=0 escape hatch.
                    #
                    # NETT_EXTRA_KIT_ARGS is appended AFTER them, so it wins on any
                    # flag it repeats. Used to pin experiment-specific values, e.g.
                    # "--/rtx-transient/resourcemanager/maxMipCount=8" (removes the
                    # monitor-texture flicker AND the frame-0 startup blank while
                    # leaving streaming enabled).
                    existing=kit_texture_args(os.environ.get("NETT_EXTRA_KIT_ARGS", "")),
                ),
            ).app

        # DO NOT arm crash_guard here. It is tempting: NETTEnv(cfg) below builds the
        # scene and the tiled-camera canvas, which is exactly where an over-size
        # num_envs runs out of VRAM and then wedges, and arming after load() returns
        # (as task_runner does) installs the hook after that crash. MEASURED 2026-07-15:
        # arming here made a KNOWN-GOOD 16-env probe -- 4.66GB on a 24GB card -- hang
        # for 30min and get reported "too big", because crash_guard's consumer is a
        # synchronous PYTHON callback on EVERY carb message, and the scene build emits
        # them in bulk from many threads: one GIL acquisition each wedges Kit. The
        # scene-build OOM is covered by the probe's absolute timeout instead
        # (TaskConfig.dry_run_timeout).
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
        if config.current_mode == "train":
            cfg.phase = getattr(self, "train_phase", "train")
        else:
            cfg.phase = config.current_mode
        cfg.imprint_condition = config.condition
        # Speedup lever (env-gated, default unchanged): with Fabric-autoparent ON,
        # the eye-camera pose propagates from physics during the sim step, so the
        # extra per-step cam.update(force_recompute=True) is a redundant re-render.
        # NETT_FORCE_RECOMPUTE=0 disables it -> +~4.6% throughput with BIT-IDENTICAL
        # observations (verified: campaign E2). Unset -> the cfg default (True).
        _frc = os.environ.get("NETT_FORCE_RECOMPUTE")
        if _frc is not None:
            cfg.force_camera_recompute = _frc != "0"
        # Tell the env how many REAL test episodes exist for this condition, so a
        # test num_envs that does not divide that total drops the surplus overflow
        # episodes instead of over-sampling the first design rows (see
        # nett_env_cfg.test_total_episodes + _compute_eval_num_envs). Only the test
        # phase has this fixed episode budget; train/record leave it None.
        if config.current_mode == "test":
            num_test_rows = int(self.iterations_per_test_episode.get(config.condition, 0))
            episodes_test = int((config.episodes or {}).get("test", 0))
            total = num_test_rows * episodes_test
            _set_if_present(cfg, "test_total_episodes", total if total > 0 else None)
        # PhysX (sim.device) placement is chosen at init by select_physx_strategy,
        # which encodes the parallelization policy:
        #   * kinematic locomotion MUST stay on CPU PhysX — GPU PhysX
        #     (GpuArticulationView / GpuRigidBodyView) core-dumps with an illegal
        #     memory access on the kinematic-only NETT scene at the first reset
        #     (documented in gpu_tiled_camera.py / nett_env_cfg.py / probe_device.py);
        #   * the *wheeled* agent is a replicated articulation stepped by the
        #     solver — exactly what GPU PhysX is built for — so it defaults to the
        #     render GPU (cuda:N) to keep locomotion on-device and batched across
        #     envs (rendering/cameras already run there);
        #   * BUT when free VRAM cannot hold the GPU-PhysX state for the requested
        #     num_envs, the strategy offloads the wheeled solver to CPU PhysX —
        #     only if the host has enough threads to absorb it — freeing VRAM so
        #     more rendered envs fit (the goal's "offload to CPU iff it raises the
        #     parallelism that fits").
        # NETT_SIM_DEVICE remains an absolute override (GPU-PhysX probe, or forcing
        # the wheeled agent back to CPU for an apples-to-apples compare).
        # TORCH index, not the physical one: this feeds select_physx_strategy (which builds
        # the "cuda:N" string for sim.device) and probe_free_vram_bytes (torch.cuda.
        # mem_get_info) -- both are torch-side, and both see only the pinned card. Passing
        # the physical index here is what made the first pin attempt still fail with
        # "invalid device ordinal" AFTER AppLauncher was already correct. See
        # runtime/device.py for why the two numberings must not be mixed.
        render_device_index = torch_device_index(getattr(config, "device", 0))
        strategy = select_physx_strategy(
            locomotion=getattr(self, "locomotion", "wheeled"),
            render_device_index=render_device_index,
            num_envs=int(getattr(self, "num_envs", 1) or 1),
            free_vram_bytes=(
                probe_free_vram_bytes(render_device_index)
                if getattr(self, "locomotion", "wheeled") == "wheeled"
                else None
            ),
            cpu_threads=usable_cpu_threads(),
            override=os.environ.get("NETT_SIM_DEVICE"),
        )
        cfg.sim.device = strategy.device
        log = logger.warning if strategy.vram_warning else logger.info
        log("PhysX placement: %s (%s)", strategy.device, strategy.rationale)
        self._copy_env_cfg_fields(cfg)
        if self.asset_root is not None:
            _set_if_present(cfg, "asset_root", str(self.asset_root))
        _set_if_present(cfg, "seed", config.seed if seed is None else seed)
        # Global-brain-id offset for this process -> combined with brain_ids (local) for
        # topology-invariant per-episode env seeding (nett_isaac.episode_seed).
        _set_if_present(cfg, "brain_id_offset", int(getattr(config, "brain_id_offset", 0) or 0))
        if config.dry_run:
            _set_if_present(cfg, "validation_mode", True)

    def _copy_env_cfg_fields(self, cfg) -> None:
        for cfg_path, env_attr, transform in _ENV_CFG_FIELDS:
            value = getattr(self, env_attr)
            if transform is not None:
                value = transform(value)
            _set_attr_path(cfg, cfg_path, value)
        # Declare which model owns which env, so every logged row carries its
        # brain_id. Derived from the SAME rule the trainer slices with
        # (brain_scope_sizes), not re-guessed downstream: the analysis used to
        # bucket test rows by env_id and report num_envs as "n_brains", which both
        # mislabels the model count and makes the statistic depend on the
        # test-time env count.
        from ..brain.trainer import brain_id_per_env

        _set_if_present(
            cfg, "brain_ids", tuple(brain_id_per_env(self.num_envs, self.num_brains))
        )

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
        section = recording_phase_map(self.recording, kind)
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
        Path("chick/robot_chick.usdc"),
        Path("chamber/chamber.usdc"),
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
