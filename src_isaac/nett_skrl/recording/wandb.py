"""W&B lifecycle helpers for skrl-created per-brain runs."""

from __future__ import annotations

import csv
import hashlib
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("nett.brain")

_DEFAULT_CHAMBER_HALF_X = 33.15

_wandb_runs_by_id: dict[str, Any] = {}
_wandb_init_patched = False


def wandb_run_id(run_name: str, condition: str, brain_id: int) -> str:
    """Stable run ID shared across all phases (train, test) for this brain.

    Removing phase from the hash means the test subprocess resumes the
    training run rather than creating a separate one, so all phases appear
    in a single W&B run.
    """
    raw = f"{run_name}:{condition}:brain_{brain_id}".encode("utf-8")
    return "nett-" + hashlib.sha256(raw).hexdigest()[:24]


def install_wandb_init_capture() -> None:
    """Patch ``wandb.init`` once so every skrl-created Run remains reachable."""
    global _wandb_init_patched
    if _wandb_init_patched:
        return

    try:
        import wandb
    except ImportError:
        return

    original_init = wandb.init

    def _patched_init(*args, **kwargs):
        run = original_init(*args, **kwargs)
        run_id = kwargs.get("id")
        if run is not None:
            run.define_metric("global_step")
            run.define_metric("*", step_metric="global_step")
            if run_id:
                _wandb_runs_by_id[run_id] = run
        return run

    _patched_init._nett_wraps_wandb_init = True
    wandb.init = _patched_init
    _wandb_init_patched = True


def attach_wandb_init_hook(agent) -> None:
    """Wrap agent.init to forward skrl scalars directly to the W&B run.

    W&B's sync_tensorboard file-watching fails when the event file lives
    outside os.getcwd() (the path-relative-to-cwd check in the SDK rejects
    paths under /tmp or any other absolute directory).  Patching the writer
    after agent.init() creates it gives us a reliable, path-independent path:
    each add_scalar call writes the TensorBoard event AND calls run.log()
    directly, so scalars always appear in W&B regardless of where the output
    directory is.
    """
    if getattr(agent, "_nett_wandb_init_hook_attached", False):
        return

    original_init = agent.init

    def _init_with_writer_hook(*args, **kwargs):
        result = original_init(*args, **kwargs)
        run_id = None
        try:
            run_id = agent.cfg.experiment.wandb_kwargs.get("id")
        except AttributeError:
            pass
        run = _wandb_runs_by_id.get(run_id) if run_id else None
        agent._nett_wandb_run = run
        _attach_writer_forwarding_hook(agent, run)
        return result

    agent.init = _init_with_writer_hook
    agent._nett_wandb_init_hook_attached = True


def init_agents_for_eval(agents: list) -> None:
    for agent in agents:
        if hasattr(agent, "init"):
            agent.init()


def finish_agent_wandb_runs(agents) -> None:
    """Finish every captured per-agent wandb Run."""
    for agent in agents:
        run = _wandb_run_for_agent(agent)
        if run is None:
            continue
        try:
            run.finish()
        except Exception:
            logger.debug("wandb run.finish failed", exc_info=True)
        agent._nett_wandb_run = None


def _wandb_run_for_agent(agent):
    run = getattr(agent, "_nett_wandb_run", None)
    if run is not None:
        return run

    run_id = None
    try:
        run_id = agent.cfg.experiment.wandb_kwargs.get("id")
    except AttributeError:
        return None
    run = _wandb_runs_by_id.get(run_id) if run_id else None
    if run is not None:
        agent._nett_wandb_run = run
    return run


def log_recording_videos_to_wandb(agents: list, cfg) -> None:
    """Upload exported MP4 recordings to each agent's active W&B run."""
    try:
        import wandb as _wandb
    except ImportError:
        return
    for brain_id, agent in enumerate(agents, start=1):
        run = _wandb_run_for_agent(agent)
        if run is None:
            continue
        for kind, mp4 in _iter_mp4s_for_brain(cfg.root, env_id=brain_id - 1):
            try:
                tag = f"video/{kind}/{mp4.parent.name}/{mp4.stem}"
                run.log({tag: _wandb.Video(str(mp4), fps=int(cfg.fps), format="mp4")}, commit=False)
            except Exception:
                logger.debug("failed to log video to wandb: %s", mp4, exc_info=True)


def log_test_metrics_to_wandb(agents: list, metrics: dict) -> None:
    """Log test-phase mean reward for each brain to its W&B run."""
    for brain_idx, agent in enumerate(agents):
        run = _wandb_run_for_agent(agent)
        if run is None:
            continue
        mean_reward = metrics.get(brain_idx)
        if mean_reward is not None:
            try:
                run.log({"test/mean_reward": float(mean_reward)}, commit=True)
            except Exception:
                logger.debug("failed to log test metrics to wandb", exc_info=True)


def log_eval_bar_to_wandb(agents: list, eval_bar_info: dict) -> None:
    """Log mid-training eval bar-chart preferences to each brain's W&B run.

    Reads the test CSV written by NETTEnv during the eval subprocess, computes
    mean correct-monitor preference per test condition (the same values shown
    in the ``test_viz`` bar chart), and logs them under ``eval/<test_cond>/correct_pct``
    using ``global_step`` as the x-axis so the series aligns with training
    reward curves in W&B.

    ``eval_bar_info`` must contain:
        eval_step   — total env-interaction count at which this eval was triggered
        global_step — eval_step converted to per-env steps (training x-axis units)
        condition   — imprint condition string
        path        — TaskConfig.path (parent directory that contains ``logs/``)
    """
    logs = Path(eval_bar_info["path"]) / "logs"
    eval_step = eval_bar_info["eval_step"]
    global_step = eval_bar_info["global_step"]

    by_cond: dict[str, list[float]] = defaultdict(list)
    # Each mid-training eval writes its CSV with the eval_step in the filename
    # (see environment.py::_configure_artifacts).
    for csv_path in sorted(logs.glob(f"test_*_{eval_step}.csv")):
        for tc, pct in _preferences_from_csv(csv_path).items():
            by_cond[tc].extend(pct)

    if not by_cond:
        logger.debug(
            "log_eval_bar_to_wandb: no test_*_%d.csv found under %s", eval_step, logs
        )
        return

    for agent in agents:
        run = _wandb_run_for_agent(agent)
        if run is None:
            continue
        payload: dict = {f"eval/{tc}/correct_pct": sum(v) / len(v) for tc, v in by_cond.items()}
        payload["global_step"] = global_step
        try:
            run.log(payload, commit=True)
        except Exception:
            logger.debug("log_eval_bar_to_wandb: run.log failed", exc_info=True)


def _preferences_from_csv(csv_path: Path) -> dict[str, list[float]]:
    """Return {test_cond: [correct_pct_per_row]} from a NETTEnv test log CSV."""
    by_cond: dict[str, list[dict]] = defaultdict(list)
    try:
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):
                tc = row.get("test.cond", "")
                if tc:
                    by_cond[tc].append(row)
    except Exception:
        logger.debug("_preferences_from_csv: failed to read %s", csv_path, exc_info=True)
        return {}

    result: dict[str, list[float]] = {}
    for tc, rows in by_cond.items():
        correct = 0
        total = 0
        for row in rows:
            try:
                ax = float(row["agent.x"])
                az = float(row["agent.z"])
                yaw = float(row["agent.angle"])
                correct_monitor = row.get("correct.monitor", "")
            except (KeyError, ValueError):
                continue
            looking = _looking_at_monitor(ax, az, yaw)
            correct += int(looking == correct_monitor)
            total += 1
        if total > 0:
            result[tc] = [correct / total]
    return result


def _looking_at_monitor(ax: float, az: float, yaw_deg: float) -> str:
    """Return 'left' or 'right' for the monitor the agent's gaze vector points toward."""
    half_x = _DEFAULT_CHAMBER_HALF_X
    rad = math.radians(yaw_deg)
    fx, fy = -math.sin(rad), math.cos(rad)
    left_dx, left_dy = -half_x - ax, -az
    right_dx, right_dy = half_x - ax, -az
    left_norm = math.hypot(left_dx, left_dy) or 1.0
    right_norm = math.hypot(right_dx, right_dy) or 1.0
    left_dot = (left_dx * fx + left_dy * fy) / left_norm
    right_dot = (right_dx * fx + right_dy * fy) / right_norm
    return "left" if left_dot > right_dot else "right"


def _iter_mp4s_for_brain(rec_dir, *, env_id: int):
    """Yield (kind, mp4_path) for all exported MP4s belonging to env_id."""
    env_prefix = f"env_{env_id}"
    for kind in ("egocentric", "chamber"):
        kind_dir = Path(rec_dir) / kind
        if not kind_dir.exists():
            continue
        for env_subdir in kind_dir.rglob("*"):
            if env_subdir.is_dir() and env_subdir.name.startswith(env_prefix):
                for mp4 in sorted(env_subdir.glob("*.mp4")):
                    yield kind, mp4


def _attach_writer_forwarding_hook(agent, run) -> None:
    """Wrap agent.write_tracking_data to forward all rollout scalars to W&B.

    skrl calls write_tracking_data() every write_interval steps, which aggregates
    tracking_data and writes to the TensorBoard event file.  We read tracking_data
    BEFORE that call clears it, build the same aggregations skrl would write, and
    log them all in a single run.log() call.  This bypasses the TensorBoard
    file-sync path entirely — no binary event files, no path checks, no flowcontrol
    backup from uploading large files.
    """
    import numpy as np

    if run is None:
        return
    if getattr(agent, "_nett_wandb_forwarding", False):
        return

    original_write_tracking = getattr(agent, "write_tracking_data", None)
    if original_write_tracking is None:
        return

    def _write_tracking_with_wandb(*, timestep: int, timesteps: int) -> None:
        # Snapshot tracking_data before original_write_tracking clears it.
        payload: dict = {}
        for k, v in agent.tracking_data.items():
            if not v:
                continue
            if k.endswith("(min)"):
                payload[k] = float(np.min(v))
            elif k.endswith("(max)"):
                payload[k] = float(np.max(v))
            else:
                payload[k] = float(np.mean(v))
        original_write_tracking(timestep=timestep, timesteps=timesteps)
        if payload:
            # Apply the per-chunk offset so global_step accumulates continuously
            # across eval_freq training chunks.  skrl 2.x always starts its
            # local timestep counter at 0; _nett_timestep_offset corrects this.
            offset = getattr(agent, "_nett_timestep_offset", 0)
            payload["global_step"] = timestep + offset
            try:
                run.log(payload, commit=True)
            except Exception:
                pass

    agent.write_tracking_data = _write_tracking_with_wandb
    agent._nett_wandb_forwarding = True
