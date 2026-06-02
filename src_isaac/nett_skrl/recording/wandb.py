"""W&B lifecycle helpers for skrl-created per-brain runs."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger("nett.brain")

_wandb_runs_by_id: dict[str, Any] = {}
_wandb_init_patched = False


def wandb_run_id(run_name: str, condition: str, brain_id: int, phase: str) -> str:
    raw = f"{run_name}:{condition}:brain_{brain_id}:{phase}".encode("utf-8")
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


def _attach_writer_forwarding_hook(agent, run) -> None:
    """Wrap agent.writer.add_scalar to also forward each scalar to the W&B run.

    skrl calls write_tracking_data() at rollout boundaries, which calls
    writer.add_scalar() for each tracked metric.  By forwarding here we bypass
    the TensorBoard file-sync path entirely — no binary event files, no path
    checks, no flowcontrol backup from uploading large files.
    """
    if run is None:
        return
    writer = getattr(agent, "writer", None)
    if writer is None:
        return
    if getattr(writer, "_nett_wandb_forwarding", False):
        return

    original_add_scalar = writer.add_scalar

    def _forwarding_add_scalar(*, tag: str, value: float, timestep: int) -> None:
        original_add_scalar(tag=tag, value=value, timestep=timestep)
        try:
            run.log({tag: value, "global_step": timestep}, commit=False)
        except Exception:
            pass

    writer.add_scalar = _forwarding_add_scalar
    writer._nett_wandb_forwarding = True

    # Commit the buffered scalars after each write_tracking_data flush so they
    # arrive as one batch per rollout rather than trickling in individually.
    original_write_tracking = getattr(agent, "write_tracking_data", None)
    if original_write_tracking is not None:
        def _write_tracking_with_commit(**kwargs):
            original_write_tracking(**kwargs)
            try:
                run.log({}, commit=True)
            except Exception:
                pass
        agent.write_tracking_data = _write_tracking_with_commit
