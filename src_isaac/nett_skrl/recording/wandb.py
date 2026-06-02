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
        if run is not None and run_id:
            _wandb_runs_by_id[run_id] = run
        return run

    _patched_init._nett_wraps_wandb_init = True
    wandb.init = _patched_init
    _wandb_init_patched = True


def attach_wandb_init_hook(agent) -> None:
    """Wrap agent.init to register skrl's TensorBoard logdir for wandb sync.

    wandb.init(sync_tensorboard=True) patches EventFileWriter at the module
    level, but skrl imports it with ``from ... import EventFileWriter`` at
    load time — its local reference is the unpatched class, so skrl's writer
    never self-registers and skrl's TensorBoard events never reach W&B.
    Calling _register_skrl_logdir_for_sync right after agent.init() (when
    wandb.init has run and experiment_dir exists) is the only reliable fix.
    """
    if getattr(agent, "_nett_wandb_init_hook_attached", False):
        return

    original_init = agent.init

    def _init_with_logdir_sync(*args, **kwargs):
        result = original_init(*args, **kwargs)
        run_id = None
        try:
            run_id = agent.cfg.experiment.wandb_kwargs.get("id")
        except AttributeError:
            pass
        run = _wandb_runs_by_id.get(run_id) if run_id else None
        agent._nett_wandb_run = run
        _register_skrl_logdir_for_sync(agent, run)
        return result

    agent.init = _init_with_logdir_sync
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
        _register_skrl_logdir_for_sync(agent, run)
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


def _register_skrl_logdir_for_sync(agent, run) -> None:
    """Tell wandb to tail skrl's TensorBoard event directory for this Run."""
    if run is None:
        return
    logdir = getattr(agent, "experiment_dir", None)
    callback = getattr(run, "_tensorboard_callback", None)
    if not logdir or callback is None:
        return
    try:
        callback(str(logdir), save=True)
    except Exception:
        logger.debug("wandb tensorboard sync registration failed", exc_info=True)
