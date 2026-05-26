"""skrl experiment, checkpoint, and wandb helpers."""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Any

logger = logging.getLogger("nett.brain")

# Registry of every wandb Run created in this process, keyed by the
# ``id`` we pass to ``wandb.init`` (a deterministic hash of
# run_name/condition/brain_id/phase — see ``_wandb_run_id``). skrl's
# multi-brain path calls ``wandb.init(reinit="create_new")`` once per
# brain, and ``wandb.run`` only ever points at the most-recently-created
# Run; post-training file uploads need each brain's own Run object so we
# capture them here via a ``wandb.init`` monkey-patch.
_wandb_runs_by_id: dict[str, Any] = {}
_wandb_init_patched = False

WANDB_CFG_DEFAULTS: dict[str, Any] = {
    "project": "nett-skrl",
    "entity": None,
    "mode": "online",
    "tags": [],
    "notes": None,
}


def normalize_wandb_cfg(value: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user wandb cfg over defaults and validate ``mode``."""
    merged = {**WANDB_CFG_DEFAULTS, **(value or {})}
    valid_modes = {"online", "offline", "disabled"}
    if merged["mode"] not in valid_modes:
        raise ValueError(
            f"brain.wandb.mode must be one of {sorted(valid_modes)}; got {merged['mode']!r}"
        )
    merged["tags"] = list(merged["tags"] or [])
    return merged


def apply_experiment_cfg(
    cfg,
    *,
    wandb_cfg: dict[str, Any],
    checkpoint_freq: int | None,
    condition: str,
    brain_id: int,
    phase: str,
    output_dir: Path,
    run_name: str,
) -> None:
    """Configure skrl's ``cfg.experiment`` for one brain."""
    if not hasattr(cfg, "experiment"):
        return

    exp = cfg.experiment
    exp.directory = str(output_dir / "wandb_runs")
    exp.experiment_name = f"brain_{brain_id}"
    if getattr(exp, "write_interval", "auto") == "auto":
        exp.write_interval = 100
    exp.checkpoint_interval = int(checkpoint_freq) if checkpoint_freq else 0
    exp.store_separately = False

    if wandb_cfg["mode"] == "disabled":
        return

    exp.wandb = True
    _install_wandb_init_capture()
    wandb_kwargs: dict[str, Any] = {
        "project": wandb_cfg["project"],
        "mode": wandb_cfg["mode"],
        "group": condition,
        "name": f"{run_name}/{condition}/brain_{brain_id}",
        "job_type": phase,
        "tags": [run_name, condition, f"brain_{brain_id}", phase] + list(wandb_cfg["tags"]),
        "dir": str(output_dir),
        "reinit": "create_new",
        "id": _wandb_run_id(run_name, condition, brain_id, phase),
        "resume": "allow",
    }
    if wandb_cfg["entity"] is not None:
        wandb_kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg["notes"] is not None:
        wandb_kwargs["notes"] = wandb_cfg["notes"]
    exp.wandb_kwargs = wandb_kwargs


def attach_wandb_scalar_mirror(agent) -> None:
    """Mirror skrl's tracked scalars into wandb on every flush.

    skrl writes scalars (rewards, losses, etc.) to local tensorboard event
    files via its own ``SummaryWriter``. Rather than relying on wandb's
    tensorboard sync (which conflicts with explicit ``step=`` arguments),
    we read the same ``agent.tracking_data`` skrl is about to flush and
    call ``run.log(...)`` ourselves with an explicit timestep.

    Implementation is per-instance method wrapping (NOT class
    mutation): we replace ``agent.init`` and ``agent.write_tracking_data``
    on each agent we build. ``agent.init`` captures the just-created
    ``wandb.run`` as ``agent._nett_wandb_run`` so each brain's scalars
    attribute to that brain's wandb run (in multi-brain runs the global
    ``wandb.run`` only points at the most-recently-init'd one, which
    would mis-attribute).

    Safe when wandb is disabled or not installed: the wrapper falls
    through to ``original_write`` and skips the wandb log call.
    """
    import numpy as np

    if getattr(agent, "_nett_wandb_scalar_mirror_attached", False):
        return

    original_init = agent.init
    original_write = agent.write_tracking_data

    def init_with_wandb_capture(*args, **kwargs):
        result = original_init(*args, **kwargs)
        # NOTE: skrl calls ``wandb.init(reinit="create_new", ...)``. Under
        # that mode the wandb library returns the new Run but does NOT
        # set the global ``wandb.run`` — that field stays at whatever it
        # was (often ``None``). So we cannot use ``wandb.run`` here; we
        # have to look up the run by id. ``_install_wandb_init_capture``
        # already populates ``_wandb_runs_by_id[run_id]`` for us via its
        # own monkey-patch on ``wandb.init``, and the id we passed to
        # ``wandb.init`` lives on ``agent.cfg.experiment.wandb_kwargs``.
        run_id = None
        try:
            run_id = agent.cfg.experiment.wandb_kwargs.get("id")
        except AttributeError:
            pass
        agent._nett_wandb_run = _wandb_runs_by_id.get(run_id) if run_id else None
        return result

    def write_with_wandb_mirror(*, timestep, timesteps):
        # Snapshot tracking_data BEFORE the original call — skrl clears it
        # at the end of its own write.
        run = getattr(agent, "_nett_wandb_run", None)
        if run is None or not agent.tracking_data:
            return original_write(timestep=timestep, timesteps=timesteps)

        snapshot = {k: list(v) for k, v in agent.tracking_data.items() if v}
        result = original_write(timestep=timestep, timesteps=timesteps)
        log_dict: dict[str, float] = {}
        for k, values in snapshot.items():
            if k.endswith("(min)"):
                log_dict[k] = float(np.min(values))
            elif k.endswith("(max)"):
                log_dict[k] = float(np.max(values))
            else:
                log_dict[k] = float(np.mean(values))
        try:
            run.log(log_dict, step=int(timestep))
        except Exception:
            logger.debug("wandb scalar mirror failed", exc_info=True)
        return result

    agent.init = init_with_wandb_capture
    agent.write_tracking_data = write_with_wandb_mirror
    agent._nett_wandb_scalar_mirror_attached = True


def finish_agent_wandb_runs(agents) -> None:
    """Call ``run.finish()`` on every agent's captured wandb Run.

    skrl doesn't tear wandb down on its own, so without this the wandb UI
    leaves runs in the ``crashed`` state (parent process exits before
    ``wandb.finish`` is called). Calling ``.finish()`` on each Run flushes
    pending uploads and marks the run ``finished``. Also clears
    ``agent._nett_wandb_run`` so subsequent calls on the same agent are
    no-ops.
    """
    for agent in agents:
        run = getattr(agent, "_nett_wandb_run", None)
        if run is None:
            continue
        try:
            run.finish()
        except Exception:
            logger.debug("wandb run.finish failed", exc_info=True)
        agent._nett_wandb_run = None


def pick_checkpoint(ckpt_dir: Path) -> Path | None:
    """Return preferred skrl checkpoint in ``ckpt_dir`` for eval, or None."""
    if not ckpt_dir.exists():
        return None
    final = ckpt_dir / "final_agent.pt"
    if final.exists():
        return final
    step_ckpts: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("agent_*.pt"):
        stem = p.stem.removeprefix("agent_")
        if stem.isdigit():
            step_ckpts.append((int(stem), p))
    if step_ckpts:
        return max(step_ckpts, key=lambda t: t[0])[1]
    best = ckpt_dir / "best_agent.pt"
    if best.exists():
        return best
    return None


def _wandb_run_id(run_name: str, condition: str, brain_id: int, phase: str) -> str:
    raw = f"{run_name}:{condition}:brain_{brain_id}:{phase}".encode("utf-8")
    return "nett-" + hashlib.sha256(raw).hexdigest()[:24]


def wandb_run_id(run_name: str, condition: str, brain_id: int, phase: str) -> str:
    """Public alias of the deterministic per-brain wandb run id."""
    return _wandb_run_id(run_name, condition, brain_id, phase)


def get_wandb_run_by_id(run_id: str):
    """Return the captured wandb Run for ``run_id``, or ``None`` if not seen.

    Only populated after ``_install_wandb_init_capture`` has run and the
    matching ``wandb.init`` has fired (which happens inside skrl's
    ``Agent.init``). Callers should tolerate ``None`` (wandb disabled,
    init not yet run, or different run_id).
    """
    return _wandb_runs_by_id.get(run_id)


def _install_wandb_init_capture() -> None:
    """Patch ``wandb.init`` once so we keep a handle on each created Run.

    skrl initializes wandb per agent via ``wandb.init(reinit="create_new")``.
    The global ``wandb.run`` is overwritten by each call, so the Runs for
    earlier brains become unreachable without an explicit capture.
    """
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

    _patched_init._nett_wraps_wandb_init = True  # debugging marker
    wandb.init = _patched_init
    _wandb_init_patched = True


def init_agents_for_eval(agents: list) -> None:
    for agent in agents:
        if hasattr(agent, "init"):
            agent.init()


def load_latest_checkpoints(agents: list, config, *, context: str = "test") -> None:
    """Load each brain's most-recent skrl checkpoint into ``agents[i]``."""
    for i, agent in enumerate(agents):
        ckpt_dir = Path(config.path) / "wandb_runs" / f"brain_{i + 1}" / "checkpoints"
        chosen = pick_checkpoint(ckpt_dir)
        if chosen is None:
            logger.info("%s: no checkpoint found under %s, using fresh weights", context, ckpt_dir)
            continue
        try:
            agent.load(str(chosen))
            logger.info("%s: brain_%d loaded %s", context, i + 1, chosen.name)
        except Exception:
            logger.exception("%s: failed to load %s for brain_%d", context, chosen, i + 1)
