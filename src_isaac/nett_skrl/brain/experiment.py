"""skrl experiment configuration for NETT brains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from skrl.agents.torch import ExperimentCfg

from ..recording import install_wandb_init_capture, wandb_run_id

WANDB_CFG_DEFAULTS: dict[str, Any] = {
    "project": "nett-skrl",
    "entity": None,
    "mode": "online",
    "tags": [],
    "notes": None,
    "kwargs": {},
}
WANDB_MODES = {"online", "offline", "disabled"}


def normalize_wandb_cfg(value: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user wandb config over defaults and validate the mode."""
    cfg = {**WANDB_CFG_DEFAULTS, **(value or {})}
    if cfg["mode"] not in WANDB_MODES:
        raise ValueError(
            f"brain.wandb.mode must be one of {sorted(WANDB_MODES)}; got {cfg['mode']!r}"
        )
    cfg["tags"] = list(cfg["tags"] or [])
    cfg["kwargs"] = dict(cfg["kwargs"] or {})
    return cfg


def apply_experiment_cfg(cfg, *, brain, config, brain_id: int) -> None:
    """Configure ``cfg.experiment`` using skrl's official ExperimentCfg."""
    if not hasattr(cfg, "experiment"):
        return

    previous = getattr(cfg, "experiment", None)
    experiment_name = f"brain_{brain_id}"
    output_dir = Path(config.path)
    wandb_enabled = brain.wandb_cfg["mode"] != "disabled"

    kwargs = {
        "directory": str(output_dir / "wandb_runs"),
        "experiment_name": experiment_name,
        "write_interval": _write_interval(previous),
        "checkpoint_interval": int(brain.checkpoint_freq or 0),
        "store_separately": False,
        "wandb": False,
        "wandb_kwargs": {},
    }
    if wandb_enabled:
        install_wandb_init_capture()
        kwargs.update(
            wandb=True,
            wandb_kwargs=_wandb_kwargs(
                brain=brain,
                config=config,
                brain_id=int(brain_id),
                experiment_name=experiment_name,
            ),
        )
    cfg.experiment = ExperimentCfg(**kwargs)


def _write_interval(previous: Any) -> int:
    value = getattr(previous, "write_interval", "auto")
    return 100 if value == "auto" else value


def _wandb_kwargs(*, brain, config, brain_id: int, experiment_name: str) -> dict[str, Any]:
    wandb_cfg = brain.wandb_cfg
    run_name = Path(config.path).parent.name
    condition = config.condition
    phase = config.current_mode

    # Unified mode (NETT_UNIFIED_WANDB=1): all brains in one process share ONE
    # wandb run id + name, so every agent's metrics land in a single run
    # (namespaced per brain in the forwarding hook). Default = per-brain runs.
    import os as _os
    unified = _os.environ.get("NETT_UNIFIED_WANDB") == "1"
    run_id = wandb_run_id(run_name, condition, 0 if unified else brain_id)
    run_display = (f"{run_name}/{condition}" if unified
                   else f"{run_name}/{condition}/{experiment_name}")

    # Grouping: when many independent single-brain runs share NETT_WANDB_GROUP,
    # wandb overlays them as labeled lines (one per run name) on shared metric
    # charts. Falls back to the condition for the default single-run case.
    wandb_group = _os.environ.get("NETT_WANDB_GROUP") or condition

    kwargs = dict(wandb_cfg.get("kwargs") or {})
    kwargs.update(
        {
            "project": wandb_cfg["project"],
            "mode": wandb_cfg["mode"],
            "group": wandb_group,
            "name": run_display,
            "job_type": phase,
            "tags": [run_name, condition, experiment_name, phase, *wandb_cfg["tags"]],
            "dir": str(config.path),
            "reinit": "create_new",
            "id": run_id,
            "resume": "allow",
            "sync_tensorboard": False,
            "config": _wandb_config_payload(
                brain=brain,
                config=config,
                brain_id=brain_id,
                run_name=run_name,
                phase=phase,
            ),
        }
    )
    for optional_key in ("entity", "notes"):
        if wandb_cfg[optional_key] is not None:
            kwargs[optional_key] = wandb_cfg[optional_key]
        else:
            kwargs.pop(optional_key, None)
    return kwargs


def _wandb_config_payload(*, brain, config, brain_id: int, run_name: str, phase: str) -> dict:
    payload = {
        "run": {
            "run_name": run_name,
            "condition": config.condition,
            "phase": phase,
            "brain_id": brain_id,
            "num_brains": getattr(config, "num_brains", None),
            "num_envs": getattr(config, "num_envs", None),
            "seed": getattr(config, "seed", None),
            "train_global_step": getattr(config, "train_global_step", None),
            "eval_step": getattr(config, "eval_step", None),
            "eval_metrics_only": getattr(config, "eval_metrics_only", None),
            "dry_run": getattr(config, "dry_run", None),
            "device": getattr(config, "device", None),
        },
        "brain": {
            "algorithm": _name_of(brain.algorithm),
            "encoder": _name_of(brain.encoder),
            "reward": str(brain.reward_spec),
            "model": vars(brain.model_cfg),
            "encoder_cfg": brain.encoder_cfg.as_dict(),
            "algorithm_cfg": brain.algorithm_cfg.as_dict(),
            "reward_cfg": brain.reward_cfg.as_dict(),
        },
        "training": {
            "checkpoint_freq": brain.checkpoint_freq,
            "envs_per_brain": getattr(brain, "envs_per_brain", None),
            "train_iterations": getattr(brain, "train_iterations", None),
            "test_iterations": getattr(brain, "test_iterations", None),
            "steps_per_episode": getattr(brain, "steps_per_episode", None),
            "n_tasks": getattr(brain, "n_tasks", None),
        },
    }
    return _jsonable(payload)


def _name_of(value) -> str:
    return getattr(value, "__name__", str(value))


def _jsonable(value):
    if callable(value):
        return "callable"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
