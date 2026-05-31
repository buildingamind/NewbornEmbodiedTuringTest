"""skrl experiment, checkpoint, and wandb helpers."""

from __future__ import annotations

import logging
import hashlib
import time
from pathlib import Path
from typing import Any

from skrl.agents.torch import ExperimentCfg

logger = logging.getLogger("nett.brain")

# Registry of every wandb Run created in this process, keyed by the
# ``id`` we pass to ``wandb.init`` (a deterministic hash of
# run_name/condition/brain_id/phase — see ``_wandb_run_id``). skrl's
# multi-brain path calls ``wandb.init(reinit="create_new")`` once per
# brain, and ``wandb.run`` only ever points at the most-recently-created
# Run; capture each handle so we can finish all of them cleanly.
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

    cfg.experiment = make_experiment_cfg(
        previous=getattr(cfg, "experiment", None),
        wandb_cfg=wandb_cfg,
        checkpoint_freq=checkpoint_freq,
        condition=condition,
        brain_id=brain_id,
        phase=phase,
        output_dir=output_dir,
        run_name=run_name,
    )


def make_experiment_cfg(
    *,
    previous: Any = None,
    wandb_cfg: dict[str, Any],
    checkpoint_freq: int | None,
    condition: str,
    brain_id: int,
    phase: str,
    output_dir: Path,
    run_name: str,
) -> ExperimentCfg:
    """Build skrl's official experiment config for one brain."""
    write_interval = getattr(previous, "write_interval", "auto")
    if write_interval == "auto":
        write_interval = 100

    wandb = False
    wandb_kwargs: dict[str, Any] = {}
    if wandb_cfg["mode"] == "disabled":
        return ExperimentCfg(
            directory=str(output_dir / "wandb_runs"),
            experiment_name=f"brain_{brain_id}",
            write_interval=write_interval,
            checkpoint_interval=int(checkpoint_freq) if checkpoint_freq else 0,
            store_separately=False,
            wandb=wandb,
            wandb_kwargs=wandb_kwargs,
        )

    wandb = True
    _install_wandb_init_capture()
    wandb_kwargs = _wandb_kwargs(
        wandb_cfg=wandb_cfg,
        condition=condition,
        brain_id=brain_id,
        phase=phase,
        output_dir=output_dir,
        run_name=run_name,
    )
    return ExperimentCfg(
        directory=str(output_dir / "wandb_runs"),
        experiment_name=f"brain_{brain_id}",
        write_interval=write_interval,
        checkpoint_interval=int(checkpoint_freq) if checkpoint_freq else 0,
        store_separately=False,
        wandb=wandb,
        wandb_kwargs=wandb_kwargs,
    )


def _wandb_kwargs(
    *,
    wandb_cfg: dict[str, Any],
    condition: str,
    brain_id: int,
    phase: str,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
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
        "sync_tensorboard": True,
    }
    if wandb_cfg["entity"] is not None:
        kwargs["entity"] = wandb_cfg["entity"]
    if wandb_cfg["notes"] is not None:
        kwargs["notes"] = wandb_cfg["notes"]
    return kwargs


def attach_tensorboard_tracking(agent) -> None:
    """Add supplemental metrics to skrl's TensorBoard stream.

    skrl writes scalars (rewards, losses, etc.) to local tensorboard event
    files via its own ``SummaryWriter``; ``sync_tensorboard=True`` watches
    those files and uploads them to wandb automatically. This function tracks
    NETT-specific rollout metrics, SB3-compatible PPO ``train/...`` aliases,
    and live SPS (steps per second) through skrl's ``agent.track_data`` so
    TensorBoard and wandb share one scalar source of truth.

    Implementation is per-instance method wrapping (NOT class
    mutation): we replace ``agent.init`` and ``agent.write_tracking_data``
    on each agent we build. ``agent.init`` still captures the just-created
    ``wandb.run`` as ``agent._nett_wandb_run`` so each per-brain run can be
    finished cleanly (in multi-brain runs the global ``wandb.run`` only points
    at the most-recently-init'd one).

    Safe when wandb is disabled or not installed: the wrapper falls
    through to skrl's original methods.
    """
    import numpy as np
    import torch

    if getattr(agent, "_nett_tensorboard_tracking_attached", False):
        return

    original_init = agent.init
    original_write = agent.write_tracking_data
    original_record = getattr(agent, "record_transition", None)
    _n_updates: list[int] = [0]
    _episode_returns: list[torch.Tensor | None] = [None]
    _episode_lengths: list[torch.Tensor | None] = [None]
    _episode_count: list[int] = [0]

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
        run = _wandb_runs_by_id.get(run_id) if run_id else None
        agent._nett_wandb_run = run
        return result

    _last_flush_time: list[float] = [time.perf_counter()]
    _last_flush_step: list[int] = [0]

    def record_with_episode_return_tracking(*args, **kwargs):
        result = original_record(*args, **kwargs)
        try:
            rewards = kwargs.get("rewards")
            if rewards is None:
                return result
            reward_tensor = torch.as_tensor(rewards).detach().reshape(-1).cpu()
            if _episode_returns[0] is None or _episode_returns[0].numel() != reward_tensor.numel():
                _episode_returns[0] = torch.zeros_like(reward_tensor, dtype=torch.float32)
                _episode_lengths[0] = torch.zeros_like(reward_tensor, dtype=torch.float32)
            returns = _episode_returns[0]
            lengths = _episode_lengths[0]
            returns += reward_tensor.to(dtype=returns.dtype)
            lengths += 1.0

            terminated = kwargs.get("terminated", kwargs.get("dones"))
            truncated = kwargs.get("truncated")
            done = _done_tensor(terminated, truncated, reward_tensor.numel(), torch)
            if not bool(done.any()):
                return result

            for env_index in done.nonzero(as_tuple=False).reshape(-1).tolist():
                total_reward = float(returns[env_index].item())
                total_length = float(lengths[env_index].item())
                _episode_count[0] += 1
                _track_many(agent, {
                    "rollout/episode": _episode_count[0],
                    "rollout/env_index": int(env_index),
                    "rollout/ep_len": total_length,
                    "rollout/ep_rew_total": total_reward,
                })
                returns[env_index] = 0.0
                lengths[env_index] = 0.0
        except Exception:
            logger.debug("episode return tracking failed", exc_info=True)
        return result

    def write_with_tensorboard_tracking(*, timestep, timesteps):
        tracking_payload = _tracking_data_payload(getattr(agent, "tracking_data", {}))
        train_payload = _sb3_train_aliases(agent, tracking_payload, torch, np)

        now = time.perf_counter()
        elapsed = now - _last_flush_time[0]
        step_delta = int(timestep) - _last_flush_step[0]
        _last_flush_time[0] = now
        _last_flush_step[0] = int(timestep)

        if train_payload:
            _n_updates[0] += 1
            train_payload["train/n_updates"] = _n_updates[0]

        payload = dict(train_payload)
        payload["Stats/nett_timestep"] = int(timestep)
        if elapsed > 0 and step_delta > 0:
            payload["Stats/nett_sps"] = step_delta / elapsed

        _track_many(agent, payload)
        return original_write(timestep=timestep, timesteps=timesteps)

    agent.init = init_with_wandb_capture
    if original_record is not None:
        agent.record_transition = record_with_episode_return_tracking
    agent.write_tracking_data = write_with_tensorboard_tracking
    agent._nett_tensorboard_tracking_attached = True


def _track_many(agent, payload: dict[str, float]) -> None:
    """Track scalars through skrl so TensorBoard and wandb stay aligned."""
    if not payload:
        return
    track_data = getattr(agent, "track_data", None)
    if callable(track_data):
        for key, value in payload.items():
            track_data(key, float(value))
        return

    tracking_data = getattr(agent, "tracking_data", None)
    if isinstance(tracking_data, dict):
        for key, value in payload.items():
            tracking_data.setdefault(key, []).append(float(value))


def _tracking_data_payload(tracking_data: dict[str, list]) -> dict[str, float]:
    """Return skrl's pending scalar aggregates before its writer clears them."""
    import numpy as np

    payload: dict[str, float] = {}
    for key, values in tracking_data.items():
        if not values:
            continue
        if key.endswith("(min)"):
            payload[key] = float(np.min(values))
        elif key.endswith("(max)"):
            payload[key] = float(np.max(values))
        else:
            payload[key] = float(np.mean(values))
    return payload


def _done_tensor(terminated, truncated, size: int, torch) -> Any:
    """Return a flat bool tensor marking completed env rows."""
    done = torch.zeros(size, dtype=torch.bool)
    if terminated is not None:
        done |= torch.as_tensor(terminated).detach().reshape(-1).cpu().bool()
    if truncated is not None:
        done |= torch.as_tensor(truncated).detach().reshape(-1).cpu().bool()
    return done


def _sb3_train_aliases(agent, tracking_payload: dict[str, float], torch, np) -> dict[str, float]:
    """Map skrl PPO metrics to the SB3-style ``train/...`` wandb keys."""
    payload: dict[str, float] = {}
    key_map = {
        "Loss / Entropy loss": "train/entropy_loss",
        "Loss / Policy loss": "train/policy_gradient_loss",
        "Loss / Value loss": "train/value_loss",
        "Policy / Standard deviation": "train/std",
    }
    for source, target in key_map.items():
        if source in tracking_payload:
            payload[target] = float(tracking_payload[source])

    loss_parts = [
        payload[key]
        for key in ("train/policy_gradient_loss", "train/value_loss", "train/entropy_loss")
        if key in payload
    ]
    if loss_parts:
        payload["train/loss"] = float(sum(loss_parts))

    memory_payload = _ppo_memory_stats(agent, torch, np)
    if not payload and not memory_payload:
        return {}

    cfg = getattr(agent, "cfg", None)
    if cfg is not None:
        if hasattr(cfg, "ratio_clip"):
            payload["train/clip_range"] = float(cfg.ratio_clip)
        value_clip = getattr(cfg, "value_clip", None)
        if value_clip is not None:
            payload["train/clip_range_vf"] = float(value_clip)

    payload.update(memory_payload)
    return payload


def _ppo_memory_stats(agent, torch, np) -> dict[str, float]:
    """Compute SB3-style PPO stats that skrl 2.x does not track directly."""
    memory = getattr(agent, "memory", None)
    policy = getattr(agent, "policy", None)
    cfg = getattr(agent, "cfg", None)
    if memory is None or policy is None or cfg is None:
        return {}

    try:
        observations = memory.get_tensor_by_name("observations")
        states = memory.get_tensor_by_name("states")
        actions = memory.get_tensor_by_name("actions")
        old_log_prob = memory.get_tensor_by_name("log_prob")
        values = memory.get_tensor_by_name("values")
        returns = memory.get_tensor_by_name("returns")
    except Exception:
        return {}

    payload: dict[str, float] = {}
    try:
        with torch.no_grad():
            inputs = {
                "observations": agent._observation_preprocessor(observations),
                "states": agent._state_preprocessor(states),
                "taken_actions": actions,
            }
            _, outputs = policy.act(inputs, role="policy")
            log_ratio = outputs["log_prob"] - old_log_prob
            ratio = torch.exp(log_ratio)
            payload["train/approx_kl"] = float(((ratio - 1) - log_ratio).mean().item())
            ratio_clip = float(getattr(cfg, "ratio_clip", 0.0) or 0.0)
            payload["train/clip_fraction"] = float((torch.abs(ratio - 1) > ratio_clip).float().mean().item())
    except Exception:
        logger.debug("wandb PPO KL/clip stats failed", exc_info=True)

    try:
        y_pred = values.detach().flatten().cpu().numpy()
        y_true = returns.detach().flatten().cpu().numpy()
        variance = np.var(y_true)
        payload["train/explained_variance"] = float(np.nan if variance == 0 else 1 - np.var(y_true - y_pred) / variance)
    except Exception:
        logger.debug("wandb PPO explained variance failed", exc_info=True)

    return payload


def finish_agent_wandb_runs(agents) -> None:
    """Call ``run.finish()`` on every agent's captured wandb Run.

    skrl doesn't tear wandb down on its own, so without this the wandb UI
    leaves runs in the ``crashed`` state (parent process exits before
    ``wandb.finish`` is called). Calling ``.finish()`` on each Run flushes
    pending TensorBoard sync and marks the run ``finished``. Also clears
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
