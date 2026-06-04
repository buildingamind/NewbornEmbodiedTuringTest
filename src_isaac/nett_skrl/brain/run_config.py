"""Small run-time helpers for :mod:`nett_skrl.brain.brain`."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from ..runtime.task import TaskConfig
from .trainer import TrainCfg


def policy_device(config: TaskConfig) -> torch.device:
    return torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")


def dry_run_timesteps(brain) -> int:
    return brain.algorithm_cfg.dry_run_timesteps()


def train_timesteps(brain, config: TaskConfig) -> int:
    base = int(config.train_timesteps or brain.train_iterations)
    envs_per_brain = max(1, getattr(brain, "envs_per_brain", 1))
    return max(1, base // envs_per_brain)


def eval_timesteps(brain, config: TaskConfig) -> int:
    total_episodes = brain.test_iterations.get(config.condition, 1)
    # Split test episodes evenly across parallel envs so the eval loop runs
    # for exactly ceil(total/num_envs) × steps_per_episode iterations rather
    # than the full serial budget (which would repeat each task num_envs times).
    num_envs = max(1, int(getattr(config, "num_envs", None) or 1))
    episodes_per_env = math.ceil(total_episodes / num_envs)
    return episodes_per_env * brain.steps_per_episode


def train_cfg_for(brain, config: TaskConfig) -> TrainCfg:
    output_dir = Path(config.path).parent
    timesteps = train_timesteps(brain, config)
    # Offset skrl's internal timestep counter so global_step in W&B accumulates
    # continuously across training chunks created by eval_freq splitting.
    start_step = int(getattr(config, "train_start_step", None) or 0)
    envs_per_brain = max(1, getattr(brain, "envs_per_brain", 1))
    initial_timestep = start_step // envs_per_brain
    return TrainCfg(
        total_timesteps=timesteps,
        initial_timestep=initial_timestep,
        hparams_dir=Path(config.path),
        hparams=train_hparams(brain, config, timesteps),
        output_dir=output_dir,
        condition=config.condition,
        phase=config.current_mode,
        run_name=output_dir.name,
    )


def train_hparams(brain, config: TaskConfig, timesteps: int) -> dict:
    return {
        "algorithm": _name_of(brain.algorithm),
        "encoder": _name_of(brain.encoder),
        "model": brain.model_cfg.__dict__,
        "reward": str(brain.reward_spec),
        "encoder_cfg": brain.encoder_cfg.as_dict(),
        "algorithm_cfg": _jsonable_cfg(brain.algorithm_cfg.as_dict()),
        "reward_cfg": brain.reward_cfg.as_dict(),
        "checkpoint_freq": brain.checkpoint_freq,
        "envs_per_brain": brain.envs_per_brain,
        "total_timesteps": brain.train_iterations,
        "chunk_timesteps": timesteps,
        "global_step": config.train_global_step,
    }


def _name_of(value) -> str:
    return getattr(value, "__name__", str(value))


def _jsonable_cfg(data: dict) -> dict:
    return {k: ("callable" if callable(v) else v) for k, v in data.items()}
