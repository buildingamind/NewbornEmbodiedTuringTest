"""Small run-time helpers for :mod:`nett_skrl.brain.brain`."""

from __future__ import annotations

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
    return brain.test_iterations.get(config.condition, 1) * brain.steps_per_episode


def train_cfg_for(brain, config: TaskConfig) -> TrainCfg:
    output_dir = Path(config.path).parent
    timesteps = train_timesteps(brain, config)
    return TrainCfg(
        total_timesteps=timesteps,
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
