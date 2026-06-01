"""Run output recording/finalization for skrl training and rollouts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .export import RecordingCfg, export_recordings
from .tensorboard import log_recording_videos_to_tensorboard, log_wrapper_egocentric_to_tensorboard

if TYPE_CHECKING:
    from nett_skrl.brain.trainer import TrainCfg

logger = logging.getLogger("nett.recorder")


class RunRecorder:
    """Writes run outputs around skrl training without owning the train loop."""

    def __init__(
        self,
        agents: list,
        num_envs: int,
        *,
        egocentric_recorder=None,
    ) -> None:
        self.agents = agents
        self.num_envs = int(num_envs)
        self._egocentric_recorder = egocentric_recorder

    def before_train(self, cfg: TrainCfg, *, dry_run: bool = False) -> None:
        if cfg.hparams_dir and not dry_run:
            self._write_hparams(cfg)

    def after_train(
        self,
        cfg: TrainCfg,
        *,
        elapsed_s: float,
        record_cfg: RecordingCfg | None = None,
        dry_run: bool = False,
    ) -> None:
        if dry_run:
            return
        if cfg.hparams_dir:
            self._write_train_timing(cfg, elapsed_s)
        self.save_final_checkpoints()
        try:
            if record_cfg:
                self._export_and_log_recordings(record_cfg)
        finally:
            self.finish_wandb_runs()

    def after_rollout(self, record_cfg: RecordingCfg | None = None) -> None:
        """Finalize a non-training rollout while skrl/wandb runs are alive."""
        try:
            if record_cfg:
                self._export_and_log_recordings(record_cfg)
        finally:
            self.finish_wandb_runs()

    def finish_wandb_runs(self) -> None:
        from .wandb import finish_agent_wandb_runs

        finish_agent_wandb_runs(self.agents)

    def save_final_checkpoints(self) -> None:
        """Write ``{experiment_dir}/checkpoints/final_agent.pt`` per agent."""
        for agent in self.agents:
            exp_dir = getattr(agent, "experiment_dir", None)
            if not exp_dir:
                continue
            ckpt_dir = Path(exp_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(ckpt_dir / "final_agent.pt"))

    def _export_and_log_recordings(self, record_cfg: RecordingCfg) -> None:
        export_recordings(record_cfg)
        log_recording_videos_to_tensorboard(self.agents, record_cfg)
        if self._egocentric_recorder is not None:
            episodes = self._egocentric_recorder.drain_completed_episodes()
            if episodes:
                log_wrapper_egocentric_to_tensorboard(
                    episodes, self.agents, fps=record_cfg.fps
                )

    def _write_hparams(self, cfg: TrainCfg) -> None:
        out = Path(cfg.hparams_dir) / "logs"
        out.mkdir(parents=True, exist_ok=True)
        payload = dict(cfg.hparams or {})
        payload.setdefault("total_timesteps", cfg.total_timesteps)
        with (out / "hparams.json").open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _write_train_timing(self, cfg: TrainCfg, elapsed_s: float) -> None:
        out = Path(cfg.hparams_dir) / "logs"
        out.mkdir(parents=True, exist_ok=True)
        env_timesteps = int(cfg.total_timesteps)
        train_steps = env_timesteps * max(1, self.num_envs)
        payload = {
            "skrl_train_total_s": elapsed_s,
            "env_timesteps": env_timesteps,
            "train_steps": train_steps,
            "env_steps_per_second": float(env_timesteps) / elapsed_s if elapsed_s > 0 else 0.0,
            "train_steps_per_second": float(train_steps) / elapsed_s if elapsed_s > 0 else 0.0,
        }
        with (out / "train_timing.json").open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
