"""Run output recording for :mod:`nett_skrl.brain.trainer`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from ..recording.export import RecordingCfg, export_recordings

if TYPE_CHECKING:
    from .trainer import TrainCfg


class RunRecorder:
    """Writes run outputs around skrl training without owning the train loop."""

    def __init__(self, agents: list, num_envs: int) -> None:
        self.agents = agents
        self.num_envs = int(num_envs)

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
        if record_cfg:
            export_recordings(record_cfg)
        if self._can_sync_outputs(cfg):
            from .wandb_sync import sync_outputs_to_wandb

            sync_outputs_to_wandb(
                self.agents,
                output_dir=cfg.output_dir,
                condition=cfg.condition,
                phase=cfg.phase,
                run_name=cfg.run_name,
            )
        self.finish_wandb_runs()

    def finish_wandb_runs(self) -> None:
        from .experiment import finish_agent_wandb_runs

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

    @staticmethod
    def _can_sync_outputs(cfg: TrainCfg) -> bool:
        return all(
            getattr(cfg, attr) is not None
            for attr in ("output_dir", "condition", "phase", "run_name")
        )
