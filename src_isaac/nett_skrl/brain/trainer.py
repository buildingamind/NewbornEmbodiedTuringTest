"""``MultiBrainTrainer`` — NETT wrapper around skrl's ``SequentialTrainer``.

One skrl agent owns one contiguous vectorized env scope. The env wrappers used
during the training loop (intrinsic-reward injection + the small skrl-compat
shim) live in :mod:`nett_skrl.brain.env_wrappers`.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from skrl.trainers.torch import SequentialTrainer

from ..recording.export import RecordingCfg, export_recordings
from .env_wrappers import (
    IntrinsicRewardEnvWrapper,
    SkrlEnvCompatibilityWrapper,
    needs_skrl_compat,
)

logger = logging.getLogger("nett.trainer")


@dataclass
class TrainCfg:
    """Per-run training parameters.

    Checkpoint cadence is delegated to skrl (driven by
    ``Brain.checkpoint_freq`` → ``cfg.experiment.checkpoint_interval``); see
    ``brain/experiment.py::apply_experiment_cfg``. ``hparams_dir`` controls
    only where the JSON hparams backup file is written; skrl captures hparams
    into wandb independently.

    ``output_dir`` / ``condition`` / ``phase`` / ``run_name`` are passed
    through to the post-train wandb upload (``wandb_sync``). Leave them
    ``None`` to skip the upload entirely — the trainer still runs.
    """

    total_timesteps: int
    hparams_dir: Path | None = None
    hparams: dict | None = None
    output_dir: Path | None = None
    condition: str | None = None
    phase: str | None = None
    run_name: str | None = None


class MultiBrainTrainer:
    """Thin NETT wrapper over skrl ``SequentialTrainer``.

    One skrl agent owns one vectorized env scope. ``SequentialTrainer`` handles
    the act/step/record/update loop through its native ``agents`` + ``scopes``
    support; this class preserves NETT output layout and eval/record hooks.
    """

    def __init__(self, env, agents: list, device: str | torch.device = "cuda"):
        if len(agents) < 1:
            raise ValueError("MultiBrainTrainer requires at least one agent.")
        if env.num_envs % len(agents) != 0:
            raise ValueError(
                f"env.num_envs ({env.num_envs}) must be divisible by num_brains "
                f"({len(agents)}); each brain owns one contiguous env scope."
            )
        self.env = env
        self.agents = agents
        self.scopes = [env.num_envs // len(agents)] * len(agents)
        self.device = torch.device(device)

    @staticmethod
    def _set_mode(agent, train: bool) -> None:
        agent.enable_training_mode(train, apply_to_models=True)

    def train(
        self,
        cfg: TrainCfg,
        record_cfg: RecordingCfg | None = None,
        intrinsic_reward_adapters: list | None = None,
        dry_run: bool = False,
    ) -> None:
        """Run training through skrl's native sequential trainer.

        Periodic checkpointing is handled inside skrl via
        ``cfg.experiment.checkpoint_interval``; we only handle the *final*
        snapshot here so the absolute last weights are always recorded
        regardless of whether ``total_timesteps`` lands on a checkpoint
        boundary.

        ``dry_run`` short-circuits the artifact-producing branches: skip
        ``hparams.json``, skip ``train_timing.json``, skip final checkpoints
        and recording exports.
        """
        if cfg.hparams_dir and not dry_run:
            self._write_hparams(cfg)

        train_env = (
            IntrinsicRewardEnvWrapper(self.env, intrinsic_reward_adapters)
            if intrinsic_reward_adapters else self.env
        )
        if needs_skrl_compat(train_env):
            train_env = SkrlEnvCompatibilityWrapper(train_env)
        # One contiguous skrl run — no chunking needed now that NETT no
        # longer interrupts to write its own checkpoints.
        start = time.perf_counter()
        self._run_skrl_train(train_env, cfg.total_timesteps)
        train_elapsed = time.perf_counter() - start
        if dry_run:
            return
        if cfg.hparams_dir:
            self._write_train_timing(cfg, train_elapsed)

        self._save_final_checkpoints()
        if record_cfg:
            export_recordings(record_cfg)
        # Upload checkpoints, recordings (as wandb.Video + raw MP4), per-step
        # CSV logs, profiling JSON, and the run's config.yaml to each brain's
        # wandb Run. ``sync_outputs_to_wandb`` no-ops if wandb is disabled
        # or the artifacts aren't present.
        if all(getattr(cfg, attr) is not None for attr in
               ("output_dir", "condition", "phase", "run_name")):
            from .wandb_sync import sync_outputs_to_wandb

            sync_outputs_to_wandb(
                self.agents,
                output_dir=cfg.output_dir,
                condition=cfg.condition,
                phase=cfg.phase,
                run_name=cfg.run_name,
            )

        # Mark wandb runs as ``finished`` (without this, the run sits in the
        # ``crashed`` state because the parent process exits before wandb's
        # own cleanup hook fires).
        from .experiment import finish_agent_wandb_runs
        finish_agent_wandb_runs(self.agents)

    def _save_final_checkpoints(self) -> None:
        """Write ``{experiment_dir}/checkpoints/final_agent.pt`` per agent.

        Uses skrl's own ``Agent.save`` (whole-modules dict, the same format
        as the periodic ``write_checkpoint`` output) so test-mode loaders can
        round-trip through ``Agent.load`` regardless of which checkpoint
        they pick.
        """
        for agent in self.agents:
            exp_dir = getattr(agent, "experiment_dir", None)
            if not exp_dir:
                continue
            ckpt_dir = Path(exp_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(ckpt_dir / "final_agent.pt"))

    def _run_skrl_train(self, env, timesteps: int) -> None:
        trainer = SequentialTrainer(
            env=env,
            agents=self.agents if len(self.agents) > 1 else self.agents[0],
            scopes=self.scopes if len(self.agents) > 1 else None,
            cfg={
                "timesteps": timesteps,
                "headless": True,
            },
        )
        trainer.train()

    def eval(self, total_timesteps: int) -> dict[int, float]:
        """Deterministic rollout returning mean reward per brain."""
        if total_timesteps <= 0:
            logger.info("eval skipped: total_timesteps=%d", total_timesteps)
            return {i: 0.0 for i in range(len(self.agents))}
        for agent in self.agents:
            self._set_mode(agent, False)
        totals = torch.zeros(len(self.agents), device=self.device)
        observations, _ = self.env.reset()
        with torch.no_grad():
            for t in range(total_timesteps):
                actions = self._collect_actions_for_eval(observations, t, total_timesteps)
                next_observations, rewards, *_ = self.env.step(actions)
                # Env runs on CPU (NETTEnvCfg.sim.device='cpu'); ``totals``
                # is on the policy device. Move rewards
                # to the totals device before accumulating.
                reward_rows = rewards.reshape(self.env.num_envs, -1).mean(dim=1)
                offset = 0
                for i, scope in enumerate(self.scopes):
                    scoped_rewards = reward_rows[offset : offset + scope]
                    totals[i] += scoped_rewards.to(totals.device, non_blocking=True).mean()
                    offset += scope
                observations = next_observations
        # Finish the eval-phase wandb runs so they aren't left ``crashed``.
        # Skrl creates a fresh wandb run for each phase (different ``id`` per
        # train/test), so this is independent of any train-phase finish.
        from .experiment import finish_agent_wandb_runs
        finish_agent_wandb_runs(self.agents)
        return {i: float(totals[i].item() / total_timesteps) for i in range(len(self.agents))}

    def _collect_actions_for_eval(self, observations: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        states = self.env.state() if hasattr(self.env, "state") else None
        actions = []
        offset = 0
        for agent, scope in zip(self.agents, self.scopes):
            obs_i = observations[offset : offset + scope]
            state_i = states[offset : offset + scope] if states is not None else None
            action_i, outputs = agent.act(obs_i, state_i, timestep=timestep, timesteps=timesteps)
            actions.append(outputs.get("mean_actions", action_i))
            offset += scope
        return torch.cat(actions, dim=0)

    @staticmethod
    def _write_hparams(cfg: TrainCfg) -> None:
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
        train_steps = env_timesteps * max(1, self.env.num_envs)
        payload = {
            "skrl_train_total_s": elapsed_s,
            "env_timesteps": env_timesteps,
            "train_steps": train_steps,
            "env_steps_per_second": float(env_timesteps) / elapsed_s if elapsed_s > 0 else 0.0,
            "train_steps_per_second": float(train_steps) / elapsed_s if elapsed_s > 0 else 0.0,
        }
        with (out / "train_timing.json").open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
