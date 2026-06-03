"""``BrainTrainer`` — NETT wrapper around skrl's ``SequentialTrainer``."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from skrl.trainers.torch import SequentialTrainer
from tqdm import tqdm

from ..recording import RecordingCfg, RunRecorder
from .env_wrappers import IntrinsicRewardEnvWrapper

logger = logging.getLogger("nett.trainer")


@dataclass
class TrainCfg:
    """Per-run training parameters.

    Checkpoint cadence is delegated to skrl (driven by
    ``Brain.checkpoint_freq`` → ``cfg.experiment.checkpoint_interval``); see
    ``brain/experiment.py::apply_experiment_cfg``. ``hparams_dir`` controls
    only where the JSON hparams backup file is written; skrl captures hparams
    into wandb independently.

    ``output_dir`` / ``condition`` / ``phase`` / ``run_name`` are retained for
    run-layout compatibility; checkpoints, scalars, and videos are written
    through skrl/TensorBoard paths.
    """

    total_timesteps: int
    hparams_dir: Path | None = None
    hparams: dict | None = None
    output_dir: Path | None = None
    condition: str | None = None
    phase: str | None = None
    run_name: str | None = None


class BrainTrainer:
    """Thin NETT wrapper over skrl ``SequentialTrainer``.

    One skrl agent owns one vectorized env scope. ``SequentialTrainer`` handles
    the act/step/record/update loop through its native ``agents`` + ``scopes``
    support; :class:`RunRecorder` owns the output writing around that loop.
    """

    def __init__(self, env, agents: list, device: str | torch.device = "cuda"):
        if len(agents) < 1:
            raise ValueError("BrainTrainer requires at least one agent.")
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

        ``dry_run`` short-circuits the output-producing branches: skip
        ``hparams.json``, skip ``train_timing.json``, skip final checkpoints
        and recording exports.
        """
        egocentric_rec = _find_egocentric_recorder(self.env)
        if egocentric_rec is not None and record_cfg is not None and record_cfg.egocentric_enabled:
            egocentric_rec.set_recording(True)
        recorder = RunRecorder(self.agents, self.env.num_envs, egocentric_recorder=egocentric_rec)
        recorder.before_train(cfg, dry_run=dry_run)

        train_env = (
            IntrinsicRewardEnvWrapper(self.env, intrinsic_reward_adapters)
            if intrinsic_reward_adapters else self.env
        )
        # One contiguous skrl run — no chunking needed now that NETT no
        # longer interrupts to write its own checkpoints.
        start = time.perf_counter()
        self._run_skrl_train(train_env, cfg.total_timesteps)
        train_elapsed = time.perf_counter() - start
        recorder.after_train(
            cfg,
            elapsed_s=train_elapsed,
            record_cfg=record_cfg,
            dry_run=dry_run,
        )

    def _run_skrl_train(self, env, timesteps: int) -> None:
        import gc
        # Default GC threshold (700, 10, 10) can allow USD/Gf Python wrapper
        # objects with reference cycles to accumulate for hundreds of steps before
        # gen-0 collection runs. Tighten gen-0 to keep per-step object backlog small.
        gc.set_threshold(200, 5, 5)
        trainer = SequentialTrainer(
            env=env,
            agents=self.agents if len(self.agents) > 1 else self.agents[0],
            scopes=list(self.scopes) if len(self.agents) > 1 else None,
            cfg={
                "timesteps": timesteps,
                "headless": True,
            },
        )
        trainer.train()

    def eval(
        self,
        total_timesteps: int,
        *,
        desc: str | None = None,
        show_progress: bool = True,
    ) -> dict[int, float]:
        """Deterministic rollout returning mean reward per brain."""
        if total_timesteps <= 0:
            logger.info("eval skipped: total_timesteps=%d", total_timesteps)
            return {i: 0.0 for i in range(len(self.agents))}
        for agent in self.agents:
            self._set_mode(agent, False)
        totals = torch.zeros(len(self.agents), device=self.device)
        observations, _ = self.env.reset()
        steps = range(total_timesteps)
        if show_progress:
            steps = tqdm(
                steps,
                total=total_timesteps,
                desc=desc or "eval",
                unit="timestep",
                file=sys.stdout,
            )
        with torch.no_grad():
            for t in steps:
                actions = self._collect_actions_for_eval(observations, t, total_timesteps)
                next_observations, rewards, *_ = self.env.step(actions)
                # Env runs on CPU (NETTEnvCfg.sim.device='cpu'); ``totals``
                # is on the policy device. Move rewards
                # to the totals device before accumulating.
                # Metric-only aggregation after the env step: this does not
                # feed back into training transitions or reward shaping.
                reward_rows = rewards.reshape(len(self.agents), self.scopes[0], -1).mean(dim=2)
                totals += reward_rows.to(totals.device, non_blocking=True).mean(dim=1)
                observations = next_observations
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


MultiBrainTrainer = BrainTrainer


def _find_egocentric_recorder(env):
    """Traverse the env wrapper chain to find a ChannelsFirst recorder.

    Uses duck typing (checks for set_recording / drain_completed_episodes) to
    avoid importing from the body package here.
    """
    current = env
    while current is not None:
        if hasattr(current, "set_recording") and hasattr(current, "drain_completed_episodes"):
            return current
        current = getattr(current, "_env", None) or getattr(current, "env", None)
    return None
