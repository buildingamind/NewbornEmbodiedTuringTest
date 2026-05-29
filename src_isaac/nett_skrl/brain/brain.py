"""skrl-backed Brain class — counterpart of `nett.brain.brain.Brain`.

Public surface matches the legacy class so existing YAML configs Just Work:
``Brain(encoder=, algorithm=, reward=, ...)`` plus ``.train()`` /
``.test()`` / ``.calc_iterations()``.

Internally it builds N skrl agents (one per brain), gives each agent a
contiguous scope of parallel env rows, wires each to its own memory +
optimizer, and delegates training to skrl's ``SequentialTrainer`` through a
small NETT runner wrapper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from ..runtime.task import TaskConfig
from .trainer import MultiBrainTrainer, RecordingCfg, TrainCfg
from .models import ModelCfg, model_cfg_from
from .agent_factory import build_agents as _build_skrl_agents
from .experiment import (
    init_agents_for_eval as _init_agents_for_eval_fn,
    load_latest_checkpoints as _load_latest_checkpoints_fn,
    normalize_wandb_cfg as _normalize_wandb_cfg,
)
from .intrinsic_adapter import IntrinsicRewardAdapter
from .env_adapter import NettIsaacLabWrapper
from .rewards import UnsupportedIntrinsicReward
from .registry import (
    validate_algorithm,
    validate_encoder,
    validate_reward,
)


logger = logging.getLogger("nett.brain")


class Brain:
    """Configures and trains N independent skrl agents.

    Args mirror the legacy Brain where useful — encoder/algorithm/reward are
    looked up against the registries in :mod:`nett_skrl.brain.registry`.
    """

    def __init__(
        self,
        encoder: str | type = "small",
        algorithm: str | type = "PPO",
        reward: str | type | None = "closeness",
        embedding_dim: Optional[int] = None,
        batch_size: int = 512,
        buffer_size: int = 2048,
        learning_rate: float | Callable = 1e-5,
        checkpoint_freq: Optional[int] = None,
        train_encoder: bool = True,
        custom_encoder_args: Optional[dict[str, Any]] = None,
        custom_algorithm_args: Optional[dict[str, Any]] = None,
        model: Optional[dict[str, Any] | ModelCfg] = None,
        reward_args: Optional[dict[str, Any]] = None,
        intrinsic_reward_weight: float = 1.0,
        intrinsic_reward_update: bool = True,
        wandb: Optional[dict[str, Any]] = None,
    ):
        self.reward_spec = reward
        self.encoder = validate_encoder(encoder)
        self.algorithm = validate_algorithm(algorithm)
        self.reward = validate_reward(reward)
        if isinstance(reward, str) and isinstance(self.reward, type) and issubclass(self.reward, UnsupportedIntrinsicReward):
            self.reward()

        self.embedding_dim = int(embedding_dim) if embedding_dim is not None else None
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.learning_rate = learning_rate if callable(learning_rate) else float(learning_rate)
        self.checkpoint_freq = int(checkpoint_freq) if checkpoint_freq is not None else None
        self.train_encoder = bool(train_encoder)

        self.custom_encoder_args = dict(custom_encoder_args or {})
        self.custom_algorithm_args = dict(custom_algorithm_args or {})
        self.model_cfg = model if isinstance(model, ModelCfg) else model_cfg_from(model)
        self.reward_args = dict(reward_args or {"beta": 0.2, "kappa": 0.0, "gamma": 0.99})
        self.intrinsic_reward_weight = float(intrinsic_reward_weight)
        self.intrinsic_reward_update = bool(intrinsic_reward_update)
        self.wandb_cfg = _normalize_wandb_cfg(wandb)

        # Populated by NETT.run setup via calc_iterations(...).
        self.iterations_per_test_episode: dict[str, int] = {}
        self.steps_per_episode: int = 0
        self.envs_per_agent: int = 1
        self.num_brains: int | None = None
        self.train_iterations: int = 0
        self.test_iterations: dict[str, int] = {}
        self.n_tasks: int = 0

    def env_reward_types(self) -> tuple[str, ...]:
        """Return NETTEnv reward names implied by the legacy ``brain.reward`` field."""
        if not isinstance(self.reward_spec, str):
            return ()
        if self.reward_spec in {"unsupervised", ""}:
            return ()
        names = tuple(x.strip() for x in self.reward_spec.split(",") if x.strip())
        if set(names).issubset({"closeness", "completeness"}):
            return names
        return ()

    def uses_intrinsic_reward(self) -> bool:
        return self.reward is not None and not self.env_reward_types()

    # --- Shared with legacy Brain (verbatim semantics) ---------------------
    def calc_iterations(
        self,
        num_brains: int,
        iterations_per_episode: dict[str, int],
        episodes: dict[str, int],
        steps_per_episode: int,
    ) -> None:
        """Compute step budgets for train + test modes."""
        self.num_brains = int(num_brains)
        self.steps_per_episode = steps_per_episode
        self.envs_per_agent = max(1, self.batch_size // max(1, steps_per_episode))
        self.n_tasks = len(iterations_per_episode) * num_brains
        if "train" in episodes:
            self.train_iterations = episodes["train"] * steps_per_episode
        if "test" in episodes:
            self.test_iterations = {
                k: v * episodes["test"] for k, v in iterations_per_episode.items()
            }

    # --- Public API --------------------------------------------------------
    def train(
        self,
        envs,
        config: TaskConfig,
        record_cfg: RecordingCfg | None = None,
    ) -> None:
        """Train N brains in parallel on slices of the vectorized env.

        In ``config.dry_run`` mode trains just past one rollout/update so peak
        VRAM (including optimizer state and rollout buffers) is committed; no
        wandb run, no hparams JSON, no skrl-owned checkpoints — those are all
        the caller's normal-mode responsibility.
        """
        device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
        wrapped = NettIsaacLabWrapper(envs, device=device)
        # During a dry-run, force wandb off for the per-agent skrl experiment
        # cfg so init never runs. Restore on exit so subsequent real runs
        # keep the user's configured wandb mode.
        wandb_cfg_backup = self.wandb_cfg
        if config.dry_run:
            self.wandb_cfg = {**self.wandb_cfg, "mode": "disabled"}
        try:
            agents = _build_skrl_agents(self, wrapped, device, config=config)
        finally:
            self.wandb_cfg = wandb_cfg_backup
        intrinsic_adapters = self._build_intrinsic_adapters(wrapped, device)

        if config.dry_run:
            # One rollout's worth of timesteps is enough to allocate the
            # replay buffer, run a single update, and commit peak VRAM. The
            # MultiBrainTrainer.train(dry_run=True) path skips hparams and
            # final-checkpoint writes for us.
            short_steps = max(self.batch_size, self.buffer_size)
            MultiBrainTrainer(wrapped, agents, device=device).train(
                TrainCfg(total_timesteps=short_steps),
                record_cfg=None,
                intrinsic_reward_adapters=intrinsic_adapters,
                dry_run=True,
            )
            return

        _load_latest_checkpoints_fn(agents, config, context="train")

        # Checkpoints are owned by skrl now (see ``apply_experiment_cfg`` in
        # ``brain/experiment.py``) — it writes ``agent_{step}.pt`` +
        # ``best_agent.pt`` to each agent's experiment_dir. ``hparams_dir`` is
        # just where the hparams.json backup gets written; skrl captures the
        # same hparams into wandb automatically via ``wandb.init(config=...)``.
        # ``output_dir`` is the run-name level (one above the per-condition
        # ``config.path``); the wandb sync helper needs both so it can find
        # config.yaml at the run root AND the per-condition logs/recordings.
        output_dir = Path(config.path).parent
        MultiBrainTrainer(wrapped, agents, device=device).train(TrainCfg(
            total_timesteps=int(config.train_timesteps or self.train_iterations),
            hparams_dir=Path(config.path),
            hparams={
                "algorithm": getattr(self.algorithm, "__name__", str(self.algorithm)),
                "encoder": getattr(self.encoder, "__name__", str(self.encoder)),
                "model": self.model_cfg.__dict__,
                "reward": str(self.reward_spec),
                "learning_rate": self.learning_rate if not callable(self.learning_rate) else "callable",
                "batch_size": self.batch_size,
                "buffer_size": self.buffer_size,
                "checkpoint_freq": self.checkpoint_freq,
                "envs_per_agent": self.envs_per_agent,
                "total_timesteps": self.train_iterations,
                "chunk_timesteps": int(config.train_timesteps or self.train_iterations),
                "global_step": config.train_global_step,
            },
            output_dir=output_dir,
            condition=config.condition,
            phase=config.current_mode,
            run_name=output_dir.name,
        ), record_cfg=record_cfg, intrinsic_reward_adapters=intrinsic_adapters)

    def test(self, envs, config: TaskConfig) -> dict[int, float]:
        """Greedy rollout. Returns ``{brain_id: mean_reward}``.

        If checkpoints exist under ``{config.path}/wandb_runs/brain_{i}/checkpoints/``, the
        latest one is loaded into agent ``i`` before evaluation. Otherwise
        the agents run with their freshly-initialized random weights and
        return values are essentially noise — useful as a smoke check but
        not as a real evaluation.
        """
        device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
        wrapped = NettIsaacLabWrapper(envs, device=device)
        agents = _build_skrl_agents(self, wrapped, device, config=config)
        _init_agents_for_eval_fn(agents)
        _load_latest_checkpoints_fn(agents, config)
        steps = self.test_iterations.get(config.condition, 1) * self.steps_per_episode
        return MultiBrainTrainer(wrapped, agents, device=device).eval(total_timesteps=steps)

    def record(self, envs, config: TaskConfig, timesteps: int | None = None) -> None:
        """Run a no-learning rollout to let NETTEnv produce recording artifacts."""
        device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
        wrapped = NettIsaacLabWrapper(envs, device=device)
        steps = int(timesteps or self.steps_per_episode)
        states, _ = wrapped.reset()
        for t in range(steps):
            actions = torch.zeros((wrapped.num_envs, wrapped.action_space.shape[0]), device=states.device)
            states, *_ = wrapped.step(actions)

    def _build_intrinsic_adapters(self, env, device: torch.device) -> list | None:
        if not self.uses_intrinsic_reward():
            return None
        return [
            IntrinsicRewardAdapter(
                self.reward,
                env=env,
                device=device,
                weight=self.intrinsic_reward_weight,
                update_enabled=self.intrinsic_reward_update,
                kwargs=self.reward_args,
            )
            for _ in range(env.num_envs)
        ]
