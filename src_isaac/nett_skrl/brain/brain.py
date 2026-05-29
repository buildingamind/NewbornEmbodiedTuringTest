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
from typing import Any, Optional

import torch

from ..runtime.task import TaskConfig
from .config import AlgorithmCfg, EncoderCfg, RewardCfg, algorithm_cfg_from
from .trainer import BrainTrainer, RecordingCfg, TrainCfg
from .models import ModelCfg, model_cfg_from
from .run_config import (
    dry_run_timesteps,
    eval_timesteps,
    policy_device,
    train_cfg_for,
)
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
    algorithm_spec,
    validate_encoder,
    validate_reward,
)


logger = logging.getLogger("nett.brain")

_ENV_REWARD_NAMES = {"closeness", "completeness"}
_RUNTIME_DEFAULTS = {
    "iterations_per_test_episode": {},
    "steps_per_episode": 0,
    "envs_per_brain": 1,
    "num_brains": None,
    "train_iterations": 0,
    "test_iterations": {},
    "n_tasks": 0,
}


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
        checkpoint_freq: Optional[int] = None,
        encoder_cfg: EncoderCfg | dict[str, Any] | None = None,
        algorithm_cfg: AlgorithmCfg | dict[str, Any] | None = None,
        model: Optional[dict[str, Any] | ModelCfg] = None,
        reward_cfg: RewardCfg | dict[str, Any] | None = None,
        wandb: Optional[dict[str, Any]] = None,
    ):
        self._assign_attrs({
            "reward_spec": reward,
            "encoder": validate_encoder(encoder),
            "algorithm": validate_algorithm(algorithm),
            "reward": validate_reward(reward),
        })
        self._raise_if_unsupported_intrinsic_reward(reward)
        spec = algorithm_spec(self.algorithm)

        self._assign_attrs({
            "checkpoint_freq": int(checkpoint_freq) if checkpoint_freq is not None else None,
            "encoder_cfg": EncoderCfg.from_value(encoder_cfg),
            "algorithm_cfg": algorithm_cfg_from(algorithm_cfg, spec),
            "model_cfg": model if isinstance(model, ModelCfg) else model_cfg_from(model),
            "reward_cfg": RewardCfg.from_value(reward_cfg),
            "wandb_cfg": _normalize_wandb_cfg(wandb),
        })

        self._init_runtime_state()

    def _assign_attrs(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            setattr(self, name, value)

    def _raise_if_unsupported_intrinsic_reward(self, reward: str | type | None) -> None:
        if (
            isinstance(reward, str)
            and isinstance(self.reward, type)
            and issubclass(self.reward, UnsupportedIntrinsicReward)
        ):
            self.reward()

    def _init_runtime_state(self) -> None:
        # Populated by NETT.run setup via calc_iterations(...).
        for name, value in _RUNTIME_DEFAULTS.items():
            setattr(self, name, dict(value) if isinstance(value, dict) else value)

    def env_reward_types(self) -> tuple[str, ...]:
        """Return NETTEnv reward names implied by the legacy ``brain.reward`` field."""
        if not isinstance(self.reward_spec, str):
            return ()
        if self.reward_spec in {"unsupervised", ""}:
            return ()
        names = tuple(x.strip() for x in self.reward_spec.split(",") if x.strip())
        if set(names).issubset(_ENV_REWARD_NAMES):
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
        self.envs_per_brain = self.algorithm_cfg.envs_per_brain_for(steps_per_episode)
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
        device, wrapped = self._wrapped_env(envs, config)
        agents = self._build_agents(wrapped, device, config)
        intrinsic_adapters = self._build_intrinsic_adapters(wrapped, device)

        if config.dry_run:
            # One rollout's worth of timesteps is enough to allocate the
            # replay buffer, run a single update, and commit peak VRAM. The
            # BrainTrainer.train(dry_run=True) path skips hparams and
            # final-checkpoint writes for us.
            self._trainer(wrapped, agents, device).train(
                TrainCfg(total_timesteps=dry_run_timesteps(self)),
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
        self._trainer(wrapped, agents, device).train(
            train_cfg_for(self, config),
            record_cfg=record_cfg,
            intrinsic_reward_adapters=intrinsic_adapters,
        )

    def test(self, envs, config: TaskConfig) -> dict[int, float]:
        """Greedy rollout. Returns ``{brain_id: mean_reward}``.

        If checkpoints exist under ``{config.path}/wandb_runs/brain_{i}/checkpoints/``, the
        latest one is loaded into agent ``i`` before evaluation. Otherwise
        the agents run with their freshly-initialized random weights and
        return values are essentially noise — useful as a smoke check but
        not as a real evaluation.
        """
        device, wrapped = self._wrapped_env(envs, config)
        agents = self._build_agents(wrapped, device, config)
        _init_agents_for_eval_fn(agents)
        _load_latest_checkpoints_fn(agents, config)
        return self._trainer(wrapped, agents, device).eval(
            total_timesteps=eval_timesteps(self, config)
        )

    def record(self, envs, config: TaskConfig, timesteps: int | None = None) -> None:
        """Run a no-learning rollout to let NETTEnv produce recording artifacts."""
        _, wrapped = self._wrapped_env(envs, config)
        steps = int(timesteps or self.steps_per_episode)
        states, _ = wrapped.reset()
        for _ in range(steps):
            action_shape = (wrapped.num_envs, wrapped.action_space.shape[0])
            actions = torch.zeros(action_shape, device=states.device)
            states, *_ = wrapped.step(actions)

    def _wrapped_env(self, envs, config: TaskConfig):
        device = policy_device(config)
        return device, NettIsaacLabWrapper(envs, device=device)

    def _build_agents(self, wrapped, device: torch.device, config: TaskConfig):
        # During a dry-run, force wandb off for the per-agent skrl experiment
        # cfg so init never runs. Restore on exit so subsequent real runs keep
        # the user's configured wandb mode.
        wandb_cfg_backup = self.wandb_cfg
        dry_run = bool(getattr(config, "dry_run", False))
        if dry_run:
            self.wandb_cfg = {**self.wandb_cfg, "mode": "disabled"}
        try:
            return _build_skrl_agents(self, wrapped, device, config=config)
        finally:
            self.wandb_cfg = wandb_cfg_backup

    def _trainer(self, wrapped, agents, device: torch.device) -> BrainTrainer:
        return BrainTrainer(wrapped, agents, device=device)

    def _build_intrinsic_adapters(self, env, device: torch.device) -> list | None:
        if not self.uses_intrinsic_reward():
            return None
        return [
            IntrinsicRewardAdapter(
                self.reward,
                env=env,
                device=device,
                weight=self.reward_cfg.weight,
                update_enabled=self.reward_cfg.trainable,
                kwargs=self.reward_cfg.as_kwargs(),
            )
            for _ in range(env.num_envs)
        ]
