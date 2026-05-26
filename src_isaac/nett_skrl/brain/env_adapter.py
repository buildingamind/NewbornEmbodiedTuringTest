"""skrl wrapper for NETT Isaac Lab envs, including NETT observation wrappers."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

from skrl import config
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.utils.spaces.torch import (
    flatten_tensorized_space,
    tensorize_space,
    unflatten_tensorized_space,
)


class NettIsaacLabWrapper(Wrapper):
    """Flatten ``obs["policy"]`` from raw or Gym-wrapped NETT Isaac envs."""

    def __init__(self, env: Any, device: str | torch.device | None = None) -> None:
        super().__init__(env)
        self._seed = config.torch.key
        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}
        # Move obs onto the policy device so skrl's
        # observation preprocessor — which the brain instantiates on the same
        # device — and the model forwards see matched tensors. Without this,
        # Isaac Lab's CPU-side env step would hand the agent CPU tensors while
        # the preprocessor lives on cuda, and `RunningStandardScaler.forward`
        # raises a device-mismatch error before the first PPO update.
        self._policy_device = torch.device(device or "cpu")

    @property
    def observation_space(self) -> gym.Space:
        return _policy_space(self._env)

    @property
    def action_space(self) -> gym.Space:
        """Return the per-env action space, not Isaac Lab's batched one.

        Isaac Lab's `DirectRLEnv` exposes `action_space` as
        `gym.vector.utils.batch_space(single_action_space, num_envs)`. If
        that batched space reaches skrl, `compute_space_size` reports
        `num_envs * action_dim` (e.g. 4 for num_envs=2, action_dim=2), and
        the PPO model builds a `Linear(..., 4)` `mean_layer`. The vstack
        across agents in `SequentialTrainer.train` then produces a tensor
        whose `unflatten_tensorized_space(Box(shape=(2,2)), …)` reshapes
        to `(-1, 2, 2)` — the (2, 2, 2) shape that fails motor.apply.
        Returning the per-env space here gives skrl `num_actions=2` and
        keeps the action tensor at `(num_envs, action_dim)`.
        """
        unwrapped = self._unwrapped
        if hasattr(unwrapped, "single_action_space"):
            return unwrapped.single_action_space
        return unwrapped.action_space

    @property
    def state_space(self) -> gym.Space | None:
        try:
            return self._unwrapped.single_observation_space["critic"]
        except Exception:
            return None

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._reset_once:
            observations, self._info = self._env.reset(seed=self._seed)
            self._observations = self._flatten_policy_obs(observations)
            states = observations.get("critic") if isinstance(observations, dict) else None
            if states is not None and self.state_space is not None:
                self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
            self._reset_once = False
            self._seed = None
        return self._observations, self._info

    def step(self, actions: torch.Tensor):
        actions = unflatten_tensorized_space(self.action_space, actions)
        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(actions)
        self._observations = self._flatten_policy_obs(observations)
        states = observations.get("critic") if isinstance(observations, dict) else None
        if states is not None and self.state_space is not None:
            self._states = flatten_tensorized_space(tensorize_space(self.state_space, states))
        return (
            self._observations,
            reward.view(-1, 1) if hasattr(reward, "view") else torch.as_tensor(reward).view(-1, 1),
            terminated.view(-1, 1) if hasattr(terminated, "view") else torch.as_tensor(terminated).view(-1, 1),
            truncated.view(-1, 1) if hasattr(truncated, "view") else torch.as_tensor(truncated).view(-1, 1),
            self._info,
        )

    def state(self) -> torch.Tensor | None:
        return self._states

    def render(self, *args, **kwargs) -> Any:
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    def _flatten_policy_obs(self, observations) -> torch.Tensor:
        policy_obs = observations["policy"] if isinstance(observations, dict) else observations
        flat = flatten_tensorized_space(tensorize_space(self.observation_space, policy_obs))
        return flat.to(self._policy_device, non_blocking=True)


def _policy_space(env) -> gym.Space:
    if isinstance(getattr(env, "observation_space", None), gym.spaces.Dict):
        return env.observation_space["policy"]
    try:
        single = env.single_observation_space
        if isinstance(single, gym.spaces.Dict):
            return single["policy"]
        return single
    except Exception:
        return env.observation_space
