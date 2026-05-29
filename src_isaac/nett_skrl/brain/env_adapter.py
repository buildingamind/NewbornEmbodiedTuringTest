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


class IsaacEnvWrapper(Wrapper):
    """Flatten ``obs["policy"]`` from raw or Gym-wrapped NETT Isaac envs."""

    def __init__(self, env: Any, device: str | torch.device | None = None) -> None:
        super().__init__(env)
        self._seed = config.torch.key
        self._reset_once = True
        self._observations = None
        self._states = None
        self._info = {}
        # Keep env observations on the same device as skrl preprocessors/models.
        self._policy_device = torch.device(device or "cpu")

    @property
    def num_agents(self) -> int:
        return int(getattr(self._env, "num_agents", 1))

    @property
    def observation_space(self) -> gym.Space:
        return _policy_space(self._env)

    @property
    def action_space(self) -> gym.Space:
        """Return the per-env action space, not Isaac Lab's batched space."""
        unwrapped = self._unwrapped
        if hasattr(unwrapped, "single_action_space"):
            return unwrapped.single_action_space
        return unwrapped.action_space

    @property
    def state_space(self) -> gym.Space | None:
        try:
            return _space_value(self._unwrapped.single_observation_space, "critic")
        except Exception:
            return None

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        if self._reset_once:
            observations, self._info = self._env.reset(seed=self._seed)
            self._update_observations(observations)
            self._reset_once = False
            self._seed = None
        return self._observations, self._info

    def step(self, actions: torch.Tensor):
        actions = unflatten_tensorized_space(self.action_space, actions)
        with torch.no_grad():
            observations, reward, terminated, truncated, self._info = self._env.step(actions)
        self._update_observations(observations)
        return (
            self._observations,
            _column_tensor(reward),
            _column_tensor(terminated),
            _column_tensor(truncated),
            self._info,
        )

    def state(self) -> torch.Tensor | None:
        return self._states

    def render(self, *args, **kwargs) -> Any:
        if hasattr(self._env, "render"):
            return self._env.render(*args, **kwargs)
        return None

    def close(self) -> None:
        if hasattr(self._env, "close"):
            self._env.close()

    def _flatten_policy_obs(self, observations) -> torch.Tensor:
        policy_obs = _obs_value(observations, "policy", default=observations)
        flat = flatten_tensorized_space(tensorize_space(self.observation_space, policy_obs))
        return flat.to(self._policy_device, non_blocking=True)

    def _flatten_state_obs(self, observations) -> torch.Tensor | None:
        state_space = self.state_space
        states = _obs_value(observations, "critic")
        if states is None or state_space is None:
            return None
        return flatten_tensorized_space(tensorize_space(state_space, states))

    def _update_observations(self, observations) -> None:
        self._observations = self._flatten_policy_obs(observations)
        self._states = self._flatten_state_obs(observations)


def _column_tensor(value) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.view(-1, 1)


def _obs_value(observations, key: str, default=None):
    if isinstance(observations, dict):
        return observations.get(key, default)
    return default


def _policy_space(env) -> gym.Space:
    policy_space = _space_value(getattr(env, "observation_space", None), "policy")
    if policy_space is not None:
        return policy_space

    try:
        single = env.single_observation_space
    except Exception:
        return env.observation_space

    return _space_value(single, "policy", default=single)


def _space_value(space, key: str, default=None):
    if isinstance(space, gym.spaces.Dict):
        return space[key]
    return default
