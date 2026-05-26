"""Env-side wrappers used by the skrl training loop.

These wrappers sit between skrl and the Isaac/Gym env. They are deliberately
kept out of :mod:`nett_skrl.wrappers`, which is reserved for observation
transforms such as video frame stacking, DVS, and retina preprocessing.
"""

from __future__ import annotations

import torch


class IntrinsicRewardEnvWrapper:
    """Reward-shaping wrapper used before skrl records transitions."""

    def __init__(self, env, adapters: list) -> None:
        self._env = env
        self.adapters = adapters
        self._last_observations = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        observations, infos = self._env.reset()
        self._last_observations = observations
        return observations, infos

    def step(self, actions):
        next_observations, rewards, terminated, truncated, infos = self._env.step(actions)
        if self._last_observations is not None:
            rewards = rewards.clone()
            for i, adapter in enumerate(self.adapters):
                obs_i = self._last_observations[i : i + 1]
                next_i = next_observations[i : i + 1]
                reward_i = rewards[i : i + 1]
                action_i = actions[i : i + 1]
                term_i = terminated[i : i + 1]
                trunc_i = truncated[i : i + 1]
                if hasattr(adapter, "watch"):
                    adapter.watch(obs_i, action_i, reward_i, term_i, trunc_i, next_i)
                if hasattr(adapter, "compute"):
                    intrinsic = adapter.compute(
                        observations=obs_i,
                        actions=action_i,
                        rewards=reward_i,
                        terminated=term_i,
                        truncated=trunc_i,
                        next_observations=next_i,
                    )
                else:
                    intrinsic = torch.zeros_like(reward_i)
                if hasattr(adapter, "update"):
                    adapter.update()
                rewards[i : i + 1] = reward_i + torch.as_tensor(
                    intrinsic, device=reward_i.device, dtype=reward_i.dtype
                ).view_as(reward_i)
        self._last_observations = next_observations
        return next_observations, rewards, terminated, truncated, infos


class SkrlEnvCompatibilityWrapper:
    """Adds optional skrl Wrapper attributes that simple test envs omit."""

    def __init__(self, env) -> None:
        self._env = env
        self.num_agents = getattr(env, "num_agents", 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def state(self):
        env_state = getattr(self._env, "state", None)
        return env_state() if callable(env_state) else None

    def render(self, *args, **kwargs):
        env_render = getattr(self._env, "render", None)
        return env_render(*args, **kwargs) if callable(env_render) else None

    def close(self):
        env_close = getattr(self._env, "close", None)
        return env_close() if callable(env_close) else None
