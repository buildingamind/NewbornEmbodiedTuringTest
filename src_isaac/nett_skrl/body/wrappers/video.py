"""Frame-stack observation wrapper for Isaac/skrl observations."""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import torch

from ...observation import channel_stack_frames, channel_stack_space


class Video(gym.Wrapper):
    """Stack recent policy observations while preserving dict observations."""

    def __init__(self, env, frames: int = 2, *args, **kwargs):
        super().__init__(env)
        self.frames = int(frames)
        self._frames = deque(maxlen=self.frames)
        self._policy_shape = _policy_space(env.observation_space).shape
        self.observation_space = channel_stack_space(env.observation_space, self.frames)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        policy = obs["policy"] if isinstance(obs, dict) else obs
        self._frames.clear()
        for _ in range(self.frames):
            self._frames.append(_clone_frame(policy))
        return self._with_stack(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        policy = obs["policy"] if isinstance(obs, dict) else obs
        self._frames.append(_clone_frame(policy))
        return self._with_stack(obs), reward, terminated, truncated, info

    def _with_stack(self, obs):
        stacked = channel_stack_frames(self._frames, self._policy_shape)
        if isinstance(obs, dict):
            out = dict(obs)
            out["policy"] = stacked
            return out
        return stacked


def _policy_space(space: gym.Space) -> gym.Space:
    if isinstance(space, gym.spaces.Dict):
        return space["policy"]
    return space


def _clone_frame(policy):
    if isinstance(policy, torch.Tensor):
        return policy.clone()
    return np.array(policy, copy=True)
