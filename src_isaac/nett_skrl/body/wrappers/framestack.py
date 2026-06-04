"""Frame-stacking wrapper that concatenates the last N observations on the channel axis.

Applied before ChannelsFirst so the stacked output is still HWC (ChannelsFirst
converts to CHW as the final step). Temporal encoders (3DCNN, ViViT,
GuessWhatMoves) then reshape the concatenated channels back into a (C, T, H, W)
volume internally.

Vectorised Isaac Lab envs return batched (N, H, W, C) tensors. This wrapper
correctly handles both the single-env (H, W, C) and batched (N, H, W, C) cases
and properly resets per-environment frame buffers on episode completion.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import torch


class FrameStack(gym.Wrapper):
    """Stack the last ``n_stack`` observations on the channel axis.

    The observation space shape changes from (H, W, C) → (H, W, C*n_stack)
    (single-env) or (N, H, W, C) → (N, H, W, C*n_stack) (batched).
    """

    def __init__(self, env: gym.Env, n_stack: int = 2) -> None:
        super().__init__(env)
        self.n_stack = int(n_stack)
        self._frames: deque = deque(maxlen=self.n_stack)
        self._base_obs_space = env.observation_space

        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Box):
            shape = list(obs_space.shape)
            if len(shape) == 3:
                # Single-env HWC: stack on channel (last) axis
                shape[-1] *= self.n_stack
            elif len(shape) == 4:
                # Batched NHWC: stack on channel (last) axis
                shape[-1] *= self.n_stack
            else:
                shape[-1] *= self.n_stack
            high_val = 255 if np.issubdtype(obs_space.dtype, np.integer) else 1.0
            self.observation_space = gym.spaces.Box(
                low=np.zeros(shape, dtype=obs_space.dtype),
                high=np.full(shape, high_val, dtype=obs_space.dtype),
                dtype=obs_space.dtype,
            )
        else:
            self.observation_space = obs_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill the frame buffer with the reset observation repeated n_stack times
        self._frames.clear()
        for _ in range(self.n_stack):
            self._frames.append(_obs_to_numpy(obs))
        return self._stacked(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_np = _obs_to_numpy(obs)

        # For vectorised envs: reset per-env frame buffers when episodes end.
        done = _done_mask(terminated, truncated)
        if done is not None and obs_np.ndim == 4 and done.any():
            prev = _obs_to_numpy(self._frames[-1])
            for env_id in np.where(done)[0]:
                # Replace every stacked frame for this env with the new episode's
                # first observation so stale frames from the previous episode
                # don't bleed across the boundary.
                for frame in self._frames:
                    frame[env_id] = obs_np[env_id]

        self._frames.append(obs_np)
        return self._stacked(), reward, terminated, truncated, info

    # ------------------------------------------------------------------

    def _stacked(self) -> np.ndarray:
        # Stack frames on the last (channel) axis
        return np.concatenate(list(self._frames), axis=-1)


def _obs_to_numpy(obs) -> np.ndarray:
    if isinstance(obs, torch.Tensor):
        return obs.detach().cpu().numpy().copy()
    return np.asarray(obs).copy()


def _done_mask(terminated, truncated) -> np.ndarray | None:
    try:
        t = np.asarray(terminated).ravel()
        tr = np.asarray(truncated).ravel()
        return (t | tr).astype(bool)
    except Exception:
        return None
