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

import os
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from ..observation import channel_stack_frames, channel_stack_space


class FrameStack(gym.Wrapper):
    """Stack the last ``n_stack`` observations on the channel axis.

    The observation space shape changes from (H, W, C) → (H, W, C*n_stack)
    (single-env) or (N, H, W, C) → (N, H, W, C*n_stack) (batched).
    """

    #: Campaign default. EVERY arm run before 2026-09-02 used exactly this value:
    #: ``campaign_train.py`` passes ``["framestack"]`` as a bare wrapper name with no
    #: kwargs, so nothing ever overrode it. Keep it at 2 so that history stays readable.
    DEFAULT_N_STACK = 2

    @staticmethod
    def _resolve_n_stack(n_stack: int | None) -> int:
        """Explicit argument wins; otherwise ``NETT_FRAMESTACK_N``; otherwise 2.

        ⛔ A stack depth that disagrees with an encoder's ``num_frames`` SILENTLY
        SCRAMBLES TIME INTO COLOUR and leaves the parameter count untouched, so no
        capacity check can see it. The temporal encoders now raise on a mismatch --
        see ``compact_3dcnn.__init__``. Set the two from the same place.
        """
        if n_stack is not None:
            return int(n_stack)
        raw = os.environ.get("NETT_FRAMESTACK_N")
        if raw is None or raw == "":
            return FrameStack.DEFAULT_N_STACK
        try:
            n = int(raw)
        except ValueError as exc:
            raise ValueError(f"NETT_FRAMESTACK_N must be an integer; got {raw!r}.") from exc
        if n < 2:
            raise ValueError(
                f"NETT_FRAMESTACK_N must be >= 2 (a 'stack' of one frame carries no time); "
                f"got {n}. Use framestack=False on the arm instead."
            )
        return n

    def __init__(self, env: gym.Env, n_stack: int | None = None) -> None:
        super().__init__(env)
        self.n_stack = self._resolve_n_stack(n_stack)
        self._frames: deque = deque(maxlen=self.n_stack)
        self._base_obs_space = env.observation_space

        self.observation_space = channel_stack_space(env.observation_space, self.n_stack)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill the frame buffer with the reset observation repeated n_stack times
        self._frames.clear()
        policy_obs = _policy_obs(obs)
        for _ in range(self.n_stack):
            self._frames.append(_obs_to_numpy(policy_obs))
        return _replace_policy_obs(obs, self._stacked()), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_np = _obs_to_numpy(_policy_obs(obs))

        # For vectorised envs: reset per-env frame buffers when episodes end.
        done = _done_mask(terminated, truncated)
        if done is not None and obs_np.ndim == 4 and done.any():
            for env_id in np.where(done)[0]:
                # Replace every stacked frame for this env with the new episode's
                # first observation so stale frames from the previous episode
                # don't bleed across the boundary.
                for frame in self._frames:
                    frame[env_id] = obs_np[env_id]

        self._frames.append(obs_np)
        return _replace_policy_obs(obs, self._stacked()), reward, terminated, truncated, info

    # ------------------------------------------------------------------

    def _stacked(self) -> np.ndarray:
        # Stack frames on the last (channel) axis
        return channel_stack_frames(list(self._frames))


def _policy_obs(obs):
    return obs.get("policy", obs) if isinstance(obs, dict) else obs


def _replace_policy_obs(obs, policy):
    if not isinstance(obs, dict):
        return policy
    out = dict(obs)
    out["policy"] = policy
    return out


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
