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

    #: NETT_FRAMESTACK_SHUFFLE=past draws k uniformly from this closed range (presenter spec).
    SHUFFLE_K_MIN, SHUFFLE_K_MAX = 10, 60

    @staticmethod
    def _resolve_shuffle() -> str | None:
        """``NETT_FRAMESTACK_SHUFFLE``: unset/empty = off (byte-identical); ``past`` | ``env``.

        ⭐ PRESENTER ABLATION (owner-confirmed, workspace DECISIONS §63): does temporally
        COHERENT input matter during learning? The t-1 slot of the stack is replaced, in train
        and test alike:
          past -- by the SAME env's frame from k steps back, k ~ U{10..60} drawn per env per
                  step, clamped to the frames of the CURRENT episode (k <- min(k, age)). At an
                  episode's first step t-1 is therefore the first frame itself, exactly what the
                  unshuffled stack holds after the boundary scrub. Never crosses an episode.
          env  -- by ANOTHER env's CURRENT frame, a fresh uniform derangement each step (needs
                  >= 2 envs). A different scene: the stronger control.
        Only the t-1 slot changes (with NETT_FRAMESTACK_N > 2 the older slots stay coherent).
        Any other value refuses.
        """
        raw = os.environ.get("NETT_FRAMESTACK_SHUFFLE", "").strip().lower()
        if raw == "":
            return None
        if raw not in ("past", "env"):
            raise ValueError(f"NETT_FRAMESTACK_SHUFFLE={raw!r}: expected 'past' or 'env' (or unset).")
        return raw

    def __init__(self, env: gym.Env, n_stack: int | None = None) -> None:
        super().__init__(env)
        self.n_stack = self._resolve_n_stack(n_stack)
        self._frames: deque = deque(maxlen=self.n_stack)
        self._base_obs_space = env.observation_space

        self.observation_space = channel_stack_space(env.observation_space, self.n_stack)

        self.shuffle = self._resolve_shuffle()
        if self.shuffle is not None:
            # Host-RAM ring (the stack itself lives in numpy): K_MAX + 1 frames, uint8 at the
            # eye. 448x280x3 x 64 envs x 61 = 1.47 GB; 128x80 x 64 x 61 = 120 MB. No GPU memory.
            self._ring: deque = deque(maxlen=self.SHUFFLE_K_MAX + 1)
            self._age: np.ndarray | None = None
            # Follows the run's seed: torch is seeded before the body wrappers are built.
            self._rng = np.random.default_rng(int(torch.initial_seed()) % (2 ** 32))
            # stdout, not a logger: a close-time check counts this line in the driver log.
            print(f"[NETT framestack] shuffle={self.shuffle}"
                  + (f" k~U{{{self.SHUFFLE_K_MIN}..{self.SHUFFLE_K_MAX}}}" if self.shuffle == "past" else ""),
                  flush=True)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill the frame buffer with the reset observation repeated n_stack times
        self._frames.clear()
        policy_obs = _policy_obs(obs)
        for _ in range(self.n_stack):
            self._frames.append(_obs_to_numpy(policy_obs))
        if self.shuffle is not None:
            self._ring.clear()
            self._ring.append(self._frames[-1])
            n = self._frames[-1].shape[0] if self._frames[-1].ndim == 4 else 1
            self._age = np.zeros(n, dtype=np.int64)
        return _replace_policy_obs(obs, self._stacked()), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_np = _obs_to_numpy(_policy_obs(obs))

        # For vectorised envs: reset per-env frame buffers when episodes end.
        done = _done_mask(terminated, truncated)
        if obs_np.ndim == 4 and done.any():
            for env_id in np.where(done)[0]:
                # Replace every stacked frame for this env with the new episode's
                # first observation so stale frames from the previous episode
                # don't bleed across the boundary.
                for frame in self._frames:
                    frame[env_id] = obs_np[env_id]

        self._frames.append(obs_np)
        if self.shuffle is not None:
            self._age += 1
            self._age[done] = 0                     # a size mismatch raises, never skips
            self._ring.append(obs_np)
        return _replace_policy_obs(obs, self._stacked()), reward, terminated, truncated, info

    # ------------------------------------------------------------------

    def _stacked(self) -> np.ndarray:
        # Stack frames on the last (channel) axis
        if self.shuffle is None:
            return channel_stack_frames(list(self._frames))
        frames = list(self._frames)
        frames[-2] = self._shuffled_prev(frames[-1])
        return channel_stack_frames(frames)

    def _shuffled_prev(self, cur: np.ndarray) -> np.ndarray:
        """The replacement for the t-1 slot (see ``_resolve_shuffle``)."""
        batched = cur.ndim == 4
        if self.shuffle == "env":
            if not batched or cur.shape[0] < 2:
                raise ValueError("NETT_FRAMESTACK_SHUFFLE=env needs a batched env with >= 2 envs.")
            n = cur.shape[0]
            ar = np.arange(n)
            perm = self._rng.permutation(n)
            while np.any(perm == ar):               # uniform derangement by rejection (~e tries)
                perm = self._rng.permutation(n)
            return cur[perm].copy()
        n = cur.shape[0] if batched else 1
        k = self._rng.integers(self.SHUFFLE_K_MIN, self.SHUFFLE_K_MAX + 1, size=n)
        k = np.minimum(k, self._age)
        assert int(k.max()) <= len(self._ring) - 1, (k.max(), len(self._ring))
        if not batched:
            return self._ring[-1 - int(k[0])].copy()
        out = np.empty_like(cur)
        for e in range(n):
            out[e] = self._ring[-1 - int(k[e])][e]
        return out


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


def _flag_to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _done_mask(terminated, truncated) -> np.ndarray:
    # ⛔ FIXED 2026-09-27 (lion C-L13, reproduced on three nodes). This used to be
    # np.asarray(...) inside a bare `except Exception: return None`. A live Isaac env
    # returns CUDA tensors, np.asarray raises on them, the except swallowed it, and
    # step() skipped the scrub when done was None -- so the scrub NEVER ran on Isaac and
    # stacks straddled episode boundaries. Convert tensors explicitly and let any
    # other failure raise: a silently disabled scrub is worse than a crash.
    t = _flag_to_numpy(terminated).ravel()
    tr = _flag_to_numpy(truncated).ravel()
    return (t | tr).astype(bool)
