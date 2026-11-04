"""Random brightness: each env's TRAINING episode sees its frames multiplied by one factor drawn from U[0.7, 1.3].

⭐ OWNER 2026-11-04 (BL2, binding): "train 6 brains of object 1 on Random Brightness [...] for 3DCNN + CLTT
(attached)". Definition chosen by the owner: "Per episode, ×U[0.7,1.3]" -- one factor per env per TRAINING episode,
multiplying every RGB pixel, clamped to the valid range, fixed within the episode so the two stacked frames never see
a fake brightness change; test is unaugmented; the policy and the CLTT loss both see the augmented frames.
The range is the archive SB3 brightness jitter's ([0.7, 1.3], clamp [0, 1]).

HOW EACH PART OF THAT DEFINITION IS MET:
  * PER EPISODE. The factor is drawn at ``reset`` (every env) and on each ``step`` whose terminated|truncated flag is
    set for that env. A vectorised Isaac env auto-resets, so the observation returned with a done flag is already the
    NEXT episode's first frame: the redraw happens before the factor is applied, and that frame gets the new
    episode's factor (framestack's buffer refill on done reads the same frame, observation.py / framestack.py).
  * BOTH STACKED FRAMES. ``pre``, innermost, before framestack (it refuses more than 3 channels): every frame of an
    episode carries that episode's factor, and framestack refills its buffer on done, so a stack never mixes two
    episodes. NETT_FIRST_FULL_STACK's held ticks are ordinary body steps with no done flag, so they keep the factor.
  * POLICY AND CLTT BOTH. The wrapper changes the observation itself, so the rollout buffer (NETT_UINT8_BUFFER) holds
    the augmented frame, and the CLTT views are built from that buffer.
  * CLAMP. Frames are uint8 [0, 255] (the [0, 1] range scaled): out = clamp(round(x * f), 0, 255), dtype kept. A
    floating-point frame refuses -- its range ([0, 1] or [0, 255]) would have to be guessed.
  * TEST, RECORD AND VALIDATE ARE UNAUGMENTED (factor 1, the frame returned untouched).

⛔ THERE IS NO ENVIRONMENT KNOB (acuity.py's rule): the range is a class attribute and the registry entry and the
model label name the arm, so a row cannot run the control under this arm's name or the reverse.

⛔ AN UNBOUND WRAPPER REFUSES TO STEP. Without ``bind_phase`` it cannot know whether this process trains, and guessing
(augment always, or never) would run a different arm under this arm's name. task_runner._bind_brightness binds it;
tasklist's validation check binds it as ``validate``.

The realised training draws are written to ``<run>/logs/brightness_<phase>_off<N>.csv`` as they happen (the worker
ends in ``os._exit``, so nothing is buffered to the end): verify the distribution from data.
"""

from __future__ import annotations

import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from ..observation import image_layout
from .acuity import _flag_to_numpy, _policy_obs, _replace_policy_obs


class RandomBrightness(gym.Wrapper):
    """Multiply each env's frames by its training episode's factor ~ U[LOW, HIGH]; identity outside training."""

    LOW: float = 0.7
    HIGH: float = 1.3

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._bound = False
        self._train = False
        self._phase = ""
        self._step = 0                              # vectorised steps taken in THIS process
        self._factor: np.ndarray | None = None      # per-env factor (float64)
        self._rng = np.random.default_rng()
        self._log_path: Path | None = None

    # ------------------------------------------------------------------
    def bind_phase(self, phase: str, *, log_dir, offset: int = 0, seed=None) -> None:
        """Declare the phase before the first observation (see task_runner._bind_brightness)."""
        self._phase = str(phase)
        self._train = phase == "train"
        self._rng = np.random.default_rng(seed)
        self._log_path = Path(log_dir) / f"brightness_{phase}_off{int(offset)}.csv"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a", newline="") as f:
            csv.writer(f).writerow(["step", "env_id", "factor", "phase"])
        self._bound = True

    def _check_bound(self):
        if not self._bound:
            raise RuntimeError(
                f"{type(self).__name__} was never bound: the runner must call bind_phase before the first "
                "observation (task_runner._bind_brightness). Refusing to guess the phase.")

    def _redraw(self, env_ids) -> None:
        env_ids = np.asarray(list(env_ids), dtype=np.int64)
        if env_ids.size == 0:
            return
        if not self._train:
            self._factor[env_ids] = 1.0
            return
        draws = self._rng.uniform(self.LOW, self.HIGH, size=env_ids.size)
        self._factor[env_ids] = draws
        if self._log_path is not None:
            with open(self._log_path, "a", newline="") as f:
                csv.writer(f).writerows(
                    [self._step, int(i), f"{d:.6f}", self._phase] for i, d in zip(env_ids, draws))

    # ------------------------------------------------------------------
    @staticmethod
    def _geometry(policy) -> int:
        shape = tuple(policy.shape)
        if len(shape) not in (3, 4):
            raise ValueError(f"brightness expects HWC/NHWC or CHW/NCHW, got shape {shape}")
        layout = image_layout(shape[-3:])
        c = shape[-1] if layout == "hwc" else shape[-3]
        if c not in (1, 3):
            raise ValueError(
                f"brightness received {c} channels; it multiplies RAW frames (3, or 1 grey). Anything else means a "
                "stacking or event wrapper ran first -- put random_brightness in `pre`.")
        return shape[0] if len(shape) == 4 else 1

    def reset(self, **kwargs):
        self._check_bound()
        obs, info = self.env.reset(**kwargs)
        n = self._geometry(_policy_obs(obs))
        self._factor = np.ones(n, dtype=np.float64)
        self._redraw(range(n))
        return _replace_policy_obs(obs, self._apply(_policy_obs(obs))), info

    def step(self, action):
        self._check_bound()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        policy = _policy_obs(obs)
        n = self._geometry(policy)
        if self._factor is None or len(self._factor) != n:
            raise RuntimeError("brightness: step before reset, or the env count changed mid-run")
        # ⛔ Convert CUDA flags explicitly (framestack._done_mask's 2026-09-27 lesson): a swallowed conversion error
        # would leave every env on its first draw for the whole run.
        done = (_flag_to_numpy(terminated).ravel() | _flag_to_numpy(truncated).ravel()).astype(bool)
        if done.size != n:
            raise RuntimeError(f"brightness: {done.size} done flags for {n} envs")
        if done.any():
            self._redraw(np.where(done)[0])
        return _replace_policy_obs(obs, self._apply(policy)), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _apply(self, policy):
        if not self._train:
            return policy
        is_tensor = isinstance(policy, torch.Tensor)
        t = policy if is_tensor else torch.as_tensor(np.asarray(policy))
        if t.dtype.is_floating_point:
            raise TypeError(f"brightness expects integer (uint8) frames, got {t.dtype}; refusing to guess the range.")
        batched = t.dim() == 4
        x = t if batched else t.unsqueeze(0)
        f = torch.as_tensor(self._factor, dtype=torch.float32, device=x.device).view(-1, 1, 1, 1)
        info = torch.iinfo(t.dtype)
        out = (x.to(torch.float32) * f).round().clamp(info.min, info.max).to(t.dtype).contiguous()
        if not batched:
            out = out.squeeze(0)
        return out if is_tensor else out.cpu().numpy()


def find_brightness(env) -> list:
    """Every RandomBrightness on ``env``'s wrapper chain, outermost first (find_acuity's walk)."""
    found, current, seen = [], env, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, RandomBrightness):
            found.append(current)
        current = getattr(current, "_env", None) or getattr(current, "env", None)
    return found
