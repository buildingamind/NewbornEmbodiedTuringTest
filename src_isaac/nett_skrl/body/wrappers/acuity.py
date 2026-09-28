"""Visual-acuity curriculum: Gaussian blur on the policy observation, scheduled over TRAINING.

⛔ WHAT THIS PORTS, READ FROM THE SOURCE, NOT A PARAPHRASE OF IT.
``rajankita/Visual_Acuity_Curriculum`` @ 79772b08b0, ``train_vac.py`` (read 2026-09-28):

    l.318-327  ``--epochs`` are stage DURATIONS. ``cumsum([13, 27, 160])`` -> boundaries 13, 40, 200,
               i.e. stage k runs while epoch < boundary[k]. As fractions of training: .065 / .20 / 1.0.
    l.328-331  REPLAY: in stage k, sigma is drawn from ``sigmas[:k+1]`` with p = the CUMULATIVE
               boundaries reached so far, normalised -- stage 3 draws [2, 1, 0] with
               [13, 40, 200] / 253 = .051 / .158 / .791. Fixed within a stage, not ramped.
    l.398-399  NO-REPLAY ablation (``--no_replay``): the current stage's sigma alone.
    l.405-409  ``kernel = int(6*sigma+1)``; ``gaussian_blur(sample, kernel, sigma)``; sigma 0 = no blur.
    l.425-437  ONE sigma per MINIBATCH: ``blur(input)`` is applied to the whole batch tensor.

⚠ THE ISAAC MAPPING, EACH A STATED CHOICE (FINDINGS §4dh.44h.24):
  * SIGMA BASIS = FRACTION OF IMAGE WIDTH. [2, 1, 0] at 32 px -> [1/16, 1/32, 0] x width, so
    [28, 14, 0] px on the 448x280 eye. This carries the METHOD over, not chick acuity.
  * GRANULARITY = one sigma per env per EPISODE, drawn when the episode starts. The paper's
    "per minibatch" has no embodied twin; redrawing every step would flicker blur inside an
    episode, which the paper never had. This is the one deliberate divergence.
  * PROGRESS = global vectorised train step / total vectorised train steps. It survives
    ``eval_freq`` chunks because the runner binds the chunk's global start (``bind_phase``).
  * TEST AND RECORD ARE SHARP (identity). The schedule shapes training only.
  * REV = the forward schedule TIME-REVERSED, p(sigma | 1 - progress): identical marginal
    exposure to each sigma, only the order differs.

⛔ THERE IS NO ENVIRONMENT KNOB. The schedule is a CLASS attribute and each schedule is its own
registry entry and model label, so the model NAME always says which schedule ran (the same rule
``segmentation_wrappers`` states for every body wrapper). A row cannot run the control while
logging the curriculum.

⛔ AN UNBOUND WRAPPER REFUSES TO STEP. Without ``bind_phase`` the wrapper cannot know the phase or
the progress, and guessing either (e.g. progress 0 forever = maximum blur, or identity) would run
a different arm under this arm's name.

The realised draws are written to ``<run>/logs/acuity_<mode>_off<N>.csv`` as they happen (the
worker ends in ``os._exit``, so nothing is buffered to the end): verify the schedule from data.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from ..observation import image_layout

#: ``train_vac.py``'s headline config: ``--sigmas 2 1 0 --epochs 13 27 160`` at 32 px.
SIGMA_FRACS = (2.0 / 32.0, 1.0 / 32.0, 0.0)
STAGE_DURATIONS = (13, 27, 160)


def _policy_obs(obs):
    """Identical to ``framestack._policy_obs``."""
    return obs.get("policy", obs) if isinstance(obs, dict) else obs


def _replace_policy_obs(obs, policy):
    """Identical to ``framestack._replace_policy_obs``."""
    if not isinstance(obs, dict):
        return policy
    out = dict(obs)
    out["policy"] = policy
    return out


def _flag_to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def stage_table(durations=STAGE_DURATIONS) -> np.ndarray:
    """Cumulative stage boundaries as fractions of training (l.318-319)."""
    cum = np.cumsum(np.asarray(durations, dtype=np.float64))
    return cum / cum[-1]


def stage_of(progress: float, durations=STAGE_DURATIONS) -> int:
    """Stage index at ``progress`` in [0, 1]: the first boundary strictly above it (l.325)."""
    bounds = stage_table(durations)
    p = min(max(float(progress), 0.0), 1.0)
    for k, b in enumerate(bounds):
        if b > p:
            return k
    return len(bounds) - 1  # progress == 1.0 exactly: the last stage


def replay_weights(stage: int, durations=STAGE_DURATIONS) -> np.ndarray:
    """p over ``sigmas[:stage+1]`` = cumulative boundaries, normalised (l.328-331)."""
    cum = np.cumsum(np.asarray(durations, dtype=np.float64))[: stage + 1]
    return cum / cum.sum()


def kernel_size(sigma: float) -> int:
    """``int(6*sigma+1)``, as the reference, forced odd (torchvision requires odd kernels)."""
    k = int(6 * sigma + 1)
    return k if k % 2 == 1 else k + 1


def gaussian_blur_nchw(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, reflect-padded: numerically ``torchvision.transforms.functional.
    gaussian_blur(x, kernel_size(sigma), sigma)`` (pinned by test_acuity.py), without its dense 2-D
    kernel -- at sigma 28 that kernel is 169x169, ~170 GFLOP per 16-env step, against ~2 separable."""
    if sigma <= 0:
        return x
    k = kernel_size(sigma)
    half = (k - 1) * 0.5
    t = torch.linspace(-half, half, steps=k, device=x.device, dtype=torch.float32)
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    g = (g / g.sum()).to(x.dtype)
    c = x.shape[1]
    pad = k // 2
    y = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    y = F.conv2d(y, g.view(1, 1, 1, k).expand(c, 1, 1, k), groups=c)
    y = F.pad(y, (0, 0, pad, pad), mode="reflect")
    return F.conv2d(y, g.view(1, 1, k, 1).expand(c, 1, k, 1), groups=c)


class AcuityCurriculum(gym.Wrapper):
    """Blur each env's policy frames with its episode's sigma; the schedule is the class's."""

    #: "vac" (replay, forward), "norep" (monotone, the published ablation), "rev" (time-reversed).
    SCHEDULE: str = ""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        if self.SCHEDULE not in ("vac", "norep", "rev"):
            raise TypeError(
                f"{type(self).__name__} has SCHEDULE={self.SCHEDULE!r}; use a registry entry "
                "(acuity_vac / acuity_norep / acuity_rev), never the base class.")
        self._bound = False
        self._train = False
        self._step = 0          # vectorised steps taken in THIS process
        self._start = 0         # global vectorised step this process starts at
        self._total = 1         # global vectorised steps in the whole training
        self._sigma: np.ndarray | None = None   # per-env sigma, in pixels
        self._rng = np.random.default_rng()
        self._log_path: Path | None = None

    # ------------------------------------------------------------------
    def bind_phase(self, phase: str, *, start_step: int, total_steps: int,
                   log_dir, offset: int = 0, seed=None) -> None:
        """Declare mode and progress before the first observation (see task_runner)."""
        if total_steps <= 0:
            raise ValueError(f"acuity: total_steps must be positive, got {total_steps}")
        self._train = phase == "train"
        self._start, self._total = int(start_step), int(total_steps)
        self._rng = np.random.default_rng(seed)
        self._log_path = Path(log_dir) / f"acuity_{phase}_off{int(offset)}.csv"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._log_path, "a", newline="") as f:
            csv.writer(f).writerow(["global_step", "progress", "env_id", "stage", "sigma_px",
                                    "schedule", "phase"])
        self._bound = True

    def progress(self) -> float:
        return min(1.0, (self._start + self._step) / self._total)

    def _draw(self, width: int) -> tuple[float, int]:
        """(sigma in PIXELS, schedule stage) for an episode starting now."""
        p = self.progress()
        if self.SCHEDULE == "rev":
            p = 1.0 - p
        k = stage_of(p)
        if self.SCHEDULE == "norep":
            frac = SIGMA_FRACS[k]
        else:
            frac = SIGMA_FRACS[int(self._rng.choice(k + 1, p=replay_weights(k)))]
        return frac * width, k

    def _redraw(self, env_ids, width: int) -> None:
        rows = []
        for i in env_ids:
            s, k = self._draw(width) if self._train else (0.0, -1)
            self._sigma[i] = s
            rows.append([self._start + self._step, f"{self.progress():.6f}", int(i), k,
                         f"{s:.4f}", self.SCHEDULE, "train" if self._train else "eval"])
        if rows and self._log_path is not None and (self._train or self._step == 0):
            with open(self._log_path, "a", newline="") as f:
                csv.writer(f).writerows(rows)

    # ------------------------------------------------------------------
    def _geometry(self, policy):
        shape = tuple(policy.shape)
        if len(shape) not in (3, 4):
            raise ValueError(f"acuity expects HWC/NHWC or CHW/NCHW, got shape {shape}")
        layout = image_layout(shape[-3:])
        n = shape[0] if len(shape) == 4 else 1
        width = shape[-2] if layout == "hwc" else shape[-1]
        return layout, n, width

    def _check_bound(self):
        if not self._bound:
            raise RuntimeError(
                f"{type(self).__name__} was never bound: the runner must call bind_phase before "
                "the first observation (task_runner._bind_acuity). Refusing to guess the phase.")

    def reset(self, **kwargs):
        self._check_bound()
        obs, info = self.env.reset(**kwargs)
        _, n, width = self._geometry(_policy_obs(obs))
        self._sigma = np.zeros(n, dtype=np.float64)
        self._redraw(range(n), width)
        return _replace_policy_obs(obs, self._blur(_policy_obs(obs))), info

    def step(self, action):
        self._check_bound()
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step += 1
        policy = _policy_obs(obs)
        _, n, width = self._geometry(policy)
        if self._sigma is None or len(self._sigma) != n:
            raise RuntimeError("acuity: step before reset, or the env count changed mid-run")
        # ⛔ Convert CUDA flags explicitly (framestack._done_mask's 2026-09-27 lesson): a swallowed
        # conversion error would leave every env on its first draw for the whole run.
        done = (_flag_to_numpy(terminated).ravel() | _flag_to_numpy(truncated).ravel()).astype(bool)
        if done.any():
            self._redraw(np.where(done)[0], width)
        return _replace_policy_obs(obs, self._blur(policy)), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def _blur(self, policy):
        if not np.any(self._sigma > 0):
            return policy
        is_tensor = isinstance(policy, torch.Tensor)
        t = policy if is_tensor else torch.as_tensor(np.asarray(policy))
        layout, _, _ = self._geometry(t)
        batched = t.dim() == 4
        x = t if batched else t.unsqueeze(0)
        x = x.to(torch.float32)
        if layout == "hwc":
            x = x.permute(0, 3, 1, 2)
        out = x.clone()
        for s in np.unique(self._sigma):
            if s <= 0:
                continue
            idx = torch.as_tensor(np.where(self._sigma == s)[0], device=x.device)
            out[idx] = gaussian_blur_nchw(x[idx], float(s))
        if layout == "hwc":
            out = out.permute(0, 2, 3, 1)
        if not t.dtype.is_floating_point:
            info = torch.iinfo(t.dtype)
            out = out.round().clamp(info.min, info.max)
        out = out.to(t.dtype).contiguous()
        if not batched:
            out = out.squeeze(0)
        return out if is_tensor else out.cpu().numpy()


class AcuityVAC(AcuityCurriculum):
    SCHEDULE = "vac"


class AcuityNoReplay(AcuityCurriculum):
    SCHEDULE = "norep"


class AcuityReversed(AcuityCurriculum):
    SCHEDULE = "rev"


def find_acuity(env) -> list:
    """Every AcuityCurriculum on ``env``'s wrapper chain, outermost first (find_segmenters' walk)."""
    found, current, seen = [], env, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, AcuityCurriculum):
            found.append(current)
        current = getattr(current, "_env", None) or getattr(current, "env", None)
    return found
