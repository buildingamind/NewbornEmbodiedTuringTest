"""Count encoder (+ PPO head) parameters for each campaign model at the live eye.

Builds each encoder exactly as agent_factory would (``encoder_cls(obs_space, **cfg)``) on the
observation space the arm's OWN body wrappers produce, so the number printed is the number that
runs.

Usage: python examples/count_params.py  [NETT_EYE_W=... NETT_EYE_H=... to override]

⛔ TWO ASSUMPTIONS WERE REMOVED FROM THIS FILE ON 2026-09-16, BOTH OF WHICH PUBLISHED WRONG
NUMBERS SILENTLY -- the counts still looked like parameter counts, so nothing announced it:

  1. A SQUARE EYE (``(RES, RES, C)`` from ``NETT_RES``, default 128). The live eye has been
     128 WIDE x 80 HIGH since 2026-08-02 (``ObservationCfg.eye_resolution``). For a ViT the
     token grid is H/patch x W/patch, so a square 128 reported 8x8=64 tokens where the arm runs
     5x8=40 -- and the whole point of tuning a config here is to match capacity.
  2. ``channels = 3 * (2 if framestack else 1)``. This ignores ``pre`` wrappers entirely, and
     ``dvs_polarity`` makes a frame TWO channels, not three. The ViT-CLTT-Ref-DVS row would
     have been counted at 6 input channels while running at 4.

Both are now derived rather than assumed: the eye is read out of the file that DECLARES it, and
the channel count comes from applying the spec's real wrapper list.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

from nett_skrl.body.wrappers.registry import validate_wrappers
from nett_skrl.brain.registry import encoder_mapping
from campaign_train import MODELS, segmentation_wrappers  # reuse the campaign specs

ACTION_DIM = 2  # 2D action space (turn + forward)

def _cfg_path() -> Path:
    """Locate ``nett_isaac/nett_env_cfg.py`` on ``sys.path``.

    ⚠ Found the same way the ARM finds it -- through PYTHONPATH -- rather than by a relative
    hop from this file. A hardcoded ``../../`` would resolve against wherever the worktree
    happens to sit and would go stale the first time anyone runs from a second checkout.
    """
    for root in sys.path:
        if not root:
            continue
        cand = Path(root) / "nett_isaac" / "nett_env_cfg.py"
        if cand.is_file():
            return cand
    raise SystemExit(
        "nett_isaac/nett_env_cfg.py is not on sys.path, so the eye resolution cannot be read "
        "from its declaration. Add the isaac_lab source to PYTHONPATH, or set NETT_EYE_W and "
        "NETT_EYE_H explicitly -- this script will not assume a square eye."
    )


def eye_wh() -> tuple[int, int, str]:
    """(width, height, provenance). Read from the declaring file; never hardcoded here.

    ⚠ The obvious move -- ``from nett_isaac.lens import eye_resolution`` -- does not work: that
    package pulls in Omniverse Kit and blocks on a EULA prompt. So the declaration is read as a
    LITERAL out of its own source instead of being copied into this file. That keeps one source
    of truth; it does mean a computed (rather than literal) default would not be readable, which
    is why the failure path below names the file rather than guessing a value.
    """
    w, h = os.environ.get("NETT_EYE_W"), os.environ.get("NETT_EYE_H")
    if w and h:
        return int(w), int(h), "NETT_EYE_W/H"
    try:
        path = _cfg_path()
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ObservationCfg":
                for st in node.body:
                    if isinstance(st, ast.AnnAssign) and getattr(st.target, "id", "") == "eye_resolution":
                        res = ast.literal_eval(st.value)
                        if res is not None:
                            return int(res[0]), int(res[1]), f"{path.name}:eye_resolution"
                for st in node.body:
                    if isinstance(st, ast.AnnAssign) and getattr(st.target, "id", "") == "input_resolution":
                        sq = int(ast.literal_eval(st.value))
                        return sq, sq, f"{path.name}:input_resolution (square)"
    except SystemExit:
        raise
    except Exception as exc:                                  # noqa: BLE001
        raise SystemExit(
            f"cannot read the eye resolution: {exc}. Set NETT_EYE_W and NETT_EYE_H explicitly "
            "rather than letting this script assume a square eye."
        ) from exc
    raise SystemExit("no eye_resolution/input_resolution found in ObservationCfg")


class _StubEnv(gym.Env):
    """Just enough env for a wrapper to compute its output observation space."""

    def __init__(self, space):
        self.observation_space = space
        self.action_space = gym.spaces.Discrete(2)


def wrapped_space(spec: dict, w: int, h: int) -> tuple[gym.spaces.Box, list[str]]:
    """The observation space the ENCODER sees, after this arm's own body wrappers."""
    names = segmentation_wrappers(spec)
    space = gym.spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
    env = _StubEnv(space)
    for cls in validate_wrappers(names):
        env = cls(env)
    out = env.observation_space
    return (out["policy"] if isinstance(out, gym.spaces.Dict) else out), names


def count(spec: dict, w: int, h: int) -> tuple[int, int, int, list[str]]:
    cfg = dict(spec["cfg"])
    cfg.pop("trainable", None)
    space, names = wrapped_space(spec, w, h)
    enc = encoder_mapping[spec["encoder"]](space, **cfg)
    enc_params = sum(p.numel() for p in enc.parameters())
    fd = int(getattr(enc, "features_dim", cfg.get("features_dim", 512)))
    # PPO heads with hidden_sizes=[]: gaussian policy mean Linear(fd->act) +
    # log_std param, value Linear(fd->1). (shared encoder, so counted once.)
    head_params = (fd * ACTION_DIM + ACTION_DIM) + ACTION_DIM + (fd * 1 + 1)
    return enc_params, enc_params + head_params, int(space.shape[-1]), names


if __name__ == "__main__":
    W, H, src = eye_wh()
    print(f"eye = {W} wide x {H} high   (from {src})\n")
    print(f"{'model':22} {'encoder':16} {'ch':>3} {'enc_params':>12} {'total':>12}  wrappers")
    for model, spec in MODELS.items():
        try:
            enc_p, total, ch, names = count(spec, W, H)
            print(f"{model:22} {spec['encoder']:16} {ch:>3} {enc_p:12,} {total:12,}  "
                  f"{','.join(names) or '-'}")
        except Exception as exc:                              # noqa: BLE001
            print(f"{model:22} {spec['encoder']:16} ERROR: {type(exc).__name__}: {exc}")
