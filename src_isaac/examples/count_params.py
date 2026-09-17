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


def aux_head_params(spec: dict, enc) -> tuple[int, str]:
    """(parameters the AUX optimizer group carries, note) for this arm's aux loss.

    ⛔ `aux.head` IS THE CONTRACT, NOT A CONVENIENCE. `AuxLossPPO.__init__` does exactly
    ``optimizer.add_param_group({"params": list(self._aux.head.parameters())})`` and registers
    nothing else, so a parameter not reachable from `head` is never stepped. Counting the whole
    module instead would (a) add the shared ENCODER back in -- every aux holds a reference to it
    -- and (b) count parameters that do not train. Counting `head` answers the question a
    capacity table is asked: what does this row carry ON TOP of the encoder?

    ⚠ THE EMA TEACHER IS DELIBERATELY ABSENT FROM THIS NUMBER, and it is not free: wave 17's
    terms hold a frozen deepcopy of the trunk (`ema_teacher.py` keeps it in a LIST so it is not
    registered). It costs memory, it is not trained, and it is not a capacity difference between
    rows -- every wave-17 row has exactly one. The guard below asserts the encoder itself never
    appears in `head`, which is the failure that WOULD silently double a row's count.

    ⚠ THIS BUILDS THE AUX FOR REAL. Some terms read env knobs at construction (slot_fg's
    NETT_AUX_SLOTFG_EGO decides whether the ego routing is inside its head), so this number is a
    function of the ENVIRONMENT as well as the spec -- run the script under the row's env and the
    header line below records what was set.
    """
    kind = spec.get("aux")
    if not kind:
        return 0, "-"
    from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES

    if kind not in AUX_LOSSES:
        return -1, f"UNREGISTERED aux {kind!r}"
    aux = AUX_LOSSES[kind](enc)
    head = getattr(aux, "head", None)
    if head is None:
        return -1, f"{type(aux).__name__} has no .head"
    enc_ids = {id(p) for p in enc.parameters()}
    params = list(head.parameters())
    if any(id(p) in enc_ids for p in params):
        return -1, f"{kind}: head contains ENCODER parameters (would be stepped twice)"
    return sum(p.numel() for p in params), kind


def count(spec: dict, w: int, h: int, with_aux: bool = True):
    cfg = dict(spec["cfg"])
    cfg.pop("trainable", None)
    space, names = wrapped_space(spec, w, h)
    enc = encoder_mapping[spec["encoder"]](space, **cfg)
    enc_params = sum(p.numel() for p in enc.parameters())
    fd = int(getattr(enc, "features_dim", cfg.get("features_dim", 512)))
    # PPO heads with hidden_sizes=[]: gaussian policy mean Linear(fd->act) +
    # log_std param, value Linear(fd->1). (shared encoder, so counted once.)
    head_params = (fd * ACTION_DIM + ACTION_DIM) + ACTION_DIM + (fd * 1 + 1)
    aux_params, aux_note = (0, "-")
    if with_aux:
        try:
            aux_params, aux_note = aux_head_params(spec, enc)
        except Exception as exc:                              # noqa: BLE001
            aux_params, aux_note = -1, f"{type(exc).__name__}: {exc}"[:60]
    total = enc_params + head_params + max(aux_params, 0)
    return enc_params, aux_params, total, int(space.shape[-1]), names, aux_note


if __name__ == "__main__":
    W, H, src = eye_wh()
    no_aux = "--no-aux" in sys.argv
    print(f"eye = {W} wide x {H} high   (from {src})")
    # ⚠ The aux column depends on the ENVIRONMENT (see aux_head_params). Record what was set, so
    # a pasted table cannot be read as unconditional.
    aux_env = {k: v for k, v in sorted(os.environ.items()) if k.startswith("NETT_AUX_")}
    print(f"aux heads: {'SKIPPED (--no-aux)' if no_aux else 'built'}"
          f"   NETT_AUX_* in this environment: {aux_env or 'none (all knobs at their defaults)'}\n")
    print(f"{'model':22} {'encoder':16} {'ch':>3} {'enc_params':>12} {'aux_head':>10} "
          f"{'total':>12}  aux / wrappers")
    for model, spec in MODELS.items():
        try:
            enc_p, aux_p, total, ch, names, aux_note = count(spec, W, H, with_aux=not no_aux)
            acol = "ERR" if aux_p < 0 else (f"{aux_p:,}" if aux_p else "-")
            print(f"{model:22} {spec['encoder']:16} {ch:>3} {enc_p:12,} {acol:>10} {total:12,}  "
                  f"{aux_note} / {','.join(names) or '-'}")
        except Exception as exc:                              # noqa: BLE001
            print(f"{model:22} {spec['encoder']:16} ERROR: {type(exc).__name__}: {exc}")
