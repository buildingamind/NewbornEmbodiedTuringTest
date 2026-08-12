"""Does the POLICY read the binding direction its own encoder provides?

Phase 14 (`probe_frozen_features.py`) showed the conjunction is linearly decodable
from every arm's frozen features -- the CNN's best of all (0.763) -- while the CNN's
behavioural binding is the worst (0.5428). That located the failure downstream of the
representation but did not show where. This does.

The campaign policy head is exactly ``Linear(512 -> 2)`` with a diagonal Gaussian
(`mean_layer.weight`, `log_std`), so "what the policy does with a feature difference"
is not a metaphor -- it is one matrix product, and it can be put in the policy's own
units:

    d' = || (W @ df) / sigma ||      df = mean features(foil) - mean features(imprint)

d' is the separation between the two stimuli IN ACTION SPACE, measured in units of the
policy's own exploration noise. d' ~ 0 means the policy emits statistically
indistinguishable actions for imprint and foil no matter what its features encode --
the information is present and not read.

Two references make d' interpretable:

  RANDOM      the same computation with a random direction of identical norm. The
              readout is rank 2 out of 512, so an unattended direction is nearly
              invisible; this is that floor.
  OPTIMAL     ||df|| / mean(sigma), the d' a policy would get if its 2-d readout were
              aligned with df. This is the ceiling the encoder makes available.

``capture`` is the fraction of df's squared norm lying in the readout's 2-d row space.
For a random direction in 512-d that expectation is 2/512 = 0.0039, so `capture` is
reported as a MULTIPLE of that floor.

⚠ Same standing limits as Phase 14: monitor video frames rather than rendered chamber
views, so absolute values are off-distribution; the contrasts are what carry meaning.

    cd NewbornEmbodiedTuringTest/src_isaac
    NETT_PROBE_DEVICE=0 PYTHONPATH=.:examples python examples/probe_policy_readout.py
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parent / "probe_frozen_features.py"
_spec = importlib.util.spec_from_file_location("_frozen_probe", _SRC)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)          # also forces NETT_AMP=off, before any encoder use

from nett_skrl.analysis.unity_parity import isaac_agents, _betainc  # noqa: E402


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return float("nan"), float("nan")
    r = sxy / math.sqrt(sxx * syy)
    if abs(r) >= 1:
        return r, 0.0
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return r, _betainc((n - 2) / 2, 0.5, (n - 2) / ((n - 2) + t * t))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _capture(basis: torch.Tensor, v: torch.Tensor) -> float:
    """Fraction of ``v``'s squared norm lying in the subspace spanned by ``basis``."""
    return float((basis.T @ v).pow(2).sum() / v.pow(2).sum().clamp_min(1e-12))


def readout_stats(ckpt: Path, feats: dict[str, tuple[torch.Tensor, np.ndarray]],
                  probe: str, rng: np.random.Generator) -> dict[str, float]:
    """What the policy does with each stimulus direction its encoder supplies.

    ``d_policy`` is the separation between the two stimulus classes in ACTION space,
    in units of the policy's total output spread -- both the exploration noise
    ``sigma`` and the stimulus-driven spread ``diag(W Sigma W^T)``. Omitting the second
    term (a first version did) inflates d' for any policy whose readout happens to be
    small, so it must stay in.

    ``capture`` is reported for EVERY probe direction and for the feature
    distribution's own top-2 principal subspace. That last one is the control the raw
    number needs: a readout can look well aligned with the binding direction simply
    because the whole feature cloud is low-rank, and the PC row shows when that is so.
    """
    policy = torch.load(ckpt, map_location="cpu", weights_only=False)["policy"]
    w = policy["mean_layer.weight"].float()               # (act, 512)
    sigma = policy["log_std"].float().exp()               # (act,)
    basis = torch.linalg.qr(w.T)[0]                       # (512, act) orthonormal
    floor = w.shape[0] / w.shape[1]                       # 2/512 for a random direction

    x, y = feats[probe]
    df = x[y == 1].mean(0) - x[y == 0].mean(0)
    centred = torch.cat([x[y == 0] - x[y == 0].mean(0), x[y == 1] - x[y == 1].mean(0)])
    spread = ((centred @ w.T).var(0) + sigma.pow(2)).sqrt()
    d_policy = float(torch.linalg.vector_norm((w @ df) / spread))

    rand = torch.from_numpy(rng.normal(size=df.shape).astype(np.float32))
    rand = rand / torch.linalg.vector_norm(rand) * torch.linalg.vector_norm(df)
    d_random = float(torch.linalg.vector_norm((w @ rand) / spread))

    out = {"d_policy": d_policy, "d_random": d_random,
           "capture_x": _capture(basis, df) / floor}
    for name, (fx, fy) in feats.items():
        out[f"cap_{name}"] = _capture(
            basis, fx[fy == 1].mean(0) - fx[fy == 0].mean(0)) / floor
    # generic-geometry control: the feature cloud's own leading directions
    pcs = torch.pca_lowrank(centred, q=w.shape[0])[2]      # (512, act)
    out["cap_topPC"] = float(sum(_capture(basis, pcs[:, i]) for i in range(pcs.shape[1]))
                             / pcs.shape[1]) / floor
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", type=int, default=56)
    ap.add_argument("--scale", type=float, default=0.25)
    ap.add_argument("--probe", default="binding", choices=list(P.STIMULI))
    args = ap.parse_args()

    device = torch.device(f"cuda:{os.environ.get('NETT_PROBE_DEVICE', '0')}"
                          if torch.cuda.is_available() else "cpu")
    clips = sorted({c for pair in P.STIMULI.values() for cl in pair for c in cl})
    cache = {c: P.load_clip(c) for c in clips}
    stim = {}
    for name in P.STIMULI:
        tr_x, tr_y, te_x, te_y = P.build_probe_set(cache, name)
        stim[name] = (np.concatenate([P.rescale(tr_x, args.scale, seed=1),
                                      P.rescale(te_x, args.scale, seed=2)]),
                      np.concatenate([tr_y, te_y]))
    print(f"device={device}  probe={args.probe}  scale={args.scale}  "
          f"n_stimuli={len(stim[args.probe][1])}")

    print(f"\n{'arm':<15}{'n':>4}{'d_policy':>19}{'d_random':>12}{'behav bind':>12}")
    rows = []
    for arm, (root, label) in P.ARMS.items():
        ckpts = P.agent_checkpoints(root, args.agents)
        beh = []
        for d in sorted(Path(root).glob("*/Object1/logs/test_Object1_*.csv")):
            beh.extend(isaac_agents(d))
        stats = []
        rng = np.random.default_rng(0)
        for _, ckpt in ckpts:
            enc = P.load_encoder(label, ckpt, device)
            feats = {k: (P.features(enc, imgs, device), y) for k, (imgs, y) in stim.items()}
            stats.append(readout_stats(ckpt, feats, args.probe, rng))
            del enc
        n = min(len(stats), len(beh))
        bind = [b["binding"]["score"] for b in beh[:n]]
        col = lambda k: [s[k] for s in stats[:n]]  # noqa: E731
        print(f"{arm:<15}{n:>4}{mean(col('d_policy')):>13.3f}±{sd(col('d_policy')):<5.3f}"
              f"{mean(col('d_random')):>12.3f}{mean(bind):>12.4f}")
        rows.append((arm, stats[:n], bind))

    print("\n== capture: fraction of each direction inside the policy's 2-d readout, "
          "as a MULTIPLE of the 2/512 chance floor ==")
    print("`topPC` is the geometry control -- if it is as high as the others, the readout "
          "is\naligned with the feature cloud in general, not with any stimulus direction.")
    keys = [f"cap_{k}" for k in P.STIMULI] + ["cap_topPC"]
    print(f"{'arm':<15}" + "".join(f"{k.removeprefix('cap_'):>13}" for k in keys))
    for arm, stats, _ in rows:
        print(f"{arm:<15}" + "".join(f"{mean([s[k] for s in stats]):>13.2f}" for k in keys))

    print("\n== is the policy's d' above its own random-direction floor? (paired) ==")
    print(f"{'arm':<15}{'d_policy - d_random':>22}{'agents above floor':>21}")
    for arm, stats, _ in rows:
        diff = [s["d_policy"] - s["d_random"] for s in stats]
        print(f"{arm:<15}{mean(diff):>+16.3f}±{sd(diff):<5.3f}"
              f"{sum(d > 0 for d in diff):>14}/{len(diff)}")

    print("\n== does an agent's readout d' predict its behavioural binding? ==")
    print(f"{'arm':<15}{'r(d_policy)':>22}{'r(capture)':>22}")
    for arm, stats, bind in rows:
        ra, pa = pearson([s["d_policy"] for s in stats], bind)
        rb, pb = pearson([s["capture_x"] for s in stats], bind)
        print(f"{arm:<15}{f'{ra:+.3f} (p={pa:.3f})':>22}{f'{rb:+.3f} (p={pb:.3f})':>22}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
