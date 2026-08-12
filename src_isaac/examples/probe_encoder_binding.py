"""OFF-POLICY encoder diagnostic: can an encoder REPRESENT the binding discrimination?

WHY THIS EXISTS. Three QK-ablation arms (~24 GPU-hours) each failed their positive control,
so none of them said anything about the QK hypothesis -- see SIDE_LOCK_INVESTIGATION.md
Phase 10. The question that blocks a fourth arm is not "does the ablated ViT bind" but the
cheaper, prior one: **can the ablated encoder fit the discrimination AT ALL, off-policy?**
If it cannot fit a supervised version of the task in a minute on one GPU, the encoder is
broken and no amount of RL will show anything.

THE TASK. Two "monitors" side by side on a dark background, each showing a coloured shape.
One side carries the IMPRINT (a fixed colour+shape pair); the other carries a FOIL that
shares exactly one feature -- same shape, wrong colour, or same colour, wrong shape. The
label is which side holds the imprint. That is the binding design in miniature: neither
colour alone nor shape alone is sufficient, only the CONJUNCTION at a location.

Two control tasks bracket it, so a failure can be localised:
  brightness  one side is brighter        -- trivial; any working encoder must pass
  colour      foil differs only in colour -- a single feature, no conjunction needed
  binding     the conjunction task above

⚠ WHAT THIS DOES AND DOES NOT SHOW. Passing means the encoder CAN represent the
discrimination under direct supervision; it does not mean PPO would find it from a closeness
reward. Failing is the informative direction: an encoder that cannot fit `brightness` cannot
be blamed on the RL setup. Synthetic stimuli also differ from the rendered chamber (no
fisheye, no lighting, no motion), so this is a capacity screen, not a replication.

Usage:
  NETT_PROBE_DEVICE=0 python examples/probe_encoder_binding.py
"""
from __future__ import annotations

import os
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from nett_skrl.brain.encoders.compact_vit import CompactViT
from nett_skrl.brain.encoders.nature_cnn import NatureCNN

RES = 128
FEATURES_DIM = 512
# Matched to examples/campaign_train.py so the probe tests the encoders that actually ran.
VIT = dict(features_dim=FEATURES_DIM, patch_size=16, embed_dim=144, depth=3, num_heads=4,
           mlp_ratio=2.0, pool="cls", stem="linear")
# ★ `pool="spatial"` IS THE CANDIDATE FIX for the small-object blindness the sweep below
# found. With `pool="cls"` the readout is ONE token, and under uniform/static mixing that
# token is a global average -- a one-patch object is diluted ~64x and the ablations sit at
# chance. Spatial pooling reduces each patch token, folds them back to the patch grid and
# pools to a small HxW map, so per-patch evidence survives the readout. It already exists in
# CompactViT (added for the ViViT collapse) and is applied to the qk arm too, or the
# comparison would confound pooling with routing.
SPATIAL = {"pool": "spatial", "spatial_grid": 4, "spatial_reduce_dim": 16}
ENCODERS = {
    "CNN":         (NatureCNN,  dict(features_dim=FEATURES_DIM, conv_dim=75)),
    "ViT (qk)":    (CompactViT, dict(VIT)),
    "ViT-NoQK":    (CompactViT, {**VIT, "embed_dim": 164, "attn_mode": "uniform"}),
    "ViT-Mixer":   (CompactViT, {**VIT, "embed_dim": 160, "attn_mode": "mixer"}),
    "ViT qk sp":   (CompactViT, {**VIT, **SPATIAL}),
    "ViT-NoQK sp": (CompactViT, {**VIT, **SPATIAL, "embed_dim": 164, "attn_mode": "uniform"}),
    "ViT-Mixer sp":(CompactViT, {**VIT, **SPATIAL, "embed_dim": 160, "attn_mode": "mixer"}),
}

COLOURS = {"red": (220, 40, 40), "blue": (40, 80, 220)}
IMPRINT = ("red", "square")          # the fixed imprint: a RED SQUARE


def _draw(img: np.ndarray, cx: int, cy: int, colour: str, shape: str, size: int = 20) -> None:
    """Paint one shape into the image, centred at (cx, cy)."""
    rgb = np.array(COLOURS[colour], dtype=np.uint8)
    ys, xs = np.mgrid[0:RES, 0:RES]
    if shape == "square":
        mask = (np.abs(xs - cx) <= size) & (np.abs(ys - cy) <= size)
    elif shape == "triangle":
        # Upward triangle: half-width shrinks linearly from `size` at the base to 0 at the apex.
        dy = cy + size - ys                        # 0 at apex, 2*size at base
        mask = (dy >= 0) & (dy <= 2 * size) & (np.abs(xs - cx) <= dy / 2)
    else:                                          # "disc"
        mask = ((xs - cx) ** 2 + (ys - cy) ** 2) <= size ** 2
    img[mask] = rgb


def make_batch(n: int, task: str, rng: np.random.Generator, size: int = 20,
               noise: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(images uint8 HWC, labels) -- label 1 means the imprint/brighter side is the RIGHT one.

    `size` is the shape half-width in pixels: the DIFFICULTY knob, standing in for viewing
    distance. `noise` adds uniform pixel noise. At size=20 every encoder saturates at 1.000,
    which measures nothing -- the sweep in main() shrinks the stimulus until arms separate.
    """
    imgs = np.zeros((n, RES, RES, 3), dtype=np.uint8)
    labels = rng.integers(0, 2, size=n)
    for i in range(n):
        right = bool(labels[i])
        # jitter the monitor centres so the answer cannot be read off a fixed pixel
        lx, rx = 32 + rng.integers(-4, 5), 96 + rng.integers(-4, 5)
        cy = 64 + rng.integers(-4, 5)
        if task == "brightness":
            small, big = max(2, size // 2), size
            _draw(imgs[i], lx, cy, "red", "square", size=big if not right else small)
            _draw(imgs[i], rx, cy, "red", "square", size=big if right else small)
            continue
        ic, ish = IMPRINT
        if task == "colour":
            foil = ("blue", ish)                       # same shape, wrong colour
        else:                                          # "binding"
            foil = (ic, "triangle") if rng.random() < 0.5 else ("blue", ish)
        tgt_x, foil_x = (rx, lx) if right else (lx, rx)
        _draw(imgs[i], tgt_x, cy, ic, ish, size=size)
        _draw(imgs[i], foil_x, cy, *foil, size=size)
    if noise:
        imgs = np.clip(imgs.astype(np.int16)
                       + rng.integers(-noise, noise + 1, imgs.shape), 0, 255).astype(np.uint8)
    return torch.from_numpy(imgs), torch.from_numpy(labels).long()


def fit(name: str, task: str, device: torch.device, steps: int = 400, batch: int = 64,
        seed: int = 0, size: int = 20, noise: int = 0) -> float:
    """Train encoder+linear head on the task; return held-out accuracy."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cls, cfg = ENCODERS[name]
    space = gym.spaces.Box(low=0, high=255, shape=(RES, RES, 3), dtype=np.uint8)
    enc = cls(space, **cfg).to(device)
    head = nn.Linear(FEATURES_DIM, 2).to(device)
    opt = torch.optim.Adam([*enc.parameters(), *head.parameters()], lr=3e-4)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        x, y = make_batch(batch, task, rng, size=size, noise=noise)
        logits = head(enc(x.to(device)))
        opt.zero_grad(); lossf(logits, y.to(device)).backward(); opt.step()
    enc.eval(); head.eval()
    correct = total = 0
    eval_rng = np.random.default_rng(seed + 10_000)     # held-out draws
    with torch.no_grad():
        for _ in range(16):
            x, y = make_batch(batch, task, eval_rng, size=size, noise=noise)
            correct += int((head(enc(x.to(device))).argmax(1).cpu() == y).sum()); total += batch
    return correct / total


def main() -> int:
    dev = torch.device(f"cuda:{os.environ.get('NETT_PROBE_DEVICE', '0')}"
                       if torch.cuda.is_available() else "cpu")
    tasks = ["brightness", "colour", "binding"]
    print(f"device={dev}  res={RES}  imprint={IMPRINT}  (chance = 0.500)\n")
    print(f"{'encoder':<14}" + "".join(f"{t:>13}" for t in tasks))
    for name in ENCODERS:
        row, t0 = [], time.time()
        for task in tasks:
            row.append(fit(name, task, dev))
        print(f"{name:<14}" + "".join(f"{a:>13.3f}" for a in row) + f"   ({time.time()-t0:.0f}s)")

    # ★ THE EASY REGIME SATURATES AT 1.000 FOR EVERY ENCODER, which discriminates nothing.
    # Shrink the stimulus (a stand-in for viewing distance) and add pixel noise until the
    # arms separate -- that is where a capacity difference, if any, becomes visible.
    print("\nbinding task, by stimulus size (half-width px) with noise:")
    regimes = [(20, 0), (8, 10), (4, 20), (2, 30)]
    print(f"{'encoder':<14}" + "".join(f"{f'sz{s}/n{n}':>13}" for s, n in regimes))
    for name in ENCODERS:
        row, t0 = [], time.time()
        for sz, nz in regimes:
            row.append(fit(name, "binding", dev, size=sz, noise=nz))
        print(f"{name:<14}" + "".join(f"{a:>13.3f}" for a in row) + f"   ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
