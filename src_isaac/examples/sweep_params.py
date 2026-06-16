"""Sweep encoder-internal knobs (features_dim FIXED at 512) to hit ~700K total
params at res=128. Keeps the policy head (Linear(512->act)) identical across all
models; only encoder internals (conv_dim / embed_dim) change."""
from __future__ import annotations

import os
import gymnasium as gym
import numpy as np

from nett_skrl.brain.registry import encoder_mapping

RES = int(os.environ.get("NETT_RES", "128"))
ACT = 2
FD = 512  # fixed features_dim -> identical policy head everywhere


def total(encoder, cfg, framestack):
    cfg = {k: v for k, v in cfg.items() if k != "trainable"}
    ch = 3 * (2 if framestack else 1)
    enc = encoder_mapping[encoder](gym.spaces.Box(0, 255, (RES, RES, ch), np.uint8), **cfg)
    ep = sum(p.numel() for p in enc.parameters())
    return ep + (FD * ACT + ACT) + ACT + (FD + 1)


CONV = {  # encoder, framestack, extra cfg
    "nature_cnn":       (False, {}),
    "compact_3dcnn":    (True, {"num_frames": 2}),
    "simclr_cltt":      (False, {}),
    "guess_what_moves": (True, {"num_frames": 2}),
}

if __name__ == "__main__":
    print(f"res={RES} features_dim={FD} (target ~700K)\n")
    for enc, (fs, extra) in CONV.items():
        print(f"== {enc} sweep conv_dim ==")
        for cd in (64, 72, 80, 88, 96, 104, 112, 128):
            cfg = {"features_dim": FD, "conv_dim": cd, **extra}
            try:
                print(f"  conv_dim={cd:<4} -> {total(enc, cfg, fs):,}")
            except Exception as e:
                print(f"  conv_dim={cd:<4} -> ERR {e}")
