"""Count encoder (+ PPO head) parameters for each campaign model at a given res.

Builds each encoder exactly as agent_factory would (encoder_cls(obs_space, **cfg))
and reports total parameters, so we can tune configs to ~700K at res=128.

Usage: NETT_RES=128 python examples/count_params.py
"""
from __future__ import annotations

import os
import gymnasium as gym
import numpy as np
import torch.nn as nn

from nett_skrl.brain.registry import encoder_mapping
from campaign_train import MODELS, VIT_CFG, VIVIT_CFG  # reuse the campaign specs

RES = int(os.environ.get("NETT_RES", "128"))
ACTION_DIM = 2  # 2D action space (turn + forward)


def obs_space(channels: int) -> gym.spaces.Box:
    # HWC uint8 camera image (framestack stacks frames along channel dim).
    return gym.spaces.Box(low=0, high=255, shape=(RES, RES, channels), dtype=np.uint8)


def count(model: str, spec: dict) -> tuple[int, int]:
    cfg = dict(spec["cfg"])
    cfg.pop("trainable", None)
    channels = 3 * (2 if spec["framestack"] else 1)
    enc_cls = encoder_mapping[spec["encoder"]]
    enc = enc_cls(obs_space(channels), **cfg)
    enc_params = sum(p.numel() for p in enc.parameters())
    fd = int(getattr(enc, "features_dim", cfg.get("features_dim", 512)))
    # PPO heads with hidden_sizes=[]: gaussian policy mean Linear(fd->act) +
    # log_std param, value Linear(fd->1). (shared encoder, so counted once.)
    head_params = (fd * ACTION_DIM + ACTION_DIM) + ACTION_DIM + (fd * 1 + 1)
    return enc_params, enc_params + head_params


if __name__ == "__main__":
    print(f"res={RES}  (target ~700K total)\n")
    print(f"{'model':16} {'encoder':16} {'enc_params':>12} {'total':>12}")
    for model, spec in MODELS.items():
        try:
            enc_p, total = count(model, spec)
            print(f"{model:16} {spec['encoder']:16} {enc_p:12,} {total:12,}")
        except Exception as exc:
            print(f"{model:16} {spec['encoder']:16} ERROR: {exc}")
