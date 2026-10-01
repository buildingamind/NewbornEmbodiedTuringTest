"""Initialization helpers shared by NETT skrl models."""

from __future__ import annotations

import os

import torch.nn as nn

from ..model_cfg import ModelCfg


def conv3d_ortho_init_enabled() -> bool:
    """NETT_CONV3D_ORTHO_INIT: '0' (default) = Conv3d keeps PyTorch's default init; '1' = orthogonal.

    orthogonal_init has matched only (Conv2d, Linear) since the first Isaac commit, so every Conv3d
    stem (compact_3dcnn ``conv3d``, guess_what_moves ``moves_3d``, dual_stream) kept
    kaiming_uniform(a=sqrt(5)) and a uniform bias: for the 3DCNN stem (fan_in 54) that is weight std
    .079 against ~.192 under orthogonal at gain sqrt(2), and a nonzero bias. Opt-in so that every
    existing arm rebuilds byte-identically when the variable is unset.
    """
    v = os.environ.get("NETT_CONV3D_ORTHO_INIT", "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"NETT_CONV3D_ORTHO_INIT={v!r}: expected '0' (default) or '1'")
    return v == "1"


def orthogonal_init(module: nn.Module, gain: float) -> None:
    types = (nn.Conv2d, nn.Linear)
    if conv3d_ortho_init_enabled():
        types = types + (nn.Conv3d,)
        if isinstance(module, nn.Conv3d):
            # stdout, not a logger: the spawn child's logger is not bridged to the driver log,
            # and this line is how a close-time check proves the knob reached the arm.
            print(f"[NETT init] Conv3d orthogonal x{gain:.4f} {tuple(module.weight.shape)}", flush=True)
    if isinstance(module, types):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def init_output(layer: nn.Linear, cfg: ModelCfg, gain: float | None = None) -> None:
    # SB3 uses gain=0.01 for the POLICY head but gain=1.0 for the VALUE head.
    # SKRL applied cfg.output_gain (0.01) to BOTH, which makes the critic output
    # near-zero at init; with correctly-scaled [0,1] CNN input (features ~255x
    # larger than the old crushed input) the mis-scaled critic produces garbage
    # advantages and training collapses. Pass gain=1.0 for the value head.
    if cfg.orthogonal_init:
        nn.init.orthogonal_(layer.weight, gain=cfg.output_gain if gain is None else gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
