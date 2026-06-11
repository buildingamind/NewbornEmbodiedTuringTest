"""Initialization helpers shared by NETT skrl models."""

from __future__ import annotations

import torch.nn as nn

from ..model_cfg import ModelCfg


def orthogonal_init(module: nn.Module, gain: float) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
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
