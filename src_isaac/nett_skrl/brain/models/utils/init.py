"""Initialization helpers shared by NETT skrl models."""

from __future__ import annotations

import torch.nn as nn

from ..model_cfg import ModelCfg


def orthogonal_init(module: nn.Module, gain: float) -> None:
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def init_output(layer: nn.Linear, cfg: ModelCfg) -> None:
    if cfg.orthogonal_init:
        nn.init.orthogonal_(layer.weight, gain=cfg.output_gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
