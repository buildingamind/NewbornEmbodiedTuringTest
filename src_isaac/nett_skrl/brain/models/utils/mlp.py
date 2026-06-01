"""MLP building blocks shared by NETT skrl models."""

from __future__ import annotations

import torch.nn as nn


def mlp_trunk(in_dim: int, hidden: list[int], activation: str) -> tuple[nn.Sequential, int]:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    if activation not in activations:
        raise ValueError(f"activation must be one of {sorted(activations)}; got {activation!r}")
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(in_dim, int(h)), activations[activation]()]
        in_dim = int(h)
    return nn.Sequential(*layers), in_dim
