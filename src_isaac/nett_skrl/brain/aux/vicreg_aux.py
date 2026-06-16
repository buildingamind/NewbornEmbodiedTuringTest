"""VICReg auxiliary self-supervised loss for shaping the visual encoder.

VICReg (Bardes, Ponce, LeCun 2022, "Variance-Invariance-Covariance
Regularization") is a NON-contrastive SSL objective — no negative pairs. Two
augmented views of the same image are encoded and expanded, then three terms are
applied to the expander embeddings:

  * invariance  : MSE(z1, z2) — pull the two views together (the "contrastive"
    positive-pair term; the only contrastive-flavoured part of VICReg).
  * variance    : hinge keeping each embedding dim's batch std >= 1 — the direct
    ANTI-COLLAPSE term (replaces contrastive negatives). Most relevant to our
    failure mode (encoder collapsing to input-invariant features).
  * covariance  : off-diagonal covariance -> 0, decorrelating the dimensions.

Total = inv_coeff*inv + var_coeff*var + cov_coeff*cov, back-propagated through
the SHARED encoder (like the SimCLR aux). Defaults (user 2026-06-15): inv=3,
var=30, cov=10 — variance-heavy anti-collapse + decorrelation, light attraction;
tunable via NETT_VICREG_{INV,VAR,COV}. Same plug-in interface as
:class:`SimCLRAuxLoss` (``compute(encoder, observations)`` returns a scalar;
``.head`` params should be added to the optimizer). Selected via NETT_AUX_LOSS=vicreg.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .simclr_aux import _augment


class VICRegExpander(nn.Module):
    """3-layer expander MLP (Linear-BN-ReLU x2 -> Linear), standard for VICReg."""

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Return the off-diagonal elements of a square matrix (VICReg helper)."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z1, z2, inv_coeff, var_coeff, cov_coeff, eps: float = 1e-4):
    # invariance (positive-pair attraction)
    inv = F.mse_loss(z1, z2)
    # variance (anti-collapse: keep each dim's std >= 1)
    std1 = torch.sqrt(z1.var(dim=0) + eps)
    std2 = torch.sqrt(z2.var(dim=0) + eps)
    var = torch.mean(F.relu(1.0 - std1)) + torch.mean(F.relu(1.0 - std2))
    # covariance (decorrelate dims)
    B, D = z1.shape
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov1 = (z1c.T @ z1c) / (B - 1)
    cov2 = (z2c.T @ z2c) / (B - 1)
    cov = _off_diagonal(cov1).pow(2).sum() / D + _off_diagonal(cov2).pow(2).sum() / D
    return inv_coeff * inv + var_coeff * var + cov_coeff * cov


class VICRegAuxLoss(nn.Module):
    """VICReg aug-view loss that shapes the encoder backbone (see module docstring)."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        expander_hidden: int = 512,
        expander_dim: int = 512,
        crop_scale_min: float = 0.5,
        jitter: float = 0.2,
        max_samples: int = 48,
    ) -> None:
        super().__init__()
        self.head = VICRegExpander(int(encoder.features_dim), expander_hidden, expander_dim)
        self.head.to(next(encoder.parameters()).device)
        self.crop_scale_min = float(crop_scale_min)
        self.jitter = float(jitter)
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", max_samples))
        # User-set balance (2026-06-15): variance-heavy anti-collapse, moderate
        # decorrelation, light positive-pair attraction. Tunable via env as testing proceeds.
        self.inv = float(os.environ.get("NETT_VICREG_INV", "3"))
        self.var = float(os.environ.get("NETT_VICREG_VAR", "30"))
        self.cov = float(os.environ.get("NETT_VICREG_COV", "10"))

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prepared = encoder._prepare_image(observations)        # (B, C*T, H, W) float [0,1]
            if prepared.shape[0] > self.max_samples:
                idx = torch.randperm(prepared.shape[0], device=prepared.device)[: self.max_samples]
                prepared = prepared[idx]
        v1 = _augment(prepared, scale_min=self.crop_scale_min, jitter=self.jitter)
        v2 = _augment(prepared, scale_min=self.crop_scale_min, jitter=self.jitter)
        z1 = self.head(encoder.encode_prepared(v1))                # grad ON through encoder
        z2 = self.head(encoder.encode_prepared(v2))
        return vicreg_loss(z1, z2, self.inv, self.var, self.cov)
