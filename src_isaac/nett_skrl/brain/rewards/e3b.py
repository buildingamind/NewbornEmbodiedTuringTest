"""Exploration via Elliptical Episodic Bonuses."""

from __future__ import annotations

import torch

from .utils import BaseIntrinsicReward


class E3B(BaseIntrinsicReward):
    """Exploration via Elliptical Episodic Bonuses over projected features."""

    def __init__(self, *args, ridge: float = 0.1, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ridge = float(ridge)
        self._inv_cov: torch.Tensor | None = None

    def compute(self, **samples) -> torch.Tensor:
        features = self._features(samples["next_observations"])
        inv_cov = self._ensure_cov(features.dtype)
        value = torch.sqrt((features @ inv_cov * features).sum(dim=-1, keepdim=True).clamp_min(0.0))
        return self._shape_like_reward(value * self._bonus_scale(), samples["rewards"])

    def update(self, samples: dict[str, torch.Tensor] | None = None) -> None:
        samples = self._samples(samples)
        features = self._features(samples["next_observations"]).detach()
        inv_cov = self._ensure_cov(features.dtype)
        for z in features:
            z = z.view(-1, 1)
            denom = 1.0 + (z.T @ inv_cov @ z).squeeze()
            inv_cov = inv_cov - (inv_cov @ z @ z.T @ inv_cov) / denom
        self._inv_cov = inv_cov
        super().update(samples)

    def _ensure_cov(self, dtype: torch.dtype) -> torch.Tensor:
        if self._inv_cov is None or self._inv_cov.dtype != dtype:
            self._inv_cov = torch.eye(self.latent_dim, device=self.device, dtype=dtype) / self.ridge
        return self._inv_cov
