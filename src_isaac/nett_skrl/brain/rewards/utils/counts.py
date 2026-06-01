"""Pseudo-count helpers shared by novelty-based rewards."""

from __future__ import annotations

from collections import defaultdict

import torch


class FeatureCountMixin:
    """Hash projected features into approximate state-visit counts."""

    rounding: int
    counts: defaultdict[tuple, int]
    device: torch.device

    def _pseudo_count_bonus(self, features: torch.Tensor) -> torch.Tensor:
        keys = self._keys(features)
        values = [1.0 / (self.counts[key] + 1) ** 0.5 for key in keys]
        return torch.tensor(values, device=self.device, dtype=features.dtype).view(-1, 1)

    def _increment_counts(self, features: torch.Tensor) -> None:
        for key in self._keys(features):
            self.counts[key] += 1

    def _keys(self, features: torch.Tensor) -> list[tuple]:
        rounded = torch.round(features.detach().cpu() * (10 ** self.rounding)).to(torch.int32)
        return [tuple(row.tolist()) for row in rounded]
