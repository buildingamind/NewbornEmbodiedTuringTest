"""Rewarding Impact-Driven Exploration."""

from __future__ import annotations

from collections import defaultdict

import torch

from .icm import ICM
from .utils import FeatureCountMixin


class RIDE(FeatureCountMixin, ICM):
    """Rewarding Impact-Driven Exploration: feature change scaled by novelty."""

    def __init__(self, *args, rounding: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rounding = int(rounding)
        self.counts: defaultdict[tuple, int] = defaultdict(int)

    def compute(self, **samples) -> torch.Tensor:
        obs = self._features(samples["observations"])
        next_obs = self._features(samples["next_observations"])
        impact = torch.linalg.vector_norm(next_obs - obs, dim=-1, keepdim=True)
        novelty = self._pseudo_count_bonus(next_obs)
        return self._shape_like_reward(impact * novelty * self._bonus_scale(), samples["rewards"])

    def update(self, samples: dict[str, torch.Tensor] | None = None) -> None:
        samples = self._samples(samples)
        self._increment_counts(self._features(samples["next_observations"]))
        super().update(samples)
