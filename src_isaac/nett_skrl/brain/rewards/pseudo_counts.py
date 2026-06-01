"""State novelty reward based on hashed random-projection features."""

from __future__ import annotations

from collections import defaultdict

import torch

from .utils import BaseIntrinsicReward, FeatureCountMixin


class PseudoCounts(FeatureCountMixin, BaseIntrinsicReward):
    """State novelty reward based on hashed random-projection features."""

    def __init__(self, *args, rounding: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rounding = int(rounding)
        self.counts: defaultdict[tuple, int] = defaultdict(int)

    def compute(self, **samples) -> torch.Tensor:
        features = self._features(samples["next_observations"])
        value = self._pseudo_count_bonus(features)
        return self._shape_like_reward(value * self._bonus_scale(), samples["rewards"])

    def update(self, samples: dict[str, torch.Tensor] | None = None) -> None:
        samples = self._samples(samples)
        self._increment_counts(self._features(samples["next_observations"]))
        super().update(samples)
