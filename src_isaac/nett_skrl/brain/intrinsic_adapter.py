"""Adapter between registered intrinsic rewards and skrl env stepping."""

from __future__ import annotations

from typing import Any

import torch


class IntrinsicRewardAdapter:
    """Bridges registered intrinsic reward classes onto the env-step loop.

    All shipped reward classes (and the documented contract for custom ones)
    take ``(*args, **kwargs)`` in ``__init__`` and use ``compute(**samples)``
    / ``update(samples=None)`` keyword forms.
    """

    def __init__(
        self,
        reward_cls,
        *,
        env,
        device: torch.device,
        weight: float,
        update_enabled: bool,
        kwargs: dict[str, Any],
    ) -> None:
        self.weight = weight
        self.update_enabled = update_enabled
        self.reward = reward_cls(env, device=device, **kwargs)
        self._last: tuple | None = None

    def watch(self, observations, actions, rewards, terminated, truncated, next_observations):
        self._last = (observations, actions, rewards, terminated, truncated, next_observations)
        if hasattr(self.reward, "watch"):
            return self.reward.watch(observations, actions, rewards, terminated, truncated, next_observations)
        return None

    def compute(self, **kwargs):
        value = (
            self.reward.compute(**kwargs)
            if hasattr(self.reward, "compute")
            else torch.zeros_like(kwargs["rewards"])
        )
        return torch.as_tensor(value, device=kwargs["rewards"].device, dtype=kwargs["rewards"].dtype) * self.weight

    def update(self):
        if not self.update_enabled or not hasattr(self.reward, "update"):
            return None
        samples = None
        if self._last is not None:
            observations, actions, rewards, terminated, truncated, next_observations = self._last
            samples = {
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "terminated": terminated,
                "truncated": truncated,
                "next_observations": next_observations,
            }
        return self.reward.update(samples)
