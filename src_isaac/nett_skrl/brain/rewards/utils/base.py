"""Shared building blocks for skrl-compatible intrinsic rewards."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


class BaseIntrinsicReward:
    """Common one-step intrinsic reward interface used by ``IntrinsicRewardAdapter``."""

    def __init__(
        self,
        env=None,
        *,
        device: str | torch.device = "cpu",
        beta: float = 0.2,
        kappa: float = 0.0,
        gamma: float | None = 0.99,
        latent_dim: int = 128,
        lr: float = 1e-3,
        update_proportion: float = 1.0,
        **_: Any,
    ) -> None:
        self.env = env
        self.device = torch.device(device)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.gamma = gamma
        self.latent_dim = int(latent_dim)
        self.lr = float(lr)
        self.update_proportion = float(update_proportion)
        self.global_step = 0
        self.metrics: dict[str, list] = defaultdict(list)
        self._last: dict[str, torch.Tensor] | None = None
        self._projection: torch.Tensor | None = None

    def watch(self, observations, actions, rewards, terminated, truncated, next_observations):
        self._last = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminated": terminated,
            "truncated": truncated,
            "next_observations": next_observations,
        }

    def compute(self, **samples) -> torch.Tensor:
        raise NotImplementedError

    def update(self, samples: dict[str, torch.Tensor] | None = None) -> None:
        self.global_step += 1

    def _samples(self, samples: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
        if samples is not None:
            return samples
        if self._last is None:
            raise RuntimeError("Intrinsic reward has no observed transition to use.")
        return self._last

    def _obs(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, device=self.device).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.view(x.shape[0], -1)
        if x.numel() and x.max() > 1.0:
            x = x / 255.0
        return x

    def _actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.as_tensor(actions, device=self.device).float()
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        return actions.view(actions.shape[0], -1)

    def _features(self, observations: torch.Tensor) -> torch.Tensor:
        obs = self._obs(observations)
        if self._projection is None or self._projection.shape[0] != obs.shape[-1]:
            scale = obs.shape[-1] ** -0.5
            self._projection = torch.randn(
                obs.shape[-1], self.latent_dim, device=self.device
            ) * scale
        return torch.tanh(obs @ self._projection)

    def _bonus_scale(self) -> float:
        return self.beta * ((1.0 - self.kappa) ** self.global_step)

    def _shape_like_reward(self, value: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        rewards = torch.as_tensor(rewards, device=self.device)
        return value.to(device=rewards.device, dtype=rewards.dtype).view_as(rewards)

    def _mask_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.update_proportion >= 1.0:
            return loss.mean()
        mask = torch.rand_like(loss) < self.update_proportion
        if not mask.any():
            return loss.mean() * 0.0
        return loss[mask].mean()
