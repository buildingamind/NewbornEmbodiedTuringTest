"""skrl-compatible intrinsic rewards for the Isaac NETT backend.

These classes provide the legacy NETT reward names without depending on SB3
callbacks or the optional ``rllte`` package. They operate on the flattened
policy observations emitted by :class:`nett_skrl.brain.env_adapter.NettIsaacLabWrapper`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnsupportedIntrinsicReward:
    """Explicit shim for legacy intrinsic rewards not implemented natively yet."""

    reward_name = "legacy intrinsic reward"

    def __init__(self, *args, **kwargs):
        raise ImportError(
            f"{self.reward_name} is not bundled with nett_skrl. Register a "
            "skrl-compatible implementation with register_reward(name, cls)."
        )

    @classmethod
    def named(cls, name: str):
        return type(name, (cls,), {"reward_name": name})


class _BaseIntrinsicReward:
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


class ICM(_BaseIntrinsicReward):
    """Intrinsic Curiosity Module using one-step forward prediction error."""

    def __init__(self, *args, hidden_dim: int = 256, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = int(hidden_dim)
        self.forward_model: nn.Module | None = None
        self.inverse_model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def _ensure_models(self, action_dim: int) -> None:
        if self.forward_model is not None:
            return
        self.forward_model = nn.Sequential(
            nn.Linear(self.latent_dim + action_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        ).to(self.device)
        self.inverse_model = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, action_dim),
        ).to(self.device)
        params = list(self.forward_model.parameters()) + list(self.inverse_model.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def compute(self, **samples) -> torch.Tensor:
        obs = self._features(samples["observations"])
        next_obs = self._features(samples["next_observations"])
        actions = self._actions(samples["actions"])
        self._ensure_models(actions.shape[-1])
        with torch.no_grad():
            pred_next = self.forward_model(torch.cat([obs, actions], dim=-1))
            value = F.mse_loss(pred_next, next_obs, reduction="none").mean(dim=-1, keepdim=True)
        return self._shape_like_reward(value * self._bonus_scale(), samples["rewards"])

    def update(self, samples: dict[str, torch.Tensor] | None = None) -> None:
        samples = self._samples(samples)
        obs = self._features(samples["observations"]).detach()
        next_obs = self._features(samples["next_observations"]).detach()
        actions = self._actions(samples["actions"])
        self._ensure_models(actions.shape[-1])
        pred_next = self.forward_model(torch.cat([obs, actions], dim=-1))
        pred_actions = self.inverse_model(torch.cat([obs, next_obs], dim=-1))
        fm_loss = F.mse_loss(pred_next, next_obs, reduction="none").mean(dim=-1)
        im_loss = F.mse_loss(pred_actions, actions, reduction="none").mean(dim=-1)
        loss = self._mask_loss(fm_loss + im_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.metrics["loss"].append([self.global_step, float(loss.detach().cpu())])
        super().update(samples)


class RIDE(ICM):
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


class PseudoCounts(_BaseIntrinsicReward):
    """State novelty reward based on hashed random-projection features."""

    def __init__(self, *args, rounding: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rounding = int(rounding)
        self.counts: defaultdict[tuple, int] = defaultdict(int)

    def compute(self, **samples) -> torch.Tensor:
        features = self._features(samples["next_observations"])
        keys = self._keys(features)
        values = [1.0 / (self.counts[key] + 1) ** 0.5 for key in keys]
        value = torch.tensor(values, device=self.device, dtype=features.dtype).view(-1, 1)
        return self._shape_like_reward(value * self._bonus_scale(), samples["rewards"])

    def update(self, samples: dict[str, torch.Tensor] | None = None) -> None:
        samples = self._samples(samples)
        for key in self._keys(self._features(samples["next_observations"])):
            self.counts[key] += 1
        super().update(samples)

    def _keys(self, features: torch.Tensor) -> list[tuple]:
        rounded = torch.round(features.detach().cpu() * (10 ** self.rounding)).to(torch.int32)
        return [tuple(row.tolist()) for row in rounded]


class E3B(_BaseIntrinsicReward):
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


# --- Public aliases for legacy NETT reward names ---------------------------

NGU = PseudoCounts

Disagreement = UnsupportedIntrinsicReward.named("Disagreement")
Fabric = UnsupportedIntrinsicReward.named("Fabric")
RE3 = UnsupportedIntrinsicReward.named("RE3")
RND = UnsupportedIntrinsicReward.named("RND")


__all__ = [
    "Disagreement",
    "E3B",
    "Fabric",
    "ICM",
    "NGU",
    "PseudoCounts",
    "RE3",
    "RIDE",
    "RND",
    "UnsupportedIntrinsicReward",
]
