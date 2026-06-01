"""Intrinsic Curiosity Module reward."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import BaseIntrinsicReward


class ICM(BaseIntrinsicReward):
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
