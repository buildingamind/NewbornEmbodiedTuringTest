"""Multivariate-Gaussian actor model for NETT continuous-control agents.

A drop-in alternative to :class:`GaussianActor` that swaps the action
distribution from an independent per-dim Normal (``GaussianMixin``) to a joint
multivariate Normal with a **learnable correlation structure**. Intended for the
wheeled agent, whose action is a pair of wheel-velocity commands that must be
coordinated (to drive straight, both wheels move together) — exactly the
correlation a diagonal Gaussian cannot represent. This is the single
learning-side change permitted for the wheeled agent (see
``workspace/notes/10_wheeled_physics.md``). Encoder, MLP trunk, mean head,
diagonal log-std, init, and optimizer are otherwise identical to
``GaussianActor``; the only added parameters are the off-diagonal Cholesky terms
inherent to choosing a correlated distribution.

Implementation notes:

* We subclass ``MultivariateGaussianMixin`` (the skrl distribution class) for its
  bookkeeping (clip flags, ``get_entropy``/``distribution`` reading
  ``_mg_distribution``) but override ``act`` to build the scale matrix correctly.
  skrl 2.1.0's stock ``act`` does ``MultivariateNormal(mean, scale_tril=diag(
  exp(log_std)**2))`` — it passes the *variance* as the Cholesky factor, which
  makes the effective per-dim std ``exp(2*log_std)`` (not ``exp(log_std)``) and
  ignores any correlation. Our override passes a proper lower-triangular Cholesky
  factor: diagonal ``= exp(log_std)`` (so the effective std matches
  ``GaussianActor`` exactly), off-diagonal ``= tril_offdiag`` (learnable).
* ``tril_offdiag`` is initialised to zero, so at the start of training the policy
  is an exact diagonal Gaussian identical to ``GaussianActor`` — a clean A/B
  baseline — and only departs from it as it learns useful action correlation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from skrl.models.torch import MultivariateGaussianMixin, Model

from .model_cfg import ModelCfg
from .utils.backbone import FeatureBackbone
from .utils.features import features_forward
from .utils.init import init_output


class MultivariateGaussianActor(MultivariateGaussianMixin, Model, FeatureBackbone):
    """Continuous correlated-Gaussian actor for PPO/A2C/TRPO/RPO/CEM."""

    def __init__(self, *, encoder_cls, encoder_kwargs, observation_space, action_space, device, cfg: ModelCfg, shared_encoder=None):
        Model.__init__(
            self,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
        )
        MultivariateGaussianMixin.__init__(
            self,
            clip_actions=cfg.clip_actions,
        )
        last = self._build_backbone(encoder_cls, encoder_kwargs, observation_space, cfg, shared_encoder=shared_encoder)
        self.mean_layer = nn.Linear(last, self.num_actions)
        init_output(self.mean_layer, cfg)
        # Diagonal log-std (identical to GaussianActor) ...
        self.log_std = nn.Parameter(torch.full((self.num_actions,), float(cfg.initial_log_std)))
        # ... plus learnable strictly-lower-triangular Cholesky terms (init 0 =>
        # starts as a pure diagonal Gaussian). Empty when num_actions == 1.
        rows, cols = torch.tril_indices(self.num_actions, self.num_actions, offset=-1)
        self.register_buffer("_tril_rows", rows, persistent=False)
        self.register_buffer("_tril_cols", cols, persistent=False)
        self.tril_offdiag = nn.Parameter(torch.zeros(int(rows.numel())))

    def compute(self, inputs, role=""):
        mean = self.mean_layer(features_forward(self, inputs))
        # 1-D log_std (shared across the batch), as the mixin/override expect.
        return mean, {"log_std": self.log_std}

    def act(self, inputs, *, role=""):
        """Override the mixin's ``act`` to build a correct lower-triangular
        Cholesky scale (diagonal = std, learnable off-diagonal correlation).
        Mirrors the mixin's clip / log_prob / outputs bookkeeping so PPO's
        ``get_entropy`` / ``distribution`` (which read ``_mg_distribution``) work
        unchanged."""
        mean_actions, outputs = self.compute(inputs, role)
        log_std = outputs["log_std"]
        if self._mg_clip_log_std:
            log_std = torch.clamp(log_std, min=self._mg_min_log_std, max=self._mg_max_log_std)
            outputs["log_std"] = log_std
        if self._mg_clip_mean_actions:
            mean_actions = torch.clamp(mean_actions, min=self._mg_min_actions, max=self._mg_max_actions)

        scale_tril = torch.diag_embed(log_std.exp())  # [n, n], diagonal = std
        if self._tril_rows.numel():
            scale_tril = scale_tril.clone()
            scale_tril[self._tril_rows, self._tril_cols] = self.tril_offdiag
        self._mg_distribution = MultivariateNormal(mean_actions, scale_tril=scale_tril)

        actions = self._mg_distribution.rsample()
        if self._mg_clip_actions:
            actions = torch.clamp(actions, min=self._mg_min_actions, max=self._mg_max_actions)

        log_prob = self._mg_distribution.log_prob(inputs.get("taken_actions", actions))
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)
        outputs["log_prob"] = log_prob
        outputs["mean_actions"] = mean_actions
        return actions, outputs
