"""CLTT: Contrastive Learning with Temporal Transformations reward.

Implements a temporal contrastive auxiliary reward following the principle of
"Contrastive Learning of Structured World Models" (Kipf et al., 2020) and
CURL (Laskin et al., 2020): temporally adjacent observations (obs_t, obs_{t+1})
are treated as positive pairs while observations from other environments in the
same batch serve as negatives. The InfoNCE / NT-Xent objective drives the
encoder's representation space to be temporally consistent.

This reward adapter works in conjunction with ``SimCLRCLTT``:
  - ``SimCLRCLTT.project(obs)`` produces L2-normalised projection vectors.
  - ``CLTTReward.compute()`` uses the projector to compute temporal similarity
    and returns a small positive intrinsic reward for consistent representations.
  - ``CLTTReward.update()`` back-propagates the NT-Xent loss through the
    projector (the backbone is trained jointly by the PPO actor-critic gradient).

The reward weight should be kept small (e.g. 0.05–0.1) so the contrastive
signal acts as a regulariser rather than dominating the extrinsic reward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import BaseIntrinsicReward


class CLTTReward(BaseIntrinsicReward):
    """NT-Xent temporal contrastive reward for ``SimCLRCLTT`` encoders.

    Expects the encoder attached to the running policy to expose a
    ``project(observations)`` method (i.e. ``SimCLRCLTT``). The reward adapter
    locates the policy model in the environment's agent at first call and caches
    the reference.

    Args:
        temperature: Softmax temperature for the InfoNCE loss (default 0.1).
        proj_lr: Learning rate for the projection-head optimiser.
    """

    def __init__(
        self,
        env=None,
        *,
        temperature: float = 0.1,
        proj_lr: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.temperature = float(temperature)
        self.proj_lr = float(proj_lr)
        self._encoder = None          # set on first call via _find_encoder()
        self._optimizer = None

    # ------------------------------------------------------------------
    # Encoder discovery
    # ------------------------------------------------------------------

    def _find_encoder(self, observations: torch.Tensor):
        """Walk the environment's agent list to find the SimCLRCLTT encoder."""
        if self._encoder is not None:
            return
        env = self.env
        # Try to locate a model with a project() method
        candidates = []
        for attr in ("_agents", "agents", "_agent"):
            agent_list = getattr(env, attr, None)
            if agent_list is None:
                continue
            if not isinstance(agent_list, (list, tuple)):
                agent_list = [agent_list]
            for agent in agent_list:
                for model in getattr(agent, "models", {}).values():
                    enc = getattr(model, "encoder", None)
                    if enc is not None and hasattr(enc, "project"):
                        candidates.append(enc)
        if candidates:
            self._encoder = candidates[0]
            self._optimizer = torch.optim.Adam(
                self._encoder.projector.parameters(), lr=self.proj_lr
            )

    # ------------------------------------------------------------------
    # Reward computation (no gradient — used during env stepping)
    # ------------------------------------------------------------------

    def compute(self, **samples) -> torch.Tensor:
        self._find_encoder(samples["observations"])
        rewards = samples["rewards"]
        if self._encoder is None:
            return self._shape_like_reward(torch.zeros_like(rewards), rewards)

        with torch.no_grad():
            z_t = self._encoder.project(samples["observations"].to(self.device))
            z_tp1 = self._encoder.project(samples["next_observations"].to(self.device))
            # Cosine similarity between temporally adjacent representations
            sim = (z_t * z_tp1).sum(dim=-1, keepdim=True).clamp(-1, 1)
            # Map [-1, 1] → [0, 1]: higher similarity → larger reward bonus
            bonus = (sim + 1.0) * 0.5 * self._bonus_scale()

        return self._shape_like_reward(bonus, rewards)

    # ------------------------------------------------------------------
    # Projection-head update (NT-Xent loss on mini-batch)
    # ------------------------------------------------------------------

    def update(self, samples: dict[str, torch.Tensor] | None = None) -> None:
        samples = self._samples(samples)
        self._find_encoder(samples["observations"])
        if self._encoder is None or self._optimizer is None:
            super().update(samples)
            return

        obs = samples["observations"].to(self.device)
        next_obs = samples["next_observations"].to(self.device)

        # Only update on a random subset to match update_proportion
        B = obs.shape[0]
        if self.update_proportion < 1.0:
            mask = torch.rand(B, device=self.device) < self.update_proportion
            if not mask.any():
                super().update(samples)
                return
            obs = obs[mask]
            next_obs = next_obs[mask]

        z_t = self._encoder.project(obs)          # (M, proj_dim)
        z_tp1 = self._encoder.project(next_obs)   # (M, proj_dim)

        loss = _nt_xent(z_t, z_tp1, self.temperature)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self.metrics["loss"].append([self.global_step, float(loss.detach().cpu())])
        super().update(samples)


def _nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric NT-Xent loss (SimCLR eq. 1) between two sets of projections.

    Positive pair: (z1[i], z2[i]). Negatives: all other z1 and z2 in the batch.
    """
    B = z1.shape[0]
    # Gather all vectors: row 0..B-1 = z1, row B..2B-1 = z2
    z = torch.cat([z1, z2], dim=0)                         # (2B, D)
    sim = torch.mm(z, z.T) / temperature                   # (2B, 2B)

    # Mask out self-similarity on the diagonal
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, float("-inf"))

    # Positive indices: each row i has its positive at row (i+B) % 2B
    labels = torch.arange(2 * B, device=z.device)
    labels = (labels + B) % (2 * B)

    return F.cross_entropy(sim, labels)
