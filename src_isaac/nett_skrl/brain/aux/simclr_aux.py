"""SimCLR-style auxiliary self-supervised loss for shaping the visual encoder.

Motivation
----------
The (weak) NETT RL reward gives the visual encoder little gradient pressure to
learn input-sensitive features. CNN encoders (3DCNN, SimCLR) cope, but the
convolution-free ViViT encoder can suffer *representation collapse*: the policy
reaches baseline reward without using vision, so the ViViT output drifts toward
a constant regardless of input.

This module adds an OPT-IN augmentation-contrastive (SimCLR / NT-Xent) loss that
back-propagates THROUGH the shared encoder backbone, independent of reward:

    1. Take the PPO minibatch's image observations (already on GPU).
    2. Prepare them to (B, C*T, H, W) normalized floats via the encoder.
    3. Build two augmented views (random-resized-crop + brightness/contrast
       jitter) as pure GPU tensor ops on the C*T channels.
    4. Encode BOTH views through the SAME encoder (gradients ON).
    5. Project (Linear->GELU->Linear, L2-normalized) and compute NT-Xent.

The encoder is shared (cfg.shared_encoder=True), so this gradient pressure
shapes the features the policy and value heads also consume.

Unlike ``CLTTReward`` (which trains only a DETACHED projector and therefore does
NOT shape the backbone), this loss explicitly flows gradients into the encoder.

Use via :class:`AuxLossPPO` (see ``brain/aux/ppo_aux.py``), wired from
``agent_factory.build_agents`` behind the ``NETT_AUX_LOSS`` env var.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRProjectionHead(nn.Module):
    """Small MLP projection head: Linear -> GELU -> Linear, L2-normalized output."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def _augment(img: torch.Tensor, *, scale_min: float = 0.5, jitter: float = 0.2) -> torch.Tensor:
    """Create one augmented view of a normalized (B, C, H, W) image on its device.

    Augmentations (all per-sample, all differentiable-safe GPU tensor ops):
      - random resized crop: sample a random box (area in [scale_min, 1.0]) per
        sample and resize back to (H, W) with bilinear ``grid_sample``.
      - brightness jitter: multiply by a per-sample factor ~U(1-jitter, 1+jitter).
      - contrast jitter: scale deviation from the per-sample mean by ~U(1-jitter,
        1+jitter).
    The C dimension here is the encoder's C*T channel stack; treating it as a
    single image is intentional (frames share the same crop/jitter).
    """
    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype

    # ---- random resized crop via affine grid_sample ----------------------
    # Sample per-image zoom (sqrt of area scale) and a random center offset so
    # the sampled box stays inside [-1, 1] normalized coordinates.
    area = torch.empty(B, device=device, dtype=dtype).uniform_(scale_min, 1.0)
    zoom = area.sqrt()  # half-extent of the crop in normalized coords, in (0, 1]
    max_off = (1.0 - zoom)
    cx = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * max_off
    cy = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * max_off

    theta = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = zoom
    theta[:, 1, 1] = zoom
    theta[:, 0, 2] = cx
    theta[:, 1, 2] = cy
    grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    # ---- brightness + contrast jitter ------------------------------------
    bright = torch.empty(B, 1, 1, 1, device=device, dtype=dtype).uniform_(1 - jitter, 1 + jitter)
    out = out * bright
    contrast = torch.empty(B, 1, 1, 1, device=device, dtype=dtype).uniform_(1 - jitter, 1 + jitter)
    mean = out.mean(dim=(2, 3), keepdim=True)
    out = (out - mean) * contrast + mean

    return out.clamp(0.0, 1.0)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric NT-Xent (SimCLR) loss between two batches of L2-normalized embeddings.

    Positive pair: (z1[i], z2[i]). Negatives: every other embedding in the 2B set.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)              # (2B, D)
    sim = torch.mm(z, z.t()) / temperature      # (2B, 2B)
    sim.fill_diagonal_(float("-inf"))           # mask self-similarity
    labels = torch.arange(2 * B, device=z.device)
    labels = (labels + B) % (2 * B)             # positive index for each row
    return F.cross_entropy(sim, labels)


class SimCLRAuxLoss(nn.Module):
    """Augmentation-contrastive auxiliary loss that shapes the encoder backbone.

    Owns the projection head (its params should be added to the agent's
    optimizer). Call :meth:`compute` with the shared encoder and a raw image
    observation minibatch; returns a scalar contrastive loss whose gradient
    flows through the encoder backbone (and the projection head).

    Args:
        encoder: the shared encoder (used here only to read ``features_dim`` and
            to allocate the projection head on the right device).
        proj_hidden: hidden width of the projection head.
        proj_dim: output (embedding) dimension of the projection head.
        temperature: NT-Xent softmax temperature (~0.2).
        crop_scale_min: minimum area fraction for random-resized-crop.
        jitter: half-range of brightness/contrast jitter.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        proj_hidden: int = 256,
        proj_dim: int = 128,
        temperature: float = 0.2,
        crop_scale_min: float = 0.5,
        jitter: float = 0.2,
        max_samples: int = 96,
    ) -> None:
        super().__init__()
        self.head = SimCLRProjectionHead(int(encoder.features_dim), proj_hidden, proj_dim)
        device = next(encoder.parameters()).device
        self.head.to(device)
        self.temperature = float(temperature)
        self.crop_scale_min = float(crop_scale_min)
        self.jitter = float(jitter)
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", max_samples))

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Return the NT-Xent contrastive loss for a raw image minibatch.

        ``observations`` is the PPO ``sampled_observations`` minibatch (HWC/CHW
        or flattened image). Gradients flow through ``encoder`` and the head.
        """
        # Prepare once (no grad needed for the deterministic permute/normalize),
        # then augment + re-encode WITH gradients through the backbone.
        with torch.no_grad():
            prepared = encoder._prepare_image(observations)  # (B, C*T, H, W) float [0,1]
            # Subsample: encoding two augmented views of the FULL ~500-sample PPO
            # minibatch doubles update-time activation memory and OOMs the ViViT
            # (update already peaks ~21 GB). NT-Xent needs only a modest batch.
            if prepared.shape[0] > self.max_samples:
                idx = torch.randperm(prepared.shape[0], device=prepared.device)[:self.max_samples]
                prepared = prepared[idx]

        v1 = _augment(prepared, scale_min=self.crop_scale_min, jitter=self.jitter)
        v2 = _augment(prepared, scale_min=self.crop_scale_min, jitter=self.jitter)

        f1 = encoder.encode_prepared(v1)   # (B, features_dim), grad ON
        f2 = encoder.encode_prepared(v2)
        z1 = self.head(f1)
        z2 = self.head(f2)
        return nt_xent(z1, z2, self.temperature)
