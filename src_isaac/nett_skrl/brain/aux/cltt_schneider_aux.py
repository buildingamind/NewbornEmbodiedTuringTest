"""Schneider et al. CLTT (trieschlab/CLTT, release), as ``cltt_schneider``.

Faithful defaults: B=256, temperature=1.0, projector Linear(in,256) ->
BatchNorm1d(256) -> ReLU(inplace=True) -> Linear(256,128), including both
linear biases. The head does NOT normalize. The objective uses cosine
similarity and symmetric NT-Xent over all 2(B-1) negatives. Both views pass
through encoder and projector together, as in main/train.py (including BN).
No synthetic augmentation is applied.

sample_contrast chooses uniformly from [t-tau_minus, t+tau_plus], excluding t.
Defaults tau_minus=0, tau_plus=1 give exactly the NEXT FRAME. Configure via
NETT_AUX_CLTT_SCHNEIDER_TAU_MINUS / _TAU_PLUS, temperature via
NETT_AUX_CLTT_SCHNEIDER_TEMP, and batch cap via NETT_AUX_BATCH.

Host departures, deliberate and required by the rollout integration:
* The encoder is supplied. We do not install the reference's ResNet18 ->
  BN(256) -> ReLU -> Linear(256,128,bias=False) backbone. The projector input
  width is encoder.features_dim; its hidden/output widths remain 256/128.
  Host preprocessing/resolution is retained. For stacked input, only current
  RGB is repeated across the required slots (static input to motion encoders).
* Reference view_sampling (randomwalk/uniform/window, default randomwalk)
  builds a procedural viewing sequence; it is NOT the positive-offset law.
  Here the environment/policy supplies that sequence. We cannot regenerate or
  reorder it. The existing episode_window_* helpers draw one safe contiguous
  slab, rather than the reference's shuffled offline anchor minibatches.
* We deliberately do NOT reproduce circular_sampling, which wraps buffer
  edges and can pair across episode resets in a rollout. Full temporal windows
  must be episode-contiguous, including ring seams. We omit edge anchors whose
  full support does not fit, rather than wrapping or using the reference's
  non-circular self-pair fallback. Short episodes/buffers reduce B; B<2 raises.
* The host owns optimizer, LR, schedule and update count: these are auxiliary
  gradients into the shared encoder, not the reference's standalone AdamW
  training (lrate=1e-3, N_EPOCHS=100). No separate optimizer is installed.

Sources: release commit 85e05c4b8b2ce34d5d3fa25d525e72a1b43a21e8,
config.py, utils/networks.py, utils/datasets.py, utils/losses.py, main/train.py:
https://github.com/trieschlab/CLTT/tree/release
Schneider et al., Contrastive Learning Through Time, SVRHM 2021:
https://openreview.net/forum?id=HTCRs8taN8
"""

from __future__ import annotations

import math
import os

import torch
from torch import nn
from torch.nn import functional as F

from .cltt_ref_aux import episode_window_batch, draw_episode_window
from .cltt_views import current_frame_stack
from .simclr_aux import nt_xent


class CLTTSchneiderProjectionHead(nn.Module):
    """The release MLPHead; raw output, with cosine computed by the loss."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CLTTSchneiderAuxLoss(nn.Module):
    """One temporal positive per anchor, uniform over non-self window offsets."""

    needs_memory = True

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        prefix = "NETT_AUX_CLTT_SCHNEIDER_"
        try:
            self.tau_minus = int(os.environ.get(prefix + "TAU_MINUS", "0"))
            self.tau_plus = int(os.environ.get(prefix + "TAU_PLUS", "1"))
            if min(self.tau_minus, self.tau_plus) < 0 or self.tau_minus + self.tau_plus == 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError(prefix + "TAU_MINUS/TAU_PLUS must be nonnegative integers with positive sum") from exc
        self.temperature = float(os.environ.get(prefix + "TEMP", "1.0"))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError(prefix + "TEMP must be finite and positive")
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", "256"))
        if self.max_samples < 2:
            raise ValueError("NETT_AUX_BATCH must be >= 2")
        self.head = CLTTSchneiderProjectionHead(int(encoder.features_dim))
        self.head.to(next(encoder.parameters()).device)
        self._memory = None
        self.last_scalars: dict[str, float] = {}

    def attach_memory(self, memory) -> None:
        self._memory = memory

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        if self._memory is None:
            raise RuntimeError("CLTTSchneiderAuxLoss requires attach_memory(memory) before compute()")
        memory = self._memory
        width = self.tau_minus + self.tau_plus
        batch, starts = episode_window_batch(memory, (width,), self.max_samples)
        env, t0 = draw_episode_window(starts)
        # CPU indices work with CPU-backed rollout memory; transfer only selected rows.
        anchor = torch.arange(t0 + self.tau_minus, t0 + self.tau_minus + batch)
        offsets = torch.cat((torch.arange(-self.tau_minus, 0), torch.arange(1, self.tau_plus + 1)))
        positive = anchor + offsets[torch.randint(len(offsets), (batch,))]
        raw = memory.tensors["observations"]
        device = next(encoder.parameters()).device
        with torch.no_grad():
            selected = raw[torch.cat((anchor, positive)).to(raw.device), env].to(device)
            views = current_frame_stack(encoder._prepare_image(selected))
        z = self.head(encoder.encode_prepared(views))
        # cosine_similarity in the reference uses eps=1e-8; normalize here, not in head.
        z1, z2 = F.normalize(z, dim=-1, eps=1e-8).chunk(2)
        loss = nt_xent(z1, z2, self.temperature)
        t_max = memory.memory_size if memory.filled else memory.memory_index
        self.last_scalars = {"B": float(batch), "t_max": float(t_max),
                             "chance": math.log(2 * batch - 1)}
        return loss
