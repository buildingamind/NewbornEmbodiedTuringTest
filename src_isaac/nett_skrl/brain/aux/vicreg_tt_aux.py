"""Temporal-pair VICReg auxiliary loss on the shared encoder, never a reward.

Reuse the incumbent objective, expander, coefficients, and augmentations; only
the view construction changes to aug(t) and aug(t+k). Draw contiguous windows
from one rollout stream just as cltt_ref does. The default single offset keeps
the objective strictly two-view; explicitly configured extra offsets are summed.

WHY THE DEFAULT OFFSET IS 8 AND NOT cltt_ref's 2. This loss is only different
from ``vicreg`` to the extent that t and t+k actually show different viewpoints,
so the offset was measured rather than inherited. From agent.angle in an existing
parsing test log (ViViT+VICReg fork-1 off0, 448 episodes, within-episode, wrapped
to +-180): median |d angle| is 4.0 deg at lag 2 and 6.0 deg at lag 8, mean 8.5 vs
17.6. At the live eye's 2.34 deg/px field average (300 deg over 128 px, equisolid
so this is an average) lag 2 is UNDER TWO PIXELS of pan -- far inside _augment's
own crop, which reaches 41% linear zoom at scale_min=0.5. At offset 2 the temporal
signal would be invisible under the augmentation and a null would mean nothing.
Rotation saturates by lag ~16 (mean 8.5 -> 17.6 -> 19.7 -> 20.7 at lags 2, 8, 16,
32) and translation is negligible at every lag (0.46 at lag 8 against a chamber
half-width of 33.15, i.e. 1.4%), so heading decorrelation is the only binding
limit. 8 takes 84% of the available rotation while staying short of saturation;
16 is the first retry if 8 returns a null. Offsets must stay multiples of the
realised stack depth T -- see the guard in compute(). Full derivation:
notes/researcher/vicreg-tt-plus.md section 6a in the fleet workspace.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import logger

from .simclr_aux import _augment
from .vicreg_aux import VICRegExpander, _off_diagonal, vicreg_loss


def vicreg_terms(
    z1: torch.Tensor, z2: torch.Tensor, eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unweighted (invariance, variance, covariance), matching vicreg_loss."""
    inv = F.mse_loss(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0) + eps)
    std2 = torch.sqrt(z2.var(dim=0) + eps)
    var = torch.mean(F.relu(1.0 - std1)) + torch.mean(F.relu(1.0 - std2))
    B, D = z1.shape
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov1 = (z1c.T @ z1c) / (B - 1)
    cov2 = (z2c.T @ z2c) / (B - 1)
    cov = _off_diagonal(cov1).pow(2).sum() / D + _off_diagonal(cov2).pow(2).sum() / D
    return inv, var, cov


class VICRegTemporalAuxLoss(nn.Module):
    """VICReg over independently augmented temporal windows from one stream."""

    needs_memory = True

    def __init__(
        self,
        encoder: nn.Module,
        *,
        crop_scale_min: float = 0.5,
        jitter: float = 0.2,
        max_samples: int = 48,
    ) -> None:
        super().__init__()
        self.head = VICRegExpander(int(encoder.features_dim), 512, 512)
        self.head.to(next(encoder.parameters()).device)
        offsets = os.environ.get("NETT_AUX_VICREG_TT_OFFSETS", "8")
        try:
            self.offsets = tuple(int(x) for x in offsets.split(","))
            if not self.offsets or any(k <= 0 for k in self.offsets):
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                "NETT_AUX_VICREG_TT_OFFSETS must be a comma-separated list of "
                f"at least one positive integer; got {offsets!r}."
            ) from exc
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", max_samples))
        # Preserve the incumbent's owner-set balance (2026-06-15).
        self.inv = float(os.environ.get("NETT_VICREG_INV", "3"))
        self.var = float(os.environ.get("NETT_VICREG_VAR", "30"))
        self.cov = float(os.environ.get("NETT_VICREG_COV", "10"))
        self.crop_scale_min = float(crop_scale_min)
        self.jitter = float(jitter)
        self._memory = None
        self.num_frames: int | None = None
        # GATE A diagnostic, OFF unless asked for. Computes the invariance term a
        # SECOND time against another augmentation of the anchor -- i.e. exactly the
        # incumbent `vicreg` construction -- on the same real batch. The ratio
        # temporal/control answers "does the pairing reach the loss at all?", which is
        # logically prior to whether its coefficient is large enough. Costs one extra
        # encoder forward per update, so it is opt-in and a real arm pays nothing.
        from ...nett import _env_flag  # local import: avoids a package-level cycle
        self.diag = _env_flag("NETT_AUX_VICREG_TT_DIAG")
        self.last_inv_temporal: float | None = None
        self.last_inv_control: float | None = None
        # Sum unweighted terms across offsets, matching the returned total loss.
        self.last_terms: tuple[float, float, float] | None = None

    def attach_memory(self, memory) -> None:
        self._memory = memory

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Ignore the PPO minibatch; draw temporal windows from attached memory."""
        if self._memory is None:
            raise RuntimeError(
                "VICRegTemporalAuxLoss draws its own temporal windows; "
                "call attach_memory(memory) before compute()."
            )
        memory = self._memory
        # Slice BEFORE transferring: HybridDeviceMemory.get_tensor_by_name would
        # copy the WHOLE ~2 GB observation buffer to the GPU on every call.
        raw = memory.tensors["observations"]
        t_max = memory.memory_size if memory.filled else memory.memory_index
        avail = t_max - max(self.offsets)
        batch = min(self.max_samples, avail)
        if batch < 2:
            raise ValueError(
                f"VICRegTemporalAuxLoss needs B_eff >= 2, got {batch} "
                f"(t_max={t_max}, offsets={self.offsets}, NETT_AUX_BATCH={self.max_samples}). "
                "VICReg variance/covariance terms are degenerate at B=1: "
                "single-row variance is zero or NaN and covariance divides by B-1."
            )
        env = torch.randint(raw.shape[1], ()).item()
        t0 = torch.randint(avail - batch + 1, ()).item() if avail >= self.max_samples else 0
        device = next(encoder.parameters()).device
        with torch.no_grad():
            views = [
                encoder._prepare_image(raw[t0 + k : t0 + batch + k, env].to(device))
                for k in (0, *self.offsets)
            ]
        if self.num_frames is None:
            self.num_frames = views[0].shape[1] // 3
            logger.info(
                "VICRegTemporalAuxLoss: offsets=%s, stack depth T=%s",
                self.offsets, self.num_frames,
            )
        if any(k % self.num_frames for k in self.offsets):
            raise ValueError(
                f"VICRegTemporalAuxLoss: realised stack depth T={self.num_frames}, "
                f"offsets={self.offsets}. Offsets not aligned to the stack depth "
                "can make positive views share a literally identical frame, "
                "allowing a shared-frame matching shortcut. Choose "
                f"NETT_AUX_VICREG_TT_OFFSETS that are multiples of T={self.num_frames}."
            )

        prepared = views
        views = [
            _augment(view, scale_min=self.crop_scale_min, jitter=self.jitter)
            for view in prepared
        ]
        z_anchor = self.head(encoder.encode_prepared(views[0]))  # backbone grad ON
        losses = []
        terms = []
        for view in views[1:]:
            z_view = self.head(encoder.encode_prepared(view))
            losses.append(vicreg_loss(z_anchor, z_view, self.inv, self.var, self.cov))
            with torch.no_grad():
                terms.append(torch.stack(vicreg_terms(z_anchor, z_view)))
        self.last_terms = tuple(torch.stack(terms).sum(dim=0).tolist())
        if self.diag:
            with torch.no_grad():
                # Control view: a second augmentation of the ANCHOR frames. Identical
                # pipeline, identical batch, differing ONLY in that no time has passed.
                z_ctrl = self.head(encoder.encode_prepared(
                    _augment(prepared[0], scale_min=self.crop_scale_min, jitter=self.jitter)
                ))
                z_temporal = self.head(encoder.encode_prepared(views[1]))
                self.last_inv_temporal = float(vicreg_terms(z_anchor, z_temporal)[0])
                self.last_inv_control = float(vicreg_terms(z_anchor, z_ctrl)[0])
        return sum(losses)
