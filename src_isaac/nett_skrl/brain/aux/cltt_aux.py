"""CLTT: Contrastive Learning with Temporal Transformations, as an AUXILIARY LOSS.

⛔ WHY THIS FILE EXISTS. ``campaign_train.MODELS`` has declared ``aux="cltt"`` for
``SimCLR-CLTT`` and ``ViT-CLTT`` since 2026-08-28, but ``cltt`` was NEVER REGISTERED in
``ppo_aux.AUX_LOSSES``. Launching either arm therefore raised at construction
(``ppo_aux.py:97``). That refusal is correct and is the reason no CLTT arm has ever
silently trained as vanilla PPO -- but it also means the arms were unlaunchable.
This module is the one entry the registry asked for.

WHAT CLTT IS. SimCLR builds its positive pair with AUGMENTATION (two jittered crops of the
SAME frame). CLTT builds it with TIME: the positives are two frames from the same short
clip, so the invariance the encoder is pushed toward is invariance to how the scene
TRANSFORMED -- viewpoint, pose, position -- rather than to a synthetic photometric crop.
That is the property this campaign is short of: object identity survives 30 deg of
rotation and dies at 60 deg.

⛔ THE PAIR MUST COME FROM THE CHANNEL AXIS, T-MAJOR. There is no ``next_observations`` in
the skrl PPO sample tuple. The only temporal pair available is the framestack, which
``body/observation.py`` builds with ``torch.cat`` and is therefore T-MAJOR:
    [t-1 R, t-1 G, t-1 B, t R, t G, t B]
Reading it C-major pairs two channels of ONE frame and calls them a time step -- the exact
defect that sat in ``compact_3dcnn.forward`` and could not be caught by any parameter count.
This module splits T-major explicitly and REFUSES a non-framestacked observation, matching
``eoo_aux``/``gwm_aux``, so an arm can never compare a frame with itself while logging
``aux=cltt``.

⚠⚠ ONE MODELLING CHOICE IS MINE AND IS NOT CANONICAL -- READ BEFORE TRUSTING A RESULT.
The encoder's first layer expects ``C*T`` channels, so a single 3-channel frame cannot be
pushed through it. To encode one frame as one view, this module REPLICATES that frame
across the T slots (``frame.repeat(1, T, 1, 1)``), presenting a motionless stack. For a
plain 2-D CNN over the channel stack (``simclr_cltt``, ``nature_cnn``, ``compact_vit``)
this is benign. ⛔ For a MOTION encoder (``compact_3dcnn``, ``guess_what_moves``) a
motionless stack is a degenerate input and the temporal kernel sees zero motion in BOTH
views -- do not cross CLTT with those encoders without deciding what that means first.
The alternative (two temporally offset stacks) needs a longer framestack than 2; with
``NETT_FRAMESTACK_N >= 4`` it becomes available and is the better design.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from .simclr_aux import SimCLRProjectionHead, nt_xent


class CLTTAuxLoss(nn.Module):
    """Temporal-contrastive (NT-Xent) auxiliary loss that shapes the encoder backbone.

    Positives are the FIRST and LAST frame of the framestack (maximum temporal
    separation available; for the default ``n_stack=2`` these are simply the adjacent
    pair ``(t-1, t)``). Negatives are every other frame in the 2B set, exactly as in
    SimCLR -- the contrast is what stops the encoder from collapsing to a constant.

    Owns its projection head; those parameters must be added to the agent's optimizer
    (``ppo_aux`` does this via ``add_param_group``).
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        proj_hidden: int = 256,
        proj_dim: int = 128,
        temperature: float = 0.2,
        max_samples: int = 96,
    ) -> None:
        super().__init__()
        self.head = SimCLRProjectionHead(int(encoder.features_dim), proj_hidden, proj_dim)
        self.head.to(next(encoder.parameters()).device)
        self.temperature = float(temperature)
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", max_samples))
        self.num_frames: int | None = None   # discovered on the first batch

    # ------------------------------------------------------------------ helpers
    def _split_ends(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Return (first_frame, last_frame, T) from a T-MAJOR (B, C*T, H, W) tensor.

        ⛔ T-major: frame k occupies channels [3k : 3k+3]. ``imgs[:, :3]`` is the OLDEST
        frame and ``imgs[:, -3:]`` is the CURRENT one. A C-major read would take
        ``imgs[:, 0::T]``, which spans colour rather than time.
        """
        total_c = imgs.shape[1]
        if total_c < 6 or total_c % 3:
            raise ValueError(
                f"CLTTAuxLoss needs a framestacked RGB observation (>=2 frames, channels a "
                f"multiple of 3); got {total_c} channels. Declare framestack=True on this "
                f"arm -- CLTT's positive pair IS the temporal pair, and without it the loss "
                f"would contrast a frame with itself and be identically zero."
            )
        T = total_c // 3
        return imgs[:, :3], imgs[:, -3:], T

    @staticmethod
    def _as_stack(frame: torch.Tensor, T: int) -> torch.Tensor:
        """Replicate one 3-channel frame across the T slots the encoder expects.

        ⚠ See the module docstring: this is a deliberate choice, not a canon. It keeps the
        two views dimensionally identical to what the encoder was built for.
        """
        return frame.repeat(1, T, 1, 1)

    # ------------------------------------------------------------------ interface
    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prepared = encoder._prepare_image(observations)     # (B, C*T, H, W) in [0,1]
            if prepared.shape[0] > self.max_samples:
                idx = torch.randperm(prepared.shape[0], device=prepared.device)[: self.max_samples]
                prepared = prepared[idx]

        first, last, T = self._split_ends(prepared)
        self.num_frames = T

        v1 = self._as_stack(first, T)
        v2 = self._as_stack(last, T)

        z1 = self.head(encoder.encode_prepared(v1))   # grad ON through the backbone
        z2 = self.head(encoder.encode_prepared(v2))
        return nt_xent(z1, z2, self.temperature)
