"""MoTok auxiliary loss — VQ codebook + slot attention, ported from the Unity campaign.

PORTED FROM, and read at the file rather than from the report:
    scripts/gwmEventBased1/model.py:534-665   ``class MoTokNet``, ``_VQModule``, ``_SlotAttention``
    scripts/gwmEventBased1/trainParsing.py:328-400  ``GwmPPO_MoTokNet._train_ventral``  <- THE LOSS
    scripts/wrappers/seg_wrappers.py:122-215        ``GwmSegWrapper``                   <- HOW IT REACHES THE POLICY

Architecture, faithful to the reference:
  * 3-stage CNN encoder -> 32ch at H/4
  * VQ codebook K=64, dim=32, straight-through estimator
  * Slot attention: 2 slots (fg/bg), 3 iterations
  * THE SLOT ATTENTION WEIGHTS ARE THE H/4 MASKS (upsampled to HxW)
  * Loss = MSE(recon, frame) + vq_coef * commit   (trainParsing.py:371)

⛔★★★★★ READ THIS BEFORE TREATING THIS AS "THE MoTok ARM"
--------------------------------------------------------
**IN UNITY, MoTok IS NOT AN AUXILIARY LOSS.** It is a SEPARATELY-OPTIMISED network
(``_MODEL_REF``/``_OPTIM_REF``, its own AdamW) whose foreground mask is MULTIPLIED
INTO THE OBSERVATION by ``GwmSegWrapper`` (``seg_wrappers.py:213``:
``masked = obs * M_np``). The policy never shares a gradient with it.

That matters because the aux-loss interface here exists to SHAPE THE SHARED PPO
ENCODER. MoTok has its OWN encoder, so a faithful port placed behind this
interface trains a parallel network and the policy encoder receives NOTHING.
**A faithful MoTok aux loss is, with respect to the shared encoder, a no-op** --
and this campaign has shipped five silent no-ops already, so it is stated here
rather than discovered later.

Two modes, both explicit, neither the silent default:

  ``NETT_MOTOK_MODE=faithful`` (DEFAULT)
      Reference architecture, own encoder, recon+VQ. Gradients DO NOT reach the
      PPO encoder. Use ``masks()`` with an observation wrapper to reproduce the
      Unity arm. ``compute()`` still returns the real loss so the network trains.

  ``NETT_MOTOK_MODE=shared``
      Slot attention runs on a spatial grid EXPANDED FROM the PPO encoder's flat
      feature vector (the ``photo_aux.PhotoFlowDecoder`` precedent). Gradients DO
      reach the encoder. ⚠ THIS IS NOT THE REFERENCE ARCHITECTURE: the reference
      attends over a conv map taken straight from pixels, and this attends over a
      map reconstructed from a bottleneck, so the spatial detail available to the
      slots is only whatever survived the encoder's flattening. Report it as an
      adaptation, never as "MoTok".

⚠ NO MOTION PATHWAY, AND THAT IS FAITHFUL, NOT AN OMISSION
----------------------------------------------------------
MoTok is single-frame and appearance-based. ``MoTokNet.get_masks(frame, frame_next)``
IGNORES ``frame_next`` (model.py:624). ``reconstruct()`` never touches the dorsal
stream. The reference keeps ``self.dorsal`` only for the ``get_flow()`` API and
marks it "not trained" (model.py:588) -- and ``_train_ventral`` confirms it, since
the loss is a function of ``reconstruct()`` alone.

**So this port has no motion pathway. That is the reference's property, not a
shortcut taken here.** The owner's "MoTok still needs a loss to train its motion
pathway" is exactly right and this port is evidence for it, not against it.

⚠ THREE PROPERTIES OF THE "WINNING CONFIGURATION" THAT DO NOT SURVIVE READING IT
--------------------------------------------------------------------------------
1. ``gwm_flow_reg: 1e-4`` is **INERT on the MoTok path.** ``GwmPPO_MoTokNet``
   overrides ``_train_ventral`` completely and never references
   ``self._gwm_flow_reg``; only the flow-reconstruction path uses it
   (trainParsing4.py:157). It is inherited config, not a tuned knob.
2. The dorsal stream IS placed in the optimizer (trainParsing.py:145-152, because
   ``hasattr(model,'dorsal')``) at ``gwm_backbone_lr`` with ``weight_decay=1e-4``,
   while receiving ZERO gradient. **AdamW decouples weight decay from the
   gradient, so the untrained dorsal is decayed toward zero over training.** It is
   never read on the MoTok mask path, so this is dead compute rather than a bug --
   but a reader who assumes ``gwm_backbone_lr`` tunes something is mistaken.
3. ``GwmSegWrapper``'s docstring says the foreground query is "selected by calling
   model.get_flow()". **For MoTokNet that is FALSE:** ``get_mask`` is
   ``get_masks(frame)[:, 1:2]`` (model.py:645) -- slot index 1, unconditionally,
   with ``frame_prev`` ignored. The docstring describes a different model's path.

⚠ RESOLUTION: the reference is 224 square; Isaac ships 128x80. Both dims are
divisible by 4, so the H/4 grid is 32x20 -- non-square but exact, and every conv
here is shape-agnostic. ``upsample_size`` (``gwm_upsample: 0`` in the winning
config) is DISABLED by default and would force a square resize if set; it is kept
for parity and should stay 0 on Isaac. Reported, not hidden.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with ~16 channels per group — safe for batch=1 (model.py:31)."""
    return nn.GroupNorm(max(1, channels // 16), channels)


def _conv2d_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        _gn(out_ch),
        nn.ReLU(inplace=True),
    )


class _VQModule(nn.Module):
    """Vector quantization with straight-through estimator (model.py:415-458)."""

    def __init__(self, num_embeddings: int = 64, embedding_dim: int = 32) -> None:
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor):
        B, d, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        dists = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        indices = dists.argmin(dim=1)
        z_q_flat = self.codebook(indices)
        # Straight-through: gradients flow through z_flat to the encoder
        z_q_st = z_flat + (z_q_flat - z_flat).detach()
        commit_loss = (
            F.mse_loss(z_flat, z_q_flat.detach())    # encoder commitment
            + F.mse_loss(z_q_flat, z_flat.detach())  # codebook update
        )
        z_q = z_q_st.reshape(B, H, W, d).permute(0, 3, 1, 2)
        return z_q, commit_loss


class _SlotAttention(nn.Module):
    """Slot attention with GRU updates (model.py:461-531).

    ⚠ The softmax is over SLOTS (``dim=1``), not over positions — that is the
    competition that makes slots partition the image. Softmaxing the other axis
    is the classic silent transposition of this module and it still trains.
    """

    def __init__(
        self,
        num_slots: int = 2,
        slot_dim: int = 32,
        num_iters: int = 3,
        feat_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iters = num_iters
        self.scale = slot_dim ** -0.5

        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.norm_input = nn.LayerNorm(feat_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_ff = nn.LayerNorm(slot_dim)
        self.to_k = nn.Linear(feat_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(feat_dim, slot_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.ff1 = nn.Linear(slot_dim, slot_dim * 2)
        self.ff2 = nn.Linear(slot_dim * 2, slot_dim)

    def forward(self, feats: torch.Tensor):
        B, N, _ = feats.shape
        slots = self.slot_mu.expand(B, -1, -1)
        feats_norm = self.norm_input(feats)
        k = self.to_k(feats_norm)
        v = self.to_v(feats_norm)

        attn = None
        for _ in range(self.num_iters):
            q = self.to_q(self.norm_slots(slots))
            dots = torch.bmm(q, k.transpose(1, 2)) * self.scale
            attn = F.softmax(dots, dim=1)                       # over SLOTS
            attn_norm = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)
            updates = torch.bmm(attn_norm, v)
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots.reshape(B * self.num_slots, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)
            slots = slots + self.ff2(F.relu(self.ff1(self.norm_ff(slots))))
        return slots, attn


class MoTokNet(nn.Module):
    """MoTok segmentation head: VQ codebook + slot attention (model.py:534-665).

    ``in_ch`` is the channel count of ONE frame (3 for RGB). Under framestack the
    caller splits the stack and passes a single frame — MoTok is single-frame.
    """

    def __init__(self, num_queries: int = 2, upsample_size: int = 0, in_ch: int = 3) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.upsample_size = upsample_size
        self.in_ch = in_ch

        self.enc1 = _conv2d_block(in_ch, 32, stride=2)   # H -> H/2
        self.enc2 = _conv2d_block(32, 64, stride=2)      # H/2 -> H/4
        self.enc3 = _conv2d_block(64, 32)                # H/4, channel reduction
        self.feat_proj = nn.Sequential(nn.Conv2d(32, 32, 1, bias=False), _gn(32))

        self.vq = _VQModule(num_embeddings=64, embedding_dim=32)
        self.slot_attn = _SlotAttention(
            num_slots=num_queries, slot_dim=32, num_iters=3, feat_dim=32
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv2d_block(32, 16),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            _conv2d_block(16, 8),
            nn.Conv2d(8, in_ch, 1),
            nn.Sigmoid(),
        )
        # ⛔ NO DORSAL STREAM. The reference carries one for the get_flow() API and
        # never trains it (model.py:588). Porting an untrained module would import
        # dead parameters that AdamW's decoupled weight decay still acts on -- see
        # the module docstring. Its absence here is deliberate and is why this port
        # has no motion pathway.

    def _maybe_upsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample_size and self.upsample_size != x.shape[-1]:
            x = F.interpolate(
                x, size=(self.upsample_size, self.upsample_size),
                mode="bilinear", align_corners=False,
            )
        return x

    def _encode_quantize(self, frame: torch.Tensor):
        z = self.feat_proj(self.enc3(self.enc2(self.enc1(frame))))
        return self.vq(z)

    def _slots_from_feats(self, z_q: torch.Tensor):
        B, d, Hf, Wf = z_q.shape
        feats_flat = z_q.permute(0, 2, 3, 1).reshape(B, Hf * Wf, d)
        slots, attn = self.slot_attn(feats_flat)
        return slots, attn, (B, d, Hf, Wf)

    def reconstruct(self, frame: torch.Tensor):
        """(B,C,H,W) -> ((B,C,H,W) reconstruction, scalar commit_loss). model.py:599."""
        H_orig, W_orig = frame.shape[2:]
        z_q, commit_loss = self._encode_quantize(self._maybe_upsample(frame))
        slots, attn, (B, d, Hf, Wf) = self._slots_from_feats(z_q)
        agg = torch.bmm(attn.transpose(1, 2), slots)
        agg = agg.permute(0, 2, 1).reshape(B, d, Hf, Wf)
        recon = self.decoder(agg)
        if recon.shape[2:] != (H_orig, W_orig):
            recon = F.interpolate(
                recon, size=(H_orig, W_orig), mode="bilinear", align_corners=False
            )
        return recon, commit_loss

    def get_masks(self, frame: torch.Tensor, frame_next: torch.Tensor = None) -> torch.Tensor:
        """(B,C,H,W) -> (B, num_queries, H, W) softmax masks.

        ⚠ ``frame_next`` is accepted and IGNORED — matching model.py:624 exactly.
        The parameter exists so this is drop-in for the reference's call sites; it
        is not a motion input and there is no motion pathway to feed.
        """
        H_orig, W_orig = frame.shape[2:]
        z_q, _ = self._encode_quantize(self._maybe_upsample(frame))
        _, attn, (B, _, Hf, Wf) = self._slots_from_feats(z_q)
        masks_low = F.softmax(attn.reshape(B, self.num_queries, Hf, Wf), dim=1)
        masks = F.interpolate(
            masks_low, size=(H_orig, W_orig), mode="bilinear", align_corners=False
        )
        return F.softmax(masks, dim=1)

    def get_mask(self, frame: torch.Tensor, frame_prev: torch.Tensor = None) -> torch.Tensor:
        """Foreground mask = SLOT INDEX 1, unconditionally (model.py:645).

        ⚠ Slot 1 is foreground BY CONVENTION ONLY — nothing in the objective binds
        slot 1 to the object. With 2 slots and a permutation-symmetric init, which
        slot lands on the object is a coin flip per seed. Any parsing score built
        on this must be permutation-invariant or it measures the coin flip.
        """
        return self.get_masks(frame)[:, 1:2]

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        return self.get_masks(frame)


class _SpatialSeed(nn.Module):
    """Expand a flat (B, in_dim) feature vector to a (B, 32, Hf, Wf) grid.

    Used ONLY by ``NETT_MOTOK_MODE=shared``. Same device as the
    ``photo_aux.PhotoFlowDecoder`` precedent: the shared encoder yields a flat
    vector and a spatial objective has to invent the grid.
    """

    def __init__(self, in_dim: int, hw: tuple[int, int], ch: int = 32) -> None:
        super().__init__()
        self.hw = hw
        self.ch = ch
        self.proj = nn.Linear(in_dim, ch * hw[0] * hw[1])
        self.refine = _conv2d_block(ch, ch)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        x = self.proj(feats).reshape(B, self.ch, self.hw[0], self.hw[1])
        return self.refine(x)


class MoTokAuxLoss(nn.Module):
    """MoTok reconstruction + VQ commitment as an aux loss.

    Loss matches ``GwmPPO_MoTokNet._train_ventral`` (trainParsing.py:371):
        ``F.mse_loss(recon, frame) + gwm_vq_coef * commit_loss``

    Winning-config knobs, env-overridable:
        NETT_MOTOK_VQ_COEF   (0.1)   gwm_vq_coef
        NETT_MOTOK_QUERIES   (2)     gwm_num_queries
        NETT_MOTOK_UPSAMPLE  (0)     gwm_upsample -- keep 0 on non-square Isaac
        NETT_MOTOK_MODE      faithful | shared   (see module docstring)
        NETT_AUX_BATCH               sample cap (gwm_batch_size was 8 in Unity)
    """

    def __init__(self, encoder: nn.Module, *, max_samples: int = 48) -> None:
        super().__init__()
        self.vq_coef = float(os.environ.get("NETT_MOTOK_VQ_COEF", "0.1"))
        self.num_queries = int(os.environ.get("NETT_MOTOK_QUERIES", "2"))
        self.upsample = int(os.environ.get("NETT_MOTOK_UPSAMPLE", "0"))
        self.mode = os.environ.get("NETT_MOTOK_MODE", "faithful").strip().lower()
        if self.mode not in ("faithful", "shared"):
            raise ValueError(
                f"NETT_MOTOK_MODE={self.mode!r} is not recognised. "
                "Use 'faithful' (reference architecture, does NOT shape the PPO "
                "encoder) or 'shared' (adapted, DOES shape it). Refusing rather "
                "than falling back, because a silent default here would decide "
                "the experiment."
            )
        # ⛔ NOT read from NETT_AUX_BATCH's default silently: NETT_AUX_BATCH=1 is a
        # known silent zero-gradient trap elsewhere in this family, and MoTok's
        # GroupNorm is batch-safe but its VQ codebook update is not meaningful at
        # B=1. Refuse rather than train a degenerate codebook.
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", max_samples))
        if self.max_samples < 2:
            raise ValueError(
                f"NETT_AUX_BATCH={self.max_samples} is too small for MoTok: the VQ "
                "codebook update is degenerate at batch < 2."
            )

        self._device = next(encoder.parameters()).device
        self._net: MoTokNet | None = None
        self._seed: _SpatialSeed | None = None
        self._features_dim = int(encoder.features_dim)
        # ⛔ SAME DEFECT AS dsecmos_aux, FIXED HERE TOO: AuxLossPPO registers
        # ``.head.parameters()`` at CONSTRUCTION. A lazily-built head is empty then, so
        # the optimizer captures nothing and the aux trains nothing while logging a
        # falling loss. Build eagerly from the observation space instead.
        from ...body.observation import image_channels_hw
        _c, _h, _w = image_channels_hw(getattr(encoder, "observation_space", None))
        self._in_ch = 3 if (_c % 3 == 0 and _c >= 3) else _c
        self._net = MoTokNet(num_queries=self.num_queries, upsample_size=self.upsample,
                             in_ch=self._in_ch).to(self._device)
        mods = [self._net]
        if self.mode == "shared":
            hf, wf = max(_h // 4, 1), max(_w // 4, 1)
            self._seed = _SpatialSeed(self._features_dim, (hf, wf)).to(self._device)
            mods.append(self._seed)
        self.head = nn.ModuleList(mods)

    # ------------------------------------------------------------------
    def _single_frame(self, prepared: torch.Tensor) -> torch.Tensor:
        """Take the CURRENT frame out of a (B, C*T, H, W) framestacked tensor.

        MoTok is single-frame. Under ``framestack=True, num_frames=2`` the body
        packs both frames into the channel dim, and ``compact_3dcnn.py:83`` shows
        the split convention. The CURRENT frame is the LAST block of channels --
        the same convention the egocentric PNG writer uses for "the frame at t".
        """
        c = prepared.shape[1]
        if c % 3 == 0 and c > 3:
            return prepared[:, -3:]
        return prepared


    # ------------------------------------------------------------------
    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        prepared = encoder._prepare_image(observations)     # (B, C*T, H, W) float [0,1]
        if prepared.shape[0] > self.max_samples:
            idx = torch.randperm(prepared.shape[0], device=prepared.device)[: self.max_samples]
            prepared = prepared[idx]
        frame = self._single_frame(prepared)

        if self.mode == "faithful":
            # Reference path. Gradient stays inside self._net -- it does NOT reach
            # the PPO encoder, and the module docstring says so in full.
            recon, commit = self._net.reconstruct(frame.detach())
            return F.mse_loss(recon, frame.detach()) + self.vq_coef * commit

        # shared: slots attend over a grid expanded from the encoder's features,
        # so the gradient DOES reach the encoder. Not the reference architecture.
        feats = encoder.encode_prepared(prepared)           # grad ON through encoder
        z = self._seed(feats)
        z_q, commit = self._net.vq(z)
        slots, attn, (B, d, hf, wf) = self._net._slots_from_feats(z_q)
        agg = torch.bmm(attn.transpose(1, 2), slots).permute(0, 2, 1).reshape(B, d, hf, wf)
        recon = self._net.decoder(agg)
        if recon.shape[2:] != frame.shape[2:]:
            recon = F.interpolate(
                recon, size=frame.shape[2:], mode="bilinear", align_corners=False
            )
        return F.mse_loss(recon, frame) + self.vq_coef * commit

    # ------------------------------------------------------------------
    def masks(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """(B, num_queries, H, W) masks — for an observation wrapper mirroring GwmSegWrapper.

        ⚠ Inference only. The Unity wrapper RAISES if the mask carries grad
        (seg_wrappers.py:190), so this detaches rather than relying on the caller.
        """
        with torch.no_grad():
            prepared = encoder._prepare_image(observations)
            frame = self._single_frame(prepared)
            return self._net.get_masks(frame).detach()
