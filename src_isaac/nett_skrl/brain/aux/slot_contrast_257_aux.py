"""`slot_contrast_257`: slot contrast with the s257 slot module (BASELINE series, owner 2026-10-28).

Owner: "Slot Contrast (the 257 version)" -- `slot_contrast` (slot_contrast_aux.py) with the slot module of the
presenter's offline SLOTCOMB config s257 (workspace notes/chicken/presenter_i800_i805/bind16/slotoff/slot_ae.py,
s257 = c2 + c5 + c7), trained JOINTLY with the policy encoder (not detached). Registered next to `slot_contrast`;
`slot_contrast` itself is unchanged.

WHAT IS KEPT FROM `slot_contrast`
  * the episode-contiguous (t, t+1) draw from the rollout buffer (`SlotContrastAuxLoss._temporal_pair`, reused);
  * the slot-slot temporal contrast (Manasyan et al. 2025's Slot_Slot_Contrastive_Loss): the t+1 pass is seeded
    with the t slots (the reference's recurrence), slots L2-normalised, batch x slot merged so negatives span the
    batch, cross-entropy on the identity, temperature NETT_SLOTC_TEMP (.1), weight NETT_SLOTC_W_SS (.5);
  * the diagnostics (`slot_contrast_diagnostics`, the mask-variance pair) under NETT_SLOTC_DIAG=1, same keys.

WHAT s257 REPLACES
  * slot init (c2/c7): K = NETT_SLOTC_SLOTS (default 5) per-slot LEARNABLE queries (1, K, 64), init randn * .1,
    with the BO-QSA straight-through at the last iteration (`slots.detach() + init - init.detach()`), in place of
    slot_contrast's per-slot Gaussian draw. Iterations NETT_SLOTC_ITERS (3, as in every slot_ae config).
    The slot-attention module is s257's: LayerNorm'd inputs, bias-free q/k/v, GRU, residual LN-MLP (64 -> 128 -> 64),
    softmax over slots + 1e-8.
  * the reconstruction target (c7): the feature reconstruction against a stop-gradient EMA trunk is GONE (and with
    it the EMA copy, its deepcopy and its doubled activation memory). In its place s257's foreground-weighted RGB
    reconstruction of the CURRENT frame: a spatial-broadcast decoder (each slot tiled at (H/8, W/8) + a soft
    position embedding, three stride-2 5x5 transposed convs to H x W, 5x5 conv, 3x3 conv to RGB + alpha, alpha
    softmaxed over slots, RGB composited) and the loss mean(((recon - x)^2) * (1 + fg_weight * fg)), where fg marks
    object pixels: max - min over the RGB channels > NETT_SLOTC_FG_THRESH (30, on the 0..255 scale) and
    fg_weight = NETT_SLOTC_FG_WEIGHT (10, so object pixels count x11). Weight NETT_SLOTC_W_REC (1.0).
  * the slot-attention input: s257's encoder tail -- the feature map + a soft position embedding (Linear(4 -> C)
    of the (y, x, 1-y, 1-x) grid), flattened, then LayerNorm -> Linear(C, 64) -> ReLU -> Linear(64, 64) -- applied
    to the BACKBONE's spatial features:
        NatureCNN   the unpooled final conv map (`encode_spatial_prepared`), 64 x 6 x 12 at 80x128 (D512P);
        CompactViT  the post-norm patch tokens (`encode_tokens_prepared`), folded to the patch grid: 148 x 10 x 16
                    for ViT-P8-666K (CLS dropped);
        otherwise   slot_contrast's `spatial_features` (a `cnn` Sequential cut before its pool), or a refusal.

⛔ DEVIATION FROM s257, DECLARED: THE c5 LEVER (a 20 x 32 feature grid) IS NOT REPRODUCED. s257's own encoder was
a 4-layer 5x5 CNN at stride 4; here the slots read the policy backbone's grid, which is 6 x 12 = 72 positions for
D512P and 10 x 16 = 160 for ViT-P8. The pixel decoder still reconstructs at the full 80 x 128.

DECISIONS THIS PORT HAD TO MAKE (s257 is a single-frame autoencoder; slot_contrast is a two-frame contrast):
  * The t+1 pass starts from the t slots (slot_contrast's recurrence), not from fresh queries. With learnable
    queries the slot index is already a stable role, so fresh queries would also give a positive; the recurrence
    is kept because it is the term being ported.
  * The BO-QSA straight-through applies only to the query-initialised (t) pass. The recurrent t+1 pass runs plain
    slot attention from the t slots with the gradient flowing along the chain, as slot_contrast's reference does
    (s257 has no rule for a non-query init).
  * Only the t frame is reconstructed (s257 reconstructs its one input frame).
  * Loss weights keep slot_contrast's .5 / 1.0. ⚠ The pixel MSE on [0, 1] images is much smaller than a cross-entropy
    over B*K logits, so at these defaults the contrast dominates the gradient; watch `rec` vs `ss` at G3.

REFUSES: an input that is not one RGB frame (the foreground test is defined on RGB chroma), an eye not divisible by 8,
and NETT_SLOTC_EMA / NETT_SLOTC_NO_DETACH (they configure the EMA feature target this term does not have; setting
them would be a silent no-op).

Every trainable parameter (soft position, input MLP, slot attention, decoder) is in ONE submodule, `head`, which
AuxLossPPO adds to its optimizer and grad clip (see `SlotContrastHead` for why that name is the interface). No
pretrained weights; PyTorch default init, as slot_ae trained.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.compact_vit import CompactViT
from ..encoders.nature_cnn import NatureCNN
from .knobs import (_env_flag_strict, _env_nonneg_float, _env_positive_float, _env_positive_int)
from .slot_contrast_aux import SlotContrastAuxLoss, slot_contrast_diagnostics, spatial_features


def slot_feature_map(encoder: nn.Module, prepared: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) spatial features of the policy backbone, with gradient (joint training)."""
    if isinstance(encoder, CompactViT):
        tok, (n_h, n_w) = encoder.encode_tokens_prepared(prepared)       # post-norm, CLS dropped
        return tok.transpose(1, 2).reshape(tok.shape[0], tok.shape[2], n_h, n_w)
    if isinstance(encoder, NatureCNN):
        return encoder.encode_spatial_prepared(prepared)                  # pre-pool, pre-flatten
    return spatial_features(encoder, prepared)


def soft_grid(h: int, w: int, device, dtype=torch.float32) -> torch.Tensor:
    """slot_ae.grid: (h, w, 4) = (y, x, 1 - y, 1 - x), each in [0, 1]."""
    ys = torch.linspace(0, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([yy, xx, 1 - yy, 1 - xx], -1)


class SoftPos(nn.Module):
    """slot_ae.SoftPos: x (B, C, h, w) + Linear(4 -> C) of the soft grid."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = soft_grid(x.shape[2], x.shape[3], x.device, x.dtype)
        return x + self.lin(g).permute(2, 0, 1)[None]


class SlotInput(nn.Module):
    """slot_ae.Encoder's tail on a backbone map: SoftPos -> flatten -> LN -> Linear(C, D) -> ReLU -> Linear(D, D)."""

    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.softpos = SoftPos(in_dim)
        self.mlp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.softpos(fmap).flatten(2).transpose(1, 2))   # (B, N, D)


class QuerySlotAttention(nn.Module):
    """slot_ae.SlotAttention(init="query"): K learnable per-slot queries, BO-QSA straight-through.

    forward(x) starts from the queries (straight-through at the last iteration); forward(x, slots_init) starts from
    given slots (slot_contrast's recurrence) and runs plain slot attention. Returns slots (B, K, D) and the last
    iteration's attention (B, K, N), softmaxed over slots.
    """

    def __init__(self, slots: int, dim: int, iters: int = 3, mlp_hidden: int = 128) -> None:
        super().__init__()
        self.K, self.iters, self.scale = slots, iters, dim ** -0.5
        self.q = nn.Parameter(torch.randn(1, slots, dim) * .1)
        self.ni, self.ns = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, dim))

    def step(self, slots, k, v):
        q = self.to_q(self.ns(slots))
        logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
        attn = logits.softmax(1) + 1e-8                                    # softmax over slots
        upd = torch.einsum("bkn,bnd->bkd", attn / attn.sum(-1, keepdim=True), v)
        b = slots.shape[0]
        slots = self.gru(upd.reshape(-1, upd.shape[-1]), slots.reshape(-1, slots.shape[-1])).reshape(b, self.K, -1)
        return slots + self.mlp(slots), attn

    def forward(self, x: torch.Tensor, slots_init: torch.Tensor | None = None):
        b = x.shape[0]
        x = self.ni(x)
        k, v = self.to_k(x), self.to_v(x)
        query_init = slots_init is None
        init = self.q.expand(b, -1, -1) if query_init else slots_init
        slots, attn = init, None
        for i in range(self.iters):
            if query_init and i == self.iters - 1:                         # BO-QSA straight-through
                slots = slots.detach() + init - init.detach()
            slots, attn = self.step(slots, k, v)
        return slots, attn


class BroadcastDecoder(nn.Module):
    """slot_ae.Decoder: each slot tiled at (H/8, W/8) + SoftPos, three stride-2 transposed convs to H x W, then
    C channels + one alpha per slot; alpha softmaxed over slots; returns (recon (B, C, H, W), alpha (B, K, H, W))."""

    def __init__(self, dim: int, out_ch: int, h: int, w: int) -> None:
        super().__init__()
        self.h, self.w, self.out_ch = h, w, out_ch
        self.softpos = SoftPos(dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 5, 2, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim, dim, 5, 2, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim, dim, 5, 2, 2, 1), nn.ReLU(),
            nn.Conv2d(dim, dim, 5, padding=2), nn.ReLU(),
            nn.Conv2d(dim, out_ch + 1, 3, padding=1))

    def forward(self, slots: torch.Tensor):
        b, k, d = slots.shape
        x = slots.reshape(b * k, d, 1, 1).expand(-1, -1, self.h // 8, self.w // 8)
        out = self.net(self.softpos(x)).reshape(b, k, self.out_ch + 1, self.h, self.w)
        rgb, alpha = out[:, :, :self.out_ch], out[:, :, self.out_ch:].softmax(1)
        return (rgb * alpha).sum(1), alpha[:, :, 0]


class SlotContrast257Head(nn.Module):
    """The whole trainable set of the term (the `head` AuxLossPPO optimizes)."""

    def __init__(self, inp: nn.Module, attn: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.inp = inp
        self.attn = attn
        self.decoder = decoder


def foreground_mask(x: torch.Tensor, thresh: int) -> torch.Tensor:
    """slot_ae's object pixels on a prepared [0, 1] RGB image: (B, 1, H, W) float, 1 where the uint8 chroma
    (max - min channel) exceeds ``thresh``. Rounded back to the uint8 scale so an integer difference of exactly
    ``thresh`` is not counted by float error."""
    chroma = torch.round((x.max(1).values - x.min(1).values) * 255.0)
    return (chroma > thresh).to(x.dtype)[:, None]


class SlotContrast257AuxLoss(nn.Module):
    """slot_contrast's temporal slot-slot contrast + s257's slot module and fg-weighted pixel reconstruction."""

    needs_memory = True
    attach_memory = SlotContrastAuxLoss.attach_memory
    _temporal_pair = SlotContrastAuxLoss._temporal_pair

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        for knob in ("NETT_SLOTC_EMA", "NETT_SLOTC_NO_DETACH"):
            if os.environ.get(knob) is not None:
                raise ValueError(
                    f"{knob} is set, but slot_contrast_257 has no EMA feature target (s257's pixel reconstruction "
                    f"replaces it); the knob would be a silent no-op. Unset it.")
        self._memory = None
        self.slots = _env_positive_int("NETT_SLOTC_SLOTS", 5)
        self.slot_dim = _env_positive_int("NETT_SLOTC_DIM", 64)
        self.iters = _env_positive_int("NETT_SLOTC_ITERS", 3)
        self.temperature = _env_positive_float("NETT_SLOTC_TEMP", 0.1)
        self.w_ss = _env_nonneg_float("NETT_SLOTC_W_SS", 0.5)
        self.w_rec = _env_nonneg_float("NETT_SLOTC_W_REC", 1.0)
        self.fg_weight = _env_nonneg_float("NETT_SLOTC_FG_WEIGHT", 10.0)
        self.fg_thresh = _env_positive_int("NETT_SLOTC_FG_THRESH", 30)
        self.max_samples = _env_positive_int("NETT_AUX_BATCH", 32)
        self.diag = _env_flag_strict("NETT_SLOTC_DIAG", False)
        if self.w_ss + self.w_rec <= 0:
            raise ValueError("slot_contrast_257: NETT_SLOTC_W_SS + NETT_SLOTC_W_REC must be > 0.")

        from ...body.observation import image_channels_hw
        c, h, w = image_channels_hw(encoder.observation_space)
        if c != 3:
            raise ValueError(
                f"slot_contrast_257 reconstructs ONE RGB frame (s257's foreground test is RGB chroma); the encoder "
                f"input has {c} channels. Refusing rather than redefining the target.")
        if h % 8 or w % 8:
            raise ValueError(f"slot_contrast_257: the {h}x{w} eye is not divisible by 8 (the broadcast decoder).")
        device = next(encoder.parameters()).device
        with torch.no_grad():
            fmap = slot_feature_map(encoder, torch.zeros(1, c, h, w, device=device))
        if fmap.dim() != 4:
            raise TypeError(f"slot_contrast_257: expected a (B, C, h, w) map, got {tuple(fmap.shape)}")
        self.in_dim, self.grid_h, self.grid_w = (int(v) for v in fmap.shape[1:])
        self.image_hw = (h, w)
        self.head = SlotContrast257Head(
            SlotInput(self.in_dim, self.slot_dim),
            QuerySlotAttention(self.slots, self.slot_dim, self.iters),
            BroadcastDecoder(self.slot_dim, c, h, w),
        ).to(device)
        self.last_terms: tuple[float, float, float] | None = None
        self.last_scalars: dict = {}

    def _slots(self, encoder, prepared, slots_init=None):
        fmap = slot_feature_map(encoder, prepared)
        return self.head.attn(self.head.inp(fmap), slots_init)

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        from ...body.observation import prepare_image_tensor

        obs_t, obs_next = self._temporal_pair(encoder, observations)
        space = encoder.observation_space
        prep_t = prepare_image_tensor(obs_t, space)
        prep_n = prepare_image_tensor(obs_next, space)

        slots_t, attn_t = self._slots(encoder, prep_t)                 # from the learned queries
        slots_n, _ = self._slots(encoder, prep_n, slots_init=slots_t)  # slot_contrast's recurrence

        # --- slot-slot temporal contrast (slot_contrast's term, unchanged) ---
        s1 = F.normalize(slots_t, p=2.0, dim=-1).reshape(-1, self.slot_dim)
        s2 = F.normalize(slots_n, p=2.0, dim=-1).reshape(-1, self.slot_dim)
        sim = (s1 @ s2.T) / self.temperature
        ss = F.cross_entropy(sim, torch.arange(sim.shape[0], device=sim.device))

        # --- s257: foreground-weighted RGB reconstruction of frame t ---
        recon, _alpha = self.head.decoder(slots_t)
        fg = foreground_mask(prep_t, self.fg_thresh)
        rec = ((recon - prep_t) ** 2 * (1.0 + self.fg_weight * fg)).mean()

        mask_var, mask_null = -1.0, -1.0
        ss_diag = {"ss_pos_acc": -1.0, "ss_shuffled_acc_within": -1.0, "ss_shuffled_acc_across": -1.0,
                   "ss_chance": -1.0, "ss_batch": -1.0, "ss_pos_acc_x_batch": -1.0, "ss_input_dependence": -1.0,
                   "ss_pos_sim": -1.0, "ss_neg_sim": -1.0}
        if self.diag:
            # Same instruments as slot_contrast. With query init the slot init carries no noise, so the
            # identical-input null is the input-blind floor (0 up to float error) rather than an init-noise floor.
            mask_var = float(attn_t.var(dim=0).mean().detach())
            with torch.no_grad():
                _, attn_null = self._slots(encoder, prep_t[:1].expand_as(prep_t).contiguous())
                mask_null = float(attn_null.var(dim=0).mean())
                ss_diag = slot_contrast_diagnostics(s1.detach(), s2.detach(), self.temperature, self.slots)

        total = self.w_ss * ss + self.w_rec * rec
        self.last_terms = (float(ss.detach()), float(rec.detach()), float(total.detach()))
        self.last_scalars = {"slots": float(self.slots),
                             "positions": float(self.grid_h * self.grid_w),
                             "ss": float(ss.detach()),
                             "rec": float(rec.detach()),
                             "fg_frac": float(fg.mean()),
                             "mask_variance": mask_var,
                             "mask_variance_null": mask_null,
                             "mask_variance_excess": mask_var - mask_null if self.diag else -1.0,
                             **{k: float(v) for k, v in ss_diag.items()}}
        return total
