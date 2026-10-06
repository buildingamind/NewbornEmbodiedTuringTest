"""`mae`: masked autoencoding as an auxiliary loss on the policy's own encoder (owner, 2026-10-05).

Owner request (workspace DECISIONS §71 and its follow-up): "MAE ... for ViT-CLTT-Ref. Additionally,
MAE applied to 3DCNN-CLTT-Ref. It would be interesting to see what partial masking of a stacked
observation space would do. ... VideoMAE and VideoMAE-CLTT-Ref."

ONE TERM, THREE ENCODER FAMILIES, CHOSEN BY THE ENCODER'S TYPE (anything else REFUSES):

  CompactViT   -> TOKEN MAE (He et al. 2022). A random ``NETT_AUX_MAE_RATIO`` (default .75) of
                  the patch tokens is dropped; the SAME trunk encodes only the visible ones (+CLS,
                  `CompactViT.encode_visible_prepared`); a light decoder with its own position
                  embedding fills mask tokens back in and regresses each masked patch's pixels.
                  ⚠ WITH A FRAME STACK THIS IS TUBE MASKING, BY CONSTRUCTION: CompactViT's patch
                  embed is one Conv2d over all C*T stacked channels, so a token IS a location in
                  every frame, and dropping it hides that location in BOTH frames. Per-frame
                  masking is not expressible on this trunk; ``NETT_AUX_MAE_TUBE=0`` refuses here.
  Compact3DCNN -> MASKED-INPUT reconstruction (SimMIM-style, Xie et al. 2022): a CNN has no tokens
                  to drop, so masked patches are REPLACED in the input by a learned per-channel
                  value and the full image is encoded. ``NETT_AUX_MAE_TUBE`` defaults to 0 here:
                  each frame of the stack draws its OWN mask, so a location hidden in frame t-1 may
                  be visible in frame t and vice versa -- "partial masking of a stacked
                  observation", the owner's question. ``NETT_AUX_MAE_TUBE=1`` masks the same
                  patches in every frame, as the contrast. The decoder reads the encoder's
                  unpooled map (`Compact3DCNN.encode_spatial`, stride 4) and pixel-shuffles back
                  to every frame; the loss is per frame, on masked (frame, patch) cells only.
  CompactViViT -> VIDEOMAE (Tong et al. 2022). ViViT's joint mode has one token per (frame, patch);
                  ``NETT_AUX_MAE_TUBE`` defaults to 1 (VideoMAE's tube masking: the same patches
                  in every frame) at ratio default .90; the visible tokens alone go through the
                  trunk (`CompactViViT.encode_visible_prepared`), the decoder reconstructs every
                  masked (frame, patch) token's pixels.

LOSS: per-patch normalized pixel MSE on masked patches (He et al.'s norm_pix_loss, default on;
``NETT_AUX_MAE_NORM_PIX=0`` regresses raw [0, 1] pixels). Each patch target is normalized over
its own pixels -- for the CNN and ViViT modes that is one frame's patch, for the ViT mode the
C*T-channel stacked patch the token covers.

WHERE THE IMAGES COME FROM: the PPO minibatch (``observations``), like SimCLR -- MAE needs no
temporal pair -- subsampled to ``NETT_AUX_MAE_BATCH`` (default 64). Its own knob, NOT
NETT_AUX_BATCH: that one sets cltt_ref's B, and inside WithCLTTRef the control's objective must
not move. ⚠ The subsample and the mask draw consume the global RNG AFTER cltt_ref has drawn
(WithCLTTRef computes cltt_ref first), so within each aux call cltt_ref draws first, as in the
control; across calls the stream has been advanced by these draws, as for every WithCLTTRef term,
so the composed row's cltt_ref windows match the control's only on the first call.

DEVIATIONS, DECLARED:
- Decoder size: depth ``NETT_AUX_MAE_DEC_DEPTH`` (1), width ``NETT_AUX_MAE_DEC_DIM`` (64), heads
  ``NETT_AUX_MAE_DEC_HEADS`` (4) -- He et al. use 8 x 512 for ViT-L. Scaled to this ~700K-param
  trunk so the decoder does not out-size the encoder it trains.
- Patches: the ViT/ViViT modes use the encoder's own 16-px patch; the 3DCNN mode's
  ``NETT_AUX_MAE_PATCH`` (16) must divide the eye (80x128 -> 5x8 = 40 cells per frame).
- The CNN mode's mask value is LEARNED (one per input channel, in ``head``), not zero: zero is a
  real colour in this scene (black), so a zeroed patch is indistinguishable from dark content;
  SimMIM likewise uses a learned mask token. Initialized at 0.5.
- The policy reads UNMASKED observations; only this term sees masks (as in MAE fine-tuning).
- No pretrained weights anywhere: every parameter starts from scratch, as the campaign requires.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.compact_3dcnn import Compact3DCNN
from ..encoders.compact_vit import CompactViT, _TransformerBlock
from ..encoders.compact_vivit import CompactViViT
from .knobs import _env_flag_strict, _env_positive_int, _env_unit_interval

MODES = {CompactViT: "vit", Compact3DCNN: "cnn3d", CompactViViT: "vivit"}
_DEFAULT_RATIO = {"vit": 0.75, "cnn3d": 0.75, "vivit": 0.90}
_DEFAULT_TUBE = {"vit": True, "cnn3d": False, "vivit": True}


def patchify(img: torch.Tensor, p: int) -> torch.Tensor:
    """(B, C, H, W) -> (B, (H/p)*(W/p), C*p*p), row-major over the patch grid.

    Row-major is the order CompactViT's Conv2d patch embed flattens and `compact_vivit._patchify`
    produces, so patch n here is token n there.
    """
    b, c, h, w = img.shape
    x = img.reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // p) * (w // p), c * p * p)


def frames_tmajor(prepared: torch.Tensor, num_frames: int) -> torch.Tensor:
    """(B, C*T, H, W) -> (B, T, C, H, W). ⛔ T-MAJOR: channels are [t0 RGB, t1 RGB, ...]
    (observation.py's torch.cat on the channel axis; see eoo_aux's docstring for the C-major bug
    that once paired two channels of ONE frame and called it motion)."""
    b, ct, h, w = prepared.shape
    if ct % num_frames:
        raise ValueError(f"{ct} channels do not split into {num_frames} frames.")
    return prepared.view(b, num_frames, ct // num_frames, h, w)


def random_mask(b: int, n: int, ratio: float, device) -> torch.Tensor:
    """MAE's per-sample shuffle as a (B, N) bool mask, True = masked; every row masks the same
    count, round(N * ratio), and keeps at least one patch."""
    n_keep = max(1, int(round(n * (1.0 - ratio))))
    if n_keep >= n:
        raise ValueError(f"mask ratio {ratio} leaves all {n} patches visible; nothing to predict.")
    ranks = torch.rand(b, n, device=device).argsort(dim=1).argsort(dim=1)
    return ranks >= n_keep


def split_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, N) bool mask with equal per-row counts -> (keep (B, N_vis) ascending, ids_restore (B, N)).

    ``order`` lists the visible indices (ascending) then the masked ones; ``ids_restore`` is its
    inverse, so ``gather(cat([visible, mask_tokens]), ids_restore)`` puts every token back in its
    own position -- He et al.'s reference unshuffle. ⛔ gather + a STABLE sort, never scatter: the
    PPO update's forward runs under use_deterministic_algorithms(True), where an index-writing
    forward op can raise at the first optimizer step (see token_term.parked_transit).
    """
    n_vis = int((~mask[0]).sum())
    if bool(((~mask).sum(dim=1) != n_vis).any()):
        raise ValueError("split_mask: rows mask different counts; MAE batches need equal counts.")
    order = mask.to(torch.int8).sort(dim=1, stable=True).indices
    return order[:, :n_vis], order.argsort(dim=1)


def masked_patch_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     norm_pix: bool) -> torch.Tensor:
    """Mean over masked patches of the per-patch MSE; target normalized per patch if asked."""
    if norm_pix:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
    per = ((pred - target) ** 2).mean(dim=-1)
    m = mask.to(per.dtype)
    return (per * m).sum() / m.sum().clamp_min(1.0)


class _TokenDecoder(nn.Module):
    """MAE decoder: embed visible tokens, insert a shared mask token at the masked positions,
    add a decoder position embedding (CLS + N positions), run blocks, regress pixels."""

    def __init__(self, enc_dim: int, n_pos: int, out_dim: int, dim: int, depth: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"NETT_AUX_MAE_DEC_DIM={dim} is not divisible by NETT_AUX_MAE_DEC_HEADS={heads}.")
        self.embed = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + n_pos, dim))
        self.blocks = nn.ModuleList([_TransformerBlock(dim, heads, 2.0) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.pred = nn.Linear(dim, out_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, enc_tokens: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """enc_tokens (B, 1+N_vis, D_enc) CLS first -> pixel predictions (B, N, out_dim)."""
        x = self.embed(enc_tokens)
        b, n_vis, d = x.shape[0], x.shape[1] - 1, x.shape[-1]
        n = ids_restore.shape[1]
        seq = torch.cat([x[:, 1:], self.mask_token.expand(b, n - n_vis, d)], dim=1)
        seq = torch.gather(seq, 1, ids_restore.unsqueeze(-1).expand(-1, -1, d))
        x = torch.cat([x[:, :1], seq], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.pred(self.norm(x)[:, 1:])


class _ConvDecoder(nn.Module):
    """SimMIM-style: unpooled map -> 3x3 conv -> 1x1 conv to T*C*s*s -> pixel shuffle by s."""

    def __init__(self, in_ch: int, out_ch: int, scale: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, dim, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(dim, out_ch * scale * scale, kernel_size=1),
            nn.PixelShuffle(scale),
        )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.net(fmap)


class MAETerm(nn.Module):
    """Masked autoencoding on the policy encoder; usable alone or inside WithCLTTRef."""

    needs_memory = False

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        mode = next((m for cls, m in MODES.items() if type(encoder) is cls), None)
        if mode is None:
            raise TypeError(
                f"MAETerm supports CompactViT (token MAE), Compact3DCNN (masked input) and "
                f"CompactViViT (VideoMAE); got {type(encoder).__name__}. Refusing to substitute "
                f"another masking scheme under the same name.")
        self.mode = mode
        self.ratio = _env_unit_interval("NETT_AUX_MAE_RATIO", _DEFAULT_RATIO[mode])
        if not 0.0 < self.ratio < 1.0:
            raise ValueError(f"NETT_AUX_MAE_RATIO={self.ratio} must be strictly between 0 and 1.")
        self.tube = _env_flag_strict("NETT_AUX_MAE_TUBE", _DEFAULT_TUBE[mode])
        if mode == "vit" and not self.tube:
            raise ValueError(
                "NETT_AUX_MAE_TUBE=0 on CompactViT: one token covers a location in EVERY stacked "
                "frame, so per-frame masking is not expressible on this trunk. Use 3DCNN or "
                "ViViT for per-frame masks.")
        self.norm_pix = _env_flag_strict("NETT_AUX_MAE_NORM_PIX", True)
        self.max_samples = _env_positive_int("NETT_AUX_MAE_BATCH", 64)
        dec_dim = _env_positive_int("NETT_AUX_MAE_DEC_DIM", 64)
        dec_depth = _env_positive_int("NETT_AUX_MAE_DEC_DEPTH", 1)
        dec_heads = _env_positive_int("NETT_AUX_MAE_DEC_HEADS", 4)

        if mode == "vit":
            if encoder.attn_mode != "qk":
                raise ValueError(f"MAETerm: CompactViT attn_mode={encoder.attn_mode!r}; need 'qk'.")
            if not isinstance(encoder.patch_embed, nn.Conv2d):
                raise ValueError("MAETerm: token MAE needs CompactViT's linear (Conv2d) patch "
                                 "stem; the conv stem's tokens are not disjoint pixel patches.")
            self.patch = int(encoder.patch_embed.kernel_size[0])
            self.n_spatial = None
            self.n_pos = encoder._n_h * encoder._n_w
            chans = encoder.patch_embed.in_channels
            self.num_frames = None
            self.head = nn.ModuleDict({"dec": _TokenDecoder(
                encoder.pos_embed.shape[-1], self.n_pos, chans * self.patch ** 2,
                dec_dim, dec_depth, dec_heads)})
        elif mode == "vivit":
            if encoder.temporal_mode != "joint" or not encoder.use_cls:
                raise ValueError("MAETerm: VideoMAE needs CompactViViT temporal_mode='joint', pool='cls'.")
            self.patch = int(encoder.patch_size)
            self.num_frames = int(encoder.num_frames)
            self.n_spatial = int(encoder.n_spatial)
            self.n_pos = self.num_frames * self.n_spatial
            self.head = nn.ModuleDict({"dec": _TokenDecoder(
                encoder.embed_dim, self.n_pos, encoder.base_channels * self.patch ** 2,
                dec_dim, dec_depth, dec_heads)})
        else:
            if encoder.duplicate_frame:
                raise ValueError("MAETerm: 3DCNN duplicate_frame sees ONE frame twice; masking "
                                 "'each frame independently' would be meaningless there.")
            self.patch = _env_positive_int("NETT_AUX_MAE_PATCH", 16)
            self.num_frames = int(encoder.num_frames)
            c = int(encoder.base_channels)
            _, h, w = _chw(encoder)
            if h % self.patch or w % self.patch:
                raise ValueError(f"NETT_AUX_MAE_PATCH={self.patch} does not divide the {h}x{w} eye.")
            with torch.no_grad():
                fmap = encoder.encode_spatial_prepared(torch.zeros(1, c * self.num_frames, h, w))
            fh, fw = fmap.shape[-2:]
            if h % fh or w % fw or h // fh != w // fw:
                raise ValueError(f"3DCNN map {fh}x{fw} is not an integer downsampling of {h}x{w}.")
            self.scale = h // fh
            self.n_h, self.n_w = h // self.patch, w // self.patch
            self.n_pos = self.n_h * self.n_w
            self.head = nn.ModuleDict({
                "dec": _ConvDecoder(fmap.shape[1], c * self.num_frames, self.scale, dec_dim),
                "mask_value": _MaskValue(c),
            })
        self.last_scalars: dict = {}

    # ---------------------------------------------------------------- masks
    def draw_masks(self, b: int, device) -> torch.Tensor:
        """vit: (B, N). vivit: (B, T*N) over the joint frame-major sequence. cnn3d: (B, T, N).
        True = masked. Tube masking repeats one spatial mask over every frame."""
        if self.mode == "vit":
            return random_mask(b, self.n_pos, self.ratio, device)
        t = self.num_frames
        n = self.n_spatial if self.mode == "vivit" else self.n_pos
        if self.tube:
            m = random_mask(b, n, self.ratio, device).unsqueeze(1).expand(b, t, n)
        else:
            m = torch.stack([random_mask(b, n, self.ratio, device) for _ in range(t)], dim=1)
        m = m.contiguous()
        return m.reshape(b, t * n) if self.mode == "vivit" else m

    def mask_input(self, prepared: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """cnn3d: replace masked (frame, patch) cells by the learned per-channel value."""
        fr = frames_tmajor(prepared, self.num_frames)            # (B, T, C, H, W)
        b, t, c, h, w = fr.shape
        pix = mask.view(b, t, self.n_h, 1, self.n_w, 1).expand(
            b, t, self.n_h, self.patch, self.n_w, self.patch).reshape(b, t, 1, h, w)
        fill = self.head["mask_value"]().view(1, 1, c, 1, 1).to(fr.dtype)
        out = torch.where(pix, fill.expand_as(fr), fr)
        return out.reshape(b, t * c, h, w)

    # ---------------------------------------------------------------- loss
    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prepared = encoder._prepare_image(observations)
            if prepared.shape[0] > self.max_samples:
                idx = torch.randperm(prepared.shape[0], device=prepared.device)[:self.max_samples]
                prepared = prepared[idx]
        b, device = prepared.shape[0], prepared.device
        mask = self.draw_masks(b, device)
        if self.mode in ("vit", "vivit"):
            keep, ids_restore = split_mask(mask)
            enc = encoder.encode_visible_prepared(prepared, keep)
            pred = self.head["dec"](enc, ids_restore)
            if self.mode == "vit":
                target = patchify(prepared, self.patch)
            else:
                fr = frames_tmajor(prepared, self.num_frames)
                target = torch.cat([patchify(fr[:, i], self.patch) for i in range(self.num_frames)], 1)
        else:
            fmap = encoder.encode_spatial_prepared(self.mask_input(prepared, mask))
            recon = self.head["dec"](fmap)                         # (B, T*C, H, W)
            rf, tf = frames_tmajor(recon, self.num_frames), frames_tmajor(prepared, self.num_frames)
            pred = torch.cat([patchify(rf[:, i], self.patch) for i in range(self.num_frames)], 1)
            target = torch.cat([patchify(tf[:, i], self.patch) for i in range(self.num_frames)], 1)
            mask = mask.reshape(b, -1)                             # (B, T*N), frame-major
        loss = masked_patch_mse(pred, target, mask, self.norm_pix)
        self.last_scalars = {
            "B": float(b),
            "mask_frac": float(mask.float().mean()),
            "ratio": float(self.ratio),
            "tube": float(self.tube),
        }
        return loss


class _MaskValue(nn.Module):
    """One learned fill value per input channel (the CNN mode's mask token), initialized at 0.5."""

    def __init__(self, channels: int):
        super().__init__()
        self.value = nn.Parameter(torch.full((channels,), 0.5))

    def forward(self) -> torch.Tensor:
        return self.value


def _chw(encoder: nn.Module) -> tuple[int, int, int]:
    from ...body.observation import image_channels_hw
    return image_channels_hw(encoder.observation_space)
