"""I-JEPA and V-JEPA objectives on the policy encoder (owner, 2026-10-15; brain/encoders/jepa.py).

Both predict, in latent space, the EMA target encoder's tokens at masked positions from the context
encoder's tokens at visible positions. There is no pixel decoder and no contrastive negative. The
context encoder IS the policy encoder: PPO and this loss both train it. The predictor (``head``)
is trained by this loss only.

Faithful to the reference code (facebookresearch/ijepa, facebookresearch/jepa @ main, read
2026-10-15). Each item says where it comes from:

* Target: ``F.layer_norm(target_encoder(x), (D,))``, then the masked positions (ijepa train.py
  forward_target; jepa app/vjepa/train.py forward_target).
* I-JEPA masks (src/masks/multiblock.py, configs/in1k_vith14_ep300.yaml):
  - 4 target blocks, scale (0.15, 0.2), aspect (0.75, 1.5);
  - 1 context block, scale (0.85, 1.0), aspect 1, restricted to the complement of the targets;
  - min_keep 10; one block size per batch; every sample's index list truncated to the batch minimum.
* I-JEPA loss: ``F.smooth_l1_loss`` (train.py loss_fn).
* V-JEPA masks (src/masks/multiblock3d.py, configs/pretrain/vitl16.yaml): two (context, target) pairs.
  - short-range: 8 blocks at spatial scale 0.15;
  - long-range: 2 blocks at 0.7;
  - both: aspect (0.75, 1.5), temporal scale 1.0; target = union of the blocks, context = complement.
* V-JEPA loss: ``mean |z - h|`` (loss_exp 1), averaged over the pairs. One zero-initialised mask token
  per pair.
* Predictors: Linear(D -> pred_dim) + positions (fixed sin-cos) + mask tokens at target positions ->
  ``pred_depth`` pre-norm blocks -> LayerNorm -> Linear(pred_dim -> D). Init trunc_normal(0.02) and
  sqrt(2 * layer_id) rescale. I-JEPA's mask token is trunc_normal; V-JEPA's are zero.

Not the papers (researcher2; listed for the owner):

* Sizes: scaled so encoder + probe + predictor is ~500k (owner 2026-10-15 answer 3: "Keep them to 500k
  parameters, ignoring EMA target"); ``pred_dim`` 48, ``pred_depth`` 3, ``pred_heads`` 2.
* EMA: a CONSTANT decay at each paper's starting value (I 0.996, V 0.998). The papers ramp it to 1.0
  over the run, and the run length is not known here.
* Optimiser: the PPO Adam (lr 7.5e-4, linear anneal), shared with PPO as every aux head here is. The
  papers use AdamW with weight-decay and lr schedules.
* Batch: the PPO minibatch, subsampled to ``aux_batch``.
* No image augmentation, as in I-JEPA.
* Mask RNG: a CPU generator seeded from ``torch.initial_seed()``, so masks are reproducible and the
  policy's RNG stream is untouched.

NOT CHECKPOINTED: the predictor and the EMA target are not in ``final_agent.pt``. The test phase
reads the policy encoder only.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.jepa import JEPABlock, JEPAEncoder, init_jepa_weights, rescale_blocks, sincos_2d, sincos_3d
from ..encoders.nature_cnn import _amp_dtype
from .ema_teacher import EMATeacher

#: (num_blocks, spatial scale) of V-JEPA's two mask configs (configs/pretrain/vitl16.yaml).
VJEPA_MASKS = ((8, 0.15), (2, 0.7))
VJEPA_ASPECT = (0.75, 1.5)
#: I-JEPA (configs/in1k_vith14_ep300.yaml).
IJEPA_NPRED, IJEPA_PRED_SCALE, IJEPA_ENC_SCALE, IJEPA_ASPECT, IJEPA_MIN_KEEP = 4, (0.15, 0.2), (0.85, 1.0), (0.75, 1.5), 10


class JEPAPredictor(nn.Module):
    """ijepa / jepa VisionTransformerPredictor, on this trunk's token grid."""

    def __init__(self, enc: JEPAEncoder, pred_dim: int, depth: int, heads: int, num_mask_tokens: int,
                 zero_mask_tokens: bool) -> None:
        super().__init__()
        d = enc.embed_dim
        self.embed = nn.Linear(d, pred_dim)
        self.mask_tokens = nn.ParameterList(nn.Parameter(torch.zeros(1, 1, pred_dim)) for _ in range(num_mask_tokens))
        pos = sincos_2d(pred_dim, enc.n_h, enc.n_w) if enc.num_frames == 1 else sincos_3d(pred_dim, enc.n_t, enc.n_h, enc.n_w)
        self.register_buffer("pos", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList(JEPABlock(pred_dim, heads, 4.0) for _ in range(depth))
        self.norm = nn.LayerNorm(pred_dim)
        self.proj = nn.Linear(pred_dim, d)
        self.apply(init_jepa_weights)
        rescale_blocks(self.blocks)
        if not zero_mask_tokens:
            for t in self.mask_tokens:
                nn.init.trunc_normal_(t, std=0.02)

    def forward(self, z: torch.Tensor, ctx: torch.Tensor, tgts: list[torch.Tensor], mask_index: int) -> torch.Tensor:
        """z (B, Kc, D) context tokens at indices ``ctx`` (B, Kc); one prediction per target index set,
        returned block-major as (len(tgts)*B, Kt, D), the order of ijepa's ``x.repeat(len(masks), 1, 1)``."""
        b = z.shape[0]
        pos = self.pos.to(z.dtype).expand(b, -1, -1)
        x = self.embed(z) + _gather(pos, ctx)
        n_ctx = x.shape[1]
        outs = []
        for tgt in tgts:
            p = self.mask_tokens[mask_index].to(x.dtype).expand(b, tgt.shape[1], -1) + _gather(pos, tgt)
            outs.append(torch.cat([x, p], dim=1))
        h = torch.cat(outs, dim=0)
        for block in self.blocks:
            h = block(h)
        return self.proj(self.norm(h)[:, n_ctx:])


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


# ── mask samplers: the reference collators, on a (n_h, n_w) grid ─────────────────────────────────────
def _block_size(g: torch.Generator, n_h: int, n_w: int, scale, aspect, clip_strict: bool,
                shared_rand: bool) -> tuple[int, int]:
    """I-JEPA draws ONE uniform for scale and aspect (shared_rand) and clips h < n_h, w < n_w
    (clip_strict); V-JEPA draws them separately and clips h <= n_h, w <= n_w."""
    r1 = torch.rand(1, generator=g).item()
    keep = int(n_h * n_w * (scale[0] + r1 * (scale[1] - scale[0])))
    r2 = r1 if shared_rand else torch.rand(1, generator=g).item()
    ar = aspect[0] + r2 * (aspect[1] - aspect[0])
    h, w = int(round(math.sqrt(keep * ar))), int(round(math.sqrt(keep / ar)))
    if clip_strict:
        while h >= n_h:
            h -= 1
        while w >= n_w:
            w -= 1
    else:
        h, w = min(h, n_h), min(w, n_w)
    return h, w


def _place(g: torch.Generator, n_h: int, n_w: int, h: int, w: int, inclusive: bool) -> tuple[int, int]:
    top = int(torch.randint(0, n_h - h + (1 if inclusive else 0), (1,), generator=g))
    left = int(torch.randint(0, n_w - w + (1 if inclusive else 0), (1,), generator=g))
    return top, left


def ijepa_masks(g: torch.Generator, b: int, n_h: int, n_w: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """ijepa src/masks/multiblock.MaskCollator.__call__ (nenc 1, npred 4, allow_overlap False):
    (context (B, Kc), [target (B, Kt)] * 4), each row ascending and truncated to the batch minimum."""
    p_size = _block_size(g, n_h, n_w, IJEPA_PRED_SCALE, IJEPA_ASPECT, True, True)
    e_size = _block_size(g, n_h, n_w, IJEPA_ENC_SCALE, (1.0, 1.0), True, True)
    preds, encs = [], []
    for _ in range(b):
        ps, comps = [], []
        for _ in range(IJEPA_NPRED):
            top, left = _place(g, n_h, n_w, *p_size, inclusive=False)
            m = torch.zeros(n_h, n_w, dtype=torch.bool)
            m[top:top + p_size[0], left:left + p_size[1]] = True
            ps.append(m.flatten().nonzero().squeeze(1))
            comps.append(~m)
        preds.append(ps)
        # Context block, constrained to the complement of the targets. As the reference does, a too-small
        # result is retried 20 times, then one more constraint is dropped, and so on.
        tries, timeout = 0, 20
        while True:
            top, left = _place(g, n_h, n_w, *e_size, inclusive=False)
            m = torch.zeros(n_h, n_w, dtype=torch.bool)
            m[top:top + e_size[0], left:left + e_size[1]] = True
            for c in comps[:max(len(comps) - tries, 0)]:
                m &= c
            idx = m.flatten().nonzero().squeeze(1)
            if len(idx) > IJEPA_MIN_KEEP:
                break
            timeout -= 1
            if timeout == 0:
                tries, timeout = tries + 1, 20
        encs.append(idx)
    kp = min(len(p) for ps in preds for p in ps)
    ke = min(len(e) for e in encs)
    ctx = torch.stack([e[:ke] for e in encs])
    tgts = [torch.stack([ps[j][:kp] for ps in preds]) for j in range(IJEPA_NPRED)]
    return ctx, tgts


def vjepa_masks(g: torch.Generator, b: int, n_h: int, n_w: int, num_blocks: int, scale: float, n_t: int = 1
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """jepa src/masks/multiblock3d._MaskGenerator.__call__ at temporal scale 1.0: target = union of
    ``num_blocks`` blocks, context = the rest. At temporal scale 1.0 every block spans the whole duration
    (``n_t`` tubelets; t-start is always 0), so a spatial mask is repeated over t. Indices are t-major,
    ascending, as the reference's ``nonzero`` over the flattened (T, H, W) mask."""
    # Temporal scale draw first (its block always spans all n_t at scale 1.0; kept so the draw order matches).
    torch.rand(1, generator=g)
    size = _block_size(g, n_h, n_w, (scale, scale), VJEPA_ASPECT, False, False)
    ctxs, tgts = [], []
    for _ in range(b):
        while True:
            keep = torch.ones(n_h, n_w, dtype=torch.bool)
            for _ in range(num_blocks):
                top, left = _place(g, n_h, n_w, *size, inclusive=True)
                torch.randint(0, 1, (1,), generator=g)          # the reference's temporal start draw
                keep[top:top + size[0], left:left + size[1]] = False
            flat = keep.flatten()
            if flat.any():
                break
        flat = flat.repeat(n_t)
        ctxs.append(flat.nonzero().squeeze(1))
        tgts.append((~flat).nonzero().squeeze(1))
    kc, kt = min(len(c) for c in ctxs), min(len(t) for t in tgts)
    return torch.stack([c[:kc] for c in ctxs]), torch.stack([t[:kt] for t in tgts])


class JEPATerm(nn.Module):
    """``AUX_LOSSES['ijepa']`` / ``['vjepa']``. ``head`` = the predictor (the PPO optimiser trains it)."""

    needs_memory = False

    def __init__(self, encoder: nn.Module, mode: str) -> None:
        super().__init__()
        if type(encoder) is not JEPAEncoder:
            raise TypeError(f"{mode} needs the JEPAEncoder trunk; got {type(encoder).__name__}.")
        cfg = encoder.jepa
        if cfg["mode"] != mode:
            raise ValueError(f"aux {mode!r} on an encoder configured for {cfg['mode']!r}")
        want_loss = "smooth_l1" if mode == "ijepa" else "l1"
        if cfg["loss"] != want_loss:
            raise ValueError(f"{mode}: loss must be the reference {want_loss!r}; got {cfg['loss']!r}")
        if int(cfg["aux_batch"]) < 2:
            raise ValueError(f"{mode}: aux_batch must be >= 2; got {cfg['aux_batch']}")
        self.mode = mode
        self.max_samples = int(cfg["aux_batch"])
        self.n_h, self.n_w, self.n_t = encoder.n_h, encoder.n_w, encoder.n_t
        device = next(encoder.parameters()).device
        n_tokens = 1 if mode == "ijepa" else len(VJEPA_MASKS)
        self.head = JEPAPredictor(encoder, int(cfg["pred_dim"]), int(cfg["pred_depth"]), int(cfg["pred_heads"]),
                                  num_mask_tokens=n_tokens, zero_mask_tokens=(mode == "vjepa")).to(device)
        self.teacher = EMATeacher(encoder, decay=float(cfg["ema"]))
        self._gen = torch.Generator().manual_seed(int(torch.initial_seed()) % (2 ** 63))
        self.last_scalars: dict = {}

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prepared = encoder._prepare_image(observations)
            if prepared.shape[0] > self.max_samples:
                idx = torch.randperm(prepared.shape[0], device=prepared.device)[:self.max_samples]
                prepared = prepared[idx]
        b, dev = prepared.shape[0], prepared.device
        # EMA once per compute, i.e. after the previous minibatch's optimiser step (the papers step it
        # right after theirs).
        self.teacher.step(encoder)
        amp = _amp_dtype()
        use_amp = amp is not None and prepared.is_cuda
        ctx_amp = torch.autocast("cuda", dtype=amp) if use_amp else torch.autocast("cpu", enabled=False)
        with ctx_amp:
            with torch.no_grad():
                h_raw = self.teacher.module.tokens_prepared(prepared).float()
            h = F.layer_norm(h_raw, (h_raw.shape[-1],))
            if self.mode == "ijepa":
                ctx, tgts = ijepa_masks(self._gen, b, self.n_h, self.n_w)
                ctx, tgts = ctx.to(dev), [t.to(dev) for t in tgts]
                z = encoder.tokens_prepared(prepared, keep=ctx)
                pred = self.head(z, ctx, tgts, mask_index=0).float()
                target = torch.cat([_gather(h, t) for t in tgts], dim=0)
                loss = F.smooth_l1_loss(pred, target)
                n_ctx, n_tgt = ctx.shape[1], sum(t.shape[1] for t in tgts)
            else:
                loss, n_ctx, n_tgt = 0.0, 0, 0
                for i, (nb, sc) in enumerate(VJEPA_MASKS):
                    ctx, tgt = vjepa_masks(self._gen, b, self.n_h, self.n_w, nb, sc, self.n_t)
                    ctx, tgt = ctx.to(dev), tgt.to(dev)
                    z = encoder.tokens_prepared(prepared, keep=ctx)
                    pred = self.head(z, ctx, [tgt], mask_index=i).float()
                    loss = loss + torch.mean(torch.abs(pred - _gather(h, tgt)))
                    n_ctx, n_tgt = n_ctx + ctx.shape[1], n_tgt + tgt.shape[1]
                loss = loss / len(VJEPA_MASKS)
        with torch.no_grad():
            # Collapse monitor: spread across the batch of the RAW (pre-layer-norm) mean target token.
            tgt_std = float(h_raw.mean(dim=1).std(dim=0).mean())
        self.last_scalars = {
            "B": float(b), "n_ctx": float(n_ctx), "n_tgt": float(n_tgt),
            "ema": float(self.teacher.decay), "target_std": tgt_std,
        }
        return loss


def build_ijepa(encoder: nn.Module) -> JEPATerm:
    return JEPATerm(encoder, "ijepa")


def build_vjepa(encoder: nn.Module) -> JEPATerm:
    return JEPATerm(encoder, "vjepa")
