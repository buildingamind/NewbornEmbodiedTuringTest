"""I-JEPA / V-JEPA context encoder (owner, 2026-10-15, chat to researcher2).

Owner's words: "I'd like to create an I-JEPA and V-JEPA implementation and run it. Essentially, keeping
with the standards set by Unity match ++. Try to keep the parameters of the model to 500k. ... run the
binding on I-JEPA and parsing on V-JEPA. Try them out using bf16 on 64 env."

One class, two modes, chosen by ``num_frames``:

* ``num_frames=1`` -- I-JEPA (Assran et al. 2023, facebookresearch/ijepa). A plain ViT over 2-D patches.
  Owner 2026-10-15 (answer 1): I-JEPA runs WITHOUT the decision loop ("only use a 1-tick loop").
* ``num_frames=2`` -- V-JEPA (Bardes et al. 2024, facebookresearch/jepa). A video ViT whose patch embed is
  a Conv3d over a TUBELET of ``tubelet`` x patch x patch. Owner 2026-10-15 (answer 2): "V-JEPA use
  tubelet 1" -- each tick of the 2-tick loop is tokenised separately (2 x 640 = 1280 tokens), with
  3-D positions over (t, h, w). The paper's own tubelet is 2.

Shared, from the papers: no CLS token; FIXED sin-cos position embeddings (2-D for I-JEPA, 3-D for V-JEPA,
as each repo builds them); pre-norm blocks with an MLP ratio of 4; a final LayerNorm; trunc_normal(0.02)
init. Patch 14 divides the Unity-recipe++ eye (448 x 280 -> 32 x 20 = 640 tokens per frame).

POLICY READOUT = each paper's own probe (owner 2026-10-15, answer 4: "Use the JEPA papers' probe convention"):
* I-JEPA ``probe="mean_linear"`` -- "the average-pooled patch representation of the last layer" into a
  linear head (I-JEPA Appendix A.2; its batch-norm variant is not used).
* V-JEPA ``probe="attentive"`` -- jepa src/models/attentive_pooler.AttentiveClassifier: one learnable query
  token, one cross-attention block (qkv bias, MLP ratio 4, the encoder's head count), then a linear head.
Here the probe is trained by PPO (it is the policy's input), not on frozen features as in the papers.

The JEPA OBJECTIVE lives in ``brain/aux/jepa_aux.py``. Its settings ride on this encoder as ``jepa``
(a dict, explicit keys only) so one MODELS line names every knob of the arm; the encoder itself never
reads them.

⛔ ATTENTION IS ``F.scaled_dot_product_attention``, NOT ``nn.MultiheadAttention``. The PPO update runs under
``torch.use_deterministic_algorithms(True, warn_only=False)``. Measured on insect gpu2 (torch 2.7.0, A10,
bf16, 640 tokens): the flash and memory-efficient SDPA backward run and are bit-repeatable; the math path
raises under that mode. nn.MultiheadAttention's training path also materialises every (N x N) weight map.

bf16: the forward runs under CUDA autocast when ``NETT_AMP`` selects it (the same reader as NatureCNN) and
returns fp32 features, so the skrl heads stay full precision.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor
from .nature_cnn import _amp_dtype
from .utils.temporal import validate_framestack_depth

#: Keys the JEPA objective reads from ``jepa`` (brain/aux/jepa_aux.py). Explicit, so a typo refuses.
JEPA_OBJECTIVE_KEYS = frozenset({
    "mode", "pred_dim", "pred_depth", "pred_heads", "ema", "aux_batch", "loss",
})


# ── fixed sin-cos position embeddings, as the reference repos build them ────────────────────────────
def _sincos_1d(dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1).astype(np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_2d(dim: int, n_h: int, n_w: int) -> torch.Tensor:
    """(n_h*n_w, dim), row-major over (h, w). ijepa/src/models/vision_transformer.get_2d_sincos_pos_embed
    on a rectangular grid: its meshgrid puts w first, so the FIRST half of the channels encodes the column
    (w) and the second half the row (h)."""
    if dim % 4:
        raise ValueError(f"2-D sin-cos needs embed_dim % 4 == 0; got {dim}")
    gh, gw = np.meshgrid(np.arange(n_h, dtype=np.float64), np.arange(n_w, dtype=np.float64), indexing="ij")
    emb = np.concatenate([_sincos_1d(dim // 2, gw), _sincos_1d(dim // 2, gh)], axis=1)
    return torch.from_numpy(emb).float()


def sincos_3d(dim: int, n_t: int, n_h: int, n_w: int) -> torch.Tensor:
    """(n_t*n_h*n_w, dim), t-major then row-major. jepa/src/models/utils/pos_embs.get_3d_sincos_pos_embed
    with ``uniform_power=True`` (configs/pretrain/vitl16.yaml): t, h and w each take ceil(dim/6)*2
    channels, concatenated [t, h, w] and truncated to ``dim``."""
    part = int(np.ceil(dim / 6) * 2)
    gt, gh, gw = np.meshgrid(np.arange(n_t, dtype=np.float64), np.arange(n_h, dtype=np.float64),
                             np.arange(n_w, dtype=np.float64), indexing="ij")
    emb = np.concatenate([_sincos_1d(part, gt), _sincos_1d(part, gh), _sincos_1d(part, gw)], axis=1)[:, :dim]
    return torch.from_numpy(emb).float()


class JEPABlock(nn.Module):
    """Pre-norm ViT block: x + attn(LN(x)); x + MLP(LN(x)). Attention through SDPA (see module note)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"embed_dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.num_heads, d // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)              # each (B, H, N, d_h)
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.norm2(x))


def init_jepa_weights(m: nn.Module) -> None:
    """ijepa/src/models/vision_transformer._init_weights: trunc_normal(0.02) on Linear, zero bias;
    LayerNorm 1/0; the patch conv as a Linear over its flattened kernel."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def rescale_blocks(blocks: nn.ModuleList) -> None:
    """ijepa fix_init_weight / jepa _rescale_blocks: divide each block's attention output projection and
    second MLP layer by sqrt(2 * layer_id), layer_id counted from 1."""
    with torch.no_grad():
        for layer_id, block in enumerate(blocks, start=1):
            block.proj.weight.div_(math.sqrt(2.0 * layer_id))
            block.mlp[2].weight.div_(math.sqrt(2.0 * layer_id))


class CrossAttentionBlock(nn.Module):
    """jepa src/models/utils/modules.CrossAttentionBlock: q + xattn(q, LN(x)); q + MLP(LN(q)). SDPA."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        b, n, d = q.shape
        h = self.num_heads
        qh = self.q(q).view(b, n, h, d // h).transpose(1, 2)
        kv = self.kv(self.norm1(x)).view(b, x.shape[1], 2, h, d // h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(qh, kv[0], kv[1]).transpose(1, 2).reshape(b, n, d)
        q = q + self.proj(y)
        return q + self.mlp(self.norm2(q))


#: Each paper's probe (module docstring). The mode fixes it, so a label cannot pair a paper with the other's probe.
PROBE_FOR_MODE = {"ijepa": "mean_linear", "vjepa": "attentive"}


class JEPAEncoder(HWCFeatureExtractor):
    """The JEPA context encoder; its EMA copy is the target encoder (brain/aux/jepa_aux.py)."""

    #: The builder leaves this encoder's init alone (models/utils/init.py): the reference trunc_normal.
    OWNS_INIT = True

    def __init__(
        self,
        observation_space: gym.Space,
        num_frames: int,
        jepa: dict,
        probe: str,
        tubelet: int = 1,
        features_dim: int = 128,
        patch_size: int = 14,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
    ) -> None:
        # Explicit keywords only (no ``**_``): a cfg key this encoder does not read refuses here.
        super().__init__(observation_space, features_dim)
        total_c, height, width = image_channels_hw(observation_space)
        self.num_frames = int(num_frames)
        if self.num_frames == 1:
            if total_c != 3:
                raise ValueError(f"I-JEPA (num_frames=1) needs one RGB frame; got {total_c} channels. "
                                 f"Run it with framestack=False.")
        else:
            validate_framestack_depth(total_c, self.num_frames, "JEPAEncoder(V-JEPA)", channels_per_frame=3)
        if self.num_frames % int(tubelet):
            raise ValueError(f"tubelet={tubelet} does not divide num_frames={self.num_frames}")
        if height % patch_size or width % patch_size:
            raise ValueError(f"patch_size={patch_size} does not divide the {height}x{width} eye")
        unknown = set(jepa) - JEPA_OBJECTIVE_KEYS
        missing = JEPA_OBJECTIVE_KEYS - set(jepa)
        if unknown or missing:
            raise ValueError(f"JEPAEncoder jepa cfg: unknown {sorted(unknown)}, missing {sorted(missing)}")
        want = "ijepa" if self.num_frames == 1 else "vjepa"
        if jepa["mode"] != want:
            raise ValueError(f"JEPAEncoder(num_frames={self.num_frames}) is {want}; jepa mode={jepa['mode']!r}")
        if probe != PROBE_FOR_MODE[want]:
            raise ValueError(f"{want}'s paper probe is {PROBE_FOR_MODE[want]!r}; got probe={probe!r}")
        self.probe = probe
        self.tubelet = int(tubelet)
        self.n_t = self.num_frames // self.tubelet
        self.jepa = dict(jepa)
        self.patch = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.n_h, self.n_w = height // patch_size, width // patch_size
        self.n_spatial = self.n_h * self.n_w
        self.n_tokens = self.n_t * self.n_spatial
        if self.num_frames == 1:
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            pos = sincos_2d(embed_dim, self.n_h, self.n_w)
        else:
            k = (self.tubelet, patch_size, patch_size)
            self.patch_embed = nn.Conv3d(3, embed_dim, kernel_size=k, stride=k)
            pos = sincos_3d(embed_dim, self.n_t, self.n_h, self.n_w)
        # A buffer, not a Parameter: fixed, so it is neither trained nor counted.
        self.register_buffer("pos_embed", pos.unsqueeze(0), persistent=False)
        self.blocks = nn.ModuleList(JEPABlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(embed_dim)
        if probe == "attentive":
            self.probe_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.probe_block = CrossAttentionBlock(embed_dim, num_heads, mlp_ratio)
        self.head = nn.Linear(embed_dim, features_dim)
        self.apply(init_jepa_weights)
        rescale_blocks(self.blocks)
        if probe == "attentive":
            # AttentivePooler.__init__: trunc_normal query, and its one block rescaled at layer_id 1.
            nn.init.trunc_normal_(self.probe_query, std=0.02)
            with torch.no_grad():
                self.probe_block.proj.weight.div_(math.sqrt(2.0))
                self.probe_block.mlp[2].weight.div_(math.sqrt(2.0))

    # ---------------------------------------------------------------- trunk
    def _embed(self, prepared: torch.Tensor) -> torch.Tensor:
        """(B, C*T, H, W) prepared floats -> (B, N, D) patch tokens + positions, row-major over the grid."""
        if self.num_frames == 1:
            x = self.patch_embed(prepared)
        else:
            b, _, h, w = prepared.shape
            # The stack is oldest-first on the channel axis (framestack.py) -> (B, C, T, H, W).
            x = prepared.view(b, self.num_frames, 3, h, w).transpose(1, 2)
            x = self.patch_embed(x)                                # (B, D, n_t, n_h, n_w): t-major tokens
        return x.flatten(2).transpose(1, 2) + self.pos_embed.to(x.dtype)

    def tokens_prepared(self, prepared: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """Post-norm tokens. ``keep`` (B, K) LongTensor: only those patches enter the blocks (context
        encoder of both papers); positions are added BEFORE the gather, so each keeps its own."""
        x = self._embed(prepared)
        if keep is not None:
            x = torch.gather(x, 1, keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def encode_tokens(self, observations: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """(B, N, D) post-norm tokens and the (n_h, n_w) grid (aux/token_features.spatial_tokens)."""
        return self.tokens_prepared(self._prepare_image(observations)), (self.n_h, self.n_w)

    def _readout(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.probe == "mean_linear":
            return self.head(tokens.mean(dim=1))
        q = self.probe_query.to(tokens.dtype).expand(tokens.shape[0], -1, -1)
        return self.head(self.probe_block(q, tokens).squeeze(1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self._prepare_image(observations)
        amp = _amp_dtype()
        if amp is not None and x.is_cuda:
            if amp is torch.bfloat16:
                from .matched import warn_if_bf16_without_tensor_cores
                warn_if_bf16_without_tensor_cores(x.device, "NETT_AMP")
            with torch.autocast("cuda", dtype=amp):
                out = self._readout(self.tokens_prepared(x))
            return out.float()
        return self._readout(self.tokens_prepared(x))
