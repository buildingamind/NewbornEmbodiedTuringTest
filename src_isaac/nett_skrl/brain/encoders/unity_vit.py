"""Repo A's ORIGINAL Unity ViT encoder, vendored and run under the Isaac body.

This is not a new architecture. It is the encoder that `src/nett/brain/encoders/vit.py` builds on
the Unity branch (`origin/dev`), brought over so the fleet can measure it on the Isaac standard
instead of arguing about it from the source. Its value is as a REFERENCE POINT: every ViT number
this fleet has published came from ``compact_vit``, which was written here; nobody has run the
one the upstream project actually ships.

## Provenance, and why the code is copied rather than imported

`vit.py` builds ``LitClassifier(configuration)`` from ``disembodied_models/vit_contrastive.py``,
which imports ``vit_pytorch`` and ``lightning``. NEITHER IS INSTALLED IN THIS FLEET'S VENV, and
installing them is not available: the venv at /home/zlaborde/code/.venv/nett-private is shared with
running arms, and adding a dependency to it mutates the environment under live processes. So the
ONE class on the forward path -- ``vit_pytorch.simple_vit.SimpleViT`` -- is copied below verbatim.

    upstream       vit_pytorch 1.26.6, vit_pytorch/simple_vit.py
    ⚠ repo A pins `lightning==2.2.5` but leaves `vit-pytorch` UNPINNED (pyproject.toml:27), so
      "the ViT Unity ran" is not a single artefact -- it is whatever release was current on the
      day someone installed. This file fixes that moving target at one version and says which.

`einops` (0.8.2) IS present, so the two ``Rearrange``/``rearrange`` calls are the upstream ones.

## What `vit.py` does that this reproduces -- including the parts that do nothing

Three properties of the upstream wrapper are PRESERVED ON PURPOSE. They look like bugs; they are
what actually ran, and silently repairing them would make this arm a measurement of a model Unity
never trained:

  1. ``init_weights()`` -- the Xavier-uniform init -- is defined TWICE in vit_contrastive.py and
     CALLED NOWHERE. Every weight here therefore keeps SimpleViT's own default init. ⛔ DO NOT
     "fix" this by calling it; the fix would be the deviation.
  2. ``self.model.fc = nn.Identity()`` (vit.py:52) assigns to the LightningModule, which has no
     ``fc`` and never reads one. It is a no-op, so the SimpleViT's ``linear_head`` survives and
     the output width is ``num_classes`` -- i.e. ``features_dim``. Reproduced by simply not
     having an ``fc``.
  3. ``LitClassifier`` also builds a 512->512->128 SimCLR ``Projection`` head. ``forward()``
     never touches it, but under SB3 its parameters still entered the policy optimiser.

⛔ DEVIATION 1, DELIBERATE: the ``Projection`` head is NOT carried over. It would add 329,216
parameters that receive zero gradient, and this wave's entire claim structure is capacity-matched
-- an arm carrying 40% dead weight cannot be compared to one that does not. The omission changes
no output value, only the parameter count, and this note is the record of it.

⛔ DEVIATION 2, FORCED: ``vit.py:37`` passes ``image_size=observation_space.shape[1]`` -- ONE
scalar, which ``pair()`` expands to a square. Unity's eye was square; ours is 80 high x 128 wide.
A scalar is not merely unfaithful here, it is unbuildable: 80 would make the patch grid assert on
a 128-wide input. This passes the true ``(H, W)``. There is no faithful alternative.

## What the defaults cost

``vit.py``'s defaults are ``patch_size=4, dim=64, depth=3, heads=3, mlp_dim=128``. On an 80x128
eye patch 4 gives 20x32 = 640 TOKENS, against 40 for this fleet's patch-16 standard. Attention is
quadratic in tokens, so the attention matrix is 640^2 = 409,600 entries per head per layer versus
1,600 -- ~256x. ⚠ THE PARAMETER COUNT DOES NOT SHOW THIS: at dim=64 the model is small, and its
cost is entirely in activations and attention compute. Budget wall-clock accordingly.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor

# ======================================================================================
# BEGIN VERBATIM COPY -- vit_pytorch 1.26.6, vit_pytorch/simple_vit.py (MIT, lucidrains)
# ⛔ Do not refactor, rename or "improve" anything between these markers. Its only job is to be
# bit-for-bit the upstream module; any edit here silently redefines what the arm measures.
# ======================================================================================


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature**omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([Attention(dim, heads=heads, dim_head=dim_head),
                               FeedForward(dim, mlp_dim)])
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 channels=3, dim_head=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        self.patch_size = patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            "Image dimensions must be divisible by the patch size."

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        x = self.transformer(x)
        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.linear_head(x)


# ======================================================================================
# END VERBATIM COPY
# ======================================================================================


class UnityViT(HWCFeatureExtractor):
    """`src/nett/brain/encoders/vit.py` from repo A's Unity branch, on the Isaac body.

    Keyword defaults are ``vit.py``'s own, under ``vit.py``'s own names, so an arm config reads
    the same as the upstream call.

    ⚠ ``pos_embedding`` is a PLAIN ATTRIBUTE upstream, not a registered buffer: it is absent from
    ``state_dict()`` and ``.to(device)`` on the module does not move it, which is why ``forward``
    copies it to the input's device on every call. Both facts are upstream's; it is deterministic
    from the config, so the checkpoint omission loses nothing.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        patch_size: int = 4,
        depth: int = 3,
        heads: int = 3,
        intermediate_size: int = 128,
        hidden_size: int = 64,
    ) -> None:
        super().__init__(observation_space, features_dim)
        channels, height, width = image_channels_hw(observation_space)
        self.n_input_channels = int(channels)
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"UnityViT: the {height}x{width} eye is not divisible by patch_size={patch_size}."
            )
        self.model = SimpleViT(
            image_size=(height, width),      # ⛔ DEVIATION 2, see module docstring
            patch_size=patch_size,
            num_classes=features_dim,
            dim=hidden_size,
            depth=depth,
            heads=heads,
            mlp_dim=intermediate_size,
            channels=self.n_input_channels,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(self._prepare_image(observations))
