"""ViT-SlotRecon encoder: a compact ViT trunk + Slot Attention over frame t; the policy reads the detached slots.

Owner, workspace DECISIONS §162 (item 4, as changed by the owner: "the encoder [should] receive frame t and the
loss calculated against frame t+1 ... similar to what we did with the XSP2 loss"). Label "ViT-SlotRecon", aux
`slot_recon` (brain/aux/slot_recon_aux.py), which trains EVERY parameter here with its own Adam.

```
obs    = [x_{t-1}, x_t]                           framestack 2, T-major (observation.py:96); ONLY x_t is read
tokens = CompactViT._trunk(x_t)[:, 1:]            VIT_CFG's trunk (patch 16, 144 wide, depth 3, 4 heads) on ONE
                                                  RGB frame: (B, 40, 144) post-norm patch tokens at 80x128
f      = MLP(LayerNorm(tokens))                   144 -> 64 ReLU -> 64, the paper's per-location encoder MLP
slots  = SlotAttention(f)                         3 slots x 64, 3 iterations (Locatello et al. 2020, Alg. 1)
policy = flatten(slots).detach()                  (B, 192) = features_dim, under no_grad, fixed slot noise
```

⛔ DETACHED BY CONSTRUCTION, AS XSP2 (encoders/xsp2.py forward). The policy features are computed under
``torch.no_grad`` from x_t alone, so no reward gradient can reach the trunk or the slots and no autograd graph
references their weights -- which is what makes it safe for the aux to step them in place between PPO's forward
and PPO's backward inside one minibatch. No projection to 512: a PPO-trained Linear here would be an RL-shaped
layer inside the encoder; the per-head MLP trunk after the encoder already reads the 192 features (XSP2 reads
its 160-wide map the same way). NOT NETT_DECOUPLE_ENCODER: that is an env knob, and MODELS has no per-model env
field, so a row that forgot it would train this label with reward reaching the slots.

⛔ THE POLICY'S SLOT NOISE IS FIXED. Slot Attention draws its initial slots from a learned Gaussian. skrl rolls
out in eval() and updates in train(): a mode-following draw would give the update different features from the
rollout for the same observation (a PPO ratio that moves without the policy changing) and would consume the
global RNG inside the policy forward. The policy path always reads SlotAttention's persistent ``eval_noise``
(one sample of the same N(0, I), saved with the weights); the aux path draws fresh noise, as the paper trains.

SLOT DIM 64. The paper's value (Table 11; slot_attention_ae.SlotAttention's default), and the right side of the
bottleneck: three 64-d slots (192) summarise 40 tokens x 144 (5,760), so the decoder must reconstruct the next
frame through a 30x narrower code -- the pressure that makes slots bind to parts of the scene. Wider slots
(e.g. 144, the token width) would let 3 slots carry most of the token grid and weaken that pressure; narrower
would drop below the paper's validated setting for no stated reason.

DECLARED: the trunk's CLS token is kept (it attends with the patches, as in every CompactViT) and not read; the
fleet's orthogonal init applies to this encoder as to every ViT label (no OWNS_INIT), so the trunk starts as
ViT-CLTT-Ref's does; NETT_AMP is refused (the aux steps its own optimizer outside any GradScaler, and a bf16
setting would never reach ``_trunk``, which this encoder calls directly).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw
from ..aux.slot_attention_ae import SlotAttention
from .compact_vit import CompactViT, vit_amp_dtype
from .hwc_feature_extractor import HWCFeatureExtractor
from .utils.temporal import validate_framestack_depth


class ViTSlotRecon(HWCFeatureExtractor):
    """CompactViT trunk + 3-slot Slot Attention on the newest frame; detached slots are the policy features."""

    N_FRAMES = 2

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 192,
        patch_size: int = 16,
        embed_dim: int = 144,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        num_slots: int = 3,
        slot_dim: int = 64,
        slot_iters: int = 3,
        slot_mlp_hidden: int = 128,
        num_frames: int = 2,
        channels_per_frame: int = 3,
    ) -> None:
        super().__init__(observation_space, features_dim)
        channels, height, width = image_channels_hw(observation_space)
        if int(num_frames) != self.N_FRAMES:
            raise ValueError(f"ViTSlotRecon reads obs = [x_(t-1), x_t] (framestack 2); num_frames={num_frames}. "
                             f"Set NETT_FRAMESTACK_N=2.")
        validate_framestack_depth(channels, num_frames, type(self).__name__, channels_per_frame)
        if features_dim != num_slots * slot_dim:
            raise ValueError(f"ViTSlotRecon's policy features ARE the flattened {num_slots} x {slot_dim} slots, so "
                             f"features_dim must be {num_slots * slot_dim}; got {features_dim}.")
        if vit_amp_dtype() is not None:
            raise ValueError("ViTSlotRecon runs fp32: its aux steps its own optimizer outside any GradScaler and "
                             "the trunk is called below CompactViT's autocast wrappers. Unset NETT_AMP.")
        self.cpf = int(channels_per_frame)
        self.image_hw = (height, width)
        self.num_slots, self.slot_dim = int(num_slots), int(slot_dim)
        frame_space = gym.spaces.Box(low=0, high=255, shape=(self.cpf, height, width), dtype=np.uint8)
        # features_dim = embed_dim makes CompactViT's head an Identity: no parameter here goes unread.
        self.vit = CompactViT(frame_space, features_dim=embed_dim, patch_size=patch_size, embed_dim=embed_dim,
                              depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, pool="cls", stem="linear")
        self.token_grid = (self.vit._n_h, self.vit._n_w)
        self.patch_size = int(patch_size)
        self.slot_in = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, slot_dim), nn.ReLU(),
                                     nn.Linear(slot_dim, slot_dim))
        self.slot_attention = SlotAttention(num_slots, dim=slot_dim, iters=slot_iters, mlp_hidden=slot_mlp_hidden)

    # ------------------------------------------------------------------ parameter groups
    def stream_parameters(self) -> list[nn.Parameter]:
        """Trunk + slot MLP + Slot Attention: every parameter of this encoder. Trained ONLY by slot_recon."""
        return [*self.vit.parameters(), *self.slot_in.parameters(), *self.slot_attention.parameters()]

    # ------------------------------------------------------------------ slots
    def current_frame(self, prepared: torch.Tensor) -> torch.Tensor:
        """x_t: the NEWEST frame of a prepared T-major stack -- the only frame this encoder reads."""
        if prepared.shape[1] != self.cpf * self.N_FRAMES:
            raise ValueError(f"ViTSlotRecon was built for {self.N_FRAMES} stacked {self.cpf}-channel frames; "
                             f"got {prepared.shape[1]} channels.")
        return prepared[:, -self.cpf:]

    def encode_slots(self, frame: torch.Tensor, sample: bool) -> torch.Tensor:
        """(B, K, D) slots of ONE prepared frame (B, cpf, H, W), gradients on. ``sample``: fresh slot noise
        (the aux) or the fixed ``eval_noise`` (the policy)."""
        tokens = self.vit._trunk(frame)[:, 1:]                       # (B, N, embed_dim), CLS dropped
        return self.slot_attention(self.slot_in(tokens), sample=sample)

    # ------------------------------------------------------------------ policy path
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Policy features: flatten(slots of x_t), (B, K*D), under no_grad, fixed slot noise. See the module doc."""
        with torch.no_grad():
            x_t = self.current_frame(self._prepare_image(observations))
            slots = self.encode_slots(x_t, sample=False)
        return slots.flatten(1).detach()
