"""XSP -- cross-stream predictive learning: a siamese conv trunk feeding a ventral (form) and a
dorsal (motion) stream, trained end-to-end by predicting the next frame.

Owner design (2026-10-09, researcher2 relay), aligned with the paper's Methods and the owner's
figure the same day; label "XSP" in examples/campaign_train.py. Methods (verbatim): "The shared
encoder phi_e maps each frame x_t to a feature map, z_t = phi_e(x_t). The ventral pathway
processes the current feature map and applies a two-way softmax at each location,
v_t = softmax(phi_v(z_t)). The dorsal pathway processes successive feature maps,
d_t = phi_d(z_{t-1}, z_t). Their outputs are combined multiplicatively and passed to the predictor
alongside the current frame, x^_{t+1} = g(x_t, v_t (.) d_t)."

```
x_{t-T+1} .. x_t = the T frames of the T-major stack, oldest .. newest   (T = NETT_FRAMESTACK_N >= 2)
z_s  = phi_e(x_s)        s = t-T+1 .. t   SIAMESE: one conv trunk, the same weights, every frame
m_t  = sigmoid(phi_v(z_t))                (B, 1, h, w): the FIGURE map; ground = 1 - m_t
       [m_t, 1 - m_t] = softmax([l, 0]) = the Methods' two-way softmax v_t (one redundant logit
       removed: a softmax is unchanged by adding a constant to both logits)
d_t  = phi_d(cat(z_{t-T+1}, .., z_t))     (B, K, h, w): motion field over the stack, linear out
x^_{t+1} = g(x_t, v_t (.) d_t)            g lives in the aux head (brain/aux/xsp_aux.py)
policy features (owner ruling "option (b)"):  ReLU(Linear(flatten(pool(z_t * m_t))))
```

STACK DEPTH: T = 2 IS THE METHODS EXACTLY (d_t = phi_d(z_{t-1}, z_t), "successive feature maps").
T = 3 is the owner's FIGURE, which draws the dorsal stream over a stack of frames; phi_d's first
conv simply widens to T * conv_dim inputs. T = 2 is the default (NETT_FRAMESTACK_N unset).

The figure: TOP, the current frame -> shared encoder -> ventral (3 conv maps) -> sigmoid -> predict
next frame; BOTTOM, the frame stack over time -> the SAME shared encoder -> dorsal (3 conv maps) ->
predict next frame; MIDDLE, the current frame goes straight into the prediction, whose error
against the next frame is the one loss.

This module holds phi_e, phi_v, phi_d and the policy readout, so all three streams are in the
policy checkpoint and in PPO's optimizer. The ventral stream and the policy read ONLY x_t; the
older frames reach nothing but phi_d. The RL gradient reaches phi_e and phi_v (the readout uses
z_t and m_t); phi_d is reached ONLY by the predictive loss, because the policy never reads it.
That asymmetry is the design: the policy reads FORM, gated by a figure map that motion
prediction had to make useful.

DESIGN CHOICES (each stated, none silent):

* T is the observation's channel count / channels-per-frame, so the dorsal input width is
  T * conv_dim and T = 2 and T = 3 both build. ``num_frames`` (the label passes
  campaign_train's _FRAMESTACK_N, the value the FrameStack wrapper also reads) is checked
  EXACTLY against it with temporal.validate_framestack_depth: a stack depth or a
  channels-per-frame that disagrees refuses to build instead of scrambling time into colour.
  T < 2 is refused: one frame has no motion.
* phi_e runs ONCE on the batch-concatenation of all T frames (siamese by construction).
* phi_e is NatureCNN's three convolutions (8/4, 4/2, 3/1; 32, 64, conv_dim channels) WITH
  PADDING (2, 1, 1). Unpadded, the 80x128 eye gives a 6x12 map whose cells are centred 17.5 px
  in from each border, so a figure map upsampled onto the image is misregistered by up to ~11 px
  and the outermost ~17 px have no cell centred on them at all. Padded, the map is exactly
  (H/8, W/8) = 10x16 on an 8-px grid, so every decoder upsampling stage is an integer 2x and a
  map cell IS an 8x8 image block. conv_dim defaults to 75, the "CNN" label's.
* The policy readout pools z_t * m_t with ``DeterministicAvgPool2d(pool_grid_for(h, w))`` --
  the SAME rule and conv_dim the "CNN" label uses -- so the readout Linear is capacity-matched
  to that label (2x8 = 16 cells x 75 = 1200 -> 512, against CNN's 3x6 = 18 x 75 = 1350). The
  unpooled flatten (10x16x75 = 12000 -> 512, 6.1M parameters) is refused on the evidence in
  nature_cnn.py's docstring: an oversized unpooled readout cost training consistency there.
  The gate is applied BEFORE the pool, so a cell the figure map rejects contributes nothing.
* ``policy_input="map"`` is the knob's alternative: the raw figure map alone,
  ReLU(Linear(flatten(m_t))) (10x16 = 160 -> 512) -- the policy then sees WHERE the figure is
  and nothing about what it looks like.
* ``policy_input="both"``: pool(z_t * m_t) and pool(z_t * (1 - m_t)) concatenated (2400 -> 512),
  figure and ground. A readout that does not depend on which side of the sigmoid the object
  lands: swapping m <-> 1 - m permutes its two halves. Under NETT_XSP_COMBINE=outer (the
  default, = the Methods' v_t (.) d_t) the loss is symmetric under m <-> 1 - m, so nothing
  decides that the object is the HIGH side: on the synthetic moving square it landed high in
  3/10 seeds at T=2 and 6/10 at T=3 (tests/test_xsp.py section 4), and "gated" then gates on
  the background. Under NETT_XSP_COMBINE=fg it landed high in 20/20. Its effect on the POLICY
  is untested (no RL run exists).
* phi_v / phi_d are two 3x3 conv layers and a 1x1 output. Ventral width 32 -> 1 logit, dorsal
  width 64 (it reads T concatenated maps), ``dorsal_dim`` (K) = 8 motion channels.
* No autocast here (NatureCNN's NETT_AMP bf16 path is NOT copied): this is a new label with no
  replay record to keep, and fp32 keeps the sigmoid and the predictive loss exact. Cost: slower
  on GPU than the CNN label's bf16 forward.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor
from .utils.pool import DeterministicAvgPool2d, pool_grid_for
from .utils.temporal import validate_framestack_depth

#: The ``policy_input`` values (owner ruling (1) and its alternatives). Index 0 is the default.
POLICY_INPUTS = ("gated", "map", "both")


def _conv_stack(in_ch: int, width: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(),
        nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
        nn.Conv2d(width, out_ch, 1),
    )


class XSPEncoder(HWCFeatureExtractor):
    """Two-stream (ventral/dorsal) encoder over a siamese per-frame trunk. See module docstring."""

    MIN_FRAMES = 2

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        conv_dim: int = 75,
        ventral_dim: int = 32,
        dorsal_width: int = 64,
        dorsal_dim: int = 8,
        policy_input: str = "gated",
        num_frames: int | None = None,
        channels_per_frame: int | None = None,
        **_,
    ):
        super().__init__(observation_space, features_dim)
        if policy_input not in POLICY_INPUTS:
            raise ValueError(f"XSPEncoder policy_input={policy_input!r} is not one of {list(POLICY_INPUTS)}.")
        for name, value in (("conv_dim", conv_dim), ("ventral_dim", ventral_dim),
                            ("dorsal_width", dorsal_width), ("dorsal_dim", dorsal_dim)):
            if type(value) is not int or value < 1:
                raise ValueError(f"XSPEncoder {name} must be a positive int; got {value!r}.")
        from ..aux.cltt_views import resolve_channels_per_frame
        self.cpf = resolve_channels_per_frame(channels_per_frame)
        channels, height, width = image_channels_hw(observation_space)
        if channels % self.cpf or channels // self.cpf < self.MIN_FRAMES:
            raise ValueError(
                f"XSPEncoder needs a frame stack of >= {self.MIN_FRAMES} frames of {self.cpf} channels "
                f"(framestack=True); the observation has {channels} channels. The dorsal stream "
                f"reads motion across the stack, and a single frame has none -- refusing rather "
                f"than comparing a frame with itself.")
        if num_frames is not None:
            validate_framestack_depth(channels, num_frames, type(self).__name__, channels_per_frame=self.cpf)
        self.n_frames = channels // self.cpf
        if height % 8 or width % 8:
            raise ValueError(f"XSPEncoder needs an eye divisible by 8 (the trunk's stride); got {height}x{width}.")
        self.policy_input = policy_input
        self.dorsal_dim = dorsal_dim

        # phi_e -- applied PER FRAME with these weights (siamese).
        self.trunk = nn.Sequential(
            nn.Conv2d(self.cpf, 32, kernel_size=8, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, conv_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        self.map_hw = (height // 8, width // 8)
        with torch.no_grad():
            probe = self.trunk(torch.zeros(1, self.cpf, height, width))
        if tuple(probe.shape[-2:]) != self.map_hw:      # measured, not derived (see nature_cnn.py)
            raise RuntimeError(f"XSP trunk map {tuple(probe.shape[-2:])} != expected {self.map_hw}")
        self.conv_dim = conv_dim
        # phi_v -- ventral (form): z_t -> ONE logit per location; m_t = sigmoid(logit).
        self.ventral = _conv_stack(conv_dim, ventral_dim, 1)
        # phi_d -- dorsal (motion): cat(z_{t-T+1}, .., z_t) -> K-channel motion field. Aux-only gradient.
        self.dorsal = _conv_stack(self.n_frames * conv_dim, dorsal_width, dorsal_dim)

        if policy_input in ("gated", "both"):
            self.pool_grid = pool_grid_for(*self.map_hw)
            self.readout_pool = DeterministicAvgPool2d(self.pool_grid)
            n_in = conv_dim * self.pool_grid[0] * self.pool_grid[1] * (2 if policy_input == "both" else 1)
        else:
            self.pool_grid = None
            self.readout_pool = None
            n_in = self.map_hw[0] * self.map_hw[1]
        self.linear = nn.Sequential(nn.Linear(n_in, features_dim), nn.ReLU())

    # ------------------------------------------------------------------ streams
    def split_frames(self, prepared: torch.Tensor) -> list[torch.Tensor]:
        """[x_{t-T+1}, .., x_t]: the T frames of a prepared T-major stack, oldest .. newest."""
        c, t = self.cpf, self.n_frames
        if prepared.shape[1] != c * t:
            raise ValueError(f"XSP was built for {t} stacked {c}-channel frames ({c * t} channels); "
                             f"got {prepared.shape[1]} channels.")
        return [prepared[:, k * c:(k + 1) * c] for k in range(t)]

    def current_frame(self, prepared: torch.Tensor) -> torch.Tensor:
        """x_t: the NEWEST frame -- the only frame the ventral stream and the policy read."""
        return self.split_frames(prepared)[-1]

    def ventral_map(self, z: torch.Tensor) -> torch.Tensor:
        """m = sigmoid(phi_v(z)): (B, 1, h, w), the figure map; the ground map is 1 - m."""
        return torch.sigmoid(self.ventral(z))

    def encode_streams(self, prepared: torch.Tensor) -> dict[str, torch.Tensor]:
        """All three streams on an already-PREPARED (B, C*T, H, W) image, gradients on.

        phi_e runs ONCE on the batch-concatenation of all T frames, so every frame goes through
        literally the same module call (siamese by construction, one kernel launch).
        ``z_frames`` is (B, T, conv_dim, h, w), oldest .. newest; ``z_t`` is its last slice.
        """
        frames = self.split_frames(prepared)
        b, t = prepared.shape[0], self.n_frames
        z_all = self.trunk(torch.cat(frames, dim=0))                     # (T*B, C, h, w), frame-major
        z_frames = z_all.view(t, b, *z_all.shape[1:]).transpose(0, 1)    # (B, T, C, h, w)
        z_t = z_frames[:, -1]
        m = self.ventral_map(z_t)
        d = self.dorsal(z_frames.flatten(1, 2))                          # channels: oldest .. newest
        return {"x_t": frames[-1], "z_frames": z_frames, "z_t": z_t, "m": m, "d": d}

    def readout(self, z_t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        if self.policy_input == "gated":
            h = self.readout_pool(z_t * m)
        elif self.policy_input == "both":
            h = torch.cat([self.readout_pool(z_t * m), self.readout_pool(z_t * (1.0 - m))], dim=1)
        else:
            h = m
        return self.linear(h.flatten(1))

    # ------------------------------------------------------------------ policy path
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Policy features. Only x_t is encoded: the policy reads the ventral stream, not phi_d."""
        z_t = self.trunk(self.current_frame(self._prepare_image(observations)))
        return self.readout(z_t, self.ventral_map(z_t))
