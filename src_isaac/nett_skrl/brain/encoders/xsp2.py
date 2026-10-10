"""XSP2 -- cross-stream predictive learning, owner spec of workspace DECISIONS §87 (2026-10-10).

Label "XSP2" in examples/campaign_train.py; aux brain/aux/xsp2_aux.py. A NEW label beside "XSP"
(encoders/xsp.py), which is unchanged. What differs from XSP, all by owner ruling:

* the POLICY READS THE FIGURE MAP ITSELF: the ventral sigmoid map m_t at its native 10x16
  resolution, flattened (160 numbers) and DETACHED, IS the encoder's output (owner: "we can just
  flatten the 1 channel map into a vector and feed that into the policy"). No readout Linear: the
  160 values go straight into the standard policy/value MLP, so ``features_dim`` = 160.
* the encoder is DETACHED FROM THE REWARD: no RL gradient reaches phi_e, phi_v or phi_d. They are
  trained ONLY by the next-frame prediction loss, with their OWN Adam, and are excluded from
  PPO's optimizer (brain/aux/xsp2_aux.py, ``XSP2Term.bind_agent``).
* the dorsal stream is a Conv3D over the 2-frame stack, the predictor works at the FULL 80x128
  eye, and the loss is a SUM of squares (xsp2_aux.py).

```
x_{t-1}, x_t   = the 2 frames of the T-major stack (T = 2 exactly; cpf = 3, RGB in [0, 1])
z_s  = phi_e(x_s)       s = t-1, t   SIAMESE: ONE module call on the batch-concatenation
                        phi_e = NatureCNN's 3 convs (8/4, 4/2, 3/1; 32, 64, 75) WITH padding
                        (2, 1, 1): the 128x80 (W x H) eye gives exactly (H/8, W/8) = 10x16
m_t  = sigmoid(phi_v(z_t))           (B, 1, 10, 16)  phi_v: 3x Conv2d 3x3 (75->32->32->1), ReLU between
d_t  = phi_d(z_{t-1}, z_t)           (B, 2, 10, 16)  phi_d: Conv3d (2,3,3) over the depth-2 volume
                                                     (no temporal padding -> depth 1), then
                                                     2x Conv2d 3x3 (32->32->2); ReLU between; LINEAR out
c_t  = up8(m_t * d_t)                (B, 2, 80, 128) the product AT THE 10x16 GRID, then NEAREST 8x
policy features = flatten(m_t).detach()   (B, 160)   -> the standard policy/value MLP
```

d_t is a motion vector field in PIXEL units of the 80x128 eye: channel 0 = x (width)
displacement, channel 1 = y (height). Nothing but the predictor's warp convention (xsp2_aux.py)
gives it that meaning.

EPISODE START. The 2-frame stack at an episode's first step is [x_0, x_0]:
nett_skrl/body/wrappers/framestack.py ``FrameStack.reset`` fills the deque with the reset
observation repeated n_stack times, and ``FrameStack.step`` overwrites EVERY stacked frame of an
env whose episode ended with that env's new first observation (the per-env scrub, made reliable
on Isaac's CUDA done flags by the 2026-09-27 ``_done_mask`` fix). So d_t at t = 0 sees two equal
feature maps: no motion, by construction, never a frame from the previous episode.

UPSAMPLING. Owner default "output": both streams run at 10x16, the product m*d is formed at
10x16 and only the 2-channel product is upsampled, NEAREST, for the predictor. The upsample is an
``expand`` + ``reshape`` (``upsample_nearest``): bitwise the same values as
``F.interpolate(mode="nearest")`` at an integer factor (tests/test_xsp2.py), and its backward is a
plain sum, deterministic on CUDA by construction. The owner's per-layer alternative ("layer") was
dropped from the run plan before it was built (DECISIONS §87, spec change 2) and is NOT
implemented; there is therefore no NETT_XSP2_UPSAMPLE / NETT_XSP2_POOL knob.

INITIALISATION (owner, spec change 2). The encoder OWNS its init (``OWNS_INIT = True``):
models/builder.py and models/utils/backbone.py skip the fleet's orthogonal_init for it
(models/utils/init.py ``encoder_owns_init``), for this encoder only.
* phi_e (the 3 trunk convs): Kaiming He, ``kaiming_normal_(mode="fan_in", nonlinearity="relu")``,
  bias 0.
* phi_v and phi_d (Conv3d included): PyTorch's default init, whatever nn.Conv2d / nn.Conv3d
  construct with, drawn from the brain's seeded RNG (agent_factory seeds before the build).
* There is no readout. The policy/value MLP and heads keep the fleet's init (ModelCfg
  orthogonal_init / hidden_gain / output_gain) -- that choice lives in ModelCfg alone, and the
  exemption above does not reach it.

INFERENCE. ``forward`` runs phi_e on x_t and phi_v, under ``torch.no_grad()``: x_{t-1}, phi_d
and the predictor never run on the policy path (tests/test_xsp2.py counts calls).

DESIGN CHOICES the spec left open (each stated):
* phi_d's width is 32, the ventral width (the spec gives "~32" for phi_v and "same depth" for
  phi_d).
* phi_v's output conv is 3x3 like its other two (the spec: "3 Conv2D layers (3x3 ...)").
* ``features_dim`` must be passed and must EQUAL the map size (160 at the 128x80 eye). The
  campaign cfg states 160; another eye refuses at construction, naming the value it needs,
  rather than resizing anything silently.
* fp32 throughout, no autocast (the aux step also disables it).
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...body.observation import image_channels_hw
from .hwc_feature_extractor import HWCFeatureExtractor
from .utils.temporal import validate_framestack_depth

#: The trunk's total stride: the stream grid is (H/8, W/8).
TRUNK_STRIDE = 8


def upsample_nearest(x: torch.Tensor, factor: int) -> torch.Tensor:
    """(B, C, h, w) -> (B, C, h*factor, w*factor), nearest neighbour, by expand + reshape.

    Equal, value for value, to ``F.interpolate(x, scale_factor=factor, mode="nearest")``; the
    backward is the sum over each factor x factor block, which has no nondeterministic kernel.
    """
    b, c, h, w = x.shape
    return x[:, :, :, None, :, None].expand(b, c, h, factor, w, factor).reshape(b, c, h * factor, w * factor)


class XSP2Dorsal(nn.Module):
    """phi_d: Conv3d (2,3,3) over the depth-2 volume (z_{t-1}, z_t), then 2 Conv2d 3x3; linear out."""

    def __init__(self, in_ch: int, width: int, out_ch: int = 2) -> None:
        super().__init__()
        # Depth axis = time, oldest first; kernel depth 2 with no temporal padding -> depth 1.
        self.conv3d = nn.Conv3d(in_ch, width, kernel_size=(2, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.conv3 = nn.Conv2d(width, out_ch, 3, padding=1)

    def forward(self, z_prev: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        h = self.conv3d(torch.stack([z_prev, z_t], dim=2))              # (B, width, 1, h, w)
        if h.shape[2] != 1:
            raise RuntimeError(f"XSP2 dorsal Conv3d left temporal depth {h.shape[2]}, expected 1")
        h = F.relu(h[:, :, 0])
        h = F.relu(self.conv2(h))
        return self.conv3(h)


class XSP2Encoder(HWCFeatureExtractor):
    """Siamese trunk + ventral/dorsal streams; the policy reads the detached figure map only."""

    N_FRAMES = 2
    CPF = 3
    #: Read by models/utils/init.py ``encoder_owns_init``: the builder and the backbone skip the
    #: fleet's orthogonal_init for this encoder (owner, DECISIONS §87 spec change 2).
    OWNS_INIT = True

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 160,
        conv_dim: int = 75,
        ventral_width: int = 32,
        dorsal_width: int = 32,
        num_frames: int | None = None,
        channels_per_frame: int | None = None,
    ):
        super().__init__(observation_space, features_dim)
        for name, value in (("conv_dim", conv_dim), ("ventral_width", ventral_width),
                            ("dorsal_width", dorsal_width), ("features_dim", features_dim)):
            if type(value) is not int or value < 1:
                raise ValueError(f"XSP2Encoder {name} must be a positive int; got {value!r}.")
        from ..aux.cltt_views import resolve_channels_per_frame
        self.cpf = resolve_channels_per_frame(channels_per_frame)
        if self.cpf != self.CPF:
            raise ValueError(
                f"XSP2Encoder needs RGB frames (channels-per-frame {self.CPF}); got {self.cpf}. The "
                f"predictor's warp and loss are defined on the RGB eye.")
        channels, height, width = image_channels_hw(observation_space)
        if num_frames is not None:
            validate_framestack_depth(channels, num_frames, type(self).__name__, channels_per_frame=self.cpf)
            if int(num_frames) != self.N_FRAMES:
                raise ValueError(
                    f"XSP2Encoder needs a stack of exactly {self.N_FRAMES} frames (the dorsal Conv3d has a "
                    f"depth-2 kernel over (z_t-1, z_t)); num_frames={num_frames}. Set NETT_FRAMESTACK_N=2.")
        if channels != self.cpf * self.N_FRAMES:
            raise ValueError(
                f"XSP2Encoder needs a frame stack of exactly {self.N_FRAMES} frames of {self.cpf} channels "
                f"({self.cpf * self.N_FRAMES} channels, framestack=True, NETT_FRAMESTACK_N=2); the "
                f"observation has {channels}.")
        if height % TRUNK_STRIDE or width % TRUNK_STRIDE:
            raise ValueError(f"XSP2Encoder needs an eye divisible by {TRUNK_STRIDE}; got {height}x{width}.")
        self.n_frames = self.N_FRAMES
        self.image_hw = (height, width)
        self.map_hw = (height // TRUNK_STRIDE, width // TRUNK_STRIDE)
        n_map = self.map_hw[0] * self.map_hw[1]
        if features_dim != n_map:
            raise ValueError(
                f"XSP2Encoder's policy features ARE the flattened {self.map_hw[0]}x{self.map_hw[1]} figure "
                f"map, so features_dim must be {n_map} at the {width}x{height} eye; got {features_dim}. "
                f"Refusing to resize the map to fit a cfg value.")
        self.conv_dim = conv_dim

        # phi_e -- applied to x_{t-1} and x_t with these weights (siamese). NatureCNN's convs, padded.
        self.trunk = nn.Sequential(
            nn.Conv2d(self.cpf, 32, kernel_size=8, stride=4, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, conv_dim, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        )
        # phi_v -- ventral (form): z_t -> one logit per location; m_t = sigmoid(logit).
        self.ventral = nn.Sequential(
            nn.Conv2d(conv_dim, ventral_width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ventral_width, ventral_width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ventral_width, 1, 3, padding=1),
        )
        # phi_d -- dorsal (motion): (z_{t-1}, z_t) -> 2-channel pixel-unit motion field, linear.
        self.dorsal = XSP2Dorsal(conv_dim, dorsal_width, 2)
        self._init_trunk()

        with torch.no_grad():
            probe = self.trunk(torch.zeros(1, self.cpf, height, width))
        if tuple(probe.shape[-2:]) != self.map_hw:      # measured, not derived (see nature_cnn.py)
            raise RuntimeError(f"XSP2 trunk map {tuple(probe.shape[-2:])} != expected {self.map_hw}")

    def _init_trunk(self) -> None:
        """phi_e: Kaiming He (fan_in, ReLU) weights, zero bias. phi_v / phi_d keep PyTorch's default."""
        for module in self.trunk:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ parameter groups
    def stream_parameters(self) -> list[nn.Parameter]:
        """phi_e, phi_v, phi_d: every parameter of this encoder. Trained ONLY by the xsp2 aux's Adam."""
        return [*self.trunk.parameters(), *self.ventral.parameters(), *self.dorsal.parameters()]

    # ------------------------------------------------------------------ streams
    def split_frames(self, prepared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(x_{t-1}, x_t): the two frames of a prepared T-major stack, oldest first."""
        c = self.cpf
        if prepared.shape[1] != c * self.N_FRAMES:
            raise ValueError(f"XSP2 was built for {self.N_FRAMES} stacked {c}-channel frames; "
                             f"got {prepared.shape[1]} channels.")
        return prepared[:, :c], prepared[:, c:]

    def current_frame(self, prepared: torch.Tensor) -> torch.Tensor:
        """x_t: the NEWEST frame -- the only frame the ventral stream and the policy read."""
        return self.split_frames(prepared)[1]

    def ventral_map(self, z_t: torch.Tensor) -> torch.Tensor:
        """m = sigmoid(phi_v(z_t)): (B, 1, h, w), the figure map at the stream grid."""
        return torch.sigmoid(self.ventral(z_t))

    def encode_streams(self, prepared: torch.Tensor) -> dict[str, torch.Tensor]:
        """All three streams on a PREPARED (B, 6, H, W) stack, gradients on (the aux path).

        phi_e runs ONCE on cat(x_{t-1}, x_t) along the batch. ``c`` is m * d formed at the stream
        grid and upsampled NEAREST by 8 to the eye: the predictor's flow input.
        """
        x_prev, x_t = self.split_frames(prepared)
        b = prepared.shape[0]
        z = self.trunk(torch.cat([x_prev, x_t], dim=0))                 # (2B, C, h, w), frame-major
        z_prev, z_t = z[:b], z[b:]
        m = self.ventral_map(z_t)
        d = self.dorsal(z_prev, z_t)
        c = upsample_nearest(m * d, TRUNK_STRIDE)
        return {"x_t": x_t, "z_t": z_t, "m": m, "d": d, "c": c}

    # ------------------------------------------------------------------ policy path
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Policy features: flatten(m_t), (B, 160), computed under no_grad from x_t alone.

        ⛔ DETACHED BY CONSTRUCTION. Nothing the RL loss differentiates depends on phi_e or phi_v
        (no autograd graph is even recorded), so no reward gradient can reach them. That is also
        what makes it safe for the xsp2 aux to step these weights in place between PPO's forward
        and PPO's backward inside one minibatch.
        """
        with torch.no_grad():
            x_t = self.current_frame(self._prepare_image(observations))
            m = self.ventral_map(self.trunk(x_t))
        return m.flatten(1).detach()
