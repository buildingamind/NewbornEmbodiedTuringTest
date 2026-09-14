"""Objectness streams: the policy host is ventral; video dorsal is aux-only.

The decoder retains dec3/dec2/dec1/head from the small vendored EoO/GWM
reference, adapting dec3's input width to the host spatial map. Dorsal retains
its complete reference architecture. All weights are trained from scratch.
"""

import os

import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gn(channels: int) -> nn.GroupNorm:
    """GroupNorm with ~16 channels per group — safe for batch=1 inference."""
    return nn.GroupNorm(max(1, channels // 16), channels)


def _conv2d_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        _gn(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallCNNVentral(nn.Module):
    """Standalone reference GWM encoder-decoder, initialized from scratch.

    The segmentation observation wrapper owns this whole stream; unlike the
    auxiliary head below, it shares no encoder or parameters with the policy.
    """

    def __init__(self, num_out_channels: int = 2):
        super().__init__()
        self.enc1 = _conv2d_block(3, 32, stride=2)
        self.enc2 = _conv2d_block(32, 64, stride=2)
        self.enc3 = _conv2d_block(64, 128, stride=2)
        self.dec3 = _conv2d_block(128, 64)
        self.dec2 = _conv2d_block(64, 32)
        self.dec1 = _conv2d_block(32, 16)
        self.head = nn.Conv2d(16, num_out_channels, 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        output_size = x.shape[2:]
        x = self.enc3(self.enc2(self.enc1(x)))
        for block in (self.dec3, self.dec2, self.dec1):
            x = block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        logits = self.head(x)
        if logits.shape[2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Ventral decoder: host spatial features → segmentation logits
# ─────────────────────────────────────────────────────────────────────────────

class VentralDecoder(nn.Module):
    """Lift host spatial features to native-resolution segmentation logits."""

    def __init__(self, in_channels: int, num_out_channels: int = 2):
        super().__init__()
        self.dec3 = _conv2d_block(in_channels, 64)
        self.dec2 = _conv2d_block(64, 32)
        self.dec1 = _conv2d_block(32, 16)
        self.head = nn.Conv2d(16, num_out_channels, 1)

    def forward(self, features: th.Tensor, output_size) -> th.Tensor:
        x = features
        for block in (self.dec3, self.dec2, self.dec1):
            x = block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        logits = self.head(x)
        if logits.shape[2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Dorsal stream: small 3D CNN → bidirectional optical flow
# ─────────────────────────────────────────────────────────────────────────────

class Small3DCNNDorsal(nn.Module):
    """Small 3D CNN dorsal stream: two frames → optical flow.

    75,906 parameters, unchanged from the small reference.

    Two RGB frames are stacked as a (B, 3, 2, H, W) temporal volume.
    A 3D convolution with temporal kernel size 2 collapses the time axis in
    the first layer; subsequent 2D conv layers decode to a (B, 2, H, W) flow.

    For EoO bidirectional flow: call forward(frame_t, frame_t1) — runs the
    network twice with swapped frame order to get F_fwd and F_bwd.

    GroupNorm throughout for batch=1 safety.
    """

    def __init__(self):
        super().__init__()

        # 3D temporal encoder: collapses T=2 → T=1 in the first conv
        # kernel (T=2, H=3, W=3) with T-padding=0 → output T=1
        self.enc3d = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(2, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(2, 32),
            nn.ReLU(inplace=True),
        )  # (B, 3, 2, H, W) → (B, 32, 1, H, W)

        # 2D refinement after temporal collapse
        self.enc2d_1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),
        )  # → (B, 64, H/2, W/2)
        self.enc2d_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.ReLU(inplace=True),
        )

        # Decoder: upsample back to full resolution
        self.dec = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.GroupNorm(2, 32),
            nn.ReLU(inplace=True),
        )
        self.flow_out = nn.Conv2d(32, 2, 1)

    def _flow_one_dir(self, frame_a: th.Tensor, frame_b: th.Tensor) -> th.Tensor:
        """Estimate flow from frame_a → frame_b. Inputs: (B, 3, H, W) float [0, 1]."""
        H, W = frame_a.shape[2:]

        vol = th.stack([frame_a, frame_b], dim=2)       # (B, 3, 2, H, W)
        x   = self.enc3d(vol).squeeze(2)                # (B, 32, H, W)
        x   = self.enc2d_2(self.enc2d_1(x))             # (B, 64, H/2, W/2)
        x   = self.dec(F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False))
        return self.flow_out(x)                          # (B, 2, H, W)

    def forward(
        self, frame_t: th.Tensor, frame_t1: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """Bidirectional flows (used by EoO).

        Returns:
            F_fwd: (B, 2, H, W) — forward  flow (t  → t+1)
            F_bwd: (B, 2, H, W) — backward flow (t+1 → t)
        """
        F_fwd = self._flow_one_dir(frame_t,  frame_t1)
        F_bwd = self._flow_one_dir(frame_t1, frame_t)
        return F_fwd, F_bwd

    def forward_single(
        self, frame_t: th.Tensor, frame_t1: th.Tensor
    ) -> th.Tensor:
        """Forward flow only (used by GWM)."""
        return self._flow_one_dir(frame_t, frame_t1)


class DualStreamHead(nn.Module):
    """Own the ventral decoder and separate dorsal under PPO's head contract."""

    def __init__(self, in_channels: int, slots: int = 2):
        super().__init__()
        self.ventral = VentralDecoder(in_channels, slots)
        self.dorsal = Small3DCNNDorsal()

    def get_masks(self, features, output_size):
        return self.ventral(features, output_size).softmax(dim=1)


class DualStreamAuxLoss(nn.Module):
    """Train the host's single-frame spatial encoder through objectness masks.

    The host is the ventral encoder and receives auxiliary gradients. Only its
    decoder and the separate, unchanged video dorsal live in ``head``. PPO owns
    their optimizer, clipping, and schedule; the host remains registered with
    the policy. Dorsal outputs never feed the policy.
    """

    def __init__(self, encoder, *, slots=2, max_samples=32):
        super().__init__()
        from ...body.observation import image_channels_hw
        total_c, height, width = image_channels_hw(encoder.observation_space)
        if total_c % 3 or total_c < 6:
            raise ValueError(f"{type(self).__name__} needs framestack=True (>=2 RGB frames); got {total_c} channels")
        if min(height, width) < 8:
            raise ValueError("Dual streams require image axes >=8 pixels")
        if slots < 2:
            raise ValueError("Dual streams require at least two mask slots")
        device = next(encoder.parameters()).device
        with th.no_grad():
            features = encoder.encode_spatial_prepared(th.zeros(1, 3, height, width, device=device))
        if features.ndim != 4:
            raise ValueError("Dual streams require a BCHW host spatial feature map")
        self.head = DualStreamHead(features.shape[1], slots).to(device)
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", max_samples))
        if self.max_samples < 1:
            raise ValueError("NETT_AUX_BATCH must be positive")
        self.parameter_counts = {
            name: sum(p.numel() for p in stream.parameters())
            for name, stream in (("host", encoder), ("ventral_decoder", self.head.ventral), ("dorsal", self.head.dorsal))
        }
        self.parameter_counts["total"] = sum(self.parameter_counts.values())

    def masks(self, encoder, curr_frame):
        features = encoder.encode_spatial_prepared(curr_frame)
        return self.head.get_masks(features, curr_frame.shape[2:])

    def frames(self, encoder, observations):
        from ...body.observation import prepare_image_tensor
        # PPO samples have no next_observations. Use the last two T-major RGB
        # frames from the host stack, at native spatial resolution (no grid_div).
        imgs = prepare_image_tensor(observations[:self.max_samples], encoder.observation_space,
                                    device=next(self.head.parameters()).device)
        return imgs[:, -6:-3], imgs[:, -3:]
