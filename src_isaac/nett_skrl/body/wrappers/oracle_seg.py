"""OracleColorSeg -- a DIAGNOSTIC CONTROL, never a candidate model.

The question it answers: IF a segmenter handed the policy a perfect object mask, would the
policy then choose by object (NF, BU above 0.5) -- or is the policy itself unable to use it?
Every motion-segmentation arm so far is a background chooser, and on real agent frames the
GWM objective does not prefer the object partition (FINDINGS §4dh.43). Before building a
better segmenter, measure what a PERFECT one buys. If this arm is also a background chooser
or sits at chance, the bottleneck is downstream of segmentation and no segmenter will fix it.

The mask is a fixed colour rule, not learned: every parsing object is saturated red on a
non-red photograph (a looser version of the rule reaches object AUC 0.983 on 512 real 128x80
test pairs, notes/researcher/offline-seg/ `red_mask`).
⛔ THE RULE MUST NOT LEAK ONTO A BACKGROUND, and leakage that differs by background would hand
the policy exactly the background cue this control exists to remove. The looser rule
(sat > 0.55, g,b < 0.6 r) keeps 1.2% of the FAMILIAR background A's pixels in the source clips
(0% on B and C), and on rendered agent frames leaves 0.04-0.23% stray red speckle that differs
by background (owner spotted it on the visualisation). Measured 2026-09-24 over the source
clips 1A/1B/1C/2A/2B/2C_00, 2A_60, 1C_60: the rule below keeps 0.00% of every background
while keeping 25-41% of the object region (the loose rule: 29-47%); on agent frames stray
speckle falls to <= 0.023% of the frame on every background. It has NO parameters and NO training, so the
train/test round trip is trivially identical; it still saves a (parameter-free) state file so
the runner's segmenter contract holds unchanged.

⛔ This uses KNOWLEDGE OF THE STIMULUS (the objects are red). That is exactly why it is an
upper-bound control and must never be reported as a model. It belongs beside the arms it
bounds, labelled ORACLE.

Rule, per pixel of each RGB frame (values in [0,1]):
    red is the max channel, saturation (max-min)/max > 0.75, max > 0.12,
    g < 0.35 r and b < 0.35 r.
Slot 0 = object (kept), slot 1 = everything else (zeroed).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..observation import image_layout
from .segmentation import SegmentationObservationWrapper


def red_object_mask(frame: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) float in [0,1] -> (B,1,H,W) float {0,1}."""
    r, g, b = frame[:, 0:1], frame[:, 1:2], frame[:, 2:3]
    mx = frame.max(dim=1, keepdim=True).values
    mn = frame.min(dim=1, keepdim=True).values
    sat = (mx - mn) / mx.clamp(min=1e-6)
    m = (r >= mx) & (sat > 0.75) & (mx > 0.12) & (g < 0.35 * r) & (b < 0.35 * r)
    return m.float()


class _OracleModel(nn.Module):
    def __init__(self):
        super().__init__()
        # One buffer so the state file is not empty and a load is still a strict check.
        self.register_buffer("rule_version", torch.tensor(1))

    def get_masks(self, frame):
        fg = red_object_mask(frame)
        return torch.cat((fg, 1.0 - fg), dim=1)


class OracleColorSeg(SegmentationObservationWrapper):
    """Multiply each frame of the stack by the fixed red-object mask."""

    def _configure(self):
        self.kind = "oracle"
        self.num_queries = 2

    def _ensure(self, in_ch):
        if self._model is None:
            self._model = _OracleModel().to(self.device).eval()

    def _frames_and_samples(self, x):
        frames, _ = super()._frames_and_samples(x)
        return frames, [None] * len(frames)     # nothing to learn, nothing to buffer

    def _mask_one(self, obs):
        # Same layout handling as GwmSeg._mask_one: the arm it bounds runs after framestack,
        # where observations may arrive CHW/NCHW; the base path is HWC/NHWC only.
        arr = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
        if arr.ndim not in (3, 4):
            raise ValueError(f"OracleColorSeg expects HWC/NHWC or CHW/NCHW, got shape {arr.shape}")
        chw = image_layout(arr.shape[-3:]) == "chw"
        if chw:
            arr = np.moveaxis(arr, -3, -1)
        result = super()._mask_one(arr)
        if chw:
            result = np.moveaxis(result, -1, -3)
        return torch.from_numpy(result) if isinstance(obs, torch.Tensor) else result

    def _pick_fg(self, masks):
        return 0                                # slot 0 is the object by construction

    def train_step(self):
        return None                             # parameter-free: there is no update

    def _loss(self, batch):
        raise RuntimeError("OracleColorSeg has no objective; train_step never calls this")
