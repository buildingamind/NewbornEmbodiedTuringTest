"""Small-3DCNN ladder (workspace DECISIONS §89): 3DCNN-W12 / -N77 / -N12 / -K16N77 / -K16N26 / -K16N12.

Pinned here, built on CPU at the campaign eye (128x80, 2 frames x 3 channels):
  (a) every rung's ENCODER parameter count is the one its MODELS comment declares, with the base
      "3DCNN" (695,981) and "3DCNN-250K" (248,762) as positive controls from the same builder;
  (b) the counts follow the closed form 6k^2 s + s + 9 s m + m + 9 m c + c + 8192 c + 512, so the
      HEAD / FILTERS / KERNEL axes are what the labels say they are;
  (c) an EVEN stem kernel pads asymmetrically ((k-1)//2 before, k//2 after) and keeps the 3x3 stem's
      grid; the module output equals an explicit F.pad + F.conv3d reference, and the mirrored pad
      (k//2 before) gives a DIFFERENT output (so (c) would catch a swapped pad);
  (d) ODD kernels take the old path exactly: no F.pad, Conv3d padding k//2, and forward equals the
      plain Conv3d reference bit for bit (3DCNN, 3DCNN-Sp1, 3DCNN-Sp5 counts unchanged).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128
LADDER = {"3DCNN": 695_981, "3DCNN-250K": 248_762, "3DCNN-W12": 125_996, "3DCNN-N77": 644_069,
          "3DCNN-N12": 102_164, "3DCNN-K16N77": 655_925, "3DCNN-K16N26": 230_738, "3DCNN-K16N12": 114_020}
ODD_CONTROLS = {"3DCNN": 695_981, "3DCNN-Sp1": 694_445, "3DCNN-Sp5": 699_053}


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train

    return campaign_train


def _encoder(campaign, label, h=H, w=W):
    spec = campaign.MODELS[label]
    kw = EncoderCfg(**spec["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    ch = 3 * (campaign._FRAMESTACK_N if spec["framestack"] else 1)
    return encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (ch, h, w), np.uint8), **kw)


def _n(m):
    return sum(p.numel() for p in m.parameters())


def _closed_form(cfg):
    k, s, m, c = cfg.get("stem_kernel_hw", 3), cfg.get("stem_dim", 32), cfg.get("mid_dim", 64), cfg["conv_dim"]
    return 6 * k * k * s + s + 9 * s * m + m + 9 * m * c + c + 16 * c * cfg["features_dim"] + cfg["features_dim"]


@pytest.mark.parametrize("label", sorted(LADDER))
def test_encoder_count_matches_declared(campaign, label):
    enc = _encoder(campaign, label)
    assert _n(enc) == LADDER[label]
    assert _closed_form(campaign.MODELS[label]["cfg"]) == LADDER[label]


@pytest.mark.parametrize("label", sorted(ODD_CONTROLS))
def test_odd_controls_unchanged(campaign, label):
    enc = _encoder(campaign, label)
    assert _n(enc) == ODD_CONTROLS[label]
    k = enc.stem_kernel_hw
    assert enc._stem_pad is None and enc.conv3d.padding == (0, k // 2, k // 2)


def test_axes(campaign):
    M = {k: campaign.MODELS[k]["cfg"] for k in LADDER}
    head = {k: v["conv_dim"] for k, v in M.items()}
    assert head == {"3DCNN": 77, "3DCNN-250K": 26, "3DCNN-W12": 12, "3DCNN-N77": 77, "3DCNN-N12": 12,
                    "3DCNN-K16N77": 77, "3DCNN-K16N26": 26, "3DCNN-K16N12": 12}
    for k, v in M.items():
        narrow = k.startswith(("3DCNN-N", "3DCNN-K16"))
        assert (v.get("stem_dim", 32), v.get("mid_dim", 64)) == ((8, 16) if narrow else (32, 64)), k
        assert v.get("stem_kernel_hw", 3) == (16 if "K16" in k else 3), k
        assert v["features_dim"] == 512 and campaign.MODELS[k]["framestack"] is True, k


@pytest.mark.parametrize("hw", [(80, 128), (160, 256), (81, 129)])
def test_even_kernel_grid_and_reference(campaign, hw):
    h, w = hw
    torch.manual_seed(0)
    # built at the campaign eye; the stem alone is applied at each size (odd sizes cannot reach the
    # 4x4 DeterministicAvgPool2d, which refuses non-divisible maps -- a pool rule, not a stem one)
    enc = _encoder(campaign, "3DCNN-K16N12").eval()
    assert enc._stem_pad == (7, 8, 7, 8) and enc.conv3d.padding == (0, 0, 0)
    x = torch.rand(2, 3, 2, h, w)
    got = enc._stem_conv(x)
    assert tuple(got.shape) == (2, 8, 1, (h - 1) // 2 + 1, (w - 1) // 2 + 1)
    ref = F.conv3d(F.pad(x, (7, 8, 7, 8)), enc.conv3d.weight, enc.conv3d.bias, stride=(1, 2, 2))
    assert torch.equal(got, ref)
    mirrored = F.conv3d(F.pad(x, (8, 7, 8, 7)), enc.conv3d.weight, enc.conv3d.bias, stride=(1, 2, 2))
    assert mirrored.shape == ref.shape and not torch.allclose(mirrored, ref), "pad order must be observable"


def test_odd_forward_is_plain_conv(campaign):
    torch.manual_seed(0)
    enc = _encoder(campaign, "3DCNN").eval()
    x = torch.rand(2, 3, 2, H, W)
    ref = F.conv3d(x, enc.conv3d.weight, enc.conv3d.bias, stride=(1, 2, 2), padding=(0, 1, 1))
    assert torch.equal(enc._stem_conv(x), ref)


@pytest.mark.parametrize("label", ["3DCNN-K16N77", "3DCNN-K16N12", "3DCNN-N12", "3DCNN-W12"])
def test_forward_shapes_and_determinism(campaign, label):
    torch.manual_seed(0)
    enc = _encoder(campaign, label).eval()
    obs = torch.randint(0, 256, (3, 6, H, W), dtype=torch.uint8)
    with torch.no_grad():
        a, b = enc(obs), enc(obs)
    assert a.shape == (3, 512) and torch.equal(a, b)
    sp = enc.encode_spatial(obs)
    assert sp.shape[0] == 3 and sp.shape[1] == campaign.MODELS[label]["cfg"]["conv_dim"]


def test_even_kernel_duplicate_frame_path(campaign):
    """duplicate_frame routes through the same _stem_conv, so an even kernel pads there too."""
    from nett_skrl.brain.encoders.compact_3dcnn import Compact3DCNN

    enc = Compact3DCNN(gym.spaces.Box(0, 255, (3, H, W), np.uint8), features_dim=512, conv_dim=12,
                       stem_dim=8, mid_dim=16, stem_kernel_hw=16, duplicate_frame=True, num_frames=2).eval()
    obs = torch.randint(0, 256, (2, 3, H, W), dtype=torch.uint8)
    with torch.no_grad():
        assert enc(obs).shape == (2, 512)
