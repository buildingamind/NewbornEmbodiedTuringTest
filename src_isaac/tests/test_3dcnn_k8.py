"""3DCNN-K8 (owner 2026-11-14, workspace FINDINGS add. 423): "(a) First layer only. 3DCNN first layer goes to 8x8."

Pinned here, built on CPU at the campaign eye (128x80, 2 frames x 3 channels):
  (a) the label is "3DCNN" verbatim except cfg["stem_kernel_hw"] = 8 (no pre, no aux, framestack on);
  (b) the encoder has 706,541 parameters = "3DCNN"'s 695,981 + the stem's 6 * (64 - 9) * 32 = 10,560, and every
      parameter outside the stem has "3DCNN"'s shape;
  (c) the stem is Conv3d (2, 8, 8), stride (1, 2, 2), Conv3d padding 0, with the even-k asymmetric pad (3, 4, 3, 4)
      (DECISIONS §89); its output equals an explicit F.pad + F.conv3d reference and keeps the 3x3 stem's grid;
  (d) the forward pass gives (B, 512), deterministic in eval, and encode_spatial gives "3DCNN"'s map shape;
  (e) "CNN-D512P" (the Base twin) already opens with an 8x8 Conv2d, as the owner said, and is not touched.
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


def test_label_is_3dcnn_plus_stem_kernel_only(campaign):
    k8, base = campaign.MODELS["3DCNN-K8"], campaign.MODELS["3DCNN"]
    assert k8["cfg"]["stem_kernel_hw"] == 8 and "stem_kernel_hw" not in base["cfg"]
    assert {k: v for k, v in k8["cfg"].items() if k != "stem_kernel_hw"} == base["cfg"]
    assert {k: v for k, v in k8.items() if k != "cfg"} == {k: v for k, v in base.items() if k != "cfg"}
    assert k8["framestack"] is True and "pre" not in k8 and "aux" not in k8


def test_parameter_count_and_shapes(campaign):
    k8, base = _encoder(campaign, "3DCNN-K8"), _encoder(campaign, "3DCNN")
    assert _n(base) == 695_981 and _n(k8) == 706_541 == 695_981 + 6 * (64 - 9) * 32
    sk, sb = dict(k8.named_parameters()), dict(base.named_parameters())
    assert sk.keys() == sb.keys()
    diff = {n for n in sk if sk[n].shape != sb[n].shape}
    assert diff == {"conv3d.weight"}
    assert tuple(sk["conv3d.weight"].shape) == (32, 3, 2, 8, 8) and tuple(sb["conv3d.weight"].shape) == (32, 3, 2, 3, 3)


@pytest.mark.parametrize("hw", [(80, 128), (160, 256)])
def test_stem_pad_grid_and_reference(campaign, hw):
    h, w = hw
    torch.manual_seed(0)
    enc = _encoder(campaign, "3DCNN-K8", h, w).eval()
    assert enc.stem_kernel_hw == 8 and enc._stem_pad == (3, 4, 3, 4)
    assert enc.conv3d.kernel_size == (2, 8, 8) and enc.conv3d.stride == (1, 2, 2) and enc.conv3d.padding == (0, 0, 0)
    x = torch.rand(2, 3, 2, h, w)
    got = enc._stem_conv(x)
    assert tuple(got.shape) == (2, 32, 1, (h - 1) // 2 + 1, (w - 1) // 2 + 1)
    ref = F.conv3d(F.pad(x, (3, 4, 3, 4)), enc.conv3d.weight, enc.conv3d.bias, stride=(1, 2, 2))
    assert torch.equal(got, ref)


def test_forward_and_spatial_map(campaign):
    torch.manual_seed(0)
    enc, base = _encoder(campaign, "3DCNN-K8").eval(), _encoder(campaign, "3DCNN").eval()
    obs = torch.randint(0, 256, (3, 6, H, W), dtype=torch.uint8)
    with torch.no_grad():
        a, b = enc(obs), enc(obs)
        assert a.shape == (3, 512) and torch.equal(a, b) and torch.isfinite(a).all()
        assert enc.encode_spatial(obs).shape == base.encode_spatial(obs).shape


def test_cnn_base_first_layer_is_already_8x8(campaign):
    enc = _encoder(campaign, "CNN-D512P")
    first = next(m for m in enc.modules() if isinstance(m, torch.nn.Conv2d))
    assert first.kernel_size == (8, 8)
