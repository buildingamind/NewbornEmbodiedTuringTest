"""Owner 2026-10-17 (workspace DECISIONS §135): CNN2F+CLTT-Ref with 1, 2 or 3 extra 3x3 conv layers.

NatureCNN ``extra_convs`` = N more Conv2d(conv_dim, conv_dim, 3, stride 1, padding 1) + ReLU after conv3. The map, the
pool grid and the head are unchanged; each layer adds conv_dim^2*9 + conv_dim params (50,700 at conv_dim 75).
0 (default) is the old Sequential, keys and outputs bit for bit.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.nature_cnn import NatureCNN
from nett_skrl.brain.registry import encoder_mapping

_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
H, W = 280, 448
SPACE = gym.spaces.Box(0, 255, (6, H, W), np.uint8)   # 2-frame RGB stack, the CNN2F eye


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _enc(spec):
    kw = EncoderCfg(**spec["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    torch.manual_seed(0)
    return encoder_mapping[spec["encoder"]](SPACE, **kw)


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_default_is_the_old_encoder_bit_for_bit():
    torch.manual_seed(0)
    a = NatureCNN(SPACE, features_dim=512, conv_dim=75)
    torch.manual_seed(0)
    b = NatureCNN(SPACE, features_dim=512, conv_dim=75, extra_convs=0)
    assert list(a.state_dict()) == list(b.state_dict())
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    x = torch.randint(0, 256, (2, 6, H, W), dtype=torch.uint8).float()
    assert torch.equal(a(x), b(x))
    assert sum(isinstance(m, nn.Conv2d) for m in a.cnn) == 3


def test_base_label_is_unchanged_and_588139(campaign):
    assert "extra_convs" not in campaign.MODELS["CNN2F+CLTT-Ref"]["cfg"]
    assert _n(_enc(campaign.MODELS["CNN2F+CLTT-Ref"])) == 588_139


@pytest.mark.parametrize("n", [1, 2, 3])
def test_xn_adds_n_same_shape_convs_after_conv3(campaign, n):
    base = _enc(campaign.MODELS["CNN2F+CLTT-Ref"])
    enc = _enc(campaign.MODELS[f"CNN2F+CLTT-Ref-X{n}"])
    assert _n(enc) - _n(base) == n * (75 * 75 * 9 + 75)          # 50,700 per layer: the knob reached the encoder
    convs = [m for m in enc.cnn if isinstance(m, nn.Conv2d)]
    assert len(convs) == 3 + n
    for c in convs[3:]:
        assert (c.in_channels, c.out_channels, c.kernel_size, c.stride, c.padding) == (75, 75, (3, 3), (1, 1), (1, 1))
    assert enc.pool_grid == base.pool_grid
    assert enc.linear[0].in_features == base.linear[0].in_features
    x = torch.randint(0, 256, (2, 6, H, W), dtype=torch.uint8).float()
    assert enc.encode_spatial(x).shape == base.encode_spatial(x).shape
    assert enc(x).shape == (2, 512)
    spec, b = campaign.MODELS[f"CNN2F+CLTT-Ref-X{n}"], campaign.MODELS["CNN2F+CLTT-Ref"]
    assert {k: v for k, v in spec.items() if k != "cfg"} == {k: v for k, v in b.items() if k != "cfg"}
    assert spec["cfg"] == {**b["cfg"], "extra_convs": n}


@pytest.mark.parametrize("bad", [-1, 1.0, "1", True])
def test_bad_extra_convs_refuse(bad):
    with pytest.raises(ValueError, match="extra_convs"):
        NatureCNN(SPACE, conv_dim=75, extra_convs=bad)
