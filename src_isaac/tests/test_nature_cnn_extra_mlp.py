"""Owner 2026-10-17 (workspace DECISIONS §138, extended by §142): CNN2F+CLTT-Ref with 1..8 extra MLP layers at the end of the encoder.

NatureCNN ``extra_mlp`` = N more Linear(features_dim, features_dim) + ReLU appended to ``self.linear``, after the
n_flatten->512 projection and before the policy/value heads. The conv stack and the projection are unchanged; each
layer adds 512*512 + 512 = 262,656 params. 0 (default) is the old encoder, keys and outputs bit for bit.
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
PER_LAYER = 512 * 512 + 512


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
    b = NatureCNN(SPACE, features_dim=512, conv_dim=75, extra_mlp=0)
    assert list(a.state_dict()) == list(b.state_dict())
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    x = torch.randint(0, 256, (2, 6, H, W), dtype=torch.uint8).float()
    assert torch.equal(a(x), b(x))
    assert sum(isinstance(m, nn.Linear) for m in a.linear) == 1


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
def test_mn_appends_n_512_linears_after_the_projection(campaign, n):
    base = _enc(campaign.MODELS["CNN2F+CLTT-Ref"])
    enc = _enc(campaign.MODELS[f"CNN2F+CLTT-Ref-M{n}"])
    assert _n(enc) - _n(base) == n * PER_LAYER                  # 262,656 per layer: the knob reached the encoder
    assert _n(enc) == 588_139 + n * PER_LAYER
    lins = [m for m in enc.linear if isinstance(m, nn.Linear)]
    assert len(lins) == 1 + n
    assert (lins[0].in_features, lins[0].out_features) == (base.linear[0].in_features, 512)
    for lin in lins[1:]:
        assert (lin.in_features, lin.out_features) == (512, 512)
    assert [type(m) for m in enc.linear] == [nn.Linear, nn.ReLU] * (1 + n)
    # the conv stack, its keys and its pool grid are untouched
    assert enc.pool_grid == base.pool_grid
    assert [k for k in enc.state_dict() if k.startswith("cnn.")] == [k for k in base.state_dict() if k.startswith("cnn.")]
    x = torch.randint(0, 256, (2, 6, H, W), dtype=torch.uint8).float()
    assert enc.encode_spatial(x).shape == base.encode_spatial(x).shape
    assert enc(x).shape == (2, 512)
    spec, b = campaign.MODELS[f"CNN2F+CLTT-Ref-M{n}"], campaign.MODELS["CNN2F+CLTT-Ref"]
    assert {k: v for k, v in spec.items() if k != "cfg"} == {k: v for k, v in b.items() if k != "cfg"}
    assert spec["cfg"] == {**b["cfg"], "extra_mlp": n}


def test_forward_runs_through_the_added_layers():
    """Every forward() consumer (the policy/value heads and the CLTT-Ref aux alike) reads the post-MLP feature:
    zeroing the LAST added layer must zero the encoder output, and perturbing it must change the output."""
    torch.manual_seed(0)
    enc = NatureCNN(SPACE, features_dim=512, conv_dim=75, extra_mlp=2)
    x = torch.randint(0, 256, (2, 6, H, W), dtype=torch.uint8).float()
    y0 = enc(x)
    last = enc.linear[-2]
    with torch.no_grad():
        last.bias.add_(1.0)
    assert not torch.equal(enc(x), y0)
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    assert torch.count_nonzero(enc(x)) == 0


@pytest.mark.parametrize("bad", [-1, 1.0, "1", True])
def test_bad_extra_mlp_refuse(bad):
    with pytest.raises(ValueError, match="extra_mlp"):
        NatureCNN(SPACE, conv_dim=75, extra_mlp=bad)
