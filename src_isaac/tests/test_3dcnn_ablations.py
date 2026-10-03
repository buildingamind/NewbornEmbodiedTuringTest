"""Presenter 3DCNN ablations (workspace DECISIONS §63): 3DCNN-1F, 3DCNN-Sp1, 3DCNN-Sp5.

1F  -- the stem sees ONE frame duplicated into both temporal taps: same Conv3d layer, same init,
       same parameter count as 3DCNN; framestack=False on the arm.
Sp1 -- stem kernel (2,1,1), stride (1,2,2): temporal integration without spatial integration.
Sp5 -- stem kernel (2,5,5), padding 2: a wider spatial stem.
Both Sp variants must leave every downstream shape identical to 3DCNN's. The two new constructor
fields default to the old layer, which must be untouched.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.compact_3dcnn import Compact3DCNN
from nett_skrl.brain.models.utils.init import orthogonal_init
from nett_skrl.brain.registry import encoder_mapping

GAIN = 2 ** 0.5
H, W = 80, 128    # the default eye
NEW = ("3DCNN-1F", "3DCNN-Sp1", "3DCNN-Sp5")


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train

    return campaign_train


def _space(channels, h=H, w=W):
    return gym.spaces.Box(low=0, high=255, shape=(channels, h, w), dtype=np.uint8)


def _from_spec(spec, h=H, w=W, seed=0):
    """Build the arm's encoder the way agent_factory does (EncoderCfg kwargs, trainable dropped)."""
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    channels = 3 * (kwargs["num_frames"] if spec["framestack"] else 1)
    torch.manual_seed(seed)
    enc = encoder_mapping[spec["encoder"]](_space(channels, h, w), **kwargs)
    enc.apply(lambda m: orthogonal_init(m, GAIN))
    return enc


def _shapes(enc, x):
    """Every intermediate shape after the stem: the stem output, then each cnn2d module."""
    out = []
    with torch.no_grad():
        if getattr(enc, "duplicate_frame", False):
            B, CT, h, w = x.shape
            z = enc.conv3d(x.unsqueeze(2).expand(B, CT, enc.num_frames, h, w).contiguous()).squeeze(2)
        else:
            B, CT, h, w = x.shape
            T = enc.num_frames
            z = enc.conv3d(x.view(B, T, CT // T, h, w).permute(0, 2, 1, 3, 4)).squeeze(2)
        out.append(tuple(z.shape))
        for m in enc.cnn2d:
            z = m(z)
            out.append(tuple(z.shape))
        out.append(tuple(enc.linear(z).shape))
    return out


def test_new_labels_are_registered_beside_3dcnn(campaign):
    base = campaign.MODELS["3DCNN"]
    for label in NEW:
        spec = campaign.MODELS[label]
        assert spec["encoder"] == base["encoder"] == "compact_3dcnn"
        assert campaign.segmentation_wrappers(spec) == (["framestack"] if spec["framestack"] else [])
    assert campaign.MODELS["3DCNN-1F"]["framestack"] is False
    assert campaign.MODELS["3DCNN-Sp1"]["framestack"] is True
    assert campaign.MODELS["3DCNN-Sp5"]["framestack"] is True
    # Each label differs from 3DCNN only in the fields it names.
    strip = lambda cfg, *ks: {k: v for k, v in cfg.items() if k not in ks}
    assert strip(campaign.MODELS["3DCNN-1F"]["cfg"], "duplicate_frame", "num_frames") == strip(base["cfg"], "num_frames")
    assert strip(campaign.MODELS["3DCNN-Sp1"]["cfg"], "stem_kernel_hw") == base["cfg"]
    assert strip(campaign.MODELS["3DCNN-Sp5"]["cfg"], "stem_kernel_hw") == base["cfg"]


def test_defaults_build_the_old_layer():
    """Unset fields == the pre-§63 stem: kernel (T,3,3), stride (1,2,2), padding (0,1,1), same state."""
    torch.manual_seed(0)
    a = Compact3DCNN(_space(6), features_dim=512, conv_dim=77, num_frames=2)
    torch.manual_seed(0)
    b = Compact3DCNN(_space(6), features_dim=512, conv_dim=77, num_frames=2, duplicate_frame=False, stem_kernel_hw=3)
    assert a.duplicate_frame is False and a.stem_kernel_hw == 3
    assert a.conv3d.kernel_size == (2, 3, 3)
    assert a.conv3d.stride == (1, 2, 2)
    assert a.conv3d.padding == (0, 1, 1)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.mark.parametrize("hw", [(80, 128), (160, 256)])
@pytest.mark.parametrize("label", NEW)
def test_new_labels_forward_with_3dcnn_downstream_shapes(campaign, label, hw):
    h, w = hw
    base = _from_spec(campaign.MODELS["3DCNN"], h, w)
    enc = _from_spec(campaign.MODELS[label], h, w)
    x6 = torch.randint(0, 256, (3, 6, h, w)).float()
    x = x6[:, 3:] if enc.duplicate_frame else x6
    assert tuple(enc(x).shape) == (3, 512)
    assert _shapes(enc, x) == _shapes(base, x6)


def test_stem_geometry(campaign):
    base = _from_spec(campaign.MODELS["3DCNN"])
    one = _from_spec(campaign.MODELS["3DCNN-1F"])
    sp1 = _from_spec(campaign.MODELS["3DCNN-Sp1"])
    sp5 = _from_spec(campaign.MODELS["3DCNN-Sp5"])
    assert one.conv3d.weight.shape == base.conv3d.weight.shape == (32, 3, 2, 3, 3)
    assert (sp1.conv3d.kernel_size, sp1.conv3d.stride, sp1.conv3d.padding) == ((2, 1, 1), (1, 2, 2), (0, 0, 0))
    assert (sp5.conv3d.kernel_size, sp5.conv3d.stride, sp5.conv3d.padding) == ((2, 5, 5), (1, 2, 2), (0, 2, 2))
    count = lambda m: sum(p.numel() for p in m.parameters())
    # 3 x 32 x 2 x (k*k - 9) stem weights apart; 1F is the same layer.
    assert count(one) == count(base) == 695_981
    assert count(sp1) == count(base) - 3 * 32 * 2 * 8 == 694_445
    assert count(sp5) == count(base) + 3 * 32 * 2 * 16 == 699_053


def test_1f_is_3dcnn_fed_the_frame_twice(campaign):
    """Same seed -> same weights; 1F(x_t) must equal 3DCNN([x_t, x_t])."""
    base = _from_spec(campaign.MODELS["3DCNN"], seed=7)
    one = _from_spec(campaign.MODELS["3DCNN-1F"], seed=7)
    sb, so = base.state_dict(), one.state_dict()
    assert sb.keys() == so.keys() and all(torch.equal(sb[k], so[k]) for k in sb)
    x = torch.randint(0, 256, (4, 3, H, W)).float()
    with torch.no_grad():
        assert torch.allclose(one(x), base(torch.cat([x, x], dim=1)), atol=1e-5, rtol=1e-5)


def test_1f_refuses_a_stacked_input_and_a_single_tap():
    with pytest.raises(ValueError):
        Compact3DCNN(_space(6), num_frames=2, duplicate_frame=True)    # 6 channels is a stack, not a frame
    with pytest.raises(ValueError):
        Compact3DCNN(_space(3), num_frames=1, duplicate_frame=True)


@pytest.mark.parametrize("k", [0, 2, 4, -1])
def test_even_or_nonpositive_stem_kernel_refuses(k):
    with pytest.raises(ValueError):
        Compact3DCNN(_space(6), num_frames=2, stem_kernel_hw=k)
