"""Owner, workspace DECISIONS §162: "3DCNN-CLTT-Ref-T1" -- Compact3DCNN with temporal_kernel_depth=1.

temporal_kernel_depth  None (every existing label) = num_frames: the (T, k, k) stem that collapses time, bit
                       for bit the pre-§162 code (pinned below against a frozen copy of the file at 72c99fc).
                       1: one (1, k, k) kernel SHARED across the frames; the per-frame stem maps are
                       concatenated T-major along channels, so the first Conv2d takes T * stem_dim.
Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU).
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.compact_3dcnn import Compact3DCNN
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128    # the default eye
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"

#: `git rev-parse 72c99fc:src_isaac/nett_skrl/brain/encoders/compact_3dcnn.py` (the commit §162 branched from),
#: copied from the command's output when the fixture was cut (same pattern as test_u35_dropout_nextframe.py).
FROZEN = Path(__file__).resolve().parent / "fixtures" / "frozen" / "compact_3dcnn_72c99fc.py.txt"
FROZEN_BLOB = "5c2575ed032e5731ecde7b9e3a0191546b311883"


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _space(channels, h=H, w=W):
    return gym.spaces.Box(low=0, high=255, shape=(channels, h, w), dtype=np.uint8)


def _kwargs(spec):
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    return kwargs


def _build(spec, channels, seed=0):
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](_space(channels), **_kwargs(spec))


def _n(module):
    return sum(p.numel() for p in module.parameters())


def _load_frozen():
    name = "nett_skrl.brain.encoders._frozen_compact_3dcnn_72c99fc"
    if name in sys.modules:
        return sys.modules[name]
    loader = importlib.machinery.SourceFileLoader(name, str(FROZEN))
    spec = importlib.util.spec_from_file_location(name, str(FROZEN), loader=loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- existing labels: byte-identical

def test_frozen_fixture_is_the_pre_s162_file():
    data = FROZEN.read_bytes()
    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == FROZEN_BLOB
    assert "_fold_time" not in vars(_load_frozen().Compact3DCNN)


@pytest.mark.parametrize("label,channels", [
    ("3DCNN", 6), ("3DCNN-CLTT-Ref", 6), ("3DCNN-1F", 3), ("3DCNN-Sp5", 6), ("3DCNN-K16N77", 6),
    ("3DCNN-250K", 6), ("3DCNN-DVS", 4), ("3DCNN-DVS6", 12),
])
def test_existing_labels_build_and_forward_bitwise_the_pre_s162_code(campaign, label, channels):
    """⛔ SAME SEED, NO load_state_dict: the construction itself (shapes AND init draws) must be the old one,
    then the seeded forward and every parameter gradient."""
    spec = campaign.MODELS[label]
    assert "temporal_kernel_depth" not in spec["cfg"]
    torch.manual_seed(3)
    old = _load_frozen().Compact3DCNN(_space(channels), **_kwargs(spec))
    new = _build(spec, channels, seed=3)
    so, sn = old.state_dict(), new.state_dict()
    assert [(k, tuple(v.shape)) for k, v in so.items()] == [(k, tuple(v.shape)) for k, v in sn.items()]
    assert all(torch.equal(so[k], sn[k]) for k in so), label
    x = torch.randint(0, 256, (3, channels, H, W), generator=torch.Generator().manual_seed(5), dtype=torch.uint8)
    a, b = old(x), new(x)
    assert torch.equal(a, b), label
    a.pow(2).sum().backward()
    b.pow(2).sum().backward()
    for (k, p), q in zip(old.named_parameters(), new.parameters()):
        assert torch.equal(p.grad, q.grad), (label, k)


def test_explicit_full_depth_is_the_default():
    torch.manual_seed(0)
    a = Compact3DCNN(_space(6), features_dim=512, conv_dim=77, num_frames=2)
    torch.manual_seed(0)
    b = Compact3DCNN(_space(6), features_dim=512, conv_dim=77, num_frames=2, temporal_kernel_depth=2)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


# ---------------------------------------------------------------- the T1 label

def test_t1_label_is_3dcnn_cltt_ref_plus_the_depth_only(campaign):
    ref, t1 = campaign.MODELS["3DCNN-CLTT-Ref"], campaign.MODELS["3DCNN-CLTT-Ref-T1"]
    assert {**t1["cfg"]} == {**ref["cfg"], "temporal_kernel_depth": 1}
    assert {k: v for k, v in t1.items() if k != "cfg"} == {k: v for k, v in ref.items() if k != "cfg"}
    assert t1["cfg"]["features_dim"] == 512 and t1["cfg"]["conv_dim"] == 77
    assert t1["aux"] == "cltt_ref" and t1["aux_weight"] == 1.0 and t1["framestack"] is True


def test_t1_shapes_and_parameter_counts_at_128x80(campaign):
    M = campaign.MODELS
    ref, t1 = _build(M["3DCNN-CLTT-Ref"], 6), _build(M["3DCNN-CLTT-Ref-T1"], 6)
    assert tuple(t1.conv3d.weight.shape) == (32, 3, 1, 3, 3)
    assert t1.cnn2d[1].in_channels == 64 and ref.cnn2d[1].in_channels == 32
    assert (_n(ref), _n(t1)) == (695_981, 713_549)
    assert (_n(ref.conv3d), _n(t1.conv3d), _n(ref.cnn2d), _n(t1.cnn2d)) == (1_760, 896, 62_925, 81_357)
    assert _n(AUX_LOSSES["cltt_ref"](t1).head) == _n(AUX_LOSSES["cltt_ref"](ref).head) == 329_216
    x = torch.randint(0, 256, (2, 6, H, W), dtype=torch.uint8).float()
    assert t1(x).shape == (2, 512)
    assert t1.encode_spatial(x).shape == ref.encode_spatial(x).shape == (2, 77, 20, 32)


def test_t1_stem_is_one_shared_2d_kernel_concatenated_t_major(campaign):
    """The stem output IS cat([conv2d(frame t-1), conv2d(frame t)]) with the Conv3d's single depth slice --
    the shared kernel and the T-major order (observation.py:96), checked against an independent conv2d."""
    enc = _build(campaign.MODELS["3DCNN-CLTT-Ref-T1"], 6)
    prepared = torch.rand(2, 6, H, W, generator=torch.Generator().manual_seed(1))
    w, b = enc.conv3d.weight[:, :, 0], enc.conv3d.bias
    want = torch.cat([F.conv2d(prepared[:, :3], w, b, stride=2, padding=1),
                      F.conv2d(prepared[:, 3:], w, b, stride=2, padding=1)], dim=1)
    torch.testing.assert_close(enc._temporal_stem(prepared), want, rtol=1e-5, atol=1e-5)
    # swapping the frames swaps the two channel blocks: nothing in the stem mixes time
    swapped = enc._temporal_stem(torch.cat([prepared[:, 3:], prepared[:, :3]], dim=1))
    torch.testing.assert_close(swapped, torch.cat([want[:, 32:], want[:, :32]], dim=1), rtol=1e-5, atol=1e-5)


def test_t1_builds_with_six_channel_frames_and_even_kernels():
    enc = Compact3DCNN(_space(12), features_dim=512, conv_dim=77, num_frames=2, channels_per_frame=6,
                       temporal_kernel_depth=1)
    assert tuple(enc.conv3d.weight.shape) == (32, 6, 1, 3, 3) and enc.cnn2d[1].in_channels == 64
    assert enc(torch.zeros(1, 12, H, W)).shape == (1, 512)
    even = Compact3DCNN(_space(6), num_frames=2, stem_kernel_hw=16, temporal_kernel_depth=1)
    assert even(torch.zeros(1, 6, H, W)).shape == (1, 128)


@pytest.mark.parametrize("kw,match", [
    ({"temporal_kernel_depth": 3, "num_frames": 3}, None),
    ({"temporal_kernel_depth": 0}, "temporal_kernel_depth must be 1 or num_frames"),
    ({"temporal_kernel_depth": 2, "num_frames": 3}, "temporal_kernel_depth must be 1 or num_frames"),
])
def test_depth_refusals(kw, match):
    channels = 3 * kw.get("num_frames", 2)
    if match is None:
        Compact3DCNN(_space(channels), **kw)
        return
    with pytest.raises(ValueError, match=match):
        Compact3DCNN(_space(channels), **kw)


def test_depth_1_refuses_duplicate_frame():
    with pytest.raises(ValueError, match="duplicate_frame"):
        Compact3DCNN(_space(3), num_frames=2, duplicate_frame=True, temporal_kernel_depth=1)
