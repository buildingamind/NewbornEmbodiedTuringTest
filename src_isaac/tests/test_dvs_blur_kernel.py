"""Owner 2026-10-17 (workspace DECISIONS §134): DVS thr 24 with a 4-pass blur, "simplified to a single passed
blur that is just a larger gaussian filter"; plus the 3DCNN-DVS6 label and the colour-DVS channel guard.

NETT_DVS_BLUR_KERNEL  odd side >= 3 of one binomial kernel (unset = 3 = the original [1,2,1], bit for bit).
                      Side 2N+1 equals N passes of the 3x3 kernel in the interior; within N px of the edge
                      it differs (one replicate pad by N is not N pads by 1) -- measured here, not assumed.
3DCNN-DVS6            3DCNN-DVS with channels_per_frame 6 (NETT_DVS_COLOR=1).
cltt_views guard      NETT_DVS_COLOR=1 with NETT_AUX_CLTT_CHANNELS_PER_FRAME unset refuses: 12 % 3 == 0 would
                      otherwise slice a 2-frame colour-event stack as four RGB frames, silently.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity
from nett_skrl.brain.aux.cltt_views import resolve_channels_per_frame
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.registry import encoder_mapping
from test_dvs_polarity import FakeEnv, drive

_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
HH, WW = 40, 56


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("NETT_DVS_BLUR_KERNEL", "NETT_DVS_BLUR", "NETT_DVS_COLOR", "NETT_AUX_CLTT_CHANNELS_PER_FRAME"):
        monkeypatch.delenv(k, raising=False)


def _wrapper(monkeypatch, kernel=None, blur=None):
    if kernel is not None:
        monkeypatch.setenv("NETT_DVS_BLUR_KERNEL", kernel)
    if blur is not None:
        monkeypatch.setenv("NETT_DVS_BLUR", blur)
    frame = np.zeros((HH, WW, 3), np.uint8)
    return DVSPolarity(FakeEnv([frame]))


def _img(seed=0):
    return torch.from_numpy(np.random.default_rng(seed).uniform(0, 255, (2, HH, WW))).float()


# ---------------------------------------------------------------- knob

def test_unset_is_side_3_and_bit_identical_to_explicit_3(monkeypatch):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (3, 8, 12, 3)).astype(np.uint8) for _ in range(4)]
    assert _wrapper(monkeypatch).blur_kernel == 3
    ref = drive(frames, n_env=3)
    monkeypatch.setenv("NETT_DVS_BLUR_KERNEL", "3")
    got = drive(frames, n_env=3)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))


def test_side_3_is_the_original_kernel_bit_for_bit(monkeypatch):
    """The pre-§134 code, inlined: [1,2,1] outer, normalised, replicate pad 1."""
    y = _img()
    k1 = torch.tensor([1.0, 2.0, 1.0])
    k = torch.outer(k1, k1)
    k = (k / k.sum()).view(1, 1, 3, 3)
    pad = torch.nn.functional.pad(y.reshape(-1, 1, HH, WW), (1, 1, 1, 1), mode="replicate")
    old = torch.nn.functional.conv2d(pad, k).reshape(y.shape)
    assert torch.equal(_wrapper(monkeypatch)._blur(y), old)


@pytest.mark.parametrize("side,passes", [(5, 2), (9, 4)])
def test_side_2n_plus_1_equals_n_passes_in_the_interior_and_differs_only_near_the_edge(monkeypatch, side, passes):
    y = _img(1)
    seq = y
    w3 = _wrapper(monkeypatch)
    for _ in range(passes):
        seq = w3._blur(seq)
    monkeypatch.setenv("NETT_DVS_BLUR_KERNEL", str(side))
    one = DVSPolarity(FakeEnv([np.zeros((HH, WW, 3), np.uint8)]))._blur(y)
    n = passes
    inner = (slice(None), slice(n, -n), slice(n, -n))
    assert torch.allclose(one[inner], seq[inner], atol=1e-3, rtol=0)
    diff = (one - seq).abs()
    assert diff.max() > 1e-3, "the edge band should differ (single pad by N vs N pads by 1) -- check could not look"
    edge = torch.ones_like(diff, dtype=torch.bool)
    edge[inner] = False
    assert diff[~edge].max() <= 1e-3 and diff[edge].max() > 1e-3


def test_kernel_sums_to_one_and_is_the_binomial_row(monkeypatch):
    """A constant image stays constant (normalised), and a centred impulse returns the binomial row."""
    w = _wrapper(monkeypatch, kernel="9")
    c = torch.full((1, HH, WW), 77.0)
    assert torch.allclose(w._blur(c), c, atol=1e-4)
    imp = torch.zeros(1, HH, WW)
    imp[0, HH // 2, WW // 2] = 1.0
    out = w._blur(imp)[0, HH // 2, WW // 2 - 4: WW // 2 + 5] * 256
    row = torch.tensor([1, 8, 28, 56, 70, 56, 28, 8, 1], dtype=torch.float32) / 256 * 256 * 70 / 256
    # centre row of the 2-D kernel = row_k * (centre tap 70/256)
    assert torch.allclose(out, row, atol=1e-5)


def test_larger_kernel_reduces_events_on_noise(monkeypatch):
    """Positive control for the knob reaching the event map: smoothing noise removes threshold crossings."""
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 256, (HH, WW, 3)).astype(np.uint8) for _ in range(3)]
    counts = {}
    for side in ("3", "9"):
        monkeypatch.setenv("NETT_DVS_BLUR_KERNEL", side)
        monkeypatch.setenv("NETT_DVS_THRESHOLD", "24")
        counts[side] = sum(int((e > 0).sum()) for e in drive(frames)[1:])
    assert counts["9"] < counts["3"]


@pytest.mark.parametrize("bad", ["4", "1", "2", "0", "-3", "abc", "9.0"])
def test_bad_values_refuse(monkeypatch, bad):
    with pytest.raises(ValueError, match="NETT_DVS_BLUR_KERNEL"):
        _wrapper(monkeypatch, kernel=bad)


def test_kernel_with_blur_off_refuses_and_blur_off_alone_still_builds(monkeypatch):
    with pytest.raises(ValueError, match="NETT_DVS_BLUR off"):
        _wrapper(monkeypatch, kernel="9", blur="0")
    monkeypatch.delenv("NETT_DVS_BLUR_KERNEL")
    assert _wrapper(monkeypatch, blur="0").blur is False


def test_colour_frames_use_the_same_kernel(monkeypatch):
    rng = np.random.default_rng(4)
    frames = [rng.integers(0, 256, (HH, WW, 3)).astype(np.uint8) for _ in range(3)]
    monkeypatch.setenv("NETT_DVS_COLOR", "1")
    monkeypatch.setenv("NETT_DVS_THRESHOLD", "24")
    a = drive(frames)
    monkeypatch.setenv("NETT_DVS_BLUR_KERNEL", "9")
    b = drive(frames)
    assert a[1].shape == b[1].shape == (HH, WW, 6)
    assert int((b[1] > 0).sum()) < int((a[1] > 0).sum())


# ---------------------------------------------------------------- colour-DVS channel guard

def test_cltt_channels_refuse_when_colour_dvs_leaves_them_unset(monkeypatch):
    monkeypatch.setenv("NETT_DVS_COLOR", "1")
    with pytest.raises(ValueError, match="NETT_AUX_CLTT_CHANNELS_PER_FRAME=6"):
        resolve_channels_per_frame()
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "6")
    assert resolve_channels_per_frame() == 6
    assert resolve_channels_per_frame(2) == 2           # an explicit argument still wins


@pytest.mark.parametrize("off", [None, "0", "false"])
def test_cltt_channels_default_unchanged_without_colour(monkeypatch, off):
    if off is not None:
        monkeypatch.setenv("NETT_DVS_COLOR", off)
    assert resolve_channels_per_frame() == 3


# ---------------------------------------------------------------- 3DCNN-DVS6

@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _build(spec, channels, h=280, w=448):
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(0)
    return encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (channels, h, w), np.uint8), **kwargs)


def test_3dcnn_dvs6_is_3dcnn_dvs_with_six_channels_per_frame(campaign):
    a, b = campaign.MODELS["3DCNN-DVS"], campaign.MODELS["3DCNN-DVS6"]
    assert {**b["cfg"]} == {**a["cfg"], "channels_per_frame": 6}
    assert b["encoder"] == a["encoder"] and b["framestack"] is True and b["pre"] == ["dvs_polarity"]
    assert campaign.segmentation_wrappers(b) == ["dvs_polarity", "framestack"]


def test_3dcnn_dvs6_builds_at_256x160_and_the_two_labels_refuse_each_other(campaign):
    """Owner 2026-10-17 (DECISIONS §134 add. 2): 3DCNN at the 256x160 eye; 448x280 cannot build (below)."""
    enc = _build(campaign.MODELS["3DCNN-DVS6"], 12, h=160, w=256)
    assert enc.base_channels == 6 and enc.num_frames == 2
    x = torch.randint(0, 256, (2, 12, 160, 256), dtype=torch.uint8).float()
    assert enc(x).shape == (2, 512)
    with pytest.raises(ValueError, match="channels_per_frame=6"):
        _build(campaign.MODELS["3DCNN-DVS6"], 4, h=160, w=256)
    with pytest.raises(ValueError, match="channels_per_frame=2"):
        _build(campaign.MODELS["3DCNN-DVS"], 12, h=160, w=256)


def test_3dcnn_dvs6_refuses_the_448x280_unity_recipe_eye(campaign):
    """Recorded limit, not a fix: the 4x4 deterministic pool gets a 70x112 map; 70 % 4 != 0."""
    with pytest.raises(ValueError, match="not divisible"):
        _build(campaign.MODELS["3DCNN-DVS6"], 12)          # refuses at construction (its shape probe)


# ---------------------------------------------------------------- ViT-CLTT-Ref-500K-P28-DVS under colour DVS

def test_vit_p28_dvs_colour_cltt_views_take_the_current_six_channels_and_encode(campaign, monkeypatch):
    """Positive control for the §134 ViT rows: 12-channel colour-event stack, NETT_AUX_CLTT_CHANNELS_PER_FRAME=6.
    The CLTT view must be the CURRENT frame's 6 channels repeated twice (not the last 3 repeated four times), and
    the encoder must take it at 280x448."""
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss

    monkeypatch.setenv("NETT_DVS_COLOR", "1")
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "6")
    enc = _build(campaign.MODELS["ViT-CLTT-Ref-500K-P28-DVS"], 12)
    assert sum(p.numel() for p in enc.parameters()) == 1_373_192
    aux = CLTTReferenceAuxLoss(enc)
    x = torch.rand(2, 12, 280, 448)
    (v,) = aux._make_views([x], enc)
    assert v.shape == x.shape
    assert torch.equal(v[:, :6], x[:, 6:]) and torch.equal(v[:, 6:], x[:, 6:])
    assert not torch.equal(v, x)                     # the older frame was replaced (the check can see a change)
    z = enc(v)
    assert z.shape[0] == 2 and torch.isfinite(z).all()
    monkeypatch.delenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME")
    with pytest.raises(ValueError, match="NETT_AUX_CLTT_CHANNELS_PER_FRAME=6"):
        aux._make_views([x], enc)
