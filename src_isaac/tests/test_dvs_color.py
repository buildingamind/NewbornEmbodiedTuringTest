"""NETT_DVS_COLOR: the 6-channel (R+, R-, G+, G-, B+, B-) event frame (owner 2026-10-16, DECISIONS §131).

Driven through reset()/step() like test_dvs_polarity.py, with the same fake env.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity
from test_dvs_polarity import FakeEnv, H, W, _pol, const, drive


@pytest.fixture
def colour(monkeypatch):
    monkeypatch.setenv("NETT_DVS_COLOR", "1")
    monkeypatch.delenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", raising=False)


# ---------------------------------------------------------------- space

def test_space_is_six_channels_hwc_chw_and_batched(colour):
    assert DVSPolarity(FakeEnv([const(0)])).observation_space["policy"].shape == (H, W, 6)
    assert DVSPolarity(FakeEnv([const(0, hwc=False)], hwc=False)).observation_space["policy"].shape == (6, H, W)
    assert DVSPolarity(FakeEnv([const(0, n_env=4)], n_env=4)).observation_space["policy"].shape == (4, H, W, 6)


@pytest.mark.parametrize("off", [None, "0", "false"])
def test_unset_or_off_keeps_two_luma_channels_bit_for_bit(monkeypatch, off):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (3, H, W, 3)).astype(np.uint8) for _ in range(4)]
    monkeypatch.delenv("NETT_DVS_COLOR", raising=False)
    ref = drive(frames, n_env=3)
    if off is not None:
        monkeypatch.setenv("NETT_DVS_COLOR", off)
    got = drive(frames, n_env=3)
    assert got[1].shape == (3, H, W, 2)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))
    assert sum(int(e.sum()) for e in ref[1:]) > 0          # positive control: the frames DO fire


# ---------------------------------------------------------------- the transform

@pytest.mark.parametrize("k,name", [(0, "R"), (1, "G"), (2, "B")])
def test_brightening_one_colour_fires_only_its_on_channel(colour, k, name):
    a = np.array([100.0, 100.0, 100.0]); b = a.copy(); b[k] = 200.0
    ev = drive([const(a), const(b)])[1]
    assert (ev[..., 2 * k] == 255).all(), name
    others = [c for c in range(6) if c != 2 * k]
    assert ev[..., others].sum() == 0


@pytest.mark.parametrize("k", [0, 1, 2])
def test_darkening_one_colour_fires_only_its_off_channel(colour, k):
    a = np.array([150.0, 150.0, 150.0]); b = a.copy(); b[k] = 20.0
    ev = drive([const(a), const(b)])[1]
    assert (ev[..., 2 * k + 1] == 255).all()
    assert ev[..., [c for c in range(6) if c != 2 * k + 1]].sum() == 0


def test_matched_luminance_colour_swap_is_visible_in_colour_and_not_in_luma(monkeypatch):
    # The luma wrapper's load-bearing blindness (test_dvs_polarity) is exactly what this knob removes.
    a = np.array([255.0, 0.0, 0.0])
    b = np.array([0.0, 255.0 * 0.299 / 0.587, 0.0])
    monkeypatch.delenv("NETT_DVS_COLOR", raising=False)
    assert drive([const(a), const(b)])[1].sum() == 0       # control: luma sees nothing
    monkeypatch.setenv("NETT_DVS_COLOR", "1")
    ev = drive([const(a), const(b)])[1]
    assert (ev[..., 1] == 255).all() and (ev[..., 2] == 255).all()   # R- and G+
    assert ev[..., [0, 3, 4, 5]].sum() == 0


@pytest.mark.parametrize("delta,fires", [(3, False), (4, True), (5, True)])
def test_threshold_is_per_colour_and_inclusive(colour, delta, fires):
    a = np.array([100.0, 100.0, 100.0]); b = a.copy(); b[2] += delta
    env = DVSPolarity(FakeEnv([const(a), const(b)]), threshold=4.0)
    env.reset()
    ev = _pol(env.step(0)[0])
    assert bool(ev[..., 4].any()) is fires


def test_chw_puts_the_six_channels_at_axis_minus_three(colour):
    a = const([100, 100, 100], hwc=False); b = const([200, 100, 20], hwc=False)
    ev = drive([a, b], hwc=False)[1]
    assert ev.shape == (6, H, W)
    assert (ev[0] == 255).all() and (ev[5] == 255).all() and ev[[1, 2, 3, 4]].sum() == 0


def test_first_frame_emits_nothing(colour):
    assert _pol(DVSPolarity(FakeEnv([const([200, 10, 90])])).reset()[0]).sum() == 0


def test_spatially_local_and_blurred_like_luma(colour):
    a = const(100); b = a.copy(); b[2:4, 3:6, 0] = 220
    ev = drive([a, b])[1]
    assert ev[2:4, 3:6, 0].all() and not ev[6:, 9:, 0].any()
    assert ev[..., 2:].sum() == 0


def test_done_zeroes_only_the_finished_env_in_colour_mode(colour):
    # ⛔ The batched check used to be `_prev.ndim == 3`, true for a luma buffer (N,H,W) and FALSE for a
    # colour buffer (N,3,H,W): every episode cut would have fired a full-field teleport event.
    n = 3
    env = DVSPolarity(FakeEnv([const(10, n_env=n), const(200, n_env=n)], n_env=n))
    inner = env.env
    env.reset()
    done = np.array([True, False, False])
    orig = inner.step
    inner.step = lambda a: (orig(a)[0], np.zeros(n), done, np.zeros(n, bool), {})
    ev = _pol(env.step(0)[0])
    assert ev[0].sum() == 0
    assert ev[1][..., 0].all() and ev[2][..., 4].all()


def test_numpy_in_numpy_out(colour):
    env = DVSPolarity(FakeEnv([const(10), const(200)], as_tensor=False))
    env.reset()
    out = env.step(0)[0]["policy"]
    assert isinstance(out, np.ndarray) and out.dtype == np.uint8 and out.shape == (H, W, 6)


def test_tensor_in_tensor_out(colour):
    env = DVSPolarity(FakeEnv([const(10), const(200)]))
    env.reset()
    out = env.step(0)[0]["policy"]
    assert isinstance(out, torch.Tensor) and out.dtype == torch.uint8


# ---------------------------------------------------------------- refusals

def test_cltt_channel_knob_mismatch_is_refused(monkeypatch):
    monkeypatch.setenv("NETT_DVS_COLOR", "1")
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "2")
    with pytest.raises(ValueError, match="6 channels"):
        DVSPolarity(FakeEnv([const(0)]))
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "6")
    assert DVSPolarity(FakeEnv([const(0)])).n_out == 6


def test_non_rgb_frame_is_refused(colour):
    env = DVSPolarity(FakeEnv([const(0)]))
    env.env.frames = [np.zeros((H, W, 6), dtype=np.uint8)]
    with pytest.raises(ValueError, match="BEFORE"):
        env.reset()


# ---------------------------------------------------------------- the +DVS label's real chain

def test_dvs_label_builds_a_six_channel_first_conv_under_the_knob(monkeypatch):
    """DVSPolarity -> ChannelsFirst -> NatureCNN, as test_fovea's 2-channel twin: in_channels comes from the space."""
    import gymnasium as gym
    from nett_skrl.body.wrappers.channels_first import ChannelsFirst
    from nett_skrl.brain.encoders.nature_cnn import NatureCNN
    from test_fovea import _StubEnv

    frame = torch.randint(0, 256, (2, 64, 96, 3), generator=torch.Generator().manual_seed(2), dtype=torch.uint8)
    for knob, want in ((None, 2), ("1", 6)):
        if knob is None:
            monkeypatch.delenv("NETT_DVS_COLOR", raising=False)
        else:
            monkeypatch.setenv("NETT_DVS_COLOR", knob)
        chain = ChannelsFirst(DVSPolarity(_StubEnv(frame), threshold=4))
        space = chain.observation_space["policy"]
        assert tuple(space.shape) == (2, want, 64, 96)
        enc = NatureCNN(gym.spaces.Box(0, 255, shape=space.shape[1:], dtype=np.uint8), features_dim=16, conv_dim=8,
                        spatial_pool=False)
        conv1 = next(m for m in enc.modules() if isinstance(m, torch.nn.Conv2d))
        assert conv1.in_channels == want
        obs, _ = chain.reset()
        assert enc(torch.as_tensor(obs["policy"]).float()).shape == (2, 16)
