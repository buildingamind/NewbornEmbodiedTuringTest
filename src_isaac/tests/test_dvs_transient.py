"""FINDINGS add. 420 (owner 2026-11-12): NETT_DVS_TRANSIENT, the transient DVS map.

Owner: "takes in 3 frames (no actions taken between them), calculates the DVS between the first two and the DVS between
the last two, then takes the positive difference values between the first two and the negative difference values between
the last two and passes through 3 channels that are the intersection between those positive and negative difference
values" -- both polarities, 6 channels, magnitudes ("size"). Per colour plane on f1, f2, f3:
    up   = min(f2 - f1, f2 - f3)   down = min(f1 - f2, f3 - f2)   kept where >= threshold, else 0
Layout (R+, R-, G+, G-, B+, B-), + = up. Unset = the plain wrapper, bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity
from test_dvs_polarity import FakeEnv, H, W, _pol, drive

KNOBS = ("NETT_DVS_TRANSIENT", "NETT_DVS_WINDOW", "NETT_DVS_WINDOW_TAU", "NETT_DECISION_LOOP", "NETT_DVS_THRESHOLD",
         "NETT_DVS_BLUR", "NETT_DVS_BLUR_KERNEL", "NETT_DVS_COLOR", "NETT_AUX_CLTT_CHANNELS_PER_FRAME",
         "NETT_DVS_MOVE_GAIN", "NETT_DVS_MOTION_COMP", "NETT_DVS_GATE_K", "NETT_DVS_GLOBAL_SUB")

R_UP, R_DN, G_UP, G_DN, B_UP, B_DN = range(6)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    # The BL2 D512P:DVS6 arm's settings: colour, threshold 4, no blur (exact integer differences).
    monkeypatch.setenv("NETT_DVS_COLOR", "1")
    monkeypatch.setenv("NETT_DVS_THRESHOLD", "4")
    monkeypatch.setenv("NETT_DVS_BLUR", "0")
    monkeypatch.setenv("NETT_DECISION_LOOP", "3")


def _on(monkeypatch, mode="1"):
    monkeypatch.setenv("NETT_DVS_TRANSIENT", mode)


def _frame(px, n_env=None, base=0):
    """A frame at ``base`` with per-(row, col, channel) overrides ``px = {(r, c, ch): value}``."""
    f = np.full((H, W, 3), base, dtype=np.uint8)
    for (r, c, ch), v in px.items():
        f[r, c, ch] = v
    return f if n_env is None else np.broadcast_to(f, (n_env, H, W, 3)).copy()


# Hand-computed 3-frame example. Every case sits in its own pixel and channel.
#   pixel (0,0) R  10 -> 50 -> 12   brighten, then return     a=+40 b=-38: up = min(40, 38) = 38
#   pixel (0,1) G 100 -> 60 -> 97   darken, then return       a=-40 b=+37: down = min(40, 37) = 37
#   pixel (0,2) B  10 -> 50 -> 50   brighten, then stay       a=+40 b=0:   up = min(40, 0) = 0
#   pixel (0,3) R  10 -> 30 -> 50   monotone ramp             a=+20 b=+20: up = min(20, -20) < 0, down < 0
#   pixel (0,4) R  10 -> 50 -> 47   returns by less than thr  a=+40 b=-3:  up = 3 < 4 -> 0
#   pixel (0,5) R  10 -> 14 -> 10   exactly at threshold      a=+4  b=-4:  up = 4 (inclusive)
#   pixel (0,6) R  10 -> 13 -> 10   one below threshold       a=+3  b=-3:  up = 3 -> 0
#   pixel (1,0) G  10 -> 200 -> 10  large swing               a=+190 b=-190: up = 190
F1 = {(0, 0, 0): 10, (0, 1, 1): 100, (0, 2, 2): 10, (0, 3, 0): 10, (0, 4, 0): 10, (0, 5, 0): 10, (0, 6, 0): 10,
      (1, 0, 1): 10}
F2 = {(0, 0, 0): 50, (0, 1, 1): 60, (0, 2, 2): 50, (0, 3, 0): 30, (0, 4, 0): 50, (0, 5, 0): 14, (0, 6, 0): 13,
      (1, 0, 1): 200}
F3 = {(0, 0, 0): 12, (0, 1, 1): 97, (0, 2, 2): 50, (0, 3, 0): 50, (0, 4, 0): 47, (0, 5, 0): 10, (0, 6, 0): 10,
      (1, 0, 1): 10}
EXPECT = {(0, 0, R_UP): 38, (0, 1, G_DN): 37, (0, 5, R_UP): 4, (1, 0, G_UP): 190}


def _expected():
    e = np.zeros((H, W, 6), dtype=np.uint8)
    for (r, c, ch), v in EXPECT.items():
        e[r, c, ch] = v
    return e


# ---------------------------------------------------------------- default unchanged

def test_unset_is_bit_identical(monkeypatch):
    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 256, (3, H, W, 3)).astype(np.uint8) for _ in range(8)]
    ref = drive(frames, n_env=3)
    monkeypatch.setenv("NETT_DVS_TRANSIENT", "")
    got = drive(frames, n_env=3)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))
    assert DVSPolarity(FakeEnv([frames[0]], n_env=3)).transient is None


# ---------------------------------------------------------------- the hand example

def test_hand_example_magnitudes(monkeypatch):
    _on(monkeypatch)
    out = drive([_frame(F1), _frame(F2), _frame(F3)])
    assert out[0].shape == (H, W, 6) and out[0].dtype == np.uint8
    assert not out[0].any() and not out[1].any()                  # fewer than 3 frames: nothing
    assert np.array_equal(out[2], _expected())


def test_mag_is_the_default_spelling(monkeypatch):
    _on(monkeypatch, "mag")
    out = drive([_frame(F1), _frame(F2), _frame(F3)])
    assert np.array_equal(out[2], _expected())


def test_bin_is_255_where_mag_is_nonzero(monkeypatch):
    _on(monkeypatch, "bin")
    out = drive([_frame(F1), _frame(F2), _frame(F3)])
    assert np.array_equal(out[2], np.where(_expected() > 0, 255, 0).astype(np.uint8))


def test_window_slides_by_one_tick(monkeypatch):
    """Frame 4 = frame 3: the map at tick 4 is (f2, f3, f4). (0,0) R 50 -> 12 -> 12 is a darkening that stays: 0.
    (0,3) R 30 -> 50 -> 50: 0. Nothing in F2..F4 returns, so the map is empty."""
    _on(monkeypatch)
    out = drive([_frame(F1), _frame(F2), _frame(F3), _frame(F3)])
    assert not out[3].any()


def test_chw_layout(monkeypatch):
    _on(monkeypatch)
    fr = [np.ascontiguousarray(_frame(f).transpose(2, 0, 1)) for f in (F1, F2, F3)]
    out = drive(fr, hwc=False)
    assert out[2].shape == (6, H, W)
    assert np.array_equal(out[2], _expected().transpose(2, 0, 1))


def test_output_space_is_six_channel_uint8(monkeypatch):
    _on(monkeypatch)
    sp = DVSPolarity(FakeEnv([_frame(F1)])).observation_space["policy"]
    assert sp.shape == (H, W, 6) and sp.dtype == np.uint8


def test_tensor_in_tensor_out(monkeypatch):
    _on(monkeypatch)
    env = DVSPolarity(FakeEnv([_frame(F1), _frame(F2), _frame(F3)]))
    env.reset()
    env.step(0)
    obs = env.step(0)[0]["policy"]
    assert isinstance(obs, torch.Tensor) and obs.dtype == torch.uint8
    assert np.array_equal(obs.numpy(), _expected())


def test_blur_on_runs_and_stays_uint8(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("NETT_DVS_BLUR", "1")
    out = drive([_frame(F1), _frame(F2), _frame(F3)])
    assert out[2].dtype == np.uint8 and out[2][1, 0, G_UP] > 0


# ---------------------------------------------------------------- episode boundaries

class DoneEnv(FakeEnv):
    """FakeEnv whose step k (1-based) reports env ``who`` done."""

    def __init__(self, frames, done_at, who, **kw):
        super().__init__(frames, **kw)
        self.k, self.done_at, self.who = 0, done_at, who

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        self.k += 1
        if self.k == self.done_at:
            trunc = trunc.copy()
            trunc[self.who] = True
        return obs, r, term, trunc, info


def test_done_restarts_only_the_finished_env(monkeypatch):
    """Envs 0 and 1 see the same frames. Env 1 is done at step 3 (map 4): its count restarts at that frame, so it is 0
    at maps 4 and 5 and returns at map 6 on frames 4..6 only. Env 0 never stops."""
    _on(monkeypatch)
    seq = [F1, F2, F3, F1, F2, F3, F1]
    env = DVSPolarity(DoneEnv([_frame(f, n_env=2) for f in seq], done_at=3, who=1, n_env=2))
    out = [_pol(env.reset()[0])] + [_pol(env.step(0)[0]) for _ in range(len(seq) - 1)]
    exp = _expected()
    assert np.array_equal(out[2][0], exp) and np.array_equal(out[2][1], exp)
    assert not out[3][1].any() and not out[4][1].any()             # env 1: 1 and 2 frames of its new episode
    assert np.array_equal(out[5][1], exp)                          # env 1: F1 F2 F3 of the new episode
    assert np.array_equal(out[5][0], exp)                          # env 0 unaffected


def test_reset_clears_the_buffer(monkeypatch):
    _on(monkeypatch)
    env = DVSPolarity(FakeEnv([_frame(F1), _frame(F2), _frame(F3)]))
    env.reset(); env.step(0); env.step(0)
    env.frames = [_frame(F2), _frame(F3), _frame(F1)]
    assert not _pol(env.reset()[0]).any()
    assert not _pol(env.step(0)[0]).any()                          # 2 frames since reset: still nothing


# ---------------------------------------------------------------- diagnostic

def test_density_line_prints_once_at_decision_ticks(monkeypatch, capsys):
    _on(monkeypatch)
    monkeypatch.setattr(DVSPolarity, "DENSITY_DECISIONS", 2)
    frames = [_frame(F1), _frame(F2), _frame(F3)] * 4
    drive(frames)
    lines = [l for l in capsys.readouterr().out.splitlines() if "density over the first" in l]
    assert len(lines) == 1 and "transient non-zero" in lines[0]


# ---------------------------------------------------------------- refusals

@pytest.mark.parametrize("val", ["0", "yes", "binary", "2"])
def test_bad_value_refuses(monkeypatch, val):
    _on(monkeypatch, val)
    with pytest.raises(ValueError, match="NETT_DVS_TRANSIENT"):
        DVSPolarity(FakeEnv([_frame(F1)]))


def test_needs_colour(monkeypatch):
    _on(monkeypatch)
    monkeypatch.delenv("NETT_DVS_COLOR")
    with pytest.raises(ValueError, match="NETT_DVS_COLOR=1"):
        DVSPolarity(FakeEnv([_frame(F1)]))


@pytest.mark.parametrize("loop", [None, "2"])
def test_needs_a_loop_of_at_least_three(monkeypatch, loop):
    _on(monkeypatch)
    if loop is None:
        monkeypatch.delenv("NETT_DECISION_LOOP")
    else:
        monkeypatch.setenv("NETT_DECISION_LOOP", loop)
    with pytest.raises(ValueError, match="NETT_DECISION_LOOP=P >= 3"):
        DVSPolarity(FakeEnv([_frame(F1)]))


@pytest.mark.parametrize("key,val", [("NETT_DVS_WINDOW", "3"), ("NETT_DVS_MOVE_GAIN", "0.5"),
                                     ("NETT_DVS_GATE_K", "0.5"), ("NETT_DVS_GLOBAL_SUB", "median:4")])
def test_refuses_the_other_temporal_knobs(monkeypatch, key, val):
    _on(monkeypatch)
    monkeypatch.setenv(key, val)
    with pytest.raises(ValueError, match="NETT_DVS_TRANSIENT"):
        DVSPolarity(FakeEnv([_frame(F1)]))
