"""Owner 2026-10-18 (workspace DECISIONS §149): "the video would play at 1 frame a tick and the new DVS would receive
5 frames but convert it into a single 6-channel observation". FINDINGS §4dh.44h.183.

NETT_DVS_WINDOW=W      the last W per-tick event maps, weighted and summed into one map of the same channels
                       (unset = one tick, bit for bit). Needs NETT_DECISION_LOOP=P >= W: the decision's last W ticks.
                       DECISIONS §149 add. 1: P = 6, W = 5 drops the move tick ("drop the first frame").
NETT_DVS_WINDOW_TAU    lam = exp(-1/tau), weight lam^k on the map k ticks old (unset = lam 1, a plain count).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity
from test_dvs_polarity import FakeEnv, H, W, _pol, const, drive

KNOBS = ("NETT_DVS_WINDOW", "NETT_DVS_WINDOW_TAU", "NETT_DECISION_LOOP", "NETT_DVS_THRESHOLD",
         "NETT_DVS_BLUR", "NETT_DVS_BLUR_KERNEL", "NETT_DVS_COLOR", "NETT_AUX_CLTT_CHANNELS_PER_FRAME")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    # Pure differences make the expected values exact: no blur smearing at the frame edge.
    monkeypatch.setenv("NETT_DVS_BLUR", "0")


def _window(monkeypatch, w, tau=None, loop=None):
    monkeypatch.setenv("NETT_DVS_WINDOW", str(w))
    monkeypatch.setenv("NETT_DECISION_LOOP", str(w if loop is None else loop))
    if tau is not None:
        monkeypatch.setenv("NETT_DVS_WINDOW_TAU", str(tau))


def _flashes(levels, n_env=None):
    """Uniform frames at the given grey levels: each step to a higher level is one full-field ON event."""
    return [const(v, n_env=n_env) for v in levels]


# ---------------------------------------------------------------- default unchanged

def test_unset_is_bit_identical_to_one_tick(monkeypatch):
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 256, (3, H, W, 3)).astype(np.uint8) for _ in range(8)]
    ref = drive(frames, n_env=3)
    # The knob absent and the knob's partner set alone must not change a single value.
    monkeypatch.setenv("NETT_DECISION_LOOP", "5")
    got = drive(frames, n_env=3)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))


def test_unset_space_and_dtype_unchanged(monkeypatch):
    env = DVSPolarity(FakeEnv([const(0)]))
    assert env.window is None
    assert env.observation_space["policy"].shape == (H, W, 2)


# ---------------------------------------------------------------- values

def test_plain_count_is_events_over_window(monkeypatch):
    """lam = 1: a pixel that fired on j of the last W ticks reads round(255 j / W)."""
    _window(monkeypatch, 5)
    # ticks: reset(0), +100 (ON), same, +100 (ON), same, same, same, same, same
    out = drive(_flashes([0, 100, 100, 200, 200, 200, 200, 200, 200]))
    on = [int(o[0, 0, 0]) for o in out]
    assert on == [0, 51, 51, 102, 102, 102, 51, 51, 0]
    assert all(int(o[..., 1].max()) == 0 for o in out)           # no OFF event anywhere


def test_decay_weights_newest_most(monkeypatch):
    """tau = 5: weights lam^k / sum, lam = exp(-1/5); an event k ticks old reads round(255 lam^k / S)."""
    _window(monkeypatch, 5, tau=5)
    lam = math.exp(-1 / 5)
    s = sum(lam ** k for k in range(5))
    out = drive(_flashes([0, 100] + [100] * 6))
    on = [int(o[0, 0, 0]) for o in out]
    want = [0] + [round(255 * lam ** k / s) for k in range(5)] + [0, 0]
    assert on == want
    assert on[1] > on[2] > on[3] > on[4] > on[5] > 0               # temporal order survives


def test_full_window_of_events_saturates_at_255(monkeypatch):
    _window(monkeypatch, 5, tau=5)
    out = drive(_flashes([0, 40, 80, 120, 160, 200]))
    assert int(out[-1][0, 0, 0]) == 255


def test_window_equals_sum_of_single_tick_maps(monkeypatch):
    """Positive control against the unwindowed wrapper on random video: out = round(sum_k w_k e_{t-k})."""
    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 256, (2, H, W, 3)).astype(np.uint8) for _ in range(9)]
    single = drive(frames, n_env=2)
    _window(monkeypatch, 5, tau=3)
    got = drive(frames, n_env=2)
    lam = math.exp(-1 / 3)
    w = np.array([lam ** k for k in range(5)])
    w = w / w.sum()
    for t in range(len(frames)):
        acc = sum(w[k] * single[t - k].astype(np.float64) for k in range(5) if t - k >= 0)
        want = np.clip(np.round(acc), 0, 255).astype(np.int16)
        # float32 (wrapper) vs float64 (here) may round an exact .5 differently: allow 1 level, nothing more.
        assert np.abs(got[t].astype(np.int16) - want).max() <= 1, t
    assert any(((g > 0) & (g < 255)).any() for g in got)          # accumulation produced intermediate levels


def test_colour_six_channels_accumulate_per_channel(monkeypatch):
    monkeypatch.setenv("NETT_DVS_COLOR", "1")
    _window(monkeypatch, 5)
    # Red rises twice, blue falls once: R+ = 2 events, B- = 1 event.
    frames = [const((0, 0, 200)), const((100, 0, 200)), const((200, 0, 100)), const((200, 0, 100))]
    out = drive(frames)
    last = out[-1][0, 0]
    assert out[-1].shape == (H, W, 6)
    assert list(map(int, last)) == [102, 0, 0, 0, 0, 51]


# ---------------------------------------------------------------- boundaries

def test_reset_clears_the_window(monkeypatch):
    _window(monkeypatch, 5)
    env = DVSPolarity(FakeEnv(_flashes([0, 100, 200])))
    env.reset()
    env.step(0)
    assert int(_pol(env.step(0)[0])[0, 0, 0]) == 102
    assert int(_pol(env.reset()[0])[0, 0, 0]) == 0                 # reset: nothing carried over
    assert int(_pol(env.step(0)[0])[0, 0, 0]) == 51                # one event in the new episode only


class _DoneEnv(FakeEnv):
    """Batched; env 0 reports done on the given step numbers (1-based)."""

    def __init__(self, frames, done_at, **kw):
        super().__init__(frames, **kw)
        self.done_at, self.k = set(done_at), 0

    def step(self, action):
        self.k += 1
        obs, r, term, trunc, info = super().step(action)
        if self.k in self.done_at:
            term = term.copy()
            term[0] = True
        return obs, r, term, trunc, info


def test_done_env_window_is_cleared_and_others_keep_theirs(monkeypatch):
    _window(monkeypatch, 5)
    frames = _flashes([0, 100, 200, 200, 200, 200], n_env=2)
    env = DVSPolarity(_DoneEnv(frames, done_at={2}, n_env=2))
    env.reset()
    env.step(0)                                                    # both: one event
    o2 = _pol(env.step(0)[0])                                      # env 0 done here
    assert int(o2[0, 0, 0, 0]) == 0 and int(o2[1, 0, 0, 0]) == 102
    o3 = _pol(env.step(0)[0])
    assert int(o3[0, 0, 0, 0]) == 0                                # the old episode's events are gone
    assert int(o3[1, 0, 0, 0]) == 102                              # env 1 untouched


def test_numpy_obs_path(monkeypatch):
    _window(monkeypatch, 5)
    out = drive(_flashes([0, 100, 200]), as_tensor=False)
    assert isinstance(out[-1], np.ndarray) and out[-1].dtype == np.uint8
    assert int(out[-1][0, 0, 0]) == 102


# ---------------------------------------------------------------- guards

@pytest.mark.parametrize("w,loop", [("5", None), ("5", "2"), ("5", "4"), ("3", ""), ("3", "x")])
def test_window_needs_a_decision_loop_at_least_as_long(monkeypatch, w, loop):
    monkeypatch.setenv("NETT_DVS_WINDOW", w)
    if loop is not None:
        monkeypatch.setenv("NETT_DECISION_LOOP", loop)
    with pytest.raises(ValueError, match="NETT_DECISION_LOOP"):
        DVSPolarity(FakeEnv([const(0)]))


@pytest.mark.parametrize("w", ["1", "0", "-5", "five", "2.5"])
def test_bad_window_refuses(monkeypatch, w):
    monkeypatch.setenv("NETT_DVS_WINDOW", w)
    monkeypatch.setenv("NETT_DECISION_LOOP", "5")
    with pytest.raises(ValueError, match="NETT_DVS_WINDOW"):
        DVSPolarity(FakeEnv([const(0)]))


@pytest.mark.parametrize("tau", ["0", "-1", "inf", "nan", "x"])
def test_bad_tau_refuses(monkeypatch, tau):
    _window(monkeypatch, 5)
    monkeypatch.setenv("NETT_DVS_WINDOW_TAU", tau)
    with pytest.raises(ValueError, match="NETT_DVS_WINDOW_TAU"):
        DVSPolarity(FakeEnv([const(0)]))


def test_tau_without_window_refuses(monkeypatch):
    monkeypatch.setenv("NETT_DVS_WINDOW_TAU", "5")
    with pytest.raises(ValueError, match="without NETT_DVS_WINDOW"):
        DVSPolarity(FakeEnv([const(0)]))


@pytest.mark.parametrize("loop", ["5", "6", "10"])
def test_window_shorter_than_the_loop_is_allowed(monkeypatch, loop):
    _window(monkeypatch, 5, loop=loop)
    assert DVSPolarity(FakeEnv([const(0)])).window == 5


def _decision_tick_maps(levels, p, w, tau=None, monkeypatch=None):
    """Drive one env tick by tick; return the map at every decision tick c = 1 + kP (c counts maps since reset)."""
    _window(monkeypatch, w, tau=tau, loop=p)
    out = drive(_flashes(levels))
    return [out[c - 1] for c in range(1, len(out) + 1) if (c - 1) % p == 0]


def test_loop6_window5_drops_the_move_tick(monkeypatch):
    """P = 6, W = 5 (DECISIONS §149 add. 1). Maps c = 2..7 are decision 1's ticks; c = 2 is tick 0, the move tick.
    An event only on the move tick is absent from the decision-tick map; one on the next tick is in it."""
    move_only = [0, 100] + [100] * 11                      # event at c = 2 only
    held = [0, 0, 100] + [100] * 10                        # event at c = 3 only
    m = _decision_tick_maps(move_only, 6, 5, monkeypatch=monkeypatch)
    assert int(m[1][0, 0, 0]) == 0                         # decision tick c = 7: window c = 3..7
    m = _decision_tick_maps(held, 6, 5, monkeypatch=monkeypatch)
    assert int(m[1][0, 0, 0]) == 51                        # plain count, 1 of 5
    assert int(m[2][0, 0, 0]) == 0                         # next decision: gone


def test_loop6_window5_covers_the_five_held_ticks(monkeypatch):
    """Events on all five held ticks of a decision (c = 3..7) saturate the decision-tick map at 255."""
    levels = [0, 0, 40, 80, 120, 160, 200] + [200] * 6
    m = _decision_tick_maps(levels, 6, 5, tau=5, monkeypatch=monkeypatch)
    assert int(m[1][0, 0, 0]) == 255


def test_banner(monkeypatch, capsys):
    _window(monkeypatch, 5, tau=5, loop=6)
    DVSPolarity(FakeEnv([const(0)]))
    assert "[NETT dvs window] W=5 tau=5.0" in capsys.readouterr().out


def test_density_line_reports_window_and_matched_single_difference(monkeypatch, capsys):
    """P = 6, W = 5. Each decision: the move tick FALLS 200 -> 0 (an OFF event the window must drop), then five +40 rises
    on the held ticks. At every decision tick the window has all 5 ON events on the 3 ON planes and nothing else:
    non-zero 0.5 of pixels x 6 channels, all at 255 (intermediate 0). One 5-frame difference over the same frames
    (move-tick frame 0 -> 200) is ON on the same planes: 0.5. Had the move tick leaked in, the OFF planes would fire."""
    _window(monkeypatch, 5, loop=6)
    levels = [0]
    for _ in range(DVSPolarity.DENSITY_DECISIONS + 2):
        levels += [0, 40, 80, 120, 160, 200]
    env = DVSPolarity(FakeEnv(_flashes(levels)))
    capsys.readouterr()
    env.reset()
    for _ in range(len(levels) - 1):
        env.step(0)
    out = capsys.readouterr().out
    assert out.count("density over the first 50 decisions") == 1
    line = [x for x in out.splitlines() if "density over" in x][0]
    nz = float(line.split("window non-zero ")[1].split(",")[0])
    mid = float(line.split("intermediate (0<v<255) ")[1].split(";")[0])
    one = float(line.split("same threshold: non-zero ")[1].split(" ")[0])
    assert (nz, mid, one) == (pytest.approx(0.5), pytest.approx(0.0), pytest.approx(0.5))
