"""NETT_DVS_MOVE_GAIN (workspace FINDINGS add. 351, owner 2026-11-06 egomotion DVS pilot, head-bob arm).

The events of a tick on which the env applied the agent's motion (repo B ``_last_decision_mask``) are multiplied by g;
hold ticks are unchanged. Unset = bit for bit. With repo B's NETT_DECISION_PHASE=last at NETT_DECISION_LOOP=2 the
2-frame stack is [hold events, g x move events].
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity
from nett_skrl.body.wrappers.framestack import FrameStack
from test_dvs_polarity import FakeEnv, H, W, _pol, const

KNOBS = ("NETT_DVS_MOVE_GAIN", "NETT_DVS_WINDOW", "NETT_DVS_WINDOW_TAU", "NETT_DECISION_LOOP", "NETT_DVS_THRESHOLD",
         "NETT_DVS_BLUR", "NETT_DVS_BLUR_KERNEL", "NETT_DVS_COLOR", "NETT_AUX_CLTT_CHANNELS_PER_FRAME",
         "NETT_FRAMESTACK_N")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NETT_DVS_BLUR", "0")
    monkeypatch.setenv("NETT_DVS_THRESHOLD", "10")      # well under the 30-per-tick ramp, so luma rounding never drops a tick


class MaskEnv(FakeEnv):
    """FakeEnv that also sets the env's decision mask per step from a schedule (list of per-env bool lists)."""

    def __init__(self, frames, masks, **kw):
        super().__init__(frames, **kw)
        self.masks = list(masks)
        self._last_decision_mask = torch.zeros(self.n_env or 1, dtype=torch.bool)

    def step(self, action):
        self._last_decision_mask = torch.as_tensor(self.masks[self.i - 1], dtype=torch.bool)
        return super().step(action)


def _ramp(n, n_env):
    """Frames whose every pixel rises by 30 per tick (n <= 9): each step is a full-field ON event on every env."""
    assert n <= 9
    return [const(30 * k, n_env=n_env) for k in range(n)]


def _drive(env, n):
    out = [_pol(env.reset()[0])]
    for _ in range(n - 1):
        out.append(_pol(env.step(0)[0]))
    return out


def _gain(monkeypatch, g, loop="2"):
    monkeypatch.setenv("NETT_DVS_MOVE_GAIN", str(g))
    if loop is not None:
        monkeypatch.setenv("NETT_DECISION_LOOP", loop)


def test_unset_is_bit_identical(monkeypatch):
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 256, (2, H, W, 3)).astype(np.uint8) for _ in range(6)]
    masks = [[True, False], [False, True]] * 3
    ref = _drive(DVSPolarity(FakeEnv(frames, n_env=2)), 6)
    monkeypatch.setenv("NETT_DECISION_LOOP", "2")
    env = DVSPolarity(MaskEnv(frames, masks, n_env=2))
    assert env.move_gain is None
    got = _drive(env, 6)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))


@pytest.mark.parametrize("g, want", [(0.0, 0), (0.5, 128), (1.0, 255), (0.25, 64)])
@pytest.mark.parametrize("as_tensor", [True, False])
def test_move_ticks_scaled_hold_ticks_not(monkeypatch, g, want, as_tensor):
    _gain(monkeypatch, g)
    masks = [[True, False], [False, True], [True, True], [False, False], [True, False]]
    env = DVSPolarity(MaskEnv(_ramp(6, 2), masks, n_env=2, as_tensor=as_tensor))
    out = _drive(env, 6)
    for t, m in enumerate(masks, start=1):
        for e in range(2):
            on = out[t][e, ..., 0]
            assert np.all(on == (want if m[e] else 255)), (t, e, m)
            assert int(out[t][e, ..., 1].max()) == 0


def test_stack_order_hold_then_move_at_phase_last(monkeypatch):
    """Loop 2, phase last after a 2-tick pre-roll: ticks pre, pre, hold, move, hold, move ...; the stack read at each
    window end (after the move tick) is [hold map, g x move map]."""
    _gain(monkeypatch, 0.5)
    monkeypatch.setenv("NETT_FRAMESTACK_N", "2")
    sched = [[False]] * 2 + [[False], [True]] * 3
    env = FrameStack(DVSPolarity(MaskEnv(_ramp(len(sched) + 1, 1), sched, n_env=1)), n_stack=2)
    out = _drive(env, len(sched) + 1)
    for t in range(4, len(sched) + 1, 2):           # after each move tick
        stack = out[t][0]                           # (H, W, 2 frames x 2 channels)
        assert np.all(stack[..., 0] == 255) and np.all(stack[..., 2] == 128), t


@pytest.mark.parametrize("bad", ["-0.1", "1.5", "x", "nan"])
def test_refuses_bad_gain(monkeypatch, bad):
    _gain(monkeypatch, bad)
    with pytest.raises(ValueError, match="NETT_DVS_MOVE_GAIN"):
        DVSPolarity(FakeEnv([const(0)]))


@pytest.mark.parametrize("loop", [None, "1", "", "x"])
def test_refuses_without_a_decision_loop(monkeypatch, loop):
    _gain(monkeypatch, 0.5, loop=loop)
    with pytest.raises(ValueError, match="NETT_DECISION_LOOP"):
        DVSPolarity(FakeEnv([const(0)]))


def test_refuses_with_a_window(monkeypatch):
    _gain(monkeypatch, 0.5)
    monkeypatch.setenv("NETT_DVS_WINDOW", "2")
    with pytest.raises(ValueError, match="NETT_DVS_WINDOW"):
        DVSPolarity(FakeEnv([const(0)]))


def test_refuses_an_env_without_a_decision_mask(monkeypatch):
    _gain(monkeypatch, 0.5)
    env = DVSPolarity(FakeEnv(_ramp(3, 2), n_env=2))
    env.reset()
    with pytest.raises(RuntimeError, match="_last_decision_mask"):
        env.step(0)
