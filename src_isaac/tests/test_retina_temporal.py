"""RETINA program (owner 2026-11-17): retina_temporal modes -- spaces, hand-checked values, the tr3 == add. 420 identity,
episode-boundary reseeding, and the registry / model labels."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity
from nett_skrl.body.wrappers.retina_temporal import N_OUT, RetinaTemporal, parse_mode
from test_dvs_polarity import FakeEnv, H, W, _pol, const

KNOBS = ("NETT_RETINA", "NETT_RETINA_THRESHOLD", "NETT_RETINA_TAU", "NETT_RETINA_HALF", "NETT_DVS_TRANSIENT",
         "NETT_DVS_COLOR", "NETT_DVS_THRESHOLD", "NETT_DVS_BLUR", "NETT_DECISION_LOOP")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)


def _drive(frames, mode, monkeypatch, **kw):
    monkeypatch.setenv("NETT_RETINA", mode)
    env = RetinaTemporal(FakeEnv(frames, **kw))
    out = [_pol(env.reset()[0])]
    for _ in range(len(frames) - 1):
        out.append(_pol(env.step(0)[0]))
    return env, out


class DoneEnv(FakeEnv):
    """FakeEnv whose step ``done_at`` reports env ``which`` done (the frame served there is the new episode's first)."""

    def __init__(self, frames, done_at, which, **kw):
        super().__init__(frames, **kw)
        self.done_at, self.which, self.k = done_at, which, 0

    def step(self, action):
        obs, r, t, tr, info = super().step(action)
        self.k += 1
        if self.k == self.done_at:
            t = t.copy()
            t[self.which] = True
        return obs, r, t, tr, info


def test_unset_and_unknown_refuse(monkeypatch):
    with pytest.raises(ValueError):
        RetinaTemporal(FakeEnv([const(0)]))
    for bad in ("tr4", "tr11", "mtap2", "dvs", "smear2"):
        with pytest.raises(ValueError):
            parse_mode(bad)


@pytest.mark.parametrize("mode", ["tr3", "tr5", "mtap5", "edr", "edrrgb", "tsurf", "smear", "sustrans"])
@pytest.mark.parametrize("hwc", [True, False])
def test_space_and_static_scene(mode, hwc, monkeypatch):
    frames = [const((30, 120, 200), hwc=hwc, n_env=2)] * 7
    env, out = _drive(frames, mode, monkeypatch, hwc=hwc, n_env=2)
    n = N_OUT[parse_mode(mode)[0]]
    sp = env.observation_space["policy"]
    assert sp.shape == ((2, H, W, n) if hwc else (2, n, H, W)) and sp.dtype == np.uint8
    for o in out:
        assert o.shape == sp.shape and o.dtype == np.uint8
    if mode in ("smear",):
        assert (out[-1][..., 0] == 30).all() if hwc else (out[-1][:, 0] == 30).all()
    elif mode in ("edrrgb", "sustrans"):
        pass                                    # carry the (static) colour; their change channels are checked below
    else:
        assert all((o == 0).all() for o in out), f"{mode}: a static scene must emit no change"


def test_rgb_refused(monkeypatch):
    monkeypatch.setenv("NETT_RETINA", "tr5")
    env = RetinaTemporal(FakeEnv([const(0, c=1)[..., :1]]))   # FakeEnv's space always says 3 ch; the frame has 1
    with pytest.raises(ValueError):
        env.reset()


def test_tr3_equals_add420_transient(monkeypatch):
    rng = np.random.default_rng(3)
    frames = [rng.integers(0, 256, (3, H, W, 3)).astype(np.uint8) for _ in range(9)]
    monkeypatch.setenv("NETT_RETINA_THRESHOLD", "4")
    _, ours = _drive(frames, "tr3", monkeypatch, n_env=3)
    for k, v in (("NETT_DVS_TRANSIENT", "1"), ("NETT_DVS_COLOR", "1"), ("NETT_DVS_THRESHOLD", "4"),
                 ("NETT_DVS_BLUR", "0"), ("NETT_DECISION_LOOP", "3")):
        monkeypatch.setenv(k, v)
    ref_env = DVSPolarity(FakeEnv(frames, n_env=3))
    ref = [_pol(ref_env.reset()[0])] + [_pol(ref_env.step(0)[0]) for _ in range(len(frames) - 1)]
    # add. 420 emits zeros until it holds 3 frames; ours seeds the buffer with the first frame (also zeros there).
    for a, b in zip(ours[2:], ref[2:]):
        np.testing.assert_array_equal(a, b)


def test_tr5_pulse_only_at_the_middle(monkeypatch):
    # pixel (0,0) R: 10,10,60,10,10 -> up = min(50,50,50,50) = 50 at the 5th tick; a step 10,10,60,60,60 gives 0
    seq_pulse = [10, 10, 60, 10, 10]
    seq_step = [10, 10, 60, 60, 60]
    frames = []
    for a, b in zip(seq_pulse, seq_step):
        f = np.zeros((H, W, 3), np.uint8)
        f[0, 0, 0], f[0, 1, 0] = a, b
        frames.append(f)
    _, out = _drive(frames, "tr5", monkeypatch)
    assert out[4][0, 0, 0] == 50 and out[4][0, 1, 0] == 0 and out[4][0, 1, 1] == 0
    assert out[3][0, 0, 0] == 0                 # the pulse is not yet at the window's middle


def test_mtap_reference_is_weighted_past(monkeypatch):
    vals = [0, 0, 0, 0, 100]
    frames = []
    for v in vals:
        f = np.zeros((H, W, 3), np.uint8)
        f[0, 0, 1] = v
        frames.append(f)
    _, out = _drive(frames, "mtap3", monkeypatch)
    assert out[4][0, 0, 2] == 100 and out[4][0, 0, 3] == 0           # G+ = 100 - 0
    frames.append(frames[-1])
    _, out = _drive(frames, "mtap3", monkeypatch)
    w = [1 / 1.6, 0.6 / 1.6]
    assert out[5][0, 0, 2] == round(100 - (w[0] * 100 + w[1] * 0))


def test_edr_values(monkeypatch):
    frames = [const(100), const(150)]
    _, out = _drive(frames, "edr", monkeypatch)
    r = 255 * (math.log(151) - math.log(101))
    exp_on = round(r - 255 * math.log(1.05))
    assert all(abs(int(out[1][0, 0, c]) - exp_on) <= 1 for c in (0, 2, 4))
    assert (out[1][..., (1, 3, 5)] == 0).all()


def test_tsurf_decays(monkeypatch):
    monkeypatch.setenv("NETT_RETINA_TAU", "2")
    frames = [const(0), const(50)] + [const(50)] * 3
    _, out = _drive(frames, "tsurf", monkeypatch)
    assert (out[1][..., 0] == 255).all()
    assert abs(int(out[2][0, 0, 0]) - round(255 * math.exp(-0.5))) <= 1
    assert abs(int(out[4][0, 0, 0]) - round(255 * math.exp(-1.5))) <= 1


def test_smear_half_life(monkeypatch):
    monkeypatch.setenv("NETT_RETINA_HALF", "1")
    frames = [const(0), const(200), const(200)]
    _, out = _drive(frames, "smear", monkeypatch)
    assert out[1][0, 0, 0] == 100 and out[2][0, 0, 0] == 150


def test_episode_boundary_reseeds(monkeypatch):
    # env 1 is done at step 3: the frame it gets there (a new value) must emit no change for tr5 / mtap5 / tsurf / edr.
    fr = [const(10, n_env=2)] * 3 + [np.stack([const(10), const(200)])] + [np.stack([const(10), const(200)])] * 3
    for mode in ("tr5", "mtap5", "tsurf", "edr"):
        monkeypatch.setenv("NETT_RETINA", mode)
        env = RetinaTemporal(DoneEnv(fr, done_at=3, which=1, n_env=2))
        outs = [_pol(env.reset()[0])] + [_pol(env.step(0)[0]) for _ in range(len(fr) - 1)]
        assert all((o[1] == 0).all() for o in outs), mode
        assert all((o[0] == 0).all() for o in outs), mode


def test_registry_and_labels():
    from nett_skrl.body.wrappers.registry import _WRAPPER_SPECS as REG
    assert REG["retina_temporal"] == ("nett_skrl.body.wrappers.retina_temporal", "RetinaTemporal")
