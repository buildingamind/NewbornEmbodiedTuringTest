"""OracleShapeGate: the oracle red mask, each component scaled by global-shape familiarity (DIAGNOSTIC)."""

import math
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from nett_skrl.body import Body
from nett_skrl.body.wrappers.framestack import FrameStack
from nett_skrl.body.wrappers.oracle_seg import OracleColorSeg
from nett_skrl.body.wrappers.oracle_shape import OracleShapeGate, component_solidity


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("NETT_SEG_", "NETT_SHAPE_GATE_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NETT_SEG_DEVICE", "cpu")


RED, SAND, WHITE = (150, 20, 15), (210, 180, 120), (255, 255, 255)
SQUARE = (slice(8, 22), slice(4, 18))            # 14x14 solid: solidity 1.0, the imprint object


def plus_mask():
    """A 15x15 plus with 5-px arms at rows 6..20, cols 36..50: solidity ~0.7, the novel object."""
    m = np.zeros((32, 64), bool)
    m[6:21, 41:46] = True
    m[11:16, 36:51] = True
    return m


class Scene(gym.Env):
    """Sand ground, red objects, a white monitor strip. ``objs``: 'square', 'plus', 'dot'."""
    num_envs = 2
    device = "cpu"

    def __init__(self, objs=("square",)):
        img = np.zeros((2, 32, 64, 3), np.uint8)
        img[:] = SAND
        img[:, :, 56:] = WHITE
        if "square" in objs:
            img[:, SQUARE[0], SQUARE[1]] = RED
        if "plus" in objs:
            img[:, plus_mask()] = RED
        if "dot" in objs:
            img[:, 26:30, 26:30] = RED                    # 4 px: below NETT_SHAPE_GATE_MIN_PX
        self.img = img
        self.observation_space = gym.spaces.Box(0, 255, img.shape, np.uint8)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, **kwargs):
        return self.img.copy(), {}

    def step(self, action):
        return self.img.copy(), np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), {}


def trained(tmp_path, objs=("square",)):
    w = OracleShapeGate(Scene(objs))
    w.bind_phase("train", tmp_path / "s.pt")
    w.observation(Scene(objs).img)
    return w, w.save_state()


def test_solidity_separates_the_test_shapes():
    _, sq = component_solidity(Scene(("square",)).img[0, :, :, 0].__eq__(RED[0]).astype(np.uint8), 10)
    _, pl = component_solidity(plus_mask().astype(np.uint8), 10)
    assert list(sq.values()) == [pytest.approx(1.0)]
    (s,) = pl.values()
    assert 0.6 < s < 0.8


def test_template_is_learned_in_train_and_frozen_in_test(tmp_path):
    w, path = trained(tmp_path)
    assert w.last_stats["seg/gate_template"] == pytest.approx(1.0)
    n_train = w.last_stats["seg/gate_template_n"]
    assert n_train == 2.0                                   # one per env frame
    t = OracleShapeGate(Scene(("square", "plus")))
    t.bind_phase("test", path)
    for _ in range(3):
        t.observation(Scene(("square", "plus")).img)
    # the plus is never folded into the template: the test phase only reads it
    assert t.last_stats["seg/gate_template_n"] == n_train
    assert t.last_stats["seg/gate_template"] == pytest.approx(1.0)
    assert t.last_stats["seg/loaded"] == 1.0 and t.last_stats["seg/frozen"] == 1.0


def test_novel_shape_is_attenuated_familiar_shape_is_kept(tmp_path):
    _, path = trained(tmp_path)
    t = OracleShapeGate(Scene(("square", "plus")))
    t.bind_phase("test", path)
    obs = t.observation(Scene(("square", "plus")).img)
    raw = Scene(("square", "plus")).img
    assert np.array_equal(obs[:, SQUARE[0], SQUARE[1]], raw[:, SQUARE[0], SQUARE[1]])  # g = 1
    _, pl = component_solidity(plus_mask().astype(np.uint8), 10)
    g = math.exp(-0.5 * ((next(iter(pl.values())) - 1.0) / 0.15) ** 2)
    assert g < 0.3
    got = obs[:, plus_mask(), 0].astype(float)
    assert np.allclose(got, RED[0] * g, atol=1.0)                                     # scaled by g
    assert (obs[:, :, 56:] == 0).all()                                                # monitor gone
    assert t.last_stats["seg/gate_min"] == pytest.approx(g)


def test_unmeasurable_component_gets_the_neutral_gate(tmp_path):
    _, path = trained(tmp_path)
    t = OracleShapeGate(Scene(("square", "dot")))
    t.bind_phase("test", path)
    obs = t.observation(Scene(("square", "dot")).img)
    assert np.array_equal(obs[:, 26:30, 26:30], Scene(("dot",)).img[:, 26:30, 26:30])


def test_no_template_yet_is_exactly_the_oracle(tmp_path):
    # the very first train frames: nothing learned, so the gate must not act
    w = OracleShapeGate(Scene(("square", "plus")))
    w._learning = False
    w.bind_phase("test", tmp_path / "missing.pt", allow_missing=True)
    o = OracleColorSeg(Scene(("square", "plus")))
    o.bind_phase("test", tmp_path / "missing2.pt", allow_missing=True)
    img = Scene(("square", "plus")).img
    assert np.array_equal(w.observation(img), o.observation(img))
    assert math.isnan(w.last_stats["seg/gate_template"])


def test_sigma_widens_the_gate(tmp_path, monkeypatch):
    _, path = trained(tmp_path)
    monkeypatch.setenv("NETT_SHAPE_GATE_SIGMA", "10")
    t = OracleShapeGate(Scene(("square", "plus")))
    t.bind_phase("test", path)
    obs = t.observation(Scene(("square", "plus")).img)
    assert (obs[:, plus_mask(), 0] >= RED[0] - 1).all()


@pytest.mark.parametrize("name,val", [("NETT_SHAPE_GATE_SIGMA", "0"), ("NETT_SHAPE_GATE_MIN_PX", "2")])
def test_bad_knobs_refuse(monkeypatch, name, val):
    monkeypatch.setenv(name, val)
    with pytest.raises(ValueError):
        OracleShapeGate(Scene())


def test_registered_arm_differs_from_the_oracle_only_in_seg(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    import campaign_train as campaign

    spec = campaign.MODELS["CNN2F+ORACLE-ShapeGate"]
    oracle = campaign.MODELS["CNN2F+ORACLE-RedSeg"]
    assert {k: v for k, v in spec.items() if k != "seg"} == {k: v for k, v in oracle.items() if k != "seg"}
    assert campaign.segmentation_wrappers(spec) == ["framestack", "oracle_shape"]
    wrapped = Body(wrappers=campaign.segmentation_wrappers(spec)).wrap(Scene())
    assert isinstance(wrapped.env, OracleShapeGate) and isinstance(wrapped.env.env, FrameStack)
    obs, _ = wrapped.reset()
    obs = np.asarray(obs)
    assert obs.shape == (2, 6, 32, 64)                      # CHW after framestack
    for f in (obs[:, :3], obs[:, 3:]):                      # BOTH frames masked
        keep = np.zeros((32, 64), bool); keep[SQUARE] = True
        assert (f[:, :, keep] > 0).all() and (f[:, :, ~keep] == 0).all()


def test_template_follows_the_largest_component(tmp_path):
    # square 196 px vs plus 125 px: the imprint object is the larger mass, whatever else is visible
    w, _ = trained(tmp_path, objs=("square", "plus"))
    assert w.last_stats["seg/gate_template"] == pytest.approx(1.0)


def test_test_phase_refuses_an_empty_template(tmp_path, monkeypatch):
    # train saw nothing measurable (only the 4-px dot): testing would be the oracle in disguise
    _, path = trained(tmp_path, objs=("dot",))
    t = OracleShapeGate(Scene(("square", "plus")))
    with pytest.raises(RuntimeError, match="template_n == 0"):
        t.bind_phase("test", path)
        t.observation(Scene(("square", "plus")).img)
    monkeypatch.setenv("NETT_SEG_ALLOW_UNTRAINED", "1")
    t = OracleShapeGate(Scene(("square", "plus")))
    t.bind_phase("test", path)
    t.observation(Scene(("square", "plus")).img)
    assert t.last_stats["seg/gate_template_n"] == 0.0


def test_resolved_knobs_are_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("NETT_SHAPE_GATE_SIGMA", "0.3")
    w, _ = trained(tmp_path)
    assert w.last_stats["seg/gate_sigma"] == 0.3 and w.last_stats["seg/gate_min_px"] == 10.0


def test_pre_gate_stats(tmp_path):
    _, path = trained(tmp_path)
    t = OracleShapeGate(Scene(("square", "plus")))
    t.bind_phase("test", path)
    t.observation(Scene(("square", "plus")).img)
    _, pl = component_solidity(plus_mask().astype(np.uint8), 10)
    g = math.exp(-0.5 * ((next(iter(pl.values())) - 1.0) / 0.15) ** 2)
    n_sq, n_pl = 14 * 14, int(plus_mask().sum())
    assert t.last_stats["seg/gate_raw_area"] == pytest.approx((n_sq + n_pl) / (32 * 64))
    assert t.last_stats["seg/gate_kept_frac"] == pytest.approx((n_sq + g * n_pl) / (n_sq + n_pl), rel=1e-5)
    assert t.last_stats["seg/gate_largest"] == pytest.approx(1.0)       # the square is the largest
