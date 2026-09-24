"""OracleColorSeg: a parameter-free DIAGNOSTIC mask -- keeps red object pixels, zeroes the rest."""

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body import Body
from nett_skrl.body.wrappers.framestack import FrameStack
from nett_skrl.body.wrappers.oracle_seg import OracleColorSeg, red_object_mask


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    for name in list(os.environ):
        if name.startswith("NETT_SEG_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NETT_SEG_DEVICE", "cpu")


RED, SAND, SKY, WHITE, GREY = (150, 20, 15), (210, 180, 120), (40, 140, 220), (255, 255, 255), (90, 90, 90)


class Scene(gym.Env):
    """A red square on sand (left) and a white monitor (right), NHWC, like one imprint frame."""
    num_envs = 2
    device = "cpu"

    def __init__(self):
        self.img = np.zeros((2, 24, 32, 3), np.uint8)
        self.img[:] = GREY
        self.img[:, :, :16] = SAND
        self.img[:, 4:10, :16] = SKY
        self.img[:, 10:18, 4:12] = RED
        self.img[:, :, 16:] = WHITE
        self.observation_space = gym.spaces.Box(0, 255, self.img.shape, np.uint8)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, **kwargs):
        return self.img.copy(), {}

    def step(self, action):
        return self.img.copy(), np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), {}


def test_rule_keeps_only_the_red_object():
    t = torch.from_numpy(Scene().img).permute(0, 3, 1, 2).float() / 255
    m = red_object_mask(t)[0, 0]
    assert m[10:18, 4:12].all()
    assert m.sum() == 8 * 8                       # sand, sky, white and grey are all rejected


def test_desert_orange_is_rejected_but_shaded_red_is_kept():
    # Background A's red rock and the loose rule's leak band (sat ~0.6, g ~0.5 r) must be dropped;
    # the object's shaded side (dark but saturated red) must survive.
    px = torch.tensor([[[[200 / 255]], [[100 / 255]], [[60 / 255]]],       # orange rock
                       [[[70 / 255]], [[8 / 255]], [[6 / 255]]]])           # shaded object red
    m = red_object_mask(px)[:, 0, 0, 0]
    assert m.tolist() == [0.0, 1.0]


def test_registered_arm_masks_after_framestack(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    import campaign_train as campaign

    spec = campaign.MODELS["CNN2F+ORACLE-RedSeg"]
    control = campaign.MODELS["CNN2F+GWM-Seg"]
    for k in ("encoder", "cfg", "framestack", "seg_after"):
        assert spec[k] == control[k], k           # the ONLY difference is the mask source
    assert campaign.segmentation_wrappers(spec) == ["framestack", "oracle_seg"]
    wrapped = Body(wrappers=campaign.segmentation_wrappers(spec)).wrap(Scene())
    assert isinstance(wrapped.env, OracleColorSeg) and isinstance(wrapped.env.env, FrameStack)
    obs, _ = wrapped.reset()
    obs = np.asarray(obs)
    assert obs.shape == (2, 6, 24, 32)            # CHW after framestack: layout preserved
    for f in (obs[:, :3], obs[:, 3:]):            # BOTH frames of the stack are masked
        assert (f[:, :, 10:18, 4:12] > 0).all()
        keep = np.zeros((24, 32), bool); keep[10:18, 4:12] = True
        assert (f[:, :, ~keep] == 0).all()        # background and the white monitor are gone


def test_no_training_and_state_round_trip(tmp_path):
    w = OracleColorSeg(Scene())
    w.bind_phase("train", tmp_path / "s.pt")
    a = w.observation(Scene().img)
    assert w.train_step() is None
    path = w.save_state()
    t = OracleColorSeg(Scene())
    t.bind_phase("test", path)
    b = t.observation(Scene().img)
    assert np.array_equal(a, b)
    assert t.last_stats["seg/loaded"] == 1.0 and t.last_stats["seg/frozen"] == 1.0
    assert t.last_stats["seg/fg_slot"] == 0.0
