"""The body segmenter survives the train -> test subprocess boundary, frozen at test.

⛔ Every mode runs in a FRESH process and the segmenter lives in the observation
wrapper, not in the skrl agent checkpoint. Before this, every segmenter arm's test
phase masked through a random init that kept training on the test stimuli.
"""

import logging
import os
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.gwm_seg import GwmSeg
from nett_skrl.body.wrappers.motok_seg import MoTokSeg
from nett_skrl.body.wrappers.segmentation import (
    body_has_segmenter,
    find_segmenters,
    state_filename,
)
from nett_skrl.runtime import task_runner


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    for name in list(os.environ):
        if name.startswith("NETT_SEG_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NETT_SEG_DEVICE", "cpu")
    monkeypatch.setenv("NETT_SEG_TRAIN_EVERY", "1")
    monkeypatch.setenv("NETT_SEG_BATCH", "2")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    torch.set_num_threads(threads)


class Frames(gym.Env):
    """A different frame every step, so the segmenter genuinely learns."""

    num_envs = 2

    def __init__(self, channels):
        self.shape = (self.num_envs, 24, 32, channels)
        self.observation_space = gym.spaces.Box(0, 255, self.shape, np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.rng = np.random.default_rng(3)

    def _obs(self):
        return self.rng.integers(0, 256, self.shape, dtype=np.uint8)

    def reset(self, **kwargs):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), np.zeros(2), np.zeros(2, bool), np.zeros(2, bool), {}


# GwmSeg sits after framestack (6+ channels); MoTokSeg before it (3 channels).
ARMS = [(GwmSeg, 6), (MoTokSeg, 3)]


def _train(cls, channels, steps=6):
    env = cls(Frames(channels))
    env.reset()
    for _ in range(steps):
        env.step(0)
    assert env.last_stats["seg/train_steps"] > 0
    return env


def _probe(channels):
    return np.random.default_rng(99).integers(0, 256, (2, 24, 32, channels), dtype=np.uint8)


@pytest.mark.parametrize("cls,channels", ARMS)
def test_test_phase_masks_with_the_trained_segmenter(cls, channels, tmp_path):
    trained = _train(cls, channels)
    path = trained.save_state(tmp_path / state_filename(0))
    steps = trained.last_stats["seg/train_steps"]
    trained.train_every = 0
    x = _probe(channels)
    expected = trained.observation(x)

    tested = cls(Frames(channels))
    tested.bind_phase("test", path)
    got = tested.observation(x)
    np.testing.assert_array_equal(got, expected)
    assert tested.last_stats["seg/loaded"] == 1.0
    assert tested.last_stats["seg/frozen"] == 1.0

    # Control: an unloaded segmenter masks differently -- the equality above is not vacuous.
    fresh = cls(Frames(channels))
    fresh.train_every = 0
    assert not np.array_equal(fresh.observation(x), expected)

    # Frozen: many test observations, no update, weights bit-identical.
    before = {k: v.clone() for k, v in tested._model.state_dict().items()}
    for _ in range(10):
        tested.step(0)
    assert tested.last_stats["seg/train_steps"] == steps
    assert all(torch.equal(before[k], v) for k, v in tested._model.state_dict().items())
    np.testing.assert_array_equal(tested.observation(x), expected)


@pytest.mark.parametrize("cls,channels", ARMS)
def test_bind_after_model_built_still_loads(cls, channels, tmp_path):
    """A reset before bind_phase builds a random model; the bind must replace it."""
    trained = _train(cls, channels)
    path = trained.save_state(tmp_path / state_filename(0))
    trained.train_every = 0
    x = _probe(channels)
    tested = cls(Frames(channels))
    tested.train_every = 0
    tested.reset()
    tested.bind_phase("record", path)
    np.testing.assert_array_equal(tested.observation(x), trained.observation(x))


@pytest.mark.parametrize("cls,channels", ARMS)
def test_missing_state_raises_unless_allowed(cls, channels, tmp_path, caplog):
    env = cls(Frames(channels))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        env.bind_phase("test", tmp_path / "absent.pt")
    env = cls(Frames(channels))
    with caplog.at_level(logging.WARNING):
        env.bind_phase("test", tmp_path / "absent.pt", allow_missing=True)
    assert "WITHOUT trained segmenter weights" in caplog.text
    assert env.train_every == 0 and env.last_stats["seg/loaded"] == 0.0


def test_state_from_another_segmenter_is_refused(tmp_path):
    path = _train(MoTokSeg, 3).save_state(tmp_path / "s.pt")
    with pytest.raises(ValueError, match="holds a MoTokSeg"):
        GwmSeg(Frames(6)).bind_phase("test", path)


def test_resumed_train_chunk_continues_weights_and_optimizer(tmp_path):
    first = _train(GwmSeg, 6, steps=4)
    path = first.save_state(tmp_path / "s.pt")
    steps = first.last_stats["seg/train_steps"]
    second = GwmSeg(Frames(6))
    second.bind_phase("train", path, resume=True)
    second.reset()
    assert second.last_stats["seg/loaded"] == 1.0
    assert second.last_stats["seg/train_steps"] >= steps  # continued, not restarted
    st1, st2 = first._optim.state_dict()["state"], second._optim.state_dict()["state"]
    assert st1 and st2.keys() == st1.keys()
    # a fresh train chunk ignores any file
    third = GwmSeg(Frames(6))
    third.bind_phase("train", path, resume=False)
    third.reset()
    assert third.last_stats["seg/loaded"] == 0.0


def test_save_without_a_frame_raises(tmp_path):
    with pytest.raises(RuntimeError, match="ever seeing a frame"):
        GwmSeg(Frames(6)).save_state(tmp_path / "s.pt")


# ---- task_runner wiring -------------------------------------------------------------

def _cfg(tmp_path, **kw):
    base = dict(path=tmp_path, dry_run=False, train_start_step=None, brain_id_offset=0,
                logger=logging.getLogger("t"))
    base.update(kw)
    return SimpleNamespace(**base)


def test_preflight_fails_before_kit_for_an_untrained_test(tmp_path, monkeypatch):
    monkeypatch.delenv("NETT_SEG_ALLOW_UNTRAINED", raising=False)
    body = SimpleNamespace(wrappers=[GwmSeg])
    assert body_has_segmenter(body.wrappers)
    assert task_runner._seg_state_preflight(body, "train", _cfg(tmp_path)) == tmp_path / "seg_state_off0.pt"
    for mode in ("test", "record"):
        with pytest.raises(FileNotFoundError, match="written by the train phase"):
            task_runner._seg_state_preflight(body, mode, _cfg(tmp_path))
    with pytest.raises(FileNotFoundError):
        task_runner._seg_state_preflight(body, "train", _cfg(tmp_path, train_start_step=100))
    # a dry-run probe, and the explicit legacy escape, may run without it
    assert task_runner._seg_state_preflight(body, "test", _cfg(tmp_path, dry_run=True))
    monkeypatch.setenv("NETT_SEG_ALLOW_UNTRAINED", "1")
    assert task_runner._seg_state_preflight(body, "test", _cfg(tmp_path))
    # offsets do not share a file
    assert task_runner._seg_state_preflight(body, "train", _cfg(tmp_path, brain_id_offset=3)).name == "seg_state_off3.pt"


def test_preflight_is_inert_without_a_segmenter(tmp_path):
    assert task_runner._seg_state_preflight(SimpleNamespace(wrappers=[]), "test", _cfg(tmp_path)) is None
    task_runner._bind_segmenters(object(), "test", _cfg(tmp_path), None)
    task_runner._save_segmenters(object(), None, logging.getLogger("t"))


class Outer(gym.Wrapper):
    pass


def test_runner_round_trip_through_a_wrapper_chain(tmp_path, monkeypatch):
    """train binds+saves via the chain walk; test binds+loads and is frozen."""
    monkeypatch.delenv("NETT_SEG_ALLOW_UNTRAINED", raising=False)
    body = SimpleNamespace(wrappers=[GwmSeg])
    cfg = _cfg(tmp_path)
    x = _probe(6)

    path = task_runner._seg_state_preflight(body, "train", cfg)
    train_env = Outer(GwmSeg(Frames(6)))
    assert len(find_segmenters(train_env)) == 1
    task_runner._bind_segmenters(train_env, "train", cfg, path)
    train_env.reset()
    for _ in range(5):
        train_env.step(0)
    task_runner._save_segmenters(train_env, path, cfg.logger)
    assert path.exists()
    task_runner._record_segmenters(train_env, "train", path, cfg)
    seg = find_segmenters(train_env)[0]
    seg.train_every = 0
    expected = seg.observation(x)

    path = task_runner._seg_state_preflight(body, "test", cfg)
    test_env = Outer(GwmSeg(Frames(6)))
    task_runner._bind_segmenters(test_env, "test", cfg, path)
    tseg = find_segmenters(test_env)[0]
    np.testing.assert_array_equal(tseg.observation(x), expected)
    assert tseg.train_every == 0
    task_runner._record_segmenters(test_env, "test", path, cfg)

    import hashlib
    import json
    tr = json.loads((tmp_path / "seg_phase_train_off0.json").read_text())
    te = json.loads((tmp_path / "seg_phase_test_off0.json").read_text())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert tr["state_sha256"] == te["state_sha256"] == sha
    assert tr["stats"]["seg/frozen"] == 0.0 and tr["train_every"] > 0
    assert te["stats"]["seg/loaded"] == 1.0 and te["stats"]["seg/frozen"] == 1.0 and te["train_every"] == 0
    assert te["stats"]["seg/train_steps"] == tr["stats"]["seg/train_steps"] > 0


def test_phase_record_never_raises(tmp_path, caplog):
    """An instrument must not cost a finished phase: a broken chain is logged, not raised."""
    cfg = _cfg(tmp_path)
    with caplog.at_level(logging.ERROR):
        task_runner._record_segmenters(object(), "test", tmp_path / "s.pt", cfg)
    assert "NOT written" in caplog.text
    task_runner._record_segmenters(object(), "test", None, cfg)  # no segmenter: inert


def test_mode_body_wires_preflight_bind_and_save_in_order():
    """The helpers are only a fix if the real mode body calls them, in this order."""
    import inspect

    src = inspect.getsource(task_runner._run_single_mode_body)
    order = [
        "_seg_state_preflight(agent.body, mode, run_config)",
        "loaded = agent.body.embed(agent.env, run_config)",
        "_bind_segmenters(loaded, mode, run_config, seg_state)",
        "agent.brain.train(",
        "agent.brain.test(",
        "_save_segmenters(loaded, seg_state, run_config.logger)",
        "_record_segmenters(loaded, mode, seg_state, run_config)",
        "_finalize_env_artifacts(loaded, run_config.logger)",
    ]
    at = [src.find(s) for s in order]
    assert all(i >= 0 for i in at) and all(src.count(s) == 1 for s in order), dict(zip(order, at))
    assert at == sorted(at), dict(zip(order, at))
    save = src[: at[5]].rsplit("\n", 2)[-2]
    assert 'mode == "train"' in save and "not config.dry_run" in save


def test_gate_fires_on_every_real_segmenter_body(monkeypatch):
    """⛔ The gate is inert if it misses a real Body. Build every campaign MODEL's body the
    way the driver does and require: segmenter arms True, every other arm False."""
    from pathlib import Path

    from nett_skrl.body import Body

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train as campaign

    seg_arms, plain = [], []
    for name, spec in campaign.MODELS.items():
        body = Body(wrappers=campaign.segmentation_wrappers(spec))
        (seg_arms if spec.get("seg") else plain).append(name)
        assert body_has_segmenter(body.wrappers) == bool(spec.get("seg")), name
    for must in ("CNN2F+GWM-Seg", "CNN2F+GWM-Seg-Q3", "CNN2F+GWM-Seg-Q5", "MoTok-Seg", "MoTok-Seg2F"):
        assert must in seg_arms, must
    assert plain
