"""MoTokSeg's Unity-parity training schedule (NETT_SEG_CADENCE / TRAIN_ON / QUANTIZE).

Reference: trainParsing.py:343-380 trains MoTok at each PPO update over every rollout frame in
stored order at batch 8, on the policy's own (masked, uint8-truncated) observation. All unset =
the online default, unchanged.
"""

import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.gwm_seg import GwmSeg
from nett_skrl.body.wrappers.motok_seg import MoTokSeg
from nett_skrl.body.wrappers.oracle_seg import OracleColorSeg


@pytest.fixture(autouse=True)
def defaults(monkeypatch):
    for name in list(os.environ):
        if name.startswith("NETT_SEG_") or name == "NETT_ROLLOUTS":
            monkeypatch.delenv(name)
    monkeypatch.setenv("NETT_SEG_DEVICE", "cpu")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        yield
    torch.set_num_threads(threads)


class Frames(gym.Env):
    num_envs = 4

    def __init__(self):
        self.observation_space = gym.spaces.Box(0, 255, (4, 16, 24, 3), np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self.rng = np.random.default_rng(0)

    def frame(self):
        return self.rng.integers(0, 256, (4, 16, 24, 3), dtype=np.uint8)


def _spy(w):
    """Record every batch the optimiser sees, as uint8 levels."""
    seen = []
    orig = w._opt_step

    def spy(batch):
        seen.append((batch * 255.0).round().to(torch.uint8).clone())
        return orig(batch)

    w._opt_step = spy
    return seen


def _run(monkeypatch, calls, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    src = Frames()
    w = MoTokSeg(src)
    seen = _spy(w)
    raws, outs = [], []
    for _ in range(calls):
        raw = src.frame()
        raws.append(raw)
        outs.append(w.observation(raw))
    return w, seen, raws, outs


def _nchw(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)


PARITY = dict(NETT_SEG_CADENCE="rollout", NETT_SEG_ROLLOUT_FRAMES="16", NETT_SEG_BATCH="8")


def test_rollout_cadence_one_ordered_pass_per_rollout(monkeypatch):
    w, seen, raws, _ = _run(monkeypatch, 4, **PARITY)        # 4 calls x 4 envs = 16 frames
    assert len(seen) == 2 and w.last_stats["seg/rollout_steps"] == 2.0
    assert w.last_stats["seg/train_steps"] == 2.0
    want = torch.cat([_nchw(r) for r in raws], dim=0)         # time-major, unshuffled
    assert torch.equal(torch.cat(seen, dim=0), want)          # raw frames by default
    assert w._roll == [] and w._roll_n == 0 and w._buf == []  # cleared; online buffer unused


def test_rollout_order_env_is_each_env_in_time_order(monkeypatch):
    w, seen, raws, _ = _run(monkeypatch, 4, NETT_SEG_ROLLOUT_ORDER="env", **PARITY)
    got = torch.cat(seen, dim=0)                              # 16 frames = 4 envs x 4 steps
    want = torch.cat([_nchw(np.stack([r[e] for r in raws])) for e in range(4)], dim=0)
    assert torch.equal(got, want)
    assert torch.equal(seen[0][:4], _nchw(np.stack([r[0] for r in raws])))   # batch 0 = env 0, t0..t3


def test_rollout_cadence_waits_for_the_full_rollout(monkeypatch):
    w, seen, _, _ = _run(monkeypatch, 3, **PARITY)
    assert seen == [] and w._roll_n == 12


def test_train_on_masked_trains_on_the_policy_observation(monkeypatch):
    w, seen, _, outs = _run(monkeypatch, 4, NETT_SEG_TRAIN_ON="masked", **PARITY)
    # the first rollout's masks come from the untrained segmenter, which is what the policy saw
    want = torch.cat([_nchw(o) for o in outs], dim=0)
    assert torch.equal(torch.cat(seen, dim=0), want)


def test_masked_differs_from_raw(monkeypatch):
    _, raw_seen, raws, _ = _run(monkeypatch, 4, **PARITY)
    torch.manual_seed(5)
    _, m_seen, _, _ = _run(monkeypatch, 4, NETT_SEG_TRAIN_ON="masked", **PARITY)
    assert not torch.equal(torch.cat(raw_seen), torch.cat(m_seen))


def test_floor_quantize_is_the_reference_truncation(monkeypatch):
    monkeypatch.setenv("NETT_SEG_QUANTIZE", "floor")
    src = Frames()
    w = MoTokSeg(src)
    raw = src.frame()
    out = w.observation(raw)
    x = _nchw(raw).float().div(255.0)
    fg = int(w.last_stats["seg/fg_slot"])
    with torch.no_grad():
        m = w._model.get_masks(x)[:, fg:fg + 1]
    ref = (_nchw(raw).float() * m).clamp(0, 255).numpy().astype(np.uint8)   # seg_wrappers.py:213
    assert np.array_equal(_nchw(out).numpy(), ref)


def test_floor_keeps_every_level_of_an_all_ones_mask(monkeypatch):
    monkeypatch.setenv("NETT_SEG_QUANTIZE", "floor")
    w = MoTokSeg(Frames())
    levels = torch.arange(256, dtype=torch.float32).view(1, 1, 16, 16) / 255.0
    got = w._masked_levels(levels, torch.ones_like(levels))
    assert torch.equal(got, torch.arange(256, dtype=torch.float32).view(1, 1, 16, 16))


def test_default_output_is_round_of_frame_times_mask(monkeypatch):
    src = Frames()
    w = MoTokSeg(src)
    raw = src.frame()
    out = w.observation(raw)
    fg = int(w.last_stats["seg/fg_slot"])
    x = _nchw(raw).float().div(255.0)
    with torch.no_grad():
        m = w._model.get_masks(x)[:, fg:fg + 1]
    ref = ((x * m).clamp(0, 1) * 255.0).round().to(torch.uint8)
    assert torch.equal(_nchw(out), ref)


def test_default_online_cadence_unchanged(monkeypatch):
    w, seen, _, _ = _run(monkeypatch, 4, NETT_SEG_TRAIN_EVERY="2", NETT_SEG_BATCH="2")
    assert len(seen) == 2 and w._roll == []
    assert seen[0].shape[0] == 2 * 4                           # 2 SAMPLES = 2 whole env batches


def test_rollout_frames_default_to_nett_rollouts(monkeypatch):
    monkeypatch.setenv("NETT_ROLLOUTS", "3072")
    assert MoTokSeg(Frames()).rollout_frames == 3072


def test_test_phase_neither_stores_nor_trains(monkeypatch, tmp_path):
    for k, v in PARITY.items():
        monkeypatch.setenv(k, v)
    src = Frames()
    tr = MoTokSeg(src)
    tr.bind_phase("train", tmp_path / "s.pt")
    tr.observation(src.frame())
    tr.save_state()
    te = MoTokSeg(src)
    te.bind_phase("test", tmp_path / "s.pt")
    seen = _spy(te)
    for _ in range(8):
        te.observation(src.frame())
    assert seen == [] and te._roll == []


@pytest.mark.parametrize("name,val", [("NETT_SEG_CADENCE", "update"), ("NETT_SEG_TRAIN_ON", "mask"),
                                      ("NETT_SEG_QUANTIZE", "trunc"), ("NETT_SEG_CADENCE", ""),
                                      ("NETT_SEG_ROLLOUT_ORDER", "env_major")])
def test_bad_values_refuse(monkeypatch, name, val):
    monkeypatch.setenv(name, val)
    with pytest.raises(ValueError, match="expected one of"):
        MoTokSeg(Frames())


def test_rollout_frames_must_be_positive(monkeypatch):
    monkeypatch.setenv("NETT_SEG_CADENCE", "rollout")
    monkeypatch.setenv("NETT_SEG_ROLLOUT_FRAMES", "0")
    with pytest.raises(ValueError, match=">= 1"):
        MoTokSeg(Frames())


@pytest.mark.parametrize("cls", [GwmSeg, OracleColorSeg])
@pytest.mark.parametrize("name,val", [("NETT_SEG_CADENCE", "rollout"), ("NETT_SEG_TRAIN_ON", "masked"),
                                      ("NETT_SEG_QUANTIZE", "floor"), ("NETT_SEG_ROLLOUT_ORDER", "env")])
def test_other_segmenters_refuse_the_knobs(monkeypatch, cls, name, val):
    monkeypatch.setenv(name, val)
    with pytest.raises(ValueError, match="MoTokSeg only"):
        cls(Frames())


def test_registered_replica_arm_is_the_recipe_policy_behind_motok(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train as campaign

    spec = campaign.MODELS["MoTok-Seg-UnityRecipe"]
    recipe = campaign.MODELS["CNN-UnityRecipe"]
    assert {k: v for k, v in spec.items() if k != "seg"} == recipe
    assert campaign.segmentation_wrappers(spec) == ["motok_seg"]


@pytest.mark.parametrize("brains,refused", [("2", True), ("7", True), ("1", False)])
def test_campaign_refuses_rollout_cadence_beside_several_brains(monkeypatch, brains, refused):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train as campaign

    monkeypatch.setenv("NETT_MODEL", "MoTok-Seg-UnityRecipe")
    monkeypatch.setenv("NETT_EXPERIMENT", "parsing")
    monkeypatch.setenv("NETT_SEG_CADENCE", "rollout")
    monkeypatch.setenv("NETT_BRAINS", brains)
    monkeypatch.setenv("NETT_TRAIN_EPS", "not-an-int")    # past the guard, main stops here
    with pytest.raises(ValueError) as e:
        campaign.main()
    assert ("needs NETT_BRAINS=1" in str(e.value)) is refused
