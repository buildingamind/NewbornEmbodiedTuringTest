"""MOTION program (owner 2026-11-20): the retina_temporal motion-sensor modes (emd, me, oms, loom, flow) -- spaces, a
static scene emits no motion, direction selectivity, egomotion cancellation, looming, and episode-boundary reseeding."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.retina_temporal import MOTION_MODES, N_OUT, RetinaTemporal, parse_mode

H, W = 32, 48


class Env(gym.Env):
    """Serves scripted (n_env, H, W, 3) uint8 frames; env ``done_env`` reports done at step ``done_at``."""

    def __init__(self, frames, done_at=None, done_env=0):
        self.frames, self.i, self.done_at, self.done_env = list(frames), 0, done_at, done_env
        n = self.frames[0].shape[0]
        self.n = n
        self.observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(low=0, high=255, shape=(n, H, W, 3), dtype=np.uint8)})
        self.action_space = gym.spaces.Discrete(2)

    def _emit(self):
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return {"policy": torch.as_tensor(f)}

    def reset(self, **kw):
        self.i = 0
        return self._emit(), {}

    def step(self, action):
        z = np.zeros(self.n, dtype=bool)
        if self.done_at is not None and self.i == self.done_at:
            z = z.copy()
            z[self.done_env] = True
        return self._emit(), np.zeros(self.n), z, np.zeros(self.n, dtype=bool), {}


def scene(xs, ys=None, *, bg=None, n_env=1, size=6, val=220):
    """One frame per entry of xs: a bright square at (x, y) on a background (flat 40, or the textured ``bg``)."""
    ys = ys if ys is not None else [H // 2 - size // 2] * len(xs)
    out = []
    for x, y in zip(xs, ys):
        f = np.full((H, W, 3), 40, np.uint8) if bg is None else bg.copy()
        f[y:y + size, x:x + size] = val
        out.append(np.broadcast_to(f, (n_env, H, W, 3)).copy())
    return out


def texture(shift=0):
    rng = np.random.default_rng(7)
    t = (rng.random((H, W + 64)) * 120 + 20).astype(np.uint8)
    t = np.repeat(np.repeat(t[::4, ::4], 4, 0), 4, 1)[:H, :W + 64]
    return np.repeat(t[:, shift:shift + W, None], 3, 2)


def run(frames, mode, monkeypatch, **kw):
    monkeypatch.setenv("NETT_RETINA", mode)
    env = RetinaTemporal(Env(frames, **kw))
    outs = [env.reset()[0]["policy"].numpy()]
    for _ in range(len(frames) - 1):
        outs.append(env.step(0)[0]["policy"].numpy())
    return env, outs


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("NETT_RETINA", "NETT_RETINA_THRESHOLD", "NETT_RETINA_TAU", "NETT_RETINA_HALF"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("mode", MOTION_MODES)
def test_parse_space_rgb_first_and_static_silence(mode, monkeypatch):
    assert parse_mode(mode) == (mode, None)
    fr = scene([10] * 8, n_env=2)
    env, outs = run(fr, mode, monkeypatch)
    n = N_OUT[mode]
    assert env.observation_space["policy"].shape == (2, H, W, n)
    for o in outs:
        assert o.shape == (2, H, W, n) and o.dtype == np.uint8
        np.testing.assert_array_equal(o[..., :3], fr[0])                  # colour passes through, first
        assert (o[..., 3:] == 0).all(), f"{mode}: a static scene must emit no motion"


@pytest.mark.parametrize("mode,pos,neg", [("emd", 3, 4), ("me", 3, 4), ("flow", 3, 4)])
def test_direction_selective_rightward(mode, pos, neg, monkeypatch):
    xs = list(range(8, 8 + 2 * 12, 2))                                     # 2 px / tick to the right
    _, outs = run(scene(xs), mode, monkeypatch)
    o = outs[-1][0].astype(float)
    assert o[..., pos].sum() > 3 * o[..., neg].sum() and o[..., pos].sum() > 0, (mode, o[..., pos].sum(), o[..., neg].sum())
    _, outs = run(scene(xs[::-1]), mode, monkeypatch)                    # and leftward flips it
    o = outs[-1][0].astype(float)
    assert o[..., neg].sum() > 3 * o[..., pos].sum(), mode


def test_oms_cancels_global_motion_keeps_object(monkeypatch):
    # (a) the whole textured image slides (self-motion): little OMS; (b) only a square moves on a still texture: OMS.
    glob = [np.broadcast_to(texture(2 * t), (1, H, W, 3)).copy() for t in range(10)]
    _, a = run(glob, "oms", monkeypatch)
    still = texture(0)
    obj = scene(list(range(8, 28, 2)), bg=still)
    _, b = run(obj, "oms", monkeypatch)
    oms_a = a[-1][0, ..., 3:].astype(float).mean()                    # anywhere, under global motion
    oms_b = b[-1][0, 10:22, 20:36, 3:].astype(float).mean()           # around the moving object
    assert oms_b > 10 * max(oms_a, 0.5), (oms_a, oms_b)
    assert (b[-1][0, :, :12, 3:] == 0).all()                           # the still background far away stays silent


def test_loom_prefers_expansion_over_translation(monkeypatch):
    sizes = [4, 6, 8, 10, 12, 14]
    loom = [scene([W // 2 - s // 2], [H // 2 - s // 2], size=s)[0] for s in sizes]
    _, a = run(loom, "loom", monkeypatch)
    _, b = run(scene([10, 12, 14, 16, 18, 20], size=8), "loom", monkeypatch)
    assert a[-1][0, ..., 4].mean() > 0 and a[-1][0, ..., 3].sum() > 0
    assert (a[-1][0, ..., 4] == a[-1][0, 0, 0, 4]).all()                 # the summed unit is broadcast


@pytest.mark.parametrize("mode", MOTION_MODES)
def test_episode_boundary_reseeds(mode, monkeypatch):
    # env 1 jumps to a new scene exactly at its episode boundary: nothing may difference across the cut.
    a = scene([10] * 3, n_env=1)[0][0]
    b = scene([30] * 3, n_env=1)[0][0]
    frames = [np.stack([a, a])] * 3 + [np.stack([a, b])] * 3
    _, outs = run(frames, mode, monkeypatch, done_at=3, done_env=1)
    for o in outs:
        assert (o[1, ..., 3:] == 0).all(), mode
        assert (o[0, ..., 3:] == 0).all(), mode


def test_dvsgm_cancels_a_global_brightness_step(monkeypatch):
    fr = [scene([10])[0], np.clip(scene([10])[0].astype(int) + 30, 0, 255).astype(np.uint8)]
    _, outs = run(fr, "dvsgm", monkeypatch)
    assert (outs[-1][..., 3:] == 0).all()
    _, outs = run(scene([10, 14]), "dvsgm", monkeypatch)              # a local change survives
    assert outs[-1][..., 3:].sum() > 0


def test_diag_writes_flow_and_channel_stats(monkeypatch, tmp_path):
    p = tmp_path / "diag.jsonl"
    monkeypatch.setenv("NETT_RETINA_DIAG", str(p))
    _, _ = run(scene(list(range(4, 34)), n_env=2, size=12), "oms", monkeypatch)
    import json
    recs = [json.loads(x) for x in p.read_text().splitlines()]
    assert recs and recs[0]["mode"] == "oms" and len(recs[0]["flow_median"]) == 2
    assert len(recs[0]["chan_mean"]) == 2 and len(recs[0]["chan_mean"][0]) == 2
    assert max(recs[0]["flow_p95"]) > 0.1                              # the 1 px/tick square is seen moving


@pytest.mark.parametrize("mode", MOTION_MODES)
def test_every_mode_responds_to_a_moving_object(mode, monkeypatch):
    _, outs = run(scene(list(range(8, 8 + 2 * 10, 2)), size=8), mode, monkeypatch)
    assert outs[-1][..., 3:].astype(float).sum() > 0, mode


# -- MOTION-ONLY pilot (presenter 2026-11-20): <mode>-only drops the 3 colour channels --------------------------------

@pytest.mark.parametrize("mode", MOTION_MODES)
def test_only_space_and_static_silence(mode, monkeypatch):
    assert parse_mode(mode + "-only") == (mode, None)
    fr = scene([10] * 6, n_env=2)
    env, outs = run(fr, mode + "-only", monkeypatch)
    n = N_OUT[mode] - 3
    assert env.observation_space["policy"].shape == (2, H, W, n)
    for o in outs:
        assert o.shape == (2, H, W, n) and o.dtype == np.uint8
        assert (o == 0).all(), f"{mode}-only: a static scene must emit nothing (no sustained channel)"


@pytest.mark.parametrize("mode", MOTION_MODES)
def test_only_carries_no_colour(mode, monkeypatch):
    # a static scene in two different colours, with no motion: the output is identical (all zero), so no static
    # appearance leaks into the observation
    a = scene([10] * 4, val=220)
    b = [np.clip(f.astype(int) * [1, 0, 0] + [0, 90, 30], 0, 255).astype(np.uint8) for f in scene([10] * 4, val=60)]
    _, oa = run(a, mode + "-only", monkeypatch)
    _, ob = run(b, mode + "-only", monkeypatch)
    for x, y in zip(oa, ob):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("mode", MOTION_MODES)
def test_only_equals_the_motion_channels_of_the_full_mode(mode, monkeypatch):
    fr = scene(list(range(8, 8 + 2 * 10, 2)), size=8)
    _, full = run(fr, mode, monkeypatch)
    _, only = run(fr, mode + "-only", monkeypatch)
    for f, o in zip(full, only):
        np.testing.assert_array_equal(f[..., 3:], o)
    assert only[-1].astype(float).sum() > 0, mode


@pytest.mark.parametrize("mode,pos,neg", [("emd", 0, 1), ("me", 0, 1), ("flow", 0, 1)])
def test_only_direction_selective(mode, pos, neg, monkeypatch):
    xs = list(range(8, 8 + 2 * 12, 2))
    _, outs = run(scene(xs), mode + "-only", monkeypatch)
    o = outs[-1][0].astype(float)
    assert o[..., pos].sum() > 3 * o[..., neg].sum() and o[..., pos].sum() > 0, mode


def test_only_diag_reports_every_channel(monkeypatch, tmp_path):
    import json
    p = tmp_path / "diag.jsonl"
    monkeypatch.setenv("NETT_RETINA_DIAG", str(p))
    run(scene(list(range(4, 34)), n_env=2, size=12), "emd-only", monkeypatch)
    rec = json.loads(p.read_text().splitlines()[0])
    assert len(rec["chan_mean"][0]) == 4                                # all 4 sensor channels, none dropped


def test_only_refuses_non_motion_modes():
    for bad in ("edr-only", "tr5-only", "rgb-only", "only"):
        with pytest.raises(ValueError):
            parse_mode(bad)
