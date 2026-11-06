"""NETT_DVS_MOTION_COMP / NETT_DVS_GATE_K / NETT_DVS_GLOBAL_SUB (workspace FINDINGS add. 351, arms MC / GATE / GFS).

The definitions are the egomotion probe's (ego/probe/evlib.py + ana.py). The parity tests load the probe's own evlib.py
and recorded ticks (ego/probe/cap/) and compare the wrapper's output to the probe's functions EXACTLY; they skip on a
host without the probe directory. Everything else is synthetic.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity, _quat_to_R, _Rz
from test_dvs_polarity import FakeEnv, H, W, _pol, const

PROBE = Path("/home/zlaborde/nett/scratch/researcher2/ego/probe")
CAP = PROBE / "cap" / "cap_130616_000.npz"
KNOBS = ("NETT_DVS_MOTION_COMP", "NETT_DVS_GATE_K", "NETT_DVS_GLOBAL_SUB", "NETT_DVS_MOVE_GAIN", "NETT_DVS_WINDOW",
         "NETT_DVS_WINDOW_TAU", "NETT_DECISION_LOOP", "NETT_DVS_THRESHOLD", "NETT_DVS_BLUR", "NETT_DVS_BLUR_KERNEL",
         "NETT_DVS_COLOR", "NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_FISHEYE_METHOD")
EGO = ("NETT_DVS_MOTION_COMP", "NETT_DVS_GATE_K", "NETT_DVS_GLOBAL_SUB")
ARM = {"NETT_DVS_MOTION_COMP": "yaw", "NETT_DVS_GATE_K": "0.5", "NETT_DVS_GLOBAL_SUB": "median:4"}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in KNOBS:
        monkeypatch.delenv(k, raising=False)


def _recipe(monkeypatch):
    """The arms' DVS: colour, threshold 4, blur off (add. 351 recipe)."""
    monkeypatch.setenv("NETT_DVS_COLOR", "1")
    monkeypatch.setenv("NETT_DVS_THRESHOLD", "4")
    monkeypatch.setenv("NETT_DVS_BLUR", "0")


class PoseEnv(gym.Env):
    """Batched (N,H,W,3) frames, plus what NETT_DVS_MOTION_COMP reads from repo B's NETTEnv: ``cfg.observation`` (the
    lens) and ``_cameras["center"].data.quat_w_world`` (wxyz), set per reset / step from a schedule."""

    def __init__(self, frames, quats, *, eye=(128, 80), fov=300.0, cameras=("center",), as_tensor=True):
        self.frames, self.quats, self.i, self.as_tensor = list(frames), list(quats), 0, as_tensor
        n, h, w, c = np.shape(frames[0])
        self.n = n
        box = gym.spaces.Box(low=0, high=255, shape=(n, h, w, c), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({"policy": box})
        self.action_space = gym.spaces.Box(-1, 1, (2,))
        self.cfg = SimpleNamespace(observation=SimpleNamespace(eye_resolution=eye, fov=fov))
        self._cameras = {name: SimpleNamespace(data=SimpleNamespace(quat_w_world=None)) for name in cameras}

    def _emit(self):
        k = min(self.i, len(self.frames) - 1)
        for cam in self._cameras.values():
            cam.data.quat_w_world = torch.as_tensor(np.asarray(self.quats[k], dtype=np.float32))
        self.i += 1
        f = self.frames[k]
        return {"policy": torch.as_tensor(f) if self.as_tensor else np.asarray(f)}

    def reset(self, **kw):
        self.i = 0
        return self._emit(), {}

    def step(self, action):
        z = np.zeros(self.n, dtype=bool)
        return self._emit(), np.zeros(self.n), z, z.copy(), {}


# ---------------------------------------------------------------------------------------------------- unset identity
def _base_module():
    """dvs_polarity.py as committed at 7188932 (before these knobs), loaded under the same package."""
    root = Path(__file__).resolve().parents[2]
    try:
        src = subprocess.run(["git", "-C", str(root), "show", "7188932:src_isaac/nett_skrl/body/wrappers/dvs_polarity.py"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history with 7188932 not available")
    name = "nett_skrl.body.wrappers._dvs_polarity_7188932"
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    mod.__package__ = "nett_skrl.body.wrappers"
    exec(compile(src, "dvs_polarity@7188932", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("color, blur", [("1", "0"), ("1", "1"), ("0", "1"), ("0", "0")])
@pytest.mark.parametrize("as_tensor", [True, False])
def test_unset_is_bit_identical_to_the_base_wrapper(monkeypatch, color, blur, as_tensor):
    monkeypatch.setenv("NETT_DVS_COLOR", color)
    monkeypatch.setenv("NETT_DVS_BLUR", blur)
    monkeypatch.setenv("NETT_DVS_THRESHOLD", "4")
    base = _base_module().DVSPolarity
    rng = np.random.default_rng(351)
    frames = [rng.integers(0, 256, (3, H, W, 3)).astype(np.uint8) for _ in range(6)]
    outs = []
    for cls in (base, DVSPolarity):
        env = cls(FakeEnv(frames, n_env=3, as_tensor=as_tensor))
        out = [_pol(env.reset()[0])]
        out += [_pol(env.step(np.zeros((3, 2), np.float32))[0]) for _ in range(5)]
        outs.append(out)
    new = DVSPolarity(FakeEnv(frames, n_env=3))
    assert (new.motion_comp, new.gate_k, new.global_sub) == (None, None, None)
    assert all(a.dtype == b.dtype and np.array_equal(a, b) for a, b in zip(*outs))


# ---------------------------------------------------------------------------------------------- the probe's own code
@pytest.fixture(scope="module")
def probe():
    """The probe's evlib.py and a handful of recorded ticks. evlib.py prepends a repo B path to sys.path at import;
    the path is restored afterwards so it cannot shadow this suite's nett_isaac."""
    if not (PROBE / "evlib.py").is_file() or not CAP.is_file():
        pytest.skip("add. 351 probe directory not on this host")
    import nett_isaac.lens  # noqa: F401  (resolved from this suite's path before evlib touches sys.path)

    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("_add351_evlib", PROBE / "evlib.py")
        evlib = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evlib)
    finally:
        sys.path[:] = saved
    cap = dict(np.load(CAP))
    sb, sie = cap["step_before"], cap["sie"]
    pairs = []
    for t in range(1, len(sb)):                                         # ana.py's in-episode consecutive-tick rule
        ok = (sb[t] == sb[t - 1] + 1) & (sie[t - 1] == sb[t - 1] + 1) & (sie[t] == sb[t] + 1)
        pairs += [(t, e) for e in np.flatnonzero(ok)]
    R = evlib.quat_to_R(cap["quat"].reshape(-1, 4).astype(np.float64)).reshape(*cap["quat"].shape[:2], 3, 3)
    dyaw = {p: abs(math.degrees(math.remainder(float(evlib.yaw_of(R[p[0], p[1]][None])[0]
                                                     - evlib.yaw_of(R[p[0] - 1, p[1]][None])[0]), 2 * math.pi)))
            for p in pairs}
    turning = [p for p in pairs if dyaw[p] >= 10][:6]
    still = [p for p in pairs if dyaw[p] < 1][:2]
    assert len(turning) == 6 and len(still) == 2
    return SimpleNamespace(evlib=evlib, cap=cap, pairs=turning + still, dyaw=dyaw)


def _run_pairs(monkeypatch, pr, pairs, **kw):
    """Feed recorded (t-1, t) pairs through the wrapper as one batched env (one pair per env); return the step's
    events (N, 6, H, W) uint8 (CHW for comparison with the probe's maps)."""
    cap = pr.cap
    t, e = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    env = DVSPolarity(PoseEnv([cap["rgb"][t - 1, e], cap["rgb"][t, e]], [cap["quat"][t - 1, e], cap["quat"][t, e]], **kw))
    env.reset()
    out = _pol(env.step(torch.as_tensor(cap["act"][t, e]))[0])
    return np.moveaxis(out, -1, 1), t, e


def test_motion_comp_equals_the_probe_on_recorded_ticks(monkeypatch, probe):
    """ana.py (b) 'b_yaw': evlib.warp_prev(yaw_only=True) with the fitted (-1, -1) convention, events at thr 4, x ok."""
    _recipe(monkeypatch)
    monkeypatch.setenv("NETT_DVS_MOTION_COMP", "yaw")
    got, t, e = _run_pairs(monkeypatch, probe, probe.pairs)
    ev, cap = probe.evlib, probe.cap
    R = ev.quat_to_R(cap["quat"].reshape(-1, 4).astype(np.float64)).reshape(*cap["quat"].shape[:2], 3, 3)
    w, ok = ev.warp_prev(cap["rgb"][t - 1, e], R[t - 1, e], R[t, e], ev.Fisheye(-1.0, -1.0), yaw_only=True)
    want = (ev.events_from_diff(ev.planes(cap["rgb"][t, e]) - w, 4.0) * ok[:, None]).numpy() * 255
    assert np.array_equal(got, want.astype(np.uint8))
    plain = ev.events(cap["rgb"][t - 1, e], cap["rgb"][t, e], 4.0, 0).numpy() * 255
    assert got[:6].sum() < plain[:6].sum()                              # the turning pairs lose events
    assert not np.array_equal(got[:6], plain[:6].astype(np.uint8))


def test_gate_equals_the_probe_on_recorded_ticks(monkeypatch, probe):
    """ana.py (c): g = max(0, 1 - k |clip(a,-1,1)| / sqrt 2) on the tick's action; the probe scaled float event MASS by
    g, the wrapper scales each 255 event to round(255 g) (uint8, as NETT_DVS_MOVE_GAIN). So the comparison is exact
    against round(255 g) x the probe's 0/1 plain map."""
    _recipe(monkeypatch)
    monkeypatch.setenv("NETT_DVS_GATE_K", "0.5")
    got, t, e = _run_pairs(monkeypatch, probe, probe.pairs)
    ev, cap = probe.evlib, probe.cap
    amag = np.linalg.norm(np.clip(cap["act"][t, e], -1, 1), axis=-1) / np.sqrt(2)
    g = np.maximum(0.0, 1 - 0.5 * amag)
    plain = ev.events(cap["rgb"][t - 1, e], cap["rgb"][t, e], 4.0, 0).numpy()
    want = np.round(255 * g.astype(np.float64))[:, None, None, None] * plain
    assert np.array_equal(got, want.astype(np.uint8))
    assert np.any((g > 0) & (g < 1))                                    # the recorded actions exercise the gate


def test_global_sub_equals_the_probe_on_recorded_ticks(monkeypatch, probe):
    """ana.py (d)(ii) 'd2_r4': d - scipy median_filter(d, (1,1,9,9), mode='nearest'), events at thr 4."""
    from scipy import ndimage as ndi

    _recipe(monkeypatch)
    monkeypatch.setenv("NETT_DVS_GLOBAL_SUB", "median:4")
    got, t, e = _run_pairs(monkeypatch, probe, probe.pairs)
    ev, cap = probe.evlib, probe.cap
    d = ev.diff(cap["rgb"][t - 1, e], cap["rgb"][t, e], 0).numpy()
    med = ndi.median_filter(d, size=(1, 1, 9, 9), mode="nearest")
    want = ev.events_from_diff(torch.as_tensor(d - med), 4.0).numpy() * 255
    assert np.array_equal(got, want.astype(np.uint8))


# ------------------------------------------------------------------------------------------- synthetic behaviour
def _render(R: np.ndarray, eye=(128, 80)) -> np.ndarray:
    """A smooth world-fixed colour panorama seen through the 300 deg equisolid eye at body->world rotation R (N,3,3):
    each pixel's ray from the lens law (u right = camera -Y, v down = camera -Z), turned into the world, coloured by
    its azimuth / elevation. Outside the image circle = 0 (the policy image is rim-masked)."""
    from nett_isaac.lens import LensSpec

    L = LensSpec(method="equisolid", fov_h_deg=300.0, width=eye[0], height=eye[1])
    v, u = np.meshgrid(np.arange(L.height) + 0.5, np.arange(L.width) + 0.5, indexing="ij")
    du, dv = u - L.cx, v - L.cy
    r = np.hypot(du, dv)
    th = 2 * np.arcsin(np.clip(r / L.cx * math.sin(L.theta_max_h / 2), -1, 1))
    phi = np.arctan2(dv, du)
    ray = np.stack([np.cos(th), -np.sin(th) * np.cos(phi), -np.sin(th) * np.sin(phi)], -1)
    world = np.einsum("nij,hwj->nhwi", R, ray)
    az, el = np.arctan2(world[..., 1], world[..., 0]), np.arcsin(np.clip(world[..., 2], -1, 1))
    img = np.stack([128 + 100 * np.sin(3 * az + k) * np.cos(2 * el + 0.3 * k) for k in range(3)], -1)
    img[:, r >= L.cx] = 0
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def _quat_of_yaw(yaw, pitch=0.0):
    """wxyz of Rz(yaw) Ry(pitch)."""
    qz = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], -1)
    qy = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
    w1, x1, y1, z1 = qz.T
    w2, x2, y2, z2 = qy
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


@pytest.mark.parametrize("turn_deg", [5.0, 15.0, -20.0])
def test_pure_yaw_is_removed(monkeypatch, turn_deg):
    """A pure turn about the world vertical, scene fixed: plain DVS fires over much of the field; motion compensation
    removes all but a residual. Measured: 36-46 residual pixel-channels of ~65k plain ones (< 0.1%), all within 11 rows
    of the top / bottom crop edge (r >= 39 px, ~73 deg off-axis), where the panorama is steep in pixel terms and the
    probe's grid admits sources up to the image edge (border padding). Bound used: 0.2% and r >= 38 px."""
    _recipe(monkeypatch)
    y0 = np.array([0.3, -2.0, 1.0])
    q0, q1 = _quat_of_yaw(y0), _quat_of_yaw(y0 + math.radians(turn_deg))
    f0, f1 = _render(_quat_to_R(q0)), _render(_quat_to_R(q1))
    plain = DVSPolarity(PoseEnv([f0, f1], [q0, q1]))
    plain.reset()
    n_plain = int((_pol(plain.step(np.zeros((3, 2)))[0]) > 0).sum())
    monkeypatch.setenv("NETT_DVS_MOTION_COMP", "yaw")
    mc = DVSPolarity(PoseEnv([f0, f1], [q0, q1]))
    mc.reset()
    mc_out = _pol(mc.step(np.zeros((3, 2)))[0])
    n_mc = int((mc_out > 0).sum())
    rim = mc._mc.valid.sum() * 3 * 6                                   # rim pixels x envs x channels
    assert n_plain > 0.05 * rim
    assert n_mc <= 2e-3 * n_plain, (n_mc, n_plain)
    _, v, u, _ = np.nonzero(mc_out)
    assert np.hypot(u + 0.5 - 64, v + 0.5 - 40).min() >= 38
    _, ok = mc._mc.grid(_quat_to_R(q0), _quat_to_R(q1))
    assert ok.float().mean() > 0.6 * mc._mc.valid.mean()               # only a sliver leaves the view


def test_motion_comp_is_about_the_world_vertical(monkeypatch):
    """With a pitched eye a world-vertical turn is not a rotation about the camera's own up axis; the warp
    (R_prev^T Rz(dyaw) R_prev) still removes it. The next tick at the same pose and frame emits nothing."""
    _recipe(monkeypatch)
    y0 = np.array([0.0, 1.0])
    q0, q1 = _quat_of_yaw(y0, 0.2), _quat_of_yaw(y0 + math.radians(12), 0.2)
    frames = [_render(_quat_to_R(q0)), _render(_quat_to_R(q1))]
    counts = []
    for mc in ("", "yaw"):
        monkeypatch.setenv("NETT_DVS_MOTION_COMP", mc)
        env = DVSPolarity(PoseEnv(frames, [q0, q1]))
        env.reset()
        counts.append(int((_pol(env.step(np.zeros((2, 2)))[0]) > 0).sum()))
        assert int(_pol(env.step(np.zeros((2, 2)))[0]).sum()) == 0
    assert counts[0] > 1000 and counts[1] <= 2e-3 * counts[0], counts


def test_motion_comp_first_tick_and_done_are_the_plain_rule(monkeypatch):
    """No previous frame => no events; a done env's events are zeroed (reset handling unchanged)."""
    _recipe(monkeypatch)
    monkeypatch.setenv("NETT_DVS_MOTION_COMP", "yaw")
    q = [_quat_of_yaw(np.zeros(2)), _quat_of_yaw(np.array([0.5, 0.5]))]
    f = [_render(_quat_to_R(x)) for x in q]
    f[1] = f[1].copy()
    f[1][:, 30:50, 50:80] = 255                                         # a local change the warp cannot explain

    class DoneEnv(PoseEnv):
        def step(self, action):
            obs, r, term, trunc, info = super().step(action)
            term[1] = True
            return obs, r, term, trunc, info

    env = DVSPolarity(DoneEnv(f, q))
    assert int(_pol(env.reset()[0]).sum()) == 0
    out = _pol(env.step(np.zeros((2, 2)))[0])
    assert int(out[0].sum()) > 0 and int(out[1].sum()) == 0


@pytest.mark.parametrize("as_tensor", [True, False])
def test_gate_scales_each_env_by_its_action(monkeypatch, as_tensor):
    monkeypatch.setenv("NETT_DVS_THRESHOLD", "10")
    monkeypatch.setenv("NETT_DVS_BLUR", "0")
    monkeypatch.setenv("NETT_DVS_GATE_K", "0.5")
    env = DVSPolarity(FakeEnv([const(0, n_env=4), const(30, n_env=4)], n_env=4, as_tensor=as_tensor))
    env.reset()
    a = np.array([[0, 0], [1, 1], [3, -3], [0.6, 0]], np.float32)       # |a| = 0, 1, 1 (clipped), .6/sqrt2
    out = _pol(env.step(torch.as_tensor(a) if as_tensor else a)[0])
    want = [255, round(255 * 0.5), round(255 * 0.5), round(255 * (1 - 0.5 * 0.6 / math.sqrt(2)))]
    for i, w in enumerate(want):
        assert np.all(out[i, ..., 0] == w) and np.all(out[i, ..., 1] == 0), i


def test_gate_k_large_silences_full_actions(monkeypatch):
    monkeypatch.setenv("NETT_DVS_THRESHOLD", "10")
    monkeypatch.setenv("NETT_DVS_BLUR", "0")
    monkeypatch.setenv("NETT_DVS_GATE_K", "2")
    env = DVSPolarity(FakeEnv([const(0, n_env=2), const(30, n_env=2)], n_env=2))
    env.reset()
    out = _pol(env.step(np.array([[1, 0], [0.1, 0.1]], np.float32))[0])
    assert int(out[0].sum()) == 0 and np.all(out[1, ..., 0] == round(255 * (1 - 2 * 0.1)))


def test_global_sub_removes_a_global_change_and_keeps_a_local_one(monkeypatch):
    monkeypatch.setenv("NETT_DVS_THRESHOLD", "10")
    monkeypatch.setenv("NETT_DVS_BLUR", "0")
    monkeypatch.setenv("NETT_DVS_COLOR", "1")
    monkeypatch.setenv("NETT_DVS_GLOBAL_SUB", "median:2")
    f1 = const(40, n_env=1)
    f1[0, 3:5, 5:7] = 200                                               # a 2x2 object brightens; the field +40
    env = DVSPolarity(FakeEnv([const(0, n_env=1), f1], n_env=1))
    env.reset()
    out = _pol(env.step(0)[0])[0]                                       # (H, W, 6)
    on = out[..., 0::2] > 0
    assert on[3:5, 5:7].all() and on.sum() == 4 * 3 and int(out[..., 1::2].sum()) == 0


# ----------------------------------------------------------------------------------------------------- refusals
@pytest.mark.parametrize("knob, bad", [
    ("NETT_DVS_MOTION_COMP", "full"), ("NETT_DVS_MOTION_COMP", "1"), ("NETT_DVS_MOTION_COMP", "yaw,pitch"),
    ("NETT_DVS_GATE_K", "0"), ("NETT_DVS_GATE_K", "-0.5"), ("NETT_DVS_GATE_K", "nan"), ("NETT_DVS_GATE_K", "inf"),
    ("NETT_DVS_GATE_K", "half"),
    ("NETT_DVS_GLOBAL_SUB", "median"), ("NETT_DVS_GLOBAL_SUB", "median:0"), ("NETT_DVS_GLOBAL_SUB", "median:4.0"),
    ("NETT_DVS_GLOBAL_SUB", "mean:4"), ("NETT_DVS_GLOBAL_SUB", "median:-1"), ("NETT_DVS_GLOBAL_SUB", "4"),
    ("NETT_DVS_GLOBAL_SUB", "median:4:4"),
])
def test_refuses_ambiguous_values(monkeypatch, knob, bad):
    monkeypatch.setenv(knob, bad)
    with pytest.raises(ValueError, match=knob):
        DVSPolarity(FakeEnv([const(0)]))


@pytest.mark.parametrize("a, b", [(EGO[0], EGO[1]), (EGO[0], EGO[2]), (EGO[1], EGO[2])])
def test_refuses_two_ego_knobs(monkeypatch, a, b):
    monkeypatch.setenv(a, ARM[a])
    monkeypatch.setenv(b, ARM[b])
    with pytest.raises(ValueError, match="at most one"):
        DVSPolarity(FakeEnv([const(0)]))


@pytest.mark.parametrize("knob", EGO)
def test_refuses_with_a_window(monkeypatch, knob):
    monkeypatch.setenv(knob, ARM[knob])
    monkeypatch.setenv("NETT_DECISION_LOOP", "2")
    monkeypatch.setenv("NETT_DVS_WINDOW", "2")
    with pytest.raises(ValueError, match="NETT_DVS_WINDOW"):
        DVSPolarity(FakeEnv([const(0)]))


@pytest.mark.parametrize("knob", EGO)
def test_refuses_with_the_move_gain(monkeypatch, knob):
    monkeypatch.setenv(knob, ARM[knob])
    monkeypatch.setenv("NETT_DECISION_LOOP", "2")
    monkeypatch.setenv("NETT_DVS_MOVE_GAIN", "0.5")
    with pytest.raises(ValueError, match="NETT_DVS_MOVE_GAIN"):
        DVSPolarity(FakeEnv([const(0)]))


@pytest.mark.parametrize("kw", [dict(fov=150.0), dict(eye=(448, 280)), dict(eye=(256, 160))])
def test_motion_comp_refuses_a_lens_the_probe_did_not_run(monkeypatch, kw):
    monkeypatch.setenv("NETT_DVS_MOTION_COMP", "yaw")
    w, h = kw.get("eye", (128, 80))
    env = DVSPolarity(PoseEnv([np.zeros((1, h, w, 3), np.uint8)], [_quat_of_yaw(np.zeros(1))], **kw))
    with pytest.raises(ValueError, match="NETT_DVS_MOTION_COMP"):
        env.reset()


def test_motion_comp_refuses_another_lens_method(monkeypatch):
    monkeypatch.setenv("NETT_DVS_MOTION_COMP", "yaw")
    monkeypatch.setenv("NETT_FISHEYE_METHOD", "equidistant")
    env = DVSPolarity(PoseEnv([np.zeros((1, 80, 128, 3), np.uint8)], [_quat_of_yaw(np.zeros(1))]))
    with pytest.raises(ValueError, match="equidistant"):
        env.reset()


def test_motion_comp_refuses_an_env_without_cameras(monkeypatch):
    monkeypatch.setenv("NETT_DVS_MOTION_COMP", "yaw")
    env = DVSPolarity(FakeEnv([const(0, n_env=2)], n_env=2))
    with pytest.raises(RuntimeError, match="_cameras"):
        env.reset()
    env = DVSPolarity(PoseEnv([np.zeros((1, 80, 128, 3), np.uint8)], [_quat_of_yaw(np.zeros(1))], cameras=("left",)))
    with pytest.raises(RuntimeError, match="center"):
        env.reset()


def test_gate_refuses_an_action_per_env_mismatch(monkeypatch):
    monkeypatch.setenv("NETT_DVS_GATE_K", "0.5")
    env = DVSPolarity(FakeEnv([const(0, n_env=3), const(30, n_env=3)], n_env=3))
    env.reset()
    with pytest.raises(RuntimeError, match="NETT_DVS_GATE_K"):
        env.step(np.zeros((2, 2), np.float32))


def test_rz_matches_the_probe_rotation_sense():
    """Rz(a) turns +X toward +Y for a > 0 (the probe's yaw_of convention)."""
    assert np.allclose(_Rz(np.array([math.pi / 2]))[0] @ [1, 0, 0], [0, 1, 0])
