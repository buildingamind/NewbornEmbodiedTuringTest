"""STP-E -- the presenter's ego-STP, approved for parsing at eye 448x280 (workspace DECISIONS §170).

Pose row body/wrappers/stp_pose.py, encoder brain/encoders/stp_e.py, aux brain/aux/stp_aux.py, labels "STP-E",
"STP-E-NoWarp", "STP-E-Cmd" and the size-matched control "CNN2F-630K" in examples/campaign_train.py.

Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU). The owner's list:
  * parameter counts per arm (printed, and pinned)
  * a = 0 -> identity, exactly
  * the dead zone
  * NoWarp residual == frame difference
  * the predictor optimizer is separate and no policy-loss gradient reaches predictor params
  * obs / space shapes at 448x280 and 128x80
  * the knobs refuse bad values
  * existing models unchanged
  * the MODELS import-time assertions
  + the remap against repo B's lens and rim mask, the warp against F.grid_sample, the byte transport, the
    wrappers on a fake NETTEnv, the aux never drawing across a reset, and the loss decreasing.
"""

from __future__ import annotations

import importlib.util
import math
import os
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.body.wrappers.channels_first import ChannelsFirst
from nett_skrl.body.wrappers.framestack import FrameStack
from nett_skrl.body.wrappers.registry import _load_wrapper
from nett_skrl.body.wrappers.stp_pose import (POSE_BYTES, STPPoseCommanded, STPPoseExecuted, bytes_to_pose,
                                              pose_to_row)
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.aux.knobs import NOT_MEASURED
from nett_skrl.brain.aux.stp_aux import STPTerm, charbonnier, masked_mse
from nett_skrl.brain.encoders.stp_e import (DEADZONE, STPEEncoder, STPPredictor, lens_azel_taps, warp_affine)
from nett_skrl.brain.registry import encoder_mapping
from nett_skrl.body.observation import prepare_image_tensor

_SRC = Path(__file__).resolve().parents[1]
_TRAIN = _SRC / "examples" / "campaign_train.py"
STP_LABELS = ("STP-E", "STP-E-NoWarp", "STP-E-Cmd")
NEW_LABELS = {*STP_LABELS, "CNN2F-630K"}
#: admitted after §170: owner 2026-10-20 (workspace DECISIONS §173), spatial-pooled ViViT (test_vivit_spatial_rect.py)
VIVSP173_LABELS = {"ViViT-SP1x4", "ViViT-SP2x8"}
# bind16 (workspace PROGRAM_BIND16, owner 2026-10-27): the conv-stem ViT-P8 label used by the presenter's frozen-encoder runs.
BIND16_LABELS = {"ViT-P8CONV-682K"}
# BASELINE series (owner 2026-10-28): the two backbones, their DVS / CLTT-Ref / MAE variations and the ViViT-P8 video twin.
BASELINE_LABELS = {"CNN-D512P", "ViT-P8-666K", "CNN-D512P+DVS", "ViT-P8-666K+DVS", "CNN-D512P+CLTT-Ref", "ViT-P8-666K+CLTT-Ref",
                   "ViT-P8-666K+MAE", "ViViT-P8-T2", "CNN-D512P+MASK75", "CNN-D512P+SC257", "ViT-P8-666K+SC257"}
# Slide-3 n=7 wave (owner 2026-10-30; workspace FINDINGS §4dh.44h.211 add. 219): the slide backbones + one variation each
# (test_slide3_wave_labels.py); mae_aux.py (early-fusion mode) and compact_vivit.py (its visible-token path) are admitted above.
S3W_LABELS = {"ViViT-T2", "CNN+DVS", "ViT+DVS", "CNN+MASK75", "ViT+MASK75", "3DCNN+MASK75", "ViViT-T2+MASK75",
              "CNN+SC257", "ViT+SC257"}
KNOBS = ("NETT_STP_LR", "NETT_STP_BATCH")
H, W = 80, 128                      # small eye for the fast tests: FOV 150 -> the same 60 x 100 grid as 448x280
FOV = 150.0

# Pinned counts (encoder only; the policy / value MLP heads on top are not included).
STP_COUNTS = {(448, 280, 150.0): {"predictor": 201_085, "policy_cnn": 427_544, "total": 628_629},
              (128, 80, 300.0): {"predictor": 201_085, "policy_cnn": 1_169_432, "total": 1_370_517}}
GRIDS = {(448, 280, 150.0): (60, 100), (128, 80, 300.0): (99, 200)}
CNN_COUNTS = {"CNN2F-630K": {(448, 280): 628_155, (128, 80): 835_530},
              "CNN2F": {(448, 280): 588_139, (128, 80): 780_139}}
#: DVSC6 = "CNN-UnityRecipe+DVS" with NETT_DVS_COLOR=1 (FINDINGS §4dh.44h.168 add. 1): 6 event channels.
DVSC6_COUNTS = {(448, 280): 13_287_712, (128, 80): 672_032}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in KNOBS + ("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_AUX_LOSS", "NETT_AUX_WEIGHT",
                         "NETT_AUX_WEIGHT_OVERRIDE", "NETT_AUX_BATCH", "NETT_AUX_STRICT", "NETT_AMP",
                         "NETT_DISABLE_FEATURE_CACHE", "NETT_DECOUPLE_ENCODER", "NETT_FISHEYE_METHOD",
                         "NETT_WHEEL_TURNMOVE", "NETT_FRAMESTACK_N", "NETT_FRAMESTACK_SHUFFLE", "NETT_DVS_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_STP_BATCH", "4")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(9)
        yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _space(h=H, w=W, layout="chw"):
    shape = (6, h + 1, w) if layout == "chw" else (h + 1, w, 6)
    return gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)


def _enc(h=H, w=W, fov=FOV, warp="affine", seed=0, features_dim=32, layout="chw"):
    torch.manual_seed(seed)
    return STPEEncoder(_space(h, w, layout), features_dim=features_dim, fov_deg=fov, warp=warp, num_frames=2)


def _frames(b, h=H, w=W, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (b, 3, h, w), generator=g, dtype=torch.uint8)


def _obs(pose, prev, cur, layout="chw"):
    """(B, 4) pose of the NEWEST frame, (B, 3, H, W) uint8 frames -> the stored 2-frame observation."""
    b, _, h, w = cur.shape
    row = pose_to_row(pose, w, 3).permute(0, 3, 1, 2)                     # (B, 3, 1, W)
    old = pose_to_row(torch.zeros(b, 4), w, 3).permute(0, 3, 1, 2)
    chw = torch.cat([torch.cat([old, prev], 2), torch.cat([row, cur], 2)], 1)
    return chw if layout == "chw" else chw.permute(0, 2, 3, 1).contiguous()


def _pose(*rows):
    return torch.tensor(rows, dtype=torch.float32)


def _n(params):
    return sum(p.numel() for p in params)


def _state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _git(*args):
    env = {k: v for k, v in os.environ.items()
           if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_COMMON_DIR")}
    try:
        return subprocess.run(["git", *args], cwd=_SRC, capture_output=True, text=True, check=True, env=env).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git unavailable here: {exc}")


def _base():
    """The commit before STP-E: the parent of the commit that added encoders/stp_e.py (HEAD while uncommitted)."""
    added = _git("log", "--diff-filter=A", "--format=%H", "--", "nett_skrl/brain/encoders/stp_e.py").split()
    return f"{added[-1]}^" if added else "HEAD"


# ================================================================ parameter counts

def _label_encoder(campaign, label, w, h, fov):
    spec = campaign.MODELS[label]
    cfg = campaign.stp_encoder_cfg(spec, fov)
    kwargs = {k: v for k, v in cfg.items() if k != "trainable"}
    extra = 1 if spec.get("pre") and spec["pre"][0].startswith("stp_pose") else 0
    channels = 6 if spec.get("framestack") else 3
    torch.manual_seed(0)
    return encoder_mapping[spec["encoder"]](
        gym.spaces.Box(0, 255, (channels, h + extra, w), np.uint8), **kwargs)


@pytest.mark.parametrize("w,h,fov", [(448, 280, 150.0), (128, 80, 300.0)])
def test_param_counts_per_arm_are_printed_and_pinned(campaign, w, h, fov, capsys):
    lines = []
    for label in STP_LABELS:
        enc = _label_encoder(campaign, label, w, h, fov)
        counts = enc.parameter_counts()
        assert counts == STP_COUNTS[(w, h, fov)], (label, counts)
        assert enc.grid_hw == GRIDS[(w, h, fov)]
        assert counts["total"] == _n(enc.parameters()) == counts["predictor"] + counts["policy_cnn"]
        lines.append(f"{label:14s} {w}x{h} fov {fov:g}: grid {enc.grid_hw}  predictor {counts['predictor']:,}  "
                     f"policy CNN {counts['policy_cnn']:,}  total {counts['total']:,}")
    for label in ("CNN2F-630K", "CNN2F"):
        enc = _label_encoder(campaign, label, w, h, fov)
        assert _n(enc.parameters()) == CNN_COUNTS[label][(w, h)], label
        lines.append(f"{label:14s} {w}x{h}: total {_n(enc.parameters()):,}")
    dvs = encoder_mapping["nature_cnn"](gym.spaces.Box(0, 255, (6, h, w), np.uint8),
                                        **{k: v for k, v in campaign.MODELS["CNN-UnityRecipe+DVS"]["cfg"].items()
                                           if k != "trainable"})
    assert _n(dvs.parameters()) == DVSC6_COUNTS[(w, h)]
    lines.append(f"{'DVSC6 ref':14s} {w}x{h}: CNN-UnityRecipe+DVS + NETT_DVS_COLOR=1, total {_n(dvs.parameters()):,}")
    with capsys.disabled():
        print("\n" + "\n".join(lines))
    if (w, h) == (448, 280):
        # the budget: ~0.63M, <= 1M; the control within 0.1% of STP-E; plain CNN2F is 6.4% short (> 5%: new key)
        stp = STP_COUNTS[(w, h, fov)]["total"]
        assert stp <= 1_000_000 and abs(stp - 630_000) / 630_000 < 0.01
        assert abs(CNN_COUNTS["CNN2F-630K"][(w, h)] - stp) / stp < 0.001
        assert (stp - CNN_COUNTS["CNN2F"][(w, h)]) / stp > 0.05


def test_nowarp_keeps_every_parameter():
    a, b = _enc(warp="affine"), _enc(warp="identity")
    assert a.parameter_counts() == b.parameter_counts()
    assert [k for k, _ in a.named_parameters()] == [k for k, _ in b.named_parameters()]
    sa, sb = _state(a), _state(b)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)                   # same seed, same init


# ================================================================ a = 0 -> identity, exactly

def test_a_zero_gives_exactly_the_identity_affine_and_the_input_frame():
    pred = STPPredictor(100, 4)
    A = pred.affines(torch.zeros(3, 3))
    eye = torch.eye(3)[:2].expand(3, 4, 2, 3)
    assert torch.equal(A, eye)
    torch.manual_seed(1)
    x = torch.rand(3, 3, 60, 100)
    vm = (torch.rand(60, 100) > 0.2).float()
    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7], [0.0, 0.0, 0.0]])
    xhat, wv, m = pred(x, a, vm)
    for i in (0, 2):
        assert torch.equal(xhat[i], x[i]) and torch.equal(wv[i, 0], vm)
    assert not torch.equal(xhat[1], x[1])
    assert torch.allclose(m.sum(1), torch.ones(3, 60, 100), atol=1e-6)


def test_still_or_dead_zone_rows_give_exactly_the_nowarp_residual():
    """valid = 0, an exactly-zero pose and a pose inside the dead zone all give the plain frame difference."""
    a, b = _enc(warp="affine"), _enc(warp="identity")
    prev, cur = _frames(4, seed=1), _frames(4, seed=2)
    pose = _pose([0, 0, 0, 1], [0.029, -0.03, 0.5, 1], [5.0, 3.0, 40.0, 0], [0.02, 0.0, -0.49, 1])
    obs = _obs(pose, prev, cur)
    assert torch.equal(a.action(a.split(obs)[0]), torch.zeros(4, 3))
    assert torch.equal(a.residual_input(obs), b.residual_input(obs))
    moving = _obs(_pose([0, 0, 6.0, 1]), prev[:1], cur[:1])
    assert not torch.equal(a.residual_input(moving), b.residual_input(moving))


# ================================================================ dead zone

def test_dead_zone_is_exact_and_soft_beyond_it():
    enc = _enc()
    assert tuple(enc.deadzone.tolist()) == pytest.approx(DEADZONE) and DEADZONE == (0.03, 0.03, 0.5)
    inside = _pose([0.03, -0.03, 0.5, 1], [-0.0299, 0.01, -0.4999, 1], [0.0, 0.0, 0.0, 1])
    assert torch.equal(enc.action(inside), torch.zeros(3, 3))
    out = enc.action(_pose([0.13, -0.53, 10.5, 1], [-1.03, 0.0, -2.5, 1]))
    assert out.tolist()[0] == pytest.approx([0.10, -0.50, 1.0], abs=1e-6)          # turn scale 10 deg
    assert out.tolist()[1] == pytest.approx([-1.00, 0.0, -0.2], abs=1e-6)
    assert torch.equal(enc.action(_pose([3.0, 2.0, 20.0, 0])), torch.zeros(1, 3))  # invalid -> 0


# ================================================================ NoWarp residual == frame difference

def test_nowarp_residual_is_the_frame_difference_and_never_runs_the_predictor():
    enc = _enc(warp="identity")

    def boom(*_a, **_k):
        raise AssertionError("NoWarp ran the predictor")

    enc.predictor.forward = boom
    prev, cur = _frames(3, seed=3), _frames(3, seed=4)
    obs = _obs(_pose([0.5, 0.1, 12.0, 1], [0, 0, 0, 1], [-1.0, 0, -18.0, 1]), prev, cur)
    out = enc.residual_input(obs)
    g_prev = enc.azel(prev.float() / 255.0)
    g_cur = enc.azel(cur.float() / 255.0)
    assert out.shape == (3, 6, 60, 100)
    assert torch.equal(out[:, :3], (g_cur - g_prev) * enc.vm)
    assert torch.equal(out[:, 3:], g_cur)
    assert enc(obs).shape == (3, 32)


# ================================================================ the remap and the warp

def test_remap_matches_repo_b_lens_and_respects_its_rim_mask():
    lens_mod = pytest.importorskip("nett_isaac.lens")
    rim_mod = pytest.importorskip("nett_isaac.eye_rim_mask")
    for (w, h, fov) in ((448, 280, 150.0), (128, 80, 300.0)):
        enc = _enc(h=h, w=w, fov=fov)
        lens = lens_mod.LensSpec(method="equisolid", fov_h_deg=fov, width=w, height=h)
        ho, wo = enc.grid_hw
        # A linear ramp is reproduced exactly by bilinear interpolation: azel(x + 0.5) is the source u in px.
        ramp_u = (torch.arange(w, dtype=torch.float32) + 0.5).expand(1, 1, h, w)
        ramp_v = (torch.arange(h, dtype=torch.float32) + 0.5).view(1, 1, h, 1).expand(1, 1, h, w)
        u_got = enc.azel(ramp_u.contiguous())[0, 0].double()
        v_got = enc.azel(ramp_v.contiguous())[0, 0].double()
        tv = math.degrees(math.asin(min(1.0, lens.cy / (lens.cx / math.sin(math.radians(fov / 4))))) * 2)
        az = torch.deg2rad((torch.arange(wo, dtype=torch.float64) + 0.5) * 1.5 - fov / 2)
        el = torch.deg2rad(tv - (torch.arange(ho, dtype=torch.float64) + 0.5) * 1.5)
        EL, AZ = torch.meshgrid(el, az, indexing="ij")
        d = torch.stack([EL.cos() * AZ.sin(), EL.sin(), EL.cos() * AZ.cos()])
        theta = torch.arccos(d[2].clamp(-1, 1))
        phi = torch.atan2(d[1], d[0])
        r = torch.as_tensor(lens.r_px(theta), dtype=torch.float64)
        u_ref, v_ref = lens.cx + r * phi.cos(), lens.cy - r * phi.sin()
        valid = enc.vm.bool()
        assert int(valid.sum()) > 0.5 * ho * wo
        assert torch.allclose(u_got[valid], u_ref[valid], atol=2e-3)
        assert torch.allclose(v_got[valid], v_ref[valid], atol=2e-3)
        assert torch.equal(u_got[~valid], torch.zeros_like(u_got[~valid]))      # invalid cells read 0
        # every tap with weight lands on a pixel repo B's rim mask keeps
        keep = rim_mod.rim_keep_mask(lens)[..., 0].reshape(-1)
        used = enc.tap_idx[enc.tap_wt > 0]
        assert bool((keep[used] == 1).all())
        # and the rim zone cannot reach the grid
        x = torch.rand(1, 3, h, w)
        loud = x.clone().reshape(1, 3, -1)
        loud[:, :, keep == 0] = 1000.0
        assert torch.equal(enc.azel(x), enc.azel(loud.view(1, 3, h, w)))
        ones = enc.azel(torch.ones(1, 1, h, w))[0, 0]
        assert torch.allclose(ones, enc.vm, atol=1e-6)


def test_lens_taps_refuse_bad_fov():
    for fov in (0.0, -10.0, 361.0):
        with pytest.raises(ValueError, match="fov_deg"):
            lens_azel_taps(128, 80, fov)


def test_warp_affine_matches_grid_sample_in_value_and_gradient():
    torch.manual_seed(3)
    x = torch.rand(2, 4, 9, 13, dtype=torch.float64)
    A = (torch.eye(3, dtype=torch.float64)[:2].expand(2, 3, 2, 3)
         + 0.15 * torch.randn(2, 3, 2, 3, dtype=torch.float64)).clone().requires_grad_(True)
    out = warp_affine(x, A)
    A2 = A.detach().clone().requires_grad_(True)
    ref = torch.stack([F.grid_sample(x, F.affine_grid(A2[:, i], list(x.shape), align_corners=False),
                                     mode="bilinear", padding_mode="zeros", align_corners=False)
                       for i in range(3)], 1)
    assert out.shape == ref.shape == (2, 3, 4, 9, 13)
    assert torch.allclose(out, ref, atol=1e-12)
    wgt = torch.randn_like(ref)
    (out * wgt).sum().backward()
    (ref * wgt).sum().backward()
    assert torch.allclose(A.grad, A2.grad, atol=1e-9)


# ================================================================ the byte transport

def test_every_byte_survives_uint8_float_and_prepare_image():
    b = torch.arange(256, dtype=torch.uint8)
    assert torch.equal(torch.round(b.float() / 255.0 * 255.0).to(torch.uint8), b)
    img = b.view(1, 1, 16, 16).expand(1, 3, 16, 16).contiguous()
    space = gym.spaces.Box(0, 255, (3, 16, 16), np.uint8)
    for stored in (img, img.float(), img.reshape(1, -1), img.reshape(1, -1).float()):   # uint8 / float buffers
        prepared = prepare_image_tensor(stored, space)
        assert prepared.dtype == torch.float32 and float(prepared.max()) <= 1.0
        assert torch.equal(torch.round(prepared * 255.0).to(torch.uint8), img)


def test_pose_row_round_trips_any_float32_bit_exactly():
    torch.manual_seed(0)
    pose = torch.cat([torch.randn(64, 4) * 100, _pose([0, -0.0, 1e-30, 1], [3.4e38, -1e-7, 179.99, 0])])
    row = pose_to_row(pose, 20, 3)
    assert row.shape == (66, 1, 20, 3) and row.dtype == torch.uint8
    assert int(row[:, :, POSE_BYTES:].abs().sum()) == 0 and int(row[..., 1:].abs().sum()) == 0
    got = bytes_to_pose(row[:, 0, :POSE_BYTES, 0])
    assert torch.equal(got.view(torch.int32), pose.view(torch.int32))
    assert torch.equal(bytes_to_pose(row[:, 0, :POSE_BYTES, 0].float()).view(torch.int32), pose.view(torch.int32))
    with pytest.raises(ValueError, match="16 px"):
        pose_to_row(pose, 15, 3)


@pytest.mark.parametrize("layout", ["chw", "hwc"])
def test_encoder_decodes_the_pose_from_uint8_and_float_storage(layout):
    enc = _enc(layout=layout)
    pose = _pose([0.25, -0.125, 7.5, 1], [0, 0, 0, 0])
    obs = _obs(pose, _frames(2, seed=5), _frames(2, seed=6), layout=layout)
    for stored in (obs, obs.float(), obs.reshape(2, -1), obs.reshape(2, -1).float()):
        assert torch.equal(enc.stored_pose(stored), pose)
    for stored in (obs, obs.float()):
        got, _, _ = enc.split(stored)
        assert torch.equal(got, pose)
    assert torch.equal(enc.residual_input(obs), enc.residual_input(obs.float()))


def test_split_refuses_an_observation_without_a_pose_row():
    enc = _enc()
    obs = torch.full((1, 6, H + 1, W), 7, dtype=torch.uint8)
    with pytest.raises(ValueError, match="does not carry a pose row"):
        enc.split(obs)


# ================================================================ the wrappers on a fake NETTEnv

class _Cfg:
    body_turn_speed_limit = 20.0
    body_move_speed_limit = 1.5


class _Motor:
    def __init__(self, n):
        self.x = torch.zeros(n)
        self.y_pos = torch.zeros(n)
        self.yaw_deg = torch.zeros(n)
        self.cfg = _Cfg()


class _FakeNett(gym.Env):
    """Just enough NETTEnv for the pose wrappers: a motor, the gated actions, batched uint8 frames in a dict."""

    def __init__(self, n=2, h=6, w=20, channels=3, dtype=np.uint8, effective=True):
        self.num_envs = n
        self._step_in_episode = torch.zeros(n)
        self._decision_period = 1
        self.motor = _Motor(n)
        if effective:
            self._effective_actions = torch.zeros(n, 2)
        self.observation_space = gym.spaces.Box(0, 255, (n, h, w, channels), dtype)
        self.action_space = gym.spaces.Box(-1, 1, (n, 2), np.float32)
        self._dtype = torch.uint8 if dtype == np.uint8 else torch.float32
        self.t = 0
        self.script = []          # per step: (x, y, yaw, done) tensors, applied to the motor
        self.h, self.w, self.c, self.n = h, w, channels, n

    def _frame(self):
        g = torch.Generator().manual_seed(100 + self.t)
        return {"policy": torch.randint(0, 256, (self.n, self.h, self.w, self.c), generator=g).to(self._dtype)}

    def reset(self, **kwargs):
        self.t = 0
        return self._frame(), {}

    def step(self, action):
        x, y, yaw, done = self.script[self.t]
        self.t += 1
        self.motor.x, self.motor.y_pos, self.motor.yaw_deg = (torch.tensor(v, dtype=torch.float32) for v in (x, y, yaw))
        if hasattr(self, "_effective_actions"):
            self._effective_actions = torch.as_tensor(action, dtype=torch.float32)
        term = torch.tensor(done, dtype=torch.bool).view(-1, 1)
        return self._frame(), torch.zeros(self.n), term, torch.zeros_like(term), {}


def test_executed_pose_is_the_body_frame_change_and_zero_at_reset_and_done():
    env = _FakeNett(n=2)
    env.motor.yaw_deg = torch.tensor([0.0, 179.0])
    wrap = _load_wrapper("stp_pose_exec")(env)
    assert type(wrap) is STPPoseExecuted
    assert wrap.observation_space.shape == (2, 7, 20, 3)
    obs, _ = wrap.reset()
    assert obs["policy"].shape == (2, 7, 20, 3) and obs["policy"].dtype == torch.uint8
    assert torch.equal(bytes_to_pose(obs["policy"][:, 0, :16, 0]), torch.zeros(2, 4))
    env.script = [
        # env 0: forward at yaw 0 is +y; env 1: yaw 179 -> -179 is a +2 deg turn
        ([0.5, 0.0], [1.0, 0.0], [0.0, -179.0], [False, False]),
        # env 0 at yaw 0 -> 90: no translation, +90; env 1 ends its episode (any pose) -> zeros, valid 0
        ([0.5, 30.0], [1.0, -7.0], [90.0, 33.0], [False, True]),
        # env 0 at yaw 90: forward is (-1, 0); env 1's delta is from the RESET pose it reported at done
        ([-1.5, 30.0], [1.0, -6.0], [90.0, 33.0], [False, False]),
    ]
    seen = []
    for _ in env.script:
        obs, *_ = wrap.step(env.action_space.sample())
        seen.append(bytes_to_pose(obs["policy"][:, 0, :16, 0]))
        assert torch.equal(seen[-1], wrap.last_pose)
        assert torch.equal(obs["policy"][:, 1:], env._frame()["policy"])      # the eye is untouched
    assert seen[0].tolist()[0] == pytest.approx([1.0, 0.5, 0.0, 1.0])      # fwd +y, lat +x (right of heading)
    assert seen[0].tolist()[1] == pytest.approx([0.0, 0.0, 2.0, 1.0], abs=1e-5)
    assert seen[1].tolist()[0] == pytest.approx([0.0, 0.0, 90.0, 1.0], abs=1e-6)
    assert seen[1][1].tolist() == [0.0, 0.0, 0.0, 0.0]
    yaw33 = math.radians(33.0)
    assert seen[2].tolist()[0] == pytest.approx([2.0, 0.0, 0.0, 1.0], abs=1e-5)
    assert seen[2].tolist()[1] == pytest.approx([math.cos(yaw33), math.sin(yaw33), 0.0, 1.0], abs=1e-5)


def test_commanded_pose_is_the_clamped_turn_move_in_physical_units():
    env = _FakeNett(n=2)
    wrap = _load_wrapper("stp_pose_cmd")(env)
    assert type(wrap) is STPPoseCommanded
    wrap.reset()
    env.script = [([9.0, 9.0], [9.0, 9.0], [9.0, 9.0], [False, True])]
    obs, *_ = wrap.step(torch.tensor([[0.5, -2.0], [1.0, 1.0]]))
    got = bytes_to_pose(obs["policy"][:, 0, :16, 0])
    assert got[0].tolist() == pytest.approx([-1.5, 0.0, 10.0, 1.0])         # move clamped to -1 x 1.5; turn 0.5 x 20
    assert got[1].tolist() == [0.0, 0.0, 0.0, 0.0]                         # done -> zero, valid 0


def test_pose_wrappers_refuse(monkeypatch):
    with pytest.raises(ValueError, match="BEFORE framestack"):
        STPPoseExecuted(_FakeNett(channels=6))
    with pytest.raises(ValueError, match="16 px"):
        STPPoseExecuted(_FakeNett(w=15))
    with pytest.raises(ValueError, match="uint8"):
        STPPoseExecuted(_FakeNett(dtype=np.float32)).reset()
    with pytest.raises(RuntimeError, match="before reset"):
        env = _FakeNett()
        env.script = [([0, 0], [0, 0], [0, 0], [False, False])]
        STPPoseExecuted(env).step(np.zeros((2, 2)))
    env = _FakeNett(effective=False)
    env.script = [([0, 0], [0, 0], [0, 0], [False, False])]
    wrap = STPPoseCommanded(env)
    wrap.reset()
    with pytest.raises(RuntimeError, match="_effective_actions"):
        wrap.step(np.zeros((2, 2)))
    monkeypatch.setenv("NETT_WHEEL_TURNMOVE", "0")
    with pytest.raises(ValueError, match="NETT_WHEEL_TURNMOVE=0"):
        STPPoseCommanded(_FakeNett())
    STPPoseExecuted(_FakeNett())                                            # executed does not care


# ================================================================ shapes at 448x280 and 128x80

@pytest.mark.parametrize("w,h,fov", [(448, 280, 150.0), (128, 80, 300.0)])
def test_obs_and_space_shapes_through_the_arm_wrapper_chain(w, h, fov):
    env = _FakeNett(n=2, h=h, w=w)
    chain = ChannelsFirst(FrameStack(STPPoseExecuted(env), n_stack=2))
    assert chain.observation_space.shape == (2, 6, h + 1, w)
    obs, _ = chain.reset()
    env.script = [([0.0, 0.0], [0.4, 0.0], [3.0, -2.0], [False, False])]
    obs, *_ = chain.step(np.zeros((2, 2), np.float32))
    pol = torch.as_tensor(np.asarray(obs["policy"]))                       # FrameStack hands numpy back
    assert tuple(pol.shape) == (2, 6, h + 1, w) and pol.dtype == torch.uint8
    enc = STPEEncoder(gym.spaces.Box(0, 255, chain.observation_space.shape[1:], np.uint8), features_dim=184,
                      fov_deg=fov, num_frames=2)
    pose, prev, cur = enc.split(pol)
    assert pose.tolist()[0] == pytest.approx([0.4, 0.0, 3.0, 1.0], abs=1e-6)
    assert prev.shape == cur.shape == (2, 3, h, w)
    assert torch.equal(torch.round(cur * 255).to(torch.uint8), env._frame()["policy"].permute(0, 3, 1, 2))
    assert enc.residual_input(pol).shape == (2, 6, *GRIDS[(w, h, fov)])
    assert enc(pol).shape == (2, 184)


# ================================================================ the agent: separate optimizer, no RL gradient

class _AgentEnv:
    num_envs = 1
    num_agents = 1
    observation_space = _space()
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    state_space = None


def _agent(monkeypatch, *, warp="affine", aux="stp", seed=0, epochs=1, mini_batches=1, rollouts=16, shared=True,
           weight="1"):
    from nett_skrl.brain.agent_factory import build_agents
    from nett_skrl.brain.brain import Brain
    if aux is None:
        monkeypatch.delenv("NETT_AUX_LOSS", raising=False)
        monkeypatch.delenv("NETT_AUX_WEIGHT", raising=False)
    else:
        monkeypatch.setenv("NETT_AUX_LOSS", aux)
        monkeypatch.setenv("NETT_AUX_WEIGHT", weight)
    torch.manual_seed(seed)
    brain = Brain(encoder="stp_e", algorithm="PPO",
                  encoder_cfg={"trainable": True, "features_dim": 32, "num_frames": 2, "warp": warp, "fov_deg": FOV},
                  algorithm_cfg={"rollouts": rollouts, "mini_batches": mini_batches, "learning_epochs": epochs,
                                 "grad_norm_clip": 0.5, "learning_rate": 1e-4},
                  model={"shared_encoder": shared, "hidden_sizes": []})
    agent = build_agents(brain, _AgentEnv(), torch.device("cpu"))[0]
    agent.init(trainer_cfg=None)
    agent.enable_training_mode(True)
    return agent


def _flat_obs(t, moving=True):
    pose = _pose([0.4, 0.0, 6.0 if moving else 0.0, 0.0 if t == 0 else 1.0])
    return _obs(pose, _frames(1, seed=2 * t), _frames(1, seed=2 * t + 1)).reshape(1, -1).float()


def _rollout(agent, seed=1, steps=16):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    obs = _flat_obs(0)
    with torch.no_grad():
        for t in range(steps):
            acts, _ = agent.act(obs, None, timestep=t, timesteps=100)
            nxt = _flat_obs(t + 1)
            agent.record_transition(observations=obs, states=None, actions=acts,
                                    rewards=torch.rand(1, 1, generator=g), next_observations=nxt,
                                    next_states=None, terminated=torch.zeros(1, 1, dtype=torch.bool),
                                    truncated=torch.zeros(1, 1, dtype=torch.bool), infos={},
                                    timestep=t, timesteps=100)
            obs = nxt
    torch.manual_seed(seed + 1000)
    agent.update(timestep=steps, timesteps=100)


def _record_ppo_grads(agent):
    steps = []
    named = [*(("policy." + k, p) for k, p in agent.policy.named_parameters()),
             *(("value." + k, p) for k, p in agent.value.named_parameters())]

    def hook(_opt, _args, _kwargs):
        steps.append({k: None if p.grad is None else p.grad.detach().clone() for k, p in named})

    agent.optimizer.register_step_pre_hook(hook)
    return steps


@pytest.mark.parametrize("warp", ["affine", "identity"])
def test_predictor_leaves_ppo_and_has_its_own_adam(monkeypatch, warp):
    agent = _agent(monkeypatch, warp=warp)
    term, enc = agent._aux, agent.policy.encoder
    assert isinstance(term, STPTerm) and agent.value.encoder is enc
    ppo_ids = {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
    pred_ids = {id(p) for p in enc.predictor_parameters()}
    assert pred_ids and not ppo_ids & pred_ids
    assert {id(p) for p in enc.policy_parameters()} <= ppo_ids              # the policy CNN stays PPO's
    assert list(term.head.parameters()) == []
    if warp == "affine":
        aux_ids = [id(p) for g in term.optimizer.param_groups for p in g["params"]]
        assert set(aux_ids) == pred_ids and len(aux_ids) == len(pred_ids)
        assert isinstance(term.optimizer, torch.optim.Adam) and term.optimizer.param_groups[0]["lr"] == 2e-3
        assert agent.checkpoint_modules["stp_optimizer"] is term.optimizer
    else:
        assert term.optimizer is None and "stp_optimizer" not in agent.checkpoint_modules


def test_policy_loss_reaches_the_policy_cnn_and_never_the_predictor(monkeypatch):
    agent = _agent(monkeypatch, epochs=2, mini_batches=2)
    steps = _record_ppo_grads(agent)
    pred_before = _state(agent.policy.encoder.predictor)
    _rollout(agent)
    assert len(steps) == 4 and agent._aux.n_steps == 4
    for step in steps:
        pred = [k for k in step if k.startswith("policy.encoder.predictor.")]
        cnn = [k for k in step if k.startswith("policy.encoder.policy_cnn.")]
        assert pred and all(step[k] is None for k in pred)
        assert cnn and all(step[k] is not None for k in cnn) and any(float(step[k].abs().sum()) > 0 for k in cnn)
    after = agent.policy.encoder.predictor.state_dict()
    assert any(not torch.equal(v, after[k]) for k, v in pred_before.items())     # the aux trained it


def test_plain_ppo_alone_never_moves_the_predictor(monkeypatch):
    """aux=None: the predictor is in PPO's optimizer but the policy loss gives it no gradient -> unchanged."""
    agent = _agent(monkeypatch, aux=None)
    before = _state(agent.policy.encoder.predictor)
    cnn_before = _state(agent.policy.encoder.policy_cnn)
    _rollout(agent)
    after = agent.policy.encoder.predictor.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items())
    assert any(not torch.equal(v, agent.policy.encoder.policy_cnn.state_dict()[k]) for k, v in cnn_before.items())


def test_bind_agent_refuses_unshared_encoder_and_inert_weight(monkeypatch):
    with pytest.raises(ValueError, match="shared_encoder=True"):
        _agent(monkeypatch, shared=False)
    with pytest.raises(ValueError, match="aux_weight=2"):
        _agent(monkeypatch, weight="2")


# ================================================================ the aux on a memory

class _Memory:
    """skrl's layout: flat (T, env, D) uint8 rows. ``moving`` (bool or a (T, env) mask): a 6 deg turn + 0.4 step,
    else a still valid row (0.02 units / 0.4 deg: inside the dead zone, so a = 0)."""

    def __init__(self, valid, moving=True, seed=0):
        t_max, n_env = valid.shape
        mov = torch.as_tensor(moving).expand(t_max, n_env) if not torch.is_tensor(moving) or moving.ndim == 0 \
            else moving
        rows = []
        for t in range(t_max):
            for e in range(n_env):
                motion = [0.4, 0.0, 6.0] if mov[t, e] else [0.02, -0.01, 0.4]
                pose = _pose([*motion, 1.0]) if valid[t, e] else torch.zeros(1, 4)
                k = seed + 2 * (t * n_env + e)
                rows.append(_obs(pose, _frames(1, seed=k), _frames(1, seed=k + 1)).reshape(-1))
        self.tensors = {"observations": torch.stack(rows).view(t_max, n_env, -1)}
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


def _term(enc):
    term = STPTerm(enc)
    term.build_optimizer()
    return term


def test_aux_never_draws_a_reset_row_and_steps_only_the_predictor():
    enc = _enc()
    valid = torch.zeros(6, 2, dtype=torch.bool)
    valid[1, 0] = valid[4, 1] = True
    term = _term(enc)
    term.attach_memory(_Memory(valid))
    for _ in range(20):
        obs, stats = term._draw(enc)
        assert stats == {"valid_rows": pytest.approx(2 / 12), "moving_rows": 2.0, "moving_rows_frac": 2 / 12}
        assert bool((enc.stored_pose(obs)[:, 3] == 1).all())
    cnn = _state(enc.policy_cnn)
    pred = _state(enc.predictor)
    out = term.compute(enc, None)
    assert out.shape == () and float(out) == 0.0 and not out.requires_grad
    assert term.n_steps == 1 and all(p.grad is None for p in enc.parameters())
    assert all(torch.equal(v, enc.policy_cnn.state_dict()[k]) for k, v in cnn.items())
    assert any(not torch.equal(v, enc.predictor.state_dict()[k]) for k, v in pred.items())
    for key in ("B", "loss", "copy_loss", "gain", "mse", "copy_mse", "gain_mse", "skill", "valid_frac",
                "moving_frac", "mask_bg", "g_turn", "valid_rows", "moving_rows", "moving_rows_frac", "skipped", "lr"):
        assert key in term.last_scalars, key
    sc = term.last_scalars
    assert sc["skipped"] == 0.0 and sc["moving_frac"] == 1.0 and sc["moving_rows"] == 2.0
    assert sc["gain"] == pytest.approx(sc["copy_loss"] / sc["loss"])
    assert sc["gain_mse"] == pytest.approx(sc["copy_mse"] / sc["mse"])


def test_aux_with_no_valid_row_takes_no_step_and_a_partial_ring_reads_only_written_rows():
    enc = _enc()
    term = _term(enc)
    term.attach_memory(_Memory(torch.zeros(4, 2, dtype=torch.bool)))
    w0 = enc.predictor.G.detach().clone()
    out = term.compute(enc, None)
    assert float(out) == 0.0 and term.n_steps == 0 and term.n_skipped == 1
    assert term.last_scalars == {**{k: NOT_MEASURED for k in STPTerm.LOSS_KEYS}, "valid_rows": 0.0,
                                 "moving_rows": 0.0, "moving_rows_frac": 0.0, "skipped": 1.0}
    assert torch.equal(enc.predictor.G, w0)
    valid = torch.zeros(6, 2, dtype=torch.bool)
    valid[:, :] = True
    mem = _Memory(valid)
    mem.tensors["observations"][3:] = 0                                     # unwritten rows: valid 0 anyway ...
    mem.tensors["observations"][3:, :, 3 * (H + 1) * W:3 * (H + 1) * W + 16] = torch.tensor(
        np.frombuffer(np.array([0, 0, 6, 1], np.float32).tobytes(), np.uint8))  # ... except this moving junk
    mem.filled, mem.memory_index = False, 3
    term.attach_memory(mem)
    for _ in range(10):
        obs, stats = term._draw(enc)
        assert stats["valid_rows"] == 1.0 and stats["moving_rows"] == 6.0
        hits = [(t, e) for t in range(6) for e in range(2) if any(torch.equal(o, mem.tensors["observations"][t, e])
                                                                  for o in obs)]
        assert hits and all(t < 3 for t, _ in hits)


def test_nowarp_aux_scores_but_never_steps():
    enc = _enc(warp="identity")
    term = _term(enc)
    assert term.optimizer is None
    term.attach_memory(_Memory(torch.ones(3, 2, dtype=torch.bool)))
    before = _state(enc)
    out = term.compute(enc, None)
    assert float(out) == 0.0 and term.n_steps == 0
    assert term.last_scalars["loss"] == pytest.approx(term.last_scalars["copy_loss"])
    assert term.last_scalars["lr"] == 0.0
    assert all(torch.equal(v, enc.state_dict()[k]) for k, v in before.items())


def test_charbonnier_is_the_masked_mean():
    p, t = torch.zeros(1, 3, 2, 2), torch.ones(1, 3, 2, 2)
    w = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    assert float(charbonnier(p, t, w)) == pytest.approx(math.sqrt(1 + 1e-6))
    assert float(charbonnier(p, t, torch.zeros_like(w))) == 0.0


def _shift_rows(n=6, shift=6, seed=0):
    """A smooth texture and the same texture shifted `shift` px in the eye, with a matching turn in the pose."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for i in range(n):
        lo = torch.rand(1, 3, 10, 18, generator=g)
        big = F.interpolate(lo, size=(H, W + shift), mode="bilinear", align_corners=False)
        prev = (big[..., shift:] * 255).round().to(torch.uint8)
        cur = (big[..., :W] * 255).round().to(torch.uint8)
        rows.append(_obs(_pose([0.0, 0.0, 6.0, 1.0]), prev, cur).reshape(-1))
    return torch.stack(rows)


def test_prediction_loss_decreases_over_cpu_steps():
    enc = _enc(seed=0)
    term = _term(enc)
    rows = _shift_rows()
    mem = _Memory(torch.ones(1, 1, dtype=torch.bool))
    mem.tensors["observations"] = rows.view(rows.shape[0], 1, -1)
    mem.memory_size = rows.shape[0]
    term.attach_memory(mem)
    torch.manual_seed(0)
    losses = []
    for _ in range(40):
        term.compute(enc, None)
        losses.append(term.last_scalars["loss"])
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert all(math.isfinite(v) for v in losses)
    assert last < 0.8 * first, (first, last)
    assert term.n_steps == 40


# ================================================================ follow-up: moving-row sampling, budget, telemetry

def test_aux_draws_only_moving_rows_distinct_when_enough_and_with_replacement_when_not():
    enc = _enc()
    term = _term(enc)                                                        # B = 4 (fixture)
    valid = torch.ones(8, 2, dtype=torch.bool)
    moving = torch.zeros(8, 2, dtype=torch.bool)
    moving[0, 0] = moving[3, 1] = moving[5, 0] = moving[6, 1] = moving[7, 0] = True     # 5 >= B
    mem = _Memory(valid, moving=moving)
    term.attach_memory(mem)
    rows = mem.tensors["observations"]

    def where(o):
        return next((t, e) for t in range(8) for e in range(2) if torch.equal(o, rows[t, e]))

    for _ in range(15):
        obs, stats = term._draw(enc)
        assert stats["moving_rows"] == 5.0 and stats["valid_rows"] == 1.0
        hits = [where(o) for o in obs]
        assert all(bool(moving[t, e]) for t, e in hits) and len(set(hits)) == 4      # distinct
        assert bool((enc.action(enc.stored_pose(obs)) != 0).any(dim=1).all())
    moving2 = torch.zeros(8, 2, dtype=torch.bool)
    moving2[2, 1] = True                                                     # 1 < B: with replacement
    mem2 = _Memory(valid, moving=moving2)
    term.attach_memory(mem2)
    obs, stats = term._draw(enc)
    assert obs.shape[0] == 4 and stats["moving_rows"] == 1.0
    assert all(torch.equal(o, mem2.tensors["observations"][2, 1]) for o in obs)


def test_aux_skips_a_rollout_of_still_rows_entirely():
    enc = _enc()
    term = _term(enc)
    term.attach_memory(_Memory(torch.ones(4, 2, dtype=torch.bool), moving=False))
    before = _state(enc)
    seen = []
    term.optimizer.register_step_pre_hook(lambda *_: seen.append(1))
    for _ in range(3):
        assert float(term.compute(enc, None)) == 0.0
    assert seen == [] and term.n_steps == 0 and term.n_skipped == 3
    assert term.last_scalars["skipped"] == 1.0 and term.last_scalars["loss"] == NOT_MEASURED
    assert term.last_scalars["valid_rows"] == 1.0 and term.last_scalars["moving_rows"] == 0.0
    assert all(torch.equal(v, enc.state_dict()[k]) for k, v in before.items())


def test_masked_mse_is_the_masked_mean_square():
    p, t = torch.zeros(1, 3, 2, 2), torch.full((1, 3, 2, 2), 2.0)
    w = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    assert float(masked_mse(p, t, w)) == 4.0 and float(masked_mse(p, t, torch.zeros_like(w))) == 0.0


class _PinnedNett(_FakeNett):
    """The agent commands a full step forward but the body never moves (pinned at the screen wall)."""

    def step(self, action):
        self._effective_actions = torch.as_tensor(action, dtype=torch.float32)
        term = torch.zeros(self.n, 1, dtype=torch.bool)
        self.t += 1
        return self._frame(), torch.zeros(self.n), term, torch.zeros_like(term), {}


def _rollout_rows(wrapper_cls, env, steps=4, action=(0.0, 0.95)):
    """Frames through [pose wrapper, framestack, channels_first], stored flat as skrl does: (T, env, D)."""
    chain = ChannelsFirst(FrameStack(wrapper_cls(env), n_stack=2))
    chain.reset()
    rows = []
    for _ in range(steps):
        obs, *_ = chain.step(torch.tensor([list(action)] * env.n))
        rows.append(torch.as_tensor(np.asarray(obs["policy"])).reshape(env.n, -1))
    return torch.stack(rows)


def test_each_arm_samples_on_its_own_motion_input():
    """Pinned frames: STP-E (executed) has no moving row and skips; STP-E-Cmd (commanded) keeps them."""
    for cls, expect in ((STPPoseExecuted, 0.0), (STPPoseCommanded, 8.0)):
        rows = _rollout_rows(cls, _PinnedNett(n=2, h=H, w=W))
        enc = _enc()
        term = _term(enc)
        mem = _Memory(torch.ones(1, 1, dtype=torch.bool))
        mem.tensors["observations"] = rows
        term.attach_memory(mem)
        term.compute(enc, None)
        assert term.last_scalars["moving_rows"] == expect, cls.__name__
        assert term.n_steps == (1 if expect else 0) and term.n_skipped == (0 if expect else 1)


def test_nowarp_predictor_is_present_but_held_by_no_optimizer(monkeypatch, capsys):
    agent = _agent(monkeypatch, warp="identity")
    enc, term = agent.policy.encoder, agent._aux
    pred = {id(p) for p in enc.predictor_parameters()}
    n_unused = _n(enc.predictor_parameters())
    assert n_unused == STP_COUNTS[(448, 280, 150.0)]["predictor"] == 201_085
    held = [o for o in (agent.optimizer, term.optimizer, *agent.checkpoint_modules.values())
            if isinstance(o, torch.optim.Optimizer)]
    assert held and term.optimizer is None
    for opt in held:
        assert not pred & {id(p) for g in opt.param_groups for p in g["params"]}
    before = _state(enc.predictor)
    _rollout(agent)
    assert all(torch.equal(v, enc.predictor.state_dict()[k]) for k, v in before.items())
    with capsys.disabled():
        print(f"\nSTP-E-NoWarp predictor: {n_unused:,} parameters present, unused, in no optimizer")


#: The P setup (Unity-recipe++ parsing, FINDINGS §4dh.44h.160 + DECISIONS §128/§129): 64 envs x 400 steps,
#: 64 minibatches, 20 learning epochs, updates = floor(500,000 / rollout).
P_ROLLOUT, P_MINIBATCHES, P_EPOCHS = 64 * 400, 64, 20


def test_aux_takes_one_step_per_ppo_minibatch_and_meets_the_offline_budget_at_the_p_setup(monkeypatch):
    agent = _agent(monkeypatch, epochs=3, mini_batches=2)
    _rollout(agent)
    assert agent._aux.n_steps == 3 * 2 and agent._aux.n_skipped == 0                # epochs x minibatches
    updates = 500_000 // P_ROLLOUT
    per_update = P_EPOCHS * P_MINIBATCHES
    assert (updates, per_update, updates * per_update) == (19, 1_280, 24_320)
    assert updates * per_update >= 3_000
    assert 19 * 10 * P_MINIBATCHES >= 3_000                                   # even at the fleet's 10 epochs


def test_per_update_telemetry_reaches_the_tracked_aux_scalars(monkeypatch):
    agent = _agent(monkeypatch, epochs=1, mini_batches=2)
    _rollout(agent)
    tracked = agent.tracking_data
    for key in ("loss", "copy_loss", "gain", "mse", "copy_mse", "gain_mse", "moving_rows", "moving_rows_frac",
                "valid_rows", "skipped"):
        assert f"Loss / Aux stp {key}" in tracked, key
    gain = tracked["Loss / Aux stp gain"][-1]
    assert math.isfinite(gain) and gain > 0
    assert tracked["Loss / Aux stp skipped"][-1] == 0.0
    assert tracked["Loss / Aux stp moving_rows"][-1] == 15.0                  # rows 1..15 move; row 0 is a reset


# ================================================================ follow-up: column order and dead-zone units

def test_turn_warm_start_sits_on_the_dyaw_column_in_every_arm(campaign):
    expect = -(10.0 / 1.5) * (2.0 / 100) * torch.linspace(1.0, 0.25, 4)
    for label in STP_LABELS:
        enc = _label_encoder(campaign, label, 448, 280, 150.0)
        G = enc.predictor.G.detach()
        assert G.shape == (3, 4, 2, 3)
        assert torch.equal(G[2, :, 0, 2], expect), label                    # the turn generator: x-shift
        rest = G[2].clone()
        rest[:, 0, 2] = 0
        assert torch.equal(rest, torch.zeros_like(rest)), label              # nothing else on the turn column
        assert float(G[:2].abs().max()) < 0.06, label                        # fwd / lat: N(0, 0.01) only
        # the pose row's dyaw (slot 2) drives that generator; fwd (0) and lat (1) do not
        a = enc.action(_pose([0.0, 0.0, 10.5, 1.0]))
        assert torch.equal(a, torch.tensor([[0.0, 0.0, 1.0]]))
        A = enc.predictor.affines(a)
        assert torch.allclose(A[0, :, 0, 2], expect, atol=1e-6)              # a pure x-shift of the warm gain


def _kinematic_env(n=1):
    motor_mod = pytest.importorskip("nett_isaac.motor_system")
    env = _FakeNett(n=n)
    env.motor = motor_mod.MotorSystem(n, motor_mod.MotorCfg(locomotion="kinematic", body_move_speed_limit=1.5),
                                      (-100.0, 100.0, -100.0, 100.0))

    def step(action):
        act = torch.as_tensor(action, dtype=torch.float32)
        env._effective_actions = act
        env.motor.apply(act)                                                 # repo B: columns [turn, move]
        env.t += 1
        term = torch.zeros(n, 1, dtype=torch.bool)
        return env._frame(), torch.zeros(n), term, torch.zeros_like(term), {}

    env.step = step
    return env


@pytest.mark.parametrize("action,row", [((0.5, 0.0), [0.0, 0.0, 10.0]), ((0.0, 0.5), [0.75, 0.0, 0.0]),
                                        ((-0.25, 0.0), [0.0, 0.0, -5.0]), ((0.0, -1.0), [-1.5, 0.0, 0.0])])
def test_env_action_turn_move_reaches_the_dyaw_and_fwd_slots_in_both_wrappers(action, row):
    """repo B MotorSystem.apply reads [turn, move]; both rows put turn on dyaw (2) and move on fwd (0)."""
    for cls in (STPPoseExecuted, STPPoseCommanded):
        wrap = cls(_kinematic_env())
        wrap.reset()
        obs, *_ = wrap.step(torch.tensor([list(action)]))
        got = bytes_to_pose(obs["policy"][:, 0, :16, 0])[0].tolist()
        assert got == pytest.approx([*row, 1.0], abs=1e-5), cls.__name__


def test_command_unit_dead_zone_at_the_default_wheeled_limits():
    """Default wheeled limits: move 1.5 units/step (nett_env_cfg wheeled override), turn 20 deg/step (MotorCfg;
    the presenter MEASURED 19.96 x command). The physical dead zone 0.03 units / 0.5 deg is then 0.02 move and
    0.025 turn in command units (0.5 / 19.96 = 0.02505 at the measured gain)."""
    motor_mod = pytest.importorskip("nett_isaac.motor_system")
    cfg_src = Path(motor_mod.__file__).with_name("nett_env_cfg.py").read_text()
    wheeled = cfg_src.split("def _apply_wheeled_overrides", 1)[1].split("\n    def ", 1)[0]
    assert "self.motor.body_move_speed_limit = 1.5\n" in wheeled
    assert motor_mod.MotorCfg().body_turn_speed_limit == 20.0 and motor_mod.MotorCfg().locomotion == "wheeled"
    move_lim, turn_lim = 1.5, 20.0
    assert DEADZONE[0] / move_lim == pytest.approx(0.02) and DEADZONE[2] / turn_lim == pytest.approx(0.025)
    assert DEADZONE[2] / 19.96 == pytest.approx(0.02505, abs=1e-5)
    enc = _enc()
    env = _FakeNett(n=1)
    env.motor.cfg.body_move_speed_limit, env.motor.cfg.body_turn_speed_limit = move_lim, turn_lim
    wrap = STPPoseCommanded(env)
    wrap.reset()
    env.script = [([0.0], [0.0], [0.0], [False])] * 8
    cases = [((0.025, 0.0), True), ((0.0, 0.02), True), ((-0.0249, 0.0199), True),
             ((0.0252, 0.0), False), ((0.0, 0.0202), False), ((-0.0252, 0.0), False)]
    for (turn, move), still in cases:
        obs, *_ = wrap.step(torch.tensor([[turn, move]]))
        a = enc.action(bytes_to_pose(obs["policy"][:, 0, :16, 0]))
        assert bool((a == 0).all()) is still, (turn, move, a)


# ================================================================ knobs and refusals

@pytest.mark.parametrize("value", ["0", "-1e-3", "nan", "inf", "abc", ""])
def test_lr_knob_refuses(monkeypatch, value):
    monkeypatch.setenv("NETT_STP_LR", value)
    with pytest.raises(ValueError, match="NETT_STP_LR"):
        STPTerm(_enc())


@pytest.mark.parametrize("value", ["1", "0", "-4", "2.5", "x", ""])
def test_batch_knob_refuses(monkeypatch, value):
    monkeypatch.setenv("NETT_STP_BATCH", value)
    with pytest.raises(ValueError, match="NETT_STP_BATCH"):
        STPTerm(_enc())


def test_knobs_accept_good_values(monkeypatch):
    monkeypatch.setenv("NETT_STP_LR", "5e-4")
    monkeypatch.setenv("NETT_STP_BATCH", "16")
    term = _term(_enc())
    assert term.lr == 5e-4 and term.batch == 16 and term.optimizer.param_groups[0]["lr"] == 5e-4
    monkeypatch.delenv("NETT_STP_LR")
    monkeypatch.delenv("NETT_STP_BATCH")
    term = STPTerm(_enc())
    assert term.lr == 2e-3 and term.batch == 32


def test_term_refuses_another_encoder():
    host = encoder_mapping["nature_cnn"](gym.spaces.Box(0, 255, (6, 80, 128), np.uint8), features_dim=16)
    with pytest.raises(TypeError, match="STP-E predictor"):
        STPTerm(host)


@pytest.mark.parametrize("kw,match", [
    ({"warp": "bogus"}, "warp must be one of"),
    ({"fov_deg": None}, "RESOLVED camera aperture"),
    ({"fov_deg": 0.0}, "RESOLVED camera aperture"),
    ({"fov_deg": 400.0}, "RESOLVED camera aperture"),
    ({"fov_deg": "150"}, "RESOLVED camera aperture"),
    ({"num_warps": 0}, "num_warps"),
    ({"features_dim": 0}, "features_dim"),
    ({"channels_per_frame": 6}, "channels_per_frame"),
    ({"num_frames": 3}, "num_frames|frames"),
])
def test_encoder_refusals(kw, match):
    args = {"features_dim": 32, "fov_deg": FOV, "num_frames": 2, **kw}
    with pytest.raises(ValueError, match=match):
        STPEEncoder(_space(), **args)


def test_encoder_refuses_wrong_lens_channels_and_room(monkeypatch):
    with pytest.raises(ValueError, match="6 channels|stacked RGB"):
        STPEEncoder(gym.spaces.Box(0, 255, (3, H + 1, W), np.uint8), features_dim=8, fov_deg=FOV)
    with pytest.raises(ValueError, match="pose row"):
        STPEEncoder(gym.spaces.Box(0, 255, (6, H + 1, 12), np.uint8), features_dim=8, fov_deg=FOV)
    for shuffle in ("past", "env"):
        monkeypatch.setenv("NETT_FRAMESTACK_SHUFFLE", shuffle)
        with pytest.raises(ValueError, match="NETT_FRAMESTACK_SHUFFLE"):
            _enc()
    monkeypatch.setenv("NETT_FRAMESTACK_SHUFFLE", "")
    _enc()                                                                  # empty = off, as framestack reads it
    monkeypatch.delenv("NETT_FRAMESTACK_SHUFFLE")
    monkeypatch.setenv("NETT_FISHEYE_METHOD", "equidistant")
    with pytest.raises(ValueError, match="equisolid"):
        _enc()


def test_stray_stp_knob_on_another_label_refuses(campaign, monkeypatch):
    monkeypatch.delenv("NETT_STP_BATCH")
    for label, spec in campaign.MODELS.items():
        assert campaign.refuse_stray_stp_knobs(label, spec) is None
    monkeypatch.setenv("NETT_STP_LR", "1e-3")
    for label in ("CNN2F", "CNN2F-630K", "XSP2", "CNN-UnityRecipe+DVS"):
        with pytest.raises(ValueError, match="not an STP-E arm"):
            campaign.refuse_stray_stp_knobs(label, campaign.MODELS[label])
    for label in STP_LABELS:
        assert campaign.refuse_stray_stp_knobs(label, campaign.MODELS[label]) is None


def test_stp_encoder_cfg_injects_the_resolved_fov_only_for_stp_e(campaign):
    for label, spec in campaign.MODELS.items():
        out = campaign.stp_encoder_cfg(spec, 150.0)
        if spec["encoder"] != "stp_e":
            assert out is spec["cfg"], label                               # the same object: nothing changes
        else:
            assert out == {**spec["cfg"], "fov_deg": 150.0} and "fov_deg" not in spec["cfg"], label
            assert type(out["fov_deg"]) is float
    with pytest.raises(ValueError, match="must not carry fov_deg"):
        campaign.stp_encoder_cfg({"encoder": "stp_e", "cfg": {"fov_deg": 150.0}}, 150.0)


def test_knobs_are_literal_reads_and_in_the_env_index():
    text = (_SRC / "nett_skrl" / "brain" / "aux" / "stp_aux.py").read_text()
    index = (_SRC / "docs" / "env_vars.md").read_text()
    for k in KNOBS:
        assert f'"{k}"' in text and f"`{k}`" in index, k


# ================================================================ existing models unchanged

def _old_module(src, name):
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = str(_TRAIN)
    exec(compile(src, name, "exec"), mod.__dict__)
    return mod


def test_labels_and_registries(campaign):
    for label in STP_LABELS:
        spec = campaign.MODELS[label]
        assert spec["encoder"] == "stp_e" and spec["aux"] == "stp" and spec["framestack"] is True
        assert spec["pre"] == (("stp_pose_cmd",) if label == "STP-E-Cmd" else ("stp_pose_exec",))
    assert campaign.MODELS["STP-E-NoWarp"]["cfg"]["warp"] == "identity"
    assert campaign.MODELS["CNN2F-630K"] == {**campaign.MODELS["CNN2F"],
                                             "cfg": {**campaign.MODELS["CNN2F"]["cfg"], "features_dim": 553}}
    assert encoder_mapping["stp_e"] is STPEEncoder and "stp" in AUX_LOSSES
    assert _load_wrapper("stp_pose_exec") is STPPoseExecuted and _load_wrapper("stp_pose_cmd") is STPPoseCommanded


def test_existing_models_and_registries_are_unchanged(campaign):
    base = _base()
    old = _old_module(_git("show", f"{base}:src_isaac/examples/campaign_train.py"), "_campaign_train_pre_stp")
    assert set(campaign.MODELS) - set(old.MODELS) == NEW_LABELS | VIVSP173_LABELS | BIND16_LABELS | BASELINE_LABELS | S3W_LABELS
    assert set(old.MODELS) <= set(campaign.MODELS)
    for label, spec_old in old.MODELS.items():
        assert campaign.MODELS[label] == spec_old, label
        assert campaign.stp_encoder_cfg(campaign.MODELS[label], 150.0) is campaign.MODELS[label]["cfg"], label
    assert campaign.EXPERIMENTS == old.EXPERIMENTS
    # no existing label reaches the pose row or the stp aux
    for label, spec in campaign.MODELS.items():
        if label not in STP_LABELS:
            assert not {"stp_pose_exec", "stp_pose_cmd"} & set(spec.get("pre", ())), label
            assert spec.get("aux") != "stp" and spec["encoder"] != "stp_e", label

    def keys(src, opener):
        body = src.split(opener, 1)[1].split("\n}", 1)[0]
        return {ln.split('"')[1] for ln in body.splitlines() if ln.strip().startswith('"')}

    for rel, opener, gained in (("nett_skrl/brain/registry.py", "encoder_mapping: dict", {"stp_e"}),
                                ("nett_skrl/brain/aux/ppo_aux.py", "AUX_LOSSES = {", {"stp", "slot_contrast_257"}),
                                ("nett_skrl/body/wrappers/registry.py", "_WRAPPER_SPECS: dict",
                                 {"stp_pose_exec", "stp_pose_cmd"})):
        before = keys(_git("show", f"{base}:src_isaac/{rel}"), opener)
        after = keys((_SRC / rel).read_text(), opener)
        assert before and after - before == gained and before <= after, rel


def test_existing_encoder_builds_are_bit_identical_to_the_base(campaign):
    """CNN2F, XSP2 and DVSC6's encoder built from the base tree's MODELS spec equal today's, weight for weight."""
    old = _old_module(_git("show", f"{_base()}:src_isaac/examples/campaign_train.py"), "_campaign_train_pre_stp2")
    for label in ("CNN2F", "CNN-UnityRecipe+DVS"):
        built = []
        for mod in (old, campaign):
            cfg = {k: v for k, v in campaign.stp_encoder_cfg(mod.MODELS[label], 150.0).items() if k != "trainable"}
            torch.manual_seed(0)
            built.append(encoder_mapping[mod.MODELS[label]["encoder"]](
                gym.spaces.Box(0, 255, (6, 80, 128), np.uint8), **cfg))
        a, b = (_state(m) for m in built)
        assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a), label


# ================================================================ the MODELS import-time assertions

@pytest.mark.parametrize("old,new", [
    # the commanded arm silently reading the executed pose
    ('"STP-E-Cmd":    dict(encoder="stp_e", cfg=dict(STP_E_CFG), framestack=True, pre=("stp_pose_cmd",)',
     '"STP-E-Cmd":    dict(encoder="stp_e", cfg=dict(STP_E_CFG), framestack=True, pre=("stp_pose_exec",)'),
    # NoWarp differing in more than the warp
    ('STP_E_NOWARP_CFG = {**STP_E_CFG, "warp": "identity"}',
     'STP_E_NOWARP_CFG = {**STP_E_CFG, "warp": "identity", "features_dim": 100}'),
    # the pose row leaking into another model's observation
    ('"CNN2F-630K":   dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 553, "conv_dim": 75}, framestack=True)',
     '"CNN2F-630K":   dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 553, "conv_dim": 75}, framestack=True, pre=("stp_pose_exec",))'),
    # the control drifting off CNN2F
    ('"CNN2F-630K":   dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 553, "conv_dim": 75}',
     '"CNN2F-630K":   dict(encoder="nature_cnn", cfg={"trainable": True, "features_dim": 553, "conv_dim": 64}'),
])
def test_import_time_assertions_fire_on_a_mutated_models(campaign, old, new):
    src = _TRAIN.read_text()
    assert src.count(old) == 1, old
    _old_module(src, "_campaign_train_unmutated")                           # the real source passes
    with pytest.raises(AssertionError):
        _old_module(src.replace(old, new), "_campaign_train_mutated")


# ================================================================ presenter review check (1): the turn sign, from physics
#
# The convention chain (each link is a file:line; repo B = nett_isaac):
#   world      +Z up; a body's forward is (-sin yaw, cos yaw) (repo B motor_system.py:141-142), so +yaw turns
#              forward from +Y toward -X: COUNTER-CLOCKWISE seen from above, i.e. a LEFT turn. The wheeled drive
#              commands wz = +d_yaw / dt for +turn (camera_rig.py:481 d_yaw, :500 vel[:, 5]).
#   yaw read   motor.yaw_deg = atan2 yaw about world +Z of the PhysX root quaternion, in [0, 360)
#              (camera_rig.py:129-137 _quat_wxyz_to_yaw_deg, assigned at :694); kinematic: yaw += turn * limit
#              (motor_system.py:136). The eye is root (x) head (x) optical-roll Rx(-90) (camera_rig.py:32-38), so the
#              camera's yaw about world Z changes by exactly the root's: the presenter's offline yaw
#              (2 atan2(q_z, q_w) of the camera sensor quaternion) has the same sign.
#   dyaw       stp_pose.py:187: (yaw_t - yaw_{t-1} + 180) mod 360 - 180, degrees, no sign flip.
#   image      ros optical frame: image +x = world +X when the body faces +Y (camera_rig.py:32-34) = forward x up,
#              the camera's RIGHT (stp_pose.py's lat axis (cos yaw, sin yaw)). Not mirrored.
#   az/el      stp_e.py:110 grid column j -> az = (j + 0.5) dpp - fov/2, increasing; :113 the camera-frame x of the
#              ray is cos(el) sin(az), so +az = camera right; :117 u = cx + r cos(phi) = image right.
#   physics    a LEFT turn by d moves every distant point d to the camera's RIGHT: content moves toward +az by d.
#   the warp   stp_e.py:206 G[TURN, :, 0, 2] = -(10 / dpp)(2 / Wo) gains; a_turn = dyaw / 10 (ACTION_SCALE); the
#              generator is a pure translation, so expm gives A = [I | a_turn G]; warp_affine (stp_e.py:170-178,
#              grid_sample's convention) samples the input at x + A[0, 2] = x - shift, so content moves toward
#              +x = +az by a_turn * 10 / dpp * gain cells = +dyaw / dpp cells for warp 0. Sign +1, as measured
#              offline (presenter_stp README §7a; tools/presenter_stp_offline.py shift_az / EgoSTP.build sgn = +1).
# The tests below do not trust that chain: they render the turn from the env's yaw convention and check the warp.

_EYE_W, _EYE_H, _EYE_FOV = 448, 280, 150.0


def _scene(bearing_deg, el_deg):
    """A distant panorama as a function of WORLD bearing (deg, CCW from +X) and elevation: a narrow red bar, a
    green step edge and a blue grating, so a horizontal shift of either sign is visible in every channel."""
    def wrap(d):
        return (d + 180.0) % 360.0 - 180.0
    red = 0.15 + 0.8 * np.exp(-0.5 * (wrap(bearing_deg - 95.0) / 3.0) ** 2)
    green = 0.2 + 0.6 / (1.0 + np.exp(-wrap(bearing_deg - 70.0) / 2.0)) * (wrap(bearing_deg - 70.0) < 60)
    blue = 0.5 + 0.35 * np.sin(np.radians(bearing_deg) * 9.0) * np.cos(np.radians(el_deg) * 2.0)
    return np.stack([red, green, blue], -1)


def _render_eye(yaw_deg, w=_EYE_W, h=_EYE_H, fov=_EYE_FOV):
    """The equisolid eye at body yaw ``yaw_deg`` in the ENV's convention (forward (-sin, cos), right (cos, sin),
    +Z up), rendered analytically at pixel centres -> (h, w, 3) uint8. Independent of stp_e's remap code."""
    cx, cy = w / 2.0, h / 2.0
    k = cx / math.sin(math.radians(fov / 4.0))
    U, V = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    du, dv = U - cx, -(V - cy)                                             # image right, image up
    r = np.hypot(du, dv)
    th = 2.0 * np.arcsin(np.clip(r / k, 0.0, 1.0))
    ph = np.arctan2(dv, du)
    xc, yc, zc = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)     # camera right, up, forward
    psi = math.radians(yaw_deg)
    fwd, right = (-math.sin(psi), math.cos(psi)), (math.cos(psi), math.sin(psi))
    wx = xc * right[0] + zc * fwd[0]
    wy = xc * right[1] + zc * fwd[1]
    bearing = np.degrees(np.arctan2(wy, wx))
    el = np.degrees(np.arcsin(np.clip(yc, -1.0, 1.0)))
    img = _scene(bearing, el) * (th <= math.radians(fov / 2.0))[..., None]
    return np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)


class _TurningNett(_FakeNett):
    """A fake NETTEnv whose eye renders the panorama at the motor's yaw (camera at the origin: pure rotation)."""

    def __init__(self, yaws):
        super().__init__(n=len(yaws), h=_EYE_H, w=_EYE_W)
        self.motor.yaw_deg = torch.tensor(yaws, dtype=torch.float32)

    def _frame(self):
        frames = np.stack([_render_eye(float(y)) for y in self.motor.yaw_deg])
        return {"policy": torch.from_numpy(frames)}

    def step(self, action):
        x, y, yaw, done = self.script[self.t]
        self.t += 1
        # camera_rig reports yaw in [0, 360) (_quat_wxyz_to_yaw_deg .remainder(360))
        self.motor.yaw_deg = torch.tensor(yaw, dtype=torch.float32).remainder(360.0)
        term = torch.tensor(done, dtype=torch.bool).view(-1, 1)
        return self._frame(), torch.zeros(self.n), term, torch.zeros_like(term), {}


def _turn_obs(yaws, delta):
    env = _TurningNett(yaws)
    chain = ChannelsFirst(FrameStack(STPPoseExecuted(env), n_stack=2))
    chain.reset()
    env.script = [([0.0] * len(yaws), [0.0] * len(yaws), [y + d for y, d in zip(yaws, delta)], [False] * len(yaws))]
    obs, *_ = chain.step(np.zeros((len(yaws), 2), np.float32))
    return torch.as_tensor(np.asarray(obs["policy"]))


def _bar_column(g):
    """Sub-cell column centroid of the red bar on the az/el grid (rows near the horizon)."""
    red = g[:, 0, 25:35].mean(1) - 0.15                                    # (B, Wo), background removed
    red = red.clamp_min(0) ** 2
    cols = torch.arange(red.shape[-1], dtype=red.dtype)
    return (red * cols).sum(-1) / red.sum(-1)


def test_a_left_turn_moves_eye_content_toward_plus_azimuth_in_the_env_convention():
    """Physics alone: +dyaw (CCW, the env's yaw) moves a distant bar toward +az by dyaw / dpp cells on the remap."""
    enc = STPEEncoder(gym.spaces.Box(0, 255, (6, _EYE_H + 1, _EYE_W), np.uint8), features_dim=8,
                      fov_deg=_EYE_FOV, num_frames=2)
    pol = _turn_obs([0.0, 0.0, 357.0], [10.0, -10.0, 10.0])               # the last one wraps 357 -> 7
    pose, x_prev, x_cur = enc.split(pol)
    assert pose[:, 2].tolist() == pytest.approx([10.0, -10.0, 10.0], abs=1e-4)   # stp_pose's dyaw, wrap included
    shift = _bar_column(enc.azel(x_cur)) - _bar_column(enc.azel(x_prev))
    assert shift.tolist() == pytest.approx([10.0 / 1.5, -10.0 / 1.5, 10.0 / 1.5], abs=0.3)


def test_positive_a_turn_moves_a_bright_column_toward_plus_azimuth():
    """The warm-start warp 0 under a_turn > 0 moves a bright grid column the way the physics test above does."""
    pred = STPPredictor(100, 4)
    g = torch.zeros(2, 3, 60, 100)
    g[:, :, :, 40] = 1.0
    a = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])                 # dyaw = +-10 deg after the scale
    out = warp_affine(g, pred.affines(a)[:, :1])[:, 0]
    prof = out[:, 0, 30]
    cols = torch.arange(100, dtype=prof.dtype)
    centroid = (prof * cols).sum(-1) / prof.sum(-1)
    assert centroid.tolist() == pytest.approx([40 + 10 / 1.5, 40 - 10 / 1.5], abs=1e-4)
    assert int(prof[0].argmax()) > 40 > int(prof[1].argmax())


def _err(a, b, w):
    return float((((a - b) ** 2) * w).sum() / (w.sum() * a.shape[1]))


@pytest.mark.parametrize("delta", [12.0, -12.0, 6.0, -6.0])
def test_warm_start_warp_predicts_the_rendered_turn_better_than_copy_and_than_the_flipped_sign(delta):
    """Render t and t+1 at yaw psi and psi + delta (env convention), carry the pose row stp_pose_exec writes, remap with
    the encoder, and apply warp 0 of the warm-start predictor at a and at -a."""
    enc = STPEEncoder(gym.spaces.Box(0, 255, (6, _EYE_H + 1, _EYE_W), np.uint8), features_dim=8,
                      fov_deg=_EYE_FOV, num_frames=2)
    pol = _turn_obs([30.0, 355.0], [delta, delta])
    pose, x_prev, x_cur = enc.split(pol)
    assert pose[:, 2].tolist() == pytest.approx([delta, delta], abs=1e-4)
    g_prev, g_cur = enc.azel(x_prev), enc.azel(x_cur)
    a = enc.action(pose)
    assert a[:, :2].abs().max() == 0 and bool((a[:, 2] * delta > 0).all())
    vmb = enc.vm.expand(2, 1, 60, 100)
    warped = {}
    for name, act in (("warm", a), ("flip", -a)):
        J = warp_affine(torch.cat([g_prev, vmb], 1), enc.predictor.affines(act)[:, :1])[:, 0]
        warped[name] = J
    w = enc.vm * (warped["warm"][:, 3:] > 0.999) * (warped["flip"][:, 3:] > 0.999)     # the same cells for all three
    assert float(w.mean()) > 0.4
    e_copy = _err(g_prev, g_cur, w)
    e_warm = _err(warped["warm"][:, :3], g_cur, w)
    e_flip = _err(warped["flip"][:, :3], g_cur, w)
    print(f"\ndelta {delta:+.0f}: error / copy  warm {e_warm / e_copy:.3f}  flipped {e_flip / e_copy:.3f}")
    assert e_warm < 0.25 * e_copy, (e_warm, e_copy)
    assert e_warm < 0.25 * e_flip, (e_warm, e_flip)
    assert e_flip > e_copy                                                 # the wrong sign is worse than copy


@pytest.mark.parametrize("delta", [12.0, -12.0])
def test_full_encoder_residual_shrinks_on_a_rendered_turn_and_grows_with_the_sign_flipped(delta):
    """The real residual path (split -> remap -> action -> predictor -> r), masks pinned to warp 0: smaller than the
    NoWarp frame difference, and larger again if the turn generator's sign is flipped."""
    space = gym.spaces.Box(0, 255, (6, _EYE_H + 1, _EYE_W), np.uint8)
    pol = _turn_obs([30.0, 200.0], [delta, delta])
    out = {}
    for name in ("warm", "flip", "nowarp"):
        torch.manual_seed(0)
        enc = STPEEncoder(space, features_dim=8, fov_deg=_EYE_FOV, num_frames=2,
                          warp="identity" if name == "nowarp" else "affine")
        if name == "flip":
            with torch.no_grad():
                enc.predictor.G[2].neg_()

        def onehot(x, m=enc.predictor.M):
            masks = torch.zeros(x.shape[0], m + 1, *x.shape[-2:])
            masks[:, 0] = 1.0
            return masks

        enc.predictor.masks = onehot
        out[name] = enc.residual_input(pol)[:, :3]
    common = (out["warm"] != 0) | (out["flip"] != 0) | (out["nowarp"] != 0)
    energy = {k: float((v ** 2).sum()) for k, v in out.items()}
    assert energy["warm"] < 0.3 * energy["nowarp"], energy
    assert energy["flip"] > energy["warm"] * 3, energy
    assert bool(common.any())
