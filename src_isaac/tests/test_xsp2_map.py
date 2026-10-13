"""XSP2-MAP20x32 -- XSP2 with a finer figure map (owner, workspace DECISIONS §104, 2026-10-13: "try using a
map that is more detailed than just 16x10"). encoders/xsp2.py ``map_stride``; label in examples/campaign_train.py.

Known answers, each with a control that can fail:
  1. map_stride 8 (the default, the "XSP2" label) is the pre-change encoder EXACTLY: same weights from the
     same seed, same streams and policy features, bit for bit, against xsp2.py as it was before map_stride.
  2. The label builds through MODELS + EncoderCfg: 20x32 map, 640 features, and the SAME 166,510 parameters
     split the same way; only phi_e's first conv differs (stride 2, padding 3 vs 4, 2).
  3. Streams at 20x32; c_t is m*d upsampled NEAREST by 4 to the 80x128 eye (and NOT by 8: control).
  4. The policy reads 640 detached numbers through the real builder, and one PPO update with the xsp2 aux runs.
  5. Refusals: another stride, a features_dim that is not the map size, an eye not divisible by 4.
  6. The xsp2 aux step and its orientation telemetry run on the 20x32 map (block size from the shapes).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nett_skrl.brain.aux.xsp2_aux import XSP2Term
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders import xsp2 as xsp2_mod
from nett_skrl.brain.encoders.xsp2 import MAP_STRIDES, XSP2Encoder
from nett_skrl.brain.registry import encoder_mapping

H, W, CPF = 80, 128, 3
C = 2 * CPF
LABEL = "XSP2-MAP20x32"
_SRC = Path(__file__).resolve().parents[1]
_TRAIN = _SRC / "examples" / "campaign_train.py"
_ENC_PATH = "nett_skrl/brain/encoders/xsp2.py"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("NETT_XSP2_BATCH", "NETT_XSP2_PREDICTOR", "NETT_XSP2_HIDDEN", "NETT_XSP2_REFINE_HIDDEN",
                 "NETT_XSP2_LR_SCALE", "NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_AUX_LOSS", "NETT_AUX_WEIGHT",
                 "NETT_AUX_WEIGHT_OVERRIDE", "NETT_AUX_BATCH", "NETT_AUX_STRICT", "NETT_DISABLE_FEATURE_CACHE",
                 "NETT_CONV3D_ORTHO_INIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_XSP2_BATCH", "8")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _space(channels=C, h=H, w=W):
    return gym.spaces.Box(low=0, high=255, shape=(channels, h, w), dtype=np.uint8)


def _n(module):
    return sum(p.numel() for p in module.parameters())


def _prepared(n=4, seed=5):
    return torch.rand(n, C, H, W, generator=torch.Generator().manual_seed(seed))


def _label_kwargs(campaign, label):
    kwargs = EncoderCfg(**campaign.MODELS[label]["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    return kwargs


def _git(*args):
    env = {k: v for k, v in os.environ.items()
           if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_COMMON_DIR")}
    try:
        return subprocess.run(["git", *args], cwd=_SRC, capture_output=True, text=True, check=True, env=env).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git unavailable here: {exc}")


def _pre_map_encoder_class():
    """XSP2Encoder as it was before map_stride: xsp2.py at the parent of the first commit that has MAP_STRIDES
    (HEAD while the change is uncommitted), loaded inside the package so its relative imports resolve."""
    commits = _git("log", "--format=%H", "-S", "MAP_STRIDES", "--", _ENC_PATH).split()
    base = f"{commits[-1]}^" if commits else "HEAD"
    src = _git("show", f"{base}:src_isaac/{_ENC_PATH}")
    if not src.strip():
        pytest.fail(f"xsp2.py at {base} is empty -- the comparison cannot look")
    assert "MAP_STRIDES" not in src and "map_stride" not in src, base
    name = "nett_skrl.brain.encoders._xsp2_pre_map"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "nett_skrl.brain.encoders"
    exec(compile(src, f"xsp2.py@{base}", "exec"), mod.__dict__)
    return mod.XSP2Encoder


# ================================================================ 1. the default is the old encoder, bit for bit

def test_default_stride_is_the_pre_change_encoder_bit_for_bit():
    Old = _pre_map_encoder_class()
    torch.manual_seed(3)
    old = Old(_space(), features_dim=160)
    torch.manual_seed(3)
    new = XSP2Encoder(_space(), features_dim=160)
    torch.manual_seed(3)
    explicit = XSP2Encoder(_space(), features_dim=160, map_stride=8)
    so, sn, se = old.state_dict(), new.state_dict(), explicit.state_dict()
    assert list(so) == list(sn) == list(se)
    assert all(torch.equal(so[k], sn[k]) and torch.equal(so[k], se[k]) for k in so)
    p = _prepared()
    a, b = old.encode_streams(p), new.encode_streams(p)
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    obs = torch.randint(0, 256, (4, C * H * W)).float()
    assert torch.equal(old(obs), new(obs))
    # control: the finer map from the same seed is a DIFFERENT forward (it does not alias the old path)
    torch.manual_seed(3)
    fine = XSP2Encoder(_space(), features_dim=640, map_stride=4)
    assert fine(obs).shape == (4, 640)


# ================================================================ 2. the label: shapes, counts, the one changed conv

def test_label_builds_with_the_same_parameters_and_only_conv1_moves(campaign):
    spec, base = campaign.MODELS[LABEL], campaign.MODELS["XSP2"]
    assert {k: v for k, v in spec.items() if k != "cfg"} == {k: v for k, v in base.items() if k != "cfg"}
    assert spec["cfg"] == {**base["cfg"], "features_dim": 640, "map_stride": 4}
    fine = encoder_mapping["xsp2"](_space(), **_label_kwargs(campaign, LABEL))
    coarse = encoder_mapping["xsp2"](_space(), **_label_kwargs(campaign, "XSP2"))
    assert fine.map_hw == (20, 32) and fine.features_dim == 640 and fine.map_stride == 4
    assert coarse.map_hw == (10, 16) and coarse.features_dim == 160 and coarse.map_stride == 8
    counts = lambda e: (_n(e), _n(e.trunk), _n(e.ventral), _n(e.dorsal))
    assert counts(fine) == counts(coarse) == (166_510, 82_283, 31_169, 53_058)
    convs = lambda e: [(m.kernel_size, m.stride, m.padding, m.in_channels, m.out_channels)
                       for m in e.modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))]
    cf, cc = convs(fine), convs(coarse)
    assert cf[0] == ((8, 8), (2, 2), (3, 3), 3, 32) and cc[0] == ((8, 8), (4, 4), (2, 2), 3, 32)
    assert cf[1:] == cc[1:]
    assert MAP_STRIDES == {8: (4, 2), 4: (2, 3)} and xsp2_mod.TRUNK_STRIDE == 8


# ================================================================ 3. streams and the x4 upsample

def test_streams_are_20x32_and_c_is_upsampled_by_4():
    torch.manual_seed(0)
    enc = XSP2Encoder(_space(), features_dim=640, map_stride=4)
    s = enc.encode_streams(_prepared())
    assert s["z_t"].shape == (4, 75, 20, 32)
    assert s["m"].shape == (4, 1, 20, 32) and s["d"].shape == (4, 2, 20, 32)
    assert s["c"].shape == (4, 2, H, W)
    assert torch.equal(s["c"], F.interpolate(s["m"] * s["d"], scale_factor=4, mode="nearest"))
    # control: the XSP2 factor (8) would overshoot the eye at this grid
    assert F.interpolate(s["m"] * s["d"], scale_factor=8, mode="nearest").shape[-2:] == (2 * H, 2 * W)
    obs = torch.randint(0, 256, (4, C * H * W)).float()
    feats = enc(obs)
    assert feats.shape == (4, 640) and feats.requires_grad is False
    m = enc.ventral_map(enc.trunk(enc.current_frame(enc._prepare_image(obs))))
    assert torch.equal(feats, m.flatten(1))


# ================================================================ 4. the real builder, one PPO update with the aux

class _Env:
    num_envs = 1
    num_agents = 1
    observation_space = _space()
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    state_space = None


def test_policy_reads_640_through_the_real_builder_and_one_update_runs(campaign, monkeypatch):
    from nett_skrl.brain.agent_factory import build_agents
    from nett_skrl.brain.brain import Brain
    monkeypatch.setenv("NETT_AUX_LOSS", "xsp2")
    monkeypatch.setenv("NETT_AUX_WEIGHT", "1")
    cfg = {k: v for k, v in campaign.MODELS[LABEL]["cfg"].items() if k != "trainable"}
    torch.manual_seed(0)
    brain = Brain(encoder="xsp2", algorithm="PPO", encoder_cfg=cfg,
                  algorithm_cfg={"rollouts": 16, "mini_batches": 1, "learning_epochs": 1, "grad_norm_clip": 0.5,
                                 "learning_rate": 1e-4},
                  model={"shared_encoder": True, "hidden_sizes": [64]})
    agent = build_agents(brain, _Env(), torch.device("cpu"))[0]
    agent.init(trainer_cfg=None)
    agent.enable_training_mode(True)
    enc = agent.policy.encoder
    assert enc.map_hw == (20, 32) and enc.features_dim == 640
    lin = [m for m in agent.policy.trunk.modules() if isinstance(m, nn.Linear)]
    assert lin[0].in_features == 640
    vlin = [m for m in agent.value.trunk.modules() if isinstance(m, nn.Linear)]
    assert vlin[0].in_features == 640
    before = {k: v.detach().clone() for k, v in enc.state_dict().items()}
    g = torch.Generator().manual_seed(1)
    torch.manual_seed(1)
    obs = torch.randint(0, 256, (1, C * H * W), generator=g).float()
    with torch.no_grad():
        for t in range(16):
            acts, _ = agent.act(obs, None, timestep=t, timesteps=100)
            nxt = torch.randint(0, 256, (1, C * H * W), generator=g).float()
            agent.record_transition(observations=obs, states=None, actions=acts,
                                    rewards=torch.rand(1, 1, generator=g), next_observations=nxt, next_states=None,
                                    terminated=torch.zeros(1, 1, dtype=torch.bool),
                                    truncated=torch.zeros(1, 1, dtype=torch.bool), infos={}, timestep=t,
                                    timesteps=100)
            obs = nxt
    agent.update(timestep=16, timesteps=100)
    after = enc.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before), "the aux Adam did not step the streams"


# ================================================================ 5. refusals

@pytest.mark.parametrize("kw, match", [
    ({"features_dim": 640, "map_stride": 2}, "map_stride must be one of"),
    ({"features_dim": 640, "map_stride": 16}, "map_stride must be one of"),
    ({"features_dim": 640, "map_stride": "4"}, "map_stride must be one of"),
    ({"features_dim": 160, "map_stride": 4}, "features_dim must be 640"),
    ({"features_dim": 640}, "features_dim must be 160"),
])
def test_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        XSP2Encoder(_space(), **kw)


def test_eye_must_divide_by_the_map_stride():
    with pytest.raises(ValueError, match="divisible by 4"):
        XSP2Encoder(_space(h=82, w=128), features_dim=640, map_stride=4)
    # control: 84x128 divides by 4 (not by 8), so the finer map builds where XSP2 refuses
    assert XSP2Encoder(_space(h=84, w=128), features_dim=21 * 32, map_stride=4).map_hw == (21, 32)
    with pytest.raises(ValueError, match="divisible by 8"):
        XSP2Encoder(_space(h=84, w=128), features_dim=160)


# ================================================================ 6. the aux step and orientation telemetry

class _NoiseMemory:
    def __init__(self, t_max=24, n_env=2, seed=3):
        g = torch.Generator().manual_seed(seed)
        acts = torch.zeros(t_max, n_env, 2)
        acts[:, :, 0] = torch.linspace(-0.5, 0.5, t_max).view(t_max, 1)
        self.tensors = {
            "observations": torch.randint(0, 255, (t_max, n_env, H, W, C), generator=g, dtype=torch.uint8),
            "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "actions": acts,
        }
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


def test_aux_step_and_orientation_telemetry_run_on_the_20x32_map():
    torch.manual_seed(1)
    enc = XSP2Encoder(_space(), features_dim=640, map_stride=4)
    term = XSP2Term(enc)
    term.build_optimizer(1e-3)
    term.attach_memory(_NoiseMemory())
    before = {k: v.detach().clone() for k, v in enc.state_dict().items()}
    torch.manual_seed(2)
    term.compute(enc, None)
    s = term.last_scalars
    for key in ("loss", "skill", "fg_frac", "fg_entropy", "fg_motion_sep", "flow_absmean"):
        assert key in s and np.isfinite(s[key]), (key, s.get(key))
    assert 0 < s["fg_frac"] < 1
    assert any(not torch.equal(before[k], v) for k, v in enc.state_dict().items())
