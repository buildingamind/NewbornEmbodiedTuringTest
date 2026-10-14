"""XSP2-WHAT -- XSP2 whose policy also reads what is under the figure (owner, workspace DECISIONS §105/§106,
2026-10-14). encoders/xsp2.py ``policy_input`` "fg" / ``pool_beta`` / ``trunk_widths``; label in
examples/campaign_train.py.

Known answers, each with a control that can fail:
  1. The defaults (policy_input "map", trunk_widths (32, 64)) are the pre-change encoder EXACTLY: same weights
     from the same seed, same streams and policy features, bit for bit, against xsp2.py before policy_input.
  2. The "fg" features are cat(flatten(m), sum_loc softmax(beta * standardised logits) * z_t), recomputed here
     from the streams by hand; a one-hot logit picks that location's z_t (and NOT another location's).
  3. The pool is scale-free in the logits (x10 on the ventral output gives the same attention) and detached.
  4. The label builds through MODELS + EncoderCfg with the SAME 166,510 parameters; the policy MLP reads 235,
     and one PPO update with the xsp2 aux runs and steps the streams.
  5. Refusals: an unknown policy_input, a non-positive pool_beta, bad trunk_widths, a features_dim that is not
     map + conv_dim (and the map mode still refuses 235).
  6. trunk_widths shrinks phi_e only where asked and the streams/aux still run (the size-reduction path).
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

from nett_skrl.brain.aux.xsp2_aux import XSP2Term
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.xsp2 import POLICY_INPUTS, XSP2Encoder
from nett_skrl.brain.registry import encoder_mapping

H, W, CPF = 80, 128, 3
C = 2 * CPF
LABEL = "XSP2-WHAT"
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


def _obs(n=4, seed=7):
    return torch.randint(0, 256, (n, C * H * W), generator=torch.Generator().manual_seed(seed)).float()


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


def _pre_what_encoder_class():
    """XSP2Encoder before policy_input: xsp2.py at the parent of the first commit that has POLICY_INPUTS (HEAD
    while the change is uncommitted), loaded inside the package so its relative imports resolve."""
    commits = _git("log", "--format=%H", "-S", "POLICY_INPUTS", "--", _ENC_PATH).split()
    base = f"{commits[-1]}^" if commits else "HEAD"
    src = _git("show", f"{base}:src_isaac/{_ENC_PATH}")
    if not src.strip():
        pytest.fail(f"xsp2.py at {base} is empty -- the comparison cannot look")
    assert "POLICY_INPUTS" not in src and "policy_input" not in src and "MAP_STRIDES" in src, base
    name = "nett_skrl.brain.encoders._xsp2_pre_what"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "nett_skrl.brain.encoders"
    exec(compile(src, f"xsp2.py@{base}", "exec"), mod.__dict__)
    return mod.XSP2Encoder


# ================================================================ 1. the defaults are the old encoder, bit for bit

@pytest.mark.parametrize("stride, n_map", [(8, 160), (4, 640)])
def test_default_is_the_pre_change_encoder_bit_for_bit(stride, n_map):
    Old = _pre_what_encoder_class()
    torch.manual_seed(3)
    old = Old(_space(), features_dim=n_map, map_stride=stride)
    torch.manual_seed(3)
    new = XSP2Encoder(_space(), features_dim=n_map, map_stride=stride)
    torch.manual_seed(3)
    explicit = XSP2Encoder(_space(), features_dim=n_map, map_stride=stride, policy_input="map", pool_beta=4.0,
                           trunk_widths=(32, 64))
    so, sn, se = old.state_dict(), new.state_dict(), explicit.state_dict()
    assert list(so) == list(sn) == list(se)
    assert all(torch.equal(so[k], sn[k]) and torch.equal(so[k], se[k]) for k in so)
    p = torch.rand(4, C, H, W, generator=torch.Generator().manual_seed(5))
    a, b = old.encode_streams(p), new.encode_streams(p)
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)
    obs = _obs()
    assert torch.equal(old(obs), new(obs)) and torch.equal(old(obs), explicit(obs))
    # control: "fg" from the same seed is a DIFFERENT, wider output whose first n_map values are the old map
    torch.manual_seed(3)
    fg = XSP2Encoder(_space(), features_dim=n_map + 75, map_stride=stride, policy_input="fg")
    out = fg(obs)
    assert out.shape == (4, n_map + 75) and torch.equal(out[:, :n_map], old(obs))


# ================================================================ 2. the fg features, by hand

def _by_hand(enc, obs, beta):
    x_t = enc.current_frame(enc._prepare_image(obs))
    z = enc.trunk(x_t)
    logit = enc.ventral(z).flatten(1)
    lh = (logit - logit.mean(1, keepdim=True)) / (logit.std(1, keepdim=True, unbiased=False) + 1e-6)
    a = torch.softmax(beta * lh, 1)
    fg = torch.einsum("bcl,bl->bc", z.flatten(2), a)
    return torch.cat([torch.sigmoid(logit), fg], 1), z, a


@pytest.mark.parametrize("beta", [1.0, 4.0, 8.0])
def test_fg_features_are_map_plus_figure_attention_pool(beta):
    torch.manual_seed(0)
    enc = XSP2Encoder(_space(), features_dim=235, policy_input="fg", pool_beta=beta)
    obs = _obs()
    out = enc(obs)
    with torch.no_grad():
        want, z, a = _by_hand(enc, obs, beta)
    assert out.shape == (4, 235)
    torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(a.sum(1), torch.ones(4))
    # control: a uniform average pool (no figure weighting) is NOT what the encoder returns
    assert not torch.allclose(out[:, 160:], z.flatten(2).mean(-1), atol=1e-4)


def test_one_hot_logit_selects_that_location():
    torch.manual_seed(0)
    enc = XSP2Encoder(_space(), features_dim=235, policy_input="fg", pool_beta=8.0)
    last = enc.ventral[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
    target = 3 * 16 + 11                                       # row 3, col 11 of the 10x16 grid
    hook = enc.ventral.register_forward_hook(
        lambda mod, inp, out: out + 100.0 * nn.functional.one_hot(torch.tensor(target), 160).view(1, 1, 10, 16))
    obs = _obs(n=2)
    out = enc(obs)
    hook.remove()
    with torch.no_grad():
        z = enc.trunk(enc.current_frame(enc._prepare_image(obs))).flatten(2)
    torch.testing.assert_close(out[:, 160:], z[:, :, target], rtol=1e-4, atol=1e-5)
    # control: a neighbouring location's features differ
    assert not torch.allclose(out[:, 160:], z[:, :, target + 1], atol=1e-3)


# ================================================================ 3. scale-free and detached

def test_attention_is_scale_free_and_output_is_detached():
    torch.manual_seed(0)
    enc = XSP2Encoder(_space(), features_dim=235, policy_input="fg")
    logit = torch.randn(3, 160)
    torch.testing.assert_close(enc.figure_attention(logit), enc.figure_attention(10.0 * logit + 2.0))
    # control: beta does change the attention (the knob is live)
    enc8 = XSP2Encoder(_space(), features_dim=235, policy_input="fg", pool_beta=8.0)
    assert not torch.allclose(enc.figure_attention(logit), enc8.figure_attention(logit))
    out = enc(_obs())
    assert out.requires_grad is False and out.grad_fn is None


# ================================================================ 4. the label, the real builder, one PPO update

def test_label_is_xsp2_with_the_policy_input_moved(campaign):
    spec, base = campaign.MODELS[LABEL], campaign.MODELS["XSP2"]
    assert {k: v for k, v in spec.items() if k != "cfg"} == {k: v for k, v in base.items() if k != "cfg"}
    assert spec["cfg"] == {**base["cfg"], "features_dim": 235, "policy_input": "fg", "pool_beta": 4.0}
    what = encoder_mapping["xsp2"](_space(), **_label_kwargs(campaign, LABEL))
    plain = encoder_mapping["xsp2"](_space(), **_label_kwargs(campaign, "XSP2"))
    assert what.policy_input == "fg" and what.pool_beta == 4.0 and what.features_dim == 235
    assert plain.policy_input == "map" and plain.features_dim == 160
    counts = lambda e: (_n(e), _n(e.trunk), _n(e.ventral), _n(e.dorsal))
    assert counts(what) == counts(plain) == (166_510, 82_283, 31_169, 53_058)


class _Env:
    num_envs = 1
    num_agents = 1
    observation_space = _space()
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    state_space = None


def test_policy_reads_235_through_the_real_builder_and_one_update_runs(campaign, monkeypatch):
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
    assert enc.policy_input == "fg" and enc.features_dim == 235
    assert [m for m in agent.policy.trunk.modules() if isinstance(m, nn.Linear)][0].in_features == 235
    assert [m for m in agent.value.trunk.modules() if isinstance(m, nn.Linear)][0].in_features == 235
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
    ({"features_dim": 235, "policy_input": "both"}, "policy_input must be one of"),
    ({"features_dim": 235, "policy_input": "fg", "pool_beta": 0.0}, "pool_beta must be a positive"),
    ({"features_dim": 235, "policy_input": "fg", "pool_beta": "4"}, "pool_beta must be a positive"),
    ({"features_dim": 160, "trunk_widths": (32,)}, "trunk_widths must be two positive ints"),
    ({"features_dim": 160, "trunk_widths": (32, 0)}, "trunk_widths must be two positive ints"),
    ({"features_dim": 160, "policy_input": "fg"}, "features_dim must be 235"),
    ({"features_dim": 235}, "features_dim must be 160"),
])
def test_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        XSP2Encoder(_space(), **kw)
    assert POLICY_INPUTS == ("map", "fg")


# ================================================================ 6. the size-reduction path

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


def test_trunk_widths_shrink_phi_e_and_the_aux_step_runs():
    torch.manual_seed(1)
    enc = XSP2Encoder(_space(), features_dim=160 + 38, conv_dim=38, ventral_width=16, dorsal_width=16,
                      trunk_widths=(16, 32), policy_input="fg")
    convs = [(m.in_channels, m.out_channels) for m in enc.trunk if isinstance(m, nn.Conv2d)]
    assert convs == [(3, 16), (16, 32), (32, 38)]
    full = XSP2Encoder(_space(), features_dim=160)
    assert _n(enc) < _n(full) / 3                              # control: the full encoder is 166,510
    assert enc(_obs()).shape == (4, 198)
    term = XSP2Term(enc)
    term.build_optimizer(1e-3)
    term.attach_memory(_NoiseMemory())
    before = {k: v.detach().clone() for k, v in enc.state_dict().items()}
    torch.manual_seed(2)
    term.compute(enc, None)
    assert np.isfinite(term.last_scalars["loss"]) and 0 < term.last_scalars["fg_frac"] < 1
    assert any(not torch.equal(before[k], v) for k, v in enc.state_dict().items())
