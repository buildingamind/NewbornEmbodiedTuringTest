"""XSP2 -- owner spec, workspace DECISIONS §87 with spec changes 1-3: encoder brain/encoders/xsp2.py,
aux brain/aux/xsp2_aux.py, label "XSP2" in examples/campaign_train.py.

Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU). Sections follow the brief's items:
  1. shapes, pinned parameter counts, the policy width through the real builder, refusals
  2. the gather-based bilinear warp: known answer, bitwise identity at c = 0, grid_sample cross-check
  3. the summed loss and its target (the NEWEST frame of obs[t+1])
  4. reward detachment (XSP v1 as the positive control)
  5. optimizer exclusion, one aux Adam step per compute(), PPO bit-identical to the aux-disabled model
  6. determinism; init (Kaiming trunk, PyTorch default elsewhere, orthogonal for every other label)
  7. synthetic learning on a textured moving square, orientation reported per seed
  8. knobs, env index, and invariance against the commit before XSP2
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
import torch.nn as nn
import torch.nn.functional as F

from nett_skrl.brain.aux.knobs import NOT_MEASURED
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.aux.xsp2_aux import (ConvPredictor, WarpRefiner, XSP2Term, exclude_from_optimizer,
                                          warp_bilinear)
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.xsp import XSPEncoder
from nett_skrl.brain.encoders.xsp2 import XSP2Encoder, upsample_nearest
from nett_skrl.brain.registry import encoder_mapping

H, W, CPF = 80, 128, 3
C = 2 * CPF
_SRC = Path(__file__).resolve().parents[1]
_TRAIN = _SRC / "examples" / "campaign_train.py"
XSP2_KNOBS = ("NETT_XSP2_BATCH", "NETT_XSP2_PREDICTOR", "NETT_XSP2_HIDDEN", "NETT_XSP2_REFINE_HIDDEN",
              "NETT_XSP2_LR_SCALE")
TELEMETRY = ("B", "loss", "mse", "copy_mse", "skill", "fg_frac", "fg_frac_std", "fg_entropy",
             "fg_motion_sep", "fg_motion_sep_parked", "fg_motion_n_parked", "flow_absmean", "d_absmean",
             "enc_grad_norm")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in XSP2_KNOBS + ("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_AUX_LOSS", "NETT_AUX_WEIGHT",
                              "NETT_AUX_WEIGHT_OVERRIDE", "NETT_AUX_BATCH", "NETT_AUX_STRICT",
                              "NETT_DISABLE_FEATURE_CACHE", "NETT_CONV3D_ORTHO_INIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_XSP2_BATCH", "8")
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


def _space(channels=C, h=H, w=W):
    return gym.spaces.Box(low=0, high=255, shape=(channels, h, w), dtype=np.uint8)


def _enc(seed=0, **kw):
    torch.manual_seed(seed)
    return XSP2Encoder(_space(), **{"features_dim": 160, **kw})


def _n(module):
    return sum(p.numel() for p in module.parameters())


def _prepared(n=4, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, C, H, W, generator=g)


def _term(enc, lr=1e-3):
    term = XSP2Term(enc)
    term.build_optimizer(lr)
    return term


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


# ---------------------------------------------------------------- a real skrl agent (no simulator)
class _Env:
    num_envs = 1
    num_agents = 1
    observation_space = _space()
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    state_space = None


def _agent(monkeypatch, *, aux="xsp2", seed=0, epochs=1, mini_batches=1, rollouts=16, shared=True,
           hidden=(), lr=1e-4, weight="1"):
    """The production path: Brain -> build_agents -> AuxLossPPO (aux="xsp2") or MetricsPPO (aux=None)."""
    from nett_skrl.brain.agent_factory import build_agents
    from nett_skrl.brain.brain import Brain
    if aux is None:
        monkeypatch.delenv("NETT_AUX_LOSS", raising=False)
        monkeypatch.delenv("NETT_AUX_WEIGHT", raising=False)
    else:
        monkeypatch.setenv("NETT_AUX_LOSS", aux)
        monkeypatch.setenv("NETT_AUX_WEIGHT", weight)
    torch.manual_seed(seed)
    brain = Brain(encoder="xsp2", algorithm="PPO",
                  encoder_cfg={"features_dim": 160, "conv_dim": 75, "num_frames": 2},
                  algorithm_cfg={"rollouts": rollouts, "mini_batches": mini_batches, "learning_epochs": epochs,
                                 "grad_norm_clip": 0.5, "learning_rate": lr},
                  model={"shared_encoder": shared, "hidden_sizes": list(hidden)})
    agent = build_agents(brain, _Env(), torch.device("cpu"))[0]
    agent.init(trainer_cfg=None)
    agent.enable_training_mode(True)
    return agent


def _rollout(agent, seed=1, steps=16):
    """skrl's trainer loop shape: act + record_transition under no_grad, then update()."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)                                   # the policy's action sampling
    n = C * H * W
    obs = torch.randint(0, 256, (1, n), generator=g).float()
    with torch.no_grad():
        for t in range(steps):
            acts, _ = agent.act(obs, None, timestep=t, timesteps=100)
            nxt = torch.randint(0, 256, (1, n), generator=g).float()
            agent.record_transition(observations=obs, states=None, actions=acts,
                                    rewards=torch.rand(1, 1, generator=g), next_observations=nxt,
                                    next_states=None, terminated=torch.zeros(1, 1, dtype=torch.bool),
                                    truncated=torch.zeros(1, 1, dtype=torch.bool), infos={},
                                    timestep=t, timesteps=100)
            obs = nxt
    torch.manual_seed(seed + 1000)                            # the update's minibatch permutation
    agent.update(timestep=steps, timesteps=100)


def _record_ppo_grads(agent):
    """Every gradient PPO's optimizer sees at each of its steps, by parameter NAME (policy + value)."""
    steps = []
    named = [*(("policy." + k, p) for k, p in agent.policy.named_parameters()),
             *(("value." + k, p) for k, p in agent.value.named_parameters())]

    def hook(_opt, _args, _kwargs):
        steps.append({k: None if p.grad is None else p.grad.detach().clone() for k, p in named})

    agent.optimizer.register_step_pre_hook(hook)
    return steps


def _state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _states_equal(a, b):
    return list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)


# ================================================================ 1. shapes, counts, builder width, refusals

def test_label_and_registries(campaign):
    spec = campaign.MODELS["XSP2"]
    assert spec == dict(encoder="xsp2", cfg=campaign.XSP2_CFG, framestack=True, aux="xsp2", aux_weight=1.0)
    assert spec["cfg"]["features_dim"] == 160 and spec["cfg"]["num_frames"] == 2
    assert encoder_mapping["xsp2"] is XSP2Encoder
    assert "xsp2" in AUX_LOSSES and "xsp" in AUX_LOSSES
    assert XSP2Encoder.N_FRAMES == 2 and XSP2Encoder.CPF == 3 and XSP2Encoder.OWNS_INIT is True


def test_label_encoder_builds_with_pinned_counts(campaign):
    kwargs = EncoderCfg(**campaign.MODELS["XSP2"]["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    enc = encoder_mapping["xsp2"](_space(), **kwargs)
    assert (_n(enc), _n(enc.trunk), _n(enc.ventral), _n(enc.dorsal)) == (166_510, 82_283, 31_169, 53_058)
    assert enc.features_dim == 160 and enc.map_hw == (10, 16)
    assert len(enc.stream_parameters()) == len(list(enc.parameters()))
    assert _n(WarpRefiner(3, 32)) == 12_451 and _n(ConvPredictor(3, 64)) == 78_531
    assert _n(XSP2Term(enc).predictor) == 12_451


def test_stream_shapes_and_nearest_upsample():
    enc = _enc()
    s = enc.encode_streams(_prepared())
    assert s["z_t"].shape == (4, 75, 10, 16)
    assert s["m"].shape == (4, 1, 10, 16) and bool(((s["m"] > 0) & (s["m"] < 1)).all())
    assert s["d"].shape == (4, 2, 10, 16) and bool((s["d"] < 0).any())          # linear, signed
    assert s["c"].shape == (4, 2, H, W)
    assert torch.equal(s["c"], F.interpolate(s["m"] * s["d"], scale_factor=8, mode="nearest"))
    x = torch.randn(2, 3, 5, 7)
    assert torch.equal(upsample_nearest(x, 8), F.interpolate(x, scale_factor=8, mode="nearest"))
    assert enc(torch.randint(0, 256, (4, C * H * W)).float()).shape == (4, 160)


def test_trunk_is_siamese_one_call_and_inference_runs_only_trunk_and_ventral():
    enc = _enc()
    calls = {"trunk": [], "ventral": 0, "dorsal": 0}
    enc.trunk.register_forward_hook(lambda m, i, o: calls["trunk"].append(i[0].shape[0]))
    enc.ventral.register_forward_hook(lambda m, i, o: calls.__setitem__("ventral", calls["ventral"] + 1))
    enc.dorsal.register_forward_hook(lambda m, i, o: calls.__setitem__("dorsal", calls["dorsal"] + 1))
    p = _prepared(n=3)
    enc.encode_streams(p)
    assert calls == {"trunk": [6], "ventral": 1, "dorsal": 1}           # ONE trunk call on cat(x_{t-1}, x_t)
    calls.update(trunk=[], ventral=0, dorsal=0)
    obs = torch.randint(0, 256, (3, C * H * W)).float()
    enc(obs)
    assert calls == {"trunk": [3], "ventral": 1, "dorsal": 0}           # inference: phi_e + phi_v only
    # the policy features read x_t only: changing x_{t-1} changes nothing
    obs2 = obs.view(3, C, H, W).clone()
    obs2[:, :CPF] = torch.randint(0, 256, obs2[:, :CPF].shape).float()
    assert torch.equal(enc(obs), enc(obs2.view(3, -1)))
    # and the same trunk weights on both frames: x_{t-1} == x_t gives z_prev == z_t
    q = p.clone()
    q[:, :CPF] = q[:, CPF:]
    z = enc.trunk(torch.cat([q[:, :CPF], q[:, CPF:]]))
    assert torch.equal(z[:3], z[3:])


def test_policy_width_is_160_through_the_real_builder(monkeypatch):
    for hidden, first in (((), None), ((64,), 160)):
        agent = _agent(monkeypatch, hidden=hidden)
        assert agent.policy.encoder is agent.value.encoder
        assert agent.policy.encoder.features_dim == 160
        if first is None:
            assert agent.policy.mean_layer.in_features == 160 and agent.value.value_head.in_features == 160
        else:
            lin = [m for m in agent.policy.trunk.modules() if isinstance(m, nn.Linear)]
            assert lin[0].in_features == 160 and agent.policy.mean_layer.in_features == 64
            vlin = [m for m in agent.value.trunk.modules() if isinstance(m, nn.Linear)]
            assert vlin[0].in_features == 160


@pytest.mark.parametrize("kw, match", [
    ({"features_dim": 512}, "features_dim must be 160"),
    ({"features_dim": 640}, "features_dim must be 160"),
    ({"num_frames": 3}, "num_frames|stack"),
    ({"channels_per_frame": 1}, "RGB"),
    ({"conv_dim": 0}, "positive int"),
])
def test_encoder_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        _enc(**kw)


def test_encoder_refuses_wrong_channels_and_eye():
    with pytest.raises(ValueError, match="exactly 2 frames"):
        XSP2Encoder(_space(channels=9), features_dim=160)
    with pytest.raises(ValueError, match="divisible by 8"):
        XSP2Encoder(_space(h=84, w=128), features_dim=160)


def test_term_refusals(monkeypatch):
    torch.manual_seed(0)
    v1 = XSPEncoder(_space(), features_dim=32, conv_dim=16)
    with pytest.raises(TypeError, match="XSP2 encoder"):
        XSP2Term(v1)
    enc = _enc()
    for env, value, match in (("NETT_XSP2_BATCH", "3", ">= 4"),
                              ("NETT_XSP2_PREDICTOR", "layer", "NETT_XSP2_PREDICTOR"),
                              ("NETT_XSP2_LR_SCALE", "0", "NETT_XSP2_LR_SCALE"),
                              ("NETT_XSP2_HIDDEN", "64", "sizes the 'conv' predictor")):
        monkeypatch.setenv(env, value)
        with pytest.raises(ValueError, match=match):
            XSP2Term(enc)
        monkeypatch.delenv(env)
    monkeypatch.setenv("NETT_XSP2_BATCH", "8")
    monkeypatch.setenv("NETT_XSP2_PREDICTOR", "conv")
    monkeypatch.setenv("NETT_XSP2_REFINE_HIDDEN", "16")
    with pytest.raises(ValueError, match="sizes the 'warp' refiner"):
        XSP2Term(enc)
    monkeypatch.delenv("NETT_XSP2_REFINE_HIDDEN")
    monkeypatch.setenv("NETT_XSP2_HIDDEN", "16")
    term = XSP2Term(enc)
    assert isinstance(term.predictor, ConvPredictor) and term.predictor.net[0].out_channels == 16
    with pytest.raises(RuntimeError, match="no optimizer"):
        term.step(enc, _prepared(), _prepared(seed=6), torch.zeros(4, 2))


# ================================================================ 2. the warp

def test_warp_known_answer_integer_shift():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, H, W, generator=g)
    c = torch.zeros(2, 2, H, W)
    c[:, 0], c[:, 1] = 3.0, -2.0                      # out[y, x] = x_t[y + 2, x - 3]
    out = warp_bilinear(x, c)
    assert torch.equal(out[:, :, :H - 2, 3:], x[:, :, 2:, :W - 3])
    # border clamp: the left 3 columns repeat column 0, the bottom 2 rows repeat row H-1
    assert torch.equal(out[:, :, :H - 2, :3], x[:, :, 2:, :1].expand(-1, -1, -1, 3))
    assert torch.equal(out[:, :, H - 2:, 3:], x[:, :, H - 1:, :W - 3].expand(-1, -1, 2, -1))


def test_warp_is_bitwise_identity_at_zero_flow_and_bare_warp_at_m_zero():
    x = _prepared(n=2)[:, CPF:]
    assert torch.equal(warp_bilinear(x, torch.zeros(2, 2, H, W)), x)
    # m == 0 everywhere -> c == 0 -> the refiner's x_w is x_t bitwise
    ref = WarpRefiner(3, 8)
    pred, x_w = ref(x, torch.zeros(2, 2, H, W))
    assert torch.equal(x_w, x)
    assert torch.equal(pred, x + ref.refine(torch.cat([x, x, torch.zeros(2, 2, H, W)], 1)))


def _grid_sample_reference(x, c):
    b, _, h, w = x.shape
    ys = torch.arange(h, dtype=c.dtype).view(1, h, 1)
    xs = torch.arange(w, dtype=c.dtype).view(1, 1, w)
    gx = 2.0 * (xs - c[:, 0]) / (w - 1) - 1.0
    gy = 2.0 * (ys - c[:, 1]) / (h - 1) - 1.0
    return F.grid_sample(x, torch.stack([gx, gy], dim=-1), mode="bilinear", padding_mode="border",
                         align_corners=True)


def test_warp_matches_grid_sample_forward_and_gradient():
    g = torch.Generator().manual_seed(1)
    x = torch.rand(2, 3, H, W, generator=g, dtype=torch.float64)
    c = (torch.rand(2, 2, H, W, generator=g, dtype=torch.float64) - 0.5) * 9.0
    c1 = c.clone().requires_grad_(True)
    c2 = c.clone().requires_grad_(True)
    out1, out2 = warp_bilinear(x, c1), _grid_sample_reference(x, c2)
    assert torch.allclose(out1, out2, atol=1e-10)
    w_up = torch.rand(out1.shape, generator=g, dtype=torch.float64)
    (out1 * w_up).sum().backward()
    (out2 * w_up).sum().backward()
    # interior: the sample point lies strictly inside the image (no clamp)
    ys = torch.arange(H, dtype=c.dtype).view(1, H, 1)
    xs = torch.arange(W, dtype=c.dtype).view(1, 1, W)
    sx, sy = xs - c[:, 0], ys - c[:, 1]
    inside = (sx > 0.5) & (sx < W - 1.5) & (sy > 0.5) & (sy < H - 1.5)
    assert inside.float().mean() > 0.8
    for ch in (0, 1):
        assert torch.allclose(c1.grad[:, ch][inside], c2.grad[:, ch][inside], atol=1e-8)


def test_warp_edges_are_finite_with_a_live_gradient():
    x = _prepared(n=1)[:, CPF:].double()
    c = torch.full((1, 2, H, W), 0.4, dtype=torch.float64)
    c[:, :, :, :4] = 200.0                               # far outside on the left -> clamped
    c.requires_grad_(True)
    out = warp_bilinear(x, c)
    out.sum().backward()
    assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(c.grad).all())
    assert float(c.grad[:, :, 4:].abs().sum()) > 0
    assert torch.equal(c.grad[:, 0, :, :4], torch.zeros_like(c.grad[:, 0, :, :4]))  # clamp: no x-grad


# ================================================================ 3. loss and target

def test_loss_is_the_sum_of_squares_against_the_newest_frame():
    enc = _enc()
    term = _term(enc)
    pt, ptk = _prepared(seed=1), _prepared(seed=2)
    loss = term.score(enc, pt, ptk, torch.zeros(4, 2))
    with torch.no_grad():
        pred, _, _ = term.predict(enc, pt)
        manual = ((pred - ptk[:, CPF:]) ** 2).sum()
    assert torch.equal(loss.detach(), manual)
    n = 4 * CPF * H * W
    assert math.isclose(term.last_scalars["mse"], float(manual) / n, rel_tol=1e-6)
    assert math.isclose(term.last_scalars["copy_mse"], float(((pt[:, CPF:] - ptk[:, CPF:]) ** 2).mean()),
                        rel_tol=1e-6)
    # obs[t+1]'s OLDER frame (== x_t, the leak) is never read; its newest frame is
    ptk_old = ptk.clone()
    ptk_old[:, :CPF] = torch.rand_like(ptk_old[:, :CPF])
    assert torch.equal(term.score(enc, pt, ptk_old, torch.zeros(4, 2)).detach(), manual)
    ptk_new = ptk.clone()
    ptk_new[:, CPF:] = ptk_new[:, CPF:] * 0.5
    assert not torch.equal(term.score(enc, pt, ptk_new, torch.zeros(4, 2)).detach(), manual)


def test_compute_targets_obs_t_plus_1_from_memory():
    """compute() draws (obs_t, obs_{t+1}) windows with draw_action_window(memory, k=1)."""
    enc = _enc()
    term = _term(enc)
    mem = _NoiseMemory()
    term.attach_memory(mem)
    seen = {}
    real = term.step

    def spy(encoder, pt, ptk, actions, turn):
        seen.update(pt=pt, ptk=ptk)
        return real(encoder, pt, ptk, actions, turn)

    term.step = spy
    out = term.compute(enc, None)
    assert out.shape == () and float(out) == 0.0 and not out.requires_grad
    obs = mem.tensors["observations"].float().permute(0, 1, 4, 2, 3) / 255.0      # (T, env, C, H, W)
    pt, ptk = seen["pt"], seen["ptk"]
    assert pt.shape == ptk.shape == (8, C, H, W)
    for i in range(8):
        t, e = next((t, e) for t in range(obs.shape[0] - 1) for e in range(obs.shape[1])
                    if torch.equal(obs[t, e], pt[i]))
        assert torch.equal(ptk[i], obs[t + 1, e])
    assert set(TELEMETRY) <= set(term.last_scalars)
    assert term.last_scalars["B"] == 8.0


def test_static_frames_mark_skill_not_measured():
    enc = _enc()
    term = _term(enc)
    pt = _prepared(seed=1)
    term.step(enc, pt, pt, torch.zeros(4, 2))
    assert term.last_scalars["copy_mse"] == 0.0 and term.last_scalars["skill"] == NOT_MEASURED
    assert all(math.isfinite(v) for k, v in term.last_scalars.items() if isinstance(v, float))


# ================================================================ 4. reward detachment

def test_no_reward_gradient_reaches_the_encoder_and_v1_is_the_positive_control():
    torch.manual_seed(0)
    v1 = XSPEncoder(_space(), features_dim=32, conv_dim=16)
    enc = _enc()
    obs = torch.randint(0, 256, (4, C * H * W)).float()
    for model, expect_grad in ((v1, True), (enc, False)):
        head = nn.Linear(model.features_dim, 1)
        feats = model(obs)
        assert feats.requires_grad is expect_grad
        head(feats).sum().backward()
        trunk_grads = [p.grad for p in model.trunk.parameters()]
        if expect_grad:                         # the probe CAN see a reward gradient when there is one
            assert any(g is not None and float(g.abs().sum()) > 0 for g in trunk_grads)
        else:
            assert all(p.grad is None for p in model.parameters())
        assert head.weight.grad is not None


# ================================================================ 5. optimizer, one step per call, bit-identity

def test_streams_leave_ppo_optimizer_and_aux_adam_owns_them(monkeypatch):
    agent = _agent(monkeypatch)
    term, enc = agent._aux, agent.policy.encoder
    ppo_ids = {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
    stream_ids = {id(p) for p in enc.stream_parameters()}
    assert not ppo_ids & stream_ids
    mlp = [p for n, p in [*agent.policy.named_parameters(), *agent.value.named_parameters()]
           if not n.startswith("encoder.")]
    assert {id(p) for p in mlp} == ppo_ids
    aux_ids = [id(p) for g in term.optimizer.param_groups for p in g["params"]]
    assert set(aux_ids) == stream_ids | {id(p) for p in term.predictor.parameters()}
    assert len(aux_ids) == len(set(aux_ids))
    assert isinstance(term.optimizer, torch.optim.Adam) and list(term.head.parameters()) == []
    assert agent.checkpoint_modules["xsp2_optimizer"] is term.optimizer
    assert agent.checkpoint_modules["xsp2_predictor"] is term.predictor
    assert not ppo_ids & {id(p) for p in term.predictor.parameters()}


def test_exclude_from_optimizer_is_in_place():
    a, b = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(3))
    opt = torch.optim.Adam([a, b])
    a.grad, b.grad = torch.ones(2), torch.ones(3)
    opt.step()
    assert exclude_from_optimizer(opt, [a]) == 1
    assert opt.param_groups[0]["params"] == [b] and a not in opt.state and b in opt.state


def test_one_aux_adam_step_per_compute_and_per_minibatch(monkeypatch):
    agent = _agent(monkeypatch, epochs=2, mini_batches=2)
    term = agent._aux
    adam_steps = []
    term.optimizer.register_step_post_hook(lambda o, a, k: adam_steps.append(1))
    computes = []
    real = term.compute
    term.compute = lambda e, o: (computes.append(1), real(e, o))[1]
    ppo_grads = _record_ppo_grads(agent)
    _rollout(agent)
    assert len(computes) == len(adam_steps) == term.n_steps == 4 == len(ppo_grads)
    # at every PPO step the encoder holds NO gradient (the aux sets them back to None)
    for step in ppo_grads:
        assert all(v is None for k, v in step.items() if ".encoder." in k or k.startswith(("policy.encoder",
                                                                                            "value.encoder")))
    assert set(TELEMETRY) <= set(term.last_scalars) and term.last_scalars["B"] == 8.0


def test_aux_lr_follows_ppo_lr_times_scale(monkeypatch):
    monkeypatch.setenv("NETT_XSP2_LR_SCALE", "0.5")
    agent = _agent(monkeypatch, lr=3e-4)
    term = agent._aux
    assert term.optimizer.param_groups[0]["lr"] == pytest.approx(1.5e-4)
    agent.optimizer.param_groups[0]["lr"] = 2e-4                       # e.g. a scheduler moved it
    _rollout(agent)
    assert term.last_scalars["lr"] == pytest.approx(1e-4)
    assert term.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def _two_agents_bitwise(monkeypatch, other):
    a = _agent(monkeypatch, seed=0)
    b = other()
    # identical models before training (the predictor exists only in `a`)
    for name in ("policy", "value"):
        assert _states_equal(_state(getattr(a, name)), _state(getattr(b, name))), name
    ga, gb = _record_ppo_grads(a), _record_ppo_grads(b)
    enc_before = _state(a.policy.encoder)
    _rollout(a)
    _rollout(b)
    assert a._aux.n_steps == 1
    assert len(ga) == len(gb) == 1
    for k in ga[0]:
        va, vb = ga[0][k], gb[0][k]
        assert (va is None) == (vb is None), k
        if va is not None:
            assert torch.equal(va, vb), k
    assert all(ga[0][k] is None for k in ga[0] if "encoder." in k)
    mlp = lambda m: {k: v for k, v in _state(m).items() if not k.startswith("encoder.")}
    for name in ("policy", "value"):
        assert _states_equal(mlp(getattr(a, name)), mlp(getattr(b, name))), name
    assert not _states_equal(enc_before, _state(a.policy.encoder))          # the aux DID train phi
    return a, b


def test_ppo_is_bit_identical_to_the_aux_stubbed_agent(monkeypatch):
    def stubbed():
        b = _agent(monkeypatch, seed=0)
        b._aux.compute = lambda e, o: torch.zeros(())
        return b
    _, b = _two_agents_bitwise(monkeypatch, stubbed)
    assert b._aux.n_steps == 0


@pytest.mark.parametrize("cache", ["on", "off"])
def test_ppo_is_bit_identical_to_plain_metricsppo(monkeypatch, cache):
    if cache == "off":
        monkeypatch.setenv("NETT_DISABLE_FEATURE_CACHE", "1")

    def plain():
        return _agent(monkeypatch, aux=None, seed=0)
    a, b = _two_agents_bitwise(monkeypatch, plain)
    assert type(a).__name__ == "AuxLossPPO" and type(b).__name__ != "AuxLossPPO"


def test_bind_agent_refuses_unshared_encoder_and_inert_weight(monkeypatch):
    with pytest.raises(ValueError, match="shared_encoder=True"):
        _agent(monkeypatch, shared=False)
    with pytest.raises(ValueError, match="aux_weight=2"):
        _agent(monkeypatch, weight="2")


def test_checkpoint_round_trip(monkeypatch, tmp_path):
    a = _agent(monkeypatch, seed=0)
    _rollout(a)
    path = str(tmp_path / "agent.pt")
    a.save(path)
    saved = torch.load(path, weights_only=False)
    assert {"policy", "value", "optimizer", "xsp2_optimizer", "xsp2_predictor"} <= set(saved)
    b = _agent(monkeypatch, seed=7)
    assert not _states_equal(_state(a._aux.predictor), _state(b._aux.predictor))
    b.load(path)
    assert _states_equal(_state(a._aux.predictor), _state(b._aux.predictor))
    assert _states_equal(_state(a.policy.encoder), _state(b.policy.encoder))
    sa, sb = a._aux.optimizer.state_dict()["state"], b._aux.optimizer.state_dict()["state"]
    assert list(sa) == list(sb) and all(torch.equal(sa[i]["exp_avg"], sb[i]["exp_avg"]) for i in sa)
    # the restored Adam still drives the restored tensors (no stale parameter references)
    ids = {id(p) for g in b._aux.optimizer.param_groups for p in g["params"]}
    assert ids == {id(p) for p in b.policy.encoder.parameters()} | {id(p) for p in b._aux.predictor.parameters()}


# ================================================================ 6. determinism and init

def test_two_runs_are_bitwise_identical(monkeypatch):
    runs = []
    for _ in range(2):
        a = _agent(monkeypatch, seed=3, epochs=1, mini_batches=2)
        _rollout(a, seed=4)
        runs.append((_state(a.policy), _state(a.value), _state(a._aux.predictor), dict(a._aux.last_scalars)))
    for x, y in zip(runs[0][:3], runs[1][:3]):
        assert _states_equal(x, y)
    assert runs[0][3] == runs[1][3]


def test_aux_step_is_deterministic_under_strict_mode():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    out = []
    try:
        torch.use_deterministic_algorithms(True)
        for _ in range(2):
            enc = _enc(seed=2)
            term = _term(enc)
            term.attach_memory(_NoiseMemory())
            torch.manual_seed(5)
            term.compute(enc, None)
            term.compute(enc, None)
            out.append((_state(enc), _state(term.predictor), dict(term.last_scalars)))
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
    assert _states_equal(out[0][0], out[1][0]) and _states_equal(out[0][1], out[1][1])
    assert out[0][2] == out[1][2]


def _is_orthogonal(w, gain=math.sqrt(2.0)):
    """orthogonal_(w, gain): the Gram matrix of the flattened weight is gain^2 * I (on its smaller
    side). The scale is part of the check, so a one-row weight is not trivially "orthogonal"."""
    m = w.detach().double().reshape(w.shape[0], -1)
    g = m @ m.T if m.shape[0] <= m.shape[1] else m.T @ m
    return torch.allclose(g, gain ** 2 * torch.eye(g.shape[0], dtype=g.dtype), atol=1e-4)


def _default_init_modules(enc, pred):
    return [*(m for m in enc.ventral if isinstance(m, nn.Conv2d)),
            enc.dorsal.conv3d, enc.dorsal.conv2, enc.dorsal.conv3,
            *(m for m in pred.modules() if isinstance(m, nn.Conv2d))]


def _check_xsp2_init(enc, pred):
    for conv in (m for m in enc.trunk if isinstance(m, nn.Conv2d)):
        fan_in = conv.weight[0].numel()
        assert conv.weight.std().item() == pytest.approx(math.sqrt(2.0 / fan_in), rel=0.1), conv
        assert torch.equal(conv.bias, torch.zeros_like(conv.bias))
    for mod in _default_init_modules(enc, pred):
        bound = 1.0 / math.sqrt(mod.weight[0].numel())
        assert float(mod.weight.abs().max()) <= bound + 1e-7, mod
        assert float(mod.bias.abs().max()) <= bound + 1e-7, mod
        assert not _is_orthogonal(mod.weight), mod


def test_init_kaiming_trunk_default_streams_not_orthogonal():
    from nett_skrl.brain.models.utils.init import orthogonal_init
    enc = _enc(seed=0)
    for kind, env in (("warp", None), ("conv", "conv")):
        os.environ.pop("NETT_XSP2_PREDICTOR", None)
        if env:
            os.environ["NETT_XSP2_PREDICTOR"] = env
        try:
            _check_xsp2_init(enc, XSP2Term(enc).predictor)
        finally:
            os.environ.pop("NETT_XSP2_PREDICTOR", None)
    # positive control: the SAME modules under the fleet's orthogonal_init ARE orthogonal
    probe = _enc(seed=0)
    probe.apply(lambda m: orthogonal_init(m, math.sqrt(2)))
    for mod in [*(m for m in probe.trunk if isinstance(m, nn.Conv2d)), *_default_init_modules(probe, WarpRefiner(3, 32).apply(
            lambda m: orthogonal_init(m, math.sqrt(2))))]:
        assert _is_orthogonal(mod.weight), mod


def test_init_seeds(monkeypatch):
    assert _states_equal(_state(_enc(seed=4)), _state(_enc(seed=4)))
    assert not _states_equal(_state(_enc(seed=4)), _state(_enc(seed=5)))
    torch.manual_seed(1)
    p1 = _state(WarpRefiner(3, 32))
    torch.manual_seed(1)
    assert _states_equal(p1, _state(WarpRefiner(3, 32)))


def test_builder_skips_orthogonal_init_for_xsp2_only(monkeypatch):
    """Through the real builder: the XSP2 encoder keeps its own init on both the shared and the
    per-model paths; the MLP heads keep the fleet's orthogonal init; nature_cnn still gets it."""
    from nett_skrl.brain.models.builder import build_models_for_algorithm
    from nett_skrl.brain.models.model_cfg import ModelCfg
    from nett_skrl.brain.brain import Brain
    from nett_skrl.brain.registry import algorithm_spec
    spec = algorithm_spec(Brain(encoder="xsp2", algorithm="PPO").algorithm)
    act = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    for shared in (True, False):
        cfg = ModelCfg(shared_encoder=shared, hidden_sizes=[64])
        assert cfg.orthogonal_init
        torch.manual_seed(0)
        models = build_models_for_algorithm(spec, encoder_cls=XSP2Encoder, encoder_kwargs={"features_dim": 160},
                                            observation_space=_space(), action_space=act,
                                            device=torch.device("cpu"), cfg=cfg)
        for m in models.values():
            _check_xsp2_init(m.encoder, WarpRefiner(3, 32))
            assert _states_equal(_state(m.encoder), _state(_enc(seed=0))) or not shared
            lin = [x for x in m.trunk.modules() if isinstance(x, nn.Linear)]
            assert lin and all(_is_orthogonal(x.weight) for x in lin)
        torch.manual_seed(0)
        other = build_models_for_algorithm(spec, encoder_cls=encoder_mapping["nature_cnn"],
                                           encoder_kwargs={"features_dim": 64},
                                           observation_space=_space(channels=3), action_space=act,
                                           device=torch.device("cpu"), cfg=cfg)
        for m in other.values():
            convs = [x for x in m.encoder.modules() if isinstance(x, nn.Conv2d)]
            assert convs and all(_is_orthogonal(x.weight) for x in convs)


# ================================================================ 7. synthetic learning

SQ, SPEED = 16, 4


def _texture(g, b):
    t = torch.rand(b, 3, H // 8, W // 8, generator=g)
    return 0.15 + 0.45 * F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)


def _square_texture(g, b):
    t = torch.rand(b, 3, 4, 4, generator=g)
    return 0.6 + 0.4 * F.interpolate(t, size=(SQ, SQ), mode="bilinear", align_corners=False)


def _square_batch(g, b):
    """A textured 16px square moving 4px (one of 4 directions) over a textured static background:
    (x_{t-1}, x_t) stacks, (x_t, x_{t+1}) stacks, and the square's coverage of each 8x8 map cell at t."""
    bg, sq = _texture(g, b), _square_texture(g, b)
    frames = [bg.clone() for _ in range(3)]
    y0 = torch.randint(8, H - SQ - 8, (b,), generator=g)
    x0 = torch.randint(8, W - SQ - 8, (b,), generator=g)
    dirs = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]])[torch.randint(0, 4, (b,), generator=g)]
    mask = torch.zeros(b, H, W)
    for k in range(3):
        for i in range(b):
            y = int(y0[i] + (k - 1) * SPEED * dirs[i, 0])
            x = int(x0[i] + (k - 1) * SPEED * dirs[i, 1])
            frames[k][i, :, y:y + SQ, x:x + SQ] = sq[i]
            if k == 1:
                mask[i, y:y + SQ, x:x + SQ] = 1
    cell = F.avg_pool2d(mask[:, None], 8)[:, 0]
    return torch.cat(frames[:2], 1), torch.cat(frames[1:], 1), cell


def _evaluate(enc, term):
    pt, ptk, cell = _square_batch(torch.Generator().manual_seed(999), 64)
    with torch.no_grad():
        term.score(enc, pt, ptk, torch.zeros(64, 2))
        m = enc.encode_streams(pt)["m"][:, 0]
    s = term.last_scalars
    return {"skill": s["skill"], "warp_skill": 1.0 - s["warp_mse"] / s["copy_mse"],
            "sep": float(m[cell >= 0.5].mean() - m[cell == 0].mean()), "fg_motion_sep": s["fg_motion_sep"],
            "flow_absmean": s["flow_absmean"]}


def test_synthetic_moving_square_is_learned_and_orientation_is_reported():
    """400 aux steps (batch 16, lr 1e-3) per seed, 3 seeds, under 2 minutes at 8 threads. The
    predictor must beat the copy baseline; the map's orientation (square-high or square-low) is
    REPORTED per seed, not asserted -- only that the map separates the square at all."""
    torch.set_num_threads(8)
    rows = []
    for seed in (0, 1, 2):
        enc = _enc(seed=seed)
        term = _term(enc, lr=1e-3)
        g = torch.Generator().manual_seed(100 + seed)
        init = _evaluate(enc, term)
        for _ in range(400):
            pt, ptk, _ = _square_batch(g, 16)
            term.step(enc, pt, ptk, torch.zeros(16, 2))
        final = _evaluate(enc, term)
        rows.append((seed, init, final))
        side = "square HIGH in m" if final["sep"] > 0 else "square LOW in m"
        print(f"[xsp2 synthetic] seed {seed}: skill {init['skill']:+.3f} -> {final['skill']:+.3f} "
              f"(bare warp {final['warp_skill']:+.3f}); m(square) - m(background) {final['sep']:+.3f} "
              f"({side}); fg_motion_sep {final['fg_motion_sep']:+.3f}; flow |c| {final['flow_absmean']:.3f}")
    for seed, init, final in rows:
        assert final["skill"] > 0.5 and final["skill"] > init["skill"], (seed, init, final)
        assert abs(final["sep"]) > 0.1, (seed, final)


# ================================================================ 8. knobs, env index, invariance

def test_stray_xsp2_knob_on_another_label_refuses(campaign, monkeypatch):
    monkeypatch.delenv("NETT_XSP2_BATCH")
    for label, spec in campaign.MODELS.items():
        assert campaign.refuse_stray_xsp2_knobs(label, spec) is None
    monkeypatch.setenv("NETT_XSP2_PREDICTOR", "conv")
    with pytest.raises(ValueError, match="not an XSP2 arm"):
        campaign.refuse_stray_xsp2_knobs("CNN", campaign.MODELS["CNN"])
    with pytest.raises(ValueError, match="not an XSP2 arm"):
        campaign.refuse_stray_xsp2_knobs("XSP", campaign.MODELS["XSP"])
    assert campaign.refuse_stray_xsp2_knobs("XSP2", campaign.MODELS["XSP2"]) is None
    assert campaign.refuse_stray_xsp_knobs("XSP2", campaign.MODELS["XSP2"]) is None


def test_knobs_are_literal_reads_and_in_the_env_index():
    text = (_SRC / "nett_skrl" / "brain" / "aux" / "xsp2_aux.py").read_text()
    index = (_SRC / "docs" / "env_vars.md").read_text()
    for k in XSP2_KNOBS:
        assert f'"{k}"' in text and f"`{k}`" in index, k
    assert '"NETT_XSP2_UPSAMPLE"' not in text and '"NETT_XSP2_POOL"' not in text      # never read (spec change 2)


def test_unset_knobs_equal_explicit_defaults(monkeypatch):
    def run():
        enc = _enc(seed=1)
        term = _term(enc)
        torch.manual_seed(2)
        term.attach_memory(_NoiseMemory())
        term.compute(enc, None)
        return _state(enc), _state(term.predictor), dict(term.last_scalars)
    monkeypatch.delenv("NETT_XSP2_BATCH")
    a = run()
    for k, v in (("NETT_XSP2_BATCH", "64"), ("NETT_XSP2_PREDICTOR", "warp"), ("NETT_XSP2_REFINE_HIDDEN", "32"),
                 ("NETT_XSP2_LR_SCALE", "1.0")):
        monkeypatch.setenv(k, v)
    b = run()
    assert _states_equal(a[0], b[0]) and _states_equal(a[1], b[1]) and a[2] == b[2]
    assert a[2]["B"] == 46.0          # 32 + 32 asked; the 24-step memory caps each slab at 23 anchors


NEW_FILES = ("nett_skrl/brain/encoders/xsp2.py", "nett_skrl/brain/aux/xsp2_aux.py")
ADDITIVE = ("nett_skrl/brain/aux/ppo_aux.py", "nett_skrl/brain/registry.py",
            "nett_skrl/brain/encoders/__init__.py", "examples/campaign_train.py")
#: The only existing lines XSP2 edits: the init-ownership guard (and its import) in the two places
#: the fleet's orthogonal init is applied to an encoder.
EDITED = {
    "nett_skrl/brain/models/builder.py": (
        ["from .utils.init import orthogonal_init", "if cfg.orthogonal_init:"],
        ["from .utils.init import encoder_owns_init, orthogonal_init",
         "if cfg.orthogonal_init and not encoder_owns_init(shared_enc):"]),
    "nett_skrl/brain/models/utils/backbone.py": (
        ["from .init import orthogonal_init", "if cfg.orthogonal_init:"],
        ["from .init import encoder_owns_init, orthogonal_init",
         "if cfg.orthogonal_init and not encoder_owns_init(self.encoder):"]),
}
UNCHANGED = ("nett_skrl/brain/encoders/xsp.py", "nett_skrl/brain/aux/xsp_aux.py")
# The small-3DCNN ladder (DECISIONS §89, test_3dcnn_small_ladder.py): cfg-only 3DCNN labels and an
# even-stem-kernel path in compact_3dcnn.py, held to the old counts and the plain odd-kernel Conv3d there.
SMALL3D_FILES = ("nett_skrl/brain/encoders/compact_3dcnn.py",)
# The train-abort gate (DECISIONS §90.3, test_train_abort_gate.py): TEST refuses a brain with no
# checkpoint, and a crashed train subprocess fails the arm unless every brain saved final_agent.pt.
# It touches no encoder, aux or label file.
ABORT_GATE_FILES = ("nett_skrl/recording/__init__.py", "nett_skrl/recording/checkpoints.py",
                    "nett_skrl/runtime/task_runner.py")
SMALL3D_LABELS = {"3DCNN-W12", "3DCNN-N77", "3DCNN-N12", "3DCNN-K16N77", "3DCNN-K16N26", "3DCNN-K16N12"}
# The owner's matched baseline (DECISIONS §92, test_matched_encoders.py): fourteen "M-" labels, encoder "matched",
# one new encoder file; the registry, encoders/__init__ and campaign_train edits are already in ADDITIVE/EDITED.
MATCHED_FILES = ("nett_skrl/brain/encoders/matched.py",)
MATCHED_LABELS = {"M-CNN", "M-ViT", "M-CNN2F", "M-ViT2F", "M-3DCNN", "M-ViViT", "M-CNN-CLTT-Ref", "M-ViT-CLTT-Ref",
                  "M-CNN2F-CLTT-Ref", "M-ViT2F-CLTT-Ref", "M-3DCNN-CLTT-Ref", "M-ViViT-CLTT-Ref",
                  "M-3DCNN-G6", "M-3DCNN-G6-CLTT-Ref"}
# The uint8 CPU observation buffer (DECISIONS §103; test_xsp.py holds the same exemption): two buffer files,
# and one admitted removal in campaign_train -- the NETT_UINT8_BUFFER pop on the CPU move -- by exact text, once.
UINT8_FILES = ("nett_skrl/brain/agent_factory.py", "nett_skrl/brain/hybrid_memory.py")
UINT8_REMOVED = {"examples/campaign_train.py": ('os.environ.pop("NETT_UINT8_BUFFER", None)',)}
# Workspace DECISIONS §108 (test_xsp2_windows_and_train_dp.py): the train-only decision period adds an
# Environment kwarg + its schema entry; NETT_XSP2_WINDOWS lives in xsp2_aux.py (already admitted).
DP108_FILES = ("nett_skrl/environment/environment.py", "nett_skrl/schema.json")
# Workspace DECISIONS §110 (test_xsp2_step_before_wrapper.py): NETT_XSP2_WINDOWS=hold reads the env's own
# step counter through a TRAINING-env wrapper (env_wrappers.py), applied in trainer._run_skrl_train.
DP110_FILES = ("nett_skrl/brain/env_wrappers.py", "nett_skrl/brain/trainer.py")
# Workspace DECISIONS §111 (test_cltt_step_fixes.py): the CLTT-Ref encode dedup (cltt_ref_aux.py), the RL
# encoder pass under no_grad when decoupled (features.py), and the aux forward moved after the PPO backward
# in ppo_aux.py. That move is the one §111 EDIT to existing lines -- the aux compute now sits under
# `if not aux_after:` and the split-backward `if (` became `elif (` -- admitted below by exact text, once each.
DP111_FILES = ("nett_skrl/brain/aux/cltt_ref_aux.py", "nett_skrl/brain/models/utils/features.py",
               "nett_skrl/brain/encoders/nature_cnn.py")   # + §113's pre-Ampere bf16 warning (additive)
DP111_REMOVED = {"nett_skrl/brain/aux/ppo_aux.py": ("aux_loss = self._aux_weight * self._aux.compute(",
                                                     "self._aux_encoder, sampled_observations", ")", "if (")}
# Workspace DECISIONS §114 (test_decision_loop.py): the agent-side decision loop wraps the env in the one place
# train/test/record get it (brain._wrapped_env); that test pins brain.py's diff to exactly that hook.
DP114_FILES = ("nett_skrl/brain/brain.py",)
#: Unity-recipe++ eyes (owner 2026-10-15, workspace DECISIONS §117): a new fovea wrapper, its registry entry,
#: and two labels (CNN-UnityRecipe+Fovea, CNN-UnityRecipe+DVS) that add one `pre` wrapper each.
UPP117_FILES = ("nett_skrl/body/wrappers/fovea.py", "nett_skrl/body/wrappers/registry.py",
                "nett_skrl/brain/hybrid_memory.py", "nett_skrl/brain/ppo_metrics.py")
UPP117_LABELS = {"CNN-UnityRecipe+Fovea", "CNN-UnityRecipe+DVS", "XSP2-MAP35x56"}
#: I-JEPA / V-JEPA (owner 2026-10-15, test_jepa.py): a new encoder and aux module, two labels; the shared
#: registries gain additive entries only ("jepa"; "ijepa", "vjepa").
JEPA_FILES = ("nett_skrl/brain/encoders/jepa.py", "nett_skrl/brain/aux/jepa_aux.py")
JEPA_LABELS = {"I-JEPA", "V-JEPA", "I-JEPA-13M", "V-JEPA-13M"}
#: fp16 for compact_vit (owner 2026-10-15, test_vit_fp16.py): the agent-level GradScaler hook (amp.py, called from
#: agent_factory) and CompactViT's fp16 autocast branch; nature_cnn/matched only gain the knob's fp16 plumbing.
FP16_FILES = ("nett_skrl/brain/amp.py", "nett_skrl/brain/encoders/compact_vit.py")
#: strict_determinism restores BOTH determinism flags on exit (test_reproducibility.py).
DETERMINISM_FILES = ("nett_skrl/brain/skrl_patches.py",)
#: NETT_TEST_CHECKPOINT: TEST loads a named mid-training checkpoint (test_test_checkpoint_override.py).
CHECKPOINT_FILES = ("nett_skrl/recording/checkpoints.py",)
#: owner 2026-10-16 (workspace DECISIONS §131): NETT_DVS_COLOR, the 6-channel event frame (opt-in; unset is bit-identical).
DVS6_FILES = ("nett_skrl/body/wrappers/dvs_polarity.py",)
#: owner 2026-10-17 (workspace DECISIONS §134): the colour-DVS channel guard (test_dvs_blur_kernel.py); the
#: NETT_DVS_BLUR_KERNEL knob is in dvs_polarity.py, already admitted above.
DVS134_FILES = ("nett_skrl/brain/aux/cltt_views.py",)
#: ...and 3DCNN-DVS on the 6-channel colour event frame (test_dvs_blur_kernel.py).
DVS134_LABELS = {"3DCNN-DVS6"}
#: owner 2026-10-17 (workspace DECISIONS §135): NatureCNN extra_convs + CNN2F+CLTT-Ref-X1/X2/X3 (test_nature_cnn_extra_convs.py).
XCONV135_FILES = ("nett_skrl/brain/encoders/nature_cnn.py",)
XCONV135_LABELS = {"CNN2F+CLTT-Ref-X1", "CNN2F+CLTT-Ref-X2", "CNN2F+CLTT-Ref-X3"}
XMLP138_LABELS = {"CNN2F+CLTT-Ref-M1", "CNN2F+CLTT-Ref-M2", "CNN2F+CLTT-Ref-M3", "CNN2F+CLTT-Ref-M4",
                  "CNN2F+CLTT-Ref-M5", "CNN2F+CLTT-Ref-M6", "CNN2F+CLTT-Ref-M7", "CNN2F+CLTT-Ref-M8"}  # M5..M8: §142
#: owner 2026-10-18 (workspace DECISIONS §151): XSP2-MAP35x56 at map stride 4 / 2 (campaign_train.py asserts nothing else differs).
XMAP151_LABELS = {"XSP2-MAP70x112", "XSP2-MAP140x224"}
#: owner 2026-10-16 (workspace DECISIONS §133): patch-28 twins of the patch-14 ViT-CLTT-Ref labels (test_vit_fp16.py).
P28_LABELS = {"ViT-CLTT-Ref-500K-P28", "ViT-CLTT-Ref-5M-P28", "ViT-CLTT-Ref-50M-P28", "ViT-CLTT-Ref-Base-P28",
              "ViT-CLTT-Ref-500K-P28-DVS", "ViT-CLTT-Ref-5M-P28-DVS"}
#: ...and the paper's ViT-Base trunk as a ViT-CLTT-Ref label (owner 2026-10-15, test_vit_fp16.py).
FP16_LABELS = {"ViT-CLTT-Ref-Base", "ViT-CLTT-Ref-500K-P14", "ViT-CLTT-Ref-5M", "ViT-CLTT-Ref-50M",
               "ViT-CLTT-Ref-500K-P14-DVS", "ViT-CLTT-Ref-5M-DVS"}

#: owner 2026-10-19 (workspace DECISIONS §162): ViViT channels_per_frame (test_dvs6_cltt_labels.py), the Slot Attention
#: `sample` override and ViT-SlotRecon's encoder + aux (test_vit_slot_recon.py); 3DCNN-CLTT-Ref-T1's temporal_kernel_depth
#: is in compact_3dcnn.py, already admitted above (test_3dcnn_t1.py holds every existing 3DCNN label bitwise).
S162_FILES = ("nett_skrl/brain/encoders/compact_vivit.py", "nett_skrl/brain/aux/slot_attention_ae.py",
              "nett_skrl/brain/encoders/vit_slot_recon.py", "nett_skrl/brain/aux/slot_recon_aux.py")
S162_LABELS = {"3DCNN-CLTT-Ref-T1", "ViViT-CLTT-Ref-T2", "3DCNN-CLTT-Ref-DVS6", "3DCNN-CLTT-Ref-T1-DVS6",
               "ViViT-CLTT-Ref-DVS6", "ViViT-CLTT-Ref-T2-DVS6", "ViT-SlotRecon",
               # §162 add. 3: the ~500K rungs (test_s162_500k.py)
               *(f"{a}-500K-CLTT-Ref{b}" for a in ("3DCNN", "ViViT")
                 for b in ("", "-T1" if a == "3DCNN" else "-T2", "-DVS6", "-T1-DVS6" if a == "3DCNN" else "-T2-DVS6"))}

#: owner 2026-10-20 (workspace DECISIONS §170): STP-E's pose wrapper, encoder and aux (test_stp_e.py); the shared
#: registries gain "stp_pose_exec" / "stp_pose_cmd", "stp_e" and "stp" only.
STP170_FILES = ("nett_skrl/body/wrappers/stp_pose.py", "nett_skrl/brain/encoders/stp_e.py",
                "nett_skrl/brain/aux/stp_aux.py")
STP170_LABELS = {"STP-E", "STP-E-NoWarp", "STP-E-Cmd", "CNN2F-630K"}

def _git(*args):
    env = {k: v for k, v in os.environ.items()
           if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_COMMON_DIR")}
    try:
        return subprocess.run(["git", *args], cwd=_SRC, capture_output=True, text=True,
                              check=True, env=env).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git unavailable here: {exc}")


def _base():
    """The commit before XSP2: the parent of the commit that added encoders/xsp2.py."""
    added = _git("log", "--diff-filter=A", "--format=%H", "--", NEW_FILES[0]).split()
    if not added:
        pytest.fail(f"no commit adds {NEW_FILES[0]} -- the diff cannot look")
    return f"{added[-1]}^"


def _diff_lines(path):
    lines = _git("diff", _base(), "--unified=0", "--", path).splitlines()
    removed = [ln[1:].strip() for ln in lines if ln.startswith("-") and not ln.startswith("---")]
    added = [ln[1:].strip() for ln in lines if ln.startswith("+") and not ln.startswith("+++")]
    return removed, added


@pytest.mark.parametrize("path", ADDITIVE)
def test_shared_file_change_is_additive_only(path):
    removed, added = _diff_lines(path)
    assert added, f"{path}: no XSP2 addition found -- the diff did not look"
    for rule, admitted in (("§103", UINT8_REMOVED), ("§111", DP111_REMOVED)):
        for text in admitted.get(path, ()):
            assert text in removed, f"{path}: the admitted {rule} removal {text!r} is not in the diff -- stale exemption"
            removed.remove(text)
    assert not removed, f"{path}: XSP2 removed or edited existing lines:\n" + "\n".join(removed)


@pytest.mark.parametrize("path", sorted(EDITED))
def test_init_guard_is_the_only_edit(path):
    removed, added = _diff_lines(path)
    assert (removed, added) == tuple(EDITED[path]), (removed, added)


def test_init_helper_is_additive_and_xsp_files_untouched():
    removed, added = _diff_lines("nett_skrl/brain/models/utils/init.py")
    assert added and not removed
    assert _git("diff", _base(), "--", *UNCHANGED) == ""
    changed = set(_git("diff", _base(), "--name-only", "--", "nett_skrl", "examples").split())
    allowed = {f"src_isaac/{p}" for p in (*NEW_FILES, *ADDITIVE, *EDITED, *SMALL3D_FILES, *ABORT_GATE_FILES, *MATCHED_FILES,
                                          *UINT8_FILES, *DP108_FILES, *DP110_FILES, *DP111_FILES, *DP114_FILES, *UPP117_FILES, *JEPA_FILES, *FP16_FILES, *DETERMINISM_FILES, *CHECKPOINT_FILES, *DVS6_FILES, *DVS134_FILES, *XCONV135_FILES, *S162_FILES, *STP170_FILES,
                                          "nett_skrl/brain/models/utils/init.py")}
    assert changed - allowed == set(), sorted(changed - allowed)


def test_existing_labels_resolve_to_the_identical_spec(campaign):
    src = _git("show", f"{_base()}:src_isaac/examples/campaign_train.py")
    spec = importlib.util.spec_from_loader("_campaign_train_pre_xsp2", loader=None)
    old = importlib.util.module_from_spec(spec)
    old.__dict__["__file__"] = str(_TRAIN)
    exec(compile(src, "campaign_train@pre-xsp2", "exec"), old.__dict__)
    assert set(campaign.MODELS) - set(old.MODELS) == {"XSP2", "XSP2-MAP20x32", "XSP2-WHAT", "XSP2-WHAT-S4", "XSP2-WHAT-S2", "XSP2-MAP40x64", "XSP2-MAP20x32-F3", "XSP2-MAP40x64-F3"} | SMALL3D_LABELS | MATCHED_LABELS | UPP117_LABELS | JEPA_LABELS | FP16_LABELS | P28_LABELS | DVS134_LABELS | XCONV135_LABELS | XMLP138_LABELS | XMAP151_LABELS | S162_LABELS | STP170_LABELS
    for label, spec_old in old.MODELS.items():
        assert campaign.MODELS[label] == spec_old, label
    assert campaign.EXPERIMENTS == old.EXPERIMENTS


def _literal_keys(src, opener):
    return {ln.split('"')[1] for ln in src.split(opener, 1)[1].split("\n}", 1)[0].splitlines()
            if ln.strip().startswith('"')}


def test_registries_only_gained_the_xsp2_entries():
    for rel, opener, live in (("nett_skrl/brain/aux/ppo_aux.py", "AUX_LOSSES = {", AUX_LOSSES),
                              ("nett_skrl/brain/registry.py", "encoder_mapping: dict", encoder_mapping)):
        old = _literal_keys(_git("show", f"{_base()}:src_isaac/{rel}"), opener)
        new = _literal_keys((_SRC / rel).read_text(), opener)
        assert old and new - old == ({"xsp2", "matched", "jepa", "vit_slot_recon", "stp_e"} if "registry" in rel
                                  else {"xsp2", "ijepa", "vjepa", "slot_recon", "stp"}) and old <= new, rel
        assert new <= set(live), rel
