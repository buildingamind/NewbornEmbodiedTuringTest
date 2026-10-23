"""NETT_MIRROR_EQUIV: the mirror-symmetrised policy mean and value, per model family (workspace DECISIONS §187;
the patch is the presenter's §184 one).

    mu_sym(o) = 0.5 (mu(o) + M mu(F o)),  M = diag(-1, +1);    V_sym(o) = 0.5 (V(o) + V(F o))

F flips the image WIDTH axis and nothing else. Every model here is built from campaign_train MODELS, through the
builder the driver uses, on the observation space of that label's REAL body chain (Body(segmentation_wrappers(spec))
over a batched HWC fake env, so the channel count and order are the driver's: DVS polarity / colour, then the
framestack's time-major stack, then ChannelsFirst), at the 80x128 eye. Checked per label:
  * the chain commutes with F: chain(flip_W(frames)) == mirror_obs(chain(frames)), bit for bit -- so a width flip of
    the policy input is the observation of the mirrored world, with polarity, colour and frame order untouched;
  * knob on: mu_turn(F o) == -mu_turn(o), mu_move(F o) == mu_move(o), V(F o) == V(o) to 1e-5 (relative to the output
    scale), on real chain observations AND dense random ones; equal to 0.5 (f(o) + M f(F o)) from the flag-off
    model; and every parameter that gets a gradient with the knob off gets one with it on;
  * POSITIVE CONTROL: knob off, the same weights and inputs break all three equalities by orders of magnitude (the
    heads are rescaled to O(1) outputs first, so "small" cannot pass for "equal").
The aux losses (CLTT-Ref, VICReg, VICReg-TT) call `model.encoder` directly on UNFLIPPED observations -- the mirror
is a property of the RL heads only -- and their first loss is bit-identical with the knob on and off.
Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "examples"))

import campaign_train as ct  # noqa: E402
from nett_skrl.body.body import Body  # noqa: E402
from nett_skrl.brain.brain import Brain  # noqa: E402
from nett_skrl.brain.config import EncoderCfg  # noqa: E402
from nett_skrl.brain.models import builder as builder_mod  # noqa: E402
from nett_skrl.brain.models.model_cfg import ModelCfg  # noqa: E402
from nett_skrl.brain.models.utils.features import (  # noqa: E402
    _CACHE_ATTR, _MIRROR_CACHE_ATTR, MIRROR_CKPT_ENV, MIRROR_ENV, features_forward_mirrored, mirror_equiv_enabled,
    mirror_obs, shared_feature_cache)
from nett_skrl.brain.registry import algorithm_spec, encoder_mapping  # noqa: E402

H, W, N = 80, 128, 3
#: (label, NETT_DVS_COLOR). CNN-UnityRecipe+DVS runs in both DVS modes (the presenter's rows used colour).
LABELS = [
    ("CNN-UnityRecipe", False),
    ("CNN-UnityRecipe+DVS", False),
    ("CNN-UnityRecipe+DVS", True),
    ("CNN2F", False),
    ("CNN2F-UnityRecipe", False),
    ("3DCNN", False),
    ("3DCNN-DVS6", True),
    ("3DCNN-CLTT-Ref-DVS6", True),
    ("ViT-CLTT-Ref", False),
    ("ViViT-CLTT-Ref", False),
    ("ViViT-CLTT-Ref-DVS6", True),
]
IDS = [f"{lbl}{'-colour' if c else ''}" for lbl, c in LABELS]
KNOBS = (MIRROR_ENV, MIRROR_CKPT_ENV, "NETT_DECOUPLE_ENCODER", "NETT_DISABLE_FEATURE_CACHE", "NETT_DVS_COLOR",
         "NETT_DVS_THRESHOLD", "NETT_DVS_BLUR", "NETT_DVS_BLUR_KERNEL", "NETT_DVS_WINDOW", "NETT_DVS_WINDOW_TAU",
         "NETT_FRAMESTACK_N", "NETT_FRAMESTACK_SHUFFLE", "NETT_INIT_SCHEME", "NETT_CONV3D_ORTHO_INIT",
         "NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_AUX_LOSS", "NETT_AUX_WEIGHT", "NETT_AUX_BATCH",
         "NETT_AUX_AFTER_PPO_BACKWARD", "NETT_AUX_STRICT")
M = torch.tensor([-1.0, 1.0])


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            yield
    finally:
        torch.set_num_threads(threads)


class _FakeEnv(gym.Env):
    """Batched (N, H, W, 3) uint8 frames in a {"policy": ...} dict, the Isaac layout the body chain consumes."""

    def __init__(self, frames):
        self.frames, self.i = list(frames), 0
        self.observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, (N, H, W, 3), np.uint8)})
        self.action_space = gym.spaces.Box(-1.0, 1.0, (N, 2), np.float32)

    def _emit(self):
        f = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return {"policy": torch.as_tensor(f)}

    def reset(self, **kw):
        self.i = 0
        return self._emit(), {}

    def step(self, action):
        z = np.zeros(N, dtype=bool)
        return self._emit(), np.zeros(N), z, z.copy(), {}


def _frames(seed, n=4):
    """Smooth-ish random frames (a random field plus a moving bar), so DVS events are dense enough to test."""
    g = np.random.default_rng(seed)
    out = []
    for k in range(n):
        f = g.integers(0, 256, (N, H, W, 3), dtype=np.uint8)
        f[:, :, 10 + 7 * k: 30 + 7 * k] = 255
        out.append(f)
    return out


def _chain_obs(label, frames, steps=2):
    """The label's body chain over ``frames``: the flat (N, D) policy observation after reset, then each step."""
    env = Body(ct.segmentation_wrappers(ct.MODELS[label])).wrap(_FakeEnv(frames))
    obs = [env.reset()[0]["policy"]]
    for _ in range(steps):
        obs.append(env.step(None)[0]["policy"])
    flat = [torch.as_tensor(np.asarray(o)).reshape(N, -1) for o in obs]
    return env.observation_space["policy"].shape[1:], flat


def _setup(monkeypatch, colour):
    if colour:
        monkeypatch.setenv("NETT_DVS_COLOR", "1")


def _space(label):
    shape, _ = _chain_obs(label, _frames(0, 1), steps=0)
    return gym.spaces.Box(low=0, high=255, shape=tuple(int(s) for s in shape), dtype=np.uint8)


def _build(label, space, seed=0, hidden=()):
    spec = ct.MODELS[label]
    kw = EncoderCfg(**spec["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    torch.manual_seed(seed)
    models = builder_mod.build_models_for_algorithm(
        algorithm_spec(Brain(encoder=spec["encoder"], algorithm="PPO").algorithm),
        encoder_cls=encoder_mapping[spec["encoder"]], encoder_kwargs=kw, observation_space=space,
        action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32), device=torch.device("cpu"),
        cfg=ModelCfg(hidden_sizes=list(hidden), shared_encoder=True, clip_actions=False, value_bound=None,
                     max_log_std=0.0))
    # O(1) outputs: the heads' small-gain init would make every difference tiny, and "small" must not pass for "equal".
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for lin in (models["policy"].mean_layer, models["value"].value_head):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) / math.sqrt(lin.weight.shape[1]))
            lin.bias.copy_(torch.randn(lin.bias.shape, generator=g))
    for m in models.values():
        m.eval()
    return models


def _run(models, x):
    mu, _ = models["policy"].compute({"observations": x})
    v, _ = models["value"].compute({"observations": x})
    return mu, v


def _flat_flip(x, space):
    return mirror_obs(x, space).reshape(x.shape[0], -1)


def _inputs(label, space):
    """Real chain observations (the step-2 obs, events on every DVS label) and dense random ones."""
    _, obs = _chain_obs(label, _frames(3))
    g = torch.Generator().manual_seed(9)
    dense = torch.randint(0, 256, (N, int(np.prod(space.shape))), dtype=torch.uint8, generator=g)
    return {"chain": obs[-1], "dense": dense}


def _errs(models, x, space):
    with torch.no_grad():
        mu, v = _run(models, x)
        mu_f, v_f = _run(models, _flat_flip(x, space))
    scale = max(1.0, float(mu.abs().max()), float(v.abs().max()))
    return (float((mu_f[:, 0] + mu[:, 0]).abs().max()) / scale,
            float((mu_f[:, 1] - mu[:, 1]).abs().max()) / scale,
            float((v_f - v).abs().max()) / scale)


# ---------------------------------------------------------------- the flip is the world's mirror

@pytest.mark.parametrize("label,colour", LABELS, ids=IDS)
def test_the_real_chain_commutes_with_the_width_flip(monkeypatch, label, colour):
    _setup(monkeypatch, colour)
    space = _space(label)
    frames = _frames(1)
    _, obs = _chain_obs(label, frames, steps=3)
    _, obs_m = _chain_obs(label, [np.ascontiguousarray(f[:, :, ::-1]) for f in frames], steps=3)   # W of (N,H,W,3)
    for t, (o, om) in enumerate(zip(obs, obs_m)):
        assert torch.equal(_flat_flip(o, space), om), f"{label} step {t}"
        # CHW at the policy: W is the last axis, and only it moves.
        assert torch.equal(mirror_obs(o, space), o.view(N, *space.shape).flip(-1))
    if ct.MODELS[label].get("pre"):
        assert int((obs[-1] > 0).sum()) > 0                        # events: the check is not on blank frames
    # control: a flip of any other axis is NOT the mirrored world's observation
    assert not torch.equal(o.view(N, *space.shape).flip(-2).reshape(N, -1), om)


# ---------------------------------------------------------------- equivariance, per family

@pytest.mark.parametrize("label,colour", LABELS, ids=IDS)
def test_policy_mean_and_value_are_mirror_equivariant_with_the_knob_on(monkeypatch, label, colour, capsys):
    _setup(monkeypatch, colour)
    space = _space(label)
    monkeypatch.setenv(MIRROR_ENV, "1")
    models = _build(label, space)
    assert "[NETT mirror] NETT_MIRROR_EQUIV=1:" in capsys.readouterr().out
    for name, x in _inputs(label, space).items():
        e_turn, e_move, e_v = _errs(models, x, space)
        assert e_turn <= 1e-5 and e_move <= 1e-5 and e_v <= 1e-5, (label, name, e_turn, e_move, e_v)


@pytest.mark.parametrize("label,colour", LABELS, ids=IDS)
def test_positive_control_the_knob_off_breaks_every_equality(monkeypatch, label, colour):
    _setup(monkeypatch, colour)
    space = _space(label)
    models = _build(label, space)
    for name, x in _inputs(label, space).items():
        e_turn, e_move, e_v = _errs(models, x, space)
        assert min(e_turn, e_move, e_v) > 1e-3, (label, name, e_turn, e_move, e_v)


@pytest.mark.parametrize("label,colour", LABELS, ids=IDS)
def test_knob_on_equals_the_symmetrised_flag_off_model_and_gradients_flow(monkeypatch, label, colour):
    _setup(monkeypatch, colour)
    space = _space(label)
    models = _build(label, space)
    x = _inputs(label, space)["chain"]
    with torch.no_grad():
        mu, v = _run(models, x)
        mu_f, v_f = _run(models, _flat_flip(x, space))
    params = [*models["policy"].named_parameters(), *(("value." + k, p) for k, p in models["value"].named_parameters())]

    def grads():
        for _, p in params:
            p.grad = None
        a, b = _run(models, x)
        ((a ** 2).sum() + (b ** 2).sum()).backward()
        return {k: p.grad for k, p in params if p.grad is not None and bool(p.grad.abs().sum() > 0)}

    off = grads()
    monkeypatch.setenv(MIRROR_ENV, "1")
    with torch.no_grad():
        mu_on, v_on = _run(models, x)
    torch.testing.assert_close(mu_on, 0.5 * (mu + mu_f * M), rtol=0, atol=1e-5)
    torch.testing.assert_close(v_on, 0.5 * (v + v_f), rtol=0, atol=1e-5)
    on = grads()
    assert set(on) == set(off), sorted(set(off) ^ set(on))
    enc = [k for k in on if ".encoder." in f".{k}" or k.startswith("encoder.")]
    assert enc and "mean_layer.weight" in on and "value.value_head.weight" in on


@pytest.mark.parametrize("label,colour", [lc for lc in LABELS if ct.MODELS[lc[0]].get("pre")],
                         ids=[i for i, lc in zip(IDS, LABELS) if ct.MODELS[lc[0]].get("pre")])
def test_the_blank_dvs_reset_observation_has_turn_mean_exactly_zero(monkeypatch, label, colour):
    """Every DVS episode starts on an all-zero observation (no previous frame, no events). It is its own mirror, so
    the symmetrised turn mean there is exactly 0.0; the flag-off model's is not (control)."""
    _setup(monkeypatch, colour)
    space = _space(label)
    _, obs = _chain_obs(label, _frames(2), steps=0)
    assert int(obs[0].count_nonzero()) == 0
    models = _build(label, space)
    with torch.no_grad():
        assert bool((_run(models, obs[0])[0][:, 0] != 0).all())
        monkeypatch.setenv(MIRROR_ENV, "1")
        assert bool((_run(models, obs[0])[0][:, 0] == 0).all())


@pytest.mark.parametrize("label,colour", [("CNN-UnityRecipe+DVS", True), ("3DCNN-DVS6", True),
                                          ("ViViT-CLTT-Ref", False)])
def test_activation_checkpointing_gives_the_same_outputs_and_gradients(monkeypatch, label, colour):
    _setup(monkeypatch, colour)
    space = _space(label)
    monkeypatch.setenv(MIRROR_ENV, "1")
    models = _build(label, space)
    x = _inputs(label, space)["chain"]
    out = {}
    for ck in ("0", "1"):
        monkeypatch.setenv(MIRROR_CKPT_ENV, ck)
        for m in models.values():
            m.zero_grad(set_to_none=True)
        a, b = _run(models, x)
        ((a ** 2).sum() + (b ** 2).sum()).backward()
        out[ck] = (a.detach(), b.detach(), {k: p.grad.clone() for m in models.values()
                                            for k, p in m.named_parameters() if p.grad is not None})
    assert torch.equal(out["0"][0], out["1"][0]) and torch.equal(out["0"][1], out["1"][1])
    assert out["0"][2].keys() == out["1"][2].keys()
    for k in out["0"][2]:
        torch.testing.assert_close(out["1"][2][k], out["0"][2][k], rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------- the builder's refusals and the knob's spelling

def test_builder_refuses_models_the_mirror_does_not_cover(monkeypatch):
    space = gym.spaces.Box(0, 255, (3, H, W), np.uint8)
    monkeypatch.setenv(MIRROR_ENV, "1")
    kw = dict(encoder_cls=encoder_mapping["nature_cnn"], encoder_kwargs={"features_dim": 32}, observation_space=space,
              device=torch.device("cpu"), cfg=ModelCfg(hidden_sizes=[], shared_encoder=True))
    box2 = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    for algo in ("SAC", "DDPG"):
        with pytest.raises(ValueError, match="GaussianActor \\+ ValueCritic only"):
            builder_mod.build_models_for_algorithm(algorithm_spec(Brain(encoder="nature_cnn", algorithm=algo).algorithm),
                                                   action_space=box2, **kw)
    ppo = algorithm_spec(Brain(encoder="nature_cnn", algorithm="PPO").algorithm)
    with pytest.raises(ValueError, match="2-D \\[turn, move\\] action"):
        builder_mod.build_models_for_algorithm(ppo, action_space=gym.spaces.Box(-1.0, 1.0, (4,), np.float32), **kw)
    builder_mod.build_models_for_algorithm(ppo, action_space=box2, **kw)          # control: PPO, 2-D builds


def test_knob_is_strict(monkeypatch):
    assert mirror_equiv_enabled() is False
    for v in ("0", "off", "false", ""):
        monkeypatch.setenv(MIRROR_ENV, v)
        assert mirror_equiv_enabled() is False
    for v in ("1", "on", "true"):
        monkeypatch.setenv(MIRROR_ENV, v)
        assert mirror_equiv_enabled() is True
    monkeypatch.setenv(MIRROR_ENV, "mirror")
    with pytest.raises(ValueError, match="not a boolean"):
        mirror_equiv_enabled()


# ---------------------------------------------------------------- the aux losses see the unflipped encoder pass

C2, H2, W2 = 6, 40, 64


class _AgentEnv:
    num_envs = 1
    num_agents = 1
    observation_space = gym.spaces.Box(0, 255, (C2, H2, W2), np.uint8)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    state_space = None


def _aux_run(monkeypatch, aux, mirror):
    """Brain -> build_agents -> AuxLossPPO (the cltt_step_fixes harness), one rollout + one update. Returns the agent,
    the (encoder, loss) of every aux compute, and the encoder parameters before the update."""
    from nett_skrl.brain.agent_factory import build_agents
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("NETT_AUX_LOSS", aux)
        mp.setenv("NETT_AUX_WEIGHT", "1")
        mp.setenv("NETT_AUX_BATCH", "6")
        if mirror:
            mp.setenv(MIRROR_ENV, "1")
        torch.manual_seed(0)
        brain = Brain(encoder="nature_cnn", algorithm="PPO", encoder_cfg={"features_dim": 32},
                      algorithm_cfg={"rollouts": 16, "mini_batches": 2, "learning_epochs": 1,
                                     "grad_norm_clip": 0.5, "learning_rate": 1e-3},
                      model={"shared_encoder": True, "hidden_sizes": [16]})
        agent = build_agents(brain, _AgentEnv(), torch.device("cpu"))[0]
        agent.init(trainer_cfg=None)
        agent.enable_training_mode(True)
        seen = []
        inner = agent._aux.compute

        def compute(encoder, observations):
            loss = inner(encoder, observations)
            seen.append((encoder, loss.detach().clone()))
            return loss

        mp.setattr(agent._aux, "compute", compute)
        g = torch.Generator().manual_seed(1)
        torch.manual_seed(1)
        n = C2 * H2 * W2
        obs = torch.randint(0, 256, (1, n), generator=g).float()
        with torch.no_grad():
            for t in range(16):
                acts, _ = agent.act(obs, None, timestep=t, timesteps=100)
                nxt = torch.randint(0, 256, (1, n), generator=g).float()
                agent.record_transition(observations=obs, states=None, actions=acts,
                                        rewards=torch.rand(1, 1, generator=g), next_observations=nxt,
                                        next_states=None, terminated=torch.zeros(1, 1, dtype=torch.bool),
                                        truncated=torch.zeros(1, 1, dtype=torch.bool), infos={},
                                        timestep=t, timesteps=100)
                obs = nxt
        before = {k: p.detach().clone() for k, p in agent.policy.encoder.named_parameters()}
        torch.manual_seed(1001)
        agent.update(timestep=16, timesteps=100)
    return agent, seen, before


@pytest.mark.parametrize("aux", ["cltt_ref", "vicreg", "vicreg_tt"])
def test_aux_losses_train_the_shared_encoder_on_unflipped_inputs_with_the_mirror_on(monkeypatch, aux):
    agent, seen, before = _aux_run(monkeypatch, aux, mirror=True)
    _, seen_off, _ = _aux_run(monkeypatch, aux, mirror=False)
    assert seen and len(seen) == len(seen_off)
    enc = agent.policy.encoder
    assert agent.value.encoder is enc and all(e is enc for e, _ in seen)     # the ONE shared encoder
    assert all(bool(torch.isfinite(loss)) for _, loss in seen)
    # The aux path never runs the mirrored pass: its first loss (before any optimizer step, same RNG draws) is
    # bit-identical with the knob off. Later minibatches differ because PPO's own gradient now differs.
    assert torch.equal(seen[0][1], seen_off[0][1]), (aux, seen[0][1], seen_off[0][1])
    moved = [k for k, p in enc.named_parameters() if not torch.equal(p.detach(), before[k])]
    assert moved, "the update must move the shared encoder"


# ---------------------------------------------------------------- EMA / slot targets copy a mirror-cached encoder

def _mirror_cached_encoder(monkeypatch):
    space = gym.spaces.Box(0, 255, (C2, H2, W2), np.uint8)
    enc = encoder_mapping["nature_cnn"](space, features_dim=32)

    class _Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = enc
            self.trunk = torch.nn.Identity()
            self.observation_space = space

    monkeypatch.setenv(MIRROR_ENV, "1")
    ctx = shared_feature_cache(enc)
    ctx.__enter__()
    features_forward_mirrored(_Head(), {"observations": torch.randint(0, 256, (2, C2 * H2 * W2)).float()})
    for a in (_CACHE_ATTR, _MIRROR_CACHE_ATTR):
        cached = getattr(enc, a)
        assert cached is not None and cached[1].grad_fn is not None             # non-leaf, as on a GPU arm
    return enc, ctx


def test_ema_teacher_and_slot_target_copy_an_encoder_holding_both_cache_slots(monkeypatch):
    from nett_skrl.brain.aux.ema_teacher import EMATeacher
    from nett_skrl.brain.aux.slot_contrast_aux import SlotContrastAuxLoss
    enc, ctx = _mirror_cached_encoder(monkeypatch)
    try:
        # control: the raw copy refuses -- the failure the guard exists for, now in the mirror slot too
        setattr(enc, _CACHE_ATTR, None)
        with pytest.raises(RuntimeError, match="deepcopy"):
            copy.deepcopy(enc)
        enc, ctx2 = _mirror_cached_encoder(monkeypatch)
        live = {a: getattr(enc, a) for a in (_CACHE_ATTR, _MIRROR_CACHE_ATTR)}
        holder = EMATeacher(enc)                                                   # must not raise
        slot = SlotContrastAuxLoss(enc)
        slot._update_target(enc)                                                   # the lazy copy: must not raise
        for a, v in live.items():
            assert getattr(enc, a) is v                                            # the live cache is untouched
            assert getattr(holder.module, a, None) is None and getattr(slot._target[0], a, None) is None
        ctx2.__exit__(None, None, None)
    finally:
        ctx.__exit__(None, None, None)
