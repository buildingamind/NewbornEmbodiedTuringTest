"""ViT-SlotRecon -- owner, workspace DECISIONS §162 item 4 as the owner changed it: the encoder reads frame t, the
loss is the summed squared L2 distance to frame t+1, on the XSP2 machinery. Encoder brain/encoders/
vit_slot_recon.py, aux brain/aux/slot_recon_aux.py, label "ViT-SlotRecon" in examples/campaign_train.py.

Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU). The owner's list:
  (a) after a PPO backward the encoder and slot params hold no grad and the policy head's do
  (b) the aux gives the encoder grads
  (c) the reconstruction loss decreases over ~50 CPU steps on a tiny batch
  (d) the model builds at 128x80 with 2-frame input
  + the target is the newest frame of obs_{t+1}; no pair crosses a done; PPO's optimizer holds no encoder
    parameter; the policy loss gives the encoder no gradient.
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.aux.slot_attention_ae import SlotAttention
from nett_skrl.brain.aux.slot_recon_aux import SlotBroadcastDecoder, SlotReconTerm
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.vit_slot_recon import ViTSlotRecon
from nett_skrl.brain.registry import encoder_mapping

H, W, CPF = 80, 128, 3
C = 2 * CPF
_SRC = Path(__file__).resolve().parents[1]
_TRAIN = _SRC / "examples" / "campaign_train.py"
KNOBS = ("NETT_SLOTRECON_BATCH", "NETT_SLOTRECON_LR_SCALE")
TELEMETRY = ("B", "loss", "mse", "copy_mse", "skill", "alpha_max", "slot_share_min", "slot_share_max",
             "slot_norm", "window_turn", "enc_grad_norm", "lr")
CFG = {"features_dim": 192, "patch_size": 16, "embed_dim": 144, "depth": 3, "num_heads": 4, "mlp_ratio": 2.0,
       "num_slots": 3, "slot_dim": 64, "slot_iters": 3, "num_frames": 2}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in KNOBS + ("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_AUX_LOSS", "NETT_AUX_WEIGHT",
                         "NETT_AUX_WEIGHT_OVERRIDE", "NETT_AUX_BATCH", "NETT_AUX_STRICT", "NETT_AMP",
                         "NETT_DISABLE_FEATURE_CACHE", "NETT_DECOUPLE_ENCODER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_SLOTRECON_BATCH", "4")
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
    return ViTSlotRecon(_space(), **{**CFG, **kw})


def _n(module):
    return sum(p.numel() for p in module.parameters())


def _prepared(n=4, seed=5):
    return torch.rand(n, C, H, W, generator=torch.Generator().manual_seed(seed))


def _term(enc, lr=1e-3):
    term = SlotReconTerm(enc)
    term.build_optimizer(lr)
    return term


class _Memory:
    """Every (t, env) frame stack is distinct; `done_every` flags terminated at t % done_every == done_every - 1."""

    def __init__(self, t_max=24, n_env=2, seed=3, done_every=0):
        g = torch.Generator().manual_seed(seed)
        acts = torch.zeros(t_max, n_env, 2)
        acts[:, :, 0] = torch.linspace(-0.5, 0.5, t_max).view(t_max, 1)
        term = torch.zeros(t_max, n_env, 1, dtype=torch.bool)
        if done_every:
            term[done_every - 1::done_every] = True
        self.tensors = {
            "observations": torch.randint(0, 255, (t_max, n_env, H, W, C), generator=g, dtype=torch.uint8),
            "terminated": term,
            "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "actions": acts,
        }
        self.memory_size, self.filled, self.memory_index = t_max, True, 0

    def frames(self):
        return self.tensors["observations"].float().permute(0, 1, 4, 2, 3) / 255.0      # (T, env, C, H, W)


def _locate(frames, stack):
    return next((t, e) for t in range(frames.shape[0]) for e in range(frames.shape[1])
                if torch.equal(frames[t, e], stack))


def _spy_step(term):
    seen = {}
    real = term.step

    def spy(encoder, pt, ptk, turn):
        seen.update(pt=pt, ptk=ptk)
        return real(encoder, pt, ptk, turn)

    term.step = spy
    return seen


# ---------------------------------------------------------------- a real skrl agent (no simulator)
class _Env:
    num_envs = 1
    num_agents = 1
    observation_space = _space()
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    state_space = None


def _agent(monkeypatch, *, aux="slot_recon", seed=0, epochs=1, mini_batches=1, rollouts=16, shared=True,
           lr=1e-4, weight="1"):
    """The production path: Brain -> build_agents -> AuxLossPPO (aux="slot_recon") or MetricsPPO (aux=None)."""
    from nett_skrl.brain.agent_factory import build_agents
    from nett_skrl.brain.brain import Brain
    if aux is None:
        monkeypatch.delenv("NETT_AUX_LOSS", raising=False)
        monkeypatch.delenv("NETT_AUX_WEIGHT", raising=False)
    else:
        monkeypatch.setenv("NETT_AUX_LOSS", aux)
        monkeypatch.setenv("NETT_AUX_WEIGHT", weight)
    torch.manual_seed(seed)
    brain = Brain(encoder="vit_slot_recon", algorithm="PPO", encoder_cfg=dict(CFG),
                  algorithm_cfg={"rollouts": rollouts, "mini_batches": mini_batches, "learning_epochs": epochs,
                                 "grad_norm_clip": 0.5, "learning_rate": lr},
                  model={"shared_encoder": shared, "hidden_sizes": []})
    agent = build_agents(brain, _Env(), torch.device("cpu"))[0]
    agent.init(trainer_cfg=None)
    agent.enable_training_mode(True)
    return agent


def _rollout(agent, seed=1, steps=16):
    """skrl's trainer loop shape: act + record_transition under no_grad, then update()."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
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
    torch.manual_seed(seed + 1000)
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


# ================================================================ (d) label, build, counts

def test_label_and_registries(campaign):
    spec = campaign.MODELS["ViT-SlotRecon"]
    assert spec == dict(encoder="vit_slot_recon", cfg=campaign.VIT_SLOT_RECON_CFG, framestack=True,
                        aux="slot_recon", aux_weight=1.0)
    assert {k: v for k, v in spec["cfg"].items() if k != "trainable"} == CFG
    for k in ("patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio"):      # ViT-CLTT-Ref's trunk, patch 16
        assert spec["cfg"][k] == campaign.VIT_CFG[k], k
    assert "pre" not in spec and spec["aux"] != "cltt_ref"                          # no CLTT
    assert encoder_mapping["vit_slot_recon"] is ViTSlotRecon
    assert AUX_LOSSES["slot_recon"](_enc()).__class__ is SlotReconTerm


def test_builds_at_128x80_with_two_frames_and_pinned_counts(campaign):
    kwargs = EncoderCfg(**campaign.MODELS["ViT-SlotRecon"]["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    enc = encoder_mapping["vit_slot_recon"](_space(), **kwargs)
    term = SlotReconTerm(enc)
    assert (_n(enc), _n(enc.vit), _n(enc.slot_in), _n(enc.slot_attention)) == (687_552, 619_488, 13_728, 54_336)
    assert _n(term.decoder) == 514_948 and _n(term.head) == 0
    assert enc.token_grid == (5, 8) and enc.num_slots == 3 and enc.slot_dim == 64
    assert {id(p) for p in enc.stream_parameters()} == {id(p) for p in enc.parameters()}
    x = torch.randint(0, 256, (2, C * H * W)).float()
    out = enc(x)
    assert out.shape == (2, 192) and not out.requires_grad
    pred, alphas = term.decoder(enc.encode_slots(torch.rand(2, CPF, H, W), sample=True))
    assert pred.shape == (2, CPF, H, W) and alphas.shape == (2, 3, H, W)
    torch.testing.assert_close(alphas.sum(1), torch.ones(2, H, W))                 # softmax over SLOTS


@pytest.mark.parametrize("kw,match", [
    ({"num_frames": 1}, "framestack 2"),
    ({"features_dim": 160}, "features_dim must be 192"),
])
def test_encoder_refusals(kw, match):
    channels = CPF * kw.get("num_frames", 2)
    with pytest.raises(ValueError, match=match):
        ViTSlotRecon(_space(channels), **{**CFG, **kw})


def test_encoder_refuses_amp(monkeypatch):
    monkeypatch.setenv("NETT_AMP", "bf16")
    with pytest.raises(ValueError, match="Unset NETT_AMP"):
        _enc()


def test_term_refusals(monkeypatch):
    with pytest.raises(TypeError, match="ViT-SlotRecon"):
        SlotReconTerm(nn.Linear(2, 2))
    monkeypatch.setenv("NETT_SLOTRECON_BATCH", "3")
    with pytest.raises(ValueError, match=">= 4"):
        SlotReconTerm(_enc())
    monkeypatch.setenv("NETT_SLOTRECON_BATCH", "4")
    with pytest.raises(RuntimeError, match="attach_memory"):
        _term(_enc()).compute(_enc(), None)


# ================================================================ policy path: frame t only, detached, fixed noise

def test_policy_reads_only_the_newest_frame():
    enc = _enc()
    x = _prepared() * 255.0
    y = x.clone()
    y[:, :CPF] = torch.rand_like(y[:, :CPF]) * 255.0                  # x_(t-1) changed
    z = x.clone()
    z[:, CPF:] = z[:, CPF:] * 0.5                                     # x_t changed
    flat = lambda t: t.reshape(t.shape[0], -1)                       # flat CHW, as ChannelsFirst delivers
    assert torch.equal(enc(flat(x)), enc(flat(y)))
    assert not torch.equal(enc(flat(x)), enc(flat(z)))


def test_policy_features_are_deterministic_across_modes_and_draw_no_rng():
    enc = _enc()
    x = torch.randint(0, 256, (3, C * H * W)).float()
    enc.eval()
    rng = torch.get_rng_state()
    a = enc(x)
    enc.train()
    b = enc(x)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(a, b)


def test_slot_attention_sample_none_keeps_the_mode_rule():
    sa = SlotAttention(3, dim=8, iters=1, mlp_hidden=16)
    x = torch.rand(2, 5, 8)
    sa.eval()
    assert torch.equal(sa(x), sa(x, sample=False))
    sa.train()
    torch.manual_seed(1)
    a = sa(x)
    torch.manual_seed(1)
    assert torch.equal(a, sa(x, sample=True))
    assert not torch.equal(a, sa(x, sample=False))


def test_policy_loss_gives_the_encoder_no_gradient():
    """The policy loss gives the encoder no gradient; the same probe on the AUX path DOES see one."""
    enc = _enc()
    head = nn.Linear(192, 1)
    feats = enc(torch.randint(0, 256, (4, C * H * W)).float())
    head(feats).sum().backward()
    assert all(p.grad is None for p in enc.parameters())
    assert head.weight.grad is not None and float(head.weight.grad.abs().sum()) > 0
    # positive control: the gradient path the aux uses reaches the trunk
    nn.Linear(192, 1)(enc.encode_slots(torch.rand(4, CPF, H, W), sample=False).flatten(1)).sum().backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in enc.vit.parameters())


# ================================================================ loss and target

def test_loss_is_the_sum_of_squares_against_the_newest_frame_of_obs_t_plus_1():
    enc = _enc()
    term = _term(enc)
    pt, ptk = _prepared(seed=1), _prepared(seed=2)
    torch.manual_seed(4)
    loss = term.score(enc, pt, ptk)
    torch.manual_seed(4)
    with torch.no_grad():
        pred, _ = term.decoder(enc.encode_slots(pt[:, CPF:], sample=True))
        manual = ((pred - ptk[:, CPF:]) ** 2).sum()
    torch.testing.assert_close(loss.detach(), manual, rtol=0, atol=0)
    assert math.isclose(term.last_scalars["mse"], float(manual) / (4 * CPF * H * W), rel_tol=1e-6)
    assert math.isclose(term.last_scalars["copy_mse"], float(((pt[:, CPF:] - ptk[:, CPF:]) ** 2).mean()),
                        rel_tol=1e-6)
    # obs_{t+1}'s OLDER frame (== x_t, the encoder's own input) is never read; its newest frame is
    ptk_old = ptk.clone()
    ptk_old[:, :CPF] = torch.rand_like(ptk_old[:, :CPF])
    torch.manual_seed(4)
    assert torch.equal(term.score(enc, pt, ptk_old).detach(), manual)
    ptk_new = ptk.clone()
    ptk_new[:, CPF:] = ptk_new[:, CPF:] * 0.5
    torch.manual_seed(4)
    assert not torch.equal(term.score(enc, pt, ptk_new).detach(), manual)
    # and obs_t's older frame (x_(t-1)) is not the encoder's input
    pt_old = pt.clone()
    pt_old[:, :CPF] = torch.rand_like(pt_old[:, :CPF])
    torch.manual_seed(4)
    assert torch.equal(term.score(enc, pt_old, ptk).detach(), manual)


def test_compute_targets_obs_t_plus_1_from_memory_and_returns_a_detached_zero():
    enc = _enc()
    term = _term(enc)
    mem = _Memory()
    term.attach_memory(mem)
    seen = _spy_step(term)
    out = term.compute(enc, None)
    assert out.shape == () and float(out) == 0.0 and not out.requires_grad
    frames = mem.frames()
    assert seen["pt"].shape == seen["ptk"].shape == (4, C, H, W)
    for i in range(4):
        t, e = _locate(frames, seen["pt"][i])
        assert torch.equal(seen["ptk"][i], frames[t + 1, e])
    assert set(TELEMETRY) <= set(term.last_scalars) and term.last_scalars["B"] == 4.0
    assert all(math.isfinite(v) for v in term.last_scalars.values() if isinstance(v, float))


def test_no_drawn_pair_crosses_a_done(monkeypatch):
    """Dones every 3rd step: a draw that ignored them would cross one in 1/3 of its windows. The slab draw
    shrinks each slab to fit inside one episode (here <= 2 windows), so B can fall below the batch knob."""
    monkeypatch.setenv("NETT_SLOTRECON_BATCH", "16")
    enc = _enc()
    term = _term(enc)
    mem = _Memory(done_every=3)
    term.attach_memory(mem)
    frames, done = mem.frames(), mem.tensors["terminated"][..., 0]
    pairs = 0
    for draw in range(8):
        torch.manual_seed(draw)
        obs_t, obs_tk, _, turn = term._draw(enc)
        assert 2 <= obs_t.shape[0] <= 4
        prep = lambda o: enc._prepare_image(o)
        for a, b in zip(prep(obs_t), prep(obs_tk)):
            t, e = _locate(frames, a)
            assert not bool(done[t, e]), (t, e)                       # transition t ends no episode
            assert torch.equal(b, frames[t + 1, e])
            pairs += 1
    assert pairs >= 16


# ================================================================ (a)/(b): optimizer exclusion and gradients

def test_ppo_optimizer_holds_no_encoder_parameter_and_aux_adam_owns_encoder_and_decoder(monkeypatch):
    agent = _agent(monkeypatch)
    term, enc = agent._aux, agent.policy.encoder
    assert agent.value.encoder is enc
    ppo_ids = {id(p) for g in agent.optimizer.param_groups for p in g["params"]}
    enc_ids = {id(p) for p in enc.parameters()}
    assert not ppo_ids & enc_ids
    assert not ppo_ids & {id(p) for p in term.decoder.parameters()}
    heads = [p for n, p in [*agent.policy.named_parameters(), *agent.value.named_parameters()]
             if not n.startswith("encoder.")]
    assert {id(p) for p in heads} == ppo_ids and ppo_ids
    aux_ids = [id(p) for g in term.optimizer.param_groups for p in g["params"]]
    assert set(aux_ids) == enc_ids | {id(p) for p in term.decoder.parameters()} and len(aux_ids) == len(set(aux_ids))
    assert isinstance(term.optimizer, torch.optim.Adam) and list(term.head.parameters()) == []
    assert agent.checkpoint_modules["slot_recon_optimizer"] is term.optimizer
    assert agent.checkpoint_modules["slot_recon_decoder"] is term.decoder


@pytest.mark.parametrize("aux", ["slot_recon", None])
def test_after_a_ppo_backward_the_encoder_has_no_grad_and_the_heads_do(monkeypatch, aux):
    """(a) At every PPO optimizer step the encoder and slot params hold NO grad and the policy/value heads do.
    aux=None (plain MetricsPPO on the same encoder) shows it is the encoder's detachment, not the aux's reset."""
    agent = _agent(monkeypatch, aux=aux, epochs=2, mini_batches=2)
    steps = _record_ppo_grads(agent)
    enc_before = _state(agent.policy.encoder)
    _rollout(agent)
    assert len(steps) == 4
    for step in steps:
        enc_keys = [k for k in step if ".encoder." in k or k.startswith(("policy.encoder", "value.encoder"))]
        assert enc_keys and all(step[k] is None for k in enc_keys)
        assert any(k.startswith("policy.encoder.slot_attention") for k in enc_keys)
        head = [v for k, v in step.items() if k.startswith("policy.") and k not in enc_keys]
        assert head and all(v is not None for v in head) and any(float(v.abs().sum()) > 0 for v in head)
    trained = any(not torch.equal(v, agent.policy.encoder.state_dict()[k]) for k, v in enc_before.items())
    if aux is None:
        assert not trained                                            # PPO alone never moves the encoder
    else:
        assert trained and agent._aux.n_steps == 4                    # one aux Adam step per minibatch


def test_aux_gives_every_encoder_parameter_a_gradient():
    """(b) The summed reconstruction error reaches the trunk, the slot MLP and Slot Attention."""
    enc = _enc()
    term = _term(enc)
    loss = term.score(enc, _prepared(seed=1), _prepared(seed=2))
    loss.backward()
    groups = {"vit": enc.vit, "slot_in": enc.slot_in, "slot_attention": enc.slot_attention, "decoder": term.decoder}
    for name, module in groups.items():
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None for g in grads), name
        assert sum(float(g.abs().sum()) for g in grads) > 0, name
    term.step(enc, _prepared(seed=1), _prepared(seed=2))
    assert term.last_scalars["enc_grad_norm"] > 0
    assert all(p.grad is None for p in enc.parameters())              # back to None after the step


def test_bind_agent_refuses_unshared_encoder_and_inert_weight(monkeypatch):
    with pytest.raises(ValueError, match="shared_encoder=True"):
        _agent(monkeypatch, shared=False)
    with pytest.raises(ValueError, match="aux_weight=2"):
        _agent(monkeypatch, weight="2")


def test_aux_lr_follows_ppo_lr_times_scale(monkeypatch):
    monkeypatch.setenv("NETT_SLOTRECON_LR_SCALE", "0.5")
    agent = _agent(monkeypatch, lr=3e-4)
    term = agent._aux
    assert term.optimizer.param_groups[0]["lr"] == pytest.approx(1.5e-4)
    agent.optimizer.param_groups[0]["lr"] = 2e-4
    _rollout(agent)
    assert term.last_scalars["lr"] == pytest.approx(1e-4)


# ================================================================ (c) learning

def _square_pair(b=4, seed=0):
    """A textured square that moves 8 px right: x_t (input) and x_{t+1} (target), both learnable from x_t."""
    g = torch.Generator().manual_seed(seed)
    pt = torch.full((b, C, H, W), 0.2)
    ptk = torch.full((b, C, H, W), 0.2)
    for i in range(b):
        r, c = 10 + 12 * i, 20 + 10 * i
        tex = torch.rand(CPF, 20, 20, generator=g) * 0.5 + 0.5
        pt[i, CPF:, r:r + 20, c:c + 20] = tex
        ptk[i, CPF:, r:r + 20, c + 8:c + 28] = tex
    return pt, ptk


def test_reconstruction_loss_decreases_over_50_cpu_steps():
    enc = _enc(seed=0)
    term = _term(enc, lr=1e-3)
    pt, ptk = _square_pair()
    torch.manual_seed(0)
    losses = []
    for _ in range(50):
        term.step(enc, pt, ptk)
        losses.append(term.last_scalars["loss"])
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert all(math.isfinite(v) for v in losses)
    assert last < 0.5 * first, (first, last)
    assert term.n_steps == 50


# ================================================================ knobs

def test_stray_slot_recon_knob_on_another_label_refuses(campaign, monkeypatch):
    monkeypatch.delenv("NETT_SLOTRECON_BATCH")
    for label, spec in campaign.MODELS.items():
        assert campaign.refuse_stray_slot_recon_knobs(label, spec) is None
    monkeypatch.setenv("NETT_SLOTRECON_LR_SCALE", "2")
    with pytest.raises(ValueError, match="not a slot_recon arm"):
        campaign.refuse_stray_slot_recon_knobs("XSP2", campaign.MODELS["XSP2"])
    with pytest.raises(ValueError, match="not a slot_recon arm"):
        campaign.refuse_stray_slot_recon_knobs("ViT-CLTT-Ref", campaign.MODELS["ViT-CLTT-Ref"])
    assert campaign.refuse_stray_slot_recon_knobs("ViT-SlotRecon", campaign.MODELS["ViT-SlotRecon"]) is None


def test_knobs_are_literal_reads_and_in_the_env_index():
    text = (_SRC / "nett_skrl" / "brain" / "aux" / "slot_recon_aux.py").read_text()
    index = (_SRC / "docs" / "env_vars.md").read_text()
    for k in KNOBS:
        assert f'"{k}"' in text and f"`{k}`" in index, k


def test_decoder_is_a_spatial_broadcast_of_each_slot():
    dec = SlotBroadcastDecoder(8, (5, 8), 3, 4)
    s = torch.rand(2, 3, 8)
    pred, alphas = dec(s)
    assert pred.shape == (2, 3, 80, 128) and alphas.shape == (2, 3, 80, 128)
    # slots are decoded independently: permuting slots leaves the mixture unchanged
    pred_p, _ = dec(s[:, [2, 0, 1]])
    torch.testing.assert_close(pred, pred_p, rtol=1e-5, atol=1e-6)
