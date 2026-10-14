"""CLTT-step memory/compute fixes 1, 2 and 2b (workspace DECISIONS §111, owner 2026-10-14, "Proceed
with 1 + 2 + 2b"), and the §111 defaults (cltt_ref NETT_AUX_BATCH 128; matched bf16 is in
test_uint8_cpu_buffer.py).

Each fix must be EXACT up to float summation order, so each is checked old path vs new path at one
seed -- loss and every parameter's .grad -- and each has a positive control showing the new path
really ran (a check that cannot see the difference would pass on a no-op):
  1. Fix 1 (cltt_ref encodes B + max(offsets) rows ONCE, roles sliced): loss + encoder AND head grads
     == the three-pass path, on all seven matched encoders, the CLTT-Stack variant and the compact
     encoders; control = one encode_prepared call over B+max rows (vs 3 over B), `encode_dedup` 1.
     FALLBACK: any live dropout / training BatchNorm takes the old path, bit for bit (same RNG);
     control = the same encoder class without the stochastic layer is accepted.
  2. Fix 2 (aux forward+backward AFTER the PPO backward, no retain_graph): every gradient PPO's
     optimizer sees, at every step of a real update (Brain -> build_agents -> AuxLossPPO), equals the
     old order's; control = the recorded event order and retain_graph flags differ as intended.
     Scope: only an aux with `independent_of_ppo_graph`, never under NETT_AUX_STRICT=1.
  2b. Fix 2b (RL encoder pass under no_grad when NETT_DECOUPLE_ENCODER=1 and the aux is CLTT-Ref):
     every gradient equal; control = the RL encoder passes ran with grad disabled (and the aux's own
     passes with grad enabled); without decouple nothing changes; an aux that may read the RL graph
     is never marked.
Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU).
"""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn

from nett_skrl.brain.aux.cltt_ref_aux import DEDUP_ENV, CLTTReferenceAuxLoss, dedup_is_exact
from nett_skrl.brain.aux.cltt_ref_stack_aux import CLTTReferenceStackAuxLoss
from nett_skrl.brain.aux.ppo_aux import AuxLossPPO
from nett_skrl.brain.encoders.compact_vit import CompactViT
from nett_skrl.brain.encoders.matched import EYE_HW, VARIANTS, MatchedEncoder
from nett_skrl.brain.models.utils.features import RL_NO_GRAD_ATTR, RL_NO_GRAD_ENV

KNOBS = ("NETT_AUX_BATCH", "NETT_AUX_CLTT_REF_OFFSETS", "NETT_AUX_CLTT_STACK_OFFSETS",
         "NETT_AUX_CLTT_REF_TEMP", "NETT_AUX_CLTT_REF_DIAG", "NETT_AUX_CLTT_MASK_POSITIVE_TWIN",
         "NETT_AUX_CLTT_CHANNELS_PER_FRAME", DEDUP_ENV, "NETT_AUX_AFTER_PPO_BACKWARD", RL_NO_GRAD_ENV,
         "NETT_DECOUPLE_ENCODER", "NETT_AUX_STRICT", "NETT_AUX_LOSS", "NETT_AUX_WEIGHT",
         "NETT_AUX_WEIGHT_OVERRIDE", "NETT_DISABLE_FEATURE_CACHE", "NETT_MATCHED_AMP",
         "NETT_CONV3D_ORTHO_INIT")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            yield
    finally:
        torch.set_num_threads(threads)


class _Memory:
    """A rollout ring as cltt_ref reads it: uint8 HWC frames + both boundary flags, no resets."""

    def __init__(self, h, w, c, t_max=12, n_env=2, seed=3):
        g = torch.Generator().manual_seed(seed)
        self.tensors = {
            "observations": torch.randint(0, 256, (t_max, n_env, h * w * c), generator=g,
                                          dtype=torch.uint8),
            "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
        }
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


class _Counting:
    """Wraps encode_prepared to record the row count of each call (the positive control of fix 1)."""

    def __init__(self, encoder):
        self.rows = []
        inner = encoder.encode_prepared

        def encode_prepared(x):
            self.rows.append(int(x.shape[0]))
            return inner(x)

        encoder.encode_prepared = encode_prepared


def _matched(variant):
    frames = VARIANTS[variant][0]
    space = gym.spaces.Box(0, 255, (*EYE_HW, 3 * frames), np.uint8)
    torch.manual_seed(0)
    return MatchedEncoder(space, variant=variant), (*EYE_HW, 3 * frames)


def _compact_3dcnn():
    from nett_skrl.brain.encoders.compact_3dcnn import Compact3DCNN
    space = gym.spaces.Box(0, 255, (16, 16, 6), np.uint8)
    torch.manual_seed(0)
    return Compact3DCNN(space, features_dim=32, num_frames=2, conv_dim=8), (16, 16, 6)


def _compact_vit(**kw):
    space = gym.spaces.Box(0, 255, (80, 128, 6), np.uint8)
    torch.manual_seed(0)
    return CompactViT(space, features_dim=32, patch_size=16, embed_dim=32, depth=1, num_heads=2,
                      **kw), (80, 128, 6)


def _one_compute(encoder, aux, memory, *, dedup, seed=5):
    """One compute()+backward from a fresh copy of (encoder, aux): loss, every grad, the encode rows."""
    enc, head_aux = copy.deepcopy(encoder), copy.deepcopy(aux)
    head_aux.attach_memory(memory)
    counter = _Counting(enc)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(DEDUP_ENV, dedup)
        torch.manual_seed(seed)                       # the window draw
        loss = head_aux.compute(enc, torch.empty(0))
    loss.backward()
    grads = {f"enc.{k}": p.grad.detach().clone() for k, p in enc.named_parameters() if p.grad is not None}
    grads.update({f"head.{k}": p.grad.detach().clone() for k, p in head_aux.head.named_parameters()})
    buffers = {k: b.detach().clone() for k, b in head_aux.head.named_buffers()}
    return loss.detach(), grads, buffers, counter.rows, head_aux.last_scalars


def _assert_same(old, new, *, exact=False):
    """Loss, every grad (encoder + head) and the head's BatchNorm buffers: close, or bit-identical."""
    (l0, g0, b0, *_), (l1, g1, b1, *_) = old, new

    def check(name, a, b, scale=None):
        if exact:
            assert torch.equal(a, b), f"{name}: not bit-identical"
        else:
            assert _summation_order_close(a, b, scale), (
                f"{name}: max|diff| {float((a - b).abs().max()):.3e}, scale "
                f"{float(b.abs().max()) if scale is None else scale:.3e}")

    check("loss", l1, l0)
    assert g0.keys() == g1.keys() and len(g0) > 0
    grad_scale = max(float(g.abs().max()) for g in g0.values())
    for k in g0:
        check(k, g1[k], g0[k], grad_scale)
    assert b0.keys() == b1.keys()
    for k in b0:
        check(k, b1[k], b0[k])


def _summation_order_close(a, b, scale=None, rel=1e-5):
    """Float32 summation-order agreement: max|a-b| <= rel * scale, scale = max|b| unless given.

    For gradients the scale is the LARGEST gradient in the model, not the tensor's own: rounding
    noise is set by the magnitudes summed, not by the result. Measured: losses bit-equal; grad
    differences ~1e-6 against model-wide gradients of 0.8-6 (relative ~1e-6); and tensors whose true
    gradient is ~0 -- a bias right before the head's BatchNorm, which cancels it -- carry ~1e-7 of
    pure rounding noise that a per-tensor or elementwise tolerance would call a failure. The mutant
    test below shows this tolerance still rejects a real error of the kind fix 1 could make.
    """
    scale = float(b.abs().max()) if scale is None else scale
    return float((a - b).abs().max()) <= rel * scale + 1e-12


# ── Fix 1 ────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_fix1_matched_encoders_loss_and_every_grad_equal_the_three_pass_path(monkeypatch, variant):
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    encoder, (h, w, c) = _matched(variant)
    assert dedup_is_exact(encoder) == (True, "")
    aux = CLTTReferenceAuxLoss(encoder)
    memory = _Memory(h, w, c, t_max=8)
    old = _one_compute(encoder, aux, memory, dedup="0")
    new = _one_compute(encoder, aux, memory, dedup="1")
    _assert_same(old, new)
    # POSITIVE CONTROL: the new path really encoded once, over B + max(offsets) rows.
    assert old[3] == [4, 4, 4] and new[3] == [4 + 2]
    assert old[4]["encode_dedup"] == 0.0 and new[4]["encode_dedup"] == 1.0
    assert old[4]["B"] == new[4]["B"] == 4.0


def test_fix1_cltt_stack_variant_and_compact_3dcnn(monkeypatch):
    monkeypatch.setenv("NETT_AUX_BATCH", "5")
    for build, cls, offsets in ((lambda: _matched("3dcnn"), CLTTReferenceStackAuxLoss, (2, 4)),
                                (_compact_3dcnn, CLTTReferenceAuxLoss, (1, 2)),
                                (_compact_3dcnn, CLTTReferenceStackAuxLoss, (2, 4))):
        encoder, (h, w, c) = build()
        aux = cls(encoder)
        assert aux.offsets == offsets
        memory = _Memory(h, w, c, t_max=14)
        old = _one_compute(encoder, aux, memory, dedup="0")
        new = _one_compute(encoder, aux, memory, dedup="1")
        _assert_same(old, new)
        assert old[3] == [5, 5, 5] and new[3] == [5 + max(offsets)]


def test_fix1_tolerance_has_power_the_head_must_see_each_view_alone(monkeypatch):
    """MUTANT: applying the projection head ONCE over the whole span and slicing after (the tempting
    shortcut) changes the head BatchNorm's batch statistics. The tolerance above must reject that,
    or "equal up to summation order" would not be a test of anything."""
    encoder, (h, w, c) = _matched("vivit")
    aux = CLTTReferenceAuxLoss(encoder)
    raw = _Memory(h, w, c, t_max=8).tensors["observations"][0:6, 0]
    with torch.no_grad():
        z = encoder.encode_prepared(encoder._prepare_image(raw))
        right = [aux.head(z[k:k + 4]) for k in (0, 1, 2)]
        whole = aux.head(z)
        wrong = [whole[k:k + 4] for k in (0, 1, 2)]
    assert all(_summation_order_close(r, r.clone()) for r in right)            # control
    assert not any(_summation_order_close(wr, r) for wr, r in zip(wrong, right))


def test_fix1_compact_vit_without_dropout_is_deduped_and_equal(monkeypatch):
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    encoder, (h, w, c) = _compact_vit()
    assert dedup_is_exact(encoder) == (True, "")
    aux = CLTTReferenceAuxLoss(encoder)
    memory = _Memory(h, w, c, t_max=10)
    _assert_same(_one_compute(encoder, aux, memory, dedup="0"),
                 _one_compute(encoder, aux, memory, dedup="1"))


@pytest.mark.parametrize("scope", ["aux", "all"])
def test_fix1_falls_back_bit_for_bit_under_live_dropout(monkeypatch, scope):
    """CompactViT U35 dropout: separate passes draw separate masks, so the dedup is NOT exact and
    the default must be the old path -- same RNG consumption, so bit-identical to "0"."""
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    encoder, (h, w, c) = _compact_vit(dropout=0.2, dropout_scope=scope)
    ok, why = dedup_is_exact(encoder)
    assert not ok and why == f"dropout_scope={scope!r}"
    aux = CLTTReferenceAuxLoss(encoder)
    memory = _Memory(h, w, c, t_max=10)
    old = _one_compute(encoder, aux, memory, dedup="0")
    default = _one_compute(encoder, aux, memory, dedup="1")
    _assert_same(old, default, exact=True)
    assert default[3] == [4, 4, 4] and default[4]["encode_dedup"] == 0.0
    # Control that the dropout is live in these passes: another seed for the masks only (same
    # window -- one env, one start) changes the loss.
    one = _Memory(h, w, c, t_max=6, n_env=1)
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    a = _one_compute(encoder, aux, one, dedup="1", seed=1)[0]
    b = _one_compute(encoder, aux, one, dedup="1", seed=2)[0]
    assert not torch.equal(a, b)


class _Toy(nn.Module):
    def __init__(self, extra):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.extra = extra


@pytest.mark.parametrize("module,training,expected", [
    (nn.BatchNorm2d(4), True, False), (nn.BatchNorm2d(4), False, True),
    (nn.Dropout(0.2), True, False), (nn.Dropout(0.0), True, True), (nn.Identity(), True, True),
])
def test_fix1_static_rule_refuses_batchnorm_and_dropout_and_accepts_their_controls(module, training,
                                                                                  expected):
    toy = _Toy(module).train(training)
    ok, why = dedup_is_exact(toy)
    assert ok is expected, why
    assert (why == "") is expected


def test_fix1_rule_sees_live_p_and_drop_prob(monkeypatch):
    for attr in ("live_p", "drop_prob"):
        toy = _Toy(nn.Identity())
        setattr(toy.extra, attr, 0.1)
        assert dedup_is_exact(toy)[0] is False
        setattr(toy.extra, attr, 0.0)
        assert dedup_is_exact(toy)[0] is True


def test_fix1_knob_off_and_misspelling_refuses(monkeypatch):
    toy = _Toy(nn.Identity())
    monkeypatch.setenv(DEDUP_ENV, "0")
    assert dedup_is_exact(toy) == (False, f"{DEDUP_ENV}=0")
    monkeypatch.setenv(DEDUP_ENV, "maybe")
    with pytest.raises(ValueError, match=DEDUP_ENV):
        dedup_is_exact(toy)


# ── §111 (c): the CLTT-Ref aux batch default ─────────────────────────────────────────────────

def test_cltt_ref_aux_batch_defaults_to_128_and_a_named_value_wins(monkeypatch):
    encoder, _ = _compact_3dcnn()
    for cls in (CLTTReferenceAuxLoss, CLTTReferenceStackAuxLoss):
        assert cls(encoder).max_samples == 128
        monkeypatch.setenv("NETT_AUX_BATCH", "256")
        assert cls(encoder).max_samples == 256
        monkeypatch.delenv("NETT_AUX_BATCH")


# ── Fix 2 / 2b: through the production agent ─────────────────────────────────────────────────

H, W, C = 40, 64, 6


class _AgentEnv:
    num_envs = 1
    num_agents = 1
    observation_space = gym.spaces.Box(0, 255, (C, H, W), np.uint8)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    state_space = None


def _agent(monkeypatch, aux="cltt_ref", seed=0):
    """Brain -> build_agents -> AuxLossPPO, NatureCNN shared encoder, 2 epochs x 2 minibatches."""
    from nett_skrl.brain.agent_factory import build_agents
    from nett_skrl.brain.brain import Brain
    monkeypatch.setenv("NETT_AUX_LOSS", aux)
    monkeypatch.setenv("NETT_AUX_WEIGHT", "1")
    monkeypatch.setenv("NETT_AUX_BATCH", "6")
    torch.manual_seed(seed)
    brain = Brain(encoder="nature_cnn", algorithm="PPO", encoder_cfg={"features_dim": 32},
                  algorithm_cfg={"rollouts": 16, "mini_batches": 2, "learning_epochs": 2,
                                 "grad_norm_clip": 0.5, "learning_rate": 1e-3},
                  model={"shared_encoder": True, "hidden_sizes": [16]})
    agent = build_agents(brain, _AgentEnv(), torch.device("cpu"))[0]
    agent.init(trainer_cfg=None)
    agent.enable_training_mode(True)
    return agent


def _rollout_and_update(agent, seed=1, steps=16, before_update=None):
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
    if before_update is not None:
        before_update()
    torch.manual_seed(seed + 1000)
    agent.update(timestep=steps, timesteps=100)


def _record_steps(agent):
    """Every gradient the optimizer sees at each step: policy + value + aux head, by name."""
    named = [*(("policy." + k, p) for k, p in agent.policy.named_parameters()),
             *(("value." + k, p) for k, p in agent.value.named_parameters()),
             *(("aux." + k, p) for k, p in agent._aux.head.named_parameters())]
    steps = []

    def hook(_opt, _args, _kwargs):
        steps.append({k: None if p.grad is None else p.grad.detach().clone() for k, p in named})

    agent.optimizer.register_step_pre_hook(hook)
    return steps


def _events(monkeypatch, agent):
    """The order of aux forwards and backwards, with each backward's retain_graph."""
    events = []
    inner_compute = agent._aux.compute

    def compute(encoder, observations):
        events.append("aux_forward")
        return inner_compute(encoder, observations)

    monkeypatch.setattr(agent._aux, "compute", compute)
    inner_backward = torch.Tensor.backward

    def backward(self, *args, **kwargs):
        events.append(f"backward(retain_graph={kwargs.get('retain_graph')})")
        return inner_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", backward)
    return events


def _run(monkeypatch, env=(), aux="cltt_ref", events=False):
    with pytest.MonkeyPatch.context() as mp:
        for k, v in env:
            mp.setenv(k, v)
        agent = _agent(mp, aux=aux)
        steps = _record_steps(agent)
        log = _events(mp, agent) if events else None
        _rollout_and_update(agent)
    return agent, steps, log


def _assert_steps_equal(a, b):
    assert len(a) == len(b) == 4                          # 2 epochs x 2 minibatches
    for sa, sb in zip(a, b):
        assert sa.keys() == sb.keys()
        for k in sa:
            if sa[k] is None or sb[k] is None:
                assert sa[k] is None and sb[k] is None, k
            else:
                torch.testing.assert_close(sb[k], sa[k], rtol=1e-6, atol=1e-7, msg=lambda m, k=k: f"{k}: {m}")


def test_fix2_every_optimizer_step_sees_the_same_gradients(monkeypatch):
    _, old, old_log = _run(monkeypatch, env=[("NETT_AUX_AFTER_PPO_BACKWARD", "0")], events=True)
    agent, new, new_log = _run(monkeypatch, events=True)
    _assert_steps_equal(old, new)
    # The encoder really received aux gradient (else equality would be trivially true for it).
    enc_keys = [k for k in new[0] if k.startswith("policy.encoder.")]
    assert enc_keys and all(new[0][k] is not None and new[0][k].abs().sum() > 0 for k in enc_keys)
    # POSITIVE CONTROL: the order and the retain flag changed exactly as fix 2 says.
    assert old_log == ["aux_forward", "backward(retain_graph=True)", "backward(retain_graph=None)"] * 4
    assert new_log == ["backward(retain_graph=None)", "aux_forward", "backward(retain_graph=None)"] * 4
    assert agent._aux_after_ppo_backward() is True


def test_fix2_strict_and_out_of_scope_aux_keep_the_old_order(monkeypatch):
    _, _, strict_log = _run(monkeypatch, env=[("NETT_AUX_STRICT", "1")], events=True)
    assert strict_log == ["aux_forward", "backward(retain_graph=None)"] * 4        # fused
    agent = _agent(monkeypatch, aux="vicreg")
    assert getattr(agent._aux, "independent_of_ppo_graph", False) is False
    assert agent._aux_after_ppo_backward() is False
    monkeypatch.setenv("NETT_AUX_AFTER_PPO_BACKWARD", "sometimes")
    cl = _agent(monkeypatch)
    with pytest.raises(ValueError, match="NETT_AUX_AFTER_PPO_BACKWARD"):
        cl._aux_after_ppo_backward()


def test_with_cltt_ref_composite_is_not_in_scope():
    from nett_skrl.brain.aux.with_cltt_ref import WithCLTTRef
    assert getattr(WithCLTTRef, "independent_of_ppo_graph", False) is False
    assert CLTTReferenceStackAuxLoss.independent_of_ppo_graph is True


def _rl_grad_modes(agent, monkeypatch):
    """(grad enabled, aux pass?) per encoder pass. RL passes go through __call__ (a pre-hook sees
    them); the aux's go through encode_prepared, which calls forward() DIRECTLY (no hook), so it is
    wrapped instead."""
    calls = []
    enc = agent.policy.encoder
    enc.register_forward_pre_hook(lambda m, _a: calls.append((torch.is_grad_enabled(), False)))
    inner = enc.encode_prepared

    def encode_prepared(x):
        calls.append((torch.is_grad_enabled(), True))
        return inner(x)

    monkeypatch.setattr(enc, "encode_prepared", encode_prepared)
    return calls


#: An update's RL encoder passes: the no_grad bootstrap value, one per minibatch (actor and critic
#: share the cached output), then two no_grad health-metric passes (track_ppo_health_metrics).
MINIBATCH_PASSES = slice(1, 5)


def test_fix2b_decoupled_rl_pass_runs_without_grad_and_every_gradient_is_unchanged(monkeypatch):
    runs = {}
    for label, env in (("old", [("NETT_DECOUPLE_ENCODER", "1"), (RL_NO_GRAD_ENV, "0")]),
                       ("new", [("NETT_DECOUPLE_ENCODER", "1")])):
        with pytest.MonkeyPatch.context() as mp:
            for k, v in env:
                mp.setenv(k, v)
            agent = _agent(mp)
            assert getattr(agent.policy.encoder, RL_NO_GRAD_ATTR, False) is True
            steps = _record_steps(agent)
            calls = _rl_grad_modes(agent, mp)
            # Only the update's passes: the rollout's act() runs under no_grad either way.
            _rollout_and_update(agent, before_update=calls.clear)
            runs[label] = (steps, list(calls))
    _assert_steps_equal(runs["old"][0], runs["new"][0])
    # POSITIVE CONTROL: the update's minibatch RL passes built a graph on the old path and none on
    # the new; the no_grad passes are no_grad on both; the aux's own passes kept grad on both.
    for label, want in (("old", True), ("new", False)):
        rl = [g for g, aux in runs[label][1] if not aux]
        ax = [g for g, aux in runs[label][1] if aux]
        assert len(rl) == 7, (label, runs[label][1])
        assert all(g is want for g in rl[MINIBATCH_PASSES]), (label, runs[label][1])
        assert not any(rl[:1] + rl[5:]), (label, runs[label][1])
        assert len(ax) == 4 and all(ax), (label, runs[label][1])


def test_fix2b_without_decouple_changes_nothing_and_only_cltt_ref_is_marked(monkeypatch):
    agent = _agent(monkeypatch)
    assert getattr(agent.policy.encoder, RL_NO_GRAD_ATTR, False) is True
    calls = _rl_grad_modes(agent, monkeypatch)
    _rollout_and_update(agent, before_update=calls.clear)
    rl = [g for g, aux in calls if not aux]
    assert len(rl) == 7 and all(rl[MINIBATCH_PASSES])        # RL passes still build the graph
    other = _agent(monkeypatch, aux="vicreg")
    assert getattr(other.policy.encoder, RL_NO_GRAD_ATTR, False) is False
    monkeypatch.setenv("NETT_DECOUPLE_ENCODER", "1")
    monkeypatch.setenv(RL_NO_GRAD_ENV, "perhaps")
    marked = _agent(monkeypatch)
    from nett_skrl.brain.models.utils.features import features_forward
    with pytest.raises(ValueError, match=RL_NO_GRAD_ENV):
        features_forward(marked.policy, {"observations": torch.zeros(1, C * H * W)})
