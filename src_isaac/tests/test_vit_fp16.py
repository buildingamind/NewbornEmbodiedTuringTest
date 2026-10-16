"""fp16 for compact_vit (the ViT-CLTT-Ref encoder) and the GradScaler behind every fp16 encoder knob.

Known answers, each with a control that can fail (everything here is CPU; CUDA fp16 numerics are the
one CUDA-only test at the bottom):
  1. KNOBS. CompactViT reads NETT_AMP at the fp16 spellings ONLY: unset/off/bf16/anything else = fp32,
     as it has always run. NatureCNN's reading of NETT_AMP is unchanged (default bf16; the pre-existing
     fall-through of an unknown value to bf16 is pinned so a change to it is a decision, not drift).
     NETT_MATCHED_AMP gains fp16 and still refuses a typo.
  2. ENCODER. On CPU tensors the ViT forward / encode_tokens / encode_visible_prepared are bit-identical
     whatever NETT_AMP says (autocast is CUDA-gated). With the gate redirected to CPU autocast the fp16
     branch really runs the trunk in float16 (control: a block Linear's output dtype), returns float32,
     and is close to -- but not equal to -- the fp32 result; with the knob unset the redirected gate
     stays shut and the output is bit-identical.
  3. AGENT (Brain -> build_agents, the production path). Unset / off / bf16: skrl's scaler stays
     disabled and the checkpoint key set is unchanged. fp16 on CompactViT (AuxLossPPO with cltt_ref, and
     MetricsPPO without aux) and on NatureCNN: an enabled GradScaler, checkpointed as "grad_scaler".
  4. UPDATE. Under the enabled scaler every gradient the optimizer sees equals the default run's (on
     CPU the encoder is fp32 and the scale is a power of two, so scale -> unscale is exact): this fails
     if unscale_ were skipped (65536x grads) or came after clip_grad_norm_ (a clip at the scaled norm).
     Control: unscale_ ran once per optimizer step. An inf gradient skips that step and halves the
     scale; the scale survives save -> load.
  5. REFUSALS. fp16 on a knob the agent's encoder does not read; fp16 under an agent class that does
     not step through the scaler; fp16 with an aux that owns its optimizer.
  6. PATCH-14 LABELS (DECISIONS §124) for the Unity-recipe++ eye (448x280): -500K-P14 (depth 3), -5M and -50M
     (depth 12, head_dim 64 rule) and -Base (the paper's ViT-B trunk: depth 12, width 768, 12 heads, MLP
     3072), each ViT-CLTT-Ref with only its listed cfg fields moved; encoder and agent counts at 448x280
     equal the MODELS comments; the widths follow the §124 rule (neighbours checked); a CPU forward at
     448x280 gives (B, 512); the fp16 branch reaches them; they refuse 128x80 (80 % 14 != 0) and the
     patch-16 labels refuse 448x280 (280 % 16 != 0).
Run with CUDA_VISIBLE_DEVICES="".
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain import amp as amp_mod
from nett_skrl.brain.amp import SCALER_CHECKPOINT_KEY, configure_fp16_grad_scaler, fp16_requested_knobs
from nett_skrl.brain.encoders import compact_vit as cv
from nett_skrl.brain.encoders.compact_vit import CompactViT, vit_amp_dtype
from nett_skrl.brain.encoders.matched import matched_amp_dtype
from nett_skrl.brain.encoders.nature_cnn import _amp_dtype

KNOBS = ("NETT_AMP", "NETT_MATCHED_AMP", "NETT_AUX_LOSS", "NETT_AUX_WEIGHT", "NETT_AUX_BATCH",
         "NETT_AUX_AFTER_PPO_BACKWARD", "NETT_AUX_STRICT", "NETT_DECOUPLE_ENCODER",
         "NETT_DISABLE_FEATURE_CACHE", "NETT_CLTT_REF_DEDUP")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            yield
    finally:
        torch.set_num_threads(threads)


# ── 1. knobs ────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", [None, "", "off", "0", "none", "fp32", "bf16", "BF16", "bfloat16", "typo"])
def test_vit_reads_fp32_at_every_non_fp16_value(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("NETT_AMP", value)
    assert vit_amp_dtype() is None


@pytest.mark.parametrize("value", ["fp16", "FP16", "float16", "half"])
def test_vit_reads_fp16_at_the_fp16_spellings(monkeypatch, value):
    monkeypatch.setenv("NETT_AMP", value)
    assert vit_amp_dtype() is torch.float16
    assert _amp_dtype() is torch.float16              # one parser: NatureCNN agrees


def test_nature_cnn_reading_of_nett_amp_is_unchanged(monkeypatch):
    assert _amp_dtype() is torch.bfloat16             # default bf16
    for v in ("off", "0", "none", "fp32", "float32", ""):
        monkeypatch.setenv("NETT_AMP", v)
        assert _amp_dtype() is None
    monkeypatch.setenv("NETT_AMP", "bf16")
    assert _amp_dtype() is torch.bfloat16
    monkeypatch.setenv("NETT_AMP", "typo")            # pre-existing fall-through, pinned
    assert _amp_dtype() is torch.bfloat16


def test_matched_amp_gains_fp16_and_still_refuses_a_typo(monkeypatch):
    assert matched_amp_dtype() is torch.bfloat16
    for v in ("fp16", "float16", "half"):
        monkeypatch.setenv("NETT_MATCHED_AMP", v)
        assert matched_amp_dtype() is torch.float16
    monkeypatch.setenv("NETT_MATCHED_AMP", "fp61")
    with pytest.raises(ValueError, match="fp16"):
        matched_amp_dtype()


def test_requested_knobs_are_read_raw(monkeypatch):
    assert fp16_requested_knobs() == []
    monkeypatch.setenv("NETT_AMP", "bf16")
    monkeypatch.setenv("NETT_MATCHED_AMP", "typo")    # not parsed here: a non-matched row stays inert
    assert fp16_requested_knobs() == []
    monkeypatch.setenv("NETT_AMP", "FP16")
    monkeypatch.setenv("NETT_MATCHED_AMP", " half ")
    assert fp16_requested_knobs() == ["NETT_AMP", "NETT_MATCHED_AMP"]


# ── 2. encoder ──────────────────────────────────────────────────────────────────────────────────────

H, W, C = 48, 64, 6


def _vit(**kw):
    torch.manual_seed(0)
    cfg = dict(features_dim=32, patch_size=16, embed_dim=32, depth=2, num_heads=2, **kw)
    return CompactViT(gym.spaces.Box(0, 255, (C, H, W), np.uint8), **cfg).eval()


def _obs(n=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, C, H, W), generator=g).float()


def _passes(enc, x):
    keep = torch.tensor([[0, 2, 5], [1, 3, 4], [0, 1, 2], [3, 4, 5]])
    tokens, grid = enc.encode_tokens(x)
    return {"forward": enc(x), "tokens": tokens,
            "visible": enc.encode_visible_prepared(enc._prepare_image(x), keep)}, grid


@pytest.mark.parametrize("pool", ["cls", "spatial"])
def test_cpu_passes_are_bit_identical_whatever_nett_amp_says(monkeypatch, pool):
    kw = {"pool": pool, "spatial_grid": (3, 2)} if pool == "spatial" else {}
    enc, x = _vit(**kw), _obs()
    ref, _ = _passes(enc, x)
    for v in ("fp16", "bf16", "off"):
        monkeypatch.setenv("NETT_AMP", v)
        out, _ = _passes(enc, x)
        for k in ref:
            assert torch.equal(out[k], ref[k]), (v, k)
    # gradients too, on the default and on the fp16 knob
    grads = []
    for v in (None, "fp16"):
        if v is None:
            monkeypatch.delenv("NETT_AMP", raising=False)
        else:
            monkeypatch.setenv("NETT_AMP", v)
        enc.zero_grad()
        enc(x).square().sum().backward()
        grads.append({k: p.grad.clone() for k, p in enc.named_parameters()})
    assert all(torch.equal(grads[0][k], grads[1][k]) for k in grads[0])


def _cpu_gate(monkeypatch):
    """Redirect the CUDA-only gate to CPU fp16 autocast, still keyed on the knob (vit_amp_dtype)."""
    def gate(x):
        return torch.autocast("cpu", dtype=torch.float16) if cv.vit_amp_dtype() is torch.float16 else None
    monkeypatch.setattr(cv, "_fp16_autocast", gate)


def test_fp16_branch_runs_the_trunk_in_half_and_returns_fp32(monkeypatch):
    enc, x = _vit(), _obs()
    with torch.no_grad():
        ref, grid = _passes(enc, x)
    _cpu_gate(monkeypatch)
    seen = []
    hook = enc.blocks[0].mlp[0].register_forward_hook(lambda m, i, o: seen.append(o.dtype))
    # gate redirected but knob unset: the fp32 path, bit for bit, and no half anywhere
    with torch.no_grad():
        same, _ = _passes(enc, x)
    assert all(torch.equal(same[k], ref[k]) for k in ref) and set(seen) == {torch.float32}
    seen.clear()
    monkeypatch.setenv("NETT_AMP", "fp16")
    with torch.no_grad():
        out, grid16 = _passes(enc, x)
    hook.remove()
    assert seen and set(seen) == {torch.float16}, "the trunk did not run in float16"
    assert grid16 == grid
    for k in ref:
        assert out[k].dtype == torch.float32, k
        assert out[k].shape == ref[k].shape, k
        assert not torch.equal(out[k], ref[k]), f"{k}: fp16 branch did not engage"
        cos = torch.nn.functional.cosine_similarity(out[k].flatten(1), ref[k].flatten(1))
        assert cos.min() > 0.99, (k, float(cos.min()))


def test_fp16_branch_backpropagates_finite_fp32_grads(monkeypatch):
    enc, x = _vit(), _obs()
    _cpu_gate(monkeypatch)
    monkeypatch.setenv("NETT_AMP", "fp16")
    enc.train()
    enc(x).square().mean().backward()
    for k, p in enc.named_parameters():
        assert p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all(), k


def test_encoders_declare_the_knob_they_read(monkeypatch):
    from nett_skrl.brain.encoders.matched import MatchedEncoder
    from nett_skrl.brain.encoders.nature_cnn import NatureCNN
    assert CompactViT.AMP_KNOB == NatureCNN.AMP_KNOB == "NETT_AMP"
    assert MatchedEncoder.AMP_KNOB == "NETT_MATCHED_AMP"
    enc = _vit()
    assert enc.amp_dtype() is None
    monkeypatch.setenv("NETT_AMP", "fp16")
    assert enc.amp_dtype() is torch.float16


# ── 3-5. agent ──────────────────────────────────────────────────────────────────────────────────────

class _AgentEnv:
    num_envs = 1
    num_agents = 1
    observation_space = gym.spaces.Box(0, 255, (C, H, W), np.uint8)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    state_space = None


_VIT_CFG = {"features_dim": 32, "patch_size": 16, "embed_dim": 32, "depth": 1, "num_heads": 2}
_CNN_CFG = {"features_dim": 32}


def _agent(mp, *, encoder="compact_vit", cfg=_VIT_CFG, aux="cltt_ref", env=(), seed=0):
    from nett_skrl.brain.agent_factory import build_agents
    from nett_skrl.brain.brain import Brain
    for k, v in env:
        mp.setenv(k, v)
    if aux is not None:
        mp.setenv("NETT_AUX_LOSS", aux)
        mp.setenv("NETT_AUX_WEIGHT", "1")
        mp.setenv("NETT_AUX_BATCH", "6")
    torch.manual_seed(seed)
    brain = Brain(encoder=encoder, algorithm="PPO", encoder_cfg=dict(cfg),
                  algorithm_cfg={"rollouts": 16, "mini_batches": 2, "learning_epochs": 2,
                                 "grad_norm_clip": 0.5, "learning_rate": 1e-3},
                  model={"shared_encoder": True, "hidden_sizes": [16]})
    agent = build_agents(brain, _AgentEnv(), torch.device("cpu"))[0]
    agent.init(trainer_cfg=None)
    agent.enable_training_mode(True)
    return agent


def _rollout_and_update(agent, seed=1, steps=16):
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


def _record_steps(agent):
    """Every gradient the optimizer sees at each executed step, by optimizer slot."""
    params = [p for group in agent.optimizer.param_groups for p in group["params"]]
    steps = []
    agent.optimizer.register_step_pre_hook(
        lambda *_: steps.append([None if p.grad is None else p.grad.detach().clone() for p in params]))
    return steps


_CASES = {"vit+cltt_ref": dict(), "vit": dict(aux=None), "cnn+cltt_ref": dict(encoder="nature_cnn", cfg=_CNN_CFG)}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_default_off_and_bf16_agents_are_unchanged(case):
    keys = []
    for env in ((), (("NETT_AMP", "off"),), (("NETT_AMP", "bf16"),), (("NETT_MATCHED_AMP", "bf16"),)):
        with pytest.MonkeyPatch.context() as mp:
            agent = _agent(mp, env=env, **_CASES[case])
        assert not agent.scaler.is_enabled()
        assert SCALER_CHECKPOINT_KEY not in agent.checkpoint_modules
        keys.append(list(agent.checkpoint_modules))
    assert all(k == keys[0] for k in keys)


@pytest.mark.parametrize("case,cls", [("vit+cltt_ref", "AuxLossPPO"), ("vit", "MetricsPPO"),
                                      ("cnn+cltt_ref", "AuxLossPPO")])
def test_fp16_enables_the_scaler_and_checkpoints_it(case, cls):
    with pytest.MonkeyPatch.context() as mp:
        base = list(_agent(mp, **_CASES[case]).checkpoint_modules)
        agent = _agent(mp, env=[("NETT_AMP", "fp16")], **_CASES[case])
        assert type(agent).__name__ == cls
        assert isinstance(agent.scaler, torch.amp.GradScaler) and agent.scaler.is_enabled()
        assert agent.checkpoint_modules[SCALER_CHECKPOINT_KEY] is agent.scaler
        assert list(agent.checkpoint_modules) == base + [SCALER_CHECKPOINT_KEY]
        scaler = agent.scaler
        assert configure_fp16_grad_scaler(agent) is True and agent.scaler is scaler   # idempotent


@pytest.mark.parametrize("case", ["vit+cltt_ref", "vit"])
def test_fp16_update_sees_the_default_gradients_and_unscales_every_step(case):
    runs = []
    for env in ((), (("NETT_AMP", "fp16"),)):
        with pytest.MonkeyPatch.context() as mp:
            agent = _agent(mp, env=env, **_CASES[case])
            steps = _record_steps(agent)
            calls = {"unscale_": 0}
            inner = agent.scaler.unscale_

            def unscale_(opt, inner=inner, calls=calls):
                calls["unscale_"] += 1
                return inner(opt)

            mp.setattr(agent.scaler, "unscale_", unscale_)
            _rollout_and_update(agent)
            runs.append((agent, steps, calls["unscale_"]))
    (a0, s0, _), (a1, s1, n_unscale) = runs
    assert len(s0) == len(s1) == 4                    # 2 epochs x 2 minibatches, none skipped
    assert n_unscale == 4                             # control: the scaler path ran at every step
    assert a1.scaler.get_scale() == 65536.0           # no overflow, no back-off
    for i, (g0, g1) in enumerate(zip(s0, s1)):
        for j, (x0, x1) in enumerate(zip(g0, g1)):
            if x0 is None or x1 is None:
                assert x0 is None and x1 is None, (i, j)
            else:
                torch.testing.assert_close(x1, x0, rtol=1e-6, atol=1e-9, msg=lambda m, i=i, j=j: f"step {i} p{j}: {m}")
    for (k, p0), (_, p1) in zip(a0.policy.named_parameters(), a1.policy.named_parameters()):
        torch.testing.assert_close(p1, p0, rtol=1e-6, atol=1e-7, msg=lambda m, k=k: f"{k}: {m}")


def _inject_inf_on_first_unscale(mp, agent):
    inner, state = agent.scaler.unscale_, {"n": 0}

    def unscale_(opt):
        if state["n"] == 0:
            p = next(p for p in agent.policy.parameters() if p.grad is not None)
            p.grad.view(-1)[0] = float("inf")
        state["n"] += 1
        return inner(opt)

    mp.setattr(agent.scaler, "unscale_", unscale_)


@pytest.mark.parametrize("case", ["vit+cltt_ref", "vit"])
def test_an_inf_gradient_skips_the_step_and_backs_off_the_scale(case, tmp_path):
    with pytest.MonkeyPatch.context() as mp:
        agent = _agent(mp, env=[("NETT_AMP", "fp16")], **_CASES[case])
        steps = _record_steps(agent)
        _inject_inf_on_first_unscale(mp, agent)
        _rollout_and_update(agent)
        assert len(steps) == 3                        # the inf step was skipped, the other three ran
        assert agent.scaler.get_scale() == 32768.0    # backoff_factor 0.5
        assert all(torch.isfinite(p).all() for p in agent.policy.parameters())
        path = tmp_path / "agent.pt"
        agent.save(str(path))
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved[SCALER_CHECKPOINT_KEY]["scale"] == 32768.0
        fresh = _agent(mp, env=[("NETT_AMP", "fp16")], **_CASES[case])
        assert fresh.scaler.get_scale() == 65536.0
        fresh.load(str(path))
        assert fresh.scaler.get_scale() == 32768.0    # the loss scale survives a resume
        mp.setenv("NETT_AMP", "off")                  # a non-fp16 agent still loads an fp16 checkpoint
        plain = _agent(mp, **_CASES[case])
        plain.load(str(path))
        assert not plain.scaler.is_enabled()


def test_without_the_scaler_the_same_inf_corrupts_the_weights():
    """Power check for the test above: the skip comes from the scaler, not from the harness."""
    with pytest.MonkeyPatch.context() as mp:
        agent = _agent(mp, aux=None)
        steps = _record_steps(agent)
        inner = agent.scaler.unscale_                 # disabled scaler: unscale_ is a no-op
        state = {"n": 0}

        def unscale_(opt):
            if state["n"] == 0:
                p = next(p for p in agent.policy.parameters() if p.grad is not None)
                p.grad.view(-1)[0] = float("inf")
            state["n"] += 1
            return inner(opt)

        mp.setattr(agent.scaler, "unscale_", unscale_)
        _rollout_and_update(agent)
        assert len(steps) == 4
        assert not all(torch.isfinite(p).all() for p in agent.policy.parameters())


@pytest.mark.parametrize("encoder,cfg,env,knob", [
    ("compact_3dcnn", {"features_dim": 32, "num_frames": 2, "conv_dim": 8}, [("NETT_AMP", "fp16")], "NETT_AMP"),
    ("compact_vit", _VIT_CFG, [("NETT_MATCHED_AMP", "fp16")], "NETT_MATCHED_AMP"),
    ("compact_vit", _VIT_CFG, [("NETT_AMP", "fp16"), ("NETT_MATCHED_AMP", "fp16")], "NETT_MATCHED_AMP"),
])
def test_fp16_on_a_knob_the_encoder_does_not_read_refuses(encoder, cfg, env, knob):
    with pytest.MonkeyPatch.context() as mp:
        with pytest.raises(ValueError, match=f"{knob}=.*does not read {knob}"):
            _agent(mp, encoder=encoder, cfg=cfg, aux=None, env=env)


def test_fp16_matched_knob_on_a_matched_encoder_is_accepted_and_nett_amp_fp16_refused(monkeypatch):
    from nett_skrl.brain.encoders.matched import EYE_HW, MatchedEncoder

    class _Agent:
        device = "cpu"

    agent = _Agent()
    agent.models = {"policy": MatchedEncoder(gym.spaces.Box(0, 255, (3, *EYE_HW), np.uint8), variant="cnn")}
    monkeypatch.setenv("NETT_AMP", "fp16")
    with pytest.raises(ValueError, match="does not read NETT_AMP"):
        configure_fp16_grad_scaler(agent)
    monkeypatch.delenv("NETT_AMP")
    monkeypatch.setenv("NETT_MATCHED_AMP", "fp16")
    with pytest.raises(ValueError, match="only MetricsPPO and AuxLossPPO"):   # knob accepted; class refused
        configure_fp16_grad_scaler(agent)


def test_fp16_refuses_an_agent_class_without_the_scaler_path_and_an_owned_aux_optimizer(monkeypatch):
    from nett_skrl.brain.aux.ppo_aux import AuxLossPPO

    class _Plain:
        device = "cpu"

    plain = _Plain()
    plain.models = {"policy": _vit()}
    assert configure_fp16_grad_scaler(plain) is False          # knob unset: untouched, no refusal
    monkeypatch.setenv("NETT_AMP", "fp16")
    with pytest.raises(ValueError, match="only MetricsPPO and AuxLossPPO"):
        configure_fp16_grad_scaler(plain)

    owned = object.__new__(AuxLossPPO)
    owned.__dict__.update(models={"policy": _vit()}, device="cpu", checkpoint_modules={},
                          _aux=type("XSP2Like", (), {"owns_optimizer": True})(), _aux_kind="xsp2")
    with pytest.raises(ValueError, match="own optimizer"):
        configure_fp16_grad_scaler(owned)


def test_amp_encoders_finds_a_nested_encoder_once():
    enc = _vit()
    shared = torch.nn.Sequential(enc)

    class _A:
        models = {"policy": torch.nn.Sequential(shared), "value": shared}

    assert amp_mod.amp_encoders(_A()) == [enc]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast engages on CUDA only")
def test_cuda_fp16_forward_is_close_not_identical_and_fp32_out(monkeypatch):
    enc, x = _vit(), _obs()
    enc, x = enc.cuda(), x.cuda()
    with torch.no_grad():
        ref = enc(x)
        monkeypatch.setenv("NETT_AMP", "fp16")
        out = enc(x)
    assert out.dtype == torch.float32 and not torch.equal(out, ref)
    assert torch.nn.functional.cosine_similarity(out, ref).min() > 0.99


# ── Patch-14 ViT-CLTT-Ref labels at the Unity-recipe++ eye (DECISIONS §124): -500K-P14 / -5M / -50M / -Base ──

EYE_H, EYE_W = 280, 448               # NETT_EYE_RES=448x280 is W x H
BASE_LABEL = "ViT-CLTT-Ref-Base"
# label -> (cfg fields moved off ViT-CLTT-Ref, encoder count at 448x280 x 6 ch, (depth, width, heads, mlp hidden)).
# The counts are also written in campaign_train.py's MODELS comments; checked against them below.
P14 = {
    "ViT-CLTT-Ref-500K-P14": ({"patch_size": 14, "embed_dim": 104}, 506_160, (3, 104, 4, 208)),
    "ViT-CLTT-Ref-5M": ({"patch_size": 14, "depth": 12, "embed_dim": 177, "num_heads": 3, "mlp_ratio": 4.0},
                        4_952_441, (12, 177, 3, 708)),
    "ViT-CLTT-Ref-50M": ({"patch_size": 14, "depth": 12, "embed_dim": 576, "num_heads": 9, "mlp_ratio": 4.0},
                         49_209_920, (12, 576, 9, 2304)),
    BASE_LABEL: ({"patch_size": 14, "depth": 12, "embed_dim": 768, "num_heads": 12, "mlp_ratio": 4.0},
                 86_846_720, (12, 768, 12, 3072)),
}
TARGETS = {"ViT-CLTT-Ref-500K-P14": 500_000, "ViT-CLTT-Ref-5M": 5_000_000, "ViT-CLTT-Ref-50M": 50_000_000}


@pytest.fixture()
def campaign(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train
    return campaign_train


def _label_encoder(campaign, label, h, w):
    from nett_skrl.brain.config import EncoderCfg
    from nett_skrl.brain.registry import encoder_mapping
    spec = campaign.MODELS[label]
    kw = EncoderCfg(**spec["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    ch = 3 * (campaign._FRAMESTACK_N if spec["framestack"] else 1)
    torch.manual_seed(0)
    return encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (ch, h, w), np.uint8), **kw)


def _declared(campaign, label):
    """(enc, agent) from the label's MODELS line comment in campaign_train.py, which must name the 448x280 eye."""
    import re
    from pathlib import Path
    src = Path(campaign.__file__).read_text()
    line = next(ln for ln in src.splitlines() if ln.lstrip().startswith(f'"{label}":'))
    m = re.search(r"enc ([\d,]+) / agent ([\d,]+) @ 448x280", line)
    assert m, line
    return int(m[1].replace(",", "")), int(m[2].replace(",", ""))


@pytest.mark.parametrize("label", sorted(P14))
def test_p14_label_is_vit_cltt_ref_with_only_its_fields_moved(campaign, label):
    lab, ref = campaign.MODELS[label], campaign.MODELS["ViT-CLTT-Ref"]
    assert {k: v for k, v in lab.items() if k != "cfg"} == {k: v for k, v in ref.items() if k != "cfg"}
    diff = {k: lab["cfg"][k] for k in set(lab["cfg"]) | set(ref["cfg"]) if lab["cfg"].get(k) != ref["cfg"].get(k)}
    assert diff == P14[label][0]
    assert lab["cfg"]["pool"] == "cls" and lab["cfg"]["stem"] == "linear" and lab["cfg"]["features_dim"] == 512
    assert lab["encoder"] == "compact_vit" and lab["framestack"] is True and lab["aux"] == "cltt_ref"


@pytest.mark.parametrize("label", sorted(P14))
def test_p14_encoder_shape_and_count_at_448x280_match_the_comment(campaign, label):
    enc = _label_encoder(campaign, label, EYE_H, EYE_W)
    n = sum(p.numel() for p in enc.parameters())
    assert n == P14[label][1]
    assert _declared(campaign, label) == (n, n + 1541)
    depth, width, heads, hidden = P14[label][2]
    assert len(enc.blocks) == depth
    for blk in enc.blocks:
        assert blk.attn.num_heads == heads and blk.attn.embed_dim == width
        fc1, act, fc2 = blk.mlp
        assert (fc1.in_features, fc1.out_features, fc2.out_features) == (width, hidden, width)
        assert isinstance(act, torch.nn.GELU) and isinstance(blk.norm1, torch.nn.LayerNorm)   # pre-norm, GELU
    assert enc.patch_embed.kernel_size == (14, 14) and enc.patch_embed.in_channels == 6
    assert enc.pos_embed.shape == (1, 20 * 32 + 1, width)      # 20 x 32 patches + CLS at 448x280
    # the cltt_ref head is the same constant 329,216 as on every ViT-CLTT-Ref label (built on features_dim)
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss
    shared = {id(p) for p in enc.parameters()}
    assert sum(p.numel() for p in CLTTReferenceAuxLoss(enc).parameters() if id(p) not in shared) == 329_216


def test_p14_sizes_follow_the_section_124_rule(campaign):
    """500K-P14: the multiple of 4 nearest 500K. 5M/50M: depth 12, head_dim 64 if a multiple of 64 lands within
    +-10% (50M: 576), else the head count whose head_dim is nearest 64 and its multiple nearest the target (5M:
    no multiple of 64 within 10%, so 3 heads, width 177). Controls: the neighbouring widths really are farther."""
    from nett_skrl.brain.encoders.compact_vit import CompactViT

    def count(**over):
        cfg = {**campaign.VIT_CFG, "patch_size": 14, **over}
        cfg.pop("trainable")
        with torch.device("meta"):
            enc = CompactViT(gym.spaces.Box(0, 255, (6, EYE_H, EYE_W), np.uint8), **cfg)
        return sum(p.numel() for p in enc.parameters())

    d12 = dict(depth=12, mlp_ratio=4.0)
    err = lambda n, t: abs(n - t) / t
    assert err(P14["ViT-CLTT-Ref-500K-P14"][1], 5e5) < min(err(count(embed_dim=w), 5e5) for w in (100, 108))
    assert err(P14["ViT-CLTT-Ref-50M"][1], 5e7) < 0.10
    assert all(err(count(embed_dim=w, num_heads=w // 64, **d12), 5e7) > 0.10 for w in (512, 640))
    assert all(err(count(embed_dim=w, num_heads=w // 64, **d12), 5e6) > 0.10 for w in (128, 192))
    assert count(embed_dim=192, num_heads=3, **d12) == 5_786_816                  # the DeiT-Ti trunk: +15.74%
    assert err(P14["ViT-CLTT-Ref-5M"][1], 5e6) < min(err(count(embed_dim=w, num_heads=3, **d12), 5e6) for w in (174, 180))
    for label, target in TARGETS.items():
        assert err(P14[label][1], target) < 0.02, label


@pytest.mark.parametrize("label", sorted(P14))
def test_p14_forward_at_448x280_on_cpu(campaign, label):
    enc = _label_encoder(campaign, label, EYE_H, EYE_W).eval()
    x = torch.randint(0, 256, (2, 6, EYE_H, EYE_W), generator=torch.Generator().manual_seed(1)).float()
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (2, 512) and out.dtype == torch.float32 and torch.isfinite(out).all()
    assert not torch.equal(out[0], out[1])


@pytest.mark.parametrize("label", sorted(P14))
def test_p14_labels_refuse_the_128x80_campaign_eye(campaign, label):
    with pytest.raises(AssertionError, match="divisible by patch_size=14"):
        _label_encoder(campaign, label, 80, 128)


@pytest.mark.parametrize("label", ["ViT-CLTT-Ref", "ViT-CLTT-Ref-500K"])
def test_patch16_vit_labels_refuse_the_448x280_eye(campaign, label):
    """The Unity-recipe++ eye (448x280, W x H) is not a multiple of patch 16 in height (280 = 17.5 x 16)."""
    with pytest.raises(AssertionError, match="divisible by patch_size=16"):
        _label_encoder(campaign, label, EYE_H, EYE_W)


def test_fp16_reaches_the_p14_encoders(campaign, monkeypatch):
    """CPU fp16 autocast of the 5M trunk at 448x280 (the larger labels are the same class reading the same knob)."""
    enc = _label_encoder(campaign, "ViT-CLTT-Ref-5M", EYE_H, EYE_W).eval()
    x = torch.randint(0, 256, (1, 6, EYE_H, EYE_W), generator=torch.Generator().manual_seed(1)).float()
    with torch.no_grad():
        ref = enc(x)
    _cpu_gate(monkeypatch)
    monkeypatch.setenv("NETT_AMP", "fp16")
    assert enc.amp_dtype() is torch.float16
    seen = []
    hook = enc.blocks[-1].mlp[0].register_forward_hook(lambda m, i, o: seen.append(o.dtype))
    with torch.no_grad():
        out = enc(x)
    hook.remove()
    assert seen == [torch.float16] and out.dtype == torch.float32 and not torch.equal(out, ref)
    assert torch.nn.functional.cosine_similarity(out, ref).min() > 0.98
    with torch.device("meta"):
        base = _label_encoder(campaign, BASE_LABEL, EYE_H, EYE_W)
    assert isinstance(base, CompactViT) and base.amp_dtype() is torch.float16


class _Env448:
    num_envs = 1
    num_agents = 1
    observation_space = gym.spaces.Box(0, 255, (6, EYE_H, EYE_W), np.uint8)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    state_space = None


@pytest.mark.parametrize("label", ["ViT-CLTT-Ref-500K-P14", BASE_LABEL])
def test_p14_agent_count_at_448x280_matches_the_comment_and_fp16_enables_the_scaler(campaign, monkeypatch, label):
    """The production path at the label's own cfg: Brain -> build_agents -> AuxLossPPO(cltt_ref), 448x280."""
    from nett_skrl.brain.agent_factory import build_agents
    from nett_skrl.brain.brain import Brain

    for k, v in (("NETT_AMP", "fp16"), ("NETT_AUX_LOSS", "cltt_ref"), ("NETT_AUX_WEIGHT", "1")):
        monkeypatch.setenv(k, v)
    spec = campaign.MODELS[label]
    torch.manual_seed(0)
    brain = Brain(encoder=spec["encoder"], algorithm="PPO", encoder_cfg=dict(spec["cfg"]),
                  algorithm_cfg={"rollouts": 4},
                  model={"value_bound": None, "shared_encoder": True, "hidden_sizes": [], "clip_actions": False})
    agent = build_agents(brain, _Env448(), torch.device("cpu"))[0]
    params = {id(p): p for m in (agent.policy, agent.value) for p in m.parameters()}
    assert sum(p.numel() for p in params.values()) == _declared(campaign, label)[1]
    assert type(agent).__name__ == "AuxLossPPO" and agent.scaler.is_enabled()
    assert agent.checkpoint_modules[SCALER_CHECKPOINT_KEY] is agent.scaler
