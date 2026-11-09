"""BL2 "<trunk>+CLTT-Ref+UNIF" (owner 2026-11-09): attached CLTT-Ref + the EMBDIV uniformity term on one encoder.

  <trunk>+CLTT-Ref+UNIF = MODELS[<CLTT-Ref parent>] with aux "cltt_ref+embdiv_unif" (brain/aux/cltt_unif_aux.py):
  total = PPO + 1.0 x L_cltt_ref + lambda x L_unif, lambda = NETT_AUX_EMBDIV_WEIGHT, aux_weight 1.0 (refused otherwise).

Pinned: (a) the composite loss is exactly L_cltt + lambda x L_unif of separately built cltt_ref / embdiv_unif terms on
the same RNG stream in the same order; (b) its encoder and CLTT-head gradients are the sum of the two terms'; (c) through
the production agent, PPO updates write embdiv_diag.csv in the "+UNIF" form with aux_weight = lambda, the CLTT head is in
the optimizer exactly once, the aux runs after the PPO backward and an update equals the pre-§111 (retain_graph) path;
(d) each of the four labels builds at BL2's 128x80 eye with a finite forward and aux loss (param counts printed); (e)
every refusal; (f) two same-seed runs agree exactly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "examples"))
sys.path.insert(0, str(_SRC / "tests"))

import campaign_train as ct  # noqa: E402
from nett_skrl.brain.aux import cltt_unif_aux as cu  # noqa: E402
from nett_skrl.brain.aux import embdiv_aux as ed  # noqa: E402
from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss  # noqa: E402
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES  # noqa: E402
from nett_skrl.brain.config import EncoderCfg  # noqa: E402
from nett_skrl.brain.models.utils.features import RL_NO_GRAD_ATTR  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402
import test_embdiv_aux as base  # noqa: E402

KEY = "cltt_ref+embdiv_unif"
H, W = 80, 128                      # BL2's eye (NETT_RES=128)
LABELS = {"ViT-P8-666K+CLTT-Ref+UNIF": ("ViT-P8-666K+CLTT-Ref", "CompactViT"),
          "ViViT-P8-T2+CLTT-Ref+UNIF": ("ViViT-P8-T2+CLTT-Ref", "CompactViViT"),
          "CNN-D512P+CLTT-Ref+UNIF": ("CNN-D512P+CLTT-Ref", "NatureCNN"),
          "3DCNN-CLTT-Ref+UNIF": ("3DCNN-CLTT-Ref", "Compact3DCNN")}
#: CLTT-Ref's projection head on a 512-feature encoder (campaign_train: "the cltt_ref head adds 329,216").
CLTT_HEAD_PARAMS = 329_216
_KNOBS = (*base._KNOBS, cu.UNIF_WEIGHT_ENV, "NETT_AUX_CLTT_REF_OFFSETS", "NETT_AUX_CLTT_REF_TEMP",
          "NETT_AUX_CLTT_REF_DIAG", "NETT_AUX_CLTT_MASK_POSITIVE_TWIN", "NETT_CLTT_REF_DEDUP", "NETT_AUX_CLTT_REF_WEIGHT",
          "NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_ENCAUX_LR_SCALE", "NETT_RL_ENCODER_NO_GRAD", "NETT_FRAMESTACK_N")


@pytest.fixture(autouse=True)
def _clean_knobs(monkeypatch):
    for k in _KNOBS:
        monkeypatch.delenv(k, raising=False)


def _channels(spec):
    return 3 * ct._FRAMESTACK_N if spec["framestack"] else 3


def _space(spec, h=H, w=W):
    return gym.spaces.Box(low=0, high=255, shape=(_channels(spec), h, w), dtype=np.uint8)


def _encoder(label, seed=0):
    spec = ct.MODELS[label]
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](_space(spec), **kwargs)


class _Memory:
    """A rollout ring as cltt_ref reads it: flattened uint8 frames + both boundary flags, no resets."""

    def __init__(self, n_flat, t_max=12, n_env=2, seed=3):
        g = torch.Generator().manual_seed(seed)
        self.tensors = {"observations": torch.randint(0, 256, (t_max, n_env, n_flat), generator=g, dtype=torch.uint8),
                        "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
                        "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool)}
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


def _flat(spec):
    return _channels(spec) * H * W


def _batch(spec, n=10, seed=1):
    return torch.randint(0, 256, (n, _channels(spec), H, W), dtype=torch.uint8,
                         generator=torch.Generator().manual_seed(seed))


def _small(monkeypatch, lam="0.003"):
    monkeypatch.setenv("NETT_AUX_BATCH", "6")
    monkeypatch.setenv(ed.BATCH_ENV, "5")
    monkeypatch.setenv(cu.UNIF_WEIGHT_ENV, lam)


def _grads(*modules):
    out = []
    for m in modules:
        for p in m.parameters():
            out.append(torch.zeros_like(p).flatten() if p.grad is None else p.grad.detach().clone().flatten())
    return torch.cat(out)


# ---------------------------------------------------------------- registry, wiring

def test_key_is_registered_and_the_composite_is_the_two_existing_terms():
    enc = _encoder("ViT-P8-666K+CLTT-Ref+UNIF")
    aux = AUX_LOSSES[KEY](enc)
    assert isinstance(aux, cu.CLTTRefUnifAux)
    assert type(aux.cltt) is CLTTReferenceAuxLoss and type(aux.unif) is ed.EmbDivTerm
    assert aux.unif.key == "embdiv_unif" and aux.unif.form == "uniformity" and aux.unif.t == 2.0
    assert aux.unif_weight == 0.01 and aux.unif.diag_weight == 0.01          # default lambda: buildable without env
    assert aux.head is aux.cltt.head                                         # ONE head: CLTT's
    assert [id(p) for p in aux.head.parameters()] == [id(p) for p in aux.cltt.head.parameters()]
    assert sum(p.numel() for p in aux.head.parameters()) == CLTT_HEAD_PARAMS
    assert list(aux.unif.parameters()) == []
    assert aux.needs_memory is True and aux.independent_of_ppo_graph is True
    enc_ids = {id(p) for p in enc.parameters()}
    assert not any(id(p) in enc_ids for p in aux.head.parameters())
    # the existing +UNIF term keeps diag_weight None (its CSV still reports AuxLossPPO's weight)
    assert AUX_LOSSES["embdiv_unif"](enc).diag_weight is None


# ---------------------------------------------------------------- (a) + (b) exact sum, gradient sum

def _pair(label, monkeypatch, lam):
    """(composite, separate cltt_ref, separate embdiv_unif) on ONE encoder, heads identically initialised."""
    _small(monkeypatch, lam)
    spec = ct.MODELS[label]
    enc = _encoder(label)
    mem = _Memory(_flat(spec))
    torch.manual_seed(7)
    comp = AUX_LOSSES[KEY](enc)
    torch.manual_seed(7)
    ref = CLTTReferenceAuxLoss(enc)
    uni = ed.EmbDivTerm(enc, "embdiv_unif")
    comp.attach_memory(mem)
    ref.attach_memory(mem)
    for a, b in zip(comp.head.state_dict().values(), ref.head.state_dict().values()):
        assert torch.equal(a, b)
    return spec, enc, comp, ref, uni


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("lam", ["0.0005", "0.003"])
def test_loss_is_exactly_cltt_plus_lambda_unif_and_gradients_sum(monkeypatch, label, lam):
    spec, enc, comp, ref, uni = _pair(label, monkeypatch, lam)
    obs = _batch(spec)
    lam_f = float(lam)

    enc.zero_grad(set_to_none=True)
    comp.head.zero_grad(set_to_none=True)
    torch.manual_seed(21)
    loss_c = comp.compute(enc, obs)
    loss_c.backward()
    g_comp = _grads(enc, comp.head)

    enc.zero_grad(set_to_none=True)
    torch.manual_seed(21)
    l_ref = ref.compute(enc, obs)          # same order as the composite: CLTT first, then the uniformity term
    l_uni = uni.compute(enc, obs)
    # (a) exact: the same tensors, the same op
    assert torch.equal(loss_c.detach(), (l_ref + lam_f * l_uni).detach()), (float(loss_c), float(l_ref), float(l_uni))
    assert torch.isfinite(loss_c) and float(l_uni) < 0.0                     # uniformity is <= 0
    # (b) the two terms' gradients, accumulated separately
    l_ref.backward()
    g_ref_only = _grads(enc, ref.head)
    (lam_f * l_uni).backward()
    g_sum = _grads(enc, ref.head)
    assert torch.allclose(g_comp, g_sum, rtol=1e-5, atol=1e-8), float((g_comp - g_sum).abs().max())
    n_enc = sum(p.numel() for p in enc.parameters())
    assert not torch.equal(g_sum[:n_enc], g_ref_only[:n_enc]), "the uniformity term put no gradient on the encoder"
    assert torch.equal(g_sum[n_enc:], g_ref_only[n_enc:]), "the uniformity term reached the CLTT head"
    assert float(g_comp[n_enc:].abs().sum()) > 0.0
    # telemetry: CLTT's names unchanged, the uniformity term's under names that cannot collide
    s = comp.last_scalars
    assert {k: v for k, v in s.items() if k not in cu.OWN_SCALARS} == ref.last_scalars
    assert s["cltt_ref_loss"] == pytest.approx(float(l_ref)) and s["unif_loss"] == pytest.approx(float(l_uni))
    assert s["unif_B"] == 5.0 and s["unif_weight"] == lam_f


# ---------------------------------------------------------------- (c) the production agent path

def _small_agent(monkeypatch, tmp_path, lam, label="ViT-P8-666K+CLTT-Ref+UNIF"):
    monkeypatch.chdir(tmp_path)
    spec = ct.MODELS[label]
    base._export_like_main(monkeypatch, spec)
    monkeypatch.setenv(ed.BATCH_ENV, "8")
    monkeypatch.setenv("NETT_AUX_BATCH", "8")
    monkeypatch.setenv(cu.UNIF_WEIGHT_ENV, lam)
    agent = base._production_agent(spec, base._EnvN, {"rollouts": base.STEPS * base.NENV, "mini_batches": 2,
                                                      "learning_epochs": 2, "learning_rate": 1e-3}, dict(base._SMALL_VIT))
    agent.init(trainer_cfg=None)
    agent.enable_training_mode(True)
    agent.experiment_dir = str(tmp_path / "exp" / "brain_1")
    return agent


@pytest.mark.parametrize("lam", ["0.0005", "0.003"])
def test_production_updates_write_the_unif_diag_at_lambda_and_one_cltt_head(monkeypatch, tmp_path, capsys, lam):
    from nett_skrl.brain.aux.ppo_aux import AuxLossPPO
    agent = _small_agent(monkeypatch, tmp_path, lam)
    lam_f = float(lam)
    assert isinstance(agent, AuxLossPPO) and agent._aux_kind == KEY and agent._aux_weight == 1.0
    aux = agent._aux
    assert isinstance(aux, cu.CLTTRefUnifAux) and aux.unif_weight == lam_f
    assert aux.cltt._memory is agent.memory                                  # needs_memory -> attach_memory -> CLTT
    # the CLTT head is in the optimizer exactly once, as its own (last) group, and nothing else repeats
    ids = [id(p) for g in agent.optimizer.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids))
    assert [id(p) for p in agent.optimizer.param_groups[-1]["params"]] == [id(p) for p in aux.cltt.head.parameters()]
    # §111 fix 2 / 2b: the aux runs after the PPO backward; the encoder carries the decoupled-no-grad mark
    assert agent._aux_after_ppo_backward() is True
    assert getattr(agent.policy.encoder, RL_NO_GRAD_ATTR, False) is True
    tracked = []
    real_track = agent.track_data
    monkeypatch.setattr(agent, "track_data", lambda tag, value: (tracked.append((tag, value)), real_track(tag, value)))
    base._rollout_and_update(agent, seed=1)
    base._rollout_and_update(agent, seed=2, t0=base.STEPS)
    rows = base._read_csv(agent)
    assert tuple(rows[0]) == ed.DIAG_COLUMNS and [r[1] for r in rows[1:]] == ["0", "1"]
    for row in rows[1:]:
        rec = dict(zip(rows[0], row))
        assert rec["form"] == "uniformity" and float(rec["t"]) == 2.0 and rec["B"] == "8"
        assert float(rec["aux_weight"]) == lam_f
        assert float(rec["loss_weighted"]) == pytest.approx(lam_f * float(rec["loss_raw"]), rel=1e-12)
        assert float(rec["aux_share"]) == pytest.approx(abs(float(rec["loss_weighted"])) / abs(float(rec["ppo_total"])))
        for k in (*ed.DIAG_METRICS, *ed.SIGNAL_METRICS):
            if k != "episode_return_recent":
                assert math.isfinite(float(rec[k])), (k, rec[k])
    tags = {}
    for tag, value in tracked:
        tags.setdefault(tag, []).append(value)
    # CLTT's tfevents under their CLTTA names, the uniformity term's beside them, the per-update diag as a +UNIF arm
    for k in ("B", "t_max", "chance", "mask_regime", "cltt_ref_loss", "unif_loss", "unif_B", "unif_weight"):
        assert len(tags[f"Loss / Aux {KEY} {k}"]) == 2, k
    assert tags[f"Loss / Aux {KEY} unif_weight"] == [lam_f, lam_f]
    rec0 = dict(zip(rows[0], rows[1]))
    assert tags[f"Loss / Aux {KEY} unif_loss"][0] == pytest.approx(float(rec0["loss_raw"]))
    total = tags[f"Loss / Aux ({KEY}) loss"][0]
    assert total == pytest.approx(tags[f"Loss / Aux {KEY} cltt_ref_loss"][0]
                                  + lam_f * tags[f"Loss / Aux {KEY} unif_loss"][0], rel=1e-5)
    for k in (*ed.DIAG_METRICS, "aux_share"):
        assert tags[f"Loss / Aux {KEY} diag_{k}"][0] == pytest.approx(float(rec0[k]))
    assert f"[NETT embdiv] brain_1 update 0 timestep {base.STEPS} {KEY} B 8" in capsys.readouterr().out


def _params_after_one_update(monkeypatch, tmp_path, after):
    monkeypatch.setenv("NETT_AUX_AFTER_PPO_BACKWARD", after)
    (tmp_path / f"after{after}").mkdir()
    agent = _small_agent(monkeypatch, tmp_path / f"after{after}", "0.003")
    base._rollout_and_update(agent, seed=4)
    return [p.detach().clone() for p in agent.policy.parameters()] + \
           [p.detach().clone() for p in agent._aux.head.parameters()]


def test_after_ppo_backward_update_equals_the_retain_graph_path(monkeypatch, tmp_path):
    """Graph independence, measured: the §111 fix-2 path (aux forward+backward after the PPO backward, no retain_graph)
    gives the same update as the old fused-forward split backward. If the uniformity term shared a graph node with the
    PPO forward, fix 2 would raise (graph freed) or differ."""
    a = _params_after_one_update(monkeypatch, tmp_path, "1")
    b = _params_after_one_update(monkeypatch, tmp_path, "0")
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, rtol=1e-5, atol=1e-7), float((x - y).abs().max())


# ---------------------------------------------------------------- (d) the four labels at BL2's eye

@pytest.mark.parametrize("label", LABELS)
def test_label_builds_at_128x80_with_a_finite_forward_and_aux_loss(monkeypatch, capsys, label):
    from nett_skrl.brain.aux.ppo_aux import AuxLossPPO
    parent, trunk = LABELS[label]
    spec = ct.MODELS[label]
    assert spec == {**ct.MODELS[parent], "aux": KEY} and spec["aux_weight"] == 1.0
    base._export_like_main(monkeypatch, spec)
    monkeypatch.setenv(cu.UNIF_WEIGHT_ENV, "0.002")
    monkeypatch.setenv("NETT_AUX_BATCH", "8")

    class _Env(base._Env):
        observation_space = _space(spec)
    agent = base._production_agent(spec, env_cls=_Env)
    assert isinstance(agent, AuxLossPPO) and agent._aux_kind == KEY and agent._aux_weight == 1.0
    enc = agent._aux_encoder
    assert enc is agent.policy.encoder and type(enc).__name__ == trunk and type(enc) in ed.ENCODERS
    aux = agent._aux
    assert isinstance(aux, cu.CLTTRefUnifAux) and aux.unif_weight == 0.002
    obs = _batch(spec, n=8)
    with torch.no_grad():
        z = enc(obs)
    assert z.shape == (8, 512) and torch.isfinite(z).all()
    _, out = agent.policy.act({"observations": obs.flatten(1).float(), "states": None}, role="policy")
    assert torch.isfinite(out["mean_actions"]).all()
    aux.attach_memory(_Memory(_flat(spec), t_max=12, n_env=2))
    loss = aux.compute(enc, obs)
    assert loss.dim() == 0 and torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in aux.head.parameters())
    n_enc = sum(p.numel() for p in enc.parameters())
    n_parent_enc = sum(p.numel() for p in _encoder(parent).parameters())
    n_head = sum(p.numel() for p in aux.head.parameters())
    n_agent = sum(p.numel() for p in {id(p): p for m in (agent.policy, agent.value) for p in m.parameters()}.values())
    assert n_enc == n_parent_enc and n_head == CLTT_HEAD_PARAMS
    with capsys.disabled():
        print(f"\n[cltt+unif] {label}: encoder {n_enc:,} (= {parent}), policy+value {n_agent:,}, "
              f"CLTT head {n_head:,}, unif term 0, total trained {n_agent + n_head:,}")


# ---------------------------------------------------------------- (e) refusals

@pytest.mark.parametrize("label", LABELS)
def test_weight_override_refuses_on_every_composite_label(monkeypatch, label):
    monkeypatch.setenv("NETT_AUX_WEIGHT_OVERRIDE", "0.5")
    with pytest.raises(ValueError, match="must not move"):
        ct.refuse_cltt_unif_weight_override(label, ct.MODELS[label])
    ct.refuse_cltt_unif_weight_override(LABELS[label][0], ct.MODELS[LABELS[label][0]])   # the CLTTA parent: unchanged
    ct.refuse_cltt_unif_weight_override("ViT-P8-666K+UNIF", ct.MODELS["ViT-P8-666K+UNIF"])
    monkeypatch.setenv("NETT_AUX_WEIGHT_OVERRIDE", "")                                   # main() reads "" as unset
    ct.refuse_cltt_unif_weight_override(label, ct.MODELS[label])


@pytest.mark.parametrize("weight", ["0.5", "2", "1.0000001"])
def test_the_term_refuses_any_aux_weight_but_one(monkeypatch, weight):
    spec = ct.MODELS["ViT-P8-666K+CLTT-Ref+UNIF"]
    monkeypatch.setenv("NETT_AUX_LOSS", KEY)
    monkeypatch.setenv("NETT_AUX_WEIGHT", weight)
    with pytest.raises(ValueError, match="must be exactly 1.0"):
        base._production_agent(spec, base._EnvN, None, dict(base._SMALL_VIT))
    monkeypatch.setenv("NETT_AUX_WEIGHT", "1.0")
    assert base._production_agent(spec, base._EnvN, None, dict(base._SMALL_VIT))._aux_weight == 1.0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "x", ""])
def test_lambda_knob_refuses_junk(monkeypatch, value):
    monkeypatch.setenv(cu.UNIF_WEIGHT_ENV, value)
    with pytest.raises(ValueError, match="must be a finite number > 0"):
        AUX_LOSSES[KEY](_encoder("CNN-D512P+CLTT-Ref+UNIF"))


def test_lambda_knob_refuses_on_every_other_label_and_admits_the_composites(monkeypatch):
    monkeypatch.setenv(cu.UNIF_WEIGHT_ENV, "0.001")
    for label in ("ViT-P8-666K+UNIF", "ViT-P8-666K+CLTT-Ref", "ViT+UNIF", "3DCNN-CLTT-Ref", "ViT-P8-666K"):
        with pytest.raises(ValueError, match="not a \\+CLTT-Ref\\+UNIF arm"):
            ct.refuse_embdiv_weight_knob(label, ct.MODELS[label])
    others = [lab for lab, spec in ct.MODELS.items() if spec.get("aux") != KEY]
    assert len(others) == len(ct.MODELS) - 4
    for label in others:
        with pytest.raises(ValueError):
            ct.refuse_embdiv_weight_knob(label, ct.MODELS[label])
    for label in LABELS:
        ct.refuse_embdiv_weight_knob(label, ct.MODELS[label])
        ct.refuse_stray_embdiv_knobs(label, ct.MODELS[label])                 # NETT_AUX_EMBDIV_* admitted
    monkeypatch.delenv(cu.UNIF_WEIGHT_ENV)
    for label in ct.MODELS:                                                   # unset: nothing refuses
        ct.refuse_embdiv_weight_knob(label, ct.MODELS[label])


def test_other_embdiv_knobs_admitted_on_composites_and_still_refused_on_clltta(monkeypatch):
    monkeypatch.setenv(ed.BATCH_ENV, "32")
    monkeypatch.setenv(ed.T_ENV, "3")
    for label in LABELS:
        ct.refuse_stray_embdiv_knobs(label, ct.MODELS[label])
    with pytest.raises(ValueError, match="not an embdiv arm"):
        ct.refuse_stray_embdiv_knobs("ViT-P8-666K+CLTT-Ref", ct.MODELS["ViT-P8-666K+CLTT-Ref"])
    aux = AUX_LOSSES[KEY](_encoder("ViT-P8-666K+CLTT-Ref+UNIF"))
    assert (aux.unif.max_samples, aux.unif.t) == (32, 3.0)
    monkeypatch.setenv(ed.FORM_ENV, "cos2")
    with pytest.raises(ValueError, match="The label names the form"):
        AUX_LOSSES[KEY](_encoder("ViT-P8-666K+CLTT-Ref+UNIF"))


def test_main_calls_every_refusal_before_the_aux_export():
    src = (_SRC / "examples" / "campaign_train.py").read_text()
    calls = ["    refuse_stray_embdiv_knobs(model, spec)\n", "    refuse_embdiv_weight_knob(model, spec)\n",
             "    refuse_cltt_unif_weight_override(model, spec)\n"]
    pos = [src.index(c) for c in calls]
    assert pos == sorted(pos) and pos[-1] < src.index(base._MAIN_WEIGHT_LINE)
    assert all(f'MODELS["{lab}"] = ' in src for lab in LABELS)       # quoted literals (tools/mem_probe_condition.sh)


# ---------------------------------------------------------------- (f) determinism

@pytest.mark.parametrize("label", ["ViT-P8-666K+CLTT-Ref+UNIF", "3DCNN-CLTT-Ref+UNIF"])
def test_two_same_seed_runs_agree_exactly(monkeypatch, label):
    _small(monkeypatch, "0.003")
    spec = ct.MODELS[label]
    obs = _batch(spec, n=12)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    results = []
    try:
        torch.use_deterministic_algorithms(True)
        for seed in (5, 5, 6):
            enc = _encoder(label)
            torch.manual_seed(7)
            aux = AUX_LOSSES[KEY](enc)
            aux.attach_memory(_Memory(_flat(spec), t_max=12, n_env=2))
            torch.manual_seed(seed)
            losses = []
            for _ in range(2):                         # two minibatches: the RNG stream advances across calls
                loss = aux.compute(enc, obs)
                loss.backward()
                losses.append(loss.detach())
            results.append((torch.stack(losses), _grads(enc, aux.head), dict(aux.last_scalars)))
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
    (l0, g0, s0), (l1, g1, s1), (l2, _, _) = results
    assert torch.equal(l0, l1) and torch.equal(g0, g1) and s0 == s1
    assert not torch.equal(l0, l2), "a different seed drew the same windows / subsample"
