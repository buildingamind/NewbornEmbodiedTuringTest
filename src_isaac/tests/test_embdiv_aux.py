"""EMBDIV labels ViT+UNIF / ViT+COS2 (owner 2026-11-01): push the ViT's encoder output apart (brain/aux/embdiv_aux.py).

  ViT+UNIF  = MODELS["ViT"] + aux embdiv_unif (Wang & Isola uniformity, t = 2), aux_weight 0.01
  ViT+COS2  = MODELS["ViT"] + aux embdiv_cos2 (mean squared pairwise cosine),    aux_weight 0.01
  lambda = 0.05 = the same label + NETT_AUX_WEIGHT_OVERRIDE=0.05.

Pinned: the two losses' values (orthonormal / identical / scale / spread vs clustered / closed form / identical points
finite); the gradient reaches every encoder tensor and the term has no parameters; every knob refuses junk; the labels
are the S3 ViT plus the aux alone and build through Brain -> build_agents at weight 0.01 and the 0.05 override; the
loss is repeatable under use_deterministic_algorithms(True) with the subsample engaged; the per-update collapse
diagnostics reach tfevents, stdout and embdiv_diag.csv (finite, with the PPO learning signal) through a real
AuxLossPPO.update; and the term's batch is a random, duplicate-free subsample across envs and timesteps of the rollout.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "examples"))

import campaign_train as ct  # noqa: E402
from nett_skrl.brain.aux import embdiv_aux as ed  # noqa: E402
from nett_skrl.brain.aux.knobs import NOT_MEASURED  # noqa: E402
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES  # noqa: E402
from nett_skrl.brain.config import EncoderCfg  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402

H, W = 80, 128
LABELS = {"ViT+UNIF": ("embdiv_unif", "uniformity"), "ViT+COS2": ("embdiv_cos2", "cos2")}
_KNOBS = (ed.FORM_ENV, ed.T_ENV, ed.BATCH_ENV, ed.DIAG_EVERY_ENV, "NETT_AUX_BATCH", "NETT_AUX_LOSS", "NETT_AUX_WEIGHT",
          "NETT_AUX_WEIGHT_OVERRIDE", "NETT_AUX_ALLOW_ZERO", "NETT_AUX_STRICT", "NETT_AUX_AFTER_PPO_BACKWARD",
          "NETT_AMP", "NETT_HIDDEN_SIZES", "NETT_FEATURE_ATTN", "NETT_DECOUPLE_ENCODER", "NETT_DECOUPLE_TRUNK",
          "NETT_DISABLE_FEATURE_CACHE", "NETT_ADV_NORM", "NETT_UINT8_BUFFER", "NETT_MEMORY_DEVICE")


@pytest.fixture(autouse=True)
def _clean_knobs(monkeypatch):
    for k in _KNOBS:
        monkeypatch.delenv(k, raising=False)


def _space(channels=3, h=H, w=W):
    return gym.spaces.Box(low=0, high=255, shape=(channels, h, w), dtype=np.uint8)


def _encoder(label="ViT+UNIF", seed=0):
    spec = ct.MODELS[label]
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](_space(), **kwargs)


def _batch(n=8, seed=1):
    return torch.randint(0, 256, (n, 3, H, W), dtype=torch.uint8, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- (i) cos2

def test_cos2_is_zero_for_orthonormal_one_for_identical_and_scale_invariant():
    eye = torch.eye(6, 10)
    assert float(ed.cos2_loss(eye)) == pytest.approx(0.0, abs=1e-12)
    same = torch.randn(1, 10).expand(6, 10)
    assert float(ed.cos2_loss(same)) == pytest.approx(1.0, abs=1e-6)
    z = torch.randn(7, 10, generator=torch.Generator().manual_seed(0))
    scale = torch.tensor([0.01, 1.0, 3.0, 100.0, 0.5, 7.0, 2.0])[:, None]
    assert torch.allclose(ed.cos2_loss(z), ed.cos2_loss(z * scale), atol=1e-6)
    assert torch.allclose(ed.cos2_loss(z), ed.cos2_loss(z * 1e3), atol=1e-6)
    # anti-parallel counts as aligned: cos^2 = 1
    assert float(ed.cos2_loss(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))) == pytest.approx(1.0)


# ---------------------------------------------------------------- (ii) uniformity

def test_uniformity_is_lower_for_spread_points_than_clustered():
    g = torch.Generator().manual_seed(0)
    spread = torch.randn(32, 16, generator=g)
    clustered = torch.randn(1, 16, generator=g) + 0.05 * torch.randn(32, 16, generator=g)
    assert float(ed.uniformity_loss(spread, 2.0)) < float(ed.uniformity_loss(clustered, 2.0)) - 1.0
    # and the orthonormal batch beats both: every pair at squared distance 2
    assert float(ed.uniformity_loss(torch.eye(16), 2.0)) == pytest.approx(-4.0, abs=1e-6)


def test_uniformity_matches_the_closed_form():
    # B = 2 orthogonal unit vectors: one distance^2 of 2 -> log exp(-2t) = -2t
    for t in (0.5, 2.0, 7.0):
        assert float(ed.uniformity_loss(torch.eye(2), t)) == pytest.approx(-2.0 * t, abs=1e-6)
    # B = 3 on a plane at 0, 90, 180 degrees: distances^2 = 2, 4, 2 (each twice over i != j)
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]) * torch.tensor([[2.0], [0.3], [5.0]])
    t = 2.0
    want = math.log((2 * math.exp(-t * 2) + 2 * math.exp(-t * 2) + 2 * math.exp(-t * 4)) / 6)
    assert float(ed.uniformity_loss(z, t)) == pytest.approx(want, abs=1e-6)
    # against a direct float64 evaluation of the definition on random points
    zz = torch.randn(9, 5, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    zh = zz / zz.norm(dim=1, keepdim=True)
    d2 = ((zh[:, None] - zh[None]) ** 2).sum(-1)
    off = ~torch.eye(9, dtype=torch.bool)
    ref = math.log(float(torch.exp(-2.0 * d2[off]).mean()))
    assert float(ed.uniformity_loss(zz, 2.0)) == pytest.approx(ref, abs=1e-9)


def test_uniformity_is_finite_and_zero_for_identical_points_and_large_t():
    z = torch.randn(1, 12).repeat(5, 1).requires_grad_(True)
    loss = ed.uniformity_loss(z, 2.0)
    assert torch.isfinite(loss) and float(loss) == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    # logsumexp: no underflow to log(0) at a temperature where every exp() is ~0 in float32
    assert torch.isfinite(ed.uniformity_loss(torch.eye(4), 500.0))
    assert float(ed.uniformity_loss(torch.eye(4), 500.0)) == pytest.approx(-1000.0, rel=1e-6)


def test_degenerate_batch_refuses():
    with pytest.raises(ValueError, match="no i != j pairs"):
        ed.uniformity_loss(torch.randn(1, 4))
    with pytest.raises(ValueError, match="no i != j pairs"):
        ed.cos2_loss(torch.randn(1, 4))
    enc = _encoder()
    term = AUX_LOSSES["embdiv_cos2"](enc)
    with pytest.raises(ValueError, match="no i != j pairs"):
        term.compute(enc, _batch(1))


# ---------------------------------------------------------------- (iii) gradient reaches the encoder

@pytest.mark.parametrize("label", LABELS)
def test_gradient_reaches_every_encoder_tensor_and_the_term_has_no_parameters(label):
    enc = _encoder(label)
    key, form = LABELS[label]
    term = AUX_LOSSES[key](enc)
    assert isinstance(term, ed.EmbDivTerm) and term.form == form
    assert list(term.parameters()) == [] and list(term.head.parameters()) == []
    loss = term.compute(enc, _batch(8))
    assert loss.dim() == 0 and torch.isfinite(loss) and loss.requires_grad
    loss.backward()
    missing = [n for n, p in enc.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0
               or not torch.isfinite(p.grad).all()]
    assert not missing, f"encoder tensors the term does not reach: {missing}"
    assert set(term.last_scalars) == {"loss", "B"} and term.last_scalars["B"] == 8.0
    assert term.last_scalars["loss"] == pytest.approx(float(loss))


def test_the_loss_reads_the_encoder_output_the_policy_reads():
    """compute() == the form's loss on enc(obs), the forward the policy and value heads consume."""
    enc = _encoder("ViT+COS2")
    obs = _batch(6)
    for key in ("embdiv_unif", "embdiv_cos2"):
        term = AUX_LOSSES[key](enc)
        with torch.no_grad():
            want = term.loss(enc(obs))
            got = term.compute(enc, obs)
        assert torch.allclose(got, want, atol=1e-6), key


# ---------------------------------------------------------------- (iv) knobs refuse junk

def test_knob_defaults():
    enc = _encoder()
    u, c = AUX_LOSSES["embdiv_unif"](enc), AUX_LOSSES["embdiv_cos2"](enc)
    assert (u.form, u.t, u.max_samples, u.diag_every) == ("uniformity", 2.0, 64, 1)
    assert (c.form, c.t, c.max_samples, c.diag_every) == ("cos2", None, 64, 1)


@pytest.mark.parametrize("key,env,value,match", [
    ("embdiv_unif", ed.FORM_ENV, "unif", "is not one of"),
    ("embdiv_unif", ed.FORM_ENV, "", "is not one of"),
    ("embdiv_unif", ed.FORM_ENV, "cos2", "The label names the form"),
    ("embdiv_cos2", ed.FORM_ENV, "uniformity", "The label names the form"),
    ("embdiv_unif", ed.T_ENV, "0", "must be a finite number > 0"),
    ("embdiv_unif", ed.T_ENV, "-2", "must be a finite number > 0"),
    ("embdiv_unif", ed.T_ENV, "inf", "must be a finite number > 0"),
    ("embdiv_unif", ed.T_ENV, "two", "must be a finite number > 0"),
    ("embdiv_cos2", ed.T_ENV, "2", "no temperature"),
    ("embdiv_unif", ed.BATCH_ENV, "1", "use >= 2"),
    ("embdiv_cos2", ed.BATCH_ENV, "0", "must be an integer > 0"),
    ("embdiv_cos2", ed.BATCH_ENV, "3.5", "must be an integer > 0"),
    ("embdiv_unif", ed.DIAG_EVERY_ENV, "0", "must be an integer > 0"),
    ("embdiv_unif", ed.DIAG_EVERY_ENV, "every", "must be an integer > 0"),
])
def test_bad_knob_values_refuse(monkeypatch, key, env, value, match):
    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError, match=match):
        AUX_LOSSES[key](_encoder())


def test_good_knob_values_are_realised(monkeypatch):
    monkeypatch.setenv(ed.FORM_ENV, "uniformity")
    monkeypatch.setenv(ed.T_ENV, "3")
    monkeypatch.setenv(ed.BATCH_ENV, "5")
    monkeypatch.setenv(ed.DIAG_EVERY_ENV, "4")
    enc = _encoder()
    term = AUX_LOSSES["embdiv_unif"](enc)
    assert (term.form, term.t, term.max_samples, term.diag_every) == ("uniformity", 3.0, 5, 4)
    term.compute(enc, _batch(9))
    assert term.last_scalars["B"] == 5.0
    # NETT_AUX_BATCH (cltt_ref / simclr's B) does not move this term's batch
    monkeypatch.delenv(ed.BATCH_ENV)
    monkeypatch.setenv("NETT_AUX_BATCH", "3")
    assert AUX_LOSSES["embdiv_unif"](enc).max_samples == 64


def test_other_encoders_refuse():
    """Only the admitted trunks (ed.ENCODERS: the S3 ViT + the four BL2 trunks, owner 2026-11-03) build the term; any
    other encoder, and any SUBCLASS of an admitted one, refuses (exact type: a trunk with no label of its own)."""
    assert {c.__name__ for c in ed.ENCODERS} == {"CompactViT", "NatureCNN", "Compact3DCNN", "CompactViViT"}
    others = [encoder_mapping[k](_space()) for k in ("compact_cnn", "small")]

    class _SubViT(type(_encoder())):
        pass
    sub = _SubViT.__new__(_SubViT)
    for enc in others + [sub]:
        assert type(enc) not in ed.ENCODERS
        for key in ed.KEY_FORM:
            with pytest.raises(TypeError, match="CompactViT, NatureCNN, Compact3DCNN, CompactViViT only"):
                AUX_LOSSES[key](enc)


def test_a_stray_embdiv_knob_refuses_on_another_label(monkeypatch):
    monkeypatch.setenv(ed.BATCH_ENV, "32")
    with pytest.raises(ValueError, match="not an embdiv arm"):
        ct.refuse_stray_embdiv_knobs("ViT", ct.MODELS["ViT"])
    with pytest.raises(ValueError, match="not an embdiv arm"):
        ct.refuse_stray_embdiv_knobs("ViT+MASK75", ct.MODELS["ViT+MASK75"])
    for label in LABELS:
        ct.refuse_stray_embdiv_knobs(label, ct.MODELS[label])
    src = (_SRC / "examples" / "campaign_train.py").read_text()
    assert "    refuse_stray_embdiv_knobs(model, spec)\n" in src


# ---------------------------------------------------------------- (v) labels, the real build path, lambda

def test_labels_are_the_s3_vit_plus_the_aux_alone():
    m = ct.MODELS
    assert m["ViT"] == dict(encoder="compact_vit", cfg=dict(ct.VIT_CFG), framestack=False)
    assert m["ViT"]["cfg"]["patch_size"] == 16 and m["ViT"]["cfg"]["pool"] == "cls"
    for label, (key, _) in LABELS.items():
        assert m[label] == {**m["ViT"], "aux": key, "aux_weight": 0.01}, label
        assert key in AUX_LOSSES and ed.KEY_FORM[key] == LABELS[label][1]
    bl2 = {f"{t}+{f}" for t in ("CNN-D512P", "ViT-P8-666K", "3DCNN", "ViViT-P8-T2") for f in ("UNIF", "COS2")}
    assert {k for k, v in m.items() if str(v.get("aux", "")).startswith("embdiv")} == set(LABELS) | bl2   # test_bl2_embdiv_labels.py
    assert sum(p.numel() for p in _encoder("ViT+UNIF").parameters()) == 693_728   # the S3 ViT's count (slide-3 tests)


#: main()'s aux export, verbatim. The test below asserts this line is still in campaign_train.py, so the copy
#: cannot drift from the code it stands in for.
_MAIN_WEIGHT_LINE = ('os.environ["NETT_AUX_WEIGHT"] = os.environ.get("NETT_AUX_WEIGHT_OVERRIDE") or '
                     'str(spec.get("aux_weight", 1.0))')


def _export_like_main(monkeypatch, spec):
    import os
    monkeypatch.setenv("NETT_AUX_LOSS", str(spec["aux"]))
    monkeypatch.setenv("NETT_AUX_WEIGHT", os.environ.get("NETT_AUX_WEIGHT_OVERRIDE") or str(spec.get("aux_weight", 1.0)))


class _Env:
    num_envs = 1
    num_agents = 1
    observation_space = _space()
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    state_space = None


def _production_agent(spec, env_cls=_Env, algorithm_cfg=None, encoder_kwargs=None):
    from nett_skrl.brain.agent_factory import build_agents
    from nett_skrl.brain.brain import Brain
    if encoder_kwargs is None:
        encoder_kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
        encoder_kwargs.pop("trainable", None)
    torch.manual_seed(0)
    brain = Brain(encoder=spec["encoder"], algorithm="PPO", encoder_cfg=encoder_kwargs,
                  algorithm_cfg=algorithm_cfg or {"rollouts": 16, "mini_batches": 1, "learning_epochs": 1},
                  model={"shared_encoder": True, "hidden_sizes": []})
    brain.num_brains = 1            # one brain over all of the env's envs (build_agents defaults to one per env)
    return build_agents(brain, env_cls(), torch.device("cpu"))[0]


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("override,want", [(None, 0.01), ("0.05", 0.05)])
def test_label_builds_through_build_agents_at_its_weight_and_the_override(monkeypatch, label, override, want):
    from nett_skrl.brain.aux.ppo_aux import AuxLossPPO
    assert _MAIN_WEIGHT_LINE in (_SRC / "examples" / "campaign_train.py").read_text()
    if override is not None:
        monkeypatch.setenv("NETT_AUX_WEIGHT_OVERRIDE", override)
    spec = ct.MODELS[label]
    _export_like_main(monkeypatch, spec)
    agent = _production_agent(spec)
    assert isinstance(agent, AuxLossPPO) and agent.value.encoder is agent.policy.encoder
    assert agent._aux_kind == spec["aux"] and agent._aux_weight == pytest.approx(want)
    assert isinstance(agent._aux, ed.EmbDivTerm) and agent._aux.form == LABELS[label][1]
    assert agent._aux_encoder is agent.policy.encoder
    # the term adds no parameters: its param group is empty, the PPO groups are untouched
    assert agent.optimizer.param_groups[-1]["params"] == []


# ---------------------------------------------------------------- (vi) determinism

@pytest.mark.parametrize("key", list(ed.KEY_FORM))
def test_repeatable_under_strict_determinism_with_the_subsample_engaged(key):
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    enc = _encoder()
    term = AUX_LOSSES[key](enc)
    term.max_samples = 6
    obs = _batch(20)
    try:
        torch.use_deterministic_algorithms(True)
        losses, grads = [], []
        for seed in (5, 5, 6):
            enc.zero_grad(set_to_none=True)
            torch.manual_seed(seed)
            loss = term.compute(enc, obs)
            loss.backward()
            losses.append(loss.detach())
            grads.append(torch.cat([p.grad.flatten() for p in enc.parameters()]))
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
    assert torch.equal(losses[0], losses[1]) and torch.equal(grads[0], grads[1])
    assert not torch.equal(losses[0], losses[2]), "a different seed drew the same subsample: is it engaged?"


# ---------------------------------------------------------------- diagnostics: values

def test_diagnostics_on_known_batches():
    d = ed.embedding_diagnostics(torch.eye(8, 16) * 3.0)
    assert d["cos_mean"] == pytest.approx(0.0, abs=1e-12) and d["cos2_mean"] == pytest.approx(0.0, abs=1e-12)
    assert d["erank"] == pytest.approx(8.0, rel=1e-9) and d["unif_t2"] == pytest.approx(-4.0, abs=1e-9)
    d = ed.embedding_diagnostics(torch.randn(1, 16).repeat(8, 1) * 5.0)
    assert d["std_raw"] == pytest.approx(0.0, abs=1e-6) and d["std_norm"] == pytest.approx(0.0, abs=1e-9)
    assert d["cos_mean"] == pytest.approx(1.0) and d["cos2_mean"] == pytest.approx(1.0)
    assert d["erank"] == pytest.approx(1.0, abs=1e-6) and d["unif_t2"] == pytest.approx(0.0, abs=1e-9)
    z = torch.randn(10, 4, generator=torch.Generator().manual_seed(0))
    d = ed.embedding_diagnostics(z)
    assert d["std_raw"] == pytest.approx(float(z.double().std(0).mean()))
    assert 1.0 < d["erank"] <= 4.0
    assert d["unif_t2"] == pytest.approx(float(ed.uniformity_loss(z.double(), 2.0)))
    assert d["cos2_mean"] == pytest.approx(float(ed.cos2_loss(z.double())))


# ---------------------------------------------------------------- diagnostics through a real AuxLossPPO.update

SH, SW, NENV, STEPS = 32, 48, 4, 16
_SMALL_VIT = {"features_dim": 32, "patch_size": 16, "embed_dim": 32, "depth": 1, "num_heads": 2}


class _EnvN:
    num_envs = NENV
    num_agents = 1
    observation_space = _space(3, SH, SW)
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
    state_space = None


def _obs_with_ids(t, g):
    """(NENV, 3*SH*SW) uint8-valued floats; channel 0 is the constant id t*NENV + env, the rest noise."""
    n = 3 * SH * SW
    obs = torch.randint(0, 256, (NENV, n), generator=g).float()
    obs[:, :SH * SW] = (t * NENV + torch.arange(NENV)).float()[:, None]
    return obs


def _small_agent(monkeypatch, tmp_path, label, batch, every=None):
    monkeypatch.chdir(tmp_path)
    spec = ct.MODELS[label]
    _export_like_main(monkeypatch, spec)
    monkeypatch.setenv(ed.BATCH_ENV, str(batch))
    if every is not None:
        monkeypatch.setenv(ed.DIAG_EVERY_ENV, str(every))
    agent = _production_agent(spec, _EnvN, {"rollouts": STEPS * NENV, "mini_batches": 2, "learning_epochs": 2,
                                            "learning_rate": 1e-3}, dict(_SMALL_VIT))
    agent.init(trainer_cfg=None)
    agent.enable_training_mode(True)
    agent.experiment_dir = str(tmp_path / "exp" / "brain_1")
    return agent


def _rollout_and_update(agent, seed, t0=0):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    obs = _obs_with_ids(0, g)
    with torch.no_grad():
        for t in range(STEPS):
            acts, _ = agent.act(obs, None, timestep=t0 + t, timesteps=1000)
            nxt = _obs_with_ids(t + 1, g)
            agent.record_transition(observations=obs, states=None, actions=acts,
                                    rewards=torch.rand(NENV, 1, generator=g), next_observations=nxt,
                                    next_states=None, terminated=torch.zeros(NENV, 1, dtype=torch.bool),
                                    truncated=torch.zeros(NENV, 1, dtype=torch.bool), infos={},
                                    timestep=t0 + t, timesteps=1000)
            obs = nxt
    agent.update(timestep=t0 + STEPS, timesteps=1000)


def _read_csv(agent):
    path = Path(agent.experiment_dir) / ed.DIAG_CSV
    with path.open() as fh:
        return list(csv.reader(fh))


@pytest.mark.parametrize("label", LABELS)
def test_diag_csv_is_written_during_training_with_finite_values(monkeypatch, tmp_path, capsys, label):
    agent = _small_agent(monkeypatch, tmp_path, label, batch=8)
    tracked = []
    real_track = agent.track_data
    monkeypatch.setattr(agent, "track_data", lambda tag, value: (tracked.append((tag, value)), real_track(tag, value)))
    _rollout_and_update(agent, seed=1)
    rows = _read_csv(agent)
    assert tuple(rows[0]) == ed.DIAG_COLUMNS and len(rows) == 2
    rec = dict(zip(rows[0], rows[1]))
    assert (rec["timestep"], rec["update"], rec["form"], rec["B"]) == (str(STEPS), "0", LABELS[label][1], "8")
    assert float(rec["aux_weight"]) == pytest.approx(0.01)
    for k in (*ed.DIAG_METRICS, *ed.SIGNAL_METRICS):
        if k == "episode_return_recent":       # no episode finishes in this rollout: the sentinel, never a number
            assert float(rec[k]) == NOT_MEASURED
            continue
        assert math.isfinite(float(rec[k])) and float(rec[k]) != NOT_MEASURED, (k, rec[k])
    assert float(rec["loss_weighted"]) == pytest.approx(0.01 * float(rec["loss_raw"]))
    assert float(rec["ppo_total"]) == pytest.approx(
        float(rec["policy_loss"]) + float(rec["value_loss"]) + float(rec["entropy_loss"]))
    assert float(rec["aux_share"]) == pytest.approx(abs(float(rec["loss_weighted"])) / abs(float(rec["ppo_total"])))
    assert 1.0 <= float(rec["erank"]) <= 8.0
    # the same numbers reach tfevents (track_data) and the PPO loss components are AuxLossPPO's own
    tags = dict(tracked)
    kind = ct.MODELS[label]["aux"]
    for k in (*ed.DIAG_METRICS, "aux_share"):
        assert tags[f"Loss / Aux {kind} diag_{k}"] == pytest.approx(float(rec[k]))
    assert tags["Loss / Policy loss"] == pytest.approx(float(rec["policy_loss"]))
    assert tags["Loss / Value loss"] == pytest.approx(float(rec["value_loss"]))
    assert tags[f"Loss / Aux ({kind}) loss"] == pytest.approx(float(rec["loss_weighted"]), rel=1e-5)
    # ... and stdout, live
    out = capsys.readouterr().out
    assert f"[NETT embdiv] brain_1 update 0 timestep {STEPS} {kind} B 8" in out
    # a second update appends a second row, header once
    _rollout_and_update(agent, seed=2, t0=STEPS)
    rows = _read_csv(agent)
    assert len(rows) == 3 and rows[2][1] == "1" and rows[2][0] == str(2 * STEPS)


def test_diag_every_k_updates(monkeypatch, tmp_path):
    agent = _small_agent(monkeypatch, tmp_path, "ViT+COS2", batch=8, every=2)
    for i in range(3):
        _rollout_and_update(agent, seed=10 + i, t0=i * STEPS)
    rows = _read_csv(agent)
    assert [r[1] for r in rows[1:]] == ["0", "2"]


def test_an_update_where_the_aux_never_ran_writes_sentinels_not_numbers(tmp_path):
    term = AUX_LOSSES["embdiv_unif"](_encoder())

    class _A:
        _aux_weight, _aux_kind, memory = 0.01, "embdiv_unif", None
        experiment_dir = str(tmp_path / "brain_1")

        def track_data(self, tag, value):
            pass

    row = term.after_update(_A(), timestep=7, timesteps=10)
    assert all(row[k] == NOT_MEASURED for k in ed.DIAG_METRICS)
    assert len(_read_csv(_A())) == 2


# ---------------------------------------------------------------- sampling across the whole rollout

def test_the_batch_is_a_random_subsample_across_envs_and_timesteps(monkeypatch, tmp_path):
    """Through the production agent: every frame the term encodes is identified by its id (t * NENV + env). Each
    call's B frames are distinct, span several envs AND several timesteps, are not one contiguous run, and differ
    from call to call; the whole update covers most of the 64-row rollout."""
    agent = _small_agent(monkeypatch, tmp_path, "ViT+UNIF", batch=8)
    enc = agent._aux_encoder
    seen = []
    orig = enc.encode_prepared

    def recording(prepared):
        seen.append(torch.round(prepared[:, 0, 0, 0] * 255.0).long().tolist())
        return orig(prepared)

    monkeypatch.setattr(enc, "encode_prepared", recording)
    _rollout_and_update(agent, seed=3)
    assert len(seen) == 4                              # 2 epochs x 2 minibatches
    for ids in seen:
        assert len(ids) == 8 and len(set(ids)) == 8, ids
        assert all(0 <= i < STEPS * NENV for i in ids)
        envs, steps = {i % NENV for i in ids}, {i // NENV for i in ids}
        assert len(envs) >= 2 and len(steps) >= 3, ids
        assert sorted(ids) != list(range(min(ids), min(ids) + 8)), f"contiguous ids {ids}"
    assert len({tuple(ids) for ids in seen}) == 4
    assert len(set().union(*map(set, seen))) >= 20


@pytest.mark.parametrize("finished", [[[1.0], [3.0]], [1.0, 3.0]])
def test_episode_return_recent_reads_skrl_track_rewards_shape(finished):
    from types import SimpleNamespace
    from collections import deque
    from nett_skrl.brain.aux.embdiv_aux import _learning_signal

    agent = SimpleNamespace(_track_rewards=deque(finished))
    out = _learning_signal(agent, None, 0.0)
    assert out["episode_return_recent"] == pytest.approx(2.0)
