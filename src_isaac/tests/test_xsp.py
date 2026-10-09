"""XSP -- cross-stream predictive learning (owner 2026-10-09): encoder brain/encoders/xsp.py, aux
brain/aux/xsp_aux.py, label "XSP" in examples/campaign_train.py.

Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU). Sections:
  1. build at the parsing eye (128x80, 2 frames), pinned parameter counts, softmax over channels
  2. leak: the target is the NEWEST frame of obs[t+1]
  3. siamese: one trunk, the same weights on x_{t-1} and x_t
  4. known-answer learning on a synthetic moving square (+ static and no-dorsal controls)
  5. knobs: every value builds and trains; malformed values refuse; unset == default
  6. existing-arm invariance (additive-only diff, unchanged MODELS, untouched encoder/aux files)
  7. mutants: break the leak guard / the softmax / the siamese sharing -> the checks go red
"""

from __future__ import annotations

import copy
import importlib.util
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.brain.aux.knobs import NOT_MEASURED
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.aux.xsp_aux import XSPTerm, combine_streams
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.xsp import XSPEncoder
from nett_skrl.brain.registry import encoder_mapping

H, W, C = 80, 128, 6
_SRC = Path(__file__).resolve().parents[1]
_TRAIN = _SRC / "examples" / "campaign_train.py"
XSP_KNOBS = ("NETT_XSP_POLICY_INPUT", "NETT_XSP_COMBINE", "NETT_XSP_ACTION", "NETT_XSP_DOWNSAMPLE",
             "NETT_XSP_HIDDEN", "NETT_XSP_BATCH", "NETT_XSP_TRANSIT_FRAC")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in XSP_KNOBS + ("NETT_AUX_CLTT_CHANNELS_PER_FRAME",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_XSP_BATCH", "8")
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


def _label_encoder(campaign, seed=0, h=H, w=W):
    kwargs = EncoderCfg(**campaign.MODELS["XSP"]["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping["xsp"](_space(h=h, w=w), **kwargs)


def _small(seed=0, **kw):
    torch.manual_seed(seed)
    return XSPEncoder(_space(), **{"features_dim": 32, "conv_dim": 16, **kw})


def _n(module):
    return sum(p.numel() for p in module.parameters())


def _prepared(n=4, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, C, H, W, generator=g)


class _FrameMemory:
    """Two-frame T-major stacks whose NEWEST frame at step t is the constant 100 + 4t and whose
    OLDER frame is 100 + 4(t-1): obs[t+1]'s older half EQUALS obs[t]'s newest frame -- exactly the
    leak the target must not contain. Turns vary so both parked/transit strata are populated."""

    def __init__(self, t_max=24, n_env=2):
        obs = torch.zeros(t_max, n_env, H, W, C)
        for t in range(t_max):
            obs[t, :, :, :, :3] = 100 + 4 * (t - 1)
            obs[t, :, :, :, 3:] = 100 + 4 * t
        acts = torch.zeros(t_max, n_env, 2)
        acts[:, :, 0] = torch.linspace(-0.5, 0.5, t_max).view(t_max, 1)
        acts[:, :, 1] = 0.3
        self.tensors = {
            "observations": obs.to(torch.uint8),
            "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "actions": acts,
        }
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


class _NoiseMemory(_FrameMemory):
    def __init__(self, t_max=24, n_env=2):
        super().__init__(t_max, n_env)
        g = torch.Generator().manual_seed(3)
        self.tensors["observations"] = torch.randint(0, 255, (t_max, n_env, H, W, C),
                                                     generator=g, dtype=torch.uint8)


# ================================================================ 1. build at the parsing eye

def test_label_is_registered_with_its_declared_spec(campaign):
    assert campaign.MODELS["XSP"] == dict(
        encoder="xsp", framestack=True, aux="xsp", aux_weight=1.0,
        cfg={"trainable": True, "features_dim": 512, "conv_dim": 75, "policy_input": "gated"})
    assert encoder_mapping["xsp"] is XSPEncoder
    assert "xsp" in AUX_LOSSES


#: Pinned at the parsing eye. ⚠ The parsing rows of U37 (lion L685-L688) set no NETT_EYE_RES, so
#: they run repo B's ObservationCfg.eye_resolution = 128x80 (W x H); some ViT rows run 256x160.
PINNED = {"encoder": 852_053, "trunk": 82_283, "ventral": 30_946, "dorsal": 123_912,
          "readout": 614_912, "head": 78_019}       # head at the fg default; 78,531 under outer


@pytest.mark.parametrize("h,w", [(80, 128), (160, 256)])
def test_builds_at_the_parsing_eye_with_pinned_counts(campaign, h, w):
    enc = _label_encoder(campaign, h=h, w=w)
    term = AUX_LOSSES["xsp"](enc)
    counts = {"encoder": _n(enc), "trunk": _n(enc.trunk), "ventral": _n(enc.ventral),
              "dorsal": _n(enc.dorsal), "readout": _n(enc.linear), "head": _n(term.head)}
    print(f"XSP @ {w}x{h}: {counts}; policy path (trunk+ventral+readout) = "
          f"{counts['trunk'] + counts['ventral'] + counts['readout']}")
    assert counts == PINNED
    assert enc.map_hw == (h // 8, w // 8) and enc.pool_grid == (2, 8)
    assert term.target_hw == (h // 4, w // 4)
    x = torch.randint(0, 256, (3, C, h, w), dtype=torch.uint8)
    assert tuple(enc(x).shape) == (3, 512)


def check_streams_softmax(enc=None):
    """v is a 2-way softmax at EVERY location: non-negative, sums to 1 over channels."""
    enc = enc or _small()
    s = enc.encode_streams(_prepared())
    assert tuple(s["z_t"].shape) == tuple(s["z_prev"].shape) == (4, 16, 10, 16)
    assert tuple(s["v"].shape) == (4, 2, 10, 16)
    assert tuple(s["d"].shape) == (4, 8, 10, 16)
    v = s["v"]
    assert bool((v >= 0).all())
    assert torch.allclose(v.sum(dim=1), torch.ones(4, 10, 16), atol=1e-6)
    # the policy path computes the same ventral map
    assert torch.allclose(enc.ventral_map(enc.trunk(_prepared()[:, 3:])), v, atol=1e-6)


def test_ventral_is_a_softmax_over_channels_at_every_location():
    check_streams_softmax()


def test_refuses_a_single_frame_and_a_non_xsp_encoder():
    with pytest.raises(ValueError, match="frame stack"):
        XSPEncoder(_space(channels=3))
    nature = encoder_mapping["nature_cnn"](_space(), features_dim=32)
    with pytest.raises(TypeError, match="encode_streams"):
        XSPTerm(nature)
    with pytest.raises(ValueError, match="policy_input"):
        XSPEncoder(_space(), policy_input="pooled")


def test_loss_is_finite_and_grads_reach_every_stream_and_the_head():
    enc = _small()
    term = XSPTerm(enc)
    term.attach_memory(_NoiseMemory())
    loss = term.compute(enc, None)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    for name, mod in (("trunk", enc.trunk), ("ventral", enc.ventral), ("dorsal", enc.dorsal),
                      ("head", term.head)):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters()), name
    assert all(p.grad is None for p in enc.linear.parameters())     # readout: RL gradient only
    enc_ids = {id(p) for p in enc.parameters()}
    assert not any(id(p) in enc_ids for p in term.head.parameters())


def test_policy_gradient_reaches_trunk_and_ventral_but_not_dorsal():
    enc = _small()
    x = torch.randint(0, 256, (3, C, H, W), dtype=torch.uint8)
    enc(x).pow(2).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.trunk.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.ventral.parameters())
    assert all(p.grad is None for p in enc.dorsal.parameters())


def test_auxlossppo_registers_the_decoder_and_attaches_memory(monkeypatch):
    """The production wiring: AuxLossPPO builds `xsp` from AUX_LOSSES, adds ONLY the decoder as a
    new param group (phi_e/phi_v/phi_d are encoder parameters, already in the policy's group),
    attaches its rollout memory, and -- as for nextframe -- does not checkpoint the decoder."""
    from types import SimpleNamespace
    from nett_skrl.brain.aux.ppo_aux import PPO, AuxLossPPO
    enc = _small()
    mem = _NoiseMemory()

    def init(self, *args, **kwargs):
        self.policy = SimpleNamespace(encoder=enc)
        self.optimizer = torch.optim.Adam(enc.parameters())
        self.checkpoint_modules = {}
        self.memory = mem

    monkeypatch.setattr(PPO, "__init__", init)
    agent = AuxLossPPO(aux_loss="xsp", aux_weight=1)
    assert isinstance(agent._aux, XSPTerm) and agent._aux._memory is mem
    head = {id(p) for p in agent._aux.head.parameters()}
    assert {id(p) for p in agent.optimizer.param_groups[1]["params"]} == head
    assert not head & {id(p) for p in enc.parameters()}
    assert "aux_head" not in agent.checkpoint_modules


def test_aux_forward_is_deterministic_under_strict_mode():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    enc = _small()
    term = XSPTerm(enc)
    term.attach_memory(_NoiseMemory())
    try:
        torch.use_deterministic_algorithms(True)
        losses = []
        for _ in range(2):
            torch.manual_seed(5)
            losses.append(term.compute(enc, None))
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
    assert torch.equal(losses[0].detach(), losses[1].detach())


# ================================================================ 2. leak

def check_target_is_newest_of_next(monkeypatch):
    """newest(t) = v(t), older(t+1) = v(t): a correct target is v(t+1), so the copy baseline errs
    by exactly 4/255 per pixel. A target built from obs[t+1]'s OLDER half (= x_t) would make
    copy_mse 0; one built from obs[t] would too."""
    enc = _small()
    term = XSPTerm(enc)
    term.attach_memory(_FrameMemory())
    seen = {}
    orig_head = type(term.head).forward

    def spy(self, cur, v, m, action=None):
        seen["cur"] = cur
        return orig_head(self, cur, v, m, action)

    monkeypatch.setattr(type(term.head), "forward", spy)
    targets = []
    orig_target = XSPTerm._target

    def tspy(self, p):
        out = orig_target(self, p)
        targets.append(out)
        return out

    monkeypatch.setattr(XSPTerm, "_target", tspy)
    term.compute(enc, None)
    (target,) = targets
    assert tuple(target.shape[1:]) == (3, H // 4, W // 4)
    assert torch.allclose(target - seen["cur"], torch.full_like(target, 4 / 255), atol=1e-6)
    assert term.last_scalars["copy_mse"] == pytest.approx((4 / 255) ** 2, rel=1e-4)


def test_target_is_the_newest_frame_of_obs_t_plus_1(monkeypatch):
    check_target_is_newest_of_next(monkeypatch)


# ================================================================ 3. siamese

def check_siamese(enc=None):
    enc = enc or _small()
    p = _prepared()
    a, b = p[:, :3], p[:, 3:]
    s_ab = enc.encode_streams(torch.cat([a, b], 1))
    s_ba = enc.encode_streams(torch.cat([b, a], 1))
    # SHARED WEIGHTS: frame b encoded in the t-1 slot equals frame b encoded in the t slot.
    assert torch.allclose(s_ab["z_t"], s_ba["z_prev"], atol=1e-6)
    assert torch.allclose(s_ab["z_prev"], s_ba["z_t"], atol=1e-6)
    # Perturb x_{t-1} ONLY: the current frame's trunk map, the ventral map and the policy
    # features are bit-unchanged; the previous map and the dorsal map move (positive control).
    q = p.clone()
    q[:, :3] = torch.rand_like(q[:, :3])
    s_q = enc.encode_streams(q)
    assert torch.equal(s_q["z_t"], s_ab["z_t"]) and torch.equal(s_q["v"], s_ab["v"])
    assert not torch.allclose(s_q["z_prev"], s_ab["z_prev"])
    assert not torch.allclose(s_q["d"], s_ab["d"])
    assert torch.equal(enc.encode_prepared(q), enc.encode_prepared(p))


def test_trunk_is_siamese_and_the_previous_frame_reaches_only_the_dorsal():
    check_siamese()


def test_policy_forward_matches_the_streams():
    enc = _small().eval()
    x = torch.randint(0, 256, (3, C, H, W), dtype=torch.uint8)
    s = enc.encode_streams(enc._prepare_image(x))
    assert torch.allclose(enc(x), enc.readout(s["z_t"], s["v"]), atol=1e-6)


# ================================================================ 5. knobs

def _train_once(enc, term):
    term.attach_memory(_NoiseMemory())
    loss = term.compute(enc, None)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in term.head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.trunk.parameters())
    return term


@pytest.mark.parametrize("name,value,extra", [
    ("NETT_XSP_COMBINE", "outer", None), ("NETT_XSP_COMBINE", "fg", None),
    ("NETT_XSP_ACTION", "none", None), ("NETT_XSP_ACTION", "film", None),
    ("NETT_XSP_DOWNSAMPLE", "1", None), ("NETT_XSP_DOWNSAMPLE", "2", None), ("NETT_XSP_DOWNSAMPLE", "4", None),
    ("NETT_XSP_DOWNSAMPLE", "8", None),
    ("NETT_XSP_HIDDEN", "16", None), ("NETT_XSP_HIDDEN", "64", None),
    ("NETT_XSP_BATCH", "4", None), ("NETT_XSP_TRANSIT_FRAC", "0", None), ("NETT_XSP_TRANSIT_FRAC", "1", None),
    ("NETT_XSP_BATCH", "2", ("NETT_XSP_TRANSIT_FRAC", "0")),
])
def test_every_aux_knob_value_builds_and_trains(monkeypatch, name, value, extra):
    monkeypatch.setenv(name, value)
    if extra:
        monkeypatch.setenv(*extra)
    enc = _small()
    term = _train_once(enc, XSPTerm(enc))
    s = term.last_scalars
    for key in ("B", "mse", "copy_mse", "skill", "mse_parked", "mse_transit", "copy_parked",
                "copy_transit", "window_turn", "action_dim", "fg_frac", "fg_frac_std",
                "fg_entropy", "d_absmean"):
        assert key in s, key
    assert all(np.isfinite(v) for v in s.values())
    assert ("film_gain" in s) == (term.action == "film")
    if name == "NETT_XSP_COMBINE":
        assert term.head.inp.in_channels == 2 + (16 if value == "outer" else 8)
    if name == "NETT_XSP_DOWNSAMPLE":
        ds = int(value)
        assert term.target_hw == (H // ds, W // ds) and len(term.head.up) == {1: 3, 2: 2, 4: 1, 8: 0}[ds]
    if name == "NETT_XSP_HIDDEN":
        assert term.head.inp.out_channels == int(value)
    if name == "NETT_XSP_BATCH":
        assert s["B"] == int(value)


def test_combine_outer_is_both_groups_and_fg_is_the_figure_group():
    v = torch.softmax(torch.randn(2, 2, 3, 4), dim=1)
    d = torch.randn(2, 5, 3, 4)
    outer = combine_streams(v, d, "outer")
    assert torch.allclose(outer[:, :5], v[:, :1] * d) and torch.allclose(outer[:, 5:], v[:, 1:] * d)
    assert torch.allclose(outer[:, :5] + outer[:, 5:], d, atol=1e-6)
    assert torch.allclose(combine_streams(v, d, "fg"), v[:, :1] * d)


def test_film_is_zero_initialised_and_learns_an_action_gradient(monkeypatch):
    monkeypatch.setenv("NETT_XSP_ACTION", "film")
    enc = _small()
    term = _train_once(enc, XSPTerm(enc))
    assert term.head.film.weight.abs().sum() == 0
    assert term.head.film.weight.grad.abs().sum() > 0
    assert term.last_scalars["film_gain"] == 0.0


def test_action_none_takes_no_action_input(monkeypatch):
    """Paper-faithful default: g has no FiLM and its output does not depend on a_t."""
    enc = _small()
    term = XSPTerm(enc)
    assert term.head.film is None
    p = _prepared()
    with torch.no_grad():
        a, _ = term.predict(enc, p, torch.zeros(4, 2))
        b, _ = term.predict(enc, p, torch.ones(4, 2))
    assert torch.equal(a, b)


@pytest.mark.parametrize("policy_input,n_in", [("gated", 16 * 16), ("map", 10 * 16), ("both", 2 * 16 * 16)])
def test_policy_input_values_build_and_backprop(policy_input, n_in):
    enc = _small(policy_input=policy_input)
    assert enc.linear[0].in_features == n_in
    x = torch.randint(0, 256, (3, C, H, W), dtype=torch.uint8)
    out = enc(x)
    assert tuple(out.shape) == (3, 32)
    out.pow(2).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.ventral.parameters())
    if policy_input == "map":
        # the raw figure map alone: the readout cannot see z_t except through v
        assert enc.readout_pool is None


def test_both_readout_is_invariant_to_swapping_the_ventral_channels():
    """"both" reads [z*v_fg, z*v_bg]; swapping the channels permutes its input halves, while
    "gated" sees a different gate -- the orientation-free property that motivates the value."""
    enc = _small(policy_input="both")
    s = enc.encode_streams(_prepared())
    z, v = s["z_t"], s["v"]
    pool = lambda t: enc.readout_pool(t).flatten(1)
    a = torch.cat([pool(z * v[:, :1]), pool(z * v[:, 1:])], 1)
    b = torch.cat([pool(z * v.flip(1)[:, :1]), pool(z * v.flip(1)[:, 1:])], 1)
    half = a.shape[1] // 2
    assert torch.equal(a[:, :half], b[:, half:]) and torch.equal(a[:, half:], b[:, :half])
    assert torch.allclose(enc.readout(z, v), enc.linear(a), atol=1e-6)


def test_both_policy_input_pinned_count(campaign):
    kwargs = EncoderCfg(**{**campaign.MODELS["XSP"]["cfg"], "policy_input": "both"}).as_kwargs()
    kwargs.pop("trainable", None)
    enc = encoder_mapping["xsp"](_space(), **kwargs)
    assert _n(enc.linear) == 2400 * 512 + 512 and _n(enc) == PINNED["encoder"] - PINNED["readout"] + 1_229_312


@pytest.mark.parametrize("name,bad", [
    ("NETT_XSP_COMBINE", "OUTER"), ("NETT_XSP_COMBINE", "both"), ("NETT_XSP_COMBINE", ""),
    ("NETT_XSP_ACTION", "FiLM"), ("NETT_XSP_ACTION", "yes"),
    ("NETT_XSP_DOWNSAMPLE", "3"), ("NETT_XSP_DOWNSAMPLE", "16"), ("NETT_XSP_DOWNSAMPLE", "0"),
    ("NETT_XSP_DOWNSAMPLE", "x"),
    ("NETT_XSP_HIDDEN", "0"), ("NETT_XSP_HIDDEN", "-4"), ("NETT_XSP_HIDDEN", "1.5"),
    ("NETT_XSP_BATCH", "1"), ("NETT_XSP_BATCH", "0"), ("NETT_XSP_BATCH", "2"), ("NETT_XSP_BATCH", "3"),
    ("NETT_XSP_TRANSIT_FRAC", "1.5"), ("NETT_XSP_TRANSIT_FRAC", "nan"),
])
def test_malformed_aux_knobs_refuse(monkeypatch, name, bad):
    monkeypatch.setenv(name, bad)
    with pytest.raises(ValueError, match=name):
        XSPTerm(_small())


@pytest.mark.parametrize("bad", ["Gated", "pooled", "", " map2"])
def test_malformed_policy_input_refuses(campaign, monkeypatch, bad):
    monkeypatch.setenv("NETT_XSP_POLICY_INPUT", bad)
    with pytest.raises(ValueError, match="NETT_XSP_POLICY_INPUT"):
        campaign._xsp_policy_input()


def test_policy_input_knob_resolves(campaign, monkeypatch):
    assert campaign._xsp_policy_input() == "gated"
    for value in ("gated", "map", "both"):
        monkeypatch.setenv("NETT_XSP_POLICY_INPUT", value)
        assert campaign._xsp_policy_input() == value


def test_film_refuses_a_one_component_action(monkeypatch):
    monkeypatch.setenv("NETT_XSP_ACTION", "film")
    enc = _small()
    term = XSPTerm(enc)
    mem = _NoiseMemory()
    mem.tensors["actions"] = mem.tensors["actions"][..., :1]
    term.attach_memory(mem)
    with pytest.raises(ValueError, match="motor components"):
        term.compute(enc, None)


DEFAULTS = {"NETT_XSP_COMBINE": "fg", "NETT_XSP_ACTION": "none", "NETT_XSP_DOWNSAMPLE": "4",
            "NETT_XSP_HIDDEN": "64", "NETT_XSP_BATCH": "64", "NETT_XSP_TRANSIT_FRAC": "0.5"}


def _default_run(campaign):
    enc = _label_encoder(campaign, seed=3)
    torch.manual_seed(4)
    term = XSPTerm(enc)
    term.attach_memory(_NoiseMemory(t_max=80))
    torch.manual_seed(5)
    loss = term.compute(enc, None)
    return enc, term, loss


def test_unset_knobs_equal_explicit_defaults(campaign, monkeypatch):
    monkeypatch.delenv("NETT_XSP_BATCH")
    enc0, term0, loss0 = _default_run(campaign)
    for k, v in DEFAULTS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("NETT_XSP_POLICY_INPUT", "gated")
    assert campaign._xsp_policy_input() == campaign.MODELS["XSP"]["cfg"]["policy_input"] == "gated"
    enc1, term1, loss1 = _default_run(campaign)
    for a, b in ((enc0, enc1), (term0.head, term1.head)):
        sa, sb = a.state_dict(), b.state_dict()
        assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    assert torch.equal(loss0, loss1)
    assert term0.last_scalars == term1.last_scalars
    assert term0.last_scalars["B"] == 64


def test_stray_xsp_knob_on_another_label_refuses(campaign, monkeypatch):
    monkeypatch.delenv("NETT_XSP_BATCH")                                 # set by the autouse fixture
    for label, spec in campaign.MODELS.items():
        assert campaign.refuse_stray_xsp_knobs(label, spec) is None      # nothing set: no-op
    monkeypatch.setenv("NETT_XSP_COMBINE", "fg")
    with pytest.raises(ValueError, match="not an XSP arm"):
        campaign.refuse_stray_xsp_knobs("CNN", campaign.MODELS["CNN"])
    assert campaign.refuse_stray_xsp_knobs("XSP", campaign.MODELS["XSP"]) is None


def test_knobs_are_literal_reads_the_env_gate_can_find():
    """tools/env_reader_check.py searches non-test .py files for each key's literal name."""
    text = "".join(p.read_text() for p in (
        _SRC / "nett_skrl" / "brain" / "aux" / "xsp_aux.py", _TRAIN))
    for k in XSP_KNOBS:
        assert f'"{k}"' in text, k
    index = (_SRC / "docs" / "env_vars.md").read_text()
    for k in XSP_KNOBS:
        assert f"`{k}`" in index, k


# ================================================================ 4. known-answer learning
#
# A synthetic video: a static smooth random texture (0.15..0.60) per sample, and a 16x16 square of
# value 1.0 moving 4 px/frame in one of four directions (random per sample), at the real 80x128 eye.
# Prediction needs BOTH where the square is (x_t) and which way it moves (x_{t-1} vs x_t), so a
# model whose dorsal stream is cut can only smear -- that is the no-dorsal control.
#
# Measured before pinning (scratch ka.py, the same generator, 300 Adam steps at lr 1e-3, batch 16,
# 10 model seeds each; 8 or 4 threads):
#   fg (DEFAULT since the coordinator's 2026-10-09 ruling)
#                    skill .938-.965 in 9/10, square on channel 0 in 9/10 (diff +.145..+.461);
#                    seed 0 COLLAPSES: v_fg == 0 everywhere, entropy 0, skill .206.
#   outer (knob)     skill .951-.964 in 10/10; |fg_in - fg_out| .19-.45 in 10/10, but the square
#                    is on CHANNEL 0 (the "gated" policy gate) in only 3/10 (seeds 0, 8, 9) and on
#                    channel 1 in 7/10 -- the loss does not decide which channel is "figure".
#   d := 0           outer: skill .200-.204 (seeds 0-2); fg: .195, .205 (seeds 1, 2). The motion
#                    stream is what lifts skill from .2 to .96. Under outer v still separates the
#                    square in 2/3 of these (diff +.15) -- v enters g directly, so the map is
#                    partly a form/brightness cue, not purely motion-made.
#   untrained        |fg_in - fg_out| <= .0008; AUC .10-.76 on maps that differ by 1e-3, so the
#                    MEAN DIFFERENCE is the primary statistic and AUC is reported, not asserted.
# Thresholds: skill >= .8 (the x_t-only ceiling is ~.2); skill - no-dorsal >= .5; |diff| >= .15
# (observed minimum .19); fg direction >= .10 (observed minimum .145).

KA_S, KA_SPEED, KA_STEPS = 16, 4, 300


def _texture(g, b):
    t = torch.rand(b, 3, H // 8, W // 8, generator=g)
    return 0.15 + 0.45 * F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)


def _ka_batch(g, b, static=False):
    bg = _texture(g, b)
    frames, masks = [bg.clone() for _ in range(3)], []
    y0 = torch.randint(8, H - KA_S - 8, (b,), generator=g)
    x0 = torch.randint(8, W - KA_S - 8, (b,), generator=g)
    dirs = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]])[torch.randint(0, 4, (b,), generator=g)]
    if static:
        dirs = dirs * 0
    for k in range(3):
        m = torch.zeros(b, H, W)
        for i in range(b):
            y = int(y0[i] + (k - 1) * KA_SPEED * dirs[i, 0])
            x = int(x0[i] + (k - 1) * KA_SPEED * dirs[i, 1])
            frames[k][i, :, y:y + KA_S, x:x + KA_S] = 1.0
            m[i, y:y + KA_S, x:x + KA_S] = 1
        masks.append(m)
    cell = F.avg_pool2d(masks[1][:, None], 8)[:, 0]               # square coverage per map cell at t
    return torch.cat(frames[:2], 1), torch.cat(frames[1:], 1), cell


def _auc(score, label):
    s, lab = score.flatten(), label.flatten().bool()
    pos, neg = s[lab], s[~lab]
    r = torch.cat([pos, neg]).argsort().argsort().float() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _ka_eval(enc, term):
    pt, ptk, cell = _ka_batch(torch.Generator().manual_seed(999), 64)
    with torch.no_grad():
        term.score(enc, pt, ptk, torch.zeros(64, 2))
        v = enc.encode_streams(pt)["v"][:, 0]
    fg = cell >= 0.5
    return dict(skill=term.last_scalars["skill"], auc=_auc(v, fg),
                diff=float(v[fg].mean() - v[cell == 0].mean()),
                fg_frac=term.last_scalars["fg_frac"], fg_entropy=term.last_scalars["fg_entropy"],
                v_fg_max=float(v.max()), v_fg_min=float(v.min()), combine=term.combine)


def _ka_train(monkeypatch, combine, seed, no_dorsal=False):
    """combine=None trains the DEFAULT (knob unset)."""
    if combine is not None:
        monkeypatch.setenv("NETT_XSP_COMBINE", combine)
    torch.set_num_threads(8)                                    # restored by the autouse fixture
    if no_dorsal:
        orig = XSPEncoder.encode_streams

        def cut(self, p):
            s = orig(self, p)
            s["d"] = torch.zeros_like(s["d"])
            return s

        monkeypatch.setattr(XSPEncoder, "encode_streams", cut)
    torch.manual_seed(seed)
    enc = XSPEncoder(_space(), features_dim=32)
    term = XSPTerm(enc)
    opt = torch.optim.Adam(list(enc.parameters()) + list(term.head.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(100 + seed)
    init = _ka_eval(enc, term)
    for _ in range(KA_STEPS):
        pt, ptk, _ = _ka_batch(g, 16)
        loss = term.score(enc, pt, ptk, torch.zeros(16, 2))
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = _ka_eval(enc, term)
    print(f"KA combine={combine} seed={seed} no_dorsal={no_dorsal}: init {init} -> final {final}")
    return init, final


@pytest.mark.parametrize("seed", [1, 2])
def test_known_answer_default_puts_the_square_on_the_figure_channel(monkeypatch, seed):
    """The DEFAULT (fg): prediction beats copy AND the square is on channel 0, the policy's gate."""
    init, final = _ka_train(monkeypatch, None, seed)
    assert final["combine"] == "fg"
    assert abs(init["diff"]) < 0.01                       # untrained map does not separate it
    assert final["skill"] >= 0.8                          # prediction clearly beats copy
    assert final["diff"] >= 0.10 and final["auc"] >= 0.9  # ... on the FIGURE channel


@pytest.mark.parametrize("seed", [0, 1])
def test_known_answer_outer_knob_separates_the_square_on_either_channel(monkeypatch, seed):
    init, final = _ka_train(monkeypatch, "outer", seed)
    assert abs(init["diff"]) < 0.01
    assert final["skill"] >= 0.8
    assert abs(final["diff"]) >= 0.15                     # symmetric: no channel is "figure"
    assert np.isfinite(final["auc"])


def test_known_answer_no_dorsal_control_cannot_predict_motion(monkeypatch):
    _, full = _ka_train(monkeypatch, None, 1)
    _, cut = _ka_train(monkeypatch, None, 1, no_dorsal=True)
    assert cut["skill"] <= 0.4
    assert full["skill"] - cut["skill"] >= 0.5


def _telemetry_flags_collapse(s):
    """What a reader of the run's own logs can see: the map is decided (entropy ~0) and uniform
    (fg_frac at an extreme)."""
    return s["fg_entropy"] < 0.05 and (s["fg_frac"] < 0.05 or s["fg_frac"] > 0.95)


def _map_is_collapsed(r):
    """Ground truth from the map itself, not from the telemetry: every location of every sample
    saturated to the SAME channel (v_fg <= .005 or >= .995 everywhere; binary entropy of .005 is
    .045 bits, so a collapsed map is inside the telemetry's .05 bound by construction)."""
    return r["v_fg_max"] <= 0.005 or r["v_fg_min"] >= 0.995


def _check_collapse_detection(r):
    if _map_is_collapsed(r):
        assert _telemetry_flags_collapse(r), r          # every collapse is DETECTED
    if abs(r["diff"]) >= 0.10:
        assert not _telemetry_flags_collapse(r), r      # a separating map raises no alarm


@pytest.mark.parametrize("seed", [0, 3])
def test_known_answer_default_collapse_whenever_it_happens_is_detected(monkeypatch, seed):
    """fg can collapse to v_fg == 0 (m = 0, g sees no motion): measured in 1/10 scratch seeds. This
    does NOT pin which seeds collapse. Whatever training produces, a collapsed map must be flagged by
    fg_frac/fg_entropy and a separating one must not; the forced cases below make sure the
    detection branch is exercised even if no trained seed collapses."""
    _, final = _ka_train(monkeypatch, None, seed)
    print(f"seed {seed}: map collapsed={_map_is_collapsed(final)} "
          f"telemetry flags={_telemetry_flags_collapse(final)}")
    _check_collapse_detection(final)


@pytest.mark.parametrize("bias", [(-30.0, 30.0), (30.0, -30.0), None])
def test_forced_collapse_is_detected_and_a_healthy_map_is_not(bias):
    """Positive controls for the detector: saturate the ventral output to all-ground / all-figure
    (a collapse by construction) and leave it untrained (undecided, entropy ~1, not a collapse)."""
    torch.manual_seed(0)
    enc = XSPEncoder(_space(), features_dim=32)
    term = XSPTerm(enc)
    if bias is not None:
        with torch.no_grad():
            enc.ventral[-1].weight.zero_()
            enc.ventral[-1].bias.copy_(torch.tensor(bias))
    r = _ka_eval(enc, term)
    assert _map_is_collapsed(r) == (bias is not None)
    assert _telemetry_flags_collapse(r) == (bias is not None)
    _check_collapse_detection(r)


@pytest.mark.parametrize("collapse", [True, False])
def test_collapse_detection_check_goes_red_on_a_blind_detector(monkeypatch, collapse):
    """Mutant: a detector that never fires must fail the check on a real collapse; one that always
    fires must fail it on a separating map."""
    r = {"v_fg_max": 1e-6, "v_fg_min": 0.0, "fg_frac": 1e-6, "fg_entropy": 1e-6, "diff": 0.0} if collapse \
        else {"v_fg_max": 0.99, "v_fg_min": 0.01, "fg_frac": 0.5, "fg_entropy": 0.9, "diff": 0.4}
    monkeypatch.setitem(globals(), "_telemetry_flags_collapse", lambda s: not collapse)
    with pytest.raises(AssertionError):
        _check_collapse_detection(r)


def test_static_scene_control_is_finite_and_marks_skill_unmeasured():
    """copy_mse == 0 on a static scene: skill is NOT_MEASURED (never a division by zero), the loss
    and every gradient are finite, and a few steps drive the residual prediction toward the copy."""
    torch.set_num_threads(8)
    enc = XSPEncoder(_space(), features_dim=32)
    term = XSPTerm(enc)
    opt = torch.optim.Adam(list(enc.parameters()) + list(term.head.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(7)
    first = None
    for _ in range(30):
        pt, ptk, _ = _ka_batch(g, 8, static=True)
        loss = term.score(enc, pt, ptk, torch.zeros(8, 2))
        assert torch.isfinite(loss)
        assert term.last_scalars["copy_mse"] == 0.0 and term.last_scalars["skill"] == NOT_MEASURED
        opt.zero_grad()
        loss.backward()
        for p in list(enc.parameters()) + list(term.head.parameters()):
            assert p.grad is None or torch.isfinite(p.grad).all()
        opt.step()
        first = float(loss) if first is None else first
    assert float(loss) < first


# ================================================================ 6. existing-arm invariance
#
# By construction: XSP adds two files and one ENTRY to each of four shared files. The tests below
# prove that from git -- against the commit before XSP (the parent of the commit that added
# encoders/xsp.py; HEAD while it is still uncommitted) -- and then check the consequence that
# matters: every pre-existing MODELS label resolves to the identical spec.

SHARED = ("nett_skrl/brain/aux/ppo_aux.py", "nett_skrl/brain/registry.py",
          "nett_skrl/brain/encoders/__init__.py", "examples/campaign_train.py")
NEW_FILES = ("nett_skrl/brain/encoders/xsp.py", "nett_skrl/brain/aux/xsp_aux.py")


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=_SRC, capture_output=True, text=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git unavailable here: {exc}")


def _xsp_range():
    """(base, head): head None = the working tree (XSP not yet committed)."""
    added = _git("log", "--diff-filter=A", "--format=%H", "--", NEW_FILES[0]).split()
    return (f"{added[-1]}^", added[-1]) if added else ("HEAD", None)


def _diff(*args):
    base, head = _xsp_range()
    return _git("diff", base, *([head] if head else []), *args)


@pytest.mark.parametrize("path", SHARED)
def test_shared_file_change_is_additive_only(path):
    lines = _diff("--unified=0", "--", path).splitlines()
    removed = [ln for ln in lines if ln.startswith("-") and not ln.startswith("---")]
    added = [ln for ln in lines if ln.startswith("+") and not ln.startswith("+++")]
    assert added, f"{path}: no XSP addition found -- the diff did not look at the change"
    assert not removed, f"{path}: XSP removed or edited existing lines:\n" + "\n".join(removed)


def test_no_other_brain_file_changed():
    changed = set(_diff("--name-only", "--", "nett_skrl", "examples").split())
    allowed = {f"src_isaac/{p}" for p in SHARED + NEW_FILES}
    assert changed - allowed == set(), sorted(changed - allowed)


def test_existing_labels_resolve_to_the_identical_spec(campaign):
    base, _ = _xsp_range()
    src = _git("show", f"{base}:src_isaac/examples/campaign_train.py")
    spec = importlib.util.spec_from_loader("_campaign_train_pre_xsp", loader=None)
    old = importlib.util.module_from_spec(spec)
    old.__dict__["__file__"] = str(_TRAIN)
    exec(compile(src, "campaign_train@pre-xsp", "exec"), old.__dict__)
    assert "XSP" not in old.MODELS
    assert set(campaign.MODELS) - set(old.MODELS) == {"XSP"}
    for label, spec_old in old.MODELS.items():
        assert campaign.MODELS[label] == spec_old, label
    # NAS schemas hold validator lambdas (never equal across two module objects): compare the rest.
    nas = lambda labels: {k: {**{kk: vv for kk, vv in d.items() if kk != "schema"},
                              "schema": sorted(d["schema"])} for k, d in labels.items()}
    assert nas(campaign.NAS_LABELS) == nas(old.NAS_LABELS)
    assert campaign.EXPERIMENTS == old.EXPERIMENTS
    for label, spec_old in old.MODELS.items():
        assert campaign.segmentation_wrappers(campaign.MODELS[label]) == old.segmentation_wrappers(spec_old)


def test_registries_only_gained_the_xsp_entries():
    base, _ = _xsp_range()
    reg = _git("show", f"{base}:src_isaac/nett_skrl/brain/aux/ppo_aux.py")
    old_keys = {ln.split('"')[1] for ln in reg.split("AUX_LOSSES = {", 1)[1].split("\n}", 1)[0].splitlines()
                if ln.strip().startswith('"')}
    assert old_keys and set(AUX_LOSSES) - old_keys == {"xsp"} and old_keys <= set(AUX_LOSSES)
    enc = _git("show", f"{base}:src_isaac/nett_skrl/brain/registry.py")
    old_enc = {ln.split('"')[1] for ln in enc.split("encoder_mapping: dict", 1)[1].split("\n}", 1)[0].splitlines()
               if ln.strip().startswith('"')}
    assert old_enc and set(encoder_mapping) - old_enc == {"xsp"} and old_enc <= set(encoder_mapping)


# ================================================================ 7. mutants
#
# Each mutant breaks ONE guarantee; the matching check above must then fail. A check that still
# passed under its mutant would be one that cannot see what it claims to test.

def _mut_leaky_target(mp):
    mp.setattr(XSPTerm, "_target",
               lambda self, p: F.avg_pool2d(p[:, -2 * self.cpf:-self.cpf], self.downsample))


def _mut_no_softmax(mp):
    mp.setattr(XSPEncoder, "ventral_map", lambda self, z: torch.sigmoid(self.ventral(z)))


def _mut_unshared_trunk(mp):
    orig = XSPEncoder.encode_streams

    def es(self, prepared):
        if "_prev_trunk" not in self.__dict__:
            twin = copy.deepcopy(self.trunk)
            with torch.no_grad():
                for p in twin.parameters():
                    p.add_(0.05 * torch.randn_like(p))
            self.__dict__["_prev_trunk"] = twin
        x_prev, x_t = self.split_frames(prepared)
        z_prev, z_t = self.__dict__["_prev_trunk"](x_prev), self.trunk(x_t)
        v = self.ventral_map(z_t)
        return {"x_t": x_t, "z_prev": z_prev, "z_t": z_t, "v": v,
                "d": self.dorsal(torch.cat([z_prev, z_t], dim=1))}

    mp.setattr(XSPEncoder, "encode_streams", es)
    assert orig is not XSPEncoder.encode_streams


def _mut_dorsal_ignores_prev(mp):
    orig = XSPEncoder.encode_streams

    def es(self, prepared):
        s = orig(self, prepared)
        s["d"] = self.dorsal(torch.cat([s["z_t"], s["z_t"]], dim=1))
        return s

    mp.setattr(XSPEncoder, "encode_streams", es)


@pytest.mark.parametrize("mutant,check", [
    (_mut_leaky_target, "leak"),
    (_mut_no_softmax, "softmax"),
    (_mut_unshared_trunk, "siamese"),
    (_mut_dorsal_ignores_prev, "siamese"),
])
def test_mutant_turns_its_check_red(monkeypatch, mutant, check):
    mutant(monkeypatch)
    run = {"leak": lambda: check_target_is_newest_of_next(monkeypatch),
           "softmax": check_streams_softmax,
           "siamese": check_siamese}[check]
    with pytest.raises(AssertionError):
        run()
