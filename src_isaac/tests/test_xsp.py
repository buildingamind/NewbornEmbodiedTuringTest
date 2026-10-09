"""XSP -- cross-stream predictive learning (owner 2026-10-09): encoder brain/encoders/xsp.py, aux
brain/aux/xsp_aux.py, label "XSP" in examples/campaign_train.py. Aligned with the owner's figure:
a one-channel SIGMOID figure map m_t from x_t, a dorsal stream over the WHOLE T-frame stack.

Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU). Sections, each at T = 2 and T = 3
stacked frames where the stack depth matters:
  1. build at the parsing eye (128x80), pinned parameter counts, m_t = sigmoid (one channel)
  2. leak: the target is the NEWEST frame of obs[t+1]
  3. siamese: one trunk, the same weights on every frame; older frames reach only phi_d
  4. known-answer learning on a synthetic moving square (+ static and no-dorsal controls)
  5. knobs: every value builds and trains; malformed values refuse; unset == default
  6. existing-arm invariance (additive-only diff, unchanged MODELS, untouched encoder/aux files)
  7. mutants: break the leak guard / the sigmoid / the siamese sharing / the stack -> checks go red
"""

from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.brain.aux.knobs import NOT_MEASURED
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.aux import xsp_aux
from nett_skrl.brain.aux.xsp_aux import XSPTerm, combine_streams, motion_orientation
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.xsp import XSPEncoder
from nett_skrl.brain.registry import encoder_mapping

H, W, CPF = 80, 128, 3
C = 2 * CPF                     # the default 2-frame stack
STACKS = (2, 3)                 # T values every stack-dependent test runs at
_SRC = Path(__file__).resolve().parents[1]
_TRAIN = _SRC / "examples" / "campaign_train.py"
XSP_KNOBS = ("NETT_XSP_POLICY_INPUT", "NETT_XSP_COMBINE", "NETT_XSP_ACTION", "NETT_XSP_DOWNSAMPLE",
             "NETT_XSP_HIDDEN", "NETT_XSP_BATCH", "NETT_XSP_TRANSIT_FRAC", "NETT_XSP_G_MAP")


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


def _label_encoder(campaign, seed=0, h=H, w=W, T=2, **cfg):
    """The label's encoder, its cfg's num_frames set to T (as NETT_FRAMESTACK_N=T would)."""
    kwargs = EncoderCfg(**{**campaign.MODELS["XSP"]["cfg"], "num_frames": T, **cfg}).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping["xsp"](_space(CPF * T, h=h, w=w), **kwargs)


def _small(seed=0, T=2, **kw):
    torch.manual_seed(seed)
    return XSPEncoder(_space(CPF * T), **{"features_dim": 32, "conv_dim": 16, **kw})


def _n(module):
    return sum(p.numel() for p in module.parameters())


def _prepared(n=4, seed=5, T=2):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, CPF * T, H, W, generator=g)


class _FrameMemory:
    """T-frame T-major stacks in which frame k (k = 0 oldest .. T-1 newest) of obs[t] is the
    constant 100 + 4(t - (T-1) + k): every frame of obs[t+1] except its newest IS a frame of obs[t]
    -- exactly the leak the target must not contain. Turns vary so both strata are populated."""

    def __init__(self, t_max=24, n_env=2, T=2):
        obs = torch.zeros(t_max, n_env, H, W, CPF * T)
        for t in range(t_max):
            for k in range(T):
                obs[t, :, :, :, k * CPF:(k + 1) * CPF] = 100 + 4 * (t - (T - 1) + k)
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
    def __init__(self, t_max=24, n_env=2, T=2):
        super().__init__(t_max, n_env, T)
        g = torch.Generator().manual_seed(3)
        self.tensors["observations"] = torch.randint(0, 255, (t_max, n_env, H, W, CPF * T),
                                                     generator=g, dtype=torch.uint8)


# ================================================================ 1. build at the parsing eye

def test_label_is_registered_with_its_declared_spec(campaign):
    # This suite runs at the default stack depth (NETT_FRAMESTACK_N unset -> 2).
    assert campaign._FRAMESTACK_N == 2
    assert campaign.MODELS["XSP"] == dict(
        encoder="xsp", framestack=True, aux="xsp", aux_weight=1.0,
        cfg={"trainable": True, "features_dim": 512, "conv_dim": 75, "policy_input": "gated",
             "num_frames": campaign._FRAMESTACK_N})
    assert encoder_mapping["xsp"] is XSPEncoder
    assert "xsp" in AUX_LOSSES


#: Pinned at the parsing eye, per stack depth T. ⚠ The parsing rows of U37 (lion L685-L688) set no
#: NETT_EYE_RES, so they run repo B's ObservationCfg.eye_resolution = 128x80 (W x H); some ViT rows
#: run 256x160. Only phi_d's first conv widens with T (T * conv_dim inputs).
PINNED = {
    2: {"encoder": 852_020, "trunk": 82_283, "ventral": 30_913, "dorsal": 123_912,
        "readout": 614_912, "head": 78_403},
    3: {"encoder": 895_220, "trunk": 82_283, "ventral": 30_913, "dorsal": 167_112,
        "readout": 614_912, "head": 78_403},
}   # head at the default (outer, G_MAP=0); 78,467 with G_MAP=1; 77,891 / 77,955 under fg


@pytest.mark.parametrize("T", STACKS)
@pytest.mark.parametrize("h,w", [(80, 128), (160, 256)])
def test_builds_at_the_parsing_eye_with_pinned_counts(campaign, h, w, T):
    enc = _label_encoder(campaign, h=h, w=w, T=T)
    term = AUX_LOSSES["xsp"](enc)
    counts = {"encoder": _n(enc), "trunk": _n(enc.trunk), "ventral": _n(enc.ventral),
              "dorsal": _n(enc.dorsal), "readout": _n(enc.linear), "head": _n(term.head)}
    print(f"XSP @ {w}x{h}, T={T}: {counts}; policy path (trunk+ventral+readout) = "
          f"{counts['trunk'] + counts['ventral'] + counts['readout']}")
    assert counts == PINNED[T]
    assert enc.n_frames == T and enc.dorsal[0].in_channels == T * 75
    assert enc.map_hw == (h // 8, w // 8) and enc.pool_grid == (2, 8)
    assert term.target_hw == (h // 4, w // 4)
    x = torch.randint(0, 256, (3, CPF * T, h, w), dtype=torch.uint8)
    assert tuple(enc(x).shape) == (3, 512)


def test_label_refuses_a_stack_that_disagrees_with_num_frames(campaign, monkeypatch):
    """num_frames (= _FRAMESTACK_N) must describe the observation EXACTLY. Positive control: the
    matching pairs build."""
    for T in STACKS:
        _label_encoder(campaign, T=T)
    with pytest.raises(ValueError, match="disagree"):
        _label_encoder(campaign, T=2, num_frames=3)                # 6 channels, cfg says 3 frames
    with pytest.raises(ValueError, match="disagree"):
        _label_encoder(campaign, T=3, num_frames=2)                # 9 channels, cfg says 2 frames
    # A channels-per-frame misparse on an RGB 6-channel stack would read it as 3 two-channel
    # frames; with num_frames from the cfg that refuses instead of scrambling colour into time.
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "2")
    with pytest.raises(ValueError, match="disagree"):
        _label_encoder(campaign, T=2)


def check_streams_sigmoid(enc=None, T=2):
    """m_t is ONE channel, in [0, 1], and EQUAL to sigmoid(phi_v(z_t)); the policy path computes
    the same map from x_t alone."""
    enc = enc or _small(T=T)
    p = _prepared(T=T)
    s = enc.encode_streams(p)
    assert tuple(s["z_frames"].shape) == (4, T, 16, 10, 16)
    assert tuple(s["z_t"].shape) == (4, 16, 10, 16) and torch.equal(s["z_t"], s["z_frames"][:, -1])
    assert tuple(s["m"].shape) == (4, 1, 10, 16)
    assert tuple(s["d"].shape) == (4, 8, 10, 16)
    m = s["m"]
    assert bool((m >= 0).all()) and bool((m <= 1).all())
    assert torch.allclose(m, torch.sigmoid(enc.ventral(s["z_t"])), atol=1e-6)
    assert torch.allclose(enc.ventral_map(enc.trunk(p[:, -CPF:])), m, atol=1e-6)


@pytest.mark.parametrize("T", STACKS)
def test_ventral_is_a_one_channel_sigmoid_figure_map(T):
    check_streams_sigmoid(T=T)


def test_refuses_a_single_frame_a_wrong_stack_and_a_non_xsp_encoder():
    with pytest.raises(ValueError, match="frame stack"):
        XSPEncoder(_space(channels=3))
    nature = encoder_mapping["nature_cnn"](_space(), features_dim=32)
    with pytest.raises(TypeError, match="encode_streams"):
        XSPTerm(nature)
    with pytest.raises(ValueError, match="policy_input"):
        XSPEncoder(_space(), policy_input="pooled")
    enc = _small(T=2)
    enc.encode_streams(_prepared(T=2))                               # positive control
    with pytest.raises(ValueError, match="built for 2"):
        enc.encode_streams(_prepared(T=3))


@pytest.mark.parametrize("T", STACKS)
def test_loss_is_finite_and_grads_reach_every_stream_and_the_head(T):
    enc = _small(T=T)
    term = XSPTerm(enc)
    term.attach_memory(_NoiseMemory(T=T))
    loss = term.compute(enc, None)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    for name, mod in (("trunk", enc.trunk), ("ventral", enc.ventral), ("dorsal", enc.dorsal),
                      ("head", term.head)):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters()), name
    # every one of the T input blocks of phi_d's first conv is used
    g = enc.dorsal[0].weight.grad.view(enc.dorsal[0].out_channels, T, -1)
    assert bool((g.abs().sum(dim=(0, 2)) > 0).all())
    assert all(p.grad is None for p in enc.linear.parameters())     # readout: RL gradient only
    enc_ids = {id(p) for p in enc.parameters()}
    assert not any(id(p) in enc_ids for p in term.head.parameters())


@pytest.mark.parametrize("T", STACKS)
def test_policy_gradient_reaches_trunk_and_ventral_but_not_dorsal(T):
    enc = _small(T=T)
    x = torch.randint(0, 256, (3, CPF * T, H, W), dtype=torch.uint8)
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

def check_target_is_newest_of_next(monkeypatch, T=2):
    """Frame k of obs[t] is 100 + 4(t - (T-1) + k): x_t = newest(obs[t]) and the frames of obs[t+1]
    are x_{t-T+2} .. x_t, x_{t+1}. Only the NEWEST of them differs from x_t by exactly +4/255 per
    pixel; obs[t+1]'s next-newest frame IS x_t (difference 0, copy_mse 0) and at T=3 its oldest is
    x_{t-1} (difference -4/255). So target - cur == +4/255 everywhere proves the target is the
    newest frame and nothing else."""
    enc = _small(T=T)
    term = XSPTerm(enc)
    term.attach_memory(_FrameMemory(T=T))
    seen = {}
    orig_head = type(term.head).forward

    def spy(self, cur, m, c, action=None):
        seen["cur"] = cur
        return orig_head(self, cur, m, c, action)

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
    assert tuple(target.shape[1:]) == (CPF, H // 4, W // 4)
    assert torch.allclose(target - seen["cur"], torch.full_like(target, 4 / 255), atol=1e-6)
    assert term.last_scalars["copy_mse"] == pytest.approx((4 / 255) ** 2, rel=1e-4)


@pytest.mark.parametrize("T", STACKS)
def test_target_is_the_newest_frame_of_obs_t_plus_1(monkeypatch, T):
    check_target_is_newest_of_next(monkeypatch, T)


# ================================================================ 3. siamese

def check_siamese(enc=None, T=2):
    """ONE trunk on every frame: re-ordering the frames re-orders z_frames and nothing else. Every
    OLDER frame reaches phi_d and only phi_d: perturbing it leaves z_t, m_t and the policy features
    bit-unchanged and moves d_t (positive control)."""
    enc = enc or _small(T=T)
    p = _prepared(T=T)
    frames = [p[:, k * CPF:(k + 1) * CPF] for k in range(T)]
    s = enc.encode_streams(p)
    s_rev = enc.encode_streams(torch.cat(frames[::-1], 1))
    assert torch.allclose(s_rev["z_frames"], s["z_frames"].flip(1), atol=1e-6)
    for k in range(T - 1):                                       # every older frame, oldest first
        q = p.clone()
        q[:, k * CPF:(k + 1) * CPF] = torch.rand_like(q[:, k * CPF:(k + 1) * CPF])
        s_q = enc.encode_streams(q)
        assert torch.equal(s_q["z_t"], s["z_t"]) and torch.equal(s_q["m"], s["m"]), k
        assert torch.equal(enc.encode_prepared(q), enc.encode_prepared(p)), k
        assert not torch.allclose(s_q["z_frames"][:, k], s["z_frames"][:, k]), k
        assert not torch.allclose(s_q["d"], s["d"]), k
    # the newest frame reaches m_t and the policy (positive control for the equalities above)
    q = p.clone()
    q[:, -CPF:] = torch.rand_like(q[:, -CPF:])
    assert not torch.allclose(enc.encode_streams(q)["m"], s["m"])
    assert not torch.allclose(enc.encode_prepared(q), enc.encode_prepared(p))


@pytest.mark.parametrize("T", STACKS)
def test_trunk_is_siamese_and_older_frames_reach_only_the_dorsal(T):
    check_siamese(T=T)


@pytest.mark.parametrize("T", STACKS)
def test_policy_forward_matches_the_streams(T):
    enc = _small(T=T).eval()
    x = torch.randint(0, 256, (3, CPF * T, H, W), dtype=torch.uint8)
    s = enc.encode_streams(enc._prepare_image(x))
    assert torch.allclose(enc(x), enc.readout(s["z_t"], s["m"]), atol=1e-6)


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
    ("NETT_XSP_G_MAP", "0", None), ("NETT_XSP_G_MAP", "1", None),
    ("NETT_XSP_G_MAP", "1", ("NETT_XSP_COMBINE", "fg")),
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
                "fg_entropy", "d_absmean", "fg_motion_sep", "fg_motion_sep_parked", "fg_motion_n_parked"):
        assert key in s, key
    assert all(np.isfinite(v) for v in s.values())
    assert ("film_gain" in s) == (term.action == "film")
    if name == "NETT_XSP_COMBINE":
        assert term.head.inp.in_channels == (16 if value == "outer" else 8)
    if name == "NETT_XSP_G_MAP":
        assert term.g_map == term.head.g_map == (value == "1")
        assert term.head.inp.in_channels == (16 if term.combine == "outer" else 8) + int(value)
    if name == "NETT_XSP_DOWNSAMPLE":
        ds = int(value)
        assert term.target_hw == (H // ds, W // ds) and len(term.head.up) == {1: 3, 2: 2, 4: 1, 8: 0}[ds]
    if name == "NETT_XSP_HIDDEN":
        assert term.head.inp.out_channels == int(value)
    if name == "NETT_XSP_BATCH":
        assert s["B"] == int(value)


def test_combine_outer_is_figure_and_ground_groups_and_fg_is_the_figure_group():
    m = torch.sigmoid(torch.randn(2, 1, 3, 4))
    d = torch.randn(2, 5, 3, 4)
    outer = combine_streams(m, d, "outer")
    assert tuple(outer.shape) == (2, 10, 3, 4)
    assert torch.allclose(outer[:, :5], m * d) and torch.allclose(outer[:, 5:], (1 - m) * d)
    assert torch.allclose(outer[:, :5] + outer[:, 5:], d, atol=1e-6)
    fg = combine_streams(m, d, "fg")
    assert tuple(fg.shape) == (2, 5, 3, 4) and torch.allclose(fg, m * d)
    assert not torch.allclose(fg, (1 - m) * d)                  # the figure side, not the ground
    with pytest.raises(ValueError, match="combine"):
        combine_streams(m, d, "sum")


def test_sigmoid_figure_map_is_the_methods_two_way_softmax_and_outer_is_v_times_d():
    """Methods: v_t = softmax(phi_v(z_t)) over TWO channels, x^ = g(x_t, v_t (.) d_t). The encoder's
    one logit l gives m = sigmoid(l); softmax([l, 0]) == (m, 1 - m) at every location, so
    v_t (.) d_t = [v_fg * d, v_bg * d] == combine "outer" exactly. Checked on the real encoder's
    logits and on an extreme range (|l| up to 40) where a naive 1 - m would lose precision."""
    enc = _small()
    s = enc.encode_streams(_prepared())
    for logit in (enc.ventral(s["z_t"]), torch.linspace(-40, 40, 801).view(1, 1, 1, -1)):
        m = torch.sigmoid(logit)
        v = torch.softmax(torch.cat([logit, torch.zeros_like(logit)], dim=1), dim=1)   # (B, 2, h, w)
        assert torch.allclose(v[:, :1], m, atol=1e-7, rtol=0)
        assert torch.allclose(v[:, 1:], 1 - m, atol=1e-7, rtol=0)
        # ... and the softmax is unchanged by a shared shift of both logits (the removed redundancy)
        shift = torch.cat([logit + 3.0, torch.full_like(logit, 3.0)], dim=1)
        assert torch.allclose(torch.softmax(shift, dim=1), v, atol=1e-6)
    m, d = s["m"], s["d"]
    v = torch.softmax(torch.cat([enc.ventral(s["z_t"]), torch.zeros_like(m)], dim=1), dim=1)
    v_odot_d = (v[:, :, None] * d[:, None]).flatten(1, 2)       # [v_fg*d_1..K, v_bg*d_1..K]
    assert torch.allclose(combine_streams(m, d, "outer"), v_odot_d, atol=1e-6)
    assert torch.allclose(combine_streams(m, d, "fg"), v_odot_d[:, :8], atol=1e-6)   # fg = figure half only


def check_g_reads_only_x_and_the_product(term, enc):
    """Under the default (G_MAP=0) g's output depends on (cur, c) ONLY: any m gives the same x^.
    Positive control: changing c moves the output."""
    s = enc.encode_streams(_prepared())
    cur = term._current(_prepared())
    c = combine_streams(s["m"], s["d"], term.combine)
    with torch.no_grad():
        out = term.head(cur, s["m"], c)
        assert torch.equal(term.head(cur, 1 - s["m"], c), out)
        assert torch.equal(term.head(cur, torch.rand_like(s["m"]), c), out)
        assert not torch.allclose(term.head(cur, s["m"], c + 1.0), out)


def test_predictor_input_is_only_the_current_frame_and_the_product():
    enc = _small()
    term = XSPTerm(enc)
    assert term.g_map is False and term.head.inp.in_channels == 2 * enc.dorsal_dim
    check_g_reads_only_x_and_the_product(term, enc)


def test_g_map_knob_feeds_the_map_directly(monkeypatch):
    """NETT_XSP_G_MAP=1: the pre-Methods behaviour -- m_t is a direct input, so the default's
    check goes red (the knob is real, and the check can see a map input)."""
    monkeypatch.setenv("NETT_XSP_G_MAP", "1")
    enc = _small()
    term = XSPTerm(enc)
    assert term.g_map is True and term.head.inp.in_channels == 2 * enc.dorsal_dim + 1
    with pytest.raises(AssertionError):
        check_g_reads_only_x_and_the_product(term, enc)


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


def test_both_readout_does_not_depend_on_which_side_of_the_sigmoid_is_figure():
    """"both" reads [pool(z*m), pool(z*(1-m))]: flipping m -> 1-m is undone EXACTLY by swapping the
    two column halves of its Linear, so a readout trained on either assignment exists. "gated"
    (positive control) reads a different gate and changes."""
    enc = _small(policy_input="both")
    s = enc.encode_streams(_prepared())
    z, m = s["z_t"], s["m"]
    swapped = copy.deepcopy(enc)
    half = enc.linear[0].in_features // 2
    with torch.no_grad():
        w = enc.linear[0].weight
        swapped.linear[0].weight.copy_(torch.cat([w[:, half:], w[:, :half]], 1))
    assert torch.allclose(swapped.readout(z, 1 - m), enc.readout(z, m), atol=1e-6)
    assert not torch.allclose(enc.readout(z, 1 - m), enc.readout(z, m))
    gated = _small(policy_input="gated")
    assert not torch.allclose(gated.readout(z, 1 - m), gated.readout(z, m))


def test_both_policy_input_pinned_count(campaign):
    kwargs = EncoderCfg(**{**campaign.MODELS["XSP"]["cfg"], "policy_input": "both"}).as_kwargs()
    kwargs.pop("trainable", None)
    enc = encoder_mapping["xsp"](_space(), **kwargs)
    assert _n(enc.linear) == 2400 * 512 + 512 and _n(enc) == PINNED[2]["encoder"] - PINNED[2]["readout"] + 1_229_312


@pytest.mark.parametrize("name,bad", [
    ("NETT_XSP_COMBINE", "OUTER"), ("NETT_XSP_COMBINE", "both"), ("NETT_XSP_COMBINE", ""),
    ("NETT_XSP_G_MAP", "2"), ("NETT_XSP_G_MAP", "true"), ("NETT_XSP_G_MAP", ""), ("NETT_XSP_G_MAP", "01"),
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


DEFAULTS = {"NETT_XSP_COMBINE": "outer", "NETT_XSP_G_MAP": "0", "NETT_XSP_ACTION": "none", "NETT_XSP_DOWNSAMPLE": "4",
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
    assert term0.combine == "outer" and term0.g_map is False and term0.head.inp.in_channels == 16


def test_stray_xsp_knob_on_another_label_refuses(campaign, monkeypatch):
    monkeypatch.delenv("NETT_XSP_BATCH")                                 # set by the autouse fixture
    for label, spec in campaign.MODELS.items():
        assert campaign.refuse_stray_xsp_knobs(label, spec) is None      # nothing set: no-op
    monkeypatch.setenv("NETT_XSP_COMBINE", "fg")
    with pytest.raises(ValueError, match="not an XSP arm"):
        campaign.refuse_stray_xsp_knobs("CNN", campaign.MODELS["CNN"])
    assert campaign.refuse_stray_xsp_knobs("XSP", campaign.MODELS["XSP"]) is None
    monkeypatch.delenv("NETT_XSP_COMBINE")
    monkeypatch.setenv("NETT_XSP_G_MAP", "0")                            # the new knob too
    with pytest.raises(ValueError, match="NETT_XSP_G_MAP"):
        campaign.refuse_stray_xsp_knobs("CNN", campaign.MODELS["CNN"])


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
# obs_t stacks T frames, obs_{t+1} the next T (T + 1 frames in all). Prediction needs BOTH where the
# square is (x_t) and which way it moves (the older frames), so a model whose dorsal stream is cut
# can only smear -- that is the no-dorsal control. "diff" = mean m_t on cells the square covers
# (>= .5) minus mean m_t on cells it does not touch; > 0 means the square is on the FIGURE side.
#
# Measured before pinning (scratch ka3.py, the same generator, 300 Adam steps at lr 1e-3, batch 16,
# model seeds 0-9 per cell; 3 threads per run). sep = diff above; "undecided" = entropy > .99 with
# |sep| < .05; "collapse" = m saturated to one side (none of the cells below has one).
#
#   T=2                     skill      |sep| >= .15   object on HIGH side     AUC (high side)  coll/undec
#   outer, G_MAP=0 DEFAULT  .949-.968  8/10           3/10 (seeds 0, 3, 8)    .988-.999        0 / 0
#   outer, G_MAP=1          .954-.964  10/10          4/10 (seeds 0, 1, 3, 7) .802-.995        0 / 0
#   fg,    G_MAP=0          .950-.963  10/10          10/10, +.166..+.444     .968-1.000       0 / 0
#   fg,    G_MAP=1          .957-.964  9/10           9/10, +.142..+.466      .986-.999        0 / 1 (s7)
#   T=3 (G_MAP=0 only)
#   outer, G_MAP=0          .953-.968  8/10           6/10 (0, 1, 2, 3, 6, 9) .747-.999        0 / 0
#   fg,    G_MAP=0          .956-.966  10/10          10/10, +.244..+.482     .992-1.000       0 / 0
#
#   Default outer misses: T=2 seed 0 sep +.109 (fg_frac .90, entropy .44: leaning to all-figure),
#   seed 6 sep -.087; T=3 seed 8 sep -.036. Under outer the LOSS cannot decide the side (it is
#   symmetric under m <-> 1-m), so the side is a coin per seed, and the "gated" readout gates on
#   the background whenever the object lands low: NETT_XSP_POLICY_INPUT=both is orientation-free.
#   ⚠ The coin is not even fixed by the seed: at 8 threads (this file) T=2 seed 0 lands at -.111
#   and seed 6 at -.359 (3 threads: +.109, -.087) -- CPU reduction order alone flips weak seeds.
#   The tests below therefore use seeds whose side matched at both thread counts.
#   G_MAP=1 rows are bit-identical to the pre-Methods commit 03fddbf (same seeds, same numbers):
#   under it fg + T=3 COLLAPSED to m ~ 0 in 2/10 (seeds 1, 6; skill .200-.202) and fg + T=2
#   seed 7 stayed undecided (m .41-.59, entropy .999) while predicting at skill .961 -- skill alone
#   does not certify the map. With G_MAP=0 neither happened in 20 fg seeds.
#   d := 0                  skill .199-.208 (T=2 outer seeds 0-2; T=3 outer, fg seed 2), and m does
#                           NOT separate (|sep| <= .0004): under G_MAP=0 m reaches g only through
#                           m*d, so with d = 0 it gets no gradient. (With G_MAP=1 it could:
#                           sep up to .15-.30 at d = 0 -- m was partly a form cue there.)
#   untrained               |sep| <= .0004; AUC .08-.87 on maps that differ by 1e-3, so the MEAN
#                           DIFFERENCE is the primary statistic and AUC is reported, not asserted.
# Thresholds (unchanged): skill >= .8 (the x_t-only ceiling is ~.2); skill - no-dorsal >= .5;
# |sep| >= .15 for outer; sep >= .10 with AUC >= .9 for fg.

KA_S, KA_SPEED, KA_STEPS = 16, 4, 300


def _texture(g, b):
    t = torch.rand(b, 3, H // 8, W // 8, generator=g)
    return 0.15 + 0.45 * F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)


def _ka_batch(g, b, static=False, T=2):
    """(obs_t, obs_{t+1}, coverage of the square at t per map cell); frame k sits at offset
    (k - (T-1)) * speed, so x_t (k = T-1) is at the drawn position for every T."""
    bg = _texture(g, b)
    frames, masks = [bg.clone() for _ in range(T + 1)], []
    y0 = torch.randint(8, H - KA_S - 8, (b,), generator=g)
    x0 = torch.randint(8, W - KA_S - 8, (b,), generator=g)
    dirs = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]])[torch.randint(0, 4, (b,), generator=g)]
    if static:
        dirs = dirs * 0
    for k in range(T + 1):
        m = torch.zeros(b, H, W)
        for i in range(b):
            y = int(y0[i] + (k - (T - 1)) * KA_SPEED * dirs[i, 0])
            x = int(x0[i] + (k - (T - 1)) * KA_SPEED * dirs[i, 1])
            frames[k][i, :, y:y + KA_S, x:x + KA_S] = 1.0
            m[i, y:y + KA_S, x:x + KA_S] = 1
        masks.append(m)
    cell = F.avg_pool2d(masks[T - 1][:, None], 8)[:, 0]          # square coverage per map cell at t
    return torch.cat(frames[:T], 1), torch.cat(frames[1:], 1), cell


def _auc(score, label):
    s, lab = score.flatten(), label.flatten().bool()
    pos, neg = s[lab], s[~lab]
    r = torch.cat([pos, neg]).argsort().argsort().float() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _ka_eval(enc, term):
    T = enc.n_frames
    pt, ptk, cell = _ka_batch(torch.Generator().manual_seed(999), 64, T=T)
    with torch.no_grad():
        term.score(enc, pt, ptk, torch.zeros(64, 2))
        m = enc.encode_streams(pt)["m"][:, 0]
    fg = cell >= 0.5
    return dict(skill=term.last_scalars["skill"], auc=_auc(m, fg),
                diff=float(m[fg].mean() - m[cell == 0].mean()),
                fg_frac=term.last_scalars["fg_frac"], fg_entropy=term.last_scalars["fg_entropy"],
                m_max=float(m.max()), m_min=float(m.min()),
                m_q999=float(m.flatten().quantile(0.999)), m_q001=float(m.flatten().quantile(0.001)),
                combine=term.combine, g_map=term.g_map, T=T,
                fg_motion_sep=term.last_scalars["fg_motion_sep"])


def _ka_train(monkeypatch, combine, seed, no_dorsal=False, T=2, g_map=None):
    """combine=None / g_map=None train the DEFAULT (knob unset)."""
    if combine is not None:
        monkeypatch.setenv("NETT_XSP_COMBINE", combine)
    if g_map is not None:
        monkeypatch.setenv("NETT_XSP_G_MAP", g_map)
    torch.set_num_threads(8)                                    # restored by the autouse fixture
    if no_dorsal:
        orig = XSPEncoder.encode_streams

        def cut(self, p):
            s = orig(self, p)
            s["d"] = torch.zeros_like(s["d"])
            return s

        monkeypatch.setattr(XSPEncoder, "encode_streams", cut)
    torch.manual_seed(seed)
    enc = XSPEncoder(_space(CPF * T), features_dim=32)
    term = XSPTerm(enc)
    opt = torch.optim.Adam(list(enc.parameters()) + list(term.head.parameters()), lr=1e-3)
    g = torch.Generator().manual_seed(100 + seed)
    init = _ka_eval(enc, term)
    for _ in range(KA_STEPS):
        pt, ptk, _ = _ka_batch(g, 16, T=T)
        loss = term.score(enc, pt, ptk, torch.zeros(16, 2))
        opt.zero_grad()
        loss.backward()
        opt.step()
    final = _ka_eval(enc, term)
    print(f"KA T={T} combine={combine} g_map={g_map} seed={seed} no_dorsal={no_dorsal}: init {init} -> final {final}")
    return init, final


@pytest.mark.parametrize("g_map,T,seed", [(None, 2, 2), (None, 2, 8), (None, 3, 0), ("1", 2, 0), ("1", 2, 2)])
def test_known_answer_outer_separates_the_square_on_either_side(monkeypatch, g_map, T, seed):
    """The DEFAULT (outer = the Methods' v (.) d, G_MAP=0) and outer with G_MAP=1: prediction beats
    copy and m separates the square -- on EITHER side. T=2 seeds 2 and 8 land on opposite sides
    (scratch: -.508, +.417), as do G_MAP=1 seeds 2 and 0 (-.477, +.417)."""
    init, final = _ka_train(monkeypatch, None, seed, T=T, g_map=g_map)
    assert final["combine"] == "outer" and final["g_map"] == (g_map == "1") and final["T"] == T
    assert abs(init["diff"]) < 0.01                       # untrained map does not separate it
    assert final["skill"] >= 0.8                          # prediction clearly beats copy
    assert abs(final["diff"]) >= 0.15                     # symmetric loss: no side is "figure"
    assert np.isfinite(final["auc"])


@pytest.mark.parametrize("T,seed", [(2, 1), (2, 2), (3, 2)])
def test_known_answer_fg_puts_the_square_on_the_figure_side(monkeypatch, T, seed):
    """fg (G_MAP=0): only m * d reaches g, so the moving object must be where m is HIGH."""
    init, final = _ka_train(monkeypatch, "fg", seed, T=T)
    assert final["combine"] == "fg" and final["g_map"] is False
    assert abs(init["diff"]) < 0.01
    assert final["skill"] >= 0.8
    assert final["diff"] >= 0.10 and final["auc"] >= 0.9


@pytest.mark.parametrize("T,seed", [(2, 1), (3, 2)])
def test_known_answer_no_dorsal_control_cannot_predict_motion(monkeypatch, T, seed):
    _, full = _ka_train(monkeypatch, None, seed, T=T)
    _, cut = _ka_train(monkeypatch, None, seed, no_dorsal=True, T=T)
    assert cut["skill"] <= 0.4
    assert full["skill"] - cut["skill"] >= 0.5
    assert abs(cut["diff"]) < 0.01      # G_MAP=0: with d = 0, m gets no gradient at all


def _telemetry_flags_collapse(s):
    """What a reader of the run's own logs can see: the map is decided (entropy ~0) and uniform
    (fg_frac at an extreme)."""
    return s["fg_entropy"] < 0.05 and (s["fg_frac"] < 0.05 or s["fg_frac"] > 0.95)


def _map_is_collapsed(r):
    """Ground truth from the map itself, not from the telemetry: 99.9% of the 64 x 160 locations
    saturated to the SAME side (m <= .005, or m >= .995). The binary entropy of .005 is .045 bits,
    so such a map has mean entropy <= .999 x .045 + .001 x 1 = .046 and fg_frac <= .006 (or
    >= .994): inside the telemetry's .05 bounds by construction. A quantile, not the max: the
    T=3 seed-1 collapse (fg_frac 2e-5, entropy 2e-4) has ONE cell at .0057, which an
    every-location rule would call healthy, leaving the check with nothing to check."""
    return r["m_q999"] <= 0.005 or r["m_q001"] >= 0.995


def _check_collapse_detection(r):
    if _map_is_collapsed(r):
        assert _telemetry_flags_collapse(r), r          # every collapse is DETECTED
    if abs(r["diff"]) >= 0.10:
        assert not _telemetry_flags_collapse(r), r      # a separating map raises no alarm


@pytest.mark.parametrize("combine,g_map,T,seed", [
    (None, None, 2, 0), (None, None, 2, 6), ("fg", "1", 3, 1), ("fg", "1", 3, 2)])
def test_known_answer_collapse_whenever_it_happens_is_detected(monkeypatch, combine, g_map, T, seed):
    """Whatever training produces, a collapsed map must be flagged by fg_frac/fg_entropy and a
    separating one must not. This does NOT pin which seeds collapse. The default's weakest seeds
    (T=2 seeds 0 and 6) and fg + G_MAP=1 at T=3 (seed 1 collapsed to m ~ 0 in scratch, seed 2 did
    not) are run; the forced cases below exercise the detector even if no trained seed collapses."""
    _, final = _ka_train(monkeypatch, combine, seed, T=T, g_map=g_map)
    print(f"{combine} g_map={g_map} T={T} seed {seed}: map collapsed={_map_is_collapsed(final)} "
          f"telemetry flags={_telemetry_flags_collapse(final)}")
    assert _map_is_collapsed(final) or abs(final["diff"]) >= 0.05 or final["fg_entropy"] > 0.5, \
        f"neither collapsed, separating nor undecided -- the check below would see nothing: {final}"
    _check_collapse_detection(final)


@pytest.mark.parametrize("bias", [-30.0, 30.0, None])
def test_forced_collapse_is_detected_and_a_healthy_map_is_not(bias):
    """Positive controls for the detector: saturate the sigmoid to all-ground (m = 0) / all-figure
    (m = 1) -- a collapse by construction -- and leave it untrained (undecided, entropy ~1)."""
    torch.manual_seed(0)
    enc = XSPEncoder(_space(), features_dim=32)
    term = XSPTerm(enc)
    if bias is not None:
        with torch.no_grad():
            enc.ventral[-1].weight.zero_()
            enc.ventral[-1].bias.fill_(bias)
    r = _ka_eval(enc, term)
    assert _map_is_collapsed(r) == (bias is not None)
    assert _telemetry_flags_collapse(r) == (bias is not None)
    if bias is not None:
        assert r["fg_frac"] == pytest.approx(0.0 if bias < 0 else 1.0, abs=1e-6)
    _check_collapse_detection(r)


@pytest.mark.parametrize("collapse", [True, False])
def test_collapse_detection_check_goes_red_on_a_blind_detector(monkeypatch, collapse):
    """Mutant: a detector that never fires must fail the check on a real collapse; one that always
    fires must fail it on a separating map."""
    r = {"m_q999": 1e-6, "m_q001": 0.0, "fg_frac": 1e-6, "fg_entropy": 1e-6, "diff": 0.0} if collapse \
        else {"m_q999": 0.99, "m_q001": 0.01, "fg_frac": 0.5, "fg_entropy": 0.9, "diff": 0.4}
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


# ================================================================ 4b. orientation telemetry
#
# fg_motion_sep: a label-free per-brain flag for WHICH side of m holds the moving object (module
# docstring of xsp_aux.py). Measured before pinning, T=2, outer (default), G_MAP=0, 8 threads (as
# _ka_train), the 300-step sweep above; sep = ground-truth square-minus-background separation:
#   seed   0      1      2      3      4      5      6      7      8      9
#   sep   -.111  -.400  -.450  +.289  -.263  -.382  -.359  -.350  +.422  -.495
#   fms   +.002  -.037  -.153  +.067  -.058  -.162  -.076  -.139  +.093  -.154
# Agreement in sign on every seed with |sep| >= .15: 9/9 (seed 0 is below .15 and not counted).
# fms is smaller than sep by construction: the square moves 4 px, so only ~3-6 of the 16 "moving"
# cells carry motion; the rest are zero-motion ties (diluted toward the static mean, sign kept).

@pytest.mark.parametrize("seed,side", [(2, -1), (3, +1), (8, +1), (9, -1)])
def test_motion_orientation_flag_matches_the_true_side(monkeypatch, seed, side):
    """POSITIVE CONTROL: the label-free flag reads the side the square actually took. Both sides
    occur among the pinned seeds (2, 9 low; 3, 8 high) -- a flag stuck at one sign fails."""
    _, final = _ka_train(monkeypatch, None, seed)
    assert final["combine"] == "outer" and final["g_map"] is False
    assert abs(final["diff"]) >= 0.15 and np.sign(final["diff"]) == side
    assert np.sign(final["fg_motion_sep"]) == np.sign(final["diff"]), final


def _motion_batch(b=256, seed=4):
    pt, ptk, _ = _ka_batch(torch.Generator().manual_seed(seed), b)
    return pt[:, -CPF:], ptk[:, -CPF:]


def test_motion_orientation_is_near_zero_for_a_map_unrelated_to_motion():
    """NEGATIVE CONTROL: a constant .5 map gives exactly 0; a random map drawn independently of the
    frames gives |fg_motion_sep| < .02. Positive control on the same frames: a map equal to the
    pooled motion energy itself gives a clearly positive value."""
    x_t, x_next = _motion_batch()
    parked = torch.ones(x_t.shape[0], dtype=torch.bool)
    flat = motion_orientation(torch.full((x_t.shape[0], 1, 10, 16), 0.5), x_t, x_next, parked)
    assert flat["fg_motion_sep"] == 0.0 and flat["fg_motion_n_parked"] == x_t.shape[0]
    rand = torch.rand(x_t.shape[0], 1, 10, 16, generator=torch.Generator().manual_seed(11))
    r = motion_orientation(rand, x_t, x_next, parked)
    assert abs(r["fg_motion_sep"]) < 0.02 and abs(r["fg_motion_sep_parked"]) < 0.02, r
    energy = F.avg_pool2d((x_next - x_t).abs().mean(1, keepdim=True), 8)
    assert motion_orientation(energy / energy.max(), x_t, x_next, parked)["fg_motion_sep"] > 0.05


def test_motion_orientation_known_answer_and_parked_split():
    """Hand-built batch: motion in ONE 8x8 cell (index 2*16+3). Samples 0, 1 parked and moving,
    sample 2 transit and moving, sample 3 parked and STATIC (does not qualify). m = 1 on the moving
    cell only: the moving set is that cell + 15 zero-motion ties, the static set holds no m, so
    sep = 1/16 exactly; with m = 1 - that map, sep = -1/16."""
    x_t = torch.zeros(4, CPF, H, W)
    x_next = x_t.clone()
    x_next[:3, :, 16:24, 24:32] = 1.0
    m = torch.zeros(4, 1, 10, 16)
    m[:, 0, 2, 3] = 1.0
    parked = torch.tensor([True, True, False, True])
    r = motion_orientation(m, x_t, x_next, parked)
    assert r["fg_motion_sep"] == pytest.approx(1 / 16) and r["fg_motion_sep_parked"] == pytest.approx(1 / 16)
    assert r["fg_motion_n_parked"] == 2.0                                  # sample 3 does not qualify
    r = motion_orientation(1 - m, x_t, x_next, parked)
    assert r["fg_motion_sep"] == pytest.approx(-1 / 16)
    r = motion_orientation(m, x_t, x_next, torch.tensor([False, False, False, True]))  # only the static one parked
    assert r["fg_motion_sep_parked"] == NOT_MEASURED and r["fg_motion_n_parked"] == 0.0
    assert r["fg_motion_sep"] == pytest.approx(1 / 16)


def test_motion_orientation_is_not_measured_on_static_frames():
    """x_{t+1} == x_t: no sample has localized motion -> every orientation scalar is NOT_MEASURED
    (n = 0). Positive control: the same model on the moving version of the batch measures it."""
    enc = XSPEncoder(_space(), features_dim=32)
    term = XSPTerm(enc)
    g = torch.Generator().manual_seed(7)
    pt, ptk, _ = _ka_batch(g, 8, static=True)
    assert torch.equal(pt[:, -CPF:], ptk[:, -CPF:])
    with torch.no_grad():
        term.score(enc, pt, ptk, torch.zeros(8, 2))
    s = term.last_scalars
    assert s["fg_motion_sep"] == NOT_MEASURED and s["fg_motion_sep_parked"] == NOT_MEASURED
    assert s["fg_motion_n_parked"] == 0.0
    pt, ptk, _ = _ka_batch(g, 8)
    with torch.no_grad():
        term.score(enc, pt, ptk, torch.zeros(8, 2))
    assert term.last_scalars["fg_motion_sep"] != NOT_MEASURED and term.last_scalars["fg_motion_n_parked"] == 8.0


def test_motion_orientation_leaves_loss_and_grads_bit_identical(monkeypatch):
    """The flag is telemetry only: the same model and batch give a bit-identical loss and gradient
    on every parameter with motion_orientation replaced by a no-op (the replacement is checked to
    have taken effect, so the comparison is not vacuous)."""
    enc = _small()
    term = XSPTerm(enc)
    term.attach_memory(_NoiseMemory())
    params = list(enc.parameters()) + list(term.head.parameters())

    def run():
        for p in params:
            p.grad = None
        torch.manual_seed(5)
        loss = term.compute(enc, None)
        loss.backward()
        return loss.detach().clone(), [None if p.grad is None else p.grad.clone() for p in params]

    loss_on, grads_on = run()
    assert "fg_motion_sep" in term.last_scalars
    monkeypatch.setattr(xsp_aux, "motion_orientation", lambda *a, **k: {})
    loss_off, grads_off = run()
    assert "fg_motion_sep" not in term.last_scalars                      # the no-op really ran
    assert torch.equal(loss_on, loss_off)
    assert sum(g is not None for g in grads_on) > 0
    for a, b in zip(grads_on, grads_off):
        assert (a is None and b is None) or torch.equal(a, b)


# ================================================================ 6. existing-arm invariance
#
# By construction: XSP adds two files and one ENTRY to each of four shared files. The tests below
# prove that from git -- the commit before XSP (the parent of the commit that added
# encoders/xsp.py; HEAD while it is still uncommitted) against the WORKING TREE, so every XSP
# commit and any uncommitted edit is covered, not only the first -- and then check the
# consequence that matters: every pre-existing MODELS label resolves to the identical spec.

SHARED = ("nett_skrl/brain/aux/ppo_aux.py", "nett_skrl/brain/registry.py",
          "nett_skrl/brain/encoders/__init__.py", "examples/campaign_train.py")
NEW_FILES = ("nett_skrl/brain/encoders/xsp.py", "nett_skrl/brain/aux/xsp_aux.py")


def _git(*args):
    # A git hook (pre-push) exports GIT_DIR; inherited, it makes cwd (src_isaac/) the work-tree
    # root, so every repo-relative pathspec below matches nothing. Discover the repo from cwd.
    env = {k: v for k, v in os.environ.items()
           if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_PREFIX", "GIT_COMMON_DIR")}
    try:
        return subprocess.run(["git", *args], cwd=_SRC, capture_output=True, text=True,
                              check=True, env=env).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git unavailable here: {exc}")


def _xsp_base():
    """The last commit before XSP: the parent of the commit that added encoders/xsp.py."""
    added = _git("log", "--diff-filter=A", "--format=%H", "--", NEW_FILES[0]).split()
    if not added:
        pytest.fail(f"no commit adds {NEW_FILES[0]} -- the diff cannot look (refusing to diff against HEAD)")
    return f"{added[-1]}^"


def _diff(*args):
    """base .. the working tree: every XSP commit plus anything not yet committed."""
    return _git("diff", _xsp_base(), *args)


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
    base = _xsp_base()
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


def _literal_keys(src, opener):
    return {ln.split('"')[1] for ln in src.split(opener, 1)[1].split("\n}", 1)[0].splitlines()
            if ln.strip().startswith('"')}


def test_registries_only_gained_the_xsp_entries():
    # Source literal at the XSP base vs source literal now. The LIVE dicts are not compared for
    # equality: other tests register entries at runtime (test_validate.py's "DummyEnc"), so under
    # the full suite the live set depends on test order. The live dicts must still contain both.
    base = _xsp_base()
    for rel, opener, live in (("nett_skrl/brain/aux/ppo_aux.py", "AUX_LOSSES = {", AUX_LOSSES),
                              ("nett_skrl/brain/registry.py", "encoder_mapping: dict", encoder_mapping)):
        old = _literal_keys(_git("show", f"{base}:src_isaac/{rel}"), opener)
        new = _literal_keys((_SRC / rel).read_text(), opener)
        assert old and new - old == {"xsp"} and old <= new, rel
        assert new <= set(live), rel


# ================================================================ 7. mutants
#
# Each mutant breaks ONE guarantee; the matching check above must then fail. A check that still
# passed under its mutant would be one that cannot see what it claims to test.

def _mut_leaky_target(mp):
    """The target from obs[t+1]'s NEXT-NEWEST frame, which IS x_t (copy_mse 0)."""
    mp.setattr(XSPTerm, "_target",
               lambda self, p: F.avg_pool2d(p[:, -2 * self.cpf:-self.cpf], self.downsample))


def _mut_oldest_target(mp):
    """The target from obs[t+1]'s OLDEST frame (= x_t at T=2, x_{t-1} at T=3)."""
    mp.setattr(XSPTerm, "_target", lambda self, p: F.avg_pool2d(p[:, :self.cpf], self.downsample))


def _mut_raw_logits(mp):
    mp.setattr(XSPEncoder, "ventral_map", lambda self, z: self.ventral(z))


def _mut_tanh(mp):
    mp.setattr(XSPEncoder, "ventral_map", lambda self, z: torch.tanh(self.ventral(z)))


def _streams(self, z_frames):
    z_t = z_frames[:, -1]
    return {"z_frames": z_frames, "z_t": z_t, "m": self.ventral_map(z_t),
            "d": self.dorsal(z_frames.flatten(1, 2))}


def _mut_unshared_trunk(mp):
    """Every OLDER frame goes through a perturbed copy of the trunk; x_t through the real one."""
    def es(self, prepared):
        if "_prev_trunk" not in self.__dict__:
            twin = copy.deepcopy(self.trunk)
            with torch.no_grad():
                for p in twin.parameters():
                    p.add_(0.05 * torch.randn_like(p))
            self.__dict__["_prev_trunk"] = twin
        frames = self.split_frames(prepared)
        z = [self.__dict__["_prev_trunk"](f) for f in frames[:-1]] + [self.trunk(frames[-1])]
        return {"x_t": frames[-1], **_streams(self, torch.stack(z, 1))}

    mp.setattr(XSPEncoder, "encode_streams", es)


def _mut_dorsal_ignores_older(mp):
    """phi_d reads x_t's map in every slot: no older frame reaches it."""
    def es(self, prepared):
        frames = self.split_frames(prepared)
        z_t = self.trunk(frames[-1])
        out = _streams(self, torch.stack([self.trunk(f) for f in frames], 1))
        out["d"] = self.dorsal(torch.cat([z_t] * self.n_frames, 1))
        return {"x_t": frames[-1], **out}

    mp.setattr(XSPEncoder, "encode_streams", es)


def _mut_dorsal_ignores_oldest(mp):
    """phi_d reads the two NEWEST maps only (the oldest slot repeats the middle one): at T=3 the
    'whole stack' guarantee breaks while a two-frame check would still pass."""
    def es(self, prepared):
        frames = self.split_frames(prepared)
        z = torch.stack([self.trunk(f) for f in frames], 1)
        out = _streams(self, z)
        z_in = torch.cat([z[:, 1:2], z[:, 1:]], 1) if self.n_frames > 2 else z
        out["d"] = self.dorsal(z_in.flatten(1, 2))
        return {"x_t": frames[-1], **out}

    mp.setattr(XSPEncoder, "encode_streams", es)


def _mut_ventral_reads_the_stack(mp):
    """m_t is computed from the MEAN of all frames' maps, so an older frame leaks into the gate."""
    def es(self, prepared):
        frames = self.split_frames(prepared)
        z = torch.stack([self.trunk(f) for f in frames], 1)
        out = _streams(self, z)
        out["m"] = self.ventral_map(z.mean(1))
        return {"x_t": frames[-1], **out}

    mp.setattr(XSPEncoder, "encode_streams", es)


@pytest.mark.parametrize("mutant,check,T", [
    (_mut_leaky_target, "leak", 2), (_mut_leaky_target, "leak", 3),
    (_mut_oldest_target, "leak", 3),
    (_mut_raw_logits, "sigmoid", 2), (_mut_tanh, "sigmoid", 2), (_mut_raw_logits, "sigmoid", 3),
    (_mut_unshared_trunk, "siamese", 2), (_mut_unshared_trunk, "siamese", 3),
    (_mut_dorsal_ignores_older, "siamese", 2), (_mut_dorsal_ignores_older, "siamese", 3),
    (_mut_dorsal_ignores_oldest, "siamese", 3),
    (_mut_ventral_reads_the_stack, "siamese", 2), (_mut_ventral_reads_the_stack, "siamese", 3),
])
def test_mutant_turns_its_check_red(monkeypatch, mutant, check, T):
    mutant(monkeypatch)
    run = {"leak": lambda: check_target_is_newest_of_next(monkeypatch, T),
           "sigmoid": lambda: check_streams_sigmoid(T=T),
           "siamese": lambda: check_siamese(T=T)}[check]
    with pytest.raises(AssertionError):
        run()


def test_dorsal_ignores_oldest_mutant_is_invisible_at_two_frames(monkeypatch):
    """Why the stack checks run at T=3: at T=2 this mutant IS the correct model."""
    mp_ok = _small(T=2)
    p = _prepared(T=2)
    d_ok = mp_ok.encode_streams(p)["d"]
    _mut_dorsal_ignores_oldest(monkeypatch)
    assert torch.allclose(mp_ok.encode_streams(p)["d"], d_ok, atol=1e-6)
    check_siamese(T=2)
