"""P3 `ego_residual`: the routing uses the action, and the residual is what it cannot explain.

⛔ THE DIAGNOSTICS ARE THE RESULT, NOT THE LOSS. A forward model's loss falls whether it is
compensating ego-motion or has learned the identity; the wave-17 plan's C3 question is answered
by D1-D5, so each is tested against a fixture with a KNOWN answer and against its own null.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.brain.aux.ego_residual_aux import EgoResidualTerm, objectness_from_residual
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.aux.token_term import NOT_MEASURED, TokenWindow
from nett_skrl.brain.aux.with_cltt_ref import WithCLTTRef
from nett_skrl.brain.encoders.compact_vit import CompactViT

H, W, C = 80, 128, 6
SMALL = dict(features_dim=32, patch_size=16, embed_dim=32, depth=1, num_heads=2)
N, N_H, N_W = 40, 5, 8


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("NETT_AUX_EGO_BATCH", "NETT_AUX_EGO_OFFSET", "NETT_AUX_EGO_IDENTITY_BIAS",
                 "NETT_AUX_EGO_TRANSIT_FRAC", "NETT_AUX_EMA_DECAY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(21)
            yield
    finally:
        torch.set_num_threads(threads)


def _encoder(patch=16):
    return CompactViT(gym.spaces.Box(0, 255, shape=(H, W, C), dtype=np.uint8),
                      **{**SMALL, "patch_size": patch})


class _ShiftMemory:
    """Ego-motion as a horizontal shift that MATCHES the recorded turn, plus a centre object.

    Background: vertical stripes translated by `turn * px_per_unit` pixels per step, i.e. exactly
    what a rotation does to a coplanar screen. Object: a bright block in the middle columns that
    moves on its own, so the residual has a correct answer to find.
    """

    def __init__(self, t_max=32, n_env=2, turn=0.25, px_per_unit=16.0, object_px=3):
        cols = torch.arange(W).float()
        obs = torch.zeros(t_max, n_env, H, W, C)
        for e in range(n_env):
            sign = 1.0 if e == 0 else -1.0
            for t in range(t_max):
                phase = t * turn * px_per_unit * sign
                frame = (torch.sin((cols - phase) / 5.0) * 90 + 120).view(1, W).expand(H, W).clone()
                lo = W // 2 - 12 + int(t * object_px) % 24
                frame[H // 2 - 12: H // 2 + 12, lo: lo + 20] = 250.0
                obs[t, e] = frame.unsqueeze(-1).expand(H, W, C)
        acts = torch.zeros(t_max, n_env, 2)
        acts[:, 0, 0] = turn
        acts[:, 1, 0] = -turn
        self.tensors = {
            "observations": obs.clamp(0, 255).to(torch.uint8),
            "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "actions": acts,
        }
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


def _composite(memory=None, enc=None):
    enc = enc or _encoder()
    comp = WithCLTTRef(enc, EgoResidualTerm(enc), "ego_residual")
    comp.attach_memory(memory or _ShiftMemory())
    return enc, comp


# ------------------------------------------------------------------ the routing

def test_routing_rows_are_distributions_and_start_at_the_identity_bias():
    term = EgoResidualTerm(_encoder())
    a = torch.randn(5, 2)
    A = term.routing(a)
    assert A.shape == (5, N, N)
    assert torch.allclose(A.sum(-1), torch.ones(5, N), atol=1e-5)
    # ⛔ Zero-init on the last layer: at step 0 the transport is the SAME for every action.
    assert torch.allclose(A, term.routing(torch.zeros(5, 2)), atol=1e-6)


@pytest.mark.parametrize("bias,expected", [(4.0, math.e ** 4 / (math.e ** 4 + N - 1)),
                                           (10.0, math.e ** 10 / (math.e ** 10 + N - 1))])
def test_the_identity_bias_is_calibrated_and_the_manufactured_null_is_reachable(
        monkeypatch, bias, expected):
    """⛔ c = 10 pins p_diag at 0.998 with a softmax gradient factor ~0.002, so the routing would
    sit at the identity whatever the action is -- D5 would then read "not engaged" because of the
    knob, not because of the data. The default c = 4 gives ~0.58. Both are asserted so the
    calibration is a measured property of the code, not a claim in a comment."""
    monkeypatch.setenv("NETT_AUX_EGO_IDENTITY_BIAS", str(bias))
    term = EgoResidualTerm(_encoder())
    p_diag = float(term.routing(torch.zeros(1, 2)).diagonal(dim1=1, dim2=2).mean())
    assert p_diag == pytest.approx(expected, rel=1e-4)


# ------------------------------------------------------------------ gradients and learning

def test_gradient_reaches_the_trunk_and_both_heads_but_never_the_teacher():
    enc, comp = _composite()
    loss = comp.term.compute(enc, None)
    loss.backward()
    assert enc.patch_embed.weight.grad.abs().sum() > 0
    for name, p in comp.term.head.named_parameters():
        assert p.grad is not None, name
    # ⚠ The route head's FIRST layer sees exactly zero gradient on the first backward -- the
    # zero-init last layer blocks it -- and unblocks after one optimizer step. Both halves are
    # asserted, because "zero grad on the action head" read at step 0 looks exactly like a dead
    # pathway and is not one.
    assert comp.term.head["route"][0].weight.grad.abs().sum() == 0
    assert comp.term.head["route"][-1].weight.grad.abs().sum() > 0
    torch.optim.SGD(comp.term.head.parameters(), lr=1.0).step()
    comp.term.head.zero_grad(set_to_none=True)
    enc.zero_grad(set_to_none=True)
    comp.term.compute(enc, None).backward()
    assert comp.term.head["route"][0].weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in comp._teacher.module.parameters())


def test_the_loss_falls_on_a_fixture_built_from_a_known_shift():
    enc, comp = _composite()
    term = comp.term
    window = term.draw_mixed(enc, transit_frac=term.transit_frac)
    opt = torch.optim.Adam([*enc.parameters(), *term.head.parameters()], lr=3e-3)
    losses = []
    for _ in range(12):
        loss, _w, _s = term.loss_and_objectness(enc, window)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] - 1e-3, losses


# ------------------------------------------------------------------ residual and objectness

def _shift_matrix(shift: int) -> torch.Tensor:
    """The ideal column-transport: destination token i reads the source `shift` columns away."""
    A = torch.zeros(N, N)
    for r in range(N_H):
        for c in range(N_W):
            src = (c + shift) % N_W          # wrap, matching the rolled fixture below
            A[r * N_W + c, r * N_W + src] = 1.0
    return A


def _teacher_tokens(shift: int, object_cols=(3, 4), object_delta=1.5, d=12):
    """u0 and u1 where the whole grid moved by `shift` columns and the CENTRE also changed."""
    torch.manual_seed(2)
    base = torch.randn(1, N_H, N_W, d)
    u0 = base.clone()
    u1 = torch.roll(base, shifts=-shift, dims=2)
    for c in object_cols:
        u1[:, :, c] = u1[:, :, c] + object_delta * torch.randn(1, N_H, d)
    return u0.reshape(1, N, d), u1.reshape(1, N, d)


def test_the_residual_concentrates_on_the_object_only_once_the_routing_compensates_ego():
    """⛔ THE DISCRIMINATING CELL. With the ego transport applied, what is left is the centre
    change; with the identity transport (an uncompensated routing) the whole frame is residual
    and the centre ratio collapses toward the permutation null."""
    term = EgoResidualTerm(_encoder())
    u0, u1 = _teacher_tokens(shift=2)
    e_compensated = term._residual(_shift_matrix(2)[None], u0, u1)
    e_identity = term._residual(torch.eye(N)[None], u0, u1)
    centre, periph, _ = term._columns()

    def ratio(e):
        return float(e[:, centre].mean() / e[:, periph].mean())

    assert ratio(e_compensated) > 3.0
    assert ratio(e_compensated) > 2 * ratio(e_identity)


def test_objectness_is_scale_free_bounded_and_detached():
    e = torch.rand(3, N) + 0.1
    w = objectness_from_residual(e)
    assert w.shape == (3, N) and not w.requires_grad
    assert float(w.max()) == pytest.approx(1.0) and float(w.min()) > 0
    # A z-score is invariant to affine rescaling of the residual, which is what makes w readable
    # while the token norms are still moving.
    assert torch.allclose(w, objectness_from_residual(e * 7.0 + 3.0), atol=1e-5)
    assert int(w.argmax(dim=1)[0]) == int(e.argmax(dim=1)[0])


# ------------------------------------------------------------------ D2/D4/D5 have power

class _StubTeacher:
    def __init__(self, u0, u1):
        self.u0, self.u1, self.calls, self.updates = u0, u1, 0, 0

    def step(self, encoder):
        self.updates += 1

    def tokens(self, prepared):
        self.calls += 1
        return (self.u0 if self.calls % 2 else self.u1), (N_H, N_W)


def _window(batch, turn):
    a = torch.zeros(batch, 2)
    a[:, 0] = turn
    prepared = torch.zeros(batch, C, H, W)
    return TokenWindow(prepared_t=prepared, prepared_tk=prepared, actions=a[:, None, :],
                       a_bar=a, mean_turn=float(turn.abs().mean()), env=0, t0=0)


def test_action_gain_is_exactly_one_while_the_routing_ignores_the_action():
    """At zero-init the routing is action-blind, so permuting ā cannot change the residual. An
    action gain of exactly 1 is the honest "not engaged yet" reading at step 0."""
    term = EgoResidualTerm(_encoder())
    batch = 8
    u0, u1 = _teacher_tokens(shift=2)
    u0, u1 = u0.expand(batch, -1, -1).contiguous(), u1.expand(batch, -1, -1).contiguous()
    window = _window(batch, torch.linspace(-1, 1, batch))
    A = term.routing(window.a_bar)
    e = term._residual(A, u0, u1)
    w = objectness_from_residual(e)
    d = term._diagnostics(window, A, u0, u1, e, w, torch.randn(batch, N, term.token_dim))
    assert d["action_gain"] == pytest.approx(1.0, abs=1e-6)
    assert d["route_diag_top_quartile"] == pytest.approx(d["route_diag_bottom_quartile"], abs=1e-6)


def test_action_gain_exceeds_one_when_the_routing_genuinely_uses_the_action():
    """Hand the routing the ideal action->shift map: permuting ā then applies the WRONG transport
    and the residual rises. That is D2's engaged reading, produced by construction."""
    term = EgoResidualTerm(_encoder())
    batch = 8
    turn = torch.tensor([-2.0, -2, -1, -1, 1, 1, 2, 2])
    u0 = torch.randn(batch, N, 10)
    shifts = turn.long()
    u1 = torch.stack([(u0[b].view(N_H, N_W, -1).roll(-int(shifts[b]), dims=1)).reshape(N, -1)
                      for b in range(batch)])
    window = _window(batch, turn)
    A = torch.stack([_shift_matrix(int(shifts[b])) for b in range(batch)])
    e = term._residual(A, u0, u1)
    A_perm_source = torch.stack([_shift_matrix(int(s)) for s in shifts])
    # Emulate _diagnostics' permuted-action branch with the same ideal map.
    perm = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3])
    e_perm = term._residual(A_perm_source[perm], u0, u1)
    assert float(e_perm.mean() / e.mean()) > 2.0


def _objectness_diag(u0, u1):
    term = EgoResidualTerm(_encoder())
    batch = u0.shape[0]
    window = _window(batch, torch.zeros(batch))
    A = torch.eye(N)[None].expand(batch, -1, -1)
    e = term._residual(A, u0, u1)
    w = objectness_from_residual(e)
    return term._diagnostics(window, A, u0, u1, e, w, torch.randn(batch, N, term.token_dim))


def test_the_objectness_null_is_an_EXACTNESS_test_for_the_fixed_map_artefact():
    """⛔ The mode D4 exists for: an objectness map that does not depend on WHICH frames were
    paired (a bezel edge, a disocclusion band). Then permuting the destination frames changes
    nothing and the excess is EXACTLY 0.0. With a per-sample object it is nonzero -- and the SIGN
    is not the reading, because a destroyed pairing raises the residual everywhere. This is the
    same correction slot_contrast_aux's 200-seed sweep forced on `mask_variance_excess`."""
    batch = 6
    shared = torch.randn(1, N, 10)
    fixed = _objectness_diag(torch.randn(batch, N, 10), shared.expand(batch, -1, -1).contiguous())
    assert fixed["objectness_input_dep"] == 0.0
    assert fixed["objectness_var"] == fixed["objectness_var_null"]

    u0 = torch.randn(batch, N, 10)
    u1 = u0.clone()
    for b in range(batch):
        u1[b, (b * 3) % N] += 5.0                 # a different "object" token per sample
    varying = _objectness_diag(u0, u1)
    assert abs(varying["objectness_input_dep"]) > 1e-4


def test_a_constant_encoder_is_caught_by_token_std_not_by_the_loss():
    """⛔ The collapse mode the EMA target makes unreachable-by-gradient but does not PROVE away:
    constant tokens. The loss says nothing; `token_std` goes to 0."""
    term = EgoResidualTerm(_encoder())
    batch = 4
    u0, u1 = torch.randn(batch, N, 10), torch.randn(batch, N, 10)
    window = _window(batch, torch.zeros(batch))
    A = term.routing(window.a_bar)
    e = term._residual(A, u0, u1)
    w = objectness_from_residual(e)
    live = term._diagnostics(window, A, u0, u1, e, w, torch.randn(batch, N, term.token_dim))
    dead = term._diagnostics(window, A, u0, u1, e, w, torch.ones(batch, N, term.token_dim))
    assert dead["token_std"] == pytest.approx(0.0, abs=1e-6)
    assert live["token_std"] > 0.05


# ------------------------------------------------------------------ sampling, knobs, registry

def test_the_window_is_a_half_transit_half_uniform_mixture(monkeypatch):
    monkeypatch.setenv("NETT_AUX_EGO_BATCH", "8")
    enc, comp = _composite()
    draws = []
    real_draw = comp.term.draw

    def spy(encoder, *, transit_weighted=None, batch=None):
        draws.append((transit_weighted, batch))
        return real_draw(encoder, transit_weighted=transit_weighted, batch=batch)

    comp.term.draw = spy
    window = comp.term.draw_mixed(enc, transit_frac=comp.term.transit_frac)
    assert draws == [(True, 4), (False, 4)]
    assert window.prepared_t.shape[0] == window.a_bar.shape[0] <= 8


def test_transit_fraction_one_draws_only_weighted_windows(monkeypatch):
    monkeypatch.setenv("NETT_AUX_EGO_TRANSIT_FRAC", "1")
    monkeypatch.setenv("NETT_AUX_EGO_BATCH", "6")
    enc, comp = _composite()
    draws = []
    real_draw = comp.term.draw
    comp.term.draw = lambda e, *, transit_weighted=None, batch=None: (
        draws.append(transit_weighted) or real_draw(e, transit_weighted=transit_weighted,
                                                    batch=batch))
    comp.term.draw_mixed(enc, transit_frac=comp.term.transit_frac)
    assert draws == [True]


def test_knobs_are_read_and_invalid_values_raise(monkeypatch):
    monkeypatch.setenv("NETT_AUX_EGO_BATCH", "11")
    monkeypatch.setenv("NETT_AUX_EGO_OFFSET", "4")
    monkeypatch.setenv("NETT_AUX_EGO_IDENTITY_BIAS", "0")
    monkeypatch.setenv("NETT_AUX_EGO_TRANSIT_FRAC", "0.25")
    term = EgoResidualTerm(_encoder())
    assert (term.batch, term.offset, term.identity_bias, term.transit_frac) == (11, 4, 0.0, 0.25)
    for name, bad in (("NETT_AUX_EGO_BATCH", "0"), ("NETT_AUX_EGO_OFFSET", "x"),
                      ("NETT_AUX_EGO_IDENTITY_BIAS", "-1"),
                      ("NETT_AUX_EGO_TRANSIT_FRAC", "1.5")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError, match=name):
            EgoResidualTerm(_encoder())
        monkeypatch.delenv(name)


def test_every_diagnostic_is_emitted_on_every_call():
    enc, comp = _composite()
    comp.term.compute(enc, None)
    keys = set(comp.term.last_scalars)
    assert {"res_centre_ratio", "res_centre_ratio_parked", "res_centre_ratio_transit",
            "res_centre_ratio_noedge", "res_centre_ratio_null", "pix_centre_ratio",
            "action_gain", "action_gain_transit", "compensation_gain", "objectness_input_dep",
            "route_diag_top_quartile", "route_diag_bottom_quartile", "route_p_diag_at_zero",
            "token_std", "residual_mean", "res_ratio_rho_turn"} <= keys


def test_parameter_counts_are_what_the_spec_derived():
    term = EgoResidualTerm(_encoder())
    route = sum(p.numel() for p in term.head["route"].parameters())
    predict = sum(p.numel() for p in term.head["predict"].parameters())
    assert route == 2 * 64 + 64 + 64 * N * N + N * N
    assert predict == 2 * (term.token_dim * 2 * term.token_dim) + 2 * term.token_dim + term.token_dim


def test_the_row_is_registered_as_cltt_ref_plus_one_term():
    aux = AUX_LOSSES["ego_residual"](_encoder())
    assert isinstance(aux, WithCLTTRef) and isinstance(aux.term, EgoResidualTerm)
    assert aux.name == "ego_residual" and aux._teacher is not None
