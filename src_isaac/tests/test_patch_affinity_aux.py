"""P2 `patch_affinity`: the target is the reference's, and every degenerate mode is detected.

⛔ A FALLING CE CANNOT TELL ANY OF THESE APART, which is why the diagnostics are tested as hard
as the loss: an identity target (k too small / parked), an image-blind positional target, a
uniform target (τ too high) and the threshold wipe all produce a well-behaved loss curve.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.aux.patch_affinity_aux import PatchAffinityTerm, affinity_target
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.aux.token_term import NOT_MEASURED
from nett_skrl.brain.aux.with_cltt_ref import WithCLTTRef
from nett_skrl.brain.encoders.compact_vit import CompactViT

H, W, C = 80, 128, 6
SMALL = dict(features_dim=32, patch_size=16, embed_dim=32, depth=1, num_heads=2)
N = 40


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("NETT_AUX_AFF_BATCH", "NETT_AUX_AFF_OFFSET", "NETT_AUX_AFF_TEMP",
                 "NETT_AUX_EMA_DECAY", "NETT_AUX_CLTT_REF_WEIGHT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_AUX_BATCH", "4")          # cltt_ref's own knob, kept small
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            yield
    finally:
        torch.set_num_threads(threads)


class _Memory:
    """Rollout memory with a HORIZONTAL SHIFT that matches the recorded turn command.

    Frames are a vertical-bar pattern shifted by `shift(t)` pixels; the action at row t is the
    per-step shift, so the cumulative action over [t, t+k) IS the image displacement. That gives
    the correspondence something correct to find.
    """

    def __init__(self, t_max=24, n_env=2, px_per_step=4, noise=0.0):
        cols = torch.arange(W).float()
        obs = torch.zeros(t_max, n_env, H, W, C)
        for t in range(t_max):
            for e in range(n_env):
                phase = (t * px_per_step * (1 if e == 0 else -1))
                pattern = (torch.sin((cols - phase) / 6.0) * 110 + 128)
                obs[t, e] = pattern.view(1, W, 1).expand(H, W, C)
        if noise:
            obs = obs + torch.randn_like(obs) * noise
        self.tensors = {
            "observations": obs.clamp(0, 255).to(torch.uint8),
            "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            # actions[..., 0] = turn = the per-step shift, opposite sign per env
            "actions": torch.stack([
                torch.tensor([[float(px_per_step), -float(px_per_step)]] * t_max),
                torch.zeros(t_max, n_env),
            ], dim=-1),
        }
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


def _encoder(patch=16):
    return CompactViT(gym.spaces.Box(0, 255, shape=(H, W, C), dtype=np.uint8),
                      **{**SMALL, "patch_size": patch})


def _composite(enc=None, memory=None):
    enc = enc or _encoder()
    comp = WithCLTTRef(enc, PatchAffinityTerm(enc), "patch_affinity")
    comp.attach_memory(memory or _Memory())
    return enc, comp


# ------------------------------------------------------------------ the target is the reference's

def test_target_rows_are_a_distribution_over_destination_tokens():
    u = torch.randn(3, N, 16)
    t, neg = affinity_target(u, torch.randn(3, N, 16), 0.1)
    assert t.shape == (3, N, N)
    assert torch.allclose(t.sum(-1), torch.ones(3, N), atol=1e-5)
    assert neg.shape == (3, N)


def test_negative_similarities_are_thresholded_before_the_temperature():
    """s < 0 -> -inf, so a negative-cosine destination gets EXACTLY zero mass at any τ."""
    u_t = torch.tensor([[[1.0, 0.0]]])
    u_tk = torch.tensor([[[1.0, 0.0], [-1.0, 0.0], [0.6, 0.8]]])
    t, neg = affinity_target(u_t, u_tk, 0.25)
    assert float(t[0, 0, 1]) == 0.0
    assert not bool(neg.any())


def test_a_row_of_only_negative_similarities_becomes_uniform_not_nan():
    u_t = torch.tensor([[[1.0, 0.0]]])
    u_tk = torch.tensor([[[-1.0, 0.0], [-0.6, -0.8]]])
    t, neg = affinity_target(u_t, u_tk, 0.1)
    assert bool(neg[0, 0]) and torch.allclose(t[0, 0], torch.full((2,), 0.5))


def test_a_constant_teacher_gives_the_MAXIMUM_loss_floor_not_the_minimum():
    """⛔ The anti-collapse property, stated as a number: a constant trunk makes every cosine 1,
    so T is uniform and the CE floor is ln N -- the worst attainable, not the best."""
    const = torch.ones(2, N, 8)
    t_const, _ = affinity_target(const, const, 0.1)
    h_const = float(-(t_const * t_const.clamp_min(1e-12).log()).sum(-1).mean())
    struct = torch.randn(2, N, 8)
    t_struct, _ = affinity_target(struct, struct.roll(3, dims=1), 0.1)
    h_struct = float(-(t_struct * t_struct.clamp_min(1e-12).log()).sum(-1).mean())
    assert h_const == pytest.approx(math.log(N), abs=1e-5)
    assert h_struct < h_const - 0.5


# ------------------------------------------------------------------ gradients and the teacher

def test_gradient_reaches_the_trunk_and_never_the_teacher():
    enc, comp = _composite()
    term = comp.term
    loss = term.compute(enc, None)
    loss.backward()
    assert enc.patch_embed.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad
               for p in comp._teacher.module.parameters())
    for name, p in term.head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_the_loss_falls_on_a_fixture_whose_structure_it_can_learn():
    enc, comp = _composite()
    term = comp.term
    window = term.draw(enc)                    # ONE fixed window, so the target is fixed
    opt = torch.optim.Adam([*enc.parameters(), *term.head.parameters()], lr=3e-3)
    losses = []
    for _ in range(12):
        loss, _ = term._core(enc, window)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] - 1e-3, losses
    # And it cannot fall below the target's own entropy: the CE floor is H(T).
    _, scalars = term._core(enc, window)
    assert losses[-1] > scalars["ce_floor"] - 1e-3


# ------------------------------------------------------------------ diagnostics have power

class _StubTeacher:
    """Teacher tokens with a KNOWN structure, so each diagnostic can be checked against truth."""

    def __init__(self, mode, n=N, d=12, batch=8, n_w=8):
        self.mode, self.n, self.d, self.batch, self.n_w = mode, n, d, batch, n_w
        torch.manual_seed(4)
        self.base = torch.randn(batch, n, d)
        self.calls = 0
        self.updates = 0
        self.shifts = None

    def step(self, encoder):
        self.updates += 1

    def tokens(self, prepared):
        self.calls += 1
        if self.mode == "identity":
            return self.base.clone(), (5, self.n_w)
        if self.mode == "image_blind":
            one = torch.randn(1, self.n, self.d).expand(self.batch, -1, -1).contiguous()
            return one, (5, self.n_w)
        if self.mode == "shift":
            if self.calls == 1:
                return self.base.clone(), (5, self.n_w)
            # Roll along the COLUMN axis of the (5, 8) grid: that is what an ego rotation does
            # to a coplanar screen, and it is what `column_shift` is built to recover. A roll
            # over the FLAT token index instead would be a column shift modulo 8 whose mean over
            # a full grid is 0 for every shift -- i.e. a null fixture wearing a signal's name.
            grid = self.base.view(self.batch, 5, self.n_w, self.d)
            out = torch.stack([grid[b].roll(int(self.shifts[b]), dims=1)
                               for b in range(self.batch)])
            return out.reshape(self.batch, self.n, self.d), (5, self.n_w)
        raise AssertionError(self.mode)


def _diag_with_stub(mode, shifts=None, batch=8):
    enc = _encoder()
    term = PatchAffinityTerm(enc)
    stub = _StubTeacher(mode, batch=batch)
    stub.shifts = shifts
    term.attach_teacher(stub)
    turn = (shifts.float() if shifts is not None else torch.zeros(batch))
    window = type("W", (), {"turn": turn.abs(), "a_bar": torch.stack([turn, turn * 0], -1)})()
    u_t, _ = stub.tokens(None)
    u_tk, _ = stub.tokens(None)
    target, neg = affinity_target(u_t, u_tk, term.temperature)
    logits = torch.zeros_like(target)
    return term._diagnostics(window, target, neg, u_t, u_tk, logits)


def test_identity_target_reads_as_identity():
    d = _diag_with_stub("identity")
    assert d["identity_frac"] == 1.0


def test_an_image_blind_target_is_sharp_but_fails_the_paired_input_dependence_null():
    """⛔ THE MODE A SHARP TARGET HIDES. Entropy and identity both look healthy; only the
    across-batch variance against the permuted-pairing null says the target ignores the input."""
    blind = _diag_with_stub("image_blind")
    real = _diag_with_stub("shift", shifts=torch.tensor([-3, -2, -1, 1, 2, 3, -2, 2]))
    assert blind["target_entropy_frac"] < 0.9
    assert abs(blind["input_dep"]) < 1e-6
    assert real["input_dep"] > 1e-4
    assert real["target_var"] > real["target_var_null"]


def test_the_column_shift_correlates_with_the_action_and_the_permuted_null_does_not():
    shifts = torch.tensor([-3, -2, -1, 1, 2, 3])
    d = _diag_with_stub("shift", shifts=shifts, batch=6)
    assert abs(d["shift_rho"]) > 0.8
    assert abs(d["shift_rho"]) > abs(d["shift_rho_null"]) + 0.2
    # A rolled correspondence is NOT the identity: this is the engaged reading.
    assert d["identity_frac"] < 0.2


def test_every_statistic_is_emitted_on_every_call_with_sentinels_never_silence():
    enc, comp = _composite()
    comp.term.compute(enc, None)
    keys = set(comp.term.last_scalars)
    assert {"identity_frac", "identity_frac_parked", "identity_frac_transit", "shift_rho",
            "shift_rho_null", "target_entropy_frac", "rows_all_neg", "input_dep", "ce",
            "ce_floor", "temperature", "B", "k", "window_turn"} <= keys
    assert all(isinstance(v, float) for v in comp.term.last_scalars.values())


def test_a_single_sample_batch_reports_sentinels_rather_than_a_plausible_zero(monkeypatch):
    d = _diag_with_stub("identity", batch=1)
    assert d["target_var"] == NOT_MEASURED and d["input_dep"] == NOT_MEASURED
    assert d["shift_rho"] == NOT_MEASURED


# ------------------------------------------------------------------ knobs, shapes, registry

def test_head_shape_is_probed_from_the_encoder_not_hardcoded():
    for patch, n in ((16, 40), (8, 160)):
        term = PatchAffinityTerm(_encoder(patch))
        assert term.n_tokens == n and term.head[-1].out_features == n
        assert term.grid == (H // patch, W // patch)


def test_knobs_are_read_and_invalid_values_raise(monkeypatch):
    monkeypatch.setenv("NETT_AUX_AFF_BATCH", "7")
    monkeypatch.setenv("NETT_AUX_AFF_OFFSET", "3")
    monkeypatch.setenv("NETT_AUX_AFF_TEMP", "0.25")
    term = PatchAffinityTerm(_encoder())
    assert (term.batch, term.offset, term.temperature) == (7, 3, 0.25)
    for name, bad in (("NETT_AUX_AFF_BATCH", "0"), ("NETT_AUX_AFF_OFFSET", "-1"),
                      ("NETT_AUX_AFF_TEMP", "0"), ("NETT_AUX_AFF_TEMP", "abc")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError, match=name):
            PatchAffinityTerm(_encoder())
        monkeypatch.delenv(name)


def test_it_never_reads_the_colliding_shared_batch_knob(monkeypatch):
    """⛔ NETT_AUX_BATCH is cltt_ref's (default 512). If this term read it too, setting the
    composite's cltt_ref batch would silently resize the affinity batch as well."""
    monkeypatch.setenv("NETT_AUX_BATCH", "123")
    assert PatchAffinityTerm(_encoder()).batch == 32


def test_the_row_is_registered_as_cltt_ref_plus_one_term():
    enc = _encoder()
    aux = AUX_LOSSES["patch_affinity"](enc)
    assert isinstance(aux, WithCLTTRef) and isinstance(aux.term, PatchAffinityTerm)
    assert aux.name == "patch_affinity" and aux.cltt_ref_weight == 1.0
    assert aux._teacher is not None
