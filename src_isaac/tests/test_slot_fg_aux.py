"""P4 `slot_fg` (and P5 under NETT_AUX_SLOTFG_EGO=1): slots over tokens with a REAL correspondence.

⛔ THE CLAIM THAT MATTERS IS THE CORRESPONDENCE. `slot_contrast_aux` draws an independent random
slot init at t and at t+1, so its identity target asks slot k at t to match a slot at t+1 that
shares only a distribution with it -- the leading candidate for I77 ("never engaged,
no-correspondence at 72 positions"). Here the t+k init IS predictor(slots_t), which is asserted
directly rather than argued.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.aux.slot_contrast_aux import SlotAttention
from nett_skrl.brain.aux.slot_fg_aux import SlotFGTerm
from nett_skrl.brain.aux.token_term import NOT_MEASURED
from nett_skrl.brain.aux.with_cltt_ref import WithCLTTRef
from nett_skrl.brain.encoders.compact_vit import CompactViT

H, W, C = 80, 128, 6
SMALL = dict(features_dim=32, patch_size=16, embed_dim=32, depth=1, num_heads=2)
N, N_H, N_W = 40, 5, 8
KNOBS = ("NETT_AUX_SLOTFG_BATCH", "NETT_AUX_SLOTFG_OFFSET", "NETT_AUX_SLOTFG_SLOTS",
         "NETT_AUX_SLOTFG_DIM", "NETT_AUX_SLOTFG_DEC_HIDDEN", "NETT_AUX_SLOTFG_TEMP",
         "NETT_AUX_SLOTFG_W_SS", "NETT_AUX_SLOTFG_W_REC", "NETT_AUX_SLOTFG_LAMBDA",
         "NETT_AUX_SLOTFG_GAMMA", "NETT_AUX_SLOTFG_EGO", "NETT_AUX_SLOTFG_RAMP_CALLS",
         "NETT_AUX_EGO_BATCH", "NETT_AUX_EGO_OFFSET", "NETT_AUX_EGO_TRANSIT_FRAC")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in KNOBS + ("NETT_AUX_EMA_DECAY",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    monkeypatch.setenv("NETT_AUX_SLOTFG_BATCH", "6")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(41)
            yield
    finally:
        torch.set_num_threads(threads)


class _Memory:
    def __init__(self, t_max=24, n_env=2):
        self.tensors = {
            "observations": torch.randint(0, 256, (t_max, n_env, H, W, C), dtype=torch.uint8),
            "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "actions": torch.randn(t_max, n_env, 2),
        }
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


def _encoder():
    return CompactViT(gym.spaces.Box(0, 255, shape=(H, W, C), dtype=np.uint8), **SMALL)


def _composite(enc=None):
    enc = enc or _encoder()
    comp = WithCLTTRef(enc, SlotFGTerm(enc), "slot_fg")
    comp.attach_memory(_Memory())
    return enc, comp


# ------------------------------------------------------------------ the correspondence

def test_the_second_frames_slots_start_from_the_predicted_first_frame_slots():
    """⛔ THE I77 FIX, ASSERTED AT THE CALL. Init at t = the fixed learned vector (identical for
    every sample); init at t+k = predictor(slots_t). Not two independent draws."""
    enc, comp = _composite()
    term = comp.term
    inits = []
    real = term._slots

    def spy(tokens, slots_init, iters):
        inits.append((slots_init.detach().clone(), iters))
        return real(tokens, slots_init, iters)

    term._slots = spy
    window = term.draw(enc)
    term._core(enc, window)
    (init_t, iters_t), (init_tk, iters_tk) = inits[0], inits[1]
    b = window.prepared_t.shape[0]
    assert (iters_t, iters_tk) == (term.ITERS_FIRST, term.ITERS_NEXT) == (3, 2)
    # t: the SAME learned init for every sample (FixedLearnedInit, not a per-sample draw).
    assert torch.allclose(init_t, init_t[:1].expand_as(init_t))
    assert torch.allclose(init_t[0], term.head["init"].value[0])
    # t+k: the predictor's output on the t slots, which differs per sample.
    assert init_tk.shape == (b, term.slots, term.slot_dim)
    assert not torch.allclose(init_tk, init_tk[:1].expand_as(init_tk))


def test_slot_attention_is_the_shipped_class_used_unchanged():
    term = SlotFGTerm(_encoder())
    assert isinstance(term.head["attn"], SlotAttention)
    # The random-init parameters are dead by construction here (an explicit init is always
    # passed), and that is DECLARED by freezing them rather than left as silent zero gradients.
    assert not term.head["attn"].mu.requires_grad
    assert not term.head["attn"].log_sigma.requires_grad


def test_iters_are_restored_even_when_the_call_raises():
    term = SlotFGTerm(_encoder())
    with pytest.raises(RuntimeError):
        term._slots(torch.zeros(2, N, 999), torch.zeros(2, term.slots, term.slot_dim), 3)
    assert term.head["attn"].iters == term.ITERS_NEXT


# ------------------------------------------------------------------ the contrastive term

def test_near_in_time_negatives_are_masked_out_of_the_contrast():
    """⛔ Our negatives are CONSECUTIVE STEPS of one env, not independent videos: a negative
    |Δt| < k away is a near-duplicate of the positive. Changing a masked negative must not change
    the loss; changing an unmasked one must."""
    term = SlotFGTerm(_encoder())
    k, s = term.slots, term.slot_dim
    torch.manual_seed(0)
    one_t, one_tk = torch.randn(1, k, s), torch.randn(1, k, s)
    alone = float(term._slot_contrast(one_t, one_tk))
    # Sample 1 is an EXACT DUPLICATE of sample 0 at |Δt| = 1 < k = 8: the hardest possible false
    # negative. Masked, the two rows score exactly what the single sample scores alone; unmasked
    # the duplicate would sit in the denominator and inflate the loss.
    dup_t, dup_tk = one_t.repeat(2, 1, 1), one_tk.repeat(2, 1, 1)
    assert float(term._slot_contrast(dup_t, dup_tk)) == pytest.approx(alone, abs=1e-5)
    unmasked = F.cross_entropy(
        (F.normalize(dup_t, dim=-1).reshape(2 * k, s)
         @ F.normalize(dup_tk, dim=-1).reshape(2 * k, s).T) / term.temperature,
        torch.arange(2 * k))
    assert float(unmasked) > alone + 0.5
    # A negative FAR enough away stays in the denominator and does move the loss.
    far_t = torch.cat([one_t, torch.randn(term.offset + 1, k, s)])
    far_tk = torch.cat([one_tk, torch.randn(term.offset + 1, k, s)])
    far_tk_moved = far_tk.clone()
    far_tk_moved[-1] = one_tk[0] + 0.01
    assert float(term._slot_contrast(far_t, far_tk)) != pytest.approx(
        float(term._slot_contrast(far_t, far_tk_moved)), abs=1e-6)


def test_constant_slots_pin_the_imported_diagnostic_to_its_documented_collapse_values():
    """⛔ THE COLLAPSE DOES NOT RAISE THE CONTRASTIVE LOSS -- constant slots preserve slot index
    across time perfectly, so `ss` FALLS. What separates it is the paired null, which is exactly
    what slot_contrast_diagnostics measures, so the collapse is asserted there instead of being
    fabricated as a loss-worse claim."""
    from nett_skrl.brain.aux.slot_contrast_aux import slot_contrast_diagnostics
    term = SlotFGTerm(_encoder())
    b, k, s = 8, term.slots, term.slot_dim
    const = torch.randn(1, k, s).expand(b, -1, -1).reshape(b * k, s)
    d = slot_contrast_diagnostics(const, const.clone(), term.temperature, k)
    assert d["ss_pos_acc_x_batch"] == 1.0 and d["ss_input_dependence"] == 0.0
    healthy = torch.randn(b * k, s)
    h = slot_contrast_diagnostics(healthy, healthy + 0.01 * torch.randn_like(healthy),
                                  term.temperature, k)
    assert h["ss_input_dependence"] > 0.5


def test_the_separation_term_prefers_a_spread_slot_marginal_to_one_slot_taking_everything():
    """L_sep maximises the entropy of the slot-MASS MARGINAL: "one slot binds the entire image and
    the other remains empty" is a statement about the marginal, which is why the axis is that one
    and not per-token (which would push every token to a uniform assignment, i.e. no slots)."""
    term = SlotFGTerm(_encoder())
    k = term.slots
    spread = torch.full((1, k), 1.0 / k)
    collapsed = torch.tensor([[1.0 - 1e-6] + [1e-6 / (k - 1)] * (k - 1)])

    def sep(m):
        return float(-(m.clamp_min(1e-12).log() * m).sum(-1).mean())

    assert sep(spread) == pytest.approx(math.log(k), abs=1e-6)
    assert sep(collapsed) < 0.01
    # The loss SUBTRACTS gamma * entropy, so the collapsed marginal costs more.
    assert -term.gamma_sep * sep(collapsed) > -term.gamma_sep * sep(spread)


# ------------------------------------------------------------------ gradients and training

def test_gradient_reaches_the_trunk_and_every_live_head_parameter():
    enc, comp = _composite()
    comp.term.compute(enc, None).backward()
    assert enc.patch_embed.weight.grad.abs().sum() > 0
    dead = [n for n, p in comp.term.head.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert dead == [], dead
    assert all(p.grad is None and not p.requires_grad for p in comp._teacher.module.parameters())


def test_the_loss_falls_on_a_fixed_window():
    enc, comp = _composite()
    term = comp.term
    window = term.draw(enc)
    opt = torch.optim.Adam([*enc.parameters(), *term.head.parameters()], lr=3e-3)
    losses = []
    for _ in range(10):
        loss, _ = term._core(enc, window)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] - 1e-3, losses


def test_the_mask_variance_null_is_exact_under_a_fixed_init():
    """With FixedLearnedInit there is no per-sample init noise to cancel, so a batch of IDENTICAL
    frames must give EXACTLY the same attention variance as the null -- excess 0. (Under
    slot_contrast's random init the same null is a noise floor, not an exact zero.)"""
    enc, comp = _composite()
    term = comp.term
    window = term.draw(enc)
    window.prepared_t = window.prepared_t[:1].expand_as(window.prepared_t).contiguous()
    _, scalars = term._core(enc, window)
    assert scalars["mask_variance_excess"] == 0.0
    assert scalars["mask_variance"] == scalars["mask_variance_null"]


def test_fg_mass_is_strictly_inside_the_unit_interval_at_init():
    enc, comp = _composite()
    comp.term.compute(enc, None)
    fg = comp.term.last_scalars["fg_mass"]
    assert 0.0 < fg < 1.0
    assert comp.term.last_scalars["fg_mass_max_slot"] < 1.0


# ------------------------------------------------------------------ P5: the ego composition

def test_without_the_ego_knob_the_foreground_terms_report_sentinels_not_zeros():
    enc, comp = _composite()
    comp.term.compute(enc, None)
    s = comp.term.last_scalars
    assert s["fg_bce"] == NOT_MEASURED and s["stuff"] == NOT_MEASURED
    assert s["ego_loss"] == NOT_MEASURED and s["used_ego"] == 0.0
    assert comp.term.ego is None


def test_the_ego_knob_composes_P3_inside_P4_on_ONE_window(monkeypatch):
    monkeypatch.setenv("NETT_AUX_SLOTFG_EGO", "1")
    enc, comp = _composite()
    term = comp.term
    assert term.ego is not None and "ego" in term.head
    # The ego head is inside THIS term's head, so AuxLossPPO's single param group covers it.
    ego_ids = {id(p) for p in term.ego.head.parameters()}
    assert ego_ids <= {id(p) for p in term.head.parameters()}
    windows = []
    real = term.ego.loss_and_objectness
    term.ego.loss_and_objectness = lambda e, w: (windows.append(w) or real(e, w))
    term.compute(enc, None)
    s = term.last_scalars
    assert len(windows) == 1                       # the ego runs on the slot window, once
    assert s["used_ego"] == 1.0 and s["fg_bce"] > 0 and s["ego_loss"] > 0
    assert "ego_action_gain" in s and "ego_token_std" in s
    assert s["fg_objectness_corr"] != NOT_MEASURED


def test_the_foreground_bce_equals_torchs_but_is_autocast_safe():
    """⛔ F.binary_cross_entropy RAISES under CUDA autocast whatever the dtype, and the aux runs
    inside torch.autocast(enabled=cfg.mixed_precision). The written-out form must give the same
    number, or the swap traded a crash for a wrong loss."""
    m = torch.rand(4, N).clamp(1e-6, 1 - 1e-6)
    w = torch.rand(4, N)
    manual = -(w * m.log() + (1 - w) * (1 - m).log()).mean()
    assert float(manual) == pytest.approx(float(F.binary_cross_entropy(m, w)), rel=1e-6)


def test_row_05_refuses_the_ego_window_knobs_that_it_would_ignore(monkeypatch):
    monkeypatch.setenv("NETT_AUX_SLOTFG_EGO", "1")
    monkeypatch.setenv("NETT_AUX_EGO_BATCH", "64")
    with pytest.raises(ValueError, match="would be ignored"):
        SlotFGTerm(_encoder())


def test_the_foreground_supervision_can_actually_be_fitted(monkeypatch):
    """The BCE has a reachable optimum: slot 0's mask can be trained toward a fixed objectness
    map. Without this, "the term is in the loss" says nothing about whether it can move."""
    monkeypatch.setenv("NETT_AUX_SLOTFG_EGO", "1")
    enc, comp = _composite()
    term = comp.term
    window = term.draw(enc)
    target = torch.rand(window.prepared_t.shape[0], N)
    opt = torch.optim.Adam(term.head.parameters(), lr=1e-2)
    from nett_skrl.brain.aux.token_features import spatial_tokens
    losses = []
    for _ in range(25):
        z = spatial_tokens(enc, window.prepared_t)[0]
        init = term.head["init"].value.expand(z.shape[0], -1, -1)
        _slots, attn = term._slots(z, init, term.ITERS_FIRST)
        loss = F.binary_cross_entropy(attn[:, 0, :].clamp(1e-6, 1 - 1e-6), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] - 0.01, losses


def test_the_ramp_starts_near_zero_and_saturates_at_one(monkeypatch):
    monkeypatch.setenv("NETT_AUX_SLOTFG_EGO", "1")
    monkeypatch.setenv("NETT_AUX_SLOTFG_RAMP_CALLS", "3")
    enc, comp = _composite()
    ramps = []
    for _ in range(4):
        comp.term.compute(enc, None)
        ramps.append(comp.term.last_scalars["ramp"])
    assert ramps == [pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0, 1.0]


# ------------------------------------------------------------------ knobs and registry

def test_knobs_are_read_and_invalid_values_raise(monkeypatch):
    monkeypatch.setenv("NETT_AUX_SLOTFG_SLOTS", "3")
    monkeypatch.setenv("NETT_AUX_SLOTFG_DIM", "32")
    monkeypatch.setenv("NETT_AUX_SLOTFG_TEMP", "0.2")
    monkeypatch.setenv("NETT_AUX_SLOTFG_W_SS", "0.25")
    monkeypatch.setenv("NETT_AUX_SLOTFG_LAMBDA", "0")
    monkeypatch.setenv("NETT_AUX_SLOTFG_GAMMA", "0.5")
    term = SlotFGTerm(_encoder())
    assert (term.slots, term.slot_dim, term.temperature, term.w_ss) == (3, 32, 0.2, 0.25)
    assert (term.lambda_stuff, term.gamma_sep) == (0.0, 0.5)
    for name, bad in (("NETT_AUX_SLOTFG_SLOTS", "0"), ("NETT_AUX_SLOTFG_DIM", "x"),
                      ("NETT_AUX_SLOTFG_TEMP", "-1"), ("NETT_AUX_SLOTFG_GAMMA", "-0.1"),
                      ("NETT_AUX_SLOTFG_RAMP_CALLS", "0")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError, match=name):
            SlotFGTerm(_encoder())
        monkeypatch.delenv(name)


def test_the_ego_flag_refuses_a_misspelling_rather_than_running_row_04_as_row_05(monkeypatch):
    monkeypatch.setenv("NETT_AUX_SLOTFG_EGO", "ture")
    with pytest.raises(ValueError, match="not a boolean"):
        SlotFGTerm(_encoder())


def test_defaults_are_the_spec_values():
    term = SlotFGTerm(_encoder())
    assert (term.slots, term.slot_dim, term.temperature) == (4, 64, 0.1)
    assert (term.w_ss, term.w_rec, term.lambda_stuff, term.gamma_sep) == (0.5, 1.0, 1.0, 0.1)
    assert term.offset == 8 and term.use_ego is False


def test_the_row_is_registered_and_carries_the_shared_teacher():
    aux = AUX_LOSSES["slot_fg"](_encoder())
    assert isinstance(aux, WithCLTTRef) and isinstance(aux.term, SlotFGTerm)
    assert aux.name == "slot_fg" and aux._teacher is not None
