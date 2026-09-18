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
         "NETT_AUX_SLOTFG_DECODER", "NETT_AUX_SLOTFG_TARGET", "NETT_AUX_SLOTFG_DECODE_SCALE",
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

CELLS = [("tokmlp", "ema_tokens"), ("tokmlp", "pixels"),
         ("convsbd", "ema_tokens"), ("convsbd", "pixels")]


def _cell(monkeypatch, decoder, target, scale=None):
    monkeypatch.setenv("NETT_AUX_SLOTFG_DECODER", decoder)
    monkeypatch.setenv("NETT_AUX_SLOTFG_TARGET", target)
    # ⚠ DELETED, not left: a scale set by an earlier call in the same test would otherwise carry
    # into a token-target cell and raise -- the leak this helper exists to make impossible.
    if scale is None:
        monkeypatch.delenv("NETT_AUX_SLOTFG_DECODE_SCALE", raising=False)
    else:
        monkeypatch.setenv("NETT_AUX_SLOTFG_DECODE_SCALE", str(scale))
    return _composite()


@pytest.mark.parametrize("decoder,target", CELLS)
def test_gradient_reaches_the_trunk_and_every_live_head_parameter(monkeypatch, decoder, target):
    """⛔ ALL FOUR CELLS OF THE SCREEN, TRAINABLE. The screen's result is only checkable if the
    configuration it screened can still be BUILT and RUN from the shipped code -- and a cell
    whose decoder gets no gradient is not the cell that was measured, however it is spelled.

    `dead == []` is the other half: this file already refuses dead parameters in an optimizer
    param group (see `attn.mu`), and the decoder modules are now built per cell precisely so
    that the unused one is ABSENT rather than present-and-untrained.
    """
    enc, comp = _cell(monkeypatch, decoder, target)
    comp.term.compute(enc, None).backward()
    assert enc.patch_embed.weight.grad.abs().sum() > 0
    dead = [n for n, p in comp.term.head.named_parameters()
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    assert dead == [], dead
    if comp._teacher is not None:
        assert all(p.grad is None and not p.requires_grad
                   for p in comp._teacher.module.parameters())


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
    # ⛔ THE CONTRACT, NOT THE FIXTURE'S CURRENT ANSWER. Whether this synthetic window's
    # objectness clears its paired null is a property of the fixture and the seed; what must
    # hold on every run is that the correlation is reported only when the target it correlates
    # against was established. A near-zero correlation against a noise target and a near-zero
    # correlation against a real one are the same number and opposite facts.
    assert s["fg_target_engaged"] in (0.0, 1.0, NOT_MEASURED)
    if s["fg_target_engaged"] == 1.0:
        assert s["fg_objectness_corr"] != NOT_MEASURED
    else:
        assert s["fg_objectness_corr"] == NOT_MEASURED


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


def test_the_row_is_registered_and_carries_the_shared_teacher_ONLY_WHERE_ONE_IS_USED(
        monkeypatch):
    """⛔ THE TEACHER IS OWNED BY WHOEVER NEEDS IT, AND UNDER THE SHIPPED DEFAULT NOBODY DOES.

    `cltt_ref` reads no teacher at all (zero references in its module); the composite builds one
    only because a TERM asks. With a pixel target and the ego off, this row's reconstruction is
    against the observation, so the row asks for none -- and a teacher nobody reads is a deepcopy
    of the trunk plus an EMA step and a teacher forward per minibatch, paid for nothing.

    ⚠ AND `ema_updates` GOES SENTINEL, WHICH IS A VISIBLE CHANGE FOR A READER: on this row it now
    means "no term uses an EMA target", not "the EMA failed to step".
    """
    aux = AUX_LOSSES["slot_fg"](_encoder())
    assert isinstance(aux, WithCLTTRef) and isinstance(aux.term, SlotFGTerm)
    assert aux.name == "slot_fg"
    assert aux._teacher is None and aux.term._teacher is None, (
        "the winning cell reconstructs PIXELS; nothing in the row reads an EMA teacher")
    monkeypatch.setenv("NETT_AUX_SLOTFG_TARGET", "ema_tokens")
    tok = AUX_LOSSES["slot_fg"](_encoder())
    assert tok._teacher is not None and tok.term._teacher is tok._teacher
    monkeypatch.setenv("NETT_AUX_SLOTFG_TARGET", "pixels")
    monkeypatch.setenv("NETT_AUX_SLOTFG_EGO", "1")
    ego = AUX_LOSSES["slot_fg"](_encoder())
    assert ego._teacher is not None, "row 05's ego residual has its own EMA token target"


def test_the_ema_teacher_is_not_stepped_by_a_term_that_does_not_use_it(monkeypatch):
    """⛔ NOT MERELY UNUSED -- NOT STEPPED, AND NOT BUILT. A teacher that is stepped but unread
    still costs a trunk-sized deepcopy and an EMA pass per minibatch, and still publishes an
    `ema_updates` series that a reader would take as evidence that some target used it."""
    enc, comp = _cell(monkeypatch, "convsbd", "pixels")
    comp.compute(enc, None)
    assert comp._teacher is None
    assert comp.last_scalars["ema_updates"] == NOT_MEASURED
    enc2, comp2 = _cell(monkeypatch, "convsbd", "ema_tokens")
    comp2.compute(enc2, None)
    assert comp2._teacher is not None and comp2._teacher.updates == 1
    assert comp2.last_scalars["ema_updates"] == 1.0


def test_the_foreground_correlation_is_withheld_exactly_when_the_target_is_not_established(
        monkeypatch):
    """Both branches of the gate, forced -- the fixture cannot be relied on to visit both.

    ⛔ WHY THIS IS NOT PARANOIA. Row 05's foreground target IS the ego term's objectness. The GPU
    verification round read `fg_centre_corr` and `fg_objectness_corr` off a run whose objectness
    had not separated from its own null, and a small correlation there says nothing at all about
    the slots.
    """
    monkeypatch.setenv("NETT_AUX_SLOTFG_EGO", "1")
    enc, comp = _composite()
    term = comp.term
    real = term.ego.loss_and_objectness

    for flag, expect_number in ((1.0, True), (0.0, False), (NOT_MEASURED, False)):
        def stub(e, w, _f=flag):
            loss, objectness, scalars = real(e, w)
            return loss, objectness, {**scalars, "objectness_engaged": _f}
        term.ego.loss_and_objectness = stub
        term.compute(enc, None)
        s = term.last_scalars
        assert s["fg_target_engaged"] == flag
        assert (s["fg_objectness_corr"] != NOT_MEASURED) is expect_number


# ------------------------------------------------------------------ the screened configuration

def test_the_defaults_are_the_WINNING_PAIR_and_every_spelling_is_validated(monkeypatch):
    """⛔ THE DEFAULT IS A RESULT, NOT A PREFERENCE (FINDINGS §4by.1: 0.531 +/- 0.033 against the
    shipped 0.320 +/- 0.028), and the knobs exist so the other three cells stay reproducible.

    Every value raises rather than falling back, because these select WHICH MODEL IS TRAINED: a
    misspelling that resolved to the default would file the arm under the wrong cell of the very
    comparison it was launched for, and nothing downstream could tell.
    """
    term = SlotFGTerm(_encoder())
    assert (term.decoder_kind, term.target_kind, term.decode_scale) == ("convsbd", "pixels", 2)
    for knob, bad in (("NETT_AUX_SLOTFG_DECODER", "conv_sbd"),
                      ("NETT_AUX_SLOTFG_DECODER", "sbd"),
                      ("NETT_AUX_SLOTFG_TARGET", "pixel"),
                      ("NETT_AUX_SLOTFG_TARGET", "tokens"),
                      ("NETT_AUX_SLOTFG_DECODE_SCALE", "3"),
                      ("NETT_AUX_SLOTFG_DECODE_SCALE", "0"),
                      ("NETT_AUX_SLOTFG_DECODE_SCALE", "half")):
        monkeypatch.setenv(knob, bad)
        with pytest.raises(ValueError, match="is not one of"):
            SlotFGTerm(_encoder())
        monkeypatch.delenv(knob)
    # ⛔ AND THE COMBINATION, not just the values: a token target decodes onto the token grid,
    # where a resolution divisor is a knob nothing reads -- which is a control that is not there.
    monkeypatch.setenv("NETT_AUX_SLOTFG_TARGET", "ema_tokens")
    monkeypatch.setenv("NETT_AUX_SLOTFG_DECODE_SCALE", "2")
    with pytest.raises(ValueError, match="needs NETT_AUX_SLOTFG_TARGET=pixels"):
        SlotFGTerm(_encoder())
    # ⛔ BUT A DEFAULT NOBODY CHOSE IS NOT AN ASK. The scale default is 2 now, and if a token
    # target inherited it and raised, two of the screen's four cells would be unbuildable by
    # anyone who did not know to spell the scale back -- the reproducibility these knobs exist
    # for, lost to the default of an unrelated one.
    monkeypatch.delenv("NETT_AUX_SLOTFG_DECODE_SCALE")
    tok = SlotFGTerm(_encoder())
    assert (tok.target_kind, tok.decode_scale) == ("ema_tokens", 1)


@pytest.mark.parametrize("decoder,target", CELLS)
def test_only_the_modules_this_cell_uses_are_built(monkeypatch, decoder, target):
    """The unused decoder is ABSENT, not present-and-frozen: a parameter in the optimizer's group
    receiving no gradient is indistinguishable from a pathway meant to train and silently not."""
    monkeypatch.setenv("NETT_AUX_SLOTFG_DECODER", decoder)
    monkeypatch.setenv("NETT_AUX_SLOTFG_TARGET", target)
    keys = set(SlotFGTerm(_encoder()).head.keys())
    assert ({"sbd", "sbd_pos"} <= keys) == (decoder == "convsbd")
    assert ({"decoder", "pos"} <= keys) == (decoder == "tokmlp")
    assert not ({"sbd"} & keys and {"decoder"} & keys), "one decoder per cell, never both"


def test_the_pixel_target_is_the_TENSOR_THE_ENCODER_SAW(monkeypatch):
    """⛔ HOW THE ALIGNMENT IS VERIFIED, IN TWO INDEPENDENT WAYS.

    (1) VALUE: the reconstruction loss equals, to the bit, the MSE of this cell's decode against
        `window.prepared_t[:, -cpf:]` -- the exact tensor object `spatial_tokens` was handed in
        `_core`, sliced. Not a tensor with the same values: the same tensor.
    (2) PROVENANCE: the tensor handed to `mse_loss` SHARES STORAGE with `window.prepared_t`, so
        it is a view of that very buffer and not a copy that merely matches today. A target
        built from a second preparation could differ in normalisation, resize or which frame is
        current, and would still look plausible.
        ⚠ COUNTING `_prepare_image` CALLS DOES NOT WORK HERE, and the first version of this test
        did: `encode_tokens_prepared` calls it under `_skip_prepare`, where it is a pass-through,
        so the call count is 3 on the correct code. Storage identity is the question actually
        being asked.

    ⚠ `-cpf:` and not `3:6`: preparation orders channels [oldest, ..., current], so the CURRENT
    frame is at the END. They coincide at the campaign eye's 6 RGB channels -- which is what the
    screen measured -- and differ under dvs_polarity, where one frame is two channels.
    """
    enc, comp = _cell(monkeypatch, "convsbd", "pixels")
    term = comp.term
    window = term.draw(enc)
    from nett_skrl.brain.aux import slot_fg_aux

    seen = {}
    real_mse = slot_fg_aux.F.mse_loss

    def spy(pred, target, **kw):
        seen["target"] = target
        return real_mse(pred, target, **kw)

    # ⚠ RESTORED BY NAME, NOT BY `monkeypatch.undo()`: `slot_fg_aux.F` IS torch.nn.functional,
    # so this patch is global while it is up, and undo() would also roll back the env this
    # fixture and helper set -- a teardown that reaches further than the thing it is undoing.
    monkeypatch.setattr(slot_fg_aux.F, "mse_loss", spy)
    try:
        term._core(enc, window)
    finally:
        monkeypatch.setattr(slot_fg_aux.F, "mse_loss", real_mse)
    got = seen["target"]
    assert got.untyped_storage().data_ptr() == window.prepared_t.untyped_storage().data_ptr(), (
        "the target must be a VIEW of the tensor the encoder consumed, not a copy of it")
    assert torch.equal(got, window.prepared_t[:, -term.cpf:])
    from nett_skrl.brain.aux.token_features import spatial_tokens
    with torch.no_grad():
        z_t = spatial_tokens(enc, window.prepared_t)[0]
        init = term.head["init"].value.expand(z_t.shape[0], -1, -1)
        slots_t, _ = term._slots(z_t, init, term.ITERS_FIRST)
        target = window.prepared_t[:, -term.cpf:]
        # ⚠ THE UPSAMPLE IS ON THE PREDICTION SIDE, and re-deriving it here pins that too: at the
        # default scale of 2 the decode is 40x64 and the target stays 80x128, so a version that
        # resized the TARGET instead would fail this equality rather than quietly changing the
        # task. (An earlier draft compared the raw decode to the target and only passed because
        # the default was then scale 1 -- the shapes agreed by accident.)
        pred = F.interpolate(term._decode(slots_t), size=target.shape[-2:], mode="bilinear",
                             align_corners=False)
        expected = F.mse_loss(pred, target)
    assert term.decode_hw == (H // 2, W // 2), "this runs on the DEFAULT cell, not a special one"
    assert torch.equal(term._reconstruction(window, slots_t), expected)
    assert term.cpf == 3 and window.prepared_t.shape[1] == 6, (
        "at the campaign eye the current frame is channels 3:6 == -3:, which is what was screened")


@pytest.mark.parametrize("decoder", ["convsbd", "tokmlp"])
@pytest.mark.parametrize("scale", [1, 2, 4])
def test_decode_scale_changes_the_decoders_resolution_and_NOT_the_loss_scale(
        monkeypatch, decoder, scale):
    """⛔ THE FALLBACK MUST NOT QUIETLY BECOME A DIFFERENT OBJECTIVE. Decoding at half resolution
    is a memory choice; DOWNSAMPLING THE TARGET to meet it would be a change of task, and the
    loss would shift by a factor nobody asked for -- so a screen comparing scales would be
    comparing two things at once and would report the wrong one.

    The prediction is upsampled to the target instead. The proof is a CONSTANT prediction: its
    upsample is the same constant at every scale, so the MSE against the same target must be
    IDENTICAL, bitwise, across scales. Any target-side rescaling shows up here immediately.
    """
    enc, comp = _cell(monkeypatch, decoder, "pixels", scale)
    term = comp.term
    window = term.draw(enc)
    assert term.decode_hw == (H // scale, W // scale)
    slots = torch.randn(window.prepared_t.shape[0], term.slots, term.slot_dim)
    out = term._decode(slots)
    assert out.shape == (window.prepared_t.shape[0], term.cpf, H // scale, W // scale)
    assert torch.isfinite(term._reconstruction(window, slots))
    # the scale-invariance of the loss units, on a prediction the upsample cannot change
    term._decode = lambda s: torch.full((s.shape[0], term.cpf, *term.decode_hw), 0.25)
    pinned = float(term._reconstruction(window, slots))
    expected = float(F.mse_loss(torch.full_like(window.prepared_t[:, -term.cpf:], 0.25),
                                window.prepared_t[:, -term.cpf:]))
    assert pinned == expected, (scale, pinned, expected)


def test_the_cell_goes_out_in_the_scalars(monkeypatch):
    """A `rec` series is unreadable without its cell: token targets and pixel targets are
    different tensors in different units, so the number alone cannot say which loss it is."""
    enc, comp = _cell(monkeypatch, "convsbd", "pixels", 2)
    comp.term.compute(enc, None)
    s = comp.term.last_scalars
    assert (s["decoder"], s["target"], s["decode_scale"]) == (1.0, 1.0, 2.0)
    assert s["head_params"] == float(sum(p.numel() for p in comp.term.head.parameters()))
    enc2, comp2 = _cell(monkeypatch, "tokmlp", "ema_tokens", None)
    comp2.term.compute(enc2, None)
    assert (comp2.term.last_scalars["decoder"], comp2.term.last_scalars["target"]) == (0.0, 0.0)


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_decode_scale_moves_the_resolution_and_HOLDS_THE_DEPTH(monkeypatch, scale):
    """⛔ THE SCALE SCREEN MUST BE ABLE TO CONCLUDE SOMETHING. Decoding at half resolution by
    DELETING a ConvTranspose removes two layers along with the pixels, so "binds at 2, not at 4"
    would not say whether it was the resolution or the depth -- and the answer the knob exists to
    produce would need another port to interpret. Same discipline as the screen's own "alpha
    resolution is held with the target": move one thing.
    """
    monkeypatch.setenv("NETT_AUX_SLOTFG_DECODER", "convsbd")
    monkeypatch.setenv("NETT_AUX_SLOTFG_TARGET", "pixels")
    monkeypatch.setenv("NETT_AUX_SLOTFG_DECODE_SCALE", str(scale))
    term = SlotFGTerm(_encoder())
    convs = [m for m in term.head["sbd"] if isinstance(m, (torch.nn.Conv2d,
                                                           torch.nn.ConvTranspose2d))]
    assert len(convs) == 4, [type(m).__name__ for m in convs]
    strided = [m for m in convs if getattr(m, "stride", (1,))[0] == 2]
    assert len(strided) == 3 - int(math.log2(scale)), "only the STRIDES may move with the scale"


def test_a_width_knob_the_running_decoder_does_not_read_is_refused(monkeypatch):
    """⛔ DEC_HIDDEN SIZES THE PER-TOKEN MLP AND NOTHING ELSE. Under the default cell the
    spatial-broadcast decoder's width is a module constant, so a config that set DEC_HIDDEN would
    train the unmodified model while its launch line said otherwise -- the silent no-op this
    class already refuses for the ego window knobs and for DECODE_SCALE on a token target."""
    monkeypatch.setenv("NETT_AUX_SLOTFG_DEC_HIDDEN", "512")
    with pytest.raises(ValueError, match="sizes the per-token MLP decoder"):
        SlotFGTerm(_encoder())
    monkeypatch.setenv("NETT_AUX_SLOTFG_DECODER", "tokmlp")
    assert SlotFGTerm(_encoder()).head["decoder"][0].out_features == 512


def test_the_ego_branch_publishes_THE_SAME_KEYS_whether_or_not_it_runs(monkeypatch):
    """⛔ THE WAVE'S LAST PLACE WHERE "DID NOT RUN" AND "RAN AND FOUND NOTHING" WERE ONE ABSENCE.

    `fg_objectness_corr` was written only inside `if self.ego is not None`, so row 04 emitted 6 of
    this family's 7 keys and the 7th was simply missing. A reader -- and the provenance column
    built to tell those two states apart -- cannot distinguish a tag that was never emitted from
    one that fired and measured nothing; the sentinel-aware aggregation in ppo_aux can only
    publish a fire rate for a key it was told about.

    ⚠ Pinned against `EGO_BRANCH_KEYS` rather than a literal list, on BOTH paths, so a key added
    to the branch tomorrow cannot be sentinel-less on one of them -- the same source-of-truth
    shape as `DIAG_KEYS` in cltt_ref/cltt_patch, whose third instance this is.
    """
    from nett_skrl.brain.aux.slot_fg_aux import EGO_BRANCH_KEYS

    enc, comp = _composite()
    comp.term.compute(enc, None)
    off = comp.term.last_scalars
    assert set(EGO_BRANCH_KEYS) <= set(off), sorted(set(EGO_BRANCH_KEYS) - set(off))
    assert all(off[k] == NOT_MEASURED for k in EGO_BRANCH_KEYS), (
        {k: off[k] for k in EGO_BRANCH_KEYS})

    monkeypatch.setenv("NETT_AUX_SLOTFG_EGO", "1")
    enc2, comp2 = _composite()
    comp2.term.compute(enc2, None)
    on = comp2.term.last_scalars
    assert set(EGO_BRANCH_KEYS) <= set(on), sorted(set(EGO_BRANCH_KEYS) - set(on))
    # The two paths differ ONLY by the ego term's own namespaced scalars, which belong to a term
    # that does not exist on row 04 -- not by any key this term itself owns.
    assert {k for k in on if not k.startswith("ego_")} == {k for k in off if not k.startswith("ego_")}
    assert on["ego_loss"] != NOT_MEASURED, "the branch really ran on the ego path"
