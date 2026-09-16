"""The mask-variance null must be PAIRED to its signal: same slot-init draw.

⛔ WHY THIS EXISTS. `mask_variance_excess = mask_variance - mask_variance_null` was computed with
the null drawing its OWN slot initialisation, so the excess was a difference of two INDEPENDENT
draws of the same noise term. The module's own fresh-init calibration (B=32, K=4) measured null
3.86e-03 against signal 3.78e-03 -- an excess of -8.0e-05 at ZERO true input-dependence. A reading
of `excess <= 0` was therefore the BASELINE, not a finding, and could not separate "no input
dependence" from "input dependence smaller than the noise". That defect was load-bearing: it was
used to argue a hypothesis about row 14 that the statistic had no power to support.

⚠ WHAT PAIRING DOES AND DOES NOT BUY. It makes the null EXACT -- see the test below. It does NOT
make the statistic powerful: measured on this untrained module with synthetic inputs, mean excess
under genuinely input-dependent batches is ~1e-5 against a per-batch spread of ~2e-4, so a SINGLE
read remains uninformative regardless of pairing. Read it as a block median, and do not treat a
single positive excess as evidence.
"""
import pytest
import torch

from nett_skrl.brain.aux import slot_contrast_aux as sc

B, N, D, K = 8, 72, 32, 4      # 72 positions is nature_cnn's 6x12, K=4 as the wave ran


def _mod():
    torch.manual_seed(0)
    return sc.SlotAttention(in_dim=D, slot_dim=D, slots=K)


def _excess(m, inputs, paired):
    _, a_sig = m(inputs)
    init = m.last_init
    flat = inputs[:1].expand_as(inputs).contiguous()
    with torch.no_grad():
        _, a_null = m(flat, init if paired else None)
    return float(a_sig.var(dim=0).mean()) - float(a_null.var(dim=0).mean())


def test_paired_null_is_zero_on_IDENTICAL_inputs_which_is_only_an_identity():
    """⚠ WEAK BY CONSTRUCTION -- kept to document what it does NOT show.

    With identical batch elements the null's inputs equal the signal's BITWISE, so with a shared
    init the null pass IS the signal pass and 0.0 is forced. This tests determinism, which was
    never in doubt. The real null is the next test: distinct inputs, input-blind attention."""
    m = _mod()
    x = torch.randn(1, N, D).expand(B, N, D).contiguous()
    for _ in range(8):
        assert _excess(m, x, paired=True) == 0.0


def test_unpaired_null_is_NOT_zero_at_zero_input_dependence():
    """⛔ THE REFUTATION. The same inputs under the old, unpaired null return nonzero noise --
    which is what made `excess <= 0` unreadable. If this ever passes, the fix has been undone."""
    m = _mod()
    x = torch.randn(1, N, D).expand(B, N, D).contiguous()
    vals = [_excess(m, x, paired=False) for _ in range(8)]
    assert any(v != 0.0 for v in vals)
    assert max(abs(v) for v in vals) > 1e-9


def test_slots_init_is_actually_reused_not_redrawn():
    """Pins the mechanism, not just the symptom: passing an init must reproduce the attention."""
    m = _mod()
    x = torch.randn(B, N, D)
    _, a1 = m(x)
    init = m.last_init
    with torch.no_grad():
        _, a2 = m(x, init)
    assert torch.allclose(a1, a2, atol=0, rtol=0)


def test_last_init_records_the_most_recent_draw_only():
    """⚠ WHY compute() CAPTURES init_t IMMEDIATELY. `last_init` is overwritten by the next
    forward, so a caller that reads it late pairs the null to the WRONG pass -- which looks
    exactly like a working fix while cancelling nothing."""
    m = _mod()
    m(torch.randn(B, N, D))
    first = m.last_init.clone()
    m(torch.randn(B, N, D))
    assert not torch.allclose(first, m.last_init)


def test_passing_an_init_does_not_overwrite_last_init():
    """A null pass must not clobber the draw the signal pass recorded."""
    m = _mod()
    m(torch.randn(B, N, D))
    saved = m.last_init.clone()
    with torch.no_grad():
        m(torch.randn(B, N, D), saved)
    assert torch.allclose(saved, m.last_init)


# ---------------------------------------------------------------------------------------------
# ⛔⛔ THE TESTS ABOVE EXERCISE `SlotAttention.forward` DIRECTLY AND DO NOT CATCH THE REAL DEFECT.
# Verified by mutation: deleting the `slots_init` pass-through in `SlotContrastAux._slots_for`
# -- the ONLY path `compute()` uses -- leaves all five GREEN, because they bypass `_slots_for`
# entirely. A suite that tests the mechanism on a path the product does not take is a passing
# check on the wrong subject. These exercise the call path itself.
# ---------------------------------------------------------------------------------------------

class _Encoder(torch.nn.Module):
    """Minimal stand-in exposing the `cnn` Sequential `spatial_features()` requires."""

    def __init__(self, ch=8):
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, ch, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())


def _aux_with_encoder():
    torch.manual_seed(0)
    enc = _Encoder()
    aux = sc.SlotContrastAuxLoss.__new__(sc.SlotContrastAuxLoss)     # skip __init__'s trunk probing
    torch.nn.Module.__init__(aux)
    aux.head = torch.nn.Module()
    aux.head.attn = sc.SlotAttention(in_dim=8, slot_dim=16, slots=K)
    return aux, enc


def test_slots_for_REUSES_a_passed_init_on_the_real_call_path():
    """⭐ THE TEST THAT ACTUALLY BINDS. Goes through `_slots_for`, the path `compute()` uses.
    Deleting the pass-through there makes this fail; the forward-level tests do not notice."""
    aux, enc = _aux_with_encoder()
    x = torch.randn(B, 3, 6, 12)
    _, attn1, _, _, _ = aux._slots_for(enc, x)
    init = aux.head.attn.last_init
    with torch.no_grad():
        _, attn2, _, _, _ = aux._slots_for(enc, x, init)
    assert torch.allclose(attn1, attn2, atol=0, rtol=0), (
        "_slots_for ignored the init it was handed -- the null is not paired to its signal")


def test_paired_null_is_exactly_zero_THROUGH_slots_for():
    """The end-to-end property on the real path: identical inputs + the signal's own init."""
    aux, enc = _aux_with_encoder()
    x = torch.randn(1, 3, 6, 12).expand(B, 3, 6, 12).contiguous()
    _, attn_sig, _, _, _ = aux._slots_for(enc, x)
    init = aux.head.attn.last_init
    flat = x[:1].expand_as(x).contiguous()
    with torch.no_grad():
        _, attn_null, _, _, _ = aux._slots_for(enc, flat, init)
    excess = float(attn_sig.var(dim=0).mean()) - float(attn_null.var(dim=0).mean())
    assert excess == 0.0


def test_null_pairs_to_the_SIGNAL_pass_not_the_intervening_one():
    """⛔ THE OFF-BY-ONE-PASS BUG. `compute()` runs TWO passes before the null -- `prep_t`
    (whose attention becomes mask_variance) and then `prep_n`. `last_init` holds the MOST
    RECENT draw, so reading it at the null site pairs to `prep_n` and cancels nothing, while
    every single-pass test still passes. Reproduces that structure exactly.

    Verified by mutation: replacing `init_t` with `self.head.attn.last_init` at the null site
    leaves the rest of this file green and fails only here.
    """
    aux, enc = _aux_with_encoder()
    x_t = torch.randn(1, 3, 6, 12).expand(B, 3, 6, 12).contiguous()   # the SIGNAL's inputs
    x_n = torch.randn(B, 3, 6, 12)                                    # the intervening pass

    _, attn_t, _, _, _ = aux._slots_for(enc, x_t)
    init_t = aux.head.attn.last_init              # captured HERE, as compute() does
    aux._slots_for(enc, x_n)                      # overwrites last_init
    assert not torch.allclose(init_t, aux.head.attn.last_init), "fixture failed to overwrite"

    flat = x_t[:1].expand_as(x_t).contiguous()
    with torch.no_grad():
        _, attn_null, _, _, _ = aux._slots_for(enc, flat, init_t)
    assert float(attn_t.var(dim=0).mean()) - float(attn_null.var(dim=0).mean()) == 0.0

    # ...and pairing to the WRONG (intervening) draw does NOT cancel -- the bug's signature.
    with torch.no_grad():
        _, attn_wrong, _, _, _ = aux._slots_for(enc, flat, aux.head.attn.last_init)
    assert float(attn_t.var(dim=0).mean()) - float(attn_wrong.var(dim=0).mean()) != 0.0


def test_each_pass_returns_ITS_OWN_init_so_mispairing_is_unrepresentable():
    """⭐ THE STRUCTURAL FIX, replacing an AST test that guarded a shape.

    `_slots_for` now RETURNS the init it used. `last_init` is a mutable side-channel the next
    forward overwrites, so a caller reading it late pairs the null to whichever pass ran most
    recently -- cancelling nothing while looking like a working fix. That was previously guarded
    by asserting on compute()'s source; it is now not expressible, because the `prep_n` call
    cannot reach the `prep_t` call's init. This pins the property the design relies on.
    """
    aux, enc = _aux_with_encoder()
    _, _, _, _, init_a = aux._slots_for(enc, torch.randn(B, 3, 6, 12))
    _, _, _, _, init_b = aux._slots_for(enc, torch.randn(B, 3, 6, 12))
    assert not torch.allclose(init_a, init_b), "two fresh passes returned the same init"
    # each returned init reproduces ITS OWN pass, which is what makes pairing well-defined
    x = torch.randn(B, 3, 6, 12)
    _, attn1, _, _, used = aux._slots_for(enc, x)
    with torch.no_grad():
        _, attn2, _, _, _ = aux._slots_for(enc, x, used)
    assert torch.allclose(attn1, attn2, atol=0, rtol=0)


def test_a_passed_init_is_returned_unchanged():
    """The returned init must BE the one used, or pairing silently drifts."""
    aux, enc = _aux_with_encoder()
    _, _, _, _, init = aux._slots_for(enc, torch.randn(B, 3, 6, 12))
    with torch.no_grad():
        _, _, _, _, echoed = aux._slots_for(enc, torch.randn(B, 3, 6, 12), init)
    assert echoed is init or torch.allclose(echoed, init, atol=0, rtol=0)


class _ConstProj(torch.nn.Module):
    """Returns the same vector for every position and batch element: k/v carry NO input."""

    def __init__(self, dim):
        super().__init__()
        self.register_buffer("c", torch.randn(dim))

    def forward(self, x):
        return self.c.expand(x.shape[0], x.shape[1], -1)


def _input_blind(iters):
    """Attention provably ignores the input while slot-init variance stays LIVE.

    ⚠ An earlier attempt zeroed `to_k`/`to_v` weights and set their biases -- but those Linears
    are bias=False, so the guard silently did nothing and k became exactly 0, collapsing
    attention to a uniform 1/K with ZERO batch variance. A fixture that does not do what its
    name says, which is the defect class this whole file exists to catch. Replacing the
    projections outright is checked by the assertions in the test below, not assumed.
    """
    torch.manual_seed(1)
    m = sc.SlotAttention(in_dim=D, slot_dim=D, slots=K, iters=iters)
    m.to_k, m.to_v = _ConstProj(D), _ConstProj(D)
    return m


@pytest.mark.parametrize("iters", [1, 3])
def test_paired_null_is_zero_on_DISTINCT_inputs_with_input_blind_attention(iters):
    """⭐⭐ THE REAL NULL, and the property is exact rather than a fixture artifact.

    If attention does not depend on the input, attn(x_b, I) and attn(x_0, I) are the SAME
    tensor, so the variances are equal and the paired excess is exactly 0 -- for ANY inputs.
    Inputs here are distinct and redrawn every trial; only the module is input-blind.
    """
    m = _input_blind(iters)
    x = torch.randn(B, N, D)
    _, a, used = m(x, None, return_init=True)
    # the fixture must actually be a null: init-driven variance live, input dependence absent
    assert float(a.var(dim=0).mean()) > 1e-4, "no batch variance -- degenerate, proves nothing"
    with torch.no_grad():
        _, a2 = m(torch.randn(B, N, D), used)
    assert float((a - a2).abs().max()) == 0.0, "attention still sees the input -- not a null"

    for _ in range(12):
        xi = torch.randn(B, N, D)
        _, sig, init = m(xi, None, return_init=True)
        flat = xi[:1].expand_as(xi).contiguous()
        with torch.no_grad():
            _, null = m(flat, init)
        assert float(sig.var(dim=0).mean()) - float(null.var(dim=0).mean()) == 0.0


def test_unpaired_null_is_NOISY_at_that_same_null_not_merely_biased():
    """⛔ CORRECTS THE CHARACTERISATION. The unpaired form was described as carrying a negative
    bias of -8.0e-05. Its mean is within noise of ZERO; it is symmetrically noisy, sd ~1e-3 to
    3e-3 -- 17-34x that figure, which is one draw from this distribution. The conclusion it
    supported (excess <= 0 cannot discriminate) holds via NOISE, not bias."""
    m = _input_blind(3)
    vals = []
    for _ in range(40):
        xi = torch.randn(B, N, D)
        _, sig, _ = m(xi, None, return_init=True)
        flat = xi[:1].expand_as(xi).contiguous()
        with torch.no_grad():
            _, null = m(flat)                      # UNPAIRED: fresh draw
        vals.append(float(sig.var(dim=0).mean()) - float(null.var(dim=0).mean()))
    assert max(abs(v) for v in vals) > 1e-4, "unpaired null should be visibly noisy"
    assert any(v > 0 for v in vals) and any(v < 0 for v in vals), (
        "excursions should straddle zero -- noise, not a one-sided bias")


# ---------------------------------------------------------------------------------------------
# ⚠ RESIDUAL GAP, STATED RATHER THAN PAPERED OVER. Returning the init from `_slots_for` makes ONE
# error class unrepresentable -- pairing the null to the WRONG pass, which used to be possible by
# reading the `last_init` side-channel late and which cancelled nothing while looking correct.
# It does NOT cover "the null was not paired AT ALL": deleting the `init_t` argument at
# compute()'s null site leaves all 13 tests here GREEN, verified by mutation, because none of
# them invokes compute() (it needs a rollout buffer, temporal pairing and a live trunk).
#
# That residual mutation is a VISIBLE DELETION of an argument rather than a silent mispairing, so
# it is the less dangerous half -- but it is uncovered, and the deleted AST test did catch it.
# Recorded so the next reader does not infer coverage this file does not have.
# ---------------------------------------------------------------------------------------------
