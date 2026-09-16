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


def test_paired_null_is_EXACTLY_zero_at_zero_input_dependence():
    """⭐ THE PROPERTY. Identical inputs + shared init => the two passes are the SAME
    computation, so the excess is exactly 0.0 -- not merely small. Every trial, no tolerance."""
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
    _, attn1, _, _ = aux._slots_for(enc, x)
    init = aux.head.attn.last_init
    with torch.no_grad():
        _, attn2, _, _ = aux._slots_for(enc, x, init)
    assert torch.allclose(attn1, attn2, atol=0, rtol=0), (
        "_slots_for ignored the init it was handed -- the null is not paired to its signal")


def test_paired_null_is_exactly_zero_THROUGH_slots_for():
    """The end-to-end property on the real path: identical inputs + the signal's own init."""
    aux, enc = _aux_with_encoder()
    x = torch.randn(1, 3, 6, 12).expand(B, 3, 6, 12).contiguous()
    _, attn_sig, _, _ = aux._slots_for(enc, x)
    init = aux.head.attn.last_init
    flat = x[:1].expand_as(x).contiguous()
    with torch.no_grad():
        _, attn_null, _, _ = aux._slots_for(enc, flat, init)
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

    _, attn_t, _, _ = aux._slots_for(enc, x_t)
    init_t = aux.head.attn.last_init              # captured HERE, as compute() does
    aux._slots_for(enc, x_n)                      # overwrites last_init
    assert not torch.allclose(init_t, aux.head.attn.last_init), "fixture failed to overwrite"

    flat = x_t[:1].expand_as(x_t).contiguous()
    with torch.no_grad():
        _, attn_null, _, _ = aux._slots_for(enc, flat, init_t)
    assert float(attn_t.var(dim=0).mean()) - float(attn_null.var(dim=0).mean()) == 0.0

    # ...and pairing to the WRONG (intervening) draw does NOT cancel -- the bug's signature.
    with torch.no_grad():
        _, attn_wrong, _, _ = aux._slots_for(enc, flat, aux.head.attn.last_init)
    assert float(attn_t.var(dim=0).mean()) - float(attn_wrong.var(dim=0).mean()) != 0.0


def test_compute_CAPTURES_the_init_before_the_second_slots_for_call():
    """⚠ STRUCTURAL, NOT BEHAVIOURAL -- and deliberately so, with the limitation stated.

    ⛔ The test above reproduces compute()'s two-pass sequence but REIMPLEMENTS it, so mutating
    compute() itself leaves it green: verified by mutation, replacing `init_t` with
    `self.head.attn.last_init` at the null site passed the entire behavioural suite. Calling
    compute() for real needs a rollout buffer, temporal pairing and a live trunk, which is a
    heavier fixture than this property warrants. So this asserts on compute()'s SOURCE instead.

    It binds to the real function and catches the real mutation. It cannot notice a defect that
    keeps this shape, and it breaks on harmless refactors -- both are the price of the choice.
    """
    import ast
    import inspect

    src = inspect.getsource(sc.SlotContrastAuxLoss.compute)
    tree = ast.parse(textwrap_dedent(src))

    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "_slots_for"]
    assert len(calls) >= 3, f"expected signal, temporal-pair and null passes; saw {len(calls)}"

    # the null pass (last) must be handed a NAME bound earlier, never a fresh `.last_init` read
    null_call = max(calls, key=lambda n: n.lineno)
    assert len(null_call.args) >= 3, "the null pass is not being given an init at all"
    init_arg = null_call.args[2]
    assert isinstance(init_arg, ast.Name), (
        "the null pass reads its init inline rather than from a variable captured earlier; "
        "`last_init` holds the MOST RECENT draw, so this pairs the null to the wrong pass")

    binding = [n for n in ast.walk(tree)
               if isinstance(n, ast.Assign)
               and any(getattr(t, "id", None) == init_arg.id for t in n.targets)]
    assert binding, f"`{init_arg.id}` is never assigned in compute()"
    second_call_line = sorted(c.lineno for c in calls)[1]
    assert binding[0].lineno < second_call_line, (
        f"`{init_arg.id}` is captured at line {binding[0].lineno}, after the pass at "
        f"{second_call_line} that overwrites `last_init` -- it pairs to the wrong draw")


def textwrap_dedent(s):
    import textwrap
    return textwrap.dedent(s)
