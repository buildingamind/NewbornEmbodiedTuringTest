"""The shared-encoder feature cache: gradient-identical, and it frees what it holds.

⛔ THE CLAIM UNDER TEST IS EQUIVALENCE, NOT SPEED. Caching is only admissible if the gradient
deposited in every encoder parameter is the same one the two-pass code produced. Everything else
this change buys is worthless if that fails, so it is asserted first and on real modules.

⭐ THE FIXTURES RUN BOTH BRANCHES. NETT_DISABLE_FEATURE_CACHE restores the pre-cache path, so
these are A/B comparisons against code that actually runs -- not against a remembered number.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from nett_skrl.brain.models.utils.features import shared_feature_cache


class _Enc(nn.Module):
    """Deterministic stand-in with the interface features_forward needs."""

    def __init__(self, din=12, dout=8, bn=False):
        super().__init__()
        self.lin = nn.Linear(din, dout)
        self.bn = nn.BatchNorm1d(dout) if bn else None
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        y = self.lin(x.flatten(1))
        return self.bn(y) if self.bn is not None else y


class _Head(nn.Module):
    def __init__(self, enc, dout=8):
        super().__init__()
        self.encoder = enc
        self.trunk = nn.Linear(dout, dout)
        self.head = nn.Linear(dout, 1)

    def compute(self, inputs):
        from nett_skrl.brain.models.utils.features import features_forward
        return self.head(features_forward(self, inputs))


def _build(bn=False, seed=0):
    torch.manual_seed(seed)
    enc = _Enc(bn=bn)
    torch.manual_seed(seed + 1)
    actor = _Head(enc)
    torch.manual_seed(seed + 2)
    critic = _Head(enc)
    return enc, actor, critic


def _run(cache: bool, bn=False, seed=0):
    """One minibatch: actor pass, critic pass, one backward. Returns grads + call count."""
    prev = os.environ.get("NETT_DISABLE_FEATURE_CACHE")
    os.environ["NETT_DISABLE_FEATURE_CACHE"] = "" if cache else "1"
    try:
        enc, actor, critic = _build(bn=bn, seed=seed)
        torch.manual_seed(99)
        x = torch.randn(16, 3, 4)
        inputs = {"observations": x}
        # ⛔ ENTER THE CONTEXT, because that is the only place the cache is live. The first
        # version of these fixtures called features_forward directly and so tested a path no
        # agent takes -- they passed while exercising nothing.
        with shared_feature_cache(enc):
            la = actor.compute({**inputs, "taken_actions": torch.zeros(16, 1)}).sum()
            lc = critic.compute(inputs).sum()
            (la + lc).backward()
        grads = {n: p.grad.clone() for n, p in enc.named_parameters() if p.grad is not None}
        buffers = {n: b.clone() for n, b in enc.named_buffers()}
        return grads, buffers, enc.calls
    finally:
        if prev is None:
            os.environ.pop("NETT_DISABLE_FEATURE_CACHE", None)
        else:
            os.environ["NETT_DISABLE_FEATURE_CACHE"] = prev


def test_encoder_runs_once_with_cache_and_twice_without():
    # ⛔ The control: if this reads 2/2 the cache is inert and every equivalence test below
    # passes VACUOUSLY, comparing the two-pass path against itself.
    assert _run(cache=False)[2] == 2
    assert _run(cache=True)[2] == 1


def _worst_relative(g_a, g_b):
    """max|a-b| divided by the gradient SCALE.

    ⛔ NOT torch.testing.assert_close's per-element rtol. Gradient tensors contain entries near
    zero, where a relative test divides by ~1e-7 and reports a "9x difference" that is one unit
    in the last place. Measured: that is exactly how this suite first failed on a change that
    was numerically exact. The meaningful question is the error RELATIVE TO THE SIGNAL, so the
    denominator is the largest gradient, not each element.
    """
    worst = max((g_a[n] - g_b[n]).abs().max().item() for n in g_b)
    scale = max(g_b[n].abs().max().item() for n in g_b)
    return worst / scale


def test_gradients_are_identical():
    """THE claim. Same gradient in every encoder parameter, one pass or two."""
    g_two, _, _ = _run(cache=False)
    g_one, _, _ = _run(cache=True)
    assert set(g_two) == set(g_one) and g_two
    assert _worst_relative(g_one, g_two) < 1e-6


def test_equivalence_is_exact_and_not_merely_close():
    """⛔ THE TEST THAT SEPARATES "numerically identical" FROM "close enough", AND IT IS THE ONE
    THAT MATTERS. Summing two gradients at a shared node orders the float additions differently
    from accumulating them in two backward passes, so float32 CANNOT distinguish an exact change
    from a slightly-wrong one -- both land around 1e-8.

    In float64 an accumulation-order artefact must shrink with machine epsilon while a genuine
    semantic difference stays put. Measured on this change: float32 1.55e-08 -> float64 4.41e-16,
    a fall of ~7 orders. A real difference would not move.
    [[a-tolerance-must-be-scale-free]]
    """
    torch.set_default_dtype(torch.float64)
    try:
        g_two, _, _ = _run(cache=False, bn=True)
        g_one, _, _ = _run(cache=True, bn=True)
        assert _worst_relative(g_one, g_two) < 1e-12
    finally:
        torch.set_default_dtype(torch.float32)


def test_trunks_stay_separate():
    """⛔ Only the ENCODER output is cached. Caching the trunk would hand the critic the
    actor's value basis -- a different model that would still train and still look fine."""
    enc, actor, critic = _build()
    assert actor.trunk is not critic.trunk
    x = torch.randn(16, 3, 4)
    with shared_feature_cache(enc):
        out_a = actor.compute({"observations": x})
        out_c = critic.compute({"observations": x})
    assert not torch.allclose(out_a, out_c)


def test_a_different_minibatch_is_a_cache_miss():
    """⛔ Identity keying, so a same-shaped NEW tensor must MISS. A shape or device key would
    collide here and feed one minibatch's features to the next -- silently, and forever."""
    enc, actor, critic = _build()
    x1, x2 = torch.randn(16, 3, 4), torch.randn(16, 3, 4)
    with shared_feature_cache(enc):
        actor.compute({"observations": x1})
        assert enc.calls == 1
        critic.compute({"observations": x1})
        assert enc.calls == 1, "same tensor must hit"
        critic.compute({"observations": x2})
        assert enc.calls == 2, "a different tensor must miss"


def test_clear_releases_the_graph():
    """⛔ The cached tensor owns its autograd graph. If clear() does not drop it, the cache
    PINS the memory it was added to save -- and nothing else in this suite would notice."""
    from nett_skrl.brain.models.utils.features import clear_feature_cache, _CACHE_ATTR
    enc, actor, _ = _build()
    with shared_feature_cache(enc):
        actor.compute({"observations": torch.randn(16, 3, 4)})
        assert getattr(enc, _CACHE_ATTR, None) is not None
    assert getattr(enc, _CACHE_ATTR, None) is None, "the context must clear on exit"
    with shared_feature_cache(enc):
        actor.compute({"observations": torch.randn(16, 3, 4)})
    clear_feature_cache(enc)
    assert getattr(enc, _CACHE_ATTR, None) is None
    clear_feature_cache(enc)          # idempotent
    clear_feature_cache(None)         # tolerates a model without an encoder


def test_batchnorm_running_stats_updated_once_not_twice():
    """The latent defect this repairs: two passes applied TWO momentum updates from ONE batch.

    Outputs and gradients were identical either way; only the eval-time buffers differed.
    Asserting the buffers directly, because that is the only observable that ever changed.
    """
    _, buf_two, calls_two = _run(cache=False, bn=True)
    _, buf_one, calls_one = _run(cache=True, bn=True)
    assert (calls_two, calls_one) == (2, 1)
    assert "bn.running_mean" in buf_two
    assert not torch.equal(buf_two["bn.running_mean"], buf_one["bn.running_mean"]), \
        "if these match, the double update was not happening and the premise is wrong"
    # one update from zero-init: running = (1-m) * 0 + m * batch_mean
    assert buf_one["bn.num_batches_tracked"].item() == 1
    assert buf_two["bn.num_batches_tracked"].item() == 2


def test_gradients_identical_even_with_batchnorm():
    """⛔ The repair must not move the GRADIENT -- only the buffers. Train-mode BN normalises by
    the current batch, so both passes see identical statistics; it is the running EMA alone that
    double-counted. If this fails, the change is not a repair, it is a different model."""
    g_two, _, _ = _run(cache=False, bn=True)
    g_one, _, _ = _run(cache=True, bn=True)
    assert _worst_relative(g_one, g_two) < 1e-6


def test_decouple_detaches_the_head_path_but_not_the_cache():
    """⛔ NETT_DECOUPLE_ENCODER detaches the RL path. The CACHED tensor must stay ATTACHED, or
    the aux losses -- which call model.encoder directly -- would silently get a frozen encoder."""
    from nett_skrl.brain.models.utils.features import _CACHE_ATTR
    prev = os.environ.get("NETT_DECOUPLE_ENCODER")
    os.environ["NETT_DECOUPLE_ENCODER"] = "1"
    try:
        enc, actor, _ = _build()
        with shared_feature_cache(enc):
            actor.compute({"observations": torch.randn(16, 3, 4)}).sum().backward()
            cached = getattr(enc, _CACHE_ATTR, None)
            assert cached is not None
            assert cached[1].requires_grad, "the cached tensor must NOT be the detached one"
        assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
                   for p in enc.lin.parameters()), "decouple must still cut the RL gradient"
    finally:
        if prev is None:
            os.environ.pop("NETT_DECOUPLE_ENCODER", None)
        else:
            os.environ["NETT_DECOUPLE_ENCODER"] = prev
