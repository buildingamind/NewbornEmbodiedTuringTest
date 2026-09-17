"""P1 `cltt_patch`: the correspondence is the teacher's argmax, the filter bites, and row 00 moves not at all.

⛔ THE COMPARATOR IS THE POINT OF THIS ROW. It is `cltt_ref` + one term, and the cltt_ref term
must be the control's, computed from its own slab -- so the term's own draw is asserted to be
SEPARATE, and the loss the composite reports for cltt_ref is checked against a standalone
CLTTReferenceAuxLoss elsewhere (test_with_cltt_ref.py).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.brain.aux.cltt_patch_aux import CLTTPatchTerm
from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceProjectionHead
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.aux.with_cltt_ref import WithCLTTRef
from nett_skrl.brain.encoders.compact_vit import CompactViT

H, W, C = 80, 128, 6
SMALL = dict(features_dim=32, patch_size=16, embed_dim=32, depth=1, num_heads=2)
N, N_H, N_W = 40, 5, 8


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("NETT_AUX_PATCH_BATCH", "NETT_AUX_CLTT_PATCH_OFFSET", "NETT_AUX_PATCH_TOPG",
                 "NETT_AUX_PATCH_M", "NETT_AUX_PATCH_TEMP", "NETT_AUX_EMA_DECAY",
                 "NETT_AUX_CLTT_CHANNELS_PER_FRAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    monkeypatch.setenv("NETT_AUX_PATCH_BATCH", "6")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            yield
    finally:
        torch.set_num_threads(threads)


class _Memory:
    def __init__(self, t_max=24, n_env=2):
        cols = torch.arange(W).float()
        obs = torch.zeros(t_max, n_env, H, W, C)
        for t in range(t_max):
            for e in range(n_env):
                phase = t * 6.0 * (1 if e == 0 else -1)
                obs[t, e] = ((torch.sin((cols - phase) / 5.0) * 100 + 128)
                             .view(1, W, 1).expand(H, W, C))
        acts = torch.zeros(t_max, n_env, 2)
        acts[:, 0, 0], acts[:, 1, 0] = 0.4, -0.4
        self.tensors = {"observations": obs.clamp(0, 255).to(torch.uint8),
                        "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
                        "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
                        "actions": acts}
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


def _encoder():
    return CompactViT(gym.spaces.Box(0, 255, shape=(H, W, C), dtype=np.uint8), **SMALL)


def _composite():
    enc = _encoder()
    comp = WithCLTTRef(enc, CLTTPatchTerm(enc), "cltt_patch")
    comp.attach_memory(_Memory())
    return enc, comp


def test_the_projector_is_cltt_refs_on_the_token_dimension():
    term = CLTTPatchTerm(_encoder())
    assert isinstance(term.head, CLTTReferenceProjectionHead)
    assert term.head.net[0].in_features == term.token_dim
    assert term.head.net[-1].out_features == 128 and term.head.net[-1].bias is None
    out = term.head(torch.randn(4, term.token_dim))
    assert torch.allclose(out.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_gradient_reaches_the_trunk_through_both_views_and_never_the_teacher():
    enc, comp = _composite()
    comp.term.compute(enc, None).backward()
    assert enc.patch_embed.weight.grad.abs().sum() > 0
    for name, p in comp.term.head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    assert all(p.grad is None and not p.requires_grad for p in comp._teacher.module.parameters())


def test_the_loss_falls_on_a_fixed_window():
    enc, comp = _composite()
    term = comp.term
    window = term.draw(enc)
    opt = torch.optim.Adam([*enc.parameters(), *term.head.parameters()], lr=3e-3)
    losses = []
    for _ in range(10):
        torch.manual_seed(0)                  # hold the anchor sample fixed across steps
        loss, _ = term._core(enc, window)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] - 1e-3, losses


def test_the_filter_keeps_only_the_most_confident_tokens_and_anchors_come_from_them():
    """γ and M are not decoration: the anchors must be a subset of the top-γ by confidence."""
    enc, comp = _composite()
    term = comp.term
    seen = []
    real_head = term.head

    class _Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = real_head

        def forward(self, x):
            seen.append(x.shape)
            return self.inner(x)

    term.head = _Spy()
    window = term.draw(enc)
    loss, scalars = term._core(enc, window)
    b = window.prepared_t.shape[0]
    assert seen[0] == (b * term.samples, term.token_dim)
    assert scalars["rows"] == 2 * b * term.samples
    # The kept set is strictly more confident than the whole set, or the filter is inert.
    assert scalars["match_conf_kept"] > scalars["match_conf"]


def test_gamma_and_m_are_bounded_by_the_token_count_and_by_each_other(monkeypatch):
    monkeypatch.setenv("NETT_AUX_PATCH_TOPG", str(N + 1))
    with pytest.raises(ValueError, match="exceeds the token count"):
        CLTTPatchTerm(_encoder())
    monkeypatch.setenv("NETT_AUX_PATCH_TOPG", "4")
    monkeypatch.setenv("NETT_AUX_PATCH_M", "8")
    with pytest.raises(ValueError, match="re-admit the low-confidence"):
        CLTTPatchTerm(_encoder())


def test_views_are_the_current_frame_repeated_as_cltt_ref_builds_them():
    """The two terms of this row must look at the SAME construction, or the contrast is confounded
    with the view rather than with tokens-vs-CLS."""
    enc, comp = _composite()
    window = comp.term.draw(enc)
    view_t, view_tk = comp.term._views(window)
    assert view_t.shape == window.prepared_t.shape
    # A 2-frame RGB stack: both halves equal the current frame.
    assert torch.equal(view_t[:, :3], view_t[:, 3:])
    assert torch.equal(view_t[:, 3:], window.prepared_t[:, 3:])
    assert not torch.equal(view_t, view_tk)


def test_identity_correspondence_reads_as_identity_and_a_shifted_one_does_not():
    term = CLTTPatchTerm(_encoder())
    idx = torch.arange(N).expand(4, -1)
    turn = torch.tensor([0.1, 0.2, 0.3, 0.4])
    assert term.identity_fraction(idx, turn)["identity_frac"] == 1.0
    rolled = idx.view(4, N_H, N_W).roll(1, dims=2).reshape(4, N)
    assert term.identity_fraction(rolled, turn)["identity_frac"] == 0.0


def test_the_deranged_positive_null_is_emitted_beside_the_accuracy():
    enc, comp = _composite()
    comp.term.compute(enc, None)
    s = comp.term.last_scalars
    assert {"patch_pos_acc", "patch_shuffled_acc", "patch_chance", "patch_pos_sim",
            "patch_neg_sim", "identity_frac_parked", "identity_frac_transit", "shift_rho",
            "shift_rho_null", "match_conf", "top_gamma", "rows"} <= set(s)
    assert 0.0 <= s["patch_pos_acc"] <= 1.0


def test_it_draws_its_own_window_and_never_widens_cltt_refs_slab(monkeypatch):
    """⛔ THE COMPARATOR HAZARD. Sharing a slab by adding offset 8 to cltt_ref's offsets would
    change the set of valid slab starts for row 00's own term. The term's offsets must never
    reach cltt_ref."""
    enc, comp = _composite()
    assert comp.cltt_ref.offsets == (1, 2)          # the control's, untouched
    assert comp.term.offset == 8
    assert comp.cltt_ref.max_samples == 4 and comp.term.batch == 6


def test_knobs_are_read_and_invalid_values_raise(monkeypatch):
    monkeypatch.setenv("NETT_AUX_PATCH_TOPG", "12")
    monkeypatch.setenv("NETT_AUX_PATCH_M", "3")
    monkeypatch.setenv("NETT_AUX_PATCH_TEMP", "0.2")
    monkeypatch.setenv("NETT_AUX_CLTT_PATCH_OFFSET", "16")
    term = CLTTPatchTerm(_encoder())
    assert (term.top_gamma, term.samples, term.temperature, term.offset) == (12, 3, 0.2, 16)
    for name, bad in (("NETT_AUX_PATCH_TOPG", "0"), ("NETT_AUX_PATCH_M", "-2"),
                      ("NETT_AUX_PATCH_TEMP", "0"), ("NETT_AUX_PATCH_BATCH", "abc")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError, match=name):
            CLTTPatchTerm(_encoder())
        monkeypatch.delenv(name)
    monkeypatch.setenv("NETT_AUX_PATCH_BATCH", "6")


def test_defaults_are_the_spec_values():
    monkey_free = CLTTPatchTerm(_encoder())
    assert monkey_free.top_gamma == N // 2 and monkey_free.samples == 8
    assert monkey_free.temperature == 0.5 and monkey_free.offset == 8


def test_the_row_is_registered_as_cltt_ref_plus_one_term():
    aux = AUX_LOSSES["cltt_patch"](_encoder())
    assert isinstance(aux, WithCLTTRef) and isinstance(aux.term, CLTTPatchTerm)
    assert aux.name == "cltt_patch" and aux._teacher is not None
