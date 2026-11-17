"""CLTT-WINDOW program (owner 2026-11-17): cltt_win -- scheme parsing, offset sampling, keyed exclusion mask, compute."""

from __future__ import annotations

import math

import pytest
import torch

from nett_skrl.brain.aux.cltt_win_aux import (CLTTWindowAuxLoss, exclusion_mask, nt_xent_keyed, offset_weights,
                                              parse_scheme, sample_offsets)
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from test_cltt_ref_aux import IdentityEncoder, identity_memory


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("NETT_AUX_BATCH", "NETT_AUX_CLTT_WIN", "NETT_AUX_CLTT_WIN_M", "NETT_AUX_CLTT_REF_TEMP",
              "NETT_CLTT_REF_DEDUP", "NETT_AUX_CLTT_CHANNELS_PER_FRAME"):
        monkeypatch.delenv(k, raising=False)
    torch.manual_seed(5)


@pytest.mark.parametrize("raw,base", [("uni:8", "uni"), ("lin:16", "lin"), ("exp:4:24", "exp"), ("epi", "epi"),
                                      ("uni:1+x0.25", "uni"), ("EPI+x0.5", "epi")])
def test_parse_ok(raw, base):
    assert parse_scheme(raw)[0] == base


@pytest.mark.parametrize("raw", ["", "uni", "uni:0", "uni:65", "exp:0:4", "lin:8+x1", "lin:8+x0", "gauss:4", "epi+x"])
def test_parse_refuses(raw):
    with pytest.raises(ValueError):
        parse_scheme(raw)


@pytest.mark.parametrize("raw,wmax", [("uni:4", 4), ("lin:8", 8), ("exp:2:6", 6), ("epi", 49)])
def test_offsets_in_window_nonzero_and_bounded(raw, wmax):
    base, params, _ = parse_scheme(raw)
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        d = sample_offsets(base, params, 50, generator=g)
        tgt = torch.arange(50) + d
        assert (d != 0).all() and (tgt >= 0).all() and (tgt < 50).all() and (d.abs() <= wmax).all()


def test_lin_and_exp_drop_off_and_epi_reaches_far():
    g = torch.Generator().manual_seed(1)
    for raw in ("lin:8", "exp:2:8"):
        base, params, _ = parse_scheme(raw)
        d = torch.cat([sample_offsets(base, params, 50, generator=g) for _ in range(400)]).abs()
        counts = torch.bincount(d, minlength=9)[1:9].double()
        assert counts[0] > counts[3] > counts[7]
    base, params, _ = parse_scheme("epi")
    d = torch.cat([sample_offsets(base, params, 50, generator=g) for _ in range(100)]).abs().double()
    assert d.mean() > 12 and d.max() >= 40
    w = offset_weights("uni", [4.0], 50)
    assert w[:4].tolist() == [1, 1, 1, 1] and float(w[4:].sum()) == 0


def test_exclusion_mask_symmetric_and_spares_self_and_positive():
    # anchors frames 0..3, positives at frames 1, 0, 3, 2 (each positive IS another anchor's frame)
    ka = torch.tensor([0, 1, 2, 3])
    kp = torch.tensor([1, 0, 3, 2])
    ex = exclusion_mask(ka, kp)
    assert torch.equal(ex, ex.t())
    b = 4
    for r in range(2 * b):
        assert not ex[r, r] and not ex[r, (r + b) % (2 * b)]
    # row 0 (frame 0, positive frame 1): the anchor copy of frame 1 (col 1) and the positive copy of frame 0 (col 5) leave
    assert ex[0, 1] and ex[0, 5]
    assert not ex[0, 2] and not ex[0, 3]


def test_keyed_loss_matches_plain_ntxent_without_duplicates():
    za = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    zp = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    loss, ex, fl = nt_xent_keyed(za, zp, 0.5, torch.arange(6), torch.arange(100, 106))
    z = torch.cat([za, zp])
    sim = z @ z.t() / 0.5
    sim.fill_diagonal_(float("-inf"))
    ref = torch.nn.functional.cross_entropy(sim, (torch.arange(12) + 6) % 12)
    assert ex == 0 and torch.allclose(loss, ref) and abs(fl - math.log(11)) < 1e-9


@pytest.mark.parametrize("scheme", ["uni:1", "uni:8", "lin:16", "exp:4:24", "epi", "uni:1+x0.5"])
def test_compute_runs_with_gradient(scheme, monkeypatch):
    monkeypatch.setenv("NETT_AUX_CLTT_WIN", scheme)
    monkeypatch.setenv("NETT_AUX_BATCH", "256")
    enc = IdentityEncoder(channels=6)
    aux = CLTTWindowAuxLoss(enc)
    aux.attach_memory(identity_memory(t_max=30, num_envs=4))
    loss = aux.compute(enc, None)
    assert torch.isfinite(loss)
    loss.backward()
    assert enc.scale.grad is not None and float(enc.scale.grad.abs().sum()) > 0
    s = aux.last_scalars
    assert s["B"] == 30 and s["mask_regime"] == 3.0
    if "+x" in scheme:
        assert 0.2 < s["win_xenv_frac"] < 0.8
    else:
        assert s["win_xenv_frac"] == 0.0


def test_cross_env_positives_come_from_other_envs(monkeypatch):
    monkeypatch.setenv("NETT_AUX_CLTT_WIN", "uni:1+x0.9")
    monkeypatch.setenv("NETT_AUX_CLTT_WIN_M", "1")
    enc = IdentityEncoder(channels=6)
    aux = CLTTWindowAuxLoss(enc)
    aux.attach_memory(identity_memory(t_max=20, num_envs=3))
    aux.compute(enc, None)
    span_env = int(enc.views[0][0, 0].item()) % 1000
    assert all(int(v) % 1000 == span_env for v in enc.views[0][:, 0])
    xenvs = {int(v) % 1000 for v in enc.views[-1][:, 0]}
    assert xenvs and span_env not in xenvs


def test_registered_and_requires_scheme(monkeypatch):
    assert "cltt_win" in AUX_LOSSES
    with pytest.raises(ValueError):
        CLTTWindowAuxLoss(IdentityEncoder(channels=6))
