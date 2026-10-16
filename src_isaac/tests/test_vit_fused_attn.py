"""NETT_VIT_FUSED_ATTN: CompactViT's attention without the materialised weight map (workspace DECISIONS §131 add. 1 item 16).

nn.MultiheadAttention returns (output, weights) and, with the default need_weights=True, builds the
(B, heads, N, N) softmax map on the bmm path and keeps it for backward. At 448x280 / patch 14 (641 tokens)
that map is what the §128 ViT+DVS update runs out of memory on. need_weights=False calls
F.scaled_dot_product_attention instead: the same operator on the same parameters. Unset is the old path,
literally.
"""

from __future__ import annotations

import os

# cuBLAS reads CUBLAS_WORKSPACE_CONFIG ONCE, when the process creates its handle. In a combined run an earlier CUDA
# test creates the handle first, so the CUDA test below set it too late and strict_determinism refused the bf16
# backward ("uses CuBLAS ... CUBLAS_WORKSPACE_CONFIG"). Set at collection, before any test runs, as the driver
# sets it at startup; it only selects the deterministic workspace.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.encoders.compact_vit import CompactViT
from nett_skrl.brain.skrl_patches import strict_determinism

C, H, W = 4, 42, 56          # 3 x 4 patches at patch 14 -> 13 tokens with CLS
CFG = dict(features_dim=32, patch_size=14, embed_dim=24, depth=2, num_heads=4, mlp_ratio=2.0, pool="cls")


def _make(monkeypatch, knob, seed=0, **over):
    if knob is None:
        monkeypatch.delenv("NETT_VIT_FUSED_ATTN", raising=False)
    else:
        monkeypatch.setenv("NETT_VIT_FUSED_ATTN", knob)
    torch.manual_seed(seed)
    return CompactViT(gym.spaces.Box(0, 255, (C, H, W), np.uint8), **{**CFG, **over})


def _obs(n=5, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, C, H, W), generator=g).float()


def _record(enc, monkeypatch):
    """Record every attention call: ("mha", need_weights kwarg or None) for nn.MultiheadAttention.forward,
    ("sdpa", padded head dim) for F.scaled_dot_product_attention."""
    import torch.nn.functional as F
    seen = []
    for blk in enc.blocks:
        orig = blk.attn.forward

        def fwd(*a, _orig=orig, **kw):
            seen.append(("mha", kw.get("need_weights")))
            return _orig(*a, **kw)
        blk.attn.forward = fwd
    real = F.scaled_dot_product_attention

    def sdpa(q, *a, **kw):
        seen.append(("sdpa", q.shape[-1]))
        return real(q, *a, **kw)
    monkeypatch.setattr(F, "scaled_dot_product_attention", sdpa)
    return seen


@pytest.mark.parametrize("knob", [None, "0", "false", ""])
def test_unset_or_off_keeps_the_weight_map_path(monkeypatch, knob):
    enc = _make(monkeypatch, knob)
    assert enc.fused_attn is False and all(b.need_weights for b in enc.blocks)
    seen = _record(enc, monkeypatch)
    enc(_obs())
    assert seen == [("mha", None)] * CFG["depth"]          # the literal pre-knob call, no keyword at all


def test_on_calls_attention_without_weights(monkeypatch):
    enc = _make(monkeypatch, "1")
    assert enc.fused_attn is True and not any(b.need_weights for b in enc.blocks)
    seen = _record(enc, monkeypatch)
    enc(_obs())
    assert seen == [("sdpa", 8)] * CFG["depth"]            # head dim 24/4 = 6, zero-padded to 8


@pytest.mark.parametrize("embed,heads,want", [(32, 4, 8), (108, 4, 32), (177, 3, 64), (24, 3, 8)])
def test_head_dim_is_padded_to_a_multiple_of_8(monkeypatch, embed, heads, want):
    enc = _make(monkeypatch, "1", embed_dim=embed, num_heads=heads)
    seen = _record(enc, monkeypatch)
    enc(_obs(n=2))
    assert seen == [("sdpa", want)] * CFG["depth"]


def test_misspelled_value_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="NETT_VIT_FUSED_ATTN"):
        _make(monkeypatch, "fused")


def test_non_qk_operator_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="only 'qk'"):
        _make(monkeypatch, "1", attn_mode="uniform")
    assert _make(monkeypatch, None, attn_mode="uniform").fused_attn is False


def test_same_parameters_same_init(monkeypatch):
    a, b = _make(monkeypatch, None).state_dict(), _make(monkeypatch, "1").state_dict()
    assert list(a) == list(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("train", [False, True])
def test_outputs_and_gradients_match_the_weight_map_path(monkeypatch, train):
    ref, fused = _make(monkeypatch, None), _make(monkeypatch, "1")
    fused.load_state_dict(ref.state_dict())
    ref.train(train); fused.train(train)
    x = _obs()
    ya, yb = ref(x), fused(x)
    assert torch.allclose(ya, yb, atol=1e-5, rtol=1e-4)
    ya.square().sum().backward(); yb.square().sum().backward()
    ga = {n: p.grad for n, p in ref.named_parameters() if p.grad is not None}
    gb = {n: p.grad for n, p in fused.named_parameters() if p.grad is not None}
    assert ga.keys() == gb.keys() and len(ga) > 0
    for n in ga:
        assert torch.allclose(ga[n], gb[n], atol=1e-5, rtol=1e-3), n
    # positive control: the comparison can see a difference -- one perturbed attention weight moves the output
    with torch.no_grad():
        fused.blocks[0].attn.in_proj_weight[0, 0] += 0.5
    assert not torch.allclose(ref(x), fused(x), atol=1e-5, rtol=1e-4)


def test_dropout_path_also_skips_the_weight_map(monkeypatch):
    enc = _make(monkeypatch, "1", dropout=0.1, dropout_scope="all").train()
    seen = _record(enc, monkeypatch)
    enc(_obs())
    assert seen == [("sdpa", 8)] * CFG["depth"]


@pytest.mark.parametrize("embed,heads", [(24, 4), (32, 4)])     # padded (d=6) and aligned (d=8)
def test_fused_matches_the_weight_map_path_at_other_head_dims(monkeypatch, embed, heads):
    ref, fused = _make(monkeypatch, None, embed_dim=embed, num_heads=heads), _make(monkeypatch, "1", embed_dim=embed, num_heads=heads)
    fused.load_state_dict(ref.state_dict())
    x = _obs()
    assert torch.allclose(ref(x), fused(x), atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------- NETT_VIT_ACT_CKPT

def _make_ck(monkeypatch, ck, fused=None, **over):
    if ck is None:
        monkeypatch.delenv("NETT_VIT_ACT_CKPT", raising=False)
    else:
        monkeypatch.setenv("NETT_VIT_ACT_CKPT", ck)
    return _make(monkeypatch, fused, **over)


def test_act_ckpt_unset_is_off_and_misspelling_refused(monkeypatch):
    assert _make_ck(monkeypatch, None).act_ckpt is False
    assert _make_ck(monkeypatch, "0").act_ckpt is False
    with pytest.raises(ValueError, match="NETT_VIT_ACT_CKPT"):
        _make_ck(monkeypatch, "on please")


@pytest.mark.parametrize("fused", [None, "1"])
def test_act_ckpt_gives_identical_outputs_and_gradients(monkeypatch, fused):
    ref = _make_ck(monkeypatch, None, fused).train()
    ck = _make_ck(monkeypatch, "1", fused).train()
    ck.load_state_dict(ref.state_dict())
    calls = []
    import torch.utils.checkpoint as tuc
    real = tuc.checkpoint
    monkeypatch.setattr(tuc, "checkpoint", lambda *a, **kw: (calls.append(kw.get("use_reentrant")), real(*a, **kw))[1])
    x = _obs()
    ya, yb = ref(x), ck(x)
    assert calls == [False] * CFG["depth"]                   # positive control: the checkpoint path ran
    assert torch.equal(ya, yb)
    ya.square().sum().backward(); yb.square().sum().backward()
    for (n, pa), (_, pb) in zip(ref.named_parameters(), ck.named_parameters()):
        if pa.grad is not None:
            assert torch.allclose(pa.grad, pb.grad, atol=1e-6, rtol=1e-5), n


def test_act_ckpt_is_skipped_without_grad(monkeypatch):
    ck = _make_ck(monkeypatch, "1").eval()
    calls = []
    import torch.utils.checkpoint as tuc
    real = tuc.checkpoint
    monkeypatch.setattr(tuc, "checkpoint", lambda *a, **kw: (calls.append(1), real(*a, **kw))[1])
    with torch.no_grad():
        ck(_obs())
    assert calls == []


def test_runs_under_strict_determinism_and_repeats(monkeypatch):
    outs = []
    for _ in range(2):
        enc = _make(monkeypatch, "1").train()
        with strict_determinism():
            y = enc(_obs())
            y.sum().backward()
        outs.append((y.detach(), enc.blocks[0].attn.in_proj_weight.grad.clone()))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("label_cfg,ckpt,b_ref", [(dict(embed_dim=108, depth=3, num_heads=4), False, 100),                  # VD500 (VIT_500K_P14_DVS)
                                                  (dict(embed_dim=177, depth=12, num_heads=3, mlp_ratio=4.0), True, 50)])  # VD5M (VIT_5M_DVS)
def test_cuda_amp_448x280_update_memory_drops_and_repeats(monkeypatch, label_cfg, ckpt, b_ref):
    """The §128 shape: 4 channels at 448x280, patch 14 (641 tokens), under the autocast the card's rows run: bf16 on
    sm80+ (lion/insect A10), fp16 below (chicken sm75: no bf16 mem-efficient SDPA kernel there, so a bf16 run falls
    back to the math kernel and cannot show the drop; presenter, 2026-10-16).
    1. At b_ref, a batch where the weight-map path FITS on a 24 GB card (A10: VD500 5.3 GiB at 100, VD5M 7.5 GiB at
       50), the fused peak (VD5M: fused + activation checkpointing) is well below the weight-map peak. The weight-map
       run must not OOM: an OOM there would make the comparison vacuous, so it fails the test instead.
    2. At the 400-sample minibatch (25,600 / 64), the fused path runs, and two runs are bit-identical under
       strict_determinism."""
    amp = "bf16" if torch.cuda.get_device_capability() >= (8, 0) else "fp16"
    monkeypatch.setenv("NETT_AMP", amp)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # also set at collection (top of module)
    dev = torch.device("cuda")
    cfg = {**dict(features_dim=512, patch_size=14, pool="cls", stem="linear", mlp_ratio=2.0), **label_cfg}
    x = torch.randint(0, 256, (400, 4, 280, 448), device=dev, dtype=torch.uint8).float()

    def peak(knob, n):
        if knob:
            monkeypatch.setenv("NETT_VIT_FUSED_ATTN", "1")
            monkeypatch.setenv("NETT_VIT_ACT_CKPT", "1" if ckpt else "0")
        else:
            monkeypatch.delenv("NETT_VIT_FUSED_ATTN", raising=False)
            monkeypatch.delenv("NETT_VIT_ACT_CKPT", raising=False)
        torch.manual_seed(0)
        enc = CompactViT(gym.spaces.Box(0, 255, (4, 280, 448), np.uint8), **cfg).to(dev).train()
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with strict_determinism():
            y = enc(x[:n])
            y.float().square().mean().backward()
        torch.cuda.synchronize()
        g = enc.blocks[0].attn.in_proj_weight.grad.detach().clone()
        p = torch.cuda.max_memory_allocated()
        del enc, y
        torch.cuda.empty_cache()
        return p, g

    p_ref, _ = peak(False, b_ref)
    p_fused, _ = peak(True, b_ref)
    print(f"{amp} peak MiB at B={b_ref}: weight-map path {p_ref / 2**20:.0f}, fused {p_fused / 2**20:.0f}")
    assert p_fused < 0.6 * p_ref
    p1, g1 = peak(True, 400)
    p2, g2 = peak(True, 400)
    print(f"peak MiB at B=400: fused {p1 / 2**20:.0f}")
    assert torch.isfinite(g1).all()      # a NaN gradient would also fail torch.equal; say why
    assert torch.equal(g1, g2)
