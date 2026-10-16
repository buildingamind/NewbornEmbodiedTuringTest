"""NETT_VIT_FUSED_ATTN: CompactViT's attention without the materialised weight map (workspace DECISIONS §131 add. 1 item 16).

nn.MultiheadAttention returns (output, weights) and, with the default need_weights=True, builds the
(B, heads, N, N) softmax map on the bmm path and keeps it for backward. At 448x280 / patch 14 (641 tokens)
that map is what the §128 ViT+DVS update runs out of memory on. need_weights=False calls
F.scaled_dot_product_attention instead: the same operator on the same parameters. Unset is the old path,
literally.
"""

from __future__ import annotations

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


def _record(enc):
    """Wrap every block's attention module and record the need_weights each call passed (None = default)."""
    seen = []
    for blk in enc.blocks:
        orig = blk.attn.forward

        def fwd(*a, _orig=orig, **kw):
            seen.append(kw.get("need_weights"))
            return _orig(*a, **kw)
        blk.attn.forward = fwd
    return seen


@pytest.mark.parametrize("knob", [None, "0", "false", ""])
def test_unset_or_off_keeps_the_weight_map_path(monkeypatch, knob):
    enc = _make(monkeypatch, knob)
    assert enc.fused_attn is False and all(b.need_weights for b in enc.blocks)
    seen = _record(enc)
    enc(_obs())
    assert seen == [None] * CFG["depth"]          # the literal pre-knob call, no keyword at all


def test_on_calls_attention_without_weights(monkeypatch):
    enc = _make(monkeypatch, "1")
    assert enc.fused_attn is True and not any(b.need_weights for b in enc.blocks)
    seen = _record(enc)
    enc(_obs())
    assert seen == [False] * CFG["depth"]


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
    seen = _record(enc)
    enc(_obs())
    assert seen == [False] * CFG["depth"]


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
@pytest.mark.parametrize("label_cfg", [dict(embed_dim=108, depth=3, num_heads=4),                  # VD500 (VIT_500K_P14_DVS)
                                       dict(embed_dim=177, depth=12, num_heads=3, mlp_ratio=4.0)])  # VD5M (VIT_5M_DVS)
def test_cuda_bf16_448x280_update_memory_drops_and_repeats(monkeypatch, label_cfg):
    """The §128 shape: 4 channels at 448x280, patch 14 (641 tokens), bf16 autocast, a 400-sample minibatch
    (25,600 / 64). Fused peak must be lower, and two runs bit-identical under strict_determinism."""
    monkeypatch.setenv("NETT_AMP", "bf16")
    dev = torch.device("cuda")
    cfg = dict(features_dim=512, patch_size=14, pool="cls", stem="linear", mlp_ratio=2.0, **label_cfg)
    x = torch.randint(0, 256, (400, 4, 280, 448), device=dev, dtype=torch.uint8).float()

    def peak(knob):
        if knob:
            monkeypatch.setenv("NETT_VIT_FUSED_ATTN", "1")
        else:
            monkeypatch.delenv("NETT_VIT_FUSED_ATTN", raising=False)
        torch.manual_seed(0)
        enc = CompactViT(gym.spaces.Box(0, 255, (4, 280, 448), np.uint8), **cfg).to(dev).train()
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with strict_determinism():
            y = enc(x)
            y.float().square().mean().backward()
        torch.cuda.synchronize()
        g = enc.blocks[0].attn.in_proj_weight.grad.detach().clone()
        p = torch.cuda.max_memory_allocated()
        del enc, y
        return p, g

    try:
        p_ref, _ = peak(False)
    except torch.OutOfMemoryError:
        p_ref = float("inf")
    torch.cuda.empty_cache()
    p1, g1 = peak(True)
    p2, g2 = peak(True)
    print(f"peak MiB: weight-map path {p_ref / 2**20:.0f}, fused {p1 / 2**20:.0f}")
    assert p1 < p_ref
    assert torch.equal(g1, g2)
