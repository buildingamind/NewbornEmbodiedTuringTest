"""CompactViT token hook: the readout the wave-15 comparator runs must not move by one bit.

⛔ THE CLAIM UNDER TEST IS BITWISE IDENTITY WITH THE PRE-HOOK CODE, NOT "CLOSE". Wave 15's live
ViT arms are the only comparator wave 17 has (notes/researcher/wave17-object-centric.md,
"Comparator protection"). Adding `encode_tokens` refactored `forward` onto a shared `_trunk`, so
this suite builds the encoder from a FROZEN COPY of compact_vit.py at commit 2141520 -- real code
that runs, not a remembered number -- loads the same weights into both, and asserts
`torch.equal` on the output AND on every parameter's gradient, for every wave-15 ViT config.

⚠ The frozen copy is a committed fixture (tests/fixtures/frozen/*.py.txt), not a `git show` at
test time, so the suite needs no git object store. Its provenance is pinned by recomputing the
git BLOB hash of the fixture bytes and comparing it to the blob id `git rev-parse
2141520:src_isaac/nett_skrl/brain/encoders/compact_vit.py` printed when the fixture was cut.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "examples"))

from campaign_train import (  # noqa: E402
    VIT_CFG, VIT_CLTT_CONV_CFG, VIT_CLTT_D6_CFG, VIT_CLTT_DVS_CFG, VIT_CLTT_H8_CFG,
    VIT_CLTT_MLP4_CFG, VIT_CLTT_P8_CFG, VIT_CLTT_SP_CFG,
)
from nett_skrl.brain.encoders.compact_vit import CompactViT  # noqa: E402

FIXTURE = _SRC / "tests" / "fixtures" / "frozen" / "compact_vit_2141520.py.txt"
#: `git rev-parse 2141520:src_isaac/nett_skrl/brain/encoders/compact_vit.py`, copied from the
#: command's output when the fixture was cut (never typed).
FIXTURE_BLOB = "d803d1285e2209b0205976b64ca198f725775b8f"

H, W = 80, 128

# (cfg, input channels). ViT-CLTT-Ref runs framestack=True -> 6 channels; the DVS row is 2
# channels per frame -> 4. The attn_mode rows are the QK-ablation arms that share this file.
CASES = {
    "ViT-CLTT-Ref": (VIT_CFG, 6),
    "H8": (VIT_CLTT_H8_CFG, 6),
    "P8": (VIT_CLTT_P8_CFG, 6),
    "Sp": (VIT_CLTT_SP_CFG, 6),
    "Conv": (VIT_CLTT_CONV_CFG, 6),
    "D6": (VIT_CLTT_D6_CFG, 6),
    "MLP4": (VIT_CLTT_MLP4_CFG, 6),
    "DVS": (VIT_CLTT_DVS_CFG, 4),
    "mixer": ({**VIT_CFG, "attn_mode": "mixer"}, 6),
    "uniform-spatial": ({**VIT_CFG, "attn_mode": "uniform", "pool": "spatial",
                         "spatial_grid": (5, 4)}, 3),
}


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _load_frozen():
    name = "nett_skrl.brain.encoders._frozen_compact_vit_2141520"
    if name in sys.modules:
        return sys.modules[name]
    # A package-qualified name makes the file's RELATIVE imports resolve against the live
    # package; an explicit loader is needed because the fixture is not a .py file.
    loader = importlib.machinery.SourceFileLoader(name, str(FIXTURE))
    spec = importlib.util.spec_from_file_location(name, str(FIXTURE), loader=loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(threads)


def _space(c):
    return gym.spaces.Box(0, 255, shape=(H, W, c), dtype=np.uint8)


def _pair(cfg, c, seed=0):
    torch.manual_seed(seed)
    old = _load_frozen().CompactViT(_space(c), **cfg)
    torch.manual_seed(seed + 1)          # different init, then overwritten: equality is by weights
    new = CompactViT(_space(c), **cfg)
    assert list(old.state_dict()) == list(new.state_dict()), "state_dict keys moved"
    new.load_state_dict(old.state_dict())
    return old, new


def _obs(c, b=3, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (b, H, W, c), generator=g, dtype=torch.uint8)


def test_fixture_is_the_pre_hook_file():
    assert _git_blob_sha1(FIXTURE.read_bytes()) == FIXTURE_BLOB
    # Inherited base-class stubs do not count; the frozen class itself must predate the hook.
    assert "encode_tokens" not in vars(_load_frozen().CompactViT)
    assert "_trunk" not in vars(_load_frozen().CompactViT)


@pytest.mark.parametrize("case", sorted(CASES))
def test_forward_and_gradients_bitwise_identical_to_pre_hook_code(case):
    cfg, c = CASES[case]
    old, new = _pair(cfg, c)
    x = _obs(c)
    out_old, out_new = old(x), new(x)
    assert torch.equal(out_old, out_new), case
    # A non-uniform upstream gradient, so a permuted or summed-differently path cannot pass.
    w = torch.linspace(-1.0, 1.0, out_old.numel()).view_as(out_old)
    (out_old * w).sum().backward()
    (out_new * w).sum().backward()
    g_old = dict(old.named_parameters())
    for n, p in new.named_parameters():
        assert (p.grad is None) == (g_old[n].grad is None), n
        if p.grad is not None:
            assert torch.equal(p.grad, g_old[n].grad), f"{case}: grad differs at {n}"


@pytest.mark.parametrize("case", ["ViT-CLTT-Ref", "Sp", "P8", "DVS"])
def test_tokens_are_the_tensor_the_readout_consumes(case):
    """The hook must return what `forward` reads: CLS readout -> head(norm(x)[:,0]) is not
    reachable from patch tokens, so the SPATIAL readout is recomputed from the hook's tokens and
    compared bitwise; for CLS configs the (1+N) trunk's tail is compared instead."""
    cfg, c = CASES[case]
    _, enc = _pair(cfg, c)
    x = _obs(c)
    tokens, (n_h, n_w) = enc.encode_tokens(x)
    patch = cfg["patch_size"]
    assert (n_h, n_w) == (H // patch, W // patch)
    assert tokens.shape == (x.shape[0], n_h * n_w, cfg["embed_dim"])
    trunk = enc._trunk(enc._prepare_image(x))
    assert torch.equal(tokens, trunk[:, 1:])
    if enc.pool == "spatial":
        B = x.shape[0]
        tok = enc.token_reduce(tokens)
        grid = tok.transpose(1, 2).reshape(B, tok.shape[-1], n_h, n_w)
        assert torch.equal(enc.head(enc.spatial_pool(grid).flatten(1)), enc(x))
    else:
        assert torch.equal(enc.head(trunk[:, 0]), enc(x))


def test_tokens_are_post_final_layernorm():
    cfg, c = CASES["ViT-CLTT-Ref"]
    _, enc = _pair(cfg, c)
    with torch.no_grad():
        enc.norm.weight.fill_(3.0)
        enc.norm.bias.fill_(-2.0)
    tokens, _ = enc.encode_tokens(_obs(c))
    # Post-norm tokens carry the norm's affine exactly: per-token mean == bias, std == |gain|.
    assert torch.allclose(tokens.mean(-1), torch.full(tokens.shape[:2], -2.0), atol=1e-4)
    assert torch.allclose(tokens.std(-1, unbiased=False), torch.full(tokens.shape[:2], 3.0),
                          atol=1e-3)


def test_tokens_backpropagate_into_the_trunk_not_the_readout():
    cfg, c = CASES["Sp"]
    _, enc = _pair(cfg, c)
    tokens, _ = enc.encode_tokens(_obs(c))
    tokens.pow(2).mean().backward()
    for n, p in enc.named_parameters():
        readout = n.startswith(("head.", "token_reduce."))
        if readout:
            assert p.grad is None, n
        else:
            assert p.grad is not None and p.grad.abs().sum() > 0, n
