"""The vendored Unity ViT: is it still the upstream model, and does it run on the Isaac body?

⚠ WHAT THIS SUITE IS FOR. `unity_vit.py` carries a VERBATIM copy of `vit_pytorch`'s SimpleViT.
Neither `vit_pytorch` nor `lightning` is installed here, so no test can diff the copy against the
real thing at run time -- the equivalence was established once, out of band, against
`vit_pytorch==1.26.6` (identical parameter names/shapes, identical init from a fixed seed,
identical position embedding, and bitwise-identical output). What these tests defend is the copy
STAYING that model: every structural quantity the check turned on is pinned independently below,
so an edit inside the verbatim block moves at least one of them.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.encoders.unity_vit import SimpleViT, UnityViT, pair, posemb_sincos_2d

H, W = 80, 128


def space(c=6, hwc=True):
    shape = (H, W, c) if hwc else (c, H, W)
    return gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- the upstream defaults

def test_vit_py_defaults_are_reproduced():
    # ⛔ These are `src/nett/brain/encoders/vit.py`'s OWN signature defaults on repo A's dev
    # branch. If the upstream file changes, this arm stops being "the Unity default" and the
    # proposal's why/falsifier stop describing it.
    import inspect
    sig = inspect.signature(UnityViT.__init__)
    got = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect._empty}
    assert got == {"features_dim": 512, "patch_size": 4, "depth": 3, "heads": 3,
                   "intermediate_size": 128, "hidden_size": 64}


def test_token_count_is_640_on_the_isaac_eye():
    # ⚠ THE COST LINE. patch 4 on 80x128 is 20x32 tokens, against 40 for the fleet's patch-16
    # standard; attention is quadratic, so this is ~256x the attention work at 1/3 the params.
    m = UnityViT(space())
    assert m.model.pos_embedding.shape == (640, 64)
    assert (H // 4) * (W // 4) == 640


def test_parameter_count_is_pinned():
    # Establishes what "capacity-matched" would have to mean if anyone tried; row 7 deliberately
    # is NOT matched, because matching it would stop it being the default.
    assert n_params(UnityViT(space(6))) == 237_888
    assert n_params(UnityViT(space(3))) == 234_720


def test_output_width_is_features_dim():
    m = UnityViT(space(), features_dim=512)
    out = m(torch.randint(0, 255, (2, H, W, 6), dtype=torch.uint8))
    assert out.shape == (2, 512)
    assert m.features_dim == 512


# ---------------------------------------------------------------- the preserved quirks

def test_there_is_no_fc_attribute():
    # `vit.py:52` sets `self.model.fc = nn.Identity()` on the LightningModule, which has no `fc`
    # and never reads one -- a no-op. Reproduced by simply not having one; the SimpleViT's
    # `linear_head` is what produces the output.
    m = UnityViT(space())
    assert not hasattr(m.model, "fc")
    assert isinstance(m.model.linear_head, torch.nn.Linear)


def test_no_xavier_init_is_applied():
    # ⛔ `init_weights()` is defined TWICE in vit_contrastive.py and called NOWHERE, so every
    # weight keeps SimpleViT's default init. Calling it would be the DEVIATION, not the fix.
    torch.manual_seed(0)
    m = UnityViT(space())
    torch.manual_seed(0)
    ref = SimpleViT(image_size=(H, W), patch_size=4, num_classes=512, dim=64, depth=3,
                    heads=3, mlp_dim=128, channels=6)
    for (ka, a), (kb, b) in zip(m.model.named_parameters(), ref.named_parameters()):
        assert ka == kb and torch.equal(a, b)
    # Xavier on the head would give a bounded uniform; the default is kaiming-uniform on
    # fan_in=dim=64. The head bias under Xavier's `_init` would be ~N(0, 1e-6).
    assert m.model.linear_head.bias.abs().max() > 1e-4


def test_no_projection_head_is_carried():
    # ⛔ DEVIATION 1: LitClassifier's 512->512->128 SimCLR Projection is omitted. It is never
    # read by forward(), but under SB3 its parameters entered the optimiser as dead weight.
    # Linear(512,512)+bias, BatchNorm1d(512) affine, Linear(512,128,bias=False).
    # ⚠ The BatchNorm's 1,024 affine parameters count; its running stats are buffers and do not.
    dead = (512 * 512 + 512) + (512 + 512) + (512 * 128)
    assert dead == 329_216
    assert n_params(UnityViT(space())) == 237_888        # i.e. the dead weight is NOT here
    assert not any("projection" in k for k, _ in UnityViT(space()).named_parameters())


def test_pos_embedding_is_not_a_parameter_or_buffer():
    # Upstream assigns it as a plain attribute: absent from state_dict(), unmoved by .to().
    m = UnityViT(space())
    assert "model.pos_embedding" not in dict(m.named_parameters())
    assert "model.pos_embedding" not in dict(m.named_buffers())
    assert "model.pos_embedding" not in m.state_dict()


def test_pooling_is_mean_over_patches_not_a_cls_token():
    m = UnityViT(space())
    assert m.model.pool == "mean"
    # No CLS token exists to prepend, so the token count is exactly the patch count.
    assert m.model.pos_embedding.shape[0] == 640


# ---------------------------------------------------------------- the forced deviation

def test_rectangular_eye_is_built_as_h_by_w():
    # ⛔ DEVIATION 2: vit.py passes the SCALAR observation_space.shape[1]. On an 80x128 eye a
    # scalar is unbuildable, not merely unfaithful -- 80 would assert against a 128-wide input.
    m = UnityViT(space())
    assert m.model.to_patch_embedding[0] is not None
    m(torch.randint(0, 255, (1, H, W, 6), dtype=torch.uint8))     # would raise if square


def test_square_scalar_would_have_been_wrong():
    with pytest.raises(Exception):
        SimpleViT(image_size=H, patch_size=4, num_classes=8, dim=64, depth=1, heads=2,
                  mlp_dim=16, channels=6)(torch.zeros(1, 6, H, W))


def test_indivisible_patch_size_is_refused_with_a_useful_message():
    with pytest.raises(ValueError, match="not divisible by patch_size"):
        UnityViT(space(), patch_size=7)


# ---------------------------------------------------------------- the verbatim block itself

def test_pair_expands_scalars_and_passes_tuples():
    assert pair(4) == (4, 4) and pair((5, 4)) == (5, 4)


def test_sincos_matches_an_independent_derivation():
    # ⛔ Recomputed from the formula rather than from the function, so an edit inside the
    # verbatim block cannot make the expectation agree with itself.
    h, w, dim, temp = 3, 5, 8, 10000
    got = posemb_sincos_2d(h, w, dim, temperature=temp)
    assert got.shape == (h * w, dim)
    q = dim // 4
    omega = [1.0 / (temp ** (i / (q - 1))) for i in range(q)]
    for yy in range(h):
        for xx in range(w):
            row = [math.sin(xx * o) for o in omega] + [math.cos(xx * o) for o in omega] \
                + [math.sin(yy * o) for o in omega] + [math.cos(yy * o) for o in omega]
            assert torch.allclose(got[yy * w + xx], torch.tensor(row, dtype=torch.float32),
                                  atol=1e-6)


def test_sincos_requires_a_multiple_of_four():
    with pytest.raises(AssertionError, match="multiple of 4"):
        posemb_sincos_2d(2, 2, 6)


def test_attention_inner_dim_is_heads_times_dim_head():
    # dim=64 is NOT divisible by heads=3; SimpleViT projects to heads*dim_head=192 regardless.
    m = UnityViT(space())
    attn = m.model.transformer.layers[0][0]
    assert attn.to_qkv.weight.shape == (192 * 3, 64)
    assert attn.to_qkv.bias is None and attn.to_out.bias is None
    assert attn.scale == pytest.approx(64 ** -0.5)


def test_blocks_are_prenorm_residual():
    m = UnityViT(space())
    attn, ff = m.model.transformer.layers[0]
    assert isinstance(attn.norm, torch.nn.LayerNorm)
    assert isinstance(ff.net[0], torch.nn.LayerNorm)
    assert isinstance(m.model.transformer.norm, torch.nn.LayerNorm)


def test_depth_controls_the_number_of_blocks():
    assert len(UnityViT(space(), depth=5).model.transformer.layers) == 5


# ---------------------------------------------------------------- body integration

def test_registry_resolves_the_name():
    from nett_skrl.brain.registry import encoder_mapping, encoders_list, validate_encoder
    assert encoder_mapping["unity_vit"] is UnityViT
    assert validate_encoder("unity_vit") is UnityViT
    assert "unity_vit" in encoders_list


def test_chw_and_hwc_observation_spaces_both_work():
    for sp, x in ((space(6, hwc=True), torch.randint(0, 255, (2, H, W, 6), dtype=torch.uint8)),
                  (space(6, hwc=False), torch.randint(0, 255, (2, 6, H, W), dtype=torch.uint8))):
        assert UnityViT(sp)(x).shape == (2, 512)


def test_encode_prepared_bypasses_preparation():
    m = UnityViT(space())
    prepared = torch.rand(2, 6, H, W)
    assert m.encode_prepared(prepared).shape == (2, 512)
    assert m._skip_prepare is False              # restored afterwards


def test_gradients_reach_the_patch_embedding():
    m = UnityViT(space())
    m(torch.randint(0, 255, (2, H, W, 6), dtype=torch.uint8)).sum().backward()
    assert m.model.to_patch_embedding[2].weight.grad.abs().sum() > 0
