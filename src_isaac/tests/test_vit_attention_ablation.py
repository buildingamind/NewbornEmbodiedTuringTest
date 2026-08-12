"""The QK ablation: is a ViT's advantage CONTENT-DEPENDENT routing, or just global mixing?

WHY THIS EXISTS. At n=56 per arm, ViT beats CNN on binding by 0.147 (p = 5e-5) with 12/56
binders against 1/56 -- while being WORSE at the positive control and at colour
(SIDE_LOCK_INVESTIGATION.md Phase 9). The leading explanation is the QK self-attention:
routing between spatial locations whose weights depend on what is at those locations, which
is what binding a colour to a shape at a place needs and what a fixed convolution kernel
cannot do.

"Remove the attention" cannot test that -- it removes the global receptive field, the value
pathway and the output projection at the same time. These ablations remove ONLY the
content-dependence:

    qk       weights = softmax(QK^T/sqrt(d))   depend on the input
    uniform  weights = 1/N                     fixed
    mixer    weights = softmax(M), M learned    learned, but blind to the input

The tests below pin the three properties that make that claim honest: the mixing really is
input-independent, the rest of the block is untouched, and the parameter count is MATCHED
(an ablation that also deletes a third of the weights proves nothing).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.encoders.compact_vit import (
    CompactViT, _TokenMixer)

# The campaign's ViT arm (examples/campaign_train.py::VIT_CFG) at res 128.
QK_CFG = dict(features_dim=512, patch_size=16, embed_dim=144, depth=3, num_heads=4,
              mlp_ratio=2.0, pool="cls", stem="linear")
# embed_dim raised to hold total parameters at the QK arm's ~697K (see the param test).
ABLATION_DIM = {"uniform": 164, "mixer": 160}
# ★ The CLS-pooled ablations above are NOT valid controls: with one readout token, uniform or
# static mixing makes it a global average, so a sub-patch object is diluted ~64x. Measured on
# examples/probe_encoder_binding.py: both fall to EXACT chance (0.497) at a half-width-8
# stimulus while qk and CNN score 1.000. pool="spatial" restores per-patch readout and both
# recover to 1.000. These are the dims of the valid arms (matched to the ViT arm's 697,184).
SPATIAL = {"pool": "spatial", "spatial_grid": 4, "spatial_reduce_dim": 16}
SPATIAL_DIM = {"qk": 136, "uniform": 156, "mixer": 152}


def obs_space(res: int = 128) -> gym.spaces.Box:
    return gym.spaces.Box(low=0, high=255, shape=(res, res, 3), dtype=np.uint8)


def build(mode: str, **over) -> CompactViT:
    cfg = dict(QK_CFG)
    if mode != "qk":
        cfg["embed_dim"] = ABLATION_DIM[mode]
    cfg.update(over)
    torch.manual_seed(0)
    return CompactViT(obs_space(), attn_mode=mode, **cfg)


@pytest.mark.parametrize("mode", ["qk", "uniform", "mixer"])
def test_encoder_runs_and_returns_features_dim(mode):
    enc = build(mode).eval()
    x = torch.randint(0, 255, (4, 128, 128, 3), dtype=torch.uint8)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (4, QK_CFG["features_dim"])
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("mode", ["qk", "uniform", "mixer"])
def test_encoder_is_deterministic(mode):
    """A non-deterministic encoder would silently break every replay guarantee."""
    enc = build(mode).eval()
    x = torch.randint(0, 255, (2, 128, 128, 3), dtype=torch.uint8)
    with torch.no_grad():
        assert torch.equal(enc(x), enc(x))


def test_uniform_mixing_is_the_same_for_every_output_token():
    """`uniform` must average ALL tokens identically -- the defining property.

    If the mixer produced different rows per position it would be routing, which is the
    thing being ablated away.
    """
    torch.manual_seed(0)
    mix = _TokenMixer(dim=16, num_tokens=5, mode="uniform").eval()
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        out = mix(x)
    for i in range(1, 5):
        assert torch.allclose(out[:, 0], out[:, i], atol=1e-6)


def test_mixer_weights_do_not_depend_on_the_input():
    """The `mixer` control: it may learn WHERE to look, never based on WHAT is there.

    Scaling one token's content changes the mixed output only through that token's VALUE,
    never through the weights -- so the map stays linear in the inputs. A content-dependent
    operator (softmax(QK^T)) fails this: doubling a token changes its own attention weights.
    """
    torch.manual_seed(0)
    mix = _TokenMixer(dim=8, num_tokens=4, mode="mixer").eval()
    a, b = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    with torch.no_grad():
        # Linear (affine) in the input <=> weights are input-independent.
        lhs = mix(a + b)
        rhs = mix(a) + mix(b) - mix(torch.zeros_like(a))
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_qk_attention_is_content_dependent():
    """The positive control for the test above: the real block must FAIL linearity."""
    from nett_skrl.brain.encoders.compact_vit import _TransformerBlock
    torch.manual_seed(0)
    blk = _TransformerBlock(dim=16, num_heads=4, attn_mode="qk").eval()
    a, b = torch.randn(1, 6, 16), torch.randn(1, 6, 16)
    with torch.no_grad():
        lhs = blk(a + b)
        rhs = blk(a) + blk(b) - blk(torch.zeros_like(a))
    assert not torch.allclose(lhs, rhs, atol=1e-3), (
        "QK attention came out linear in its input -- it is not routing on content, and the "
        "ablation would have nothing to ablate")


@pytest.mark.parametrize("mode", ["uniform", "mixer"])
def test_ablation_parameter_count_is_matched(mode):
    """★ THE ABLATION MUST NOT ALSO BE A CAPACITY CUT.

    Dropping Q and K frees 2*dim^2 per block; `embed_dim` is raised to spend it back. Within
    2% of the QK arm, or the encoder contrast is confounded by size -- the same trap the
    9-model sweep avoided by matching every model to ~700K.
    """
    qk = sum(p.numel() for p in build("qk").parameters())
    ab = sum(p.numel() for p in build(mode).parameters())
    assert abs(ab - qk) / qk < 0.02, f"{mode}: {ab:,} vs qk {qk:,} ({100*(ab-qk)/qk:+.2f}%)"


def test_the_ablations_keep_the_global_receptive_field():
    """What is ablated is the ROUTING, not the reach.

    Every output token must still depend on a distant input token; if the ablation localised
    the encoder it would be testing locality, which is the CNN's property, not attention's.
    """
    for mode in ("uniform", "mixer"):
        enc = build(mode, depth=1).eval()
        x = torch.zeros(1, 128, 128, 3, dtype=torch.uint8)
        y = x.clone()
        y[0, -16:, -16:, :] = 255          # perturb the LAST patch only
        with torch.no_grad():
            assert not torch.allclose(enc(x), enc(y), atol=1e-6), (
                f"{mode}: the CLS output ignored a change in the far corner patch")


@pytest.mark.parametrize("mode", ["uniform", "mixer"])
def test_gradients_reach_the_mixing_pathway(mode):
    enc = build(mode, depth=1)
    out = enc(torch.randint(0, 255, (2, 128, 128, 3), dtype=torch.uint8))
    out.sum().backward()
    mixer = enc.blocks[0].attn
    assert mixer.v.weight.grad is not None and mixer.v.weight.grad.abs().sum() > 0
    if mode == "mixer":
        assert mixer.mix.grad is not None and mixer.mix.grad.abs().sum() > 0


def test_ablated_branch_starts_at_the_same_scale_as_qk():
    """★ THE TEST THAT WOULD HAVE SAVED 16 GPU-HOURS (added after the v1 arms failed).

    ``CompactViT._init_weights`` applies ``trunc_normal(std=0.02)`` to every ``nn.Linear``,
    but ``nn.MultiheadAttention`` initialises its own projections with ``xavier_uniform`` and
    is not reached by that pass. So the v1 ablation ALSO changed the initialisation: the
    mixing branch emitted 0.008 RMS against QK's 0.026, a 3.2x gap. Both v1 arms then failed
    their positive control (`rest` 0.54 / 0.53, ~25 of 28 agents below 0.75) while a CNN with
    no attention at all scores `rest` 0.9996 -- proof the ablation broke something unrelated
    to routing.

    An ablation must differ from its control in ONE thing. Scale is not that thing.
    """
    torch.manual_seed(0)
    x = torch.randn(8, 65, 160)

    def branch_rms(mode: str) -> float:
        enc = build("qk" if mode == "qk" else mode, embed_dim=160, depth=1).eval()
        blk = enc.blocks[0]
        with torch.no_grad():
            n = blk.norm1(x)
            out = blk.attn(n, n, n)[0] if mode == "qk" else blk.attn(n)
        return float(out.pow(2).mean().sqrt())

    qk = branch_rms("qk")
    for mode in ("uniform", "mixer"):
        r = branch_rms(mode) / qk
        # A residual gap is intrinsic: a diagonal-dominant mixer keeps per-token variance
        # that an averaging operator destroys, so an EXACT match would mean the operator had
        # not changed. 1.5x is the tolerance the eye*2.0 init was chosen against; the 3.2x
        # and 4.6x seen while fixing this were not.
        assert 0.5 < r < 1.6, f"{mode}: init output scale is {r:.2f}x the QK branch"


def test_mixer_does_not_start_at_the_uniform_degenerate_point():
    """`mix` initialised to zeros is softmax-uniform, i.e. EXACTLY the `uniform` lesion.

    v1 did that, so every mixer began at the degenerate all-tokens-identical solution and had
    to break a perfectly symmetric point to escape. It did not, in 2000 episodes. The init is
    now diagonal-dominant: each token mostly keeps itself, which is benign and asymmetric.
    """
    torch.manual_seed(0)
    n_tokens = 65                       # the real token count at 128px / patch 16
    mix = _TokenMixer(dim=8, num_tokens=n_tokens, mode="mixer").eval()
    w = torch.softmax(mix.mix, dim=-1)
    uniform = 1.0 / n_tokens
    assert w.diagonal().mean() > 4 * uniform, (
        f"mixer starts at diagonal {w.diagonal().mean():.4f} vs uniform {uniform:.4f} -- "
        "too close to the degenerate all-tokens-identical point that broke the v1 arms")
    x = torch.randn(1, n_tokens, 8)
    with torch.no_grad():
        out = mix(x)
    assert not torch.allclose(out[:, 0], out[:, 1], atol=1e-3), (
        "mixer output tokens are identical at init -- it began at the `uniform` lesion")


@pytest.mark.parametrize("mode", ["qk", "uniform", "mixer"])
def test_spatial_pooled_ablation_arms_are_parameter_matched(mode):
    """The valid arms must be matched too -- pooling changes the head size."""
    cfg = {**QK_CFG, **SPATIAL, "embed_dim": SPATIAL_DIM[mode]}
    if mode != "qk":
        cfg["attn_mode"] = mode
    torch.manual_seed(0)
    enc = CompactViT(obs_space(), **cfg)
    qk_params = sum(p.numel() for p in build("qk").parameters())
    n = sum(p.numel() for p in enc.parameters())
    assert abs(n - qk_params) / qk_params < 0.02, (
        f"{mode}+spatial: {n:,} vs cls-qk {qk_params:,} ({100*(n-qk_params)/qk_params:+.2f}%)")


# ⚠ NO UNIT TEST ASSERTS *WHY* SPATIAL POOLING FIXES THE ABLATION. Two candidate mechanisms
# were measured at initialisation and NEITHER separates the variants cleanly: position-
# specificity (spatial 0.99 vs cls 0.87 -- both position-sensitive, because pos_embed is
# injected before the blocks) and small-vs-large object response (spatial-uniform 0.39 vs
# cls-uniform 0.26, but qk sits in between at 0.31/0.32). The dilution story is a plausible
# reading, not a verified one, and an untrained forward pass may say nothing about
# trainability anyway. The EVIDENCE for the fix is the measured probe --
# campaign/logs/probe/encoder_capacity_spatial.txt: both ablations 0.497 -> 1.000 at a
# half-width-8 stimulus. Do not encode the unverified mechanism as an assertion.


def test_spatial_pool_is_deterministic_and_matches_adaptive_pooling():
    """★ THE BUG THAT KILLED THE FIRST SPATIAL ARMS, 8 processes, at the first PPO update.

    ``F.adaptive_avg_pool2d``'s CUDA backward has no deterministic implementation, so under
    this project's ``torch.use_deterministic_algorithms(True)`` it raises:

        RuntimeError: adaptive_avg_pool2d_backward_cuda does not have a deterministic
        implementation ...

    On an evenly-divisible grid adaptive pooling IS uniform average pooling, so reshape-and-
    mean gives the identical value with a deterministic backward. This test pins both halves:
    the values match, and the encoder no longer references the non-deterministic op.

    ⚠ ADAPTED ON THE PORT INTO THIS TREE (2026-08-12). isaac1 carried a private
    ``_deterministic_grid_pool`` helper; this tree already ships
    ``encoders/utils/pool.DeterministicAvgPool2d`` and routes eight other encoders through
    it, so CompactViT uses that instead of gaining a second implementation of one idea.
    The property under test is unchanged.
    """
    import torch.nn.functional as F
    from nett_skrl.brain.encoders.utils.pool import DeterministicAvgPool2d
    torch.manual_seed(0)
    x = torch.randn(3, 5, 8, 8)
    assert torch.allclose(F.adaptive_avg_pool2d(x, (4, 4)), DeterministicAvgPool2d(4)(x),
                          atol=1e-6)

    src = (Path(__file__).resolve().parents[1] / "nett_skrl" / "brain" / "encoders"
           / "compact_vit.py").read_text()
    forward = src[src.index("def forward(self, observations"):]
    assert "adaptive_avg_pool2d" not in forward, (
        "CompactViT.forward must not call adaptive_avg_pool2d -- its CUDA backward is "
        "non-deterministic and the PPO update runs under strict determinism")


def test_ragged_spatial_grid_is_rejected_rather_than_silently_non_deterministic():
    """A grid that does not divide evenly would need the non-deterministic op -- say so.

    ⚠ Now checked at CONSTRUCTION of CompactViT, not only inside the pool. That matters
    here: the default eye is 128x80, which at patch 16 gives a 5x8 token grid that
    ``spatial_grid=4`` does not divide -- so the spatial path is unusable at the default
    resolution and must say so immediately rather than at the first forward. The 2026-08
    spatial arms ran at a SQUARE 128x128 (8x8 -> 4x4).
    """
    from nett_skrl.brain.encoders.utils.pool import DeterministicAvgPool2d
    with pytest.raises(ValueError, match="divisible"):
        DeterministicAvgPool2d(3)(torch.randn(1, 2, 8, 8))

    space = gym.spaces.Box(low=0, high=255, shape=(80, 128, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not divide"):
        CompactViT(space, features_dim=512, patch_size=16, embed_dim=144, depth=3,
                   pool="spatial", spatial_grid=4)


def test_a_rectangular_spatial_grid_works_at_the_real_non_square_eye():
    """★ THE EYE IS 128x80 AND STAYING THAT WAY, so a square grid is not an option.

    5x8 tokens have no common square divisor but 1 -- which is a global average, i.e. exactly
    the CLS collapse the spatial readout exists to avoid. A RECTANGULAR grid keeps the layout.
    Without this, `pool="spatial"` would be a branch that can only ever raise.
    """
    space = gym.spaces.Box(low=0, high=255, shape=(80, 128, 3), dtype=np.uint8)
    enc = CompactViT(space, features_dim=512, patch_size=16, embed_dim=144, depth=3,
                     num_heads=4, pool="spatial", spatial_grid=(5, 4),
                     spatial_reduce_dim=16).eval()
    with torch.no_grad():
        out = enc(torch.randint(0, 255, (2, 80, 128, 3), dtype=torch.uint8))
    assert out.shape == (2, 512) and torch.isfinite(out).all()


@pytest.mark.parametrize("arm", ["ViT-Sp", "ViT-Mixer-Sp", "ViT-NoQK-Sp"])
def test_square_eye_archive_arms_still_rebuild_for_offline_reanalysis(arm):
    """★ THE n=56 CHECKPOINTS ARE ON DISK AND MUST STAY LOADABLE.

    These three arms cannot be TRAINED at the 128x80 eye and are not meant to be -- but
    examples/probe_frozen_features.py rebuilds each arm's encoder through ``MODELS[...]`` at a
    SQUARE RES=128 to load the saved weights, and that probe is how Phases 14/15 were produced.
    Re-asking the QK question with the per-agent coupling as the endpoint (p ~ 0.006 on the n
    already on disk) runs through this path, so deleting these configs would have closed the
    cheapest open experiment in the programme.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    from campaign_train import MODELS
    cfg = {k: v for k, v in MODELS[arm]["cfg"].items() if k != "trainable"}
    square = gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
    enc = CompactViT(square, **cfg)
    n = sum(p.numel() for p in enc.parameters())
    assert abs(n - 697_184) / 697_184 < 0.02, f"{arm}: {n:,} drifted from the ~697K match"


def test_an_unknown_attn_mode_is_rejected_loudly():
    """A typo must not silently fall back to a mode -- the arm's identity is the result."""
    with pytest.raises(ValueError, match="uniform.*mixer|mixer.*uniform"):
        CompactViT(obs_space(), attn_mode="qk_typo", **QK_CFG)
