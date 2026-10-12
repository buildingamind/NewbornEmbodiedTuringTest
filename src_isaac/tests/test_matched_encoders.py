"""The owner's six MATCHED encoders and their head (workspace DECISIONS §92; encoders/matched.py; spec
docs/specs/matched_architectures_owner_2026-10-12.md). Built on CPU at the spec's 160 x 256 eye.

Pinned against KNOWN ANSWERS, each with a control that the assertion can fail:
  1. ENCODER parameter counts == the spec's table, exactly, through the campaign labels' own cfg.
  2. The attention mixer == the spec's explicit softmax(q k^T / sqrt(32)) v code, and a WRONG scale
     (1/sqrt(96)) is not.
  3. FRAME ORDER AND SHARING: CNN2F / ViT2F never mix frames before LayerNorm(128), so feeding
     [cur, prev] must give exactly the [prev | cur] halves swapped (true at init, where that
     LayerNorm's affine is 1/0). 3DCNN / ViViT mix frames (and carry temporal positions), so the
     same swap must NOT commute -- the control that the identity is not vacuous.
  4. uint8 is divided by 255 ONCE: forward(uint8) == encode_prepared(float / 255).
  5. Refusals: wrong eye, wrong channel count (both directions, strict 3-per-frame), features_dim
     != 128, unknown variant, an unread cfg key.
  6. INIT: every conv/linear weight and bias inside +-1/sqrt(fan_in) on the encoder AND the head
     (built through the real builder with the label's head cfg); positions std ~ .02; LayerNorm 1/0.
     Control: the fleet's orthogonal init at sqrt(2) breaks the bound on the same layers.
  7. HEAD: separate tanh 128-128 actor and critic trunks over ONE shared encoder.
  8. cltt_ref through a matched encoder: finite loss, nonzero gradient to EVERY encoder parameter,
     single-frame (3 channels, framestack=False) and two-frame.
  9. Existing labels untouched: the new labels are additive, and the six M- names and their
     CLTT-Ref twins differ ONLY in the aux.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "examples"))

import campaign_train as ct  # noqa: E402
from nett_skrl.brain.config import EncoderCfg  # noqa: E402
from nett_skrl.brain.encoders.matched import MatchedEncoder, VARIANTS, _Attention  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402

H, W = 160, 256
SPEC_COUNTS = {"M-CNN": 644_640, "M-ViT": 507_264, "M-CNN2F": 634_336, "M-ViT2F": 496_960,
               "M-3DCNN": 1_132_192, "M-ViViT": 497_152, "M-3DCNN-G6": 510_112}


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    for k in ("NETT_AUX_BATCH", "NETT_AUX_CLTT_REF_OFFSETS", "NETT_AUX_CLTT_REF_TEMP", "NETT_DECOUPLE_ENCODER"):
        monkeypatch.delenv(k, raising=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        yield


def _space(frames, h=H, w=W):
    return gym.spaces.Box(0, 255, (3 * frames, h, w), np.uint8)


def _frames(label):
    return 2 if ct.MODELS[label]["framestack"] else 1


def _from_label(label, h=H, w=W):
    spec = ct.MODELS[label]
    kw = EncoderCfg(**spec["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    return encoder_mapping[spec["encoder"]](_space(_frames(label), h, w), **kw)


def _n(m):
    return sum(p.numel() for p in m.parameters())


# ------------------------------------------------------------------ 1. counts
@pytest.mark.parametrize("label", sorted(SPEC_COUNTS))
def test_encoder_count_is_the_spec_table(label):
    assert _n(_from_label(label)) == SPEC_COUNTS[label]
    assert _n(_from_label(label + "-CLTT-Ref")) == SPEC_COUNTS[label]


def test_counts_follow_the_closed_form():
    """Spec arithmetic, term by term, so a count that matched by accident would show here."""
    patch, pos, ln96 = 3 * 96 * 256 + 96, 160 * 96, 2 * 96
    ffn = 96 * 384 + 384 + 384 * 96 + 96
    mixer = {"conv2d": 96 * 96 * 9 + 96, "conv3d": 96 * 96 * 27 + 96, "conv3d_g6": 96 * 16 * 27 + 96,
             "attn": 96 * 288 + 288 + 96 * 96 + 96}
    for v, (T, kind, _, temporal) in VARIANTS.items():
        d = 128 // T
        readout = 96 * 32 * 16 + 32 + 32 * 32 * 12 + 32 + 160 * d + d
        want = patch + pos + (T * 96 if temporal else 0) + 3 * (2 * ln96 + mixer[kind] + ffn) + ln96 + readout + 256
        assert _n(MatchedEncoder(_space(T), variant=v)) == want, v


# ------------------------------------------------------------------ 2. attention == spec code
def _spec_attention(mod, tokens):
    M, N, _ = tokens.shape
    qkv = mod.qkv_proj(tokens).reshape(M, N, 3, 3, 32)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    a = ((q @ k.transpose(-2, -1)) / math.sqrt(32)).softmax(dim=-1)
    return mod.out_proj((a @ v).transpose(1, 2).reshape(M, N, 96)), (q, k, v)


def test_attention_is_the_spec_code_and_a_wrong_scale_is_not():
    mod, x = _Attention(), torch.randn(3, 20, 96)
    ref, (q, k, v) = _spec_attention(mod, x)
    torch.testing.assert_close(mod(x), ref, rtol=1e-5, atol=1e-5)
    wrong = mod.out_proj((((q @ k.transpose(-2, -1)) / math.sqrt(96)).softmax(-1) @ v).transpose(1, 2).reshape(3, 20, 96))
    assert not torch.allclose(wrong, ref, atol=1e-4)


# ------------------------------------------------------------------ 3. frame order / sharing
def _obs(frames, B=2):
    return torch.randint(0, 256, (B, 3 * frames, H, W), dtype=torch.uint8)


@pytest.mark.parametrize("variant", ["cnn2f", "vit2f"])
def test_frame_independent_models_swap_halves_exactly(variant):
    enc = MatchedEncoder(_space(2), variant=variant).eval()
    x = _obs(2)
    swapped = torch.cat([x[:, 3:], x[:, :3]], dim=1)
    with torch.no_grad():
        y, ys = enc(x), enc(swapped)
    torch.testing.assert_close(ys, torch.cat([y[:, 64:], y[:, :64]], dim=1), rtol=1e-5, atol=1e-5)
    # and the halves differ, so the identity is not satisfied by a constant output
    assert not torch.allclose(y[:, :64], y[:, 64:], atol=1e-3)


@pytest.mark.parametrize("variant", ["cnn2f", "vit2f"])
def test_previous_frame_fills_the_first_half(variant):
    """The swap identity is symmetric in the concat order, so it cannot see [cur | prev]. Here, with
    the coupling LayerNorm(128) removed, changing ONLY the current frame (channels 3:6, the frame
    stack's newest) must leave the first 64 features bit-identical and move the last 64."""
    enc = MatchedEncoder(_space(2), variant=variant).eval()
    enc.out_norm = nn.Identity()
    x = _obs(2)
    x2 = x.clone()
    x2[:, 3:] = 255 - x2[:, 3:]
    with torch.no_grad():
        y, y2 = enc(x), enc(x2)
    torch.testing.assert_close(y2[:, :64], y[:, :64], rtol=0, atol=0)
    assert not torch.allclose(y2[:, 64:], y[:, 64:], atol=1e-3)


@pytest.mark.parametrize("variant", ["3dcnn", "vivit"])
def test_frame_mixing_models_do_not_commute(variant):
    enc = MatchedEncoder(_space(2), variant=variant).eval()
    x = _obs(2)
    swapped = torch.cat([x[:, 3:], x[:, :3]], dim=1)
    with torch.no_grad():
        y, ys = enc(x), enc(swapped)
    assert not torch.allclose(ys, torch.cat([y[:, 64:], y[:, :64]], dim=1), atol=1e-3)


@pytest.mark.parametrize("variant,mixes", [("3dcnn", True), ("3dcnn_g6", True), ("vivit", True), ("cnn2f", False),
                                           ("vit2f", False)])
def test_previous_frame_reaches_the_current_features_only_in_frame_mixing_models(variant, mixes):
    """With the coupling LayerNorm(128) removed, change ONLY the previous frame: the current frame's
    64 features move iff the backbone mixes frames (Conv3d temporal taps; ViViT's joint 320-token
    attention). The CNN2F / ViT2F rows are the control -- bit-identical, so the check can fail.
    (A frame SWAP cannot test ViViT: without temporal positions joint attention is permutation-
    equivariant over tokens, so it commutes with the swap by construction.)"""
    enc = MatchedEncoder(_space(2), variant=variant).eval()
    enc.out_norm = nn.Identity()
    x = _obs(2)
    x2 = x.clone()
    x2[:, :3] = 255 - x2[:, :3]
    with torch.no_grad():
        y, y2 = enc(x), enc(x2)
    if mixes:
        assert not torch.allclose(y2[:, 64:], y[:, 64:], atol=1e-4)
    else:
        torch.testing.assert_close(y2[:, 64:], y[:, 64:], rtol=0, atol=0)


def test_control_zeroed_positions_make_a_frame_independent_model_commute():
    enc = MatchedEncoder(_space(2), variant="cnn2f").eval()
    enc.out_norm = nn.Identity()
    x = _obs(2)
    with torch.no_grad():
        y, ys = enc(x), enc(torch.cat([x[:, 3:], x[:, :3]], dim=1))
    torch.testing.assert_close(ys, torch.cat([y[:, 64:], y[:, :64]], dim=1), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_block_structure_is_the_spec(variant):
    enc = MatchedEncoder(_space(VARIANTS[variant][0]), variant=variant)
    assert len(enc.blocks) == 3
    for blk in enc.blocks:
        assert [type(m) for m in blk.ffn] == [nn.Linear, nn.GELU, nn.Linear]
        assert blk.ffn[1].approximate == "none"
        assert (blk.ffn[0].in_features, blk.ffn[0].out_features, blk.ffn[2].out_features) == (96, 384, 96)
    assert [type(m) for m in enc.readout] == [nn.Conv2d, nn.ReLU, nn.Conv2d, nn.ReLU, nn.Flatten, nn.Linear]


@pytest.mark.parametrize("variant", ["cnn", "vivit"])
def test_patch_projection_sees_pixels_in_unit_range(variant):
    """Known answer: an all-255 observation reaches the patch conv as exactly 1.0 (one /255)."""
    T = VARIANTS[variant][0]
    enc = MatchedEncoder(_space(T), variant=variant).eval()
    seen = []
    enc.patch.register_forward_hook(lambda m, i, o: seen.append(i[0]))
    with torch.no_grad():
        enc(torch.full((1, 3 * T, H, W), 255, dtype=torch.uint8))
    assert float(seen[0].min()) == 1.0 == float(seen[0].max())


def test_single_frame_models_read_the_one_frame():
    for v in ("cnn", "vit"):
        enc = MatchedEncoder(_space(1), variant=v).eval()
        x = _obs(1)
        with torch.no_grad():
            assert not torch.allclose(enc(x), enc(255 - x), atol=1e-3), v


# ------------------------------------------------------------------ 4. /255 once
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_uint8_is_normalised_once(variant):
    T = VARIANTS[variant][0]
    enc = MatchedEncoder(_space(T), variant=variant).eval()
    x = _obs(T)
    with torch.no_grad():
        torch.testing.assert_close(enc(x), enc.encode_prepared(x.float() / 255.0))
        assert not torch.allclose(enc(x), enc.encode_prepared(x.float()), atol=1e-3)


def test_hwc_observation_space_builds_the_same_model():
    a = MatchedEncoder(gym.spaces.Box(0, 255, (H, W, 6), np.uint8), variant="vivit")
    assert _n(a) == SPEC_COUNTS["M-ViViT"]


# ------------------------------------------------------------------ 5. refusals
@pytest.mark.parametrize("h,w", [(80, 128), (160, 240), (256, 160)])
def test_any_other_eye_refuses(h, w):
    with pytest.raises(ValueError, match="160 x 256"):
        MatchedEncoder(_space(1, h, w), variant="cnn")


@pytest.mark.parametrize("variant,channels", [("cnn", 6), ("vit", 6), ("cnn2f", 3), ("vivit", 3),
                                               ("3dcnn", 9), ("vit2f", 4)])
def test_wrong_channel_count_refuses(variant, channels):
    with pytest.raises(ValueError):
        MatchedEncoder(gym.spaces.Box(0, 255, (channels, H, W), np.uint8), variant=variant)


def test_bad_cfg_refuses():
    with pytest.raises(ValueError, match="features_dim"):
        MatchedEncoder(_space(1), variant="cnn", features_dim=512)
    with pytest.raises(ValueError, match="variant"):
        MatchedEncoder(_space(1), variant="CNN")
    with pytest.raises(TypeError):
        MatchedEncoder(_space(1), variant="cnn", conv_dim=75)


# ------------------------------------------------------------------ 6/7. init + head through the builder
def _build(label):
    from nett_skrl.brain.brain import Brain
    from nett_skrl.brain.models.builder import build_models_for_algorithm
    from nett_skrl.brain.models.model_cfg import ModelCfg
    from nett_skrl.brain.registry import algorithm_spec
    spec = ct.MODELS[label]
    kw = EncoderCfg(**spec["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    head = spec["head"]
    # the "model" block campaign_train.main assembles for a head-carrying label
    cfg = ModelCfg(value_bound=None, shared_encoder=True, hidden_sizes=list(head["hidden_sizes"]),
                   clip_actions=False, activation=head["activation"], orthogonal_init=head["orthogonal_init"])
    return build_models_for_algorithm(
        algorithm_spec(Brain(encoder="matched", algorithm="PPO").algorithm),
        encoder_cls=encoder_mapping["matched"], encoder_kwargs=kw, observation_space=_space(_frames(label)),
        action_space=gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32), device=torch.device("cpu"), cfg=cfg)


def _in_bound(mod):
    fan_in = mod.weight[0].numel()
    b = 1.0 / math.sqrt(fan_in)
    ok = bool(mod.weight.abs().max() <= b)
    return ok and (mod.bias is None or bool(mod.bias.abs().max() <= b))


@pytest.mark.parametrize("label", ["M-CNN", "M-ViViT", "M-3DCNN-CLTT-Ref"])
def test_spec_init_survives_the_builder_and_heads_are_separate_tanh_mlps(label):
    models = _build(label)
    pol, val = models["policy"], models["value"]
    assert pol.encoder is val.encoder                                  # one shared encoder
    assert pol.trunk is not val.trunk                                  # separate head weights
    for m in (pol, val):
        lin = [x for x in m.trunk if isinstance(x, nn.Linear)]
        assert [(x.in_features, x.out_features) for x in lin] == [(128, 128), (128, 128)]
        assert [type(x) for x in m.trunk] == [nn.Linear, nn.Tanh, nn.Linear, nn.Tanh]
    assert (pol.mean_layer.in_features, pol.mean_layer.out_features) == (128, 2)
    assert (val.value_head.in_features, val.value_head.out_features) == (128, 1)
    layers = [x for m in (pol, val) for x in m.modules() if isinstance(x, (nn.Conv2d, nn.Conv3d, nn.Linear))]
    assert layers and all(_in_bound(x) for x in layers), [type(x).__name__ for x in layers if not _in_bound(x)]
    enc = pol.encoder
    for p in (enc.pos_spatial, enc.pos_temporal):
        if p is not None:
            assert abs(float(p.std()) - 0.02) < 0.002 and abs(float(p.mean())) < 0.002
    for ln in (x for x in enc.modules() if isinstance(x, nn.LayerNorm)):
        assert ln.eps == 1e-6 and torch.all(ln.weight == 1) and torch.all(ln.bias == 0)


def test_compact_3dcnn_differs_from_3dcnn_only_in_the_mixer_groups():
    """Owner §92 answer 2: groups=6 in the backbone Conv3d mixers; everything else is 3dcnn's."""
    a, b = MatchedEncoder(_space(2), variant="3dcnn"), MatchedEncoder(_space(2), variant="3dcnn_g6")
    sa, sb = {k: v.shape for k, v in a.state_dict().items()}, {k: v.shape for k, v in b.state_dict().items()}
    assert sa.keys() == sb.keys()
    diff = sorted(k for k in sa if sa[k] != sb[k])
    assert diff == [f"blocks.{i}.mixer.op.weight" for i in range(3)], diff
    for i in range(3):
        op = b.blocks[i].mixer.op
        assert isinstance(op, nn.Conv3d) and op.groups == 6 and op.kernel_size == (3, 3, 3) and op.padding == (1, 1, 1)
        assert tuple(op.weight.shape) == (96, 16, 3, 3, 3)


def test_control_orthogonal_init_breaks_the_bound():
    from nett_skrl.brain.models.utils.init import encoder_owns_init, orthogonal_init
    enc = MatchedEncoder(_space(1), variant="cnn")
    assert encoder_owns_init(enc)
    enc.apply(lambda m: orthogonal_init(m, math.sqrt(2.0)))
    layers = [x for x in enc.modules() if isinstance(x, (nn.Conv2d, nn.Linear))]
    assert not all(_in_bound(x) for x in layers)


# ------------------------------------------------------------------ 8. cltt_ref through the encoder
class _Mem:
    def __init__(self, raw):
        self.tensors = {"observations": raw,
                        "terminated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool),
                        "truncated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool)}
        self.memory_size, self.filled, self.memory_index = raw.shape[0], True, 0

    def get_tensor_by_name(self, name):
        raise AssertionError("never transfer the whole buffer")


@pytest.mark.parametrize("variant", ["cnn", "vivit"])
def test_cltt_ref_trains_every_encoder_parameter(monkeypatch, variant):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    T = VARIANTS[variant][0]
    enc = MatchedEncoder(gym.spaces.Box(0, 255, (H, W, 3 * T), np.uint8), variant=variant)
    aux = CLTTReferenceAuxLoss(enc)
    aux.attach_memory(_Mem(torch.randint(256, (10, 2, H * W * 3 * T), dtype=torch.uint8)))
    loss = aux.compute(enc, torch.empty(0))
    assert torch.isfinite(loss)
    loss.backward()
    dead = [n for n, p in enc.named_parameters() if p.grad is None or float(p.grad.norm()) == 0.0]
    assert not dead, dead


# ------------------------------------------------------------------ 9. labels are additive
def test_labels_are_new_and_twins_differ_only_in_the_aux():
    for base in SPEC_COUNTS:
        a, b = ct.MODELS[base], ct.MODELS[base + "-CLTT-Ref"]
        assert {k: v for k, v in b.items() if k not in ("aux", "aux_weight")} == a
        assert (b["aux"], b["aux_weight"]) == ("cltt_ref", 1.0) and "aux" not in a
        assert a["head"] == ct.MATCHED_HEAD and a["encoder"] == "matched"
    # the plain names the spec uses are DIFFERENT registered models, untouched
    assert ct.MODELS["CNN"]["encoder"] == "nature_cnn" and ct.MODELS["ViViT"]["encoder"] == "compact_vivit"
    assert not any("head" in s for k, s in ct.MODELS.items() if not k.startswith("M-"))


def test_head_label_refuses_a_stray_trunk_knob(monkeypatch):
    """main() refuses NETT_HIDDEN_SIZES / NETT_FEATURE_ATTN on a head-carrying label before Kit."""
    src = (_SRC / "examples" / "campaign_train.py").read_text()
    assert 'stray = sorted(k for k in ("NETT_HIDDEN_SIZES", "NETT_FEATURE_ATTN") if os.environ.get(k, "").strip())' in src
    monkeypatch.setenv("NETT_MODEL", "M-CNN")
    monkeypatch.setenv("NETT_HIDDEN_SIZES", "64,64")
    monkeypatch.setenv("NETT_EYE_RES", "256x160")
    monkeypatch.delenv("NETT_NAS_CFG", raising=False)
    with pytest.raises(ValueError, match="carries its own head"):
        ct.main()
