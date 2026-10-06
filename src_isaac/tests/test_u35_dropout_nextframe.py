"""U35 (owner 2026-10-05): ViT-CLTT-Ref dropout (all / aux-only) and next-frame predictive coding.

DropAll / DropAux -- "ViT-CLTT-Ref" + two dropout fields; the scope is the label's. DropAux is
                     live only inside cltt_ref's encoder passes, so the PPO forward is untouched.
NextFrame         -- decode the NEWEST frame of obs[t+1] from the spatial map at t and a_t.

⛔ Every existing label must build byte-identically: the state_dict tests below pin keys, shapes
AND values for ViT-CLTT-Ref and 3DCNN at a fixed seed, and the forward is checked bitwise.
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

from nett_skrl.brain.aux import cltt_ref_aux
from nett_skrl.brain.aux.nextframe_aux import NextFrameTerm, spatial_map
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.aux.token_term import NOT_MEASURED
from nett_skrl.brain.aux.with_cltt_ref import WithCLTTRef
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.encoders.compact_vit import CompactViT
from nett_skrl.brain.models.utils.features import features_forward, shared_feature_cache
from nett_skrl.brain.registry import encoder_mapping

H, W, C = 80, 128, 6
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
SMALL = dict(features_dim=32, patch_size=16, embed_dim=32, depth=2, num_heads=2)
NEW = ("ViT-CLTT-Ref-DropAll", "ViT-CLTT-Ref-DropAux", "3DCNN-NextFrame", "ViT-CLTT-Ref-NextFrame")


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("NETT_VIT_DROPOUT", "NETT_AUX_NF_BATCH", "NETT_AUX_NF_DOWNSAMPLE",
                 "NETT_AUX_NF_HIDDEN", "NETT_AUX_NF_TRANSIT_FRAC", "NETT_AUX_NF_NORM_PIX",
                 "NETT_AUX_CLTT_CHANNELS_PER_FRAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    monkeypatch.setenv("NETT_AUX_NF_BATCH", "8")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(35)
        yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _space(channels=C):
    return gym.spaces.Box(low=0, high=255, shape=(channels, H, W), dtype=np.uint8)


def _build(spec, seed=0):
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](_space(), **kwargs)


def _vit(**kw):
    torch.manual_seed(0)
    return CompactViT(_space(), **{**SMALL, **kw})


def _n(module):
    return sum(p.numel() for p in module.parameters())


class _FrameMemory:
    """Two-frame T-major stacks whose NEWEST frame at step t is the constant v(t) = 100 + 4t and
    whose OLDER frame is v(t-1) -- so obs[t+1]'s older half EQUALS obs[t]'s newest frame, which is
    exactly the leak the target must not contain. Turns vary so both strata are populated."""

    def __init__(self, t_max=24, n_env=2):
        obs = torch.zeros(t_max, n_env, H, W, C)
        for t in range(t_max):
            obs[t, :, :, :, :3] = 100 + 4 * (t - 1)
            obs[t, :, :, :, 3:] = 100 + 4 * t
        acts = torch.zeros(t_max, n_env, 2)
        acts[:, :, 0] = torch.linspace(-0.5, 0.5, t_max).view(t_max, 1)
        acts[:, :, 1] = 0.3
        self.tensors = {
            "observations": obs.to(torch.uint8),
            "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "actions": acts,
        }
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


class _NoiseMemory(_FrameMemory):
    def __init__(self, t_max=24, n_env=2):
        super().__init__(t_max, n_env)
        g = torch.Generator().manual_seed(3)
        self.tensors["observations"] = torch.randint(0, 255, (t_max, n_env, H, W, C),
                                                     generator=g, dtype=torch.uint8)


# ---------------------------------------------------------------- registry

def test_new_labels_are_one_factor_off_their_bases(campaign):
    M = campaign.MODELS
    ref, cnn = M["ViT-CLTT-Ref"], M["3DCNN"]
    for lbl, scope in (("ViT-CLTT-Ref-DropAll", "all"), ("ViT-CLTT-Ref-DropAux", "aux")):
        spec = M[lbl]
        assert spec["cfg"] == {**ref["cfg"], "dropout": 0.1, "dropout_scope": scope}
        assert {k: v for k, v in spec.items() if k != "cfg"} == {k: v for k, v in ref.items() if k != "cfg"}
    assert M["3DCNN-NextFrame"] == {**cnn, "aux": "nextframe", "aux_weight": 1.0}
    assert M["ViT-CLTT-Ref-NextFrame"] == {**ref, "aux": "nextframe_with_cltt_ref"}
    for lbl in NEW:
        assert M[lbl]["aux"] in AUX_LOSSES


def test_vit_dropout_knob_refuses_out_of_range(campaign, monkeypatch):
    for bad in ("0", "1", "-0.1", "1.5", "nan", "x"):
        monkeypatch.setenv("NETT_VIT_DROPOUT", bad)
        with pytest.raises(ValueError, match="NETT_VIT_DROPOUT"):
            campaign._vit_dropout()
    monkeypatch.setenv("NETT_VIT_DROPOUT", "0.25")
    assert campaign._vit_dropout() == 0.25


def test_compact_vit_refuses_bad_dropout_fields():
    with pytest.raises(ValueError, match="dropout_scope"):
        _vit(dropout=0.1)                                  # rate without a scope
    with pytest.raises(ValueError, match="dropout_scope"):
        _vit(dropout=0.1, dropout_scope="policy")
    for bad in (0.0, 1.0, float("nan"), -0.2):
        with pytest.raises(ValueError):
            _vit(dropout=bad, dropout_scope="aux")


# ---------------------------------------------------------------- byte identity

def _sd_signature(enc):
    return [(k, tuple(v.shape)) for k, v in enc.state_dict().items()]


@pytest.mark.parametrize("base,variants", [
    ("ViT-CLTT-Ref", ("ViT-CLTT-Ref-DropAll", "ViT-CLTT-Ref-DropAux", "ViT-CLTT-Ref-NextFrame")),
    ("3DCNN", ("3DCNN-NextFrame",)),
])
def test_state_dict_keys_shapes_and_values_match_the_base(campaign, base, variants):
    ref = _build(campaign.MODELS[base], seed=11)
    sd = ref.state_dict()
    for lbl in variants:
        enc = _build(campaign.MODELS[lbl], seed=11)
        assert _sd_signature(enc) == _sd_signature(ref), lbl
        assert all(torch.equal(sd[k], v) for k, v in enc.state_dict().items()), lbl


def test_base_forwards_are_deterministic_and_unscoped_vit_has_no_live_dropout(campaign):
    """No dropout path is entered for an unscoped trunk (live_p == 0 skips F.dropout entirely).
    Bitwise identity WITH THE PRE-EDIT CODE is pinned separately: ViT by
    test_compact_vit_token_hook.py's frozen copy, 3DCNN by the frozen-fixture test below."""
    x = torch.randint(0, 255, (3, C, H, W), dtype=torch.uint8)
    for base in ("ViT-CLTT-Ref", "3DCNN"):
        enc = _build(campaign.MODELS[base], seed=5).train()
        torch.manual_seed(1)
        a = enc(x)
        torch.manual_seed(2)
        assert torch.equal(a, enc(x)), base
    vit = _build(campaign.MODELS["ViT-CLTT-Ref"], seed=5)
    assert all(b.live_p == 0.0 for b in vit.blocks)


#: `git rev-parse 71a5fec:src_isaac/nett_skrl/brain/encoders/compact_3dcnn.py`, copied from the
#: command's output when the fixture was cut (same pattern as test_compact_vit_token_hook.py).
FROZEN_3DCNN = Path(__file__).resolve().parent / "fixtures" / "frozen" / "compact_3dcnn_71a5fec.py.txt"
FROZEN_3DCNN_BLOB = "cfa6b9aa53b40b550264cdfed762a11b3288ab1f"


def _load_frozen_3dcnn():
    name = "nett_skrl.brain.encoders._frozen_compact_3dcnn_71a5fec"
    if name in sys.modules:
        return sys.modules[name]
    loader = importlib.machinery.SourceFileLoader(name, str(FROZEN_3DCNN))
    spec = importlib.util.spec_from_file_location(name, str(FROZEN_3DCNN), loader=loader)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_frozen_3dcnn_fixture_is_the_pre_u35_file():
    data = FROZEN_3DCNN.read_bytes()
    assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == FROZEN_3DCNN_BLOB
    assert "_temporal_stem" not in vars(_load_frozen_3dcnn().Compact3DCNN)


@pytest.mark.parametrize("label,channels", [
    ("3DCNN", 6), ("3DCNN-1F", 3), ("3DCNN-Sp1", 6), ("3DCNN-DVS", 4), ("3DCNN-NextFrame", 6),
])
def test_3dcnn_forward_and_grads_are_bitwise_the_pre_u35_code(campaign, label, channels):
    """⛔ BITWISE, against real pre-edit code: the `_temporal_stem` refactor must not move one bit
    of output or gradient for any 3DCNN row (duplicate-frame and DVS paths included)."""
    kwargs = EncoderCfg(**campaign.MODELS[label]["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    space = _space(channels)
    torch.manual_seed(0)
    old = _load_frozen_3dcnn().Compact3DCNN(space, **kwargs)
    torch.manual_seed(1)
    new = encoder_mapping["compact_3dcnn"](space, **kwargs)
    assert list(old.state_dict()) == list(new.state_dict())
    new.load_state_dict(old.state_dict())
    x = torch.randint(0, 256, (3, channels, H, W), generator=torch.Generator().manual_seed(5),
                      dtype=torch.uint8)
    a, b = old(x), new(x)
    assert torch.equal(a, b), label
    a.pow(2).sum().backward()
    b.pow(2).sum().backward()
    for (k, p), q in zip(old.named_parameters(), new.parameters()):
        assert torch.equal(p.grad, q.grad), (label, k)


def test_parameter_counts_per_label(campaign):
    """Encoder and aux-head sizes at the default 128x80 eye, 2-frame stack (the report's table)."""
    M = campaign.MODELS
    counts = {}
    for lbl in ("ViT-CLTT-Ref",) + NEW:
        enc = _build(M[lbl])
        aux = AUX_LOSSES[M[lbl]["aux"]](enc)
        counts[lbl] = (_n(enc), _n(aux.head))
    assert counts["ViT-CLTT-Ref"] == (804_320, 329_216)
    assert counts["ViT-CLTT-Ref-DropAll"] == counts["ViT-CLTT-Ref-DropAux"] == (804_320, 329_216)
    assert counts["3DCNN-NextFrame"] == (695_981, 44_035)
    assert counts["ViT-CLTT-Ref-NextFrame"] == (804_320, 329_216 + 122_179)


# ---------------------------------------------------------------- dropout semantics

def _prep(enc, n=4):
    x = torch.randint(0, 255, (n, C, H, W), dtype=torch.uint8)
    return x, enc._prepare_image(x)


def test_dropaux_policy_forward_is_deterministic_in_train_mode_and_aux_is_not():
    enc = _vit(dropout=0.3, dropout_scope="aux").train()
    x, prepared = _prep(enc)
    assert torch.equal(enc(x), enc(x))                       # the PPO forward: no dropout
    with enc.aux_dropout():
        a, b = enc.encode_prepared(prepared), enc.encode_prepared(prepared)
    assert not torch.equal(a, b)                             # the aux forward: dropout live
    assert all(blk.live_p == 0.0 for blk in enc.blocks)      # and switched off on exit
    assert torch.equal(enc(x), enc(x))


def test_dropaux_is_reset_when_the_aux_pass_raises():
    enc = _vit(dropout=0.3, dropout_scope="aux").train()
    with pytest.raises(RuntimeError):
        with enc.aux_dropout():
            raise RuntimeError("boom")
    assert all(blk.live_p == 0.0 for blk in enc.blocks)


def test_dropall_differs_across_train_mode_forwards():
    enc = _vit(dropout=0.3, dropout_scope="all").train()
    x, _ = _prep(enc)
    assert not torch.equal(enc(x), enc(x))
    with enc.aux_dropout():                                  # no-op for scope 'all'
        assert all(blk.live_p == 0.3 for blk in enc.blocks)
    assert all(blk.live_p == 0.3 for blk in enc.blocks)


@pytest.mark.parametrize("scope", ["all", "aux"])
def test_both_scopes_are_deterministic_in_eval_mode(scope):
    enc = _vit(dropout=0.3, dropout_scope=scope).eval()
    x, prepared = _prep(enc)
    assert torch.equal(enc(x), enc(x))
    with enc.aux_dropout():
        assert torch.equal(enc.encode_prepared(prepared), enc.encode_prepared(prepared))
    ref = _vit().eval()                                      # same seed, no dropout at all
    assert torch.equal(enc(x), ref(x))


def test_cltt_ref_runs_its_encoder_passes_inside_the_aux_scope(monkeypatch):
    """The aux-only scope is only real if cltt_ref's passes are the ones wrapped."""
    enc = _vit(dropout=0.3, dropout_scope="aux").train()
    aux = AUX_LOSSES["cltt_ref"](enc)
    aux.attach_memory(_NoiseMemory())
    seen = []
    orig = type(enc).encode_prepared

    def spy(self, prepared):
        seen.append(self.blocks[0].live_p)
        return orig(self, prepared)

    monkeypatch.setattr(type(enc), "encode_prepared", spy)
    loss = aux.compute(enc, None)
    assert torch.isfinite(loss) and seen and all(p == 0.3 for p in seen), seen
    assert enc.blocks[0].live_p == 0.0
    assert cltt_ref_aux.aux_dropout_scope(torch.nn.Linear(1, 1)) is not None   # non-ViT: no-op


def test_dropout_masks_survive_the_scope_exit_for_backward():
    """The aux backward runs AFTER the context exits; autograd must use the masks it drew."""
    enc = _vit(dropout=0.5, dropout_scope="aux").train()
    _, prepared = _prep(enc)
    with enc.aux_dropout():
        z = enc.encode_prepared(prepared)
    z.pow(2).sum().backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_feature_cache_cannot_cross_between_aux_and_ppo_forwards():
    """features_forward caches ONLY `model.encoder(x)` keyed on tensor identity; the aux calls
    encode_prepared, which neither reads nor writes that cache. So under DropAux a dropped-out aux
    vector cannot reach the policy, and a clean policy vector cannot stand in for the aux pass."""
    enc = _vit(dropout=0.3, dropout_scope="aux").train()
    model = type("M", (), {"encoder": enc, "trunk": torch.nn.Identity()})()
    x, prepared = _prep(enc)
    with shared_feature_cache(enc):
        with enc.aux_dropout():
            z_aux = enc.encode_prepared(prepared)
        assert getattr(enc, "_nett_shared_feature_cache", None) is None   # aux wrote nothing
        f1 = features_forward(model, {"observations": x})
        with enc.aux_dropout():
            z_aux2 = enc.encode_prepared(prepared)
        f2 = features_forward(model, {"observations": x})
    assert f1 is f2                                          # PPO actor/critic share one pass
    assert torch.equal(f1, enc(x))                           # and it is the dropout-free one
    assert not torch.equal(z_aux, z_aux2)                    # the aux never read the cache
    assert not torch.equal(z_aux, f1)


# ---------------------------------------------------------------- next-frame

def _nf_terms():
    vit = _vit()
    cnn = _build({"encoder": "compact_3dcnn",
                  "cfg": {"features_dim": 32, "conv_dim": 16, "num_frames": 2}})
    return [(vit, NextFrameTerm(vit)), (cnn, NextFrameTerm(cnn))]


def test_spatial_map_shapes_and_refusal():
    vit, cnn = _vit(), _build({"encoder": "compact_3dcnn",
                               "cfg": {"features_dim": 32, "conv_dim": 16, "num_frames": 2}})
    z = torch.zeros(2, C, H, W)
    assert tuple(spatial_map(vit, z).shape) == (2, 32, 5, 8)
    assert tuple(spatial_map(cnn, z).shape) == (2, 16, 20, 32)
    nature = encoder_mapping["nature_cnn"](_space(), features_dim=32)
    try:
        spatial_map(nature, z)                               # has encode_spatial: allowed
    except TypeError:
        pytest.fail("nature_cnn exposes encode_spatial and must not be refused")
    dup = encoder_mapping["compact_3dcnn"](_space(3), features_dim=32, conv_dim=16, num_frames=2,
                                           duplicate_frame=True)
    with pytest.raises(TypeError, match="spatial map"):
        spatial_map(dup, torch.zeros(2, 3, H, W))


def test_3dcnn_spatial_map_is_the_forward_trunk_before_its_pool():
    cnn = _build({"encoder": "compact_3dcnn",
                  "cfg": {"features_dim": 32, "conv_dim": 16, "num_frames": 2}})
    _, prepared = _prep(cnn)
    m = cnn.encode_spatial_prepared(prepared)
    pooled = cnn.cnn2d[-1](cnn.cnn2d[-2](m))                 # the pool + flatten forward applies
    assert torch.equal(cnn.linear(pooled), cnn.encode_prepared(prepared))


def test_nextframe_loss_is_finite_and_grads_reach_encoder_and_head():
    for enc, term in _nf_terms():
        term.attach_memory(_NoiseMemory())
        loss = term.compute(enc, None)
        assert torch.isfinite(loss) and loss.item() > 0
        loss.backward()
        enc_g = [p.grad for p in enc.parameters() if p.grad is not None]
        head_g = [p.grad for p in term.head.parameters() if p.grad is not None]
        assert enc_g and any(g.abs().sum() > 0 for g in enc_g), type(enc).__name__
        assert head_g and any(g.abs().sum() > 0 for g in head_g)
        enc_ids = {id(p) for p in enc.parameters()}
        assert not any(id(p) in enc_ids for p in term.head.parameters())


def test_film_starts_unconditioned_and_learns_an_action_gradient():
    enc, term = _nf_terms()[0]
    term.attach_memory(_NoiseMemory())
    term.compute(enc, None).backward()
    assert term.head.film.weight.abs().sum() == 0            # zero-init: action off at step 0
    assert term.head.film.weight.grad.abs().sum() > 0        # but the gradient reaches it
    assert term.last_scalars["film_gain"] == 0.0


def test_target_is_the_newest_frame_of_obs_t_plus_1(monkeypatch):
    """With newest(t) = v(t) and older(t+1) = v(t): a correct target is v(t+1), so the copy
    baseline (newest(t) -> target) errs by exactly 4/255 per pixel. A target that used obs[t+1]'s
    OLDER half, or obs[t] at all, would make copy_mse 0."""
    enc, term = _nf_terms()[1]
    term.attach_memory(_FrameMemory())
    frames = []
    orig = NextFrameTerm._frames

    def spy(self, prepared):
        out = orig(self, prepared)
        frames.append(out)
        return out

    monkeypatch.setattr(NextFrameTerm, "_frames", spy)
    term.compute(enc, None)
    target, current = frames
    assert tuple(target.shape[1:]) == (3, H // 4, W // 4)
    assert torch.allclose(target - current, torch.full_like(target, 4 / 255), atol=1e-6)
    assert term.last_scalars["copy_mse"] == pytest.approx((4 / 255) ** 2, rel=1e-4)


def test_scalars_include_copy_baseline_and_strata():
    enc, term = _nf_terms()[0]
    term.attach_memory(_NoiseMemory())
    term.compute(enc, None)
    s = term.last_scalars
    for key in ("B", "mse", "copy_mse", "skill", "mse_parked", "mse_transit", "copy_parked",
                "copy_transit", "window_turn", "action_dim", "film_gain"):
        assert key in s, key
    assert s["B"] == 8 and s["action_dim"] == 2
    assert s["copy_mse"] > 0 and s["skill"] == pytest.approx(1 - s["mse"] / s["copy_mse"])
    assert s["mse_parked"] != NOT_MEASURED and s["mse_transit"] != NOT_MEASURED
    assert s["window_turn"] >= 0.0 and term.last_window_turn >= 0.0


def test_norm_pix_normalises_the_target_per_sample(monkeypatch):
    monkeypatch.setenv("NETT_AUX_NF_NORM_PIX", "1")
    enc, term = _nf_terms()[0]
    term.attach_memory(_NoiseMemory())
    assert torch.isfinite(term.compute(enc, None))
    assert term.last_scalars["copy_mse"] > 0.5               # unit-variance target scale


def test_nextframe_refusals(monkeypatch):
    enc = _vit()
    monkeypatch.setenv("NETT_AUX_NF_DOWNSAMPLE", "3")
    with pytest.raises(ValueError, match="does not divide"):
        NextFrameTerm(enc)
    monkeypatch.delenv("NETT_AUX_NF_DOWNSAMPLE")
    monkeypatch.setenv("NETT_AUX_NF_BATCH", "1")
    with pytest.raises(ValueError, match=">= 2"):
        NextFrameTerm(enc)
    monkeypatch.setenv("NETT_AUX_NF_BATCH", "8")
    term = NextFrameTerm(enc)
    with pytest.raises(RuntimeError, match="attach_memory"):
        term.compute(enc, None)
    mem = _NoiseMemory()
    mem.tensors["actions"] = mem.tensors["actions"][..., :1]
    term.attach_memory(mem)
    with pytest.raises(ValueError, match="motor components"):
        term.compute(enc, None)


def test_registry_builders_compose_with_cltt_ref():
    enc = _vit()
    alone = AUX_LOSSES["nextframe"](enc)
    assert isinstance(alone, NextFrameTerm)
    comp = AUX_LOSSES["nextframe_with_cltt_ref"](enc)
    assert isinstance(comp, WithCLTTRef) and comp.name == "nextframe"
    assert isinstance(comp.term, NextFrameTerm) and comp._teacher is None
    comp.attach_memory(_NoiseMemory())
    loss = comp.compute(enc, None)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in comp.head[1].parameters())
