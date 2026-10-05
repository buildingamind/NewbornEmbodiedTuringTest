"""U35 MAE (owner request 2026-10-05): token MAE on CompactViT, per-frame masked input on
Compact3DCNN, VideoMAE on CompactViViT; four labels; existing labels unchanged.

Pinned here:
- every existing base (ViT-CLTT-Ref, 3DCNN-CLTT-Ref, ViViT) builds the SAME state_dict (keys, shapes,
  values at a fixed seed) and the SAME forward output as before the hooks were added;
- the new visible-token / spatial paths reproduce the existing forward when nothing is masked;
- each term trains: finite loss, gradients reach the encoder AND the head;
- the realised masked fraction is the knob's;
- 3DCNN per-frame masks are independent at TUBE=0 and identical at TUBE=1;
- the frame split is T-MAJOR, and a C-major split is caught.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.aux import mae_aux
from nett_skrl.brain.aux.mae_aux import MAETerm, frames_tmajor, patchify, random_mask, split_mask
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128    # the default eye (NETT_RES=128 -> 128x80)
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
LABELS = {"ViT-CLTT-Ref-MAE": "ViT-CLTT-Ref", "3DCNN-CLTT-Ref-MAE": "3DCNN-CLTT-Ref",
          "VideoMAE": "ViViT", "VideoMAE-CLTT-Ref": "ViViT"}
_KNOBS = ("NETT_AUX_MAE_RATIO", "NETT_AUX_MAE_TUBE", "NETT_AUX_MAE_NORM_PIX", "NETT_AUX_MAE_BATCH",
          "NETT_AUX_MAE_DEC_DIM", "NETT_AUX_MAE_DEC_DEPTH", "NETT_AUX_MAE_DEC_HEADS",
          "NETT_AUX_MAE_PATCH")


@pytest.fixture(autouse=True)
def _clean_knobs(monkeypatch):
    for k in _KNOBS:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


def _space(channels=6):
    return gym.spaces.Box(low=0, high=255, shape=(channels, H, W), dtype=np.uint8)


def _build(spec, channels=6, seed=0):
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](_space(channels), **kwargs)


def _obs(b=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (b, 6, H, W), generator=g, dtype=torch.uint8)


def _params(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- labels and byte-identity

def test_labels_are_their_base_except_aux(campaign):
    for new, base in LABELS.items():
        a, b = dict(campaign.MODELS[new]), dict(campaign.MODELS[base])
        assert a.pop("aux") in ("mae", "mae_with_cltt_ref")
        b.pop("aux", None)
        a.pop("aux_weight"), b.pop("aux_weight", None)
        assert a == b, new
    assert campaign.MODELS["VideoMAE"]["aux"] == "mae"
    assert {campaign.MODELS[k]["aux"] for k in LABELS if k != "VideoMAE"} == {"mae_with_cltt_ref"}


_BASE_SHA = "71a5fec"
_MODULE = {"ViT-CLTT-Ref": ("compact_vit", "CompactViT"),
           "3DCNN-CLTT-Ref": ("compact_3dcnn", "Compact3DCNN"),
           "ViViT": ("compact_vivit", "CompactViViT")}


def _pre_change_class(module, cls):
    """The encoder class as it was at 71a5fec, before the U35 hooks: exec'd from git into the
    live package namespace so its relative imports resolve to the same helpers."""
    import importlib.util
    import subprocess

    rel = f"src_isaac/nett_skrl/brain/encoders/{module}.py"
    try:
        src = subprocess.run(["git", "show", f"{_BASE_SHA}:{rel}"], cwd=_TRAIN.parents[1],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"{_BASE_SHA} not reachable from this checkout")
    name = f"nett_skrl.brain.encoders._pre_u35_{module}"
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    mod.__package__ = "nett_skrl.brain.encoders"
    exec(compile(src, rel, "exec"), mod.__dict__)
    return getattr(mod, cls)


@pytest.mark.parametrize("base", list(_MODULE))
def test_existing_base_is_byte_identical_to_pre_change(campaign, base):
    """ViT-CLTT-Ref, 3DCNN-CLTT-Ref, ViViT: same state_dict keys, shapes AND values at a fixed
    seed (so RNG consumption at construction is unchanged), and the same forward output."""
    spec = campaign.MODELS[base]
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(0)
    old = _pre_change_class(*_MODULE[base])(_space(), **kwargs).eval()
    new = _build(spec, seed=0).eval()
    assert type(new).__name__ == type(old).__name__
    a, b = old.state_dict(), new.state_dict()
    assert [(k, tuple(v.shape)) for k, v in a.items()] == [(k, tuple(v.shape)) for k, v in b.items()]
    assert all(torch.equal(a[k], b[k]) for k in a)
    x = _obs(4)
    with torch.no_grad():
        assert torch.equal(old(x), new(x))


def test_vit_keep_all_is_forward(campaign):
    enc = _build(campaign.MODELS["ViT-CLTT-Ref"]).eval()
    x = _obs(3)
    with torch.no_grad():
        p = enc._prepare_image(x)
        full = enc.encode_visible_prepared(p, torch.arange(40).expand(3, 40))
        assert torch.allclose(enc.head(full[:, 0]), enc(x), atol=1e-6)


def test_vivit_mirror_reproduces_forward(campaign):
    enc = _build(campaign.MODELS["ViViT"]).eval()
    x = _obs(3)
    with torch.no_grad():
        p = enc._prepare_image(x)
        full = enc.encode_visible_prepared(p, torch.arange(80).expand(3, 80))
        assert torch.allclose(enc.head(full[:, 0]), enc(x), atol=1e-6)


def test_3dcnn_spatial_map_pools_to_forward(campaign):
    enc = _build(campaign.MODELS["3DCNN-CLTT-Ref"]).eval()
    x = _obs(3)
    with torch.no_grad():
        fmap = enc.encode_spatial(x)
        assert fmap.shape[-2:] == (H // 4, W // 4)
        pooled = enc.cnn2d[-2:](fmap)
        assert torch.allclose(enc.linear(pooled), enc(x), atol=1e-6)


# ---------------------------------------------------------------- the terms train

class _Memory:
    """Rollout memory as the trainer holds it (HWC uint8), for the composed rows' cltt_ref half."""

    def __init__(self, t_max=24, n_env=2):
        self.tensors = {
            "observations": torch.randint(0, 256, (t_max, n_env, H, W, 6), dtype=torch.uint8),
            "terminated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
            "truncated": torch.zeros(t_max, n_env, 1, dtype=torch.bool),
        }
        self.memory_size, self.filled, self.memory_index = t_max, True, 0


@pytest.mark.parametrize("layout", ["chw", "hwc"])
@pytest.mark.parametrize("label", list(LABELS))
def test_term_trains_and_reports(campaign, monkeypatch, label, layout):
    """The full aux as AuxLossPPO calls it (composed rows include cltt_ref), on uint8 obs at the
    live eye in both layouts (Isaac hands HWC). Finite loss; grads reach encoder AND MAE head."""
    monkeypatch.setenv("NETT_AUX_BATCH", "4")
    spec = campaign.MODELS[label]
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(0)
    shape = (6, H, W) if layout == "chw" else (H, W, 6)
    enc = encoder_mapping[spec["encoder"]](
        gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8), **kwargs).train()
    aux = AUX_LOSSES[spec["aux"]](enc)
    term = aux if spec["aux"] == "mae" else aux.term
    if getattr(aux, "needs_memory", False):
        aux.attach_memory(_Memory())
    obs = _obs(8)
    if layout == "hwc":
        obs = obs.permute(0, 2, 3, 1).contiguous()
    loss = aux.compute(enc, obs)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    enc_g = sum(float(p.grad.abs().sum()) for p in enc.parameters() if p.grad is not None)
    head_g = sum(float(p.grad.abs().sum()) for p in term.head.parameters() if p.grad is not None)
    assert enc_g > 0 and head_g > 0
    assert all(p.grad is not None for p in term.head["dec"].parameters())
    ratio = {"vit": 0.75, "cnn3d": 0.75, "vivit": 0.90}[term.mode]
    assert abs(term.last_scalars["mask_frac"] - ratio) < 0.03
    print(f"{label}/{layout}: encoder {_params(enc):,}  mae head {_params(term.head):,}  "
          f"aux head total {_params(aux.head):,}  mode {term.mode}  tube {term.tube}")


def test_ratio_knob_is_realised(campaign, monkeypatch):
    monkeypatch.setenv("NETT_AUX_MAE_RATIO", "0.5")
    enc = _build(campaign.MODELS["ViT-CLTT-Ref"])
    term = MAETerm(enc)
    term.compute(enc, _obs(4))
    assert term.last_scalars["mask_frac"] == pytest.approx(0.5)


def test_refusals(campaign, monkeypatch):
    with pytest.raises(TypeError, match="supports CompactViT"):
        MAETerm(_build(campaign.MODELS["CNN"], channels=3))
    monkeypatch.setenv("NETT_AUX_MAE_TUBE", "0")
    with pytest.raises(ValueError, match="per-frame masking is not expressible"):
        MAETerm(_build(campaign.MODELS["ViT-CLTT-Ref"]))
    monkeypatch.setenv("NETT_AUX_MAE_TUBE", "maybe")
    with pytest.raises(ValueError):
        MAETerm(_build(campaign.MODELS["3DCNN-CLTT-Ref"]))
    monkeypatch.delenv("NETT_AUX_MAE_TUBE")
    monkeypatch.setenv("NETT_AUX_MAE_RATIO", "1.0")
    with pytest.raises(ValueError, match="strictly between"):
        MAETerm(_build(campaign.MODELS["3DCNN-CLTT-Ref"]))
    monkeypatch.setenv("NETT_AUX_MAE_RATIO", "0.75")
    monkeypatch.setenv("NETT_AUX_MAE_PATCH", "24")
    with pytest.raises(ValueError, match="does not divide"):
        MAETerm(_build(campaign.MODELS["3DCNN-CLTT-Ref"]))


# ---------------------------------------------------------------- masks

def test_mask_helpers():
    torch.manual_seed(0)
    m = random_mask(5, 40, 0.75, "cpu")
    assert (m.sum(1) == 30).all()
    keep, restore = split_mask(m)
    assert keep.shape == (5, 10) and (~m).gather(1, keep).all()
    assert torch.equal(keep, keep.sort(1).values)
    order = torch.cat([keep, torch.stack([torch.nonzero(r).flatten() for r in m])], 1)
    assert torch.equal(order.gather(1, restore), torch.arange(40).expand(5, 40))


def test_3dcnn_frames_masked_independently_or_as_tubes(campaign, monkeypatch):
    enc = _build(campaign.MODELS["3DCNN-CLTT-Ref"])
    torch.manual_seed(0)
    m = MAETerm(enc).draw_masks(16, "cpu")                 # default TUBE=0
    assert m.shape == (16, 2, 40)
    assert not torch.equal(m[:, 0], m[:, 1])
    assert (m.sum(-1) == 30).all()                          # each frame masks exactly 75%
    monkeypatch.setenv("NETT_AUX_MAE_TUBE", "1")
    m = MAETerm(enc).draw_masks(16, "cpu")
    assert torch.equal(m[:, 0], m[:, 1])


def test_vivit_tube_masks_same_patches_in_every_frame(campaign):
    term = MAETerm(_build(campaign.MODELS["ViViT"]))
    m = term.draw_masks(6, "cpu").view(6, 2, 40)
    assert torch.equal(m[:, 0], m[:, 1]) and (m[:, 0].sum(1) == 36).all()


def test_masked_input_is_tmajor_and_cmajor_is_caught(campaign):
    """Frame 0 = zeros, frame 1 = ones; mask frame 0 only at patch 0. A T-major split fills
    channels 0-2 of patch 0 and leaves 3-5 at one; a C-major split would fill (0,1),(2,3)..."""
    term = MAETerm(_build(campaign.MODELS["3DCNN-CLTT-Ref"]))
    prepared = torch.cat([torch.zeros(1, 3, H, W), torch.ones(1, 3, H, W)], 1)
    mask = torch.zeros(1, 2, 40, dtype=torch.bool)
    mask[0, 0, 0] = True
    out = term.mask_input(prepared, mask)
    fill = term.head["mask_value"]().detach()
    assert torch.allclose(out[0, :3, :16, :16], fill.view(3, 1, 1).expand(3, 16, 16))
    assert torch.equal(out[0, 3:, :16, :16], torch.ones(3, 16, 16))
    assert torch.equal(out[0, :3, :16, 16:], torch.zeros(3, 16, 112))

    def cmajor(p, t):                                     # the bug: C outer, T inner
        b, ct, h, w = p.shape
        return p.view(b, ct // t, t, h, w).transpose(1, 2)
    good, bad = frames_tmajor(prepared, 2), cmajor(prepared, 2)
    assert good[0, 0].max() == 0 and good[0, 1].min() == 1
    assert not (bad[0, 0].max() == 0 and bad[0, 1].min() == 1)


def test_patchify_matches_vit_token_order(campaign):
    enc = _build(campaign.MODELS["ViT-CLTT-Ref"])
    x = torch.rand(2, 6, H, W)
    w = enc.patch_embed.weight                            # (D, 6, 16, 16)
    tok = enc.patch_embed(x).flatten(2).transpose(1, 2)   # (2, 40, D)
    lin = patchify(x, 16) @ w.reshape(w.shape[0], -1).T + enc.patch_embed.bias
    assert torch.allclose(tok, lin, atol=1e-4)


def test_mae_module_has_no_pretrained_loading():
    src = Path(mae_aux.__file__).read_text()
    assert "load_state_dict" not in src and "torch.hub" not in src
