"""Slide-3 n=7 wave labels (owner 2026-10-30; workspace FINDINGS §4dh.44h.211 add. 219).

"I would like to run the following encoders (n=7) using the standard for those runs": the slide-3 backbones (CNN =
NatureCNN conv 75, ViT = VIT_CFG patch 16, 3DCNN, ViViT patch 16) each with ONE variation, auxes attached:

  ViViT-T2           ViViT early fusion (ViViT-CLTT-Ref-T2's encoder, no aux)
  CNN+DVS / ViT+DVS  + dvs_polarity (rows set NETT_DVS_COLOR=1: 6 event channels)
  *+MASK75           + mae: CNN / 3DCNN masked input, ViT token MAE, ViViT-T2 early-fusion token MAE (new: mode
                     vivit_early, CompactViViT.encode_visible_early_prepared)
  CNN+SC257 / ViT+SC257  + slot_contrast_257

Pinned: each label is its base plus that one field; encoder counts at 128x80 equal the base's; aux head counts;
a forward + aux + backward step on CPU with finite loss and gradients on every head parameter; the early-fusion MAE
masks 75 % of the 40 two-frame tokens, regresses both frames' pixels of the masked ones only, and its visible-token
path reproduces forward() exactly when nothing is masked; refusals (SC257 on a stack or DVS input, TUBE=0 on early
fusion, factored / late ViViT still refused).
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "examples"))

import campaign_train as ct  # noqa: E402
from nett_skrl.brain.aux.mae_aux import MAETerm, patchify, split_mask  # noqa: E402
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES  # noqa: E402
from nett_skrl.brain.config import EncoderCfg  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402

H, W = 80, 128
#: label -> (base, the one added field(s), input channels at 128x80)
WAVE = {
    "CNN+DVS":         ("CNN",      {"pre": ("dvs_polarity",)}, 6),
    "ViT+DVS":         ("ViT",      {"pre": ("dvs_polarity",)}, 6),
    "CNN+MASK75":      ("CNN",      {"aux": "mae", "aux_weight": 1.0}, 3),
    "ViT+MASK75":      ("ViT",      {"aux": "mae", "aux_weight": 1.0}, 3),
    "3DCNN+MASK75":    ("3DCNN",    {"aux": "mae", "aux_weight": 1.0}, 6),
    "ViViT-T2+MASK75": ("ViViT-T2", {"aux": "mae", "aux_weight": 1.0}, 6),
    "CNN+SC257":       ("CNN",      {"aux": "slot_contrast_257", "aux_weight": 1.0}, 3),
    "ViT+SC257":       ("ViT",      {"aux": "slot_contrast_257", "aux_weight": 1.0}, 3),
}
LABELS = ("ViViT-T2", *WAVE)
CHANNELS = {"ViViT-T2": 6, **{k: v[2] for k, v in WAVE.items()}}
#: encoder only, at 128x80 (DVS: 6 channels, so the patch embed / first conv is wider than the RGB base's)
ENCODER_COUNT = {"ViViT-T2": 804_320, "CNN+DVS": 780_139, "ViT+DVS": 804_320, "CNN+MASK75": 773_995,
                 "ViT+MASK75": 693_728, "3DCNN+MASK75": 695_981, "ViViT-T2+MASK75": 804_320, "CNN+SC257": 773_995,
                 "ViT+SC257": 693_728}
#: aux head (the set AuxLossPPO optimizes), default knobs.
#:   CNN+MASK75      resample decoder 3x3 conv (map ch -> 64) + 1x1 conv 64 -> 192 + mask value 3
#:   ViT+MASK75      token decoder, 40 positions, out 3*16*16 = 768
#:   3DCNN+MASK75    conv decoder + pixel shuffle, + mask value 3
#:   ViViT-T2+MASK75 token decoder, 40 positions, out 2*3*16*16 = 1,536
#:   *+SC257         slot input + slot attention 54,528 + broadcast decoder 412,484
HEAD_COUNT = {"CNN+MASK75": 55_747, "ViT+MASK75": 95_488, "3DCNN+MASK75": 50_659, "ViViT-T2+MASK75": 145_408,
              "CNN+SC257": 476_561, "ViT+SC257": 481_460}
AUX_LABELS = tuple(HEAD_COUNT)
_KNOBS = ("NETT_AUX_MAE_RATIO", "NETT_AUX_MAE_TUBE", "NETT_AUX_MAE_NORM_PIX", "NETT_AUX_MAE_BATCH",
          "NETT_AUX_MAE_DEC_DIM", "NETT_AUX_MAE_DEC_DEPTH", "NETT_AUX_MAE_DEC_HEADS", "NETT_AUX_MAE_PATCH",
          "NETT_AUX_BATCH", "NETT_SLOTC_SLOTS", "NETT_SLOTC_DIM", "NETT_SLOTC_ITERS", "NETT_SLOTC_TEMP",
          "NETT_SLOTC_W_SS", "NETT_SLOTC_W_REC", "NETT_SLOTC_FG_WEIGHT", "NETT_SLOTC_FG_THRESH", "NETT_SLOTC_DIAG",
          "NETT_SLOTC_EMA", "NETT_SLOTC_NO_DETACH", "NETT_AMP", "NETT_FRAMESTACK_N")


@pytest.fixture(autouse=True)
def _clean_knobs(monkeypatch):
    for k in _KNOBS:
        monkeypatch.delenv(k, raising=False)


def _space(channels):
    return gym.spaces.Box(low=0, high=255, shape=(channels, H, W), dtype=np.uint8)


def _encoder(label, channels=None, seed=0):
    spec = ct.MODELS[label]
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](_space(channels or CHANNELS[label]), **kwargs)


class _FakeMemory:
    """Rollout buffer the episode-contiguity helpers read: (T, envs, C, H, W) uint8, no resets."""

    def __init__(self, obs):
        t, envs = obs.shape[:2]
        self.tensors = {"observations": obs,
                        "terminated": torch.zeros(t, envs, 1, dtype=torch.bool),
                        "truncated": torch.zeros(t, envs, 1, dtype=torch.bool)}
        self.memory_size = self.memory_index = t
        self.filled = True


def _build(label, monkeypatch, seed=0):
    enc = _encoder(label, seed=seed)
    aux = AUX_LOSSES[ct.MODELS[label]["aux"]](enc)
    if getattr(aux, "needs_memory", False):
        monkeypatch.setattr(aux, "max_samples", 4)
        g = torch.Generator().manual_seed(seed)
        aux.attach_memory(_FakeMemory(torch.randint(0, 256, (6, 2, 3, H, W), dtype=torch.uint8, generator=g)))
    return enc, aux


def _batch(label, n=4, seed=1):
    return torch.randint(0, 256, (n, CHANNELS[label], H, W), dtype=torch.uint8,
                         generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- labels: base + one field

def test_each_label_is_its_base_plus_one_field():
    m = ct.MODELS
    assert m["ViViT-T2"] == dict(encoder="compact_vivit", cfg={**ct.VIVIT_CFG, "temporal_mode": "early"},
                                 framestack=True)
    # ViViT early fusion - CLTT-Ref is the existing label: the same encoder + cltt_ref
    assert m["ViViT-CLTT-Ref-T2"] == {**m["ViViT-T2"], "aux": "cltt_ref", "aux_weight": 1.0}
    for label, (base, extra, _) in WAVE.items():
        assert m[label] == {**m[base], **extra}, label
    # the slide-3 backbones themselves, as the wave reads them
    assert m["CNN"]["encoder"] == "nature_cnn" and m["CNN"]["cfg"]["conv_dim"] == 75 and m["CNN"]["framestack"] is False
    assert m["ViT"]["cfg"] == ct.VIT_CFG and m["ViT"]["cfg"]["patch_size"] == 16 and m["ViT"]["framestack"] is False
    assert m["3DCNN"]["encoder"] == "compact_3dcnn" and m["3DCNN"]["framestack"] is True
    for label in LABELS:
        assert "decouple" not in str(m[label]).lower()        # attached, as the owner chose


@pytest.mark.parametrize("label", LABELS)
def test_encoder_counts(label):
    enc = _encoder(label)
    n = sum(p.numel() for p in enc.parameters())
    print(f"{label}: encoder {n:,}")
    assert n == ENCODER_COUNT[label]
    base = WAVE.get(label, (None,))[0]
    if base is not None and "pre" not in WAVE[label][1]:
        assert n == sum(p.numel() for p in _encoder(base, channels=CHANNELS[label]).parameters())


@pytest.mark.parametrize("label", ("ViViT-T2", "CNN+DVS", "ViT+DVS"))
def test_no_aux_labels_forward(label):
    enc = _encoder(label)
    out = enc(_batch(label))
    assert out.shape == (4, 512) and torch.isfinite(out).all()


# ---------------------------------------------------------------- aux labels: counts, one step

@pytest.mark.parametrize("label", AUX_LABELS)
def test_aux_head_counts(label, monkeypatch):
    enc, aux = _build(label, monkeypatch)
    n_head = sum(p.numel() for p in aux.head.parameters())
    print(f"{label}: aux head {n_head:,}")
    assert n_head == HEAD_COUNT[label]
    assert {id(p) for p in aux.head.parameters()} == {id(p) for p in aux.parameters()}
    assert not ({id(p) for p in aux.head.parameters()} & {id(p) for p in enc.parameters()})


@pytest.mark.parametrize("label", AUX_LABELS)
def test_forward_aux_backward_step(label, monkeypatch):
    enc, aux = _build(label, monkeypatch)
    obs = _batch(label)
    loss, feats = aux.compute(enc, obs), enc(obs)
    assert feats.shape == (4, 512) and loss.dim() == 0 and torch.isfinite(loss) and loss.requires_grad
    loss.backward(retain_graph=True)
    missing = [n for n, p in aux.head.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()
               or float(p.grad.abs().sum()) == 0.0]
    assert not missing, f"head parameters with no aux gradient: {missing}"
    reached = [n for n, p in enc.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0.0]
    assert reached, "the aux reaches no encoder tensor: not attached"
    opt = torch.optim.Adam(list(enc.parameters()), lr=1e-3)
    opt.add_param_group({"params": list(aux.head.parameters())})
    feats.pow(2).mean().backward()
    before = [p.detach().clone() for p in aux.head.parameters()]
    opt.step()
    assert any(not torch.equal(a, p) for a, p in zip(before, aux.head.parameters()))
    if "MASK75" in label:
        assert aux.last_scalars["mask_frac"] == pytest.approx(0.75)
    else:
        assert {"positions", "slots", "ss", "rec", "fg_frac"} <= set(aux.last_scalars)


def test_mask75_modes(monkeypatch):
    modes = {}
    for label in ("CNN+MASK75", "ViT+MASK75", "3DCNN+MASK75", "ViViT-T2+MASK75"):
        _, aux = _build(label, monkeypatch)
        assert isinstance(aux, MAETerm) and aux.ratio == 0.75
        modes[label] = (aux.mode, aux.tube, aux.patch, aux.n_pos)
    assert modes == {"CNN+MASK75": ("cnn2d", True, 8, 160), "ViT+MASK75": ("vit", True, 16, 40),
                     "3DCNN+MASK75": ("cnn3d", False, 16, 40),      # per-frame masks: the module's 3DCNN default
                     "ViViT-T2+MASK75": ("vivit_early", True, 16, 40)}
    monkeypatch.setenv("NETT_AUX_MAE_RATIO", "0.75")              # the rows' explicit setting changes nothing
    _, aux = _build("ViViT-T2+MASK75", monkeypatch)
    assert aux.ratio == 0.75


# ---------------------------------------------------------------- ViViT-T2 early-fusion MAE

def test_early_visible_path_with_nothing_masked_is_forward():
    enc = _encoder("ViViT-T2", seed=3)
    obs = _batch("ViViT-T2", n=3)
    prepared = enc._prepare_image(obs)
    keep = torch.arange(enc.n_spatial).expand(3, -1)
    with torch.no_grad():
        tok = enc.encode_visible_early_prepared(prepared, keep)
        assert tok.shape == (3, 1 + 40, enc.embed_dim)
        assert torch.allclose(enc.head(tok[:, 0]), enc(obs), atol=1e-6)


def test_early_masks_75_percent_of_two_frame_tokens(monkeypatch):
    enc, aux = _build("ViViT-T2+MASK75", monkeypatch)
    mask = aux.draw_masks(5, "cpu")
    assert mask.shape == (5, 40) and (mask.sum(-1) == 30).all()
    keep, _ = split_mask(mask)
    assert keep.shape == (5, 10)
    out = enc.encode_visible_early_prepared(enc._prepare_image(_batch("ViViT-T2+MASK75", n=5)), keep)
    assert out.shape == (5, 11, enc.embed_dim)


def test_early_target_is_both_frames_and_loss_is_on_masked_tokens_only(monkeypatch):
    """With every visible token's prediction wrong and every masked one right, the loss is ~0; the target of token
    n is frame 0's patch n then frame 1's patch n (the early patch embed's input)."""
    enc, aux = _build("ViViT-T2+MASK75", monkeypatch)
    aux.norm_pix = False
    obs = _batch("ViViT-T2+MASK75", n=2)
    prepared = enc._prepare_image(obs)
    torch.manual_seed(5)
    mask = aux.draw_masks(2, "cpu")
    target = torch.cat([patchify(prepared[:, :3], 16), patchify(prepared[:, 3:], 16)], -1)
    assert target.shape == (2, 40, 1536)
    pred = torch.where(mask[:, :, None], target, target + 1.0)
    monkeypatch.setattr(aux, "draw_masks", lambda b, d: mask)
    monkeypatch.setattr(aux.head["dec"], "forward", lambda enc_tokens, ids_restore: pred)
    assert float(aux.compute(enc, obs)) == pytest.approx(0.0, abs=1e-12)
    # and a frame-1-only error on a masked token is seen (both frames are regressed)
    wrong = pred.clone()
    n = int(mask[0].nonzero()[0])
    wrong[0, n, 768:] += 1.0
    monkeypatch.setattr(aux.head["dec"], "forward", lambda enc_tokens, ids_restore: wrong)
    assert float(aux.compute(enc, obs)) > 0.0


# ---------------------------------------------------------------- refusals

def test_refusals(monkeypatch):
    monkeypatch.setenv("NETT_AUX_MAE_TUBE", "0")
    with pytest.raises(ValueError, match="BOTH"):
        MAETerm(_encoder("ViViT-T2"))
    monkeypatch.delenv("NETT_AUX_MAE_TUBE")
    for mode in ("factored", "late"):
        spec = {**ct.VIVIT_CFG, "temporal_mode": mode}
        kwargs = EncoderCfg(**spec).as_kwargs()
        kwargs.pop("trainable", None)
        with pytest.raises(ValueError, match="joint"):
            MAETerm(encoder_mapping["compact_vivit"](_space(6), **kwargs))
    with pytest.raises(ValueError, match="early"):
        _encoder("ViViT", channels=6).encode_visible_early_prepared(torch.zeros(1, 6, H, W), torch.zeros(1, 4, dtype=torch.long))
    # SC257 is one RGB frame: refuses a stack and DVS6 input
    for label in ("CNN+SC257", "ViT+SC257"):
        with pytest.raises(ValueError, match="ONE RGB frame"):
            AUX_LOSSES["slot_contrast_257"](_encoder(label, channels=6))
