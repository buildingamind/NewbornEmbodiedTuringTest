"""BASELINE series G1b aux terms (owner 2026-10-28; workspace notes/chicken/presenter_i800_i805/baseline/PLAN_BASELINE.md §4).

  CNN-D512P+MASK75     the `mae` aux's NatureCNN path: SimMIM-style masked-input reconstruction (8x8 patches, 75 %
                       replaced by a learned per-channel value, decoder on the unpooled conv map).
  CNN-D512P+SC257      `slot_contrast_257`: slot_contrast's temporal slot-slot contrast with the s257 slot module
  ViT-P8-666K+SC257    (K=5 learnable queries + BO-QSA, fg-weighted RGB reconstruction x11), joint with the encoder.

Pinned: each label builds the agent's way (encoder from the label's cfg, aux from AUX_LOSSES on that encoder) on a
(3, 80, 128) space; encoder counts are unchanged (666,272 / 659,704); aux head counts; a forward + aux + backward
step on CPU; aux gradients reach the encoder AND every head parameter; the masked fraction is 75 %; SC257's queries
are 5 learnable per-slot parameters; its reconstruction weights object pixels x11.
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
from nett_skrl.brain.aux import slot_contrast_257_aux as sc257  # noqa: E402
from nett_skrl.brain.aux.mae_aux import MAETerm, patchify  # noqa: E402
from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES  # noqa: E402
from nett_skrl.brain.config import EncoderCfg  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402

H, W = 80, 128
LABELS = ("CNN-D512P+MASK75", "CNN-D512P+SC257", "ViT-P8-666K+SC257")
ENCODER_COUNT = {"CNN-D512P+MASK75": 666_272, "CNN-D512P+SC257": 666_272, "ViT-P8-666K+SC257": 659_704}
#: aux head (the set AuxLossPPO adds to its optimizer), at 128x80 and default knobs.
#:   MASK75: decoder 3x3 conv 64->64 (36,928) + 1x1 conv 64->192 (12,480) + mask value 3 = 49,411.
#:   SC257:  input (SoftPos + LN-MLP to 64) 8,768 (CNN, 64 ch) / 14,732 (ViT, 148 ch) + slot attention 54,528 +
#:           broadcast decoder 412,484 -- the latter two are s257's modules, shape for shape.
HEAD_COUNT = {"CNN-D512P+MASK75": 49_411, "CNN-D512P+SC257": 475_780, "ViT-P8-666K+SC257": 481_744}
_KNOBS = ("NETT_AUX_MAE_RATIO", "NETT_AUX_MAE_TUBE", "NETT_AUX_MAE_NORM_PIX", "NETT_AUX_MAE_BATCH",
          "NETT_AUX_MAE_DEC_DIM", "NETT_AUX_MAE_DEC_DEPTH", "NETT_AUX_MAE_DEC_HEADS", "NETT_AUX_MAE_PATCH",
          "NETT_AUX_BATCH", "NETT_SLOTC_SLOTS", "NETT_SLOTC_DIM", "NETT_SLOTC_ITERS", "NETT_SLOTC_TEMP",
          "NETT_SLOTC_W_SS", "NETT_SLOTC_W_REC", "NETT_SLOTC_FG_WEIGHT", "NETT_SLOTC_FG_THRESH", "NETT_SLOTC_DIAG",
          "NETT_SLOTC_EMA", "NETT_SLOTC_NO_DETACH", "NETT_AMP")


@pytest.fixture(autouse=True)
def _clean_knobs(monkeypatch):
    for k in _KNOBS:
        monkeypatch.delenv(k, raising=False)


def _space(channels=3):
    return gym.spaces.Box(low=0, high=255, shape=(channels, H, W), dtype=np.uint8)


def _encoder(label, channels=3, seed=0):
    spec = ct.MODELS[label]
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](_space(channels), **kwargs)


class _FakeMemory:
    """Rollout buffer the episode-contiguity helpers read: (T, envs, C, H, W) uint8, no resets."""

    def __init__(self, obs):
        t, envs = obs.shape[:2]
        self.tensors = {"observations": obs,
                        "terminated": torch.zeros(t, envs, 1, dtype=torch.bool),
                        "truncated": torch.zeros(t, envs, 1, dtype=torch.bool)}
        self.memory_size = self.memory_index = t
        self.filled = True


def _build(label, monkeypatch, seed=0, obs=None):
    """Encoder + aux the agent's way; SC257 gets a small rollout buffer (NETT_AUX_BATCH=4)."""
    enc = _encoder(label, seed=seed)
    aux = AUX_LOSSES[ct.MODELS[label]["aux"]](enc)
    if getattr(aux, "needs_memory", False):
        monkeypatch.setattr(aux, "max_samples", 4)
        g = torch.Generator().manual_seed(seed)
        if obs is None:
            obs = torch.randint(0, 256, (6, 2, 3, H, W), dtype=torch.uint8, generator=g)
        aux.attach_memory(_FakeMemory(obs))
    return enc, aux


def _batch(n=4, seed=1):
    return torch.randint(0, 256, (n, 3, H, W), dtype=torch.uint8, generator=torch.Generator().manual_seed(seed))


# ---------------------------------------------------------------- labels, construction, counts

def test_labels_are_the_base_plus_one_aux():
    m = ct.MODELS
    assert m["CNN-D512P+MASK75"] == {**m["CNN-D512P"], "aux": "mae", "aux_weight": 1.0}
    assert m["CNN-D512P+SC257"] == {**m["CNN-D512P"], "aux": "slot_contrast_257", "aux_weight": 1.0}
    assert m["ViT-P8-666K+SC257"] == {**m["ViT-P8-666K"], "aux": "slot_contrast_257", "aux_weight": 1.0}
    for label in LABELS:
        assert m[label]["framestack"] is False and ct.MODELS[label]["aux"] in AUX_LOSSES


@pytest.mark.parametrize("label", LABELS)
def test_counts(label, monkeypatch):
    enc, aux = _build(label, monkeypatch)
    n_enc = sum(p.numel() for p in enc.parameters())
    n_head = sum(p.numel() for p in aux.head.parameters())
    print(f"{label}: encoder {n_enc:,}  aux head {n_head:,}  "
          + "  ".join(f"{k} {sum(p.numel() for p in v.parameters()):,}" for k, v in aux.head.named_children()))
    assert n_enc == ENCODER_COUNT[label]
    assert n_head == HEAD_COUNT[label]
    # the head is the whole trainable set of the term, and holds no encoder tensor
    assert {id(p) for p in aux.head.parameters()} == {id(p) for p in aux.parameters()}
    assert not ({id(p) for p in aux.head.parameters()} & {id(p) for p in enc.parameters()})


def test_mask75_is_the_masked_input_mode_at_the_owner_defaults(monkeypatch):
    _, aux = _build("CNN-D512P+MASK75", monkeypatch)
    assert isinstance(aux, MAETerm)
    assert (aux.mode, aux.ratio, aux.patch, aux.n_h, aux.n_w) == ("cnn2d", 0.75, 8, 10, 16)


# ---------------------------------------------------------------- one training step, gradients

def _aux_loss(label, enc, aux):
    obs = _batch()
    return aux.compute(enc, obs), enc(obs)


@pytest.mark.parametrize("label", LABELS)
def test_forward_aux_backward_step_and_gradients(label, monkeypatch):
    enc, aux = _build(label, monkeypatch)
    loss, feats = _aux_loss(label, enc, aux)
    assert feats.shape == (4, 512) and loss.dim() == 0 and torch.isfinite(loss) and loss.requires_grad
    # aux alone first: what does IT reach?
    loss.backward(retain_graph=True)
    missing = [n for n, p in aux.head.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()
               or float(p.grad.abs().sum()) == 0.0]
    assert not missing, f"head parameters with no aux gradient: {missing}"
    # joint: every trunk tensor gets an aux gradient; the readout projection (NatureCNN.linear, CompactViT.head after
    # the CLS token) is not on the spatial path these terms read, so it gets none
    readout = "linear." if "CNN" in label else "head."
    no_grad = [n for n, p in enc.named_parameters() if not n.startswith(readout)
               and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    assert not no_grad, f"encoder tensors the aux does not reach: {no_grad}"
    assert all(p.grad is None for n, p in enc.named_parameters() if n.startswith(readout))
    # then one full optimizer step over encoder + head, as AuxLossPPO's param groups do
    opt = torch.optim.Adam(list(enc.parameters()), lr=1e-3)
    opt.add_param_group({"params": list(aux.head.parameters())})
    (feats.pow(2).mean()).backward()
    before = [p.detach().clone() for p in aux.head.parameters()]
    opt.step()
    assert any(not torch.equal(a, p) for a, p in zip(before, aux.head.parameters()))
    assert {"positions", "slots", "ss", "rec", "fg_frac"} <= set(aux.last_scalars) if "SC257" in label \
        else aux.last_scalars["mask_frac"] == pytest.approx(0.75)


# ---------------------------------------------------------------- MASK75: what is masked

def test_mask75_masks_75_percent_of_8x8_patches_with_the_learned_value(monkeypatch):
    enc, aux = _build("CNN-D512P+MASK75", monkeypatch)
    torch.manual_seed(3)
    prepared = torch.rand(5, 3, H, W)
    with torch.no_grad():
        aux.head["mask_value"].value.copy_(torch.tensor([0.11, 0.22, 0.33]))
    mask = aux.draw_masks(5, "cpu")                                         # (B, 1, N)
    assert mask.shape == (5, 1, 160) and (mask.sum(-1) == 120).all()       # exactly 75 % of 160 per image
    out = aux.mask_input(prepared, mask)
    p_in, p_out = patchify(prepared, 8), patchify(out, 8)                  # (B, 160, 192), (c, py, px) order
    fill = torch.tensor([0.11, 0.22, 0.33]).repeat_interleave(64)
    m = mask[:, 0]
    assert torch.equal(p_out[~m], p_in[~m]), "a visible patch was changed"
    assert torch.allclose(p_out[m], fill.expand_as(p_out[m])), "a masked patch is not the learned value"
    # and compute() reports the realised fraction
    aux.compute(enc, _batch())
    assert aux.last_scalars["mask_frac"] == pytest.approx(0.75)


def test_mask75_loss_is_on_masked_patches_only(monkeypatch):
    """With every visible patch's prediction made wrong and every masked one made right, the loss is ~0."""
    enc, aux = _build("CNN-D512P+MASK75", monkeypatch)
    monkeypatch.setenv("NETT_AUX_MAE_NORM_PIX", "0")
    aux.norm_pix = False
    obs = _batch(2)
    prepared = obs.float() / 255.0
    torch.manual_seed(5)
    mask = aux.draw_masks(2, "cpu")
    target = patchify(prepared, 8)
    pred = torch.where(mask[:, 0, :, None], target, target + 1.0)
    monkeypatch.setattr(aux, "draw_masks", lambda b, d: mask)
    monkeypatch.setattr(aux.head["dec"], "forward", lambda fmap: pred)
    assert float(aux.compute(enc, obs)) == pytest.approx(0.0, abs=1e-12)


def test_mae_refusals_on_naturecnn(monkeypatch):
    monkeypatch.setenv("NETT_AUX_MAE_TUBE", "0")
    with pytest.raises(ValueError, match="ONE image"):
        MAETerm(_encoder("CNN-D512P"))
    monkeypatch.delenv("NETT_AUX_MAE_TUBE")
    monkeypatch.setenv("NETT_AUX_MAE_PATCH", "12")
    with pytest.raises(ValueError, match="does not divide"):
        MAETerm(_encoder("CNN-D512P"))
    monkeypatch.delenv("NETT_AUX_MAE_PATCH")
    enc = encoder_mapping["nature_cnn"](_space(6), features_dim=512, conv_dim=64, input_frames=1)
    with pytest.raises(ValueError, match="input_frames"):
        MAETerm(enc)


def test_resample_matches_bilinear_interpolation():
    from nett_skrl.brain.aux.mae_aux import bilinear_matrix
    x = torch.randn(2, 4, 6, 12)
    ref = torch.nn.functional.interpolate(x, size=(10, 16), mode="bilinear", align_corners=False)
    got = torch.einsum("yh,bchw,xw->bcyx", bilinear_matrix(10, 6), x, bilinear_matrix(16, 12))
    assert torch.allclose(got, ref, atol=1e-5)


# ---------------------------------------------------------------- SC257: the slot module

@pytest.mark.parametrize("label,grid", [("CNN-D512P+SC257", (64, 6, 12)), ("ViT-P8-666K+SC257", (148, 10, 16))])
def test_sc257_reads_the_backbone_grid_with_5_learnable_queries(label, grid, monkeypatch):
    enc, aux = _build(label, monkeypatch)
    assert (aux.in_dim, aux.grid_h, aux.grid_w) == grid
    q = aux.head.attn.q
    assert isinstance(q, torch.nn.Parameter) and q.requires_grad and q.shape == (1, 5, 64)
    assert aux.slots == 5 and aux.head.attn.iters == 3 and aux.fg_weight == 10.0 and aux.fg_thresh == 30
    assert (aux.w_ss, aux.w_rec, aux.temperature) == (0.5, 1.0, 0.1)
    assert not hasattr(aux.head.attn, "mu") and not hasattr(aux, "_target")    # no Gaussian draw, no EMA trunk
    # query init: the slots are a deterministic function of the image (no random draw)
    prep = _batch(2).float() / 255.0
    with torch.no_grad():
        a, _ = aux._slots(enc, prep)
        b, _ = aux._slots(enc, prep)
    assert torch.equal(a, b)


def test_sc257_straight_through_on_the_query_pass_only():
    """BO-QSA: on the query pass the queries' gradient is the identity through the LAST iteration only; the
    recurrent pass (init = given slots) runs plain slot attention, so the init gets the full chain."""
    torch.manual_seed(0)
    sa = sc257.QuerySlotAttention(5, 64, 3)
    x = torch.randn(2, 30, 64)
    s, _ = sa(x)
    g_st = torch.autograd.grad(s.sum(), sa.q)[0]
    # reference: one step from the (detached) slots after iters-1 steps, with the identity routed to q
    with torch.no_grad():
        xn = sa.ni(x)
        k, v = sa.to_k(xn), sa.to_v(xn)
        pre = sa.q.expand(2, -1, -1)
        for _ in range(sa.iters - 1):
            pre, _ = sa.step(pre, k, v)
    init = sa.q.expand(2, -1, -1)
    s_ref, _ = sa.step(pre + init - init.detach(), k, v)
    assert torch.allclose(g_st, torch.autograd.grad(s_ref.sum(), sa.q)[0], atol=1e-6)
    # recurrent pass: gradient reaches the init through every iteration (differs from a one-step ST)
    init2 = torch.randn(2, 5, 64, requires_grad=True)
    s2, _ = sa(x, slots_init=init2)
    g_full = torch.autograd.grad(s2.sum(), init2)[0]
    with torch.no_grad():
        pre2 = init2.detach()
        for _ in range(sa.iters - 1):
            pre2, _ = sa.step(pre2, k, v)
    s_st2, _ = sa.step(pre2 + init2 - init2.detach(), k, v)
    g_st2 = torch.autograd.grad(s_st2.sum(), init2)[0]
    assert float(g_full.abs().sum()) > 0 and not torch.allclose(g_full, g_st2, atol=1e-4)


def _blob_frame():
    """grey background (128,128,128), a red 10x16 blob (200, 30, 30) at rows 30..39, cols 50..65."""
    img = torch.full((3, H, W), 128, dtype=torch.uint8)
    img[:, 30:40, 50:66] = torch.tensor([200, 30, 30], dtype=torch.uint8)[:, None, None]
    return img


def test_foreground_mask_is_s257_chroma_over_30():
    img = _blob_frame()
    img[:, 0, 0] = torch.tensor([130, 100, 100], dtype=torch.uint8)     # chroma exactly 30: background
    img[:, 0, 1] = torch.tensor([131, 100, 100], dtype=torch.uint8)     # chroma 31: object
    fg = sc257.foreground_mask(img[None].float() / 255.0, 30)[0, 0]
    want = torch.zeros(H, W)
    want[30:40, 50:66] = 1
    want[0, 1] = 1
    assert torch.equal(fg, want)


@pytest.mark.parametrize("label", ["CNN-D512P+SC257", "ViT-P8-666K+SC257"])
def test_sc257_reconstruction_weights_object_pixels_x11(label, monkeypatch):
    """A recon error of 0.1 on n blob pixels costs 11x the same error on n background pixels, and a uniform error
    costs mean(1 + 10 * fg); the contrast weight is zeroed so the term returns the reconstruction alone."""
    monkeypatch.setenv("NETT_SLOTC_W_SS", "0")
    frame = _blob_frame()
    obs = frame.expand(6, 2, 3, H, W).contiguous()
    enc, aux = _build(label, monkeypatch, obs=obs)
    x = frame.float() / 255.0
    fg = torch.zeros(H, W, dtype=torch.bool)
    fg[30:40, 50:66] = True
    n = int(fg.sum())                                                        # 160 object pixels
    bg_same = torch.zeros(H, W, dtype=torch.bool)
    bg_same.view(-1)[torch.nonzero(~fg.view(-1))[:n, 0]] = True

    def loss_with_error(where):
        pred = x.expand(4, 3, H, W) + 0.1 * where.to(x.dtype)
        monkeypatch.setattr(aux.head.decoder, "forward", lambda slots: (pred, None))
        return float(aux.compute(enc, obs))

    on_obj, on_bg = loss_with_error(fg), loss_with_error(bg_same)
    assert on_obj / on_bg == pytest.approx(11.0, rel=1e-5)
    uniform = loss_with_error(torch.ones(H, W, dtype=torch.bool))
    assert uniform == pytest.approx(0.01 * (1 + 10 * n / (H * W)), rel=1e-5)
    assert aux.last_scalars["fg_frac"] == pytest.approx(n / (H * W))


def test_sc257_modules_are_s257_shape_for_shape():
    """The slot attention and decoder state_dicts have slot_ae.py's s257 keys and shapes (a s257 checkpoint's
    sa.* / dec.* load strictly; measured 2026-10-28: forward and query gradients bit-identical)."""
    sa = sc257.QuerySlotAttention(5, 64, 3)
    dec = sc257.BroadcastDecoder(64, 3, H, W)
    assert {k: tuple(v.shape) for k, v in sa.state_dict().items()} == {
        "q": (1, 5, 64), "ni.weight": (64,), "ni.bias": (64,), "ns.weight": (64,), "ns.bias": (64,),
        "to_q.weight": (64, 64), "to_k.weight": (64, 64), "to_v.weight": (64, 64),
        "gru.weight_ih": (192, 64), "gru.weight_hh": (192, 64), "gru.bias_ih": (192,), "gru.bias_hh": (192,),
        "mlp.0.weight": (64,), "mlp.0.bias": (64,), "mlp.1.weight": (128, 64), "mlp.1.bias": (128,),
        "mlp.3.weight": (64, 128), "mlp.3.bias": (64,)}
    assert {k: tuple(v.shape) for k, v in dec.state_dict().items()} == {
        "softpos.lin.weight": (64, 4), "softpos.lin.bias": (64,),
        **{f"net.{i}.weight": (64, 64, 5, 5) for i in (0, 2, 4, 6)}, **{f"net.{i}.bias": (64,) for i in (0, 2, 4, 6)},
        "net.8.weight": (4, 64, 3, 3), "net.8.bias": (4,)}
    rgb, alpha = dec(torch.randn(2, 5, 64))
    assert rgb.shape == (2, 3, H, W) and alpha.shape == (2, 5, H, W)
    assert torch.allclose(alpha.sum(1), torch.ones(2, H, W), atol=1e-5)


def test_sc257_refusals(monkeypatch):
    with pytest.raises(ValueError, match="ONE RGB frame"):
        sc257.SlotContrast257AuxLoss(_encoder("CNN-D512P", channels=6))
    for knob in ("NETT_SLOTC_EMA", "NETT_SLOTC_NO_DETACH"):
        monkeypatch.setenv(knob, "1")
        with pytest.raises(ValueError, match="no EMA feature target"):
            sc257.SlotContrast257AuxLoss(_encoder("CNN-D512P"))
        monkeypatch.delenv(knob)
    enc = _encoder("CNN-D512P")
    aux = sc257.SlotContrast257AuxLoss(enc)
    with pytest.raises(RuntimeError, match="attach_memory"):
        aux.compute(enc, _batch())


def test_sc257_diag_emits_slot_contrast_keys(monkeypatch):
    monkeypatch.setenv("NETT_SLOTC_DIAG", "1")
    enc, aux = _build("CNN-D512P+SC257", monkeypatch)
    aux.compute(enc, _batch())
    s = aux.last_scalars
    assert s["mask_variance"] >= 0 and s["ss_pos_acc"] >= 0 and s["ss_batch"] == 4
    # query init carries no noise: the identical-input null is ~0
    assert s["mask_variance_null"] == pytest.approx(0.0, abs=1e-10)
