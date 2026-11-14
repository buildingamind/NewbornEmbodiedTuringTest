"""D512P-T2 / CNN-K3 (owner 2026-11-14, first-hand; workspace FINDINGS add. 426).

"We would need to have the 3DCNN use a 2x8x8 kernel for it to be a proper comparison" -> D512P-T2, the EXACT video twin
of CNN-D512P ("keep 4x4 = exact D512P twin"); "test a 3x3 first layer kernel for CNN+CLTTA" -> CNN-K3, the 3DCNN trunk on
one frame with a real Conv2d. Pinned here, built on CPU at the campaign eye (128x80, 3 channels per frame):
  (a) each spec is its twin plus exactly the one field (and +CLTT-Ref is the label plus the attached cltt_ref);
  (b) encoder counts: D512P-T2 672,416 = CNN-D512P's 666,272 + conv1's 3 * 8 * 8 * 32 = 6,144; CNN-K3 695,117 =
      "3DCNN"'s 695,981 - the stem's 3 * 3 * 3 * 32 = 864; every parameter outside conv1 / the stem has the twin's shape;
  (c) TWIN EQUIVALENCE: D512P-T2 with conv3d.weight[:, :, -1] = CNN-D512P's conv1 weight and [:, :, 0] = 0 (all other
      parameters copied) gives CNN-D512P's features on the CURRENT frame (the last 3 channels of the T-major stack);
      with [:, :, -1] = 0 it reads only the previous frame; swapping the two frames changes the output;
  (d) "3DCNN" with conv3d.weight[:, :, -1] = CNN-K3's stem weight and [:, :, 0] = 0 reproduces CNN-K3 on the current frame;
  (e) the fleet init (NETT_INIT_SCHEME unset = family) gives the new stems Kaiming normal fan_in, zero bias, like the twins';
  (f) forward / encode_spatial shapes, and CLTT-Ref is finite with encoder gradients on both +CLTT-Ref labels;
  (g) the new cfg fields refuse the combinations they do not implement.
Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.models.utils.init import init_encoder
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train

    return campaign_train


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("NETT_AMP", "NETT_INIT_SCHEME", "NETT_CONV3D_ORTHO_INIT", "NETT_AUX_BATCH", "NETT_AUX_CLTT_CHANNELS_PER_FRAME",
              "NETT_AUX_CLTT_REF_OFFSETS", "NETT_CLTT_REF_DEDUP"):
        monkeypatch.delenv(k, raising=False)


def _encoder(campaign, label, seed=0, **over):
    spec = campaign.MODELS[label]
    kw = EncoderCfg(**{**spec["cfg"], **over}).as_kwargs()
    kw.pop("trainable", None)
    ch = 3 * (campaign._FRAMESTACK_N if spec["framestack"] else 1)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (ch, H, W), np.uint8), **kw)


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_specs_are_the_twin_plus_one_field(campaign):
    m = campaign.MODELS
    assert m["D512P-T2"] == {**m["CNN-D512P"], "cfg": {**m["CNN-D512P"]["cfg"], "temporal_frames": campaign._FRAMESTACK_N},
                             "framestack": True}
    assert m["CNN-K3"] == {**m["3DCNN"], "cfg": {**m["3DCNN"]["cfg"], "num_frames": 1, "stem_2d": True}, "framestack": False}
    for base in ("D512P-T2", "CNN-K3"):
        assert "aux" not in m[base] and "pre" not in m[base]
        assert m[f"{base}+CLTT-Ref"] == {**m[base], "aux": "cltt_ref", "aux_weight": 1.0}
    assert campaign._FRAMESTACK_N == 2


def test_parameter_counts_and_shapes(campaign):
    t2, d512p = _encoder(campaign, "D512P-T2"), _encoder(campaign, "CNN-D512P")
    k3, c3d = _encoder(campaign, "CNN-K3"), _encoder(campaign, "3DCNN")
    print(f"\nencoder params at 128x80: D512P-T2 {_n(t2):,}  CNN-D512P {_n(d512p):,}  CNN-K3 {_n(k3):,}  3DCNN {_n(c3d):,}")
    assert _n(d512p) == 666_272 and _n(t2) == 672_416 == 666_272 + 3 * 8 * 8 * 32
    assert _n(c3d) == 695_981 and _n(k3) == 695_117 == 695_981 - 3 * 3 * 3 * 32
    a, b = dict(t2.named_parameters()), dict(d512p.named_parameters())
    assert set(a) - set(b) == {"cnn.0.conv3d.weight", "cnn.0.conv3d.bias"} and set(b) - set(a) == {"cnn.0.weight", "cnn.0.bias"}
    assert all(a[n].shape == b[n].shape for n in a.keys() & b.keys())
    assert tuple(a["cnn.0.conv3d.weight"].shape) == (32, 3, 2, 8, 8)
    assert t2.cnn[0].conv3d.stride == (1, 4, 4) and t2.cnn[0].conv3d.padding == (0, 0, 0)
    assert t2.pool_grid == d512p.pool_grid == (3, 6)
    a, b = dict(k3.named_parameters()), dict(c3d.named_parameters())
    assert set(a) - set(b) == {"stem2d.weight", "stem2d.bias"} and set(b) - set(a) == {"conv3d.weight", "conv3d.bias"}
    assert all(a[n].shape == b[n].shape for n in a.keys() & b.keys())
    assert isinstance(k3.stem2d, torch.nn.Conv2d) and tuple(k3.stem2d.weight.shape) == (32, 3, 3, 3)
    assert k3.stem2d.stride == (2, 2) and k3.stem2d.padding == (1, 1)


def _stack(prev, cur):
    return torch.cat([prev, cur], dim=1)          # the framestack's T-MAJOR order: [t-1 RGB, t RGB]


def test_d512p_t2_is_the_exact_twin_of_cnn_d512p(campaign):
    t2, d512p = _encoder(campaign, "D512P-T2", seed=1).eval(), _encoder(campaign, "CNN-D512P", seed=2).eval()
    sd = {k: v for k, v in d512p.state_dict().items() if not k.startswith("cnn.0.")}
    w = torch.zeros_like(t2.cnn[0].conv3d.weight)
    w[:, :, -1] = d512p.cnn[0].weight
    sd["cnn.0.conv3d.weight"], sd["cnn.0.conv3d.bias"] = w, d512p.cnn[0].bias
    t2.load_state_dict(sd)
    torch.manual_seed(3)
    prev, cur = torch.randint(0, 256, (2, 4, 3, H, W), dtype=torch.uint8)
    with torch.no_grad():
        want = d512p(cur)
        assert torch.allclose(t2(_stack(prev, cur)), want, atol=1e-5)
        assert torch.allclose(t2.encode_spatial(_stack(prev, cur)), d512p.encode_spatial(cur), atol=1e-5)
        assert not torch.allclose(t2(_stack(cur, prev)), want, atol=1e-3)      # frame order matters
        w2 = torch.zeros_like(w)
        w2[:, :, 0] = d512p.cnn[0].weight
        t2.cnn[0].conv3d.weight.copy_(w2)                                       # previous-frame slice only
        assert torch.allclose(t2(_stack(prev, cur)), d512p(prev), atol=1e-5)
        assert torch.allclose(t2(_stack(prev, torch.zeros_like(cur))), d512p(prev), atol=1e-5)


def test_cnn_k3_is_the_3dcnn_trunk_on_one_frame(campaign):
    k3, c3d = _encoder(campaign, "CNN-K3", seed=1).eval(), _encoder(campaign, "3DCNN", seed=2).eval()
    sd = {k: v for k, v in k3.state_dict().items() if not k.startswith("stem2d.")}
    w = torch.zeros_like(c3d.conv3d.weight)
    w[:, :, -1] = k3.stem2d.weight
    sd["conv3d.weight"], sd["conv3d.bias"] = w, k3.stem2d.bias
    c3d.load_state_dict(sd)
    torch.manual_seed(3)
    prev, cur = torch.randint(0, 256, (2, 4, 3, H, W), dtype=torch.uint8)
    with torch.no_grad():
        assert torch.allclose(c3d(_stack(prev, cur)), k3(cur), atol=1e-5)
        assert torch.allclose(c3d.encode_spatial(_stack(prev, cur)), k3.encode_spatial(cur), atol=1e-5)
        assert tuple(k3(cur).shape) == (4, 512) and tuple(k3.encode_spatial(cur).shape) == (4, 77, 20, 32)


def test_family_init_gives_the_new_stems_kaiming(campaign, capsys):
    for label, stem in (("D512P-T2", lambda e: e.cnn[0].conv3d), ("CNN-K3", lambda e: e.stem2d)):
        enc = _encoder(campaign, label)
        init_encoder(enc, gain=2 ** 0.5)
        assert "kaiming (NETT_INIT_SCHEME=family" in capsys.readouterr().out
        conv = stem(enc)
        fan_in = conv.weight[0].numel()
        assert torch.count_nonzero(conv.bias) == 0
        assert abs(conv.weight.std().item() / (2.0 / fan_in) ** 0.5 - 1) < 0.1, (label, fan_in)


@pytest.mark.parametrize("label", ["D512P-T2+CLTT-Ref", "CNN-K3+CLTT-Ref"])
def test_cltt_ref_is_finite_with_encoder_gradients(campaign, monkeypatch, label):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss

    class _FakeMemory:
        def __init__(self, raw):
            self.tensors = {"observations": raw,
                            "terminated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool),
                            "truncated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool)}
            self.memory_size, self.filled, self.memory_index = raw.shape[0], True, 0

        def get_tensor_by_name(self, name):
            raise AssertionError("never transfer the whole buffer")

    c = 6 if campaign.MODELS[label]["framestack"] else 3
    monkeypatch.setenv("NETT_AUX_BATCH", "8")
    enc = _encoder(campaign, label)
    aux = CLTTReferenceAuxLoss(enc)
    torch.manual_seed(1)
    aux.attach_memory(_FakeMemory(torch.randint(0, 256, (24, 2, c * H * W)).float()))
    value = aux.compute(enc, torch.empty(0))
    assert torch.isfinite(value) and value.requires_grad
    value.backward()
    stem = enc.cnn[0].conv3d if c == 6 else enc.stem2d
    assert stem.weight.grad is not None and torch.isfinite(stem.weight.grad).all() and stem.weight.grad.abs().sum() > 0


def test_new_fields_refuse_what_they_do_not_implement(campaign):
    with pytest.raises(ValueError, match="exclusive"):
        _encoder(campaign, "D512P-T2", input_frames=1)
    with pytest.raises(ValueError, match="temporal_frames"):
        _encoder(campaign, "D512P-T2", temporal_frames=1)
    with pytest.raises(ValueError, match="stem_2d"):
        _encoder(campaign, "CNN-K3", stem_layernorm=True)
    with pytest.raises(ValueError, match="odd"):
        _encoder(campaign, "CNN-K3", stem_kernel_hw=8)
