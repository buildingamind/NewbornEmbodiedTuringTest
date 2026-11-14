"""D512P-T2 full rows (owner 2026-11-14 ~18:10Z, first-hand; workspace FINDINGS add. 426a).

"when we run the new 3DCNN runs, we will also need to run the full rows ... for the CLTT, DVS, and CLTT+DVS runs across
experiments as well as the Fovea, Fovea+CLTT, Fovea+DVS, and Fovea+DVS+CLTT runs across binding." Each label is D512P-T2
(+CLTT-Ref) plus exactly the fields of the BL2 "3DCNN" cell it replaces. Pinned here, built on CPU at 128x80:
  (a) specs: DVS6 = pre dvs_polarity + channels_per_frame 6; Fovea = pre fovea_mask; Fovea+DVS6 = pre (fovea_mask,
      dvs_polarity) + channels_per_frame 6; +CLTT-Ref = the attached cltt_ref;
  (b) the body chain equals the replaced 3DCNN cell's (fovea / events innermost, framestack last), and the real Body
      gives 12 channels on the DVS6 labels at NETT_DVS_COLOR=1;
  (c) DVS6 stem Conv3d (6 -> 32, (2, 8, 8)); counts; a 6-channel stack refuses, the 1/3/4 check is unchanged for D512P-T2;
  (d) TWIN EQUIVALENCE: D512P-T2+DVS6 with the previous-frame slice zeroed == CNN-D512P+DVS on the current 6 channels;
  (e) CLTT-Ref on the three +CLTT-Ref labels is finite with stem gradients (NETT_AUX_CLTT_CHANNELS_PER_FRAME=6 on DVS6);
  (f) channels_per_frame without temporal_frames refuses (NatureCNN builds nothing from it).
Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.body import Body
from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity
from nett_skrl.body.wrappers.framestack import FrameStack
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128
# label: (base, replaced BL2 3DCNN cell's label, input channels, encoder params at 128x80)
ROWS = {
    "D512P-T2+DVS6": ("D512P-T2", "3DCNN-DVS6", 12, 684_704),
    "D512P-T2+CLTT-Ref-DVS6": ("D512P-T2+CLTT-Ref", "3DCNN-CLTT-Ref-DVS6", 12, 684_704),
    "D512P-T2+Fovea": ("D512P-T2", "3DCNN+Fovea", 6, 672_416),
    "D512P-T2+CLTT-Ref+Fovea": ("D512P-T2+CLTT-Ref", "3DCNN-CLTT-Ref+Fovea", 6, 672_416),
    "D512P-T2+Fovea+DVS6": ("D512P-T2", None, 12, 684_704),
    "D512P-T2+Fovea+DVS6+CLTT-Ref": ("D512P-T2+CLTT-Ref", None, 12, 684_704),
}
ROW_ENV = {"NETT_DVS_COLOR": "1", "NETT_DVS_THRESHOLD": "4", "NETT_DVS_BLUR": "0"}   # the BL2 DVS6 arms' knobs


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train

    return campaign_train


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("NETT_AMP", "NETT_INIT_SCHEME", "NETT_CONV3D_ORTHO_INIT", "NETT_AUX_BATCH", "NETT_AUX_CLTT_CHANNELS_PER_FRAME",
              "NETT_AUX_CLTT_REF_OFFSETS", "NETT_CLTT_REF_DEDUP", "NETT_DVS_WINDOW", "NETT_DVS_WINDOW_TAU",
              "NETT_DECISION_LOOP", "NETT_FRAMESTACK_SHUFFLE", "NETT_FRAMESTACK_N", "NETT_DVS_BLUR_KERNEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in ROW_ENV.items():
        monkeypatch.setenv(k, v)


def _build(spec, channels, seed=0, **over):
    kw = EncoderCfg(**{**spec["cfg"], **over}).as_kwargs()
    kw.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (channels, H, W), np.uint8), **kw)


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_specs_are_the_base_plus_the_replaced_cells_fields(campaign):
    m = campaign.MODELS
    dvs_cfg = {**m["D512P-T2"]["cfg"], "channels_per_frame": 6}
    assert m["D512P-T2+DVS6"] == {**m["D512P-T2"], "cfg": dvs_cfg, "pre": ("dvs_polarity",)}
    assert m["D512P-T2+Fovea"] == {**m["D512P-T2"], "pre": ("fovea_mask",)}
    assert m["D512P-T2+Fovea+DVS6"] == {**m["D512P-T2"], "cfg": dvs_cfg, "pre": ("fovea_mask", "dvs_polarity")}
    assert m["D512P-T2+CLTT-Ref+Fovea"] == {**m["D512P-T2+CLTT-Ref"], "pre": ("fovea_mask",)}
    for base in ("D512P-T2+DVS6", "D512P-T2+Fovea+DVS6"):
        assert "aux" not in m[base]
    assert m["D512P-T2+CLTT-Ref-DVS6"] == {**m["D512P-T2+DVS6"], "aux": "cltt_ref", "aux_weight": 1.0}
    assert m["D512P-T2+Fovea+DVS6+CLTT-Ref"] == {**m["D512P-T2+Fovea+DVS6"], "aux": "cltt_ref", "aux_weight": 1.0}
    for label, (base, _, _, _) in ROWS.items():
        assert m[label]["encoder"] == "nature_cnn" and m[label]["framestack"] is True
        assert m[label].get("aux") == m[base].get("aux")


def test_body_chain_is_the_replaced_cells(campaign):
    sw = campaign.segmentation_wrappers
    for label, (_, cell, _, _) in ROWS.items():
        if cell is not None:
            assert sw(campaign.MODELS[label]) == sw(campaign.MODELS[cell]), label
    for label in ("D512P-T2+Fovea+DVS6", "D512P-T2+Fovea+DVS6+CLTT-Ref"):
        assert sw(campaign.MODELS[label]) == sw(campaign.MODELS["CNN-D512P+Fovea+DVS"]) + ["framestack"] \
            == ["fovea_mask", "dvs_polarity", "framestack"]


def test_real_body_gives_twelve_channels_on_dvs6(campaign):
    class _Env(gym.Env):
        metadata: dict = {}
        render_mode = None
        spec = None
        observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, (2, H, W, 3), np.uint8)})
        action_space = gym.spaces.Box(-1, 1, shape=(2, 2), dtype=np.float32)

    env = Body(campaign.segmentation_wrappers(campaign.MODELS["D512P-T2+DVS6"])).wrap(_Env())
    assert isinstance(env.env, FrameStack) and isinstance(env.env.env, DVSPolarity)
    assert env.env.env.color and env.env.env.n_out == 6
    assert tuple(env.observation_space["policy"].shape) == (2, 12, H, W)


@pytest.mark.parametrize("label", sorted(ROWS))
def test_builds_at_its_channels_with_counts(campaign, label):
    _, _, ch, n = ROWS[label]
    enc = _build(campaign.MODELS[label], ch)
    w = enc.cnn[0].conv3d.weight
    print(f"\n{label}: encoder params at 128x80, {ch} ch: {_n(enc):,}; stem {tuple(w.shape)}")
    assert _n(enc) == n and tuple(w.shape) == (32, ch // 2, 2, 8, 8)
    assert enc.cnn[0].conv3d.stride == (1, 4, 4) and enc.pool_grid == (3, 6)
    x = torch.randint(0, 256, (2, ch, H, W), dtype=torch.uint8)
    out = enc(x)
    assert out.shape == (2, 512) and torch.isfinite(out).all()
    if ch == 12:
        assert n == 672_416 + (6 - 3) * 2 * 8 * 8 * 32                    # D512P-T2 + the stem's 3 extra input planes
        with pytest.raises(ValueError, match="channels_per_frame=6"):
            _build(campaign.MODELS[label], 6)


def test_d512p_t2_dvs6_is_the_exact_twin_of_cnn_d512p_dvs(campaign):
    t2 = _build(campaign.MODELS["D512P-T2+DVS6"], 12, seed=1).eval()
    one = _build(campaign.MODELS["CNN-D512P+DVS"], 6, seed=2).eval()
    sd = {k: v for k, v in one.state_dict().items() if not k.startswith("cnn.0.")}
    w = torch.zeros_like(t2.cnn[0].conv3d.weight)
    w[:, :, -1] = one.cnn[0].weight
    sd["cnn.0.conv3d.weight"], sd["cnn.0.conv3d.bias"] = w, one.cnn[0].bias
    t2.load_state_dict(sd)
    torch.manual_seed(3)
    prev, cur = (torch.randint(0, 2, (2, 4, 6, H, W), dtype=torch.uint8) * 255)
    with torch.no_grad():
        want = one(cur)
        assert torch.allclose(t2(torch.cat([prev, cur], dim=1)), want, atol=1e-5)
        assert not torch.allclose(t2(torch.cat([cur, prev], dim=1)), want, atol=1e-3)


@pytest.mark.parametrize("label", ["D512P-T2+CLTT-Ref-DVS6", "D512P-T2+CLTT-Ref+Fovea", "D512P-T2+Fovea+DVS6+CLTT-Ref"])
def test_cltt_ref_is_finite_with_stem_gradients(campaign, monkeypatch, label):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss

    class _FakeMemory:
        def __init__(self, raw):
            self.tensors = {"observations": raw,
                            "terminated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool),
                            "truncated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool)}
            self.memory_size, self.filled, self.memory_index = raw.shape[0], True, 0

        def get_tensor_by_name(self, name):
            raise AssertionError("never transfer the whole buffer")

    ch = ROWS[label][2]
    monkeypatch.setenv("NETT_AUX_BATCH", "8")
    if ch == 12:
        monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "6")
    else:                                   # the Fovea rows set no NETT_DVS_* (the CLTT view guard reads NETT_DVS_COLOR)
        for k in ROW_ENV:
            monkeypatch.delenv(k, raising=False)
    enc = _build(campaign.MODELS[label], ch)
    aux = CLTTReferenceAuxLoss(enc)
    torch.manual_seed(1)
    aux.attach_memory(_FakeMemory(torch.randint(0, 256, (24, 2, ch * H * W)).float()))
    value = aux.compute(enc, torch.empty(0))
    assert torch.isfinite(value) and value.requires_grad
    value.backward()
    g = enc.cnn[0].conv3d.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_channels_per_frame_needs_temporal_frames(campaign):
    with pytest.raises(ValueError, match="only read with temporal_frames"):
        _build(campaign.MODELS["CNN-D512P"], 6, channels_per_frame=6)
    with pytest.raises(ValueError, match="not a plausible image depth"):
        _build(campaign.MODELS["D512P-T2"], 12)
