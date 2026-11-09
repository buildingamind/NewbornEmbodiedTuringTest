"""Owner 2026-11-09 (presenter relay), BL2 full rows: the colour-DVS + attached CLTT-Ref labels and the D512P fovea + DVS pair.

  CNN-D512P+DVS+CLTT-Ref / ViT-P8-666K+DVS+CLTT-Ref   parsing (i), single frame: the "+DVS" twins + attached cltt_ref
  ViViT-P8-T2+CLTT-Ref-DVS6                           parsing (ii), video (cherry-picked c407ed8; with 3DCNN-CLTT-Ref-DVS6)
  CNN-D512P+Fovea+DVS / CNN-D512P+Fovea+DVS+CLTT-Ref  binding: the D512P side of ViT-P8-666K+Fovea+DVS(+CLTT-Ref)

Pinned here:
  * each spec is its twin plus exactly the stated fields;
  * the D512P fovea + DVS chain is fovea_mask -> dvs_polarity, 6 channels per frame at NETT_DVS_COLOR=1, and its encoder is
    the CNN-D512P+DVS one;
  * CLTT-Ref at NETT_AUX_CLTT_CHANNELS_PER_FRAME=6: finite loss with nonzero encoder gradients on every CLTT label here (a
    single 6-channel frame is its own view; the 12-channel ViViT stack views its current frame), and the single-frame
    labels refuse colour DVS without the knob.
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
from nett_skrl.body.wrappers.fovea_mask import FoveaMask
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128    # the series eye
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
ROW_ENV = {"NETT_DVS_COLOR": "1", "NETT_DVS_THRESHOLD": "4", "NETT_DVS_BLUR": "0"}   # the BL2 DVS6 arms' knobs
# CLTT label -> channels the encoder sees
CLTT = {"CNN-D512P+DVS+CLTT-Ref": 6, "ViT-P8-666K+DVS+CLTT-Ref": 6, "CNN-D512P+Fovea+DVS+CLTT-Ref": 6,
        "ViViT-P8-T2+CLTT-Ref-DVS6": 12}


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


@pytest.fixture(autouse=True)
def row_env(monkeypatch):
    for k in ("NETT_DVS_BLUR_KERNEL", "NETT_DVS_WINDOW", "NETT_DVS_WINDOW_TAU", "NETT_DECISION_LOOP", "NETT_DVS_MOVE_GAIN",
              "NETT_DVS_MOTION_COMP", "NETT_DVS_GATE_K", "NETT_DVS_GLOBAL_SUB", "NETT_FOVEAM_RADIUS", "NETT_FOVEAM_MASK_SIGMA",
              "NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_AUX_BATCH", "NETT_AUX_CLTT_REF_OFFSETS", "NETT_CLTT_REF_DEDUP"):
        monkeypatch.delenv(k, raising=False)
    for k, v in ROW_ENV.items():
        monkeypatch.setenv(k, v)


def _encoder(spec, channels):
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(0)
    return encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (channels, H, W), np.uint8), **kwargs)


def _n(enc):
    return sum(p.numel() for p in enc.parameters())


# ---------------------------------------------------------------- labels

def test_specs(campaign):
    m = campaign.MODELS
    aux = {"aux": "cltt_ref", "aux_weight": 1.0}
    assert m["CNN-D512P+DVS+CLTT-Ref"] == {**m["CNN-D512P+DVS"], **aux}
    assert m["ViT-P8-666K+DVS+CLTT-Ref"] == {**m["ViT-P8-666K+DVS"], **aux}
    assert m["CNN-D512P+Fovea+DVS"] == {**m["CNN-D512P"], "pre": ("fovea_mask", "dvs_polarity")}
    assert m["CNN-D512P+Fovea+DVS+CLTT-Ref"] == {**m["CNN-D512P+Fovea+DVS"], **aux}
    # the D512P pair mirrors the ViT-P8 one field for field
    for d, v in (("CNN-D512P+Fovea+DVS", "ViT-P8-666K+Fovea+DVS"), ("CNN-D512P+Fovea+DVS+CLTT-Ref", "ViT-P8-666K+Fovea+DVS+CLTT-Ref")):
        assert {k: x for k, x in m[d].items() if k not in ("encoder", "cfg")} == {k: x for k, x in m[v].items() if k not in ("encoder", "cfg")}
    for label in ("CNN-D512P+DVS+CLTT-Ref", "ViT-P8-666K+DVS+CLTT-Ref"):
        assert campaign.segmentation_wrappers(m[label]) == ["dvs_polarity"] and m[label]["framestack"] is False
    for label in ("CNN-D512P+Fovea+DVS", "CNN-D512P+Fovea+DVS+CLTT-Ref"):
        assert campaign.segmentation_wrappers(m[label]) == ["fovea_mask", "dvs_polarity"] and m[label]["framestack"] is False
    v = m["ViViT-P8-T2+CLTT-Ref-DVS6"]
    assert v["cfg"] == {**m["ViViT-P8-T2+CLTT-Ref"]["cfg"], "channels_per_frame": 6} and list(v["pre"]) == ["dvs_polarity"]
    assert v["aux"] == "cltt_ref" and v["framestack"] is True


@pytest.mark.parametrize("label,twin", [("CNN-D512P+DVS+CLTT-Ref", "CNN-D512P+DVS"), ("ViT-P8-666K+DVS+CLTT-Ref", "ViT-P8-666K+DVS"),
                                        ("CNN-D512P+Fovea+DVS", "CNN-D512P+DVS"), ("CNN-D512P+Fovea+DVS+CLTT-Ref", "CNN-D512P+DVS")])
def test_encoder_is_the_dvs6_twins(campaign, label, twin):
    enc = _encoder(campaign.MODELS[label], 6)
    assert _n(enc) == _n(_encoder(campaign.MODELS[twin], 6))
    assert enc(torch.randint(0, 256, (2, 6, H, W)).float()).shape == (2, 512)


# ---------------------------------------------------------------- the real D512P fovea + DVS chain

class _ScriptedEnv(gym.Env):
    """Batched HWC uint8 RGB frames, one per call; never done."""

    metadata: dict = {}
    render_mode = None
    spec = None

    def __init__(self, frames):
        self.frames, self.i = frames, 0
        self.observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, tuple(frames[0].shape), np.uint8)})
        self.action_space = gym.spaces.Box(-1, 1, shape=(frames[0].shape[0], 2), dtype=np.float32)

    def reset(self, **kw):
        self.i = 0
        return {"policy": self.frames[0].clone()}, {}

    def step(self, action):
        self.i += 1
        n = self.frames[0].shape[0]
        return {"policy": self.frames[self.i].clone()}, np.zeros(n), np.zeros(n, bool), np.zeros(n, bool), {}


def test_d512p_fovea_dvs_chain_is_six_channel_events_on_the_foveated_frame(campaign):
    f0 = torch.zeros(2, H, W, 3, dtype=torch.uint8)
    f0[..., 0], f0[..., 1], f0[..., 2] = 120, 60, 90
    f1 = f0.clone()
    cy, cx = H // 2, W // 2
    f1[:, cy - 1:cy + 1, cx - 1:cx + 1, 0] = 250                       # red-only change at the centre
    body = Body(campaign.segmentation_wrappers(campaign.MODELS["CNN-D512P+Fovea+DVS"]))
    env = body.wrap(_ScriptedEnv([f0, f1]))
    assert isinstance(env.env, DVSPolarity) and isinstance(env.env.env, FoveaMask)
    assert tuple(env.observation_space["policy"].shape) == (2, 6, H, W)
    env.reset()
    ev = torch.as_tensor(env.step(None)[0]["policy"])
    patch = torch.zeros(H, W, dtype=torch.bool)
    patch[cy - 1:cy + 1, cx - 1:cx + 1] = True
    assert bool((ev[:, 0][:, patch] == 255).all()) and not ev[:, 0][:, ~patch].any() and not ev[:, 1:].any()


# ---------------------------------------------------------------- CLTT-Ref on colour-DVS frames

class _FakeMemory:
    def __init__(self, raw):
        self.tensors = {"observations": raw,
                        "terminated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool),
                        "truncated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool)}
        self.memory_size, self.filled, self.memory_index = raw.shape[0], True, 0

    def get_tensor_by_name(self, name):
        raise AssertionError("never transfer the whole buffer")


def _cltt(campaign, monkeypatch, label):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss

    c = CLTT[label]
    monkeypatch.setenv("NETT_AUX_BATCH", "8")
    enc = _encoder(campaign.MODELS[label], c)
    aux = CLTTReferenceAuxLoss(enc)
    torch.manual_seed(1)
    aux.attach_memory(_FakeMemory((torch.rand(24, 2, c * H * W) > 0.97).float() * 255.0))   # sparse binary event frames
    return enc, aux, c


@pytest.mark.parametrize("label", CLTT)
def test_cltt_at_six_channels_per_frame(campaign, monkeypatch, label):
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "6")
    enc, aux, c = _cltt(campaign, monkeypatch, label)
    x = torch.rand(3, c, H, W)
    (v,) = aux._make_views([x], enc)
    assert torch.equal(v, x if c == 6 else torch.cat([x[:, 6:], x[:, 6:]], dim=1))   # the current frame (repeated in a stack)
    value = aux.compute(enc, torch.empty(0))
    assert torch.isfinite(value) and value.requires_grad
    value.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in enc.parameters())


@pytest.mark.parametrize("label", ["CNN-D512P+DVS+CLTT-Ref", "ViT-P8-666K+DVS+CLTT-Ref", "CNN-D512P+Fovea+DVS+CLTT-Ref"])
def test_cltt_refuses_colour_dvs_without_the_channel_knob(campaign, monkeypatch, label):
    enc, aux, _ = _cltt(campaign, monkeypatch, label)
    with pytest.raises(ValueError, match="NETT_AUX_CLTT_CHANNELS_PER_FRAME=6"):
        aux.compute(enc, torch.empty(0))
