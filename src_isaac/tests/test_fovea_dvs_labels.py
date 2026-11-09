"""Owner 2026-11-09, BL2 binding pilot (presenter relay): the original fovea feeding colour DVS on the ViT-P8 twin.

"ViT-P8-666K+Fovea+DVS" = ViT-P8-666K + pre=("fovea_mask", "dvs_polarity"); "ViT-P8-666K+Fovea+DVS+CLTT-Ref" = that + attached
CLTT-Ref. `pre` is innermost first, so the events are computed on the FOVEATED frame. Pinned here:
  * the specs are their twins plus exactly those fields, and the body chain is fovea_mask -> dvs_polarity;
  * through the real Body chain at NETT_DVS_COLOR=1 one frame is 6 channels, the encoder is the ViT-P8-666K+DVS one
    (688,120 parameters), and outside the fovea the three ON (and the three OFF) planes are identical, because the
    surround is grey -- while a colour-only change inside the disk fires one colour's pair alone;
  * CLTT-Ref on these 6-channel frames: with NETT_AUX_CLTT_CHANNELS_PER_FRAME=6 a view is the frame itself (depth 1) and
    the loss is finite with gradients; without it, colour DVS is refused (cltt_views.resolve_channels_per_frame).
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
from nett_skrl.body.wrappers.fovea_mask import FoveaMask, soft_mask
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128    # the series eye
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
LABELS = ("ViT-P8-666K+Fovea+DVS", "ViT-P8-666K+Fovea+DVS+CLTT-Ref")
ROW_ENV = {"NETT_DVS_COLOR": "1", "NETT_DVS_THRESHOLD": "4", "NETT_DVS_BLUR": "0"}   # the BL2 DVS6 arms' knobs


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


@pytest.fixture()
def row_env(monkeypatch):
    for k in ("NETT_DVS_BLUR_KERNEL", "NETT_DVS_WINDOW", "NETT_DVS_WINDOW_TAU", "NETT_DECISION_LOOP", "NETT_DVS_MOVE_GAIN",
              "NETT_DVS_MOTION_COMP", "NETT_DVS_GATE_K", "NETT_DVS_GLOBAL_SUB", "NETT_FOVEAM_RADIUS", "NETT_FOVEAM_MASK_SIGMA",
              "NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_AUX_BATCH", "NETT_AUX_CLTT_REF_OFFSETS", "NETT_CLTT_REF_DEDUP"):
        monkeypatch.delenv(k, raising=False)
    for k, v in ROW_ENV.items():
        monkeypatch.setenv(k, v)


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


def _count(spec, channels):
    cfg = {k: v for k, v in spec["cfg"].items() if k != "trainable"}
    enc = encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (channels, H, W), np.uint8), **cfg)
    return enc, sum(p.numel() for p in enc.parameters())


# ---------------------------------------------------------------- labels

def test_specs_are_the_twin_plus_fovea_then_dvs(campaign):
    m = campaign.MODELS
    assert m["ViT-P8-666K+Fovea+DVS"] == {**m["ViT-P8-666K"], "pre": ("fovea_mask", "dvs_polarity")}
    assert m["ViT-P8-666K+Fovea+DVS+CLTT-Ref"] == {**m["ViT-P8-666K+Fovea+DVS"], "aux": "cltt_ref", "aux_weight": 1.0}
    # the pilot's twins on the page differ from these by exactly one wrapper / one aux
    def bare(label):
        return {k: v for k, v in m[label].items() if k != "pre"}
    assert bare("ViT-P8-666K+Fovea+DVS") == bare("ViT-P8-666K+DVS") == bare("ViT-P8-666K+Fovea") == m["ViT-P8-666K"]
    assert m["ViT-P8-666K+DVS"]["pre"] == ("dvs_polarity",) and m["ViT-P8-666K+Fovea"]["pre"] == ("fovea_mask",)
    assert bare("ViT-P8-666K+Fovea+DVS+CLTT-Ref") == bare("ViT-P8-666K+Fovea+CLTT-Ref") == m["ViT-P8-666K+CLTT-Ref"]
    for label in LABELS:
        assert campaign.segmentation_wrappers(m[label]) == ["fovea_mask", "dvs_polarity"]
        assert m[label]["framestack"] is False


@pytest.mark.parametrize("label", LABELS)
def test_encoder_is_the_vit_dvs6_one_at_six_channels(campaign, label):
    enc, n = _count(campaign.MODELS[label], 6)
    assert n == _count(campaign.MODELS["ViT-P8-666K+DVS"], 6)[1] == 688_120
    assert enc(torch.randint(0, 256, (2, 6, H, W)).float()).shape == (2, 512)


# ---------------------------------------------------------------- the real body chain

def _frames():
    """Frame 0 a saturated colour field; frame 1 brighter everywhere (a luminance step); frame 2 = frame 1 with only
    the RED channel raised inside a small centre patch (a colour change the fovea keeps)."""
    f0 = torch.zeros(2, H, W, 3, dtype=torch.uint8)
    f0[..., 0], f0[..., 1], f0[..., 2] = 120, 60, 90
    f1 = (f0.int() + 40).to(torch.uint8)
    f2 = f1.clone()
    cy, cx = H // 2, W // 2
    f2[:, cy - 1:cy + 1, cx - 1:cx + 1, 0] = 250
    return [f0, f1, f2]


def test_body_chain_is_fovea_then_six_channel_events(campaign, row_env):
    spec = campaign.MODELS["ViT-P8-666K+Fovea+DVS"]
    body = Body(campaign.segmentation_wrappers(spec))
    env = body.wrap(_ScriptedEnv(_frames()))
    inner = env.env
    assert isinstance(inner, DVSPolarity) and isinstance(inner.env, FoveaMask)
    assert tuple(env.observation_space["policy"].shape) == (2, 6, H, W)
    obs, _ = env.reset()
    assert torch.as_tensor(obs["policy"]).shape == (2, 6, H, W) and not torch.as_tensor(obs["policy"]).any()

    m = soft_mask(H, W, 3.0, 1.0)
    out = m < 1e-3                                                     # well outside the disk
    ev1 = torch.as_tensor(env.step(None)[0]["policy"])                 # luminance step: ON everywhere, in all colours
    on, off = ev1[:, 0::2], ev1[:, 1::2]                               # (R+, G+, B+), (R-, G-, B-)
    assert bool((on[:, :, out] == 255).all()) and not off.any()
    assert torch.equal(on[:, 0], on[:, 1]) and torch.equal(on[:, 1], on[:, 2])

    ev2 = torch.as_tensor(env.step(None)[0]["policy"])                 # red-only change at the centre
    assert not ev2[:, :, out].any()                                    # the surround saw nothing
    cy, cx = H // 2, W // 2
    patch = torch.zeros(H, W, dtype=torch.bool)
    patch[cy - 1:cy + 1, cx - 1:cx + 1] = True
    assert bool((ev2[:, 0][:, patch] == 255).all()) and not ev2[:, 0][:, ~patch].any()   # R+ fires on the patch only,
    assert not ev2[:, 1:].any()                                        # and nothing else does: colour survives the fovea
    # periphery identity is the general property: outside the disk all three ON planes (and OFF planes) agree
    for ev in (ev1, ev2):
        for p in (0, 1):
            assert torch.equal(ev[:, p][:, out], ev[:, p + 2][:, out]) and torch.equal(ev[:, p][:, out], ev[:, p + 4][:, out])


# ---------------------------------------------------------------- CLTT-Ref on 6-channel foveated event frames

class _FakeMemory:
    def __init__(self, raw):
        self.tensors = {"observations": raw,
                        "terminated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool),
                        "truncated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool)}
        self.memory_size, self.filled, self.memory_index = raw.shape[0], True, 0

    def get_tensor_by_name(self, name):
        raise AssertionError("never transfer the whole buffer")


def _cltt(campaign, monkeypatch, batch=8):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss

    monkeypatch.setenv("NETT_AUX_BATCH", str(batch))
    torch.manual_seed(0)
    enc, _ = _count(campaign.MODELS["ViT-P8-666K+Fovea+DVS+CLTT-Ref"], 6)
    aux = CLTTReferenceAuxLoss(enc)
    raw = (torch.rand(24, 2, 6 * H * W) > 0.97).float() * 255.0       # sparse binary event frames
    aux.attach_memory(_FakeMemory(raw))
    return enc, aux


def test_cltt_views_are_the_six_channel_frame_itself(campaign, row_env, monkeypatch):
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "6")
    enc, aux = _cltt(campaign, monkeypatch)
    x = torch.rand(3, 6, H, W)
    (v,) = aux._make_views([x], enc)
    assert torch.equal(v, x)                                           # depth 1: the view IS the current frame
    value = aux.compute(enc, torch.empty(0))
    assert aux.num_frames == 1 and torch.isfinite(value) and value.requires_grad


def test_cltt_refuses_colour_dvs_without_the_channel_knob(campaign, row_env, monkeypatch):
    enc, aux = _cltt(campaign, monkeypatch)
    with pytest.raises(ValueError, match="NETT_AUX_CLTT_CHANNELS_PER_FRAME=6"):
        aux.compute(enc, torch.empty(0))
