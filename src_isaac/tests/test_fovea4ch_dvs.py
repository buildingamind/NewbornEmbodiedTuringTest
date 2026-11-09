"""Owner 2026-11-09, "Option 1" (presenter relay): the FOVEA-4CH eye feeding colour DVS on the attached-CLTT-Ref ViT-P8.

"ViT-P8-666K+Fovea4ch+DVS+CLTT-Ref" = ViT-P8-666K + pre=("fovea_4ch", "dvs_polarity") + attached cltt_ref. Directly on
fovea_4ch, dvs_polarity at NETT_DVS_COLOR=1 runs its per-plane rule on all four planes -> (R+, R-, G+, G-, B+, B-, L+, L-).
Pinned here, through the real Body chain:
  * 8 channels; outside the fovea the colour pairs never fire (mask x RGB is 0 there) while the luma pair does;
  * inside the fovea a colour change fires its colour pair;
  * the 8-plane mode exists ONLY on fovea_4ch: a 4-channel frame from anything else is refused, as before;
  * the stated per-frame channel count must match (8 here, 6 on RGB);
  * CLTT-Ref at cpf=8: a view is the frame itself and the loss is finite with gradients; colour DVS without the knob is refused.
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
from nett_skrl.body.wrappers.fovea_4ch import Fovea4ch
from nett_skrl.body.wrappers.fovea_mask import soft_mask
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128    # the series eye
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
LABEL = "ViT-P8-666K+Fovea4ch+DVS+CLTT-Ref"
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


# ---------------------------------------------------------------- label

def test_spec_is_the_twin_plus_fovea4ch_then_dvs_plus_attached_cltt(campaign):
    m = campaign.MODELS
    assert m[LABEL] == {**m["ViT-P8-666K"], "pre": ("fovea_4ch", "dvs_polarity"), "aux": "cltt_ref", "aux_weight": 1.0}
    assert {k: v for k, v in m[LABEL].items() if k != "pre"} == m["ViT-P8-666K+CLTT-Ref"]
    assert campaign.segmentation_wrappers(m[LABEL]) == ["fovea_4ch", "dvs_polarity"] and m[LABEL]["framestack"] is False


def test_encoder_builds_at_eight_channels(campaign):
    enc, n = _count(campaign.MODELS[LABEL], 8)
    # only the patch embedding grows: 2 more input channels x 8x8 patch x embed 148 over the 6-channel ViT-P8 DVS6
    assert n == _count(campaign.MODELS["ViT-P8-666K+DVS"], 6)[1] + 2 * 8 * 8 * 148 == 707_064
    assert enc(torch.randint(0, 256, (2, 8, H, W)).float()).shape == (2, 512)


# ---------------------------------------------------------------- the real body chain

def _frames():
    """Frame 0 a saturated colour field; frame 1 brighter everywhere (a luminance step); frame 2 = frame 1 with only
    the RED channel raised inside a small centre patch."""
    f0 = torch.zeros(2, H, W, 3, dtype=torch.uint8)
    f0[..., 0], f0[..., 1], f0[..., 2] = 120, 60, 90
    f1 = (f0.int() + 40).to(torch.uint8)
    f2 = f1.clone()
    cy, cx = H // 2, W // 2
    f2[:, cy - 1:cy + 1, cx - 1:cx + 1, 0] = 250
    return [f0, f1, f2]


def test_body_chain_is_fovea4ch_then_eight_channel_events(campaign, row_env):
    body = Body(campaign.segmentation_wrappers(campaign.MODELS[LABEL]))
    env = body.wrap(_ScriptedEnv(_frames()))
    inner = env.env
    assert isinstance(inner, DVSPolarity) and isinstance(inner.env, Fovea4ch) and inner.n_out == 8
    assert tuple(env.observation_space["policy"].shape) == (2, 8, H, W)
    obs, _ = env.reset()
    assert torch.as_tensor(obs["policy"]).shape == (2, 8, H, W) and not torch.as_tensor(obs["policy"]).any()

    out = soft_mask(H, W, 3.0, 1.0) == 0                               # where mask x RGB is exactly 0
    ev1 = torch.as_tensor(env.step(None)[0]["policy"])                 # luminance step
    assert not ev1[:, 0:6][:, :, out].any()                            # no colour events outside the fovea ...
    assert bool((ev1[:, 6][:, out] == 255).all()) and not ev1[:, 7].any()   # ... the luma pair fires L+ over the frame
    cy, cx = H // 2, W // 2
    assert bool((ev1[:, 0:6:2, cy, cx] == 255).all())                  # inside the fovea R+/G+/B+ fire too

    ev2 = torch.as_tensor(env.step(None)[0]["policy"])                 # red-only change at the centre patch
    patch = torch.zeros(H, W, dtype=torch.bool)
    patch[cy - 1:cy + 1, cx - 1:cx + 1] = True
    assert bool((ev2[:, 0][:, patch] == 255).all()) and not ev2[:, 0][:, ~patch].any()   # R+ on the patch only
    assert not ev2[:, 1:6].any()                                       # no other colour event
    assert bool((ev2[:, 6][:, patch] == 255).all()) and not ev2[:, 6][:, ~patch].any() and not ev2[:, 7].any()
    # (L+ fires on the patch: 0.299 x 90 = 27 >= threshold 4)


def test_the_eight_plane_mode_exists_only_on_fovea4ch(row_env):
    f = torch.zeros(1, H, W, 4, dtype=torch.uint8)
    plain4 = DVSPolarity(_ScriptedEnv([f, f]))                         # 4 channels NOT from fovea_4ch
    assert plain4.n_out == 6
    with pytest.raises(ValueError, match="BEFORE"):
        plain4.reset()
    rgb = torch.zeros(1, H, W, 3, dtype=torch.uint8)
    assert DVSPolarity(_ScriptedEnv([rgb, rgb])).n_out == 6            # RGB unchanged
    assert DVSPolarity(Fovea4ch(_ScriptedEnv([rgb, rgb]))).n_out == 8


def test_luma_dvs_on_fovea4ch_is_still_refused(monkeypatch, row_env):
    monkeypatch.delenv("NETT_DVS_COLOR")
    rgb = torch.zeros(1, H, W, 3, dtype=torch.uint8)
    env = DVSPolarity(Fovea4ch(_ScriptedEnv([rgb, rgb])))
    assert env.n_out == 2
    with pytest.raises(ValueError, match="BEFORE"):
        env.reset()


@pytest.mark.parametrize("cpf,fovea4ch,ok", [("8", True, True), ("6", True, False), ("6", False, True), ("8", False, False)])
def test_stated_channels_per_frame_must_match(monkeypatch, row_env, cpf, fovea4ch, ok):
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", cpf)
    rgb = torch.zeros(1, H, W, 3, dtype=torch.uint8)
    inner = Fovea4ch(_ScriptedEnv([rgb, rgb])) if fovea4ch else _ScriptedEnv([rgb, rgb])
    if ok:
        DVSPolarity(inner)
    else:
        with pytest.raises(ValueError, match="NETT_AUX_CLTT_CHANNELS_PER_FRAME"):
            DVSPolarity(inner)


# ---------------------------------------------------------------- CLTT-Ref on 8-channel frames

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
    enc, _ = _count(campaign.MODELS[LABEL], 8)
    aux = CLTTReferenceAuxLoss(enc)
    raw = (torch.rand(24, 2, 8 * H * W) > 0.97).float() * 255.0       # sparse binary event frames
    aux.attach_memory(_FakeMemory(raw))
    return enc, aux


def test_cltt_views_are_the_eight_channel_frame_itself(campaign, row_env, monkeypatch):
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "8")
    enc, aux = _cltt(campaign, monkeypatch)
    x = torch.rand(3, 8, H, W)
    (v,) = aux._make_views([x], enc)
    assert torch.equal(v, x)
    value = aux.compute(enc, torch.empty(0))
    assert aux.num_frames == 1 and torch.isfinite(value) and value.requires_grad


def test_cltt_refuses_colour_dvs_without_the_channel_knob(campaign, row_env, monkeypatch):
    enc, aux = _cltt(campaign, monkeypatch)
    with pytest.raises(ValueError, match="NETT_AUX_CLTT_CHANNELS_PER_FRAME"):
        aux.compute(enc, torch.empty(0))
