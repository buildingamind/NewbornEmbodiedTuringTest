"""ViViT-P8-T2 Fovea+DVS6 full rows for binding (owner 2026-11-15 ~03:30Z; workspace FINDINGS add. 430).

"Schedule ViViT Fovea+DVS and ViViT Fovea+DVS+CLTT full rows for binding." Each label is "ViViT-P8-T2+DVS6" (channels_per_frame
6) with the fovea added innermost -- pre (fovea_mask, dvs_polarity), as "ViT-P8-666K+Fovea+DVS" / "D512P-T2+Fovea+DVS6" -- and
the +CLTT-Ref label adds the attached cltt_ref. Pinned here, built on CPU at 128x80:
  (a) specs: the base plus exactly that `pre`; +CLTT-Ref = that plus the aux; ViViT-P8-T2+DVS6 itself unchanged;
  (b) the body chain is fovea_mask -> dvs_polarity -> framestack (= D512P-T2+Fovea+DVS6's; ViT-P8-666K+Fovea+DVS's plus the
      stack), and the real Body gives 12 channels at NETT_DVS_COLOR=1;
  (c) the encoder is ViViT-P8-T2+DVS6's (fovea is a body wrapper: no parameters), 744,952 at 128x80, finite on a 12-channel
      2-frame stack;
  (d) the fovea reaches the events: a red-only change in the periphery fires R+ alone on the DVS6 twin but all three ON
      planes identically here (grey surround: only luminance survives), while a red-only change at the centre fires R+ alone;
  (e) CLTT-Ref on the +CLTT-Ref label is finite with patch-embedding gradients (NETT_AUX_CLTT_CHANNELS_PER_FRAME=6).
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
from nett_skrl.body.wrappers.framestack import FrameStack
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128    # the series eye
BASE, LABEL, CLTT = "ViViT-P8-T2+DVS6", "ViViT-P8-T2+Fovea+DVS6", "ViViT-P8-T2+Fovea+DVS6+CLTT-Ref"
LABELS = (LABEL, CLTT)
N_PARAMS = 744_952    # ViViT-P8-T2+DVS6 at 128x80, 12 channels (tests/test_bl2_fill_vivit_dvs6.py)
ROW_ENV = {"NETT_DVS_COLOR": "1", "NETT_DVS_THRESHOLD": "4", "NETT_DVS_BLUR": "0"}   # the BL2 DVS6 arms' knobs


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train

    return campaign_train


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("NETT_AMP", "NETT_INIT_SCHEME", "NETT_AUX_BATCH", "NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_AUX_CLTT_REF_OFFSETS",
              "NETT_CLTT_REF_DEDUP", "NETT_DVS_WINDOW", "NETT_DVS_WINDOW_TAU", "NETT_DVS_BLUR_KERNEL", "NETT_DVS_MOVE_GAIN",
              "NETT_DVS_MOTION_COMP", "NETT_DVS_GATE_K", "NETT_DVS_GLOBAL_SUB", "NETT_DECISION_LOOP", "NETT_FRAMESTACK_SHUFFLE",
              "NETT_FRAMESTACK_N", "NETT_FOVEAM_RADIUS", "NETT_FOVEAM_MASK_SIGMA"):
        monkeypatch.delenv(k, raising=False)
    for k, v in ROW_ENV.items():
        monkeypatch.setenv(k, v)


def _build(spec, channels, seed=0):
    kw = EncoderCfg(**spec["cfg"]).as_kwargs()
    kw.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (channels, H, W), np.uint8), **kw)


def _n(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- (a) specs

def test_specs_are_the_dvs6_twin_plus_the_fovea(campaign):
    m = campaign.MODELS
    assert m[LABEL] == {**m[BASE], "pre": ("fovea_mask", "dvs_polarity")}
    assert m[CLTT] == {**m[LABEL], "aux": "cltt_ref", "aux_weight": 1.0}
    assert "aux" not in m[LABEL]
    # the twin is untouched (its `pre` stays the list it was written with)
    assert m[BASE] == {**m["ViViT-P8-T2"], "cfg": {**m["ViViT-P8-T2"]["cfg"], "channels_per_frame": 6}, "pre": ["dvs_polarity"]}
    assert m["ViViT-P8-T2+CLTT-Ref-DVS6"] == {**m[BASE], "aux": "cltt_ref", "aux_weight": 1.0}
    # the same fovea-then-events `pre` as every other Fovea+DVS cell
    for other in ("ViT-P8-666K+Fovea+DVS", "CNN-D512P+Fovea+DVS", "D512P-T2+Fovea+DVS6"):
        assert m[other]["pre"] == m[LABEL]["pre"] == m[CLTT]["pre"], other
    for label in LABELS:
        assert m[label]["encoder"] == "compact_vivit" and m[label]["framestack"] is True
        assert m[label]["cfg"] == m[BASE]["cfg"] and m[label]["cfg"]["channels_per_frame"] == 6


# ---------------------------------------------------------------- (b) the body chain

def test_body_chain_is_fovea_then_events_then_stack(campaign):
    sw, m = campaign.segmentation_wrappers, campaign.MODELS
    for label in LABELS:
        assert sw(m[label]) == ["fovea_mask", "dvs_polarity", "framestack"]
        assert sw(m[label]) == sw(m["D512P-T2+Fovea+DVS6"]) == sw(m["ViT-P8-666K+Fovea+DVS"]) + ["framestack"]
        assert sw(m[label]) == ["fovea_mask"] + sw(m[BASE])
    assert sw(m["ViViT-P8-T2+CLTT-Ref+Fovea"]) == ["fovea_mask", "framestack"]


class _Env(gym.Env):
    metadata: dict = {}
    render_mode = None
    spec = None
    observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, (2, H, W, 3), np.uint8)})
    action_space = gym.spaces.Box(-1, 1, shape=(2, 2), dtype=np.float32)


@pytest.mark.parametrize("label", LABELS)
def test_real_body_gives_twelve_channels(campaign, label):
    env = Body(campaign.segmentation_wrappers(campaign.MODELS[label])).wrap(_Env())
    assert isinstance(env.env, FrameStack) and isinstance(env.env.env, DVSPolarity)
    assert isinstance(env.env.env.env, FoveaMask)
    assert env.env.env.color and env.env.env.n_out == 6
    assert tuple(env.observation_space["policy"].shape) == (2, 12, H, W)


# ---------------------------------------------------------------- (c) the encoder

@pytest.mark.parametrize("label", LABELS)
def test_encoder_is_the_dvs6_twins(campaign, label):
    enc = _build(campaign.MODELS[label], 12)
    n = _n(enc)
    print(f"\n{label}: encoder params at 128x80, 12 ch: {n:,}")
    assert enc.base_channels == 6 and enc.num_frames == 2
    assert n == _n(_build(campaign.MODELS[BASE], 12)) == N_PARAMS
    out = enc(torch.randint(0, 256, (2, 12, H, W), dtype=torch.uint8).float())
    assert out.shape == (2, 512) and torch.isfinite(out).all()
    with pytest.raises(ValueError, match="channels_per_frame=6"):
        _build(campaign.MODELS[label], 6)


# ---------------------------------------------------------------- (d) the fovea reaches the events

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


CY, CX = H // 2, W // 2
_CENTRE = torch.zeros(H, W, dtype=torch.bool)
_CENTRE[CY - 1:CY + 1, CX - 1:CX + 1] = True
_EDGE = torch.zeros(H, W, dtype=torch.bool)
_EDGE[4:12, 8:20] = True                                               # far outside the radius-3 disk


def _frames():
    """Frame 0 a saturated colour field; frame 1 = frame 0 with only RED raised on a peripheral patch; frame 2 = frame 1
    with only RED raised on a small centre patch."""
    f0 = torch.zeros(2, H, W, 3, dtype=torch.uint8)
    f0[..., 0], f0[..., 1], f0[..., 2] = 120, 60, 90
    f1 = f0.clone()
    f1[:, 4:12, 8:20, 0] = 250
    f2 = f1.clone()
    f2[:, CY - 1:CY + 1, CX - 1:CX + 1, 0] = 250
    return [f0, f1, f2]


def _run(campaign, label):
    env = Body(campaign.segmentation_wrappers(campaign.MODELS[label])).wrap(_ScriptedEnv(_frames()))
    obs, _ = env.reset()
    assert torch.as_tensor(obs["policy"]).shape == (2, 12, H, W) and not torch.as_tensor(obs["policy"]).any()
    cur = slice(6, 12)                                                 # the stack is oldest first: channels 6..11 = now
    return env, torch.as_tensor(env.step(None)[0]["policy"])[:, cur], torch.as_tensor(env.step(None)[0]["policy"])[:, cur]


def test_fovea_greys_the_periphery_before_the_events(campaign):
    m = soft_mask(H, W, 3.0, 1.0)
    out = m < 1e-3
    assert bool(out[_EDGE].all()) and not bool(out[_CENTRE].any())

    # the DVS6 twin (no fovea): the peripheral red-only change fires R+ alone
    _, ev1, ev2 = _run(campaign, BASE)
    assert bool((ev1[:, 0][:, _EDGE] == 255).all()) and not ev1[:, 0][:, ~_EDGE].any() and not ev1[:, 1:].any()

    # what the DVS wrapper receives (the chain's innermost wrapper alone): grey outside the disk, colour at the centre
    seen = torch.as_tensor(Body(["fovea_mask"]).wrap(_ScriptedEnv(_frames())).reset()[0]["policy"]).int()
    assert seen.shape == (2, 3, H, W)                                 # channels first
    r, g, b = (seen[:, c][:, out] for c in range(3))
    assert torch.equal(r, g) and torch.equal(g, b)
    assert bool((seen[:, :, CY, CX] != seen[:, :1, CY, CX]).any())

    for label in LABELS:
        env, ev1, ev2 = _run(campaign, label)
        assert isinstance(env.env.env.env, FoveaMask)
        # periphery: the same red-only change now fires every ON plane identically (only its luminance survives)
        on, off = ev1[:, 0::2], ev1[:, 1::2]
        assert bool((on[:, :, _EDGE] == 255).all()) and not on[:, :, ~_EDGE].any() and not off.any(), label
        assert torch.equal(on[:, 0], on[:, 1]) and torch.equal(on[:, 1], on[:, 2])
        # centre: a red-only change keeps its colour (R+ alone, on the patch only)
        assert bool((ev2[:, 0][:, _CENTRE] == 255).all()) and not ev2[:, 0][:, ~_CENTRE].any(), label
        assert not ev2[:, 1:].any(), label
        # outside the disk the three ON (and the three OFF) planes always agree
        for ev in (ev1, ev2):
            for p in (0, 1):
                assert torch.equal(ev[:, p][:, out], ev[:, p + 2][:, out]) and torch.equal(ev[:, p][:, out], ev[:, p + 4][:, out])


# ---------------------------------------------------------------- (e) CLTT-Ref

class _FakeMemory:
    def __init__(self, raw):
        self.tensors = {"observations": raw,
                        "terminated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool),
                        "truncated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool)}
        self.memory_size, self.filled, self.memory_index = raw.shape[0], True, 0

    def get_tensor_by_name(self, name):
        raise AssertionError("never transfer the whole buffer")


def test_cltt_ref_is_finite_with_patch_embedding_gradients(campaign, monkeypatch):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss

    monkeypatch.setenv("NETT_AUX_BATCH", "8")
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "6")
    enc = _build(campaign.MODELS[CLTT], 12)
    aux = CLTTReferenceAuxLoss(enc)
    torch.manual_seed(1)
    aux.attach_memory(_FakeMemory((torch.rand(24, 2, 12 * H * W) > 0.97).float() * 255.0))   # sparse binary event frames
    x = torch.rand(3, 12, H, W)
    (v,) = aux._make_views([x], enc)
    assert torch.equal(v, torch.cat([x[:, 6:], x[:, 6:]], dim=1))     # the current 6-channel frame, repeated in the stack
    value = aux.compute(enc, torch.empty(0))
    assert torch.isfinite(value) and value.requires_grad
    value.backward()
    g = enc.patch_embed.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_cltt_ref_refuses_colour_dvs_without_the_channel_knob(campaign, monkeypatch):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss

    monkeypatch.setenv("NETT_AUX_BATCH", "8")
    enc = _build(campaign.MODELS[CLTT], 12)
    aux = CLTTReferenceAuxLoss(enc)
    aux.attach_memory(_FakeMemory((torch.rand(24, 2, 12 * H * W) > 0.97).float() * 255.0))
    with pytest.raises(ValueError, match="NETT_AUX_CLTT_CHANNELS_PER_FRAME=6"):
        aux.compute(enc, torch.empty(0))
