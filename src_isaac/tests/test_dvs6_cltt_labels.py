"""Owner, workspace DECISIONS §162 add. 1: colour DVS (5x5 blur, threshold 24) for the four §162 CLTT-Ref labels.

"3DCNN-CLTT-Ref-DVS6", "3DCNN-CLTT-Ref-T1-DVS6", "ViViT-CLTT-Ref-DVS6", "ViViT-CLTT-Ref-T2-DVS6": each is its twin
+ pre=["dvs_polarity"] + channels_per_frame 6. The trace test runs the REAL wrapper chain (dvs_polarity innermost,
then framestack 2) and checks the 12-channel observation against an independent event computation:
    obs_0 = [0, 0]   obs_1 = [0, dvs(1,0)]   obs_t = [dvs(t-1,t-2), dvs(t,t-1)] for t >= 2,
and a done env restarts at [0, 0]. Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU).
"""

from __future__ import annotations

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity
from nett_skrl.body.wrappers.framestack import FrameStack
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128    # the default eye
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
LABELS = {  # label: (twin, encoder params at 128x80 with 12 input channels)
    "3DCNN-CLTT-Ref-DVS6": ("3DCNN-CLTT-Ref", 697_709),
    "3DCNN-CLTT-Ref-T1-DVS6": ("3DCNN-CLTT-Ref-T1", 714_413),
    "ViViT-CLTT-Ref-DVS6": ("ViViT-CLTT-Ref", 804_608),
    "ViViT-CLTT-Ref-T2-DVS6": ("ViViT-CLTT-Ref-T2", 1_025_504),
}
ROW_ENV = {"NETT_DVS_COLOR": "1", "NETT_AUX_CLTT_CHANNELS_PER_FRAME": "6", "NETT_DVS_THRESHOLD": "24",
           "NETT_DVS_BLUR_KERNEL": "5"}


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


@pytest.fixture()
def row_env(monkeypatch):
    for k in ("NETT_DVS_BLUR", "NETT_DVS_WINDOW", "NETT_DVS_WINDOW_TAU", "NETT_DECISION_LOOP",
              "NETT_FRAMESTACK_SHUFFLE", "NETT_FRAMESTACK_N"):
        monkeypatch.delenv(k, raising=False)
    for k, v in ROW_ENV.items():
        monkeypatch.setenv(k, v)


def _build(spec, channels, seed=0):
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(seed)
    return encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (channels, H, W), np.uint8), **kwargs)


def _n(module):
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------- labels

@pytest.mark.parametrize("label", list(LABELS))
def test_label_is_its_twin_plus_dvs_and_six_channels_per_frame(campaign, label):
    twin = campaign.MODELS[LABELS[label][0]]
    spec = campaign.MODELS[label]
    assert spec["cfg"] == {**twin["cfg"], "channels_per_frame": 6}
    assert spec["pre"] == ["dvs_polarity"] and "pre" not in twin
    assert {k: v for k, v in spec.items() if k not in ("cfg", "pre")} == {k: v for k, v in twin.items() if k != "cfg"}
    assert spec["aux"] == "cltt_ref" and spec["framestack"] is True
    assert campaign.segmentation_wrappers(spec) == ["dvs_polarity", "framestack"]


@pytest.mark.parametrize("label", list(LABELS))
def test_label_builds_at_twelve_channels_and_refuses_rgb_while_the_twin_refuses_events(campaign, label):
    enc = _build(campaign.MODELS[label], 12)
    assert enc.base_channels == 6 and enc.num_frames == 2
    assert _n(enc) == LABELS[label][1]
    x = torch.randint(0, 256, (2, 12, H, W), dtype=torch.uint8).float()
    assert enc(x).shape == (2, 512)
    with pytest.raises(ValueError, match="channels_per_frame=6"):
        _build(campaign.MODELS[label], 6)
    with pytest.raises(ValueError, match="not a plausible image depth"):
        _build(campaign.MODELS[LABELS[label][0]], 12)


@pytest.mark.parametrize("label", list(LABELS))
def test_cltt_ref_views_take_the_current_six_channels(campaign, label, row_env):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss

    enc = _build(campaign.MODELS[label], 12)
    aux = CLTTReferenceAuxLoss(enc)
    x = torch.rand(2, 12, H, W)
    (v,) = aux._make_views([x], enc)
    assert torch.equal(v[:, :6], x[:, 6:]) and torch.equal(v[:, 6:], x[:, 6:])
    z = aux.head(enc.encode_prepared(v))
    assert z.shape[0] == 2 and torch.isfinite(z).all()


# ---------------------------------------------------------------- the trace, through the real wrappers

class _ScriptedEnv(gym.Env):
    """Batched HWC RGB frames; `dones[i]` is the done mask returned by step i (1-based)."""

    def __init__(self, frames, dones):
        self.frames, self.dones, self.i = frames, dones, 0
        n, h, w, _ = frames[0].shape
        self.observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, (n, h, w, 3), np.uint8)})
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, **kw):
        self.i = 0
        return {"policy": torch.as_tensor(self.frames[0])}, {}

    def step(self, action):
        self.i += 1
        d = np.asarray(self.dones.get(self.i, [False] * self.frames[0].shape[0]))
        return {"policy": torch.as_tensor(self.frames[self.i])}, np.zeros(len(d)), d, np.zeros_like(d), {}


def _expected_events(cur, prev, thr=24.0, side=5):
    """Independent colour DVS: per-plane side x side binomial blur (replicate pad), then
    (R+, R-, G+, G-, B+, B-) at +-thr. HWC uint8 in, HWC uint8 (N, h, w, 6) out."""
    k1 = torch.tensor([float(math.comb(side - 1, i)) for i in range(side)], dtype=torch.float64)
    k = (torch.outer(k1, k1) / torch.outer(k1, k1).sum()).view(1, 1, side, side)

    def blur(f):
        x = torch.as_tensor(f, dtype=torch.float64).permute(0, 3, 1, 2)          # (N, 3, h, w)
        n, c, h, w = x.shape
        x = F.pad(x.reshape(n * c, 1, h, w), (side // 2,) * 4, mode="replicate")
        return F.conv2d(x, k).reshape(n, c, h, w)

    d = blur(cur) - blur(prev)
    planes = []
    for c in range(3):
        planes += [(d[:, c] >= thr), (d[:, c] <= -thr)]
    return (torch.stack(planes, dim=-1).numpy().astype(np.uint8) * 255)


def test_trace_colour_events_six_per_frame_and_twelve_stacked_t_major(row_env):
    g = np.random.default_rng(0)
    n, h, w = 2, 12, 16
    frames = [g.integers(0, 256, (n, h, w, 3), dtype=np.uint8) for _ in range(5)]
    env = FrameStack(DVSPolarity(_ScriptedEnv(frames, {3: [False, True]})), n_stack=2)
    assert env.observation_space["policy"].shape == (n, h, w, 12)
    inner = env.env
    assert inner.color and inner.n_out == 6 and inner.threshold == 24.0 and inner.blur_kernel == 5

    obs = [env.reset()[0]["policy"]]
    for _ in range(4):
        obs.append(env.step(0)[0]["policy"])
    obs = [np.asarray(o) for o in obs]
    e = {t: _expected_events(frames[t], frames[t - 1]) for t in range(1, 5)}
    zero = np.zeros((n, h, w, 6), np.uint8)
    assert all(o.shape == (n, h, w, 12) for o in obs)
    assert e[2].any() and e[2].min() == 0                                  # the check can see events and non-events

    np.testing.assert_array_equal(obs[0], np.concatenate([zero, zero], -1))     # reset: no previous frame
    np.testing.assert_array_equal(obs[1], np.concatenate([zero, e[1]], -1))     # the stack still holds the reset zeros
    np.testing.assert_array_equal(obs[2], np.concatenate([e[1], e[2]], -1))     # [dvs(t-1,t-2), dvs(t,t-1)]
    np.testing.assert_array_equal(obs[3][0], np.concatenate([e[2], e[3]], -1)[0])
    np.testing.assert_array_equal(obs[3][1], np.zeros((h, w, 12), np.uint8))    # done: events zeroed AND stack scrubbed
    np.testing.assert_array_equal(obs[4][0], np.concatenate([e[3], e[4]], -1)[0])
    np.testing.assert_array_equal(obs[4][1], np.concatenate([zero, e[4]], -1)[1])  # restarts from the new first frame
