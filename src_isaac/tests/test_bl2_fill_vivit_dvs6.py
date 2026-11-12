"""Owner 2026-11-12 (presenter relay; workspace FINDINGS add. 417), BL2 FILL phase 2: "ViViT-P8-T2+DVS6".

The ViT-P8 video twin on colour DVS events WITHOUT CLTT: "ViViT-P8-T2" + pre=["dvs_polarity"] + channels_per_frame 6, nothing
else -- the ViViT analogue of "3DCNN-DVS6", and "ViViT-P8-T2+CLTT-Ref-DVS6" minus its attached cltt_ref. Pinned here:
  * the spec is its twin plus exactly those fields, and the CLTT-Ref-DVS6 label is this one plus the aux;
  * the wrapper chain is dvs_polarity then framestack (as 3DCNN-DVS6), and the real Body gives 12 channels at NETT_DVS_COLOR=1;
  * the encoder builds at 12 channels (6 per frame x 2) with the same parameters as the CLTT-Ref-DVS6 encoder, refuses a
    6-channel input, and the RGB twin refuses the 12-channel event stack.
The colour-DVS + framestack observation itself is traced in tests/test_dvs6_cltt_labels.py (same wrappers).
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

H, W = 80, 128    # the series eye
LABEL, TWIN, CLTT = "ViViT-P8-T2+DVS6", "ViViT-P8-T2", "ViViT-P8-T2+CLTT-Ref-DVS6"
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
ROW_ENV = {"NETT_DVS_COLOR": "1", "NETT_DVS_THRESHOLD": "4", "NETT_DVS_BLUR": "0"}   # the BL2 DVS6 arms' knobs


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


@pytest.fixture(autouse=True)
def row_env(monkeypatch):
    for k in ("NETT_DVS_BLUR_KERNEL", "NETT_DVS_WINDOW", "NETT_DVS_WINDOW_TAU", "NETT_DECISION_LOOP", "NETT_DVS_MOVE_GAIN",
              "NETT_DVS_MOTION_COMP", "NETT_DVS_GATE_K", "NETT_DVS_GLOBAL_SUB", "NETT_FRAMESTACK_SHUFFLE", "NETT_FRAMESTACK_N",
              "NETT_AUX_CLTT_CHANNELS_PER_FRAME"):
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


def test_spec_is_the_twin_plus_dvs_and_six_channels_per_frame(campaign):
    m = campaign.MODELS
    spec, twin = m[LABEL], m[TWIN]
    assert spec["cfg"] == {**twin["cfg"], "channels_per_frame": 6}
    assert spec["pre"] == ["dvs_polarity"] and "pre" not in twin
    assert {k: v for k, v in spec.items() if k not in ("cfg", "pre")} == {k: v for k, v in twin.items() if k != "cfg"}
    assert "aux" not in spec and spec["framestack"] is True and spec["encoder"] == "compact_vivit"
    assert m[CLTT] == {**spec, "aux": "cltt_ref", "aux_weight": 1.0}


def test_wrapper_chain_matches_3dcnn_dvs6_and_gives_twelve_channels(campaign):
    assert campaign.segmentation_wrappers(campaign.MODELS[LABEL]) == ["dvs_polarity", "framestack"]
    assert campaign.segmentation_wrappers(campaign.MODELS[LABEL]) == campaign.segmentation_wrappers(campaign.MODELS["3DCNN-DVS6"])

    class _Env(gym.Env):
        metadata: dict = {}
        render_mode = None
        spec = None
        observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, (2, H, W, 3), np.uint8)})
        action_space = gym.spaces.Box(-1, 1, shape=(2, 2), dtype=np.float32)

    env = Body(campaign.segmentation_wrappers(campaign.MODELS[LABEL])).wrap(_Env())
    assert isinstance(env.env, FrameStack) and isinstance(env.env.env, DVSPolarity)
    assert env.env.env.color and env.env.env.n_out == 6
    assert tuple(env.observation_space["policy"].shape) == (2, 12, H, W)


def test_encoder_builds_at_twelve_channels_like_the_cltt_dvs6_encoder(campaign):
    enc = _build(campaign.MODELS[LABEL], 12)
    assert enc.base_channels == 6 and enc.num_frames == 2
    assert _n(enc) == _n(_build(campaign.MODELS[CLTT], 12)) == 744_952
    x = torch.randint(0, 256, (2, 12, H, W), dtype=torch.uint8).float()
    out = enc(x)
    assert out.shape == (2, 512) and torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters() if p.requires_grad)
    with pytest.raises(ValueError, match="channels_per_frame=6"):
        _build(campaign.MODELS[LABEL], 6)
    with pytest.raises(ValueError, match="not a plausible image depth"):
        _build(campaign.MODELS[TWIN], 12)
