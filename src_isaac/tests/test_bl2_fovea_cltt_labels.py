"""Owner 2026-11-09 (presenter relay), BL2 full rows on binding: the original 3-channel fovea on three attached-CLTT-Ref arms.

  ViT-P8-666K-Mixer+Fovea+CLTT-Ref = ViT-P8-666K-Mixer + pre ("fovea_mask",) + attached cltt_ref (the first Mixer with an aux)
  3DCNN-CLTT-Ref+Fovea             = 3DCNN-CLTT-Ref + pre ("fovea_mask",)
  ViViT-P8-T2+CLTT-Ref+Fovea       = ViViT-P8-T2+CLTT-Ref + pre ("fovea_mask",)

Pinned here:
  * each spec is its twin plus exactly those fields, and equals the existing fovea label plus the aux;
  * the encoder is the twin's (same parameter count) and runs on 3-channel frames (6 for the 2-frame stacks);
  * the real chain FrameStack(FoveaMask) keeps 3 channels per frame, current frame last;
  * CLTT-Ref at the default 3 channels per frame (no knob): the loss is finite with gradients on every arm, the
    current-frame view of a 2-frame stack is the last frame repeated, and a single frame is its own view.
Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.fovea_mask import FoveaMask
from nett_skrl.brain.config import EncoderCfg
from nett_skrl.brain.registry import encoder_mapping

H, W = 80, 128    # the series eye
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
# label -> (twin, the existing fovea label it extends with the aux, channels the encoder sees)
LABELS = {
    "ViT-P8-666K-Mixer+Fovea+CLTT-Ref": ("ViT-P8-666K-Mixer", None, 3),
    "3DCNN-CLTT-Ref+Fovea": ("3DCNN-CLTT-Ref", "3DCNN+Fovea", 6),
    "ViViT-P8-T2+CLTT-Ref+Fovea": ("ViViT-P8-T2+CLTT-Ref", "ViViT-P8-T2+Fovea", 6),
}


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in ("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_AUX_BATCH", "NETT_AUX_CLTT_REF_OFFSETS", "NETT_CLTT_REF_DEDUP",
              "NETT_DVS_COLOR", "NETT_FOVEAM_RADIUS", "NETT_FOVEAM_MASK_SIGMA"):
        monkeypatch.delenv(k, raising=False)


def _encoder(spec, channels):
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(0)
    return encoder_mapping[spec["encoder"]](gym.spaces.Box(0, 255, (channels, H, W), np.uint8), **kwargs)


def _n(enc):
    return sum(p.numel() for p in enc.parameters())


# ---------------------------------------------------------------- labels

@pytest.mark.parametrize("label", LABELS)
def test_spec_is_the_twin_plus_the_fovea_only(campaign, label):
    m = campaign.MODELS
    twin, fovea_twin, _ = LABELS[label]
    spec = m[label]
    assert "pre" not in m[twin] and tuple(spec["pre"]) == ("fovea_mask",)
    extra = {"aux": "cltt_ref", "aux_weight": 1.0} if label.startswith("ViT-P8-666K-Mixer") else {}
    assert spec == {**m[twin], "pre": ("fovea_mask",), **extra}
    assert spec["aux"] == "cltt_ref" and spec["aux_weight"] == 1.0
    if fovea_twin is not None:   # the existing fovea video label, plus the aux
        assert {k: v for k, v in spec.items() if k not in ("aux", "aux_weight")} == m[fovea_twin]
    # fovea innermost, then the 2-frame stack on the video arms: the encoders stack foveated frames
    assert campaign.segmentation_wrappers(spec) == (["fovea_mask", "framestack"] if spec["framestack"] else ["fovea_mask"])
    assert spec["framestack"] is (LABELS[label][2] == 6)


def test_mixer_label_keeps_the_mixer_backbone(campaign):
    m = campaign.MODELS
    spec = m["ViT-P8-666K-Mixer+Fovea+CLTT-Ref"]
    assert spec["cfg"]["attn_mode"] == "mixer" and spec["cfg"]["embed_dim"] == 156
    assert {k: v for k, v in spec.items() if k not in ("pre",)} == {**m["ViT-P8-666K-Mixer"], "aux": "cltt_ref", "aux_weight": 1.0}


@pytest.mark.parametrize("label", LABELS)
def test_encoder_is_the_twins(campaign, label):
    twin, _, c = LABELS[label]
    enc = _encoder(campaign.MODELS[label], c)
    assert _n(enc) == _n(_encoder(campaign.MODELS[twin], c))
    assert enc(torch.randint(0, 256, (2, c, H, W)).float()).shape[0] == 2


# ---------------------------------------------------------------- the real body chain

class _StubEnv(gym.Env):
    metadata: dict = {}
    render_mode = None
    spec = None

    def __init__(self, frame):
        self.observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, tuple(frame.shape), np.uint8)})
        self.action_space = gym.spaces.Box(-1, 1, shape=(frame.shape[0], 2), dtype=np.float32)
        self.frame = frame

    def reset(self, **kw):
        return {"policy": self.frame.clone()}, {}

    def step(self, action):
        return self.reset()[0], 0.0, False, False, {}


def test_two_frame_stack_of_foveated_frames_is_6_channels():
    from nett_skrl.body.wrappers.channels_first import ChannelsFirst
    from nett_skrl.body.wrappers.framestack import FrameStack

    torch.manual_seed(0)
    f = torch.randint(0, 256, (2, H, W, 3), dtype=torch.uint8)
    fov = FoveaMask(_StubEnv(f))
    one = torch.as_tensor(np.asarray(fov.reset()[0]["policy"]))
    chain = ChannelsFirst(FrameStack(FoveaMask(_StubEnv(f)), n_stack=2))
    assert tuple(chain.observation_space["policy"].shape) == (2, 6, H, W)
    p = torch.as_tensor(np.asarray(chain.reset()[0]["policy"]))
    assert tuple(p.shape) == (2, 6, H, W)
    assert torch.equal(p[:, 3:6], one.permute(0, 3, 1, 2))           # T-major: the current foveated frame is the last 3


# ---------------------------------------------------------------- CLTT-Ref at the default 3 channels per frame

class _FakeMemory:
    def __init__(self, raw):
        self.tensors = {"observations": raw,
                        "terminated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool),
                        "truncated": torch.zeros(*raw.shape[:2], 1, dtype=torch.bool)}
        self.memory_size, self.filled, self.memory_index = raw.shape[0], True, 0

    def get_tensor_by_name(self, name):
        raise AssertionError("never transfer the whole buffer")


@pytest.mark.parametrize("label", LABELS)
def test_cltt_loss_is_finite_with_gradients_at_the_default_channels(campaign, monkeypatch, label):
    from nett_skrl.brain.aux.cltt_ref_aux import CLTTReferenceAuxLoss

    _, _, c = LABELS[label]
    monkeypatch.setenv("NETT_AUX_BATCH", "8")
    enc = _encoder(campaign.MODELS[label], c)
    aux = CLTTReferenceAuxLoss(enc)
    torch.manual_seed(1)
    aux.attach_memory(_FakeMemory(torch.randint(0, 256, (24, 2, c * H * W)).float()))
    x = torch.rand(3, c, H, W)
    (v,) = aux._make_views([x], enc)
    assert torch.equal(v, x if c == 3 else torch.cat([x[:, 3:], x[:, 3:]], dim=1))   # the current frame (repeated)
    value = aux.compute(enc, torch.empty(0))
    assert torch.isfinite(value) and value.requires_grad
    value.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in enc.parameters())
