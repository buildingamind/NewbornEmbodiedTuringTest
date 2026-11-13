"""Owner 2026-11-13 (presenter relay; workspace FINDINGS add. 421, CHICK-CONE step 2): the chick-cone colour input.

Per pixel: u8 sRGB -> linear -> T3 (chicken LWS, MWS, VS+SWS 1:2 white-normalised catches) -> sRGB-encode -> u8, from a
fixed 2^24 lookup table (body/wrappers/chick_cone.py). Pinned here:
  * T3: rows sum to 1, weights >= 0, cond 1.50 (the step 1 values);
  * the table is the registered one (sha256) and equals an independent reference written as tools/chick_cone.py
    apply_transform (linearise, BLAS matmul, encode, round) on all 2^24 colours;
  * every grey maps to itself; the reference colour and the other step 1 face colours map to their values;
  * the wrapper is per pixel, keeps shape / dtype / device / layout (HWC and CHW, numpy and torch), handles a float
    0-255 frame, and refuses anything but 3 channels;
  * a tampered table is refused at boot;
  * the four labels are their BL2 core-four twin plus pre ("chick_cone",) and nothing else, and chick_cone runs
    innermost (before framestack) in the real Body.
Run with CUDA_VISIBLE_DEVICES="" (everything here is CPU).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.body import Body
from nett_skrl.body.wrappers import chick_cone as cc
from nett_skrl.body.wrappers.chick_cone import LUT_SHA256, REF_IN, REF_OUT, T3, ChickCone, build_lut, check_lut, lut_np
from nett_skrl.body.wrappers.framestack import FrameStack

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
H, W = 80, 128    # the series eye
_TRAIN = Path(__file__).resolve().parents[1] / "examples" / "campaign_train.py"
LABELS = {"CNN-D512P+Cone": "CNN-D512P", "3DCNN+Cone": "3DCNN", "ViT-P8-666K+Cone": "ViT-P8-666K",
          "ViViT-P8-T2+Cone": "ViViT-P8-T2"}
# step 1 face colours (add. 383 observation means, rounded) -> cone values (tools/chick_cone.py apply_transform)
FACES = {(155, 110, 18): (148, 110, 56), (108, 23, 141): (98, 43, 128), (155, 36, 25): (141, 48, 33),
         (14, 146, 109): (69, 142, 117), (255, 0, 0): (232, 56, 29), (0, 255, 0): (121, 246, 122),
         (0, 0, 255): (1, 56, 230), REF_IN: REF_OUT}


@pytest.fixture()
def campaign(monkeypatch):
    monkeypatch.syspath_prepend(str(_TRAIN.parent))
    import campaign_train

    return campaign_train


class _StubEnv(gym.Env):
    metadata: dict = {}
    render_mode = None
    spec = None

    def __init__(self, shape, dtype=np.uint8):
        self.observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, shape=tuple(shape), dtype=dtype)})
        self.action_space = gym.spaces.Box(-1, 1, shape=(shape[0], 2), dtype=np.float32)


def _reference(rgb_u8: np.ndarray) -> np.ndarray:
    """tools/chick_cone.py apply_transform, verbatim in form (BLAS matmul, numpy pow)."""
    c = rgb_u8.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    q = np.clip(lin @ np.asarray(T3).T, 0, 1)
    enc = np.where(q <= 0.0031308, 12.92 * q, 1.055 * q ** (1 / 2.4) - 0.055)
    return np.clip(np.round(255.0 * enc), 0, 255).astype(np.uint8)


def _all_colours() -> np.ndarray:
    i = np.arange(1 << 24, dtype=np.uint32)
    return np.stack([(i >> 16) & 255, (i >> 8) & 255, i & 255], -1).astype(np.uint8)


def test_t3_is_white_normalised_nonnegative_and_well_conditioned():
    T = np.asarray(T3)
    assert np.allclose(T.sum(1), 1.0, atol=1e-12) and (T >= 0).all()
    assert abs(np.linalg.cond(T) - 1.5035) < 1e-3


def test_table_is_registered_and_equals_the_reference_on_all_colours():
    lut = build_lut()
    assert lut.shape == (1 << 24, 3) and lut.dtype == np.uint8
    assert cc.lut_sha256(lut) == LUT_SHA256
    assert np.array_equal(lut, _reference(_all_colours()))


def test_greys_and_face_colours():
    lut = lut_np()
    v = np.arange(256)
    assert np.array_equal(lut[(v << 16) | (v << 8) | v], np.stack([v, v, v], -1))
    for (r, g, b), want in FACES.items():
        assert tuple(int(x) for x in lut[(r << 16) | (g << 8) | b]) == want, (r, g, b)


def test_tampered_table_is_refused():
    bad = lut_np().copy()
    bad[12345, 0] ^= 1
    with pytest.raises(RuntimeError, match="sha256"):
        check_lut(bad)


@pytest.mark.parametrize("device", DEVICES)
def test_wrapper_per_pixel_hwc_torch(device):
    rng = np.random.default_rng(0)
    f = torch.as_tensor(rng.integers(0, 256, (3, H, W, 3), dtype=np.uint8)).to(device)
    f[0, :10] = 77                                                # a grey band
    env = ChickCone(_StubEnv(f.shape))
    out = env.observation({"policy": f})["policy"]
    assert out.shape == f.shape and out.dtype == torch.uint8 and out.device == f.device
    want = torch.as_tensor(_reference(f.cpu().numpy().reshape(-1, 3)).reshape(f.shape))
    assert torch.equal(out.cpu(), want)
    assert bool((out[0, :10] == 77).all())
    again = env.observation({"policy": f})["policy"]              # stateless across steps
    assert torch.equal(again, out)


def test_wrapper_chw_numpy_float_and_channel_refusal():
    rng = np.random.default_rng(1)
    hwc = rng.integers(0, 256, (2, H, W, 3), dtype=np.uint8)
    chw = np.ascontiguousarray(hwc.transpose(0, 3, 1, 2))
    out = ChickCone(_StubEnv(chw.shape)).observation({"policy": chw})["policy"]
    assert isinstance(out, np.ndarray) and out.shape == chw.shape and out.dtype == np.uint8
    assert np.array_equal(out.transpose(0, 2, 3, 1), _reference(hwc.reshape(-1, 3)).reshape(hwc.shape))
    ff = torch.as_tensor(hwc).float()                             # a float 0-255 frame
    outf = ChickCone(_StubEnv(ff.shape, np.float32)).observation({"policy": ff})["policy"]
    assert outf.dtype == torch.float32 and torch.equal(outf, torch.as_tensor(_reference(hwc.reshape(-1, 3)).reshape(hwc.shape)).float())
    with pytest.raises(ValueError, match="3"):
        ChickCone(_StubEnv((1, H, W, 4)))
    with pytest.raises(ValueError, match="3"):
        ChickCone(_StubEnv((1, H, W, 6)))


def test_boot_banner_and_first_frame_line(capsys):
    f = torch.zeros(1, H, W, 3, dtype=torch.uint8)
    f[..., 0] = 252; f[..., 1] = 182; f[..., 2] = 20
    f[:, :40] = 128
    env = ChickCone(_StubEnv(f.shape))
    env.observation({"policy": f})
    env.observation({"policy": f})
    err = capsys.readouterr().out
    assert f"LUT sha256 {LUT_SHA256} OK" in err and "grey 0..255 -> itself OK" in err
    assert err.count("[NETT chick_cone] first frame:") == 1
    assert f"grey px {40 * W} unchanged {40 * W}, non-grey px {40 * W} changed {40 * W}" in err


def test_the_four_cone_labels(campaign):
    for lab, twin in LABELS.items():
        spec = campaign.MODELS[lab]
        assert {k: v for k, v in spec.items() if k != "pre"} == campaign.MODELS[twin], lab
        assert tuple(spec["pre"]) == ("chick_cone",) and "pre" not in campaign.MODELS[twin], lab


@pytest.mark.parametrize("label", ["3DCNN+Cone", "ViViT-P8-T2+Cone", "CNN-D512P+Cone", "ViT-P8-666K+Cone"])
def test_chick_cone_runs_innermost_in_the_real_body(campaign, label):
    chain = campaign.segmentation_wrappers(campaign.MODELS[label])
    assert chain[0] == "chick_cone"
    assert chain == ["chick_cone"] + campaign.segmentation_wrappers(campaign.MODELS[LABELS[label]])

    class _Env(gym.Env):
        metadata: dict = {}
        render_mode = None
        spec = None
        observation_space = gym.spaces.Dict({"policy": gym.spaces.Box(0, 255, (2, H, W, 3), np.uint8)})
        action_space = gym.spaces.Box(-1, 1, shape=(2, 2), dtype=np.float32)

    env = Body(chain).wrap(_Env())
    inner = env
    while not isinstance(inner, ChickCone):
        inner = inner.env
    assert isinstance(inner.env, _Env)                            # nothing between the raw env and chick_cone
    if campaign.MODELS[label]["framestack"]:
        assert isinstance(env.env, FrameStack) and env.env.env is inner
