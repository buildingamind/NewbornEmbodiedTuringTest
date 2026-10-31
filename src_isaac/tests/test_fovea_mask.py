"""The soft-mask foveated retina (owner 2026-10-30): a radius-3 colour disk at the image centre whose mask is
Gaussian-blurred (sigma 1), BT.601 grey (three equal channels) everywhere else, no image blur, size unchanged.

Positive controls: a saturated frame whose grey differs from every channel (so "colour kept" and "grey" are
distinguishable), and a checkerboard (so an image blur, which this wrapper must NOT apply, would show).
Also pins the four BASELINE +Fovea labels as their series twin plus fovea_mask and nothing else.
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.dvs_polarity import _LUMA
from nett_skrl.body.wrappers.fovea_mask import (DEFAULT_RADIUS, DEFAULT_SIGMA, ENV_RADIUS, ENV_SIGMA, FoveaMask,
                                                fovea_mask_params_from_env, soft_mask)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
H, W = 80, 128          # the series eye


class _StubEnv(gym.Env):
    metadata: dict = {}
    render_mode = None
    spec = None

    def __init__(self, frame, chw=False):
        self.observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(0, 255, shape=tuple(frame.shape), dtype=np.uint8)})
        self.action_space = gym.spaces.Box(-1, 1, shape=(frame.shape[0], 2), dtype=np.float32)
        self.frame = frame

    def reset(self, **kwargs):
        return {"policy": self.frame.clone() if isinstance(self.frame, torch.Tensor) else self.frame.copy()}, {}

    def step(self, action):
        return self.reset()[0], 0.0, False, False, {}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ENV_RADIUS, raising=False)
    monkeypatch.delenv(ENV_SIGMA, raising=False)


def _saturated(n=2):
    f = torch.zeros(n, H, W, 3, dtype=torch.uint8)
    f[..., 0] = 220; f[..., 1] = 30; f[..., 2] = 90
    return f


def _grey_of(rgb):
    return sum(c * l for c, l in zip(rgb, _LUMA))


def test_defaults_are_the_owner_recipe():
    p = fovea_mask_params_from_env()
    assert p == {"radius": 3.0, "sigma": 1.0} and (DEFAULT_RADIUS, DEFAULT_SIGMA) == (3.0, 1.0)


def test_section117_knobs_do_not_leak(monkeypatch):
    monkeypatch.setenv("NETT_FOVEA_COLOR_RADIUS", "7"); monkeypatch.setenv("NETT_FOVEA_SIGMA", "2")
    assert fovea_mask_params_from_env() == {"radius": 3.0, "sigma": 1.0}


def test_bad_knob_refused(monkeypatch):
    monkeypatch.setenv(ENV_SIGMA, "0")
    with pytest.raises(ValueError, match=ENV_SIGMA):
        fovea_mask_params_from_env()


def test_mask_is_the_blurred_disk_centred_and_monotone():
    m = soft_mask(H, W, 3.0, 1.0)
    assert m.shape == (H, W)
    cy, cx = (H - 1) / 2, (W - 1) / 2
    yy, xx = torch.meshgrid(torch.arange(H) - cy, torch.arange(W) - cx, indexing="ij")
    r = torch.sqrt(yy ** 2 + xx ** 2)
    assert 0.95 < float(m.max()) < 1.0                           # blurred: just under 1, not renormalised
    assert float(m[r >= 7].max()) < 1e-3                         # gone well outside the radius
    assert 0.2 < float(m[(r > 2.9) & (r < 3.6)].mean()) < 0.8    # soft at the edge, not a hard step
    assert torch.allclose(m, m.flip(0), atol=1e-6) and torch.allclose(m, m.flip(1), atol=1e-6)   # centred
    rings = [float(m[(r >= k) & (r < k + 1)].mean()) for k in range(8)]
    assert all(a >= b for a, b in zip(rings, rings[1:])), rings  # radial average non-increasing (a pixel disk
                                                                 # is not exactly round, so single pixels at
                                                                 # equal r can differ slightly)
    hard = soft_mask(H, W, 3.0, 1e-3)                            # a vanishing blur leaves the binary disk
    assert torch.equal((hard > 0.5), (r <= 3.0))


@pytest.mark.parametrize("device", DEVICES)
def test_colour_at_centre_grey_outside_size_kept(device):
    f = _saturated().to(device)
    env = FoveaMask(_StubEnv(f))
    out = env.observation({"policy": f})["policy"]
    assert out.shape == f.shape and out.dtype == torch.uint8 and out.device == f.device
    g = _grey_of((220, 30, 90))
    cy, cx = H // 2, W // 2                                      # a centre pixel, r = 0.71
    assert (out[0, cy, cx].float() - torch.tensor([220., 30., 90.], device=device)).abs().max() <= 4
    far = out[0, 0, 0].float()
    assert torch.allclose(far, torch.full((3,), round(g), device=device, dtype=torch.float32), atol=1)
    assert torch.equal(out[0, :, :, 0], out[0, :, :, 0]) and bool((out[0, 10:20, 10:20, 0] == out[0, 10:20, 10:20, 2]).all())


def test_no_image_blur():
    f = torch.zeros(1, H, W, 3, dtype=torch.uint8)
    f[:, ::2, ::2] = 255; f[:, 1::2, 1::2] = 255                 # checkerboard
    out = FoveaMask(_StubEnv(f)).observation({"policy": f})["policy"]
    # Outside the fovea every pixel is the grey of ITS OWN colour: 0 or 255, never a blurred mixture.
    assert set(out[0, :20, :20, 0].unique().tolist()) <= {0, 255}


def test_blend_matches_the_owner_formula_in_unit_range():
    rng = np.random.default_rng(0)
    f = torch.as_tensor(rng.integers(0, 256, size=(2, H, W, 3), dtype=np.uint8))
    out = FoveaMask(_StubEnv(f)).observation({"policy": f})["policy"].float() / 255.0
    x = f.float() / 255.0                                        # the owner's [0, 1] normalisation
    m = soft_mask(H, W, 3.0, 1.0)[..., None]
    y = (x * torch.tensor(_LUMA)).sum(-1, keepdim=True)
    ref = m * x + (1 - m) * y
    assert float((out - ref).abs().max()) <= 0.5 / 255 + 1e-6    # only the uint8 rounding differs


def test_chw_numpy_and_channel_refusal():
    f = _saturated(1).permute(0, 3, 1, 2).contiguous().numpy()   # NCHW numpy
    out = FoveaMask(_StubEnv(f)).observation({"policy": f})["policy"]
    assert isinstance(out, np.ndarray) and out.shape == f.shape and out.dtype == np.uint8
    bad = np.zeros((1, H, W, 4), np.uint8)
    with pytest.raises(ValueError, match="3"):
        FoveaMask(_StubEnv(bad))


def test_the_four_baseline_fovea_labels():
    import importlib, sys, os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "examples"))
    ct = importlib.import_module("campaign_train")
    for twin in ("CNN-D512P", "ViT-P8-666K", "3DCNN", "ViViT-P8-T2", "CNN", "ViT", "ViViT-T2"):
        lab = ct.MODELS[f"{twin}+Fovea"]
        assert {k: v for k, v in lab.items() if k != "pre"} == ct.MODELS[twin]
        assert tuple(lab["pre"]) == ("fovea_mask",)
