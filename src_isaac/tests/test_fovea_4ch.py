"""FOVEA-4CH pilot (owner 2026-11-09): colour only inside the fovea_mask window (0 outside) in ch0-2, plus a
separate BT.601 grey of the WHOLE frame in ch3.

Checks, all on CPU with synthetic frames:
  * periphery RGB == 0, fovea RGB == round(mask x RGB), ch3 == round(BT.601 luma of the whole input);
  * the window and edge are fovea_mask's (same soft_mask, same knobs);
  * spaces / shapes for a single frame (HWC and CHW, numpy and torch) and for a 2-frame stack (8 channels);
  * the first conv of the two pilot labels' encoders is built for 4 (CNN-D512P) and 4 per frame (3DCNN, 8 total);
  * the CLTT current-frame view slices 4-channel frames when NETT_AUX_CLTT_CHANNELS_PER_FRAME=4 and refuses at 3;
  * the first-frame self-check and the optional dump;
  * each label is its twin plus fovea_4ch and nothing else; the twins are untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC / "examples"))

import campaign_train as ct  # noqa: E402
from nett_skrl.body.wrappers.dvs_polarity import _LUMA  # noqa: E402
from nett_skrl.body.wrappers.fovea_4ch import ENV_DUMP, Fovea4ch  # noqa: E402
from nett_skrl.body.wrappers.fovea_mask import ENV_RADIUS, ENV_SIGMA, FoveaMask, soft_mask  # noqa: E402
from nett_skrl.brain.config import EncoderCfg  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402

H, W = 80, 128


class _StubEnv(gym.Env):
    metadata: dict = {}
    render_mode = None
    spec = None

    def __init__(self, frame):
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
    for k in (ENV_RADIUS, ENV_SIGMA, ENV_DUMP, "NETT_AUX_CLTT_CHANNELS_PER_FRAME", "NETT_DVS_COLOR"):
        monkeypatch.delenv(k, raising=False)


def _random(n=2, seed=0):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.integers(1, 256, size=(n, H, W, 3), dtype=np.uint8))


def _refs(f):
    xf = f.float()
    m = soft_mask(H, W, 3.0, 1.0)
    col = (m[..., None] * xf).round().clamp(0, 255)
    y = (xf * torch.tensor(_LUMA)).sum(-1).round().clamp(0, 255)
    return m, col, y


def test_channels_periphery_fovea_grey():
    f = _random()
    out = Fovea4ch(_StubEnv(f)).observation({"policy": f})["policy"]
    assert out.shape == (2, H, W, 4) and out.dtype == torch.uint8
    m, col, y = _refs(f)
    assert int(out[..., :3][:, m == 0].max()) == 0                       # periphery RGB == 0
    assert torch.equal(out[..., :3].float(), col)                         # fovea RGB == round(mask x RGB)
    assert torch.equal(out[..., 3].float(), y)                            # ch3 == luma of the WHOLE frame
    cy, cx = H // 2, W // 2
    assert (out[:, cy, cx, :3].float() - f[:, cy, cx].float()).abs().max() <= 4   # colour kept at the centre
    assert int((m > 0).sum()) < 200                                       # the window is small (r 3 + blur)


def test_same_window_and_grey_as_fovea_mask():
    """fovea_mask's blend == m*rgb + (1-m)*y; recover it from the 4 channels up to uint8 rounding."""
    f = _random(1, seed=3)
    four = Fovea4ch(_StubEnv(f)).observation({"policy": f})["policy"].float()
    blend = FoveaMask(_StubEnv(f)).observation({"policy": f})["policy"].float()
    m = soft_mask(H, W, 3.0, 1.0)[..., None]
    recon = four[..., :3] + (1 - m) * four[..., 3:4]
    assert float((recon - blend).abs().max()) <= 1.5 + 1e-6     # three uint8 roundings: m*rgb, y, the blend
    assert torch.equal(four[..., 3][:, (m[..., 0] == 0)], blend[..., 0][:, (m[..., 0] == 0)])  # grey surround agrees


def test_knobs_shared_with_fovea_mask(monkeypatch):
    monkeypatch.setenv(ENV_RADIUS, "6")
    e = Fovea4ch(_StubEnv(_random(1)))
    assert e.radius == 6.0 and e.sigma == 1.0


@pytest.mark.parametrize("layout", ["hwc_torch", "chw_numpy"])
def test_spaces_and_layouts(layout):
    f = _random(3)
    if layout == "chw_numpy":
        f = f.permute(0, 3, 1, 2).contiguous().numpy()
    env = Fovea4ch(_StubEnv(f))
    want = (3, 4, H, W) if layout == "chw_numpy" else (3, H, W, 4)
    assert tuple(env.observation_space["policy"].shape) == want
    out = env.observation({"policy": f})["policy"]
    assert tuple(out.shape) == want
    assert isinstance(out, np.ndarray) == (layout == "chw_numpy")


def test_refuses_non_rgb_input():
    with pytest.raises(ValueError, match="4 channels|received 4"):
        Fovea4ch(_StubEnv(np.zeros((1, H, W, 4), np.uint8)))


def test_two_frame_stack_is_8_channels_through_the_real_chain():
    from nett_skrl.body.wrappers.channels_first import ChannelsFirst
    from nett_skrl.body.wrappers.framestack import FrameStack

    f = _random(2)
    chain = ChannelsFirst(FrameStack(Fovea4ch(_StubEnv(f)), n_stack=2))
    space = chain.observation_space["policy"]
    assert tuple(space.shape) == (2, 8, H, W)
    obs, _ = chain.reset()
    p = torch.as_tensor(np.asarray(obs["policy"]))
    assert tuple(p.shape) == (2, 8, H, W)
    _, col, y = _refs(f)
    cur = p[:, 4:8].permute(0, 2, 3, 1).float()               # T-major: the current frame is the last 4
    assert torch.equal(cur[..., :3], col) and torch.equal(cur[..., 3], y)


def _encoder(label, channels):
    spec = ct.MODELS[label]
    kwargs = EncoderCfg(**spec["cfg"]).as_kwargs()
    kwargs.pop("trainable", None)
    torch.manual_seed(0)
    space = gym.spaces.Box(0, 255, shape=(channels, H, W), dtype=np.uint8)
    return encoder_mapping[spec["encoder"]](space, **kwargs)


@pytest.mark.parametrize("label,channels,in_ch", [("CNN-D512P+Fovea4ch", 4, 4), ("3DCNN-CLTT-Ref+Fovea4ch", 8, 4)])
def test_first_conv_in_channels_and_forward(label, channels, in_ch):
    enc = _encoder(label, channels)
    conv1 = next(m for m in enc.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)))
    assert conv1.in_channels == in_ch
    x = torch.randint(0, 256, (2, channels, H, W), dtype=torch.uint8).float()
    assert enc(x).shape[0] == 2


def test_twin_encoders_unchanged_except_first_conv():
    for label, twin, c4, c3 in (("CNN-D512P+Fovea4ch", "CNN-D512P", 4, 3), ("3DCNN-CLTT-Ref+Fovea4ch", "3DCNN-CLTT-Ref", 8, 6)):
        a = sum(p.numel() for p in _encoder(label, c4).parameters())
        b = sum(p.numel() for p in _encoder(twin, c3).parameters())
        conv_a = next(m for m in _encoder(label, c4).modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d)))
        per_in = conv_a.weight.numel() // conv_a.in_channels
        assert a - b == per_in, (label, a, b, per_in)   # exactly one more input channel's worth of first-conv weights


def test_cltt_current_frame_view_needs_4_per_frame(monkeypatch):
    from nett_skrl.brain.aux.cltt_views import current_frame_stack

    x = torch.randint(0, 256, (3, 8, H, W), dtype=torch.uint8).float()
    with pytest.raises(ValueError):
        current_frame_stack(x)                                           # default 3: 8 % 3 != 0 -> loud
    monkeypatch.setenv("NETT_AUX_CLTT_CHANNELS_PER_FRAME", "4")
    v = current_frame_stack(x)
    assert v.shape == x.shape and torch.equal(v[:, :4], x[:, 4:]) and torch.equal(v[:, 4:], x[:, 4:])


def test_self_check_skips_blank_then_runs_and_dumps(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(ENV_DUMP, str(tmp_path))
    blank = torch.zeros(1, H, W, 3, dtype=torch.uint8)
    env = Fovea4ch(_StubEnv(blank))
    env.observation({"policy": blank})
    assert not list(tmp_path.glob("*.npz"))                              # a black reset frame is not the check
    f = _random(1, seed=5)
    env.observation({"policy": f})
    out = capsys.readouterr().out
    assert "first-frame check" in out and "-> OK" in out
    files = list(tmp_path.glob("fovea4ch_first_obs_*.npz"))
    assert len(files) == 1
    d = np.load(files[0])
    assert d["input_rgb"].shape == (H, W, 3) and d["out_4ch"].shape == (H, W, 4)
    env.observation({"policy": f})
    assert len(list(tmp_path.glob("*.npz"))) == 1                        # once per process


def test_labels_are_twin_plus_fovea_4ch_only():
    from nett_skrl.body.wrappers.registry import _load_wrapper

    assert _load_wrapper("fovea_4ch") is Fovea4ch
    for label, twin in (("CNN-D512P+Fovea4ch", "CNN-D512P"), ("3DCNN-CLTT-Ref+Fovea4ch", "3DCNN-CLTT-Ref")):
        spec = ct.MODELS[label]
        assert {k: v for k, v in spec.items() if k != "pre"} == ct.MODELS[twin]
        assert tuple(spec["pre"]) == ("fovea_4ch",)
        assert "pre" not in ct.MODELS[twin]
    assert tuple(ct.MODELS["CNN-D512P+Fovea"]["pre"]) == ("fovea_mask",)   # the existing fovea label is untouched
