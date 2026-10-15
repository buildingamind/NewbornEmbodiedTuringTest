"""The foveated eye (owner 2026-10-15, workspace DECISIONS §117): colour only in a small centre spot,
crisp grey to the sharp radius, slightly blurred grey (BT.601, three equal channels) beyond it.

Each property is checked against a positive control: the colour test uses a saturated frame whose
grey differs from every channel, and the blur test uses a checkerboard that a blur must change.
Also pins the two labels as CNN-UnityRecipe plus one wrapper and nothing else.
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.fovea import Fovea, fovea_params_from_env

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
H, W = 56, 88


class _StubEnv(gym.Env):
    metadata: dict = {}
    render_mode = None
    spec = None

    def __init__(self, frame, device="cpu"):
        n, h, w, c = frame.shape
        self.observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(0, 255, shape=(n, h, w, c), dtype=np.uint8)})
        self.action_space = gym.spaces.Box(-1, 1, shape=(n, 2), dtype=np.float32)
        self.frame = frame.to(device)
        self.n = n

    def reset(self, **kwargs):
        return {"policy": self.frame.clone()}, {}

    def step(self, action):
        z = torch.zeros(self.n, device=self.frame.device)
        return {"policy": self.frame.clone()}, z, z.bool(), z.bool(), {}


def _solid(rgb, n=2):
    return torch.tensor(rgb, dtype=torch.uint8).view(1, 1, 1, 3).expand(n, H, W, 3).contiguous()


def _checker(n=2):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    g = (((yy + xx) % 2) * 255).to(torch.uint8)
    return g.view(1, H, W, 1).expand(n, H, W, 3).contiguous()


def _r():
    yy = torch.arange(H, dtype=torch.float32) - (H - 1) / 2
    xx = torch.arange(W, dtype=torch.float32) - (W - 1) / 2
    return torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)


@pytest.mark.parametrize("device", DEVICES)
def test_colour_only_in_the_spot_and_grey_is_three_equal_bt601_channels(device):
    red = _solid((200, 30, 10))
    env = Fovea(_StubEnv(red, device), color_radius=4, sharp_radius=12, sigma=2)
    out = env.reset()[0]["policy"]
    assert out.dtype == torch.uint8 and out.device.type == device and tuple(out.shape) == (2, H, W, 3)
    o = out[0].cpu().numpy().astype(int)
    r = _r().numpy()
    y = round(0.299 * 200 + 0.587 * 30 + 0.114 * 10)
    assert (o[r <= 4] == (200, 30, 10)).all()                       # positive control: colour kept
    outside = o[r >= 6]                                             # beyond the 1.5x ramp
    assert (outside == y).all(), np.unique(outside)                 # grey, all three channels equal
    assert y not in (200, 30, 10)


@pytest.mark.parametrize("device", DEVICES)
def test_crisp_inside_the_sharp_radius_blurred_beyond(device):
    chk = _checker()
    env = Fovea(_StubEnv(chk, device), color_radius=2, sharp_radius=16, sigma=1.5)
    o = env.reset()[0]["policy"][0, ..., 0].cpu().numpy().astype(int)
    src = chk[0, ..., 0].numpy().astype(int)
    r = _r().numpy()
    assert (o[r <= 16] == src[r <= 16]).all()                       # crisp
    far = r >= 24 + 4                                               # past the ramp and the kernel's reach of it
    assert np.abs(o[far] - 127.5).max() < 20                        # a checkerboard blurs to mid-grey
    assert np.abs(src[far] - 127.5).min() > 100                     # positive control: it was not grey


def test_chw_and_numpy_paths_agree_with_nhwc():
    frame = torch.randint(0, 256, (2, H, W, 3), generator=torch.Generator().manual_seed(1), dtype=torch.uint8)
    env = Fovea(_StubEnv(frame), color_radius=5, sharp_radius=15, sigma=2)
    ref = env.observation(frame.clone())
    out_np = env.observation(frame.numpy())
    assert isinstance(out_np, np.ndarray) and (out_np == ref.numpy()).all()
    out_chw = env.observation(frame.permute(0, 3, 1, 2).contiguous())
    assert (out_chw.permute(0, 2, 3, 1) == ref).all()


def test_refuses_non_rgb_and_bad_sizes(monkeypatch):
    grey = torch.zeros(2, H, W, 1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="RAW RGB"):
        Fovea(_StubEnv(grey), color_radius=2, sharp_radius=4, sigma=1)
    with pytest.raises(ValueError):
        Fovea(_StubEnv(_solid((1, 2, 3))), color_radius=8, sharp_radius=4, sigma=1)
    for k in ("NETT_FOVEA_COLOR_RADIUS", "NETT_FOVEA_SHARP_RADIUS", "NETT_FOVEA_SIGMA"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValueError, match="no default size"):
        fovea_params_from_env()
    monkeypatch.setenv("NETT_FOVEA_COLOR_RADIUS", "14")
    monkeypatch.setenv("NETT_FOVEA_SHARP_RADIUS", "56")
    monkeypatch.setenv("NETT_FOVEA_SIGMA", "0")
    with pytest.raises(ValueError, match="positive"):
        fovea_params_from_env()
    monkeypatch.setenv("NETT_FOVEA_SIGMA", "2")
    assert fovea_params_from_env() == {"color_radius": 14.0, "sharp_radius": 56.0, "sigma": 2.0}
    env = Fovea(_StubEnv(_solid((1, 2, 3))))
    assert (env.color_radius, env.sharp_radius, env.sigma) == (14.0, 56.0, 2.0)


def test_labels_add_one_wrapper_and_nothing_else(monkeypatch):
    from pathlib import Path
    from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity
    from nett_skrl.body.wrappers.registry import _load_wrapper

    assert _load_wrapper("fovea") is Fovea
    assert _load_wrapper("dvs_polarity") is DVSPolarity
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train as campaign
    base = campaign.MODELS["CNN-UnityRecipe"]
    for label, name in (("CNN-UnityRecipe+Fovea", "fovea"), ("CNN-UnityRecipe+DVS", "dvs_polarity")):
        spec = campaign.MODELS[label]
        assert campaign.segmentation_wrappers(spec) == [name]
        assert {k: v for k, v in spec.items() if k != "pre"} == base


def test_dvs_label_builds_a_two_channel_first_conv():
    """The +DVS chain: DVSPolarity -> ChannelsFirst -> NatureCNN, so in_channels=2 is checked where the
    encoder builds it, not asserted from the wrapper's own space."""
    from nett_skrl.body.wrappers.channels_first import ChannelsFirst
    from nett_skrl.body.wrappers.dvs_polarity import DVSPolarity

    from nett_skrl.brain.encoders.nature_cnn import NatureCNN

    frame = torch.randint(0, 256, (2, 64, 96, 3), generator=torch.Generator().manual_seed(2), dtype=torch.uint8)
    chain = ChannelsFirst(DVSPolarity(_StubEnv(frame), threshold=30))
    space = chain.observation_space["policy"]
    assert tuple(space.shape) == (2, 2, 64, 96)
    per_env = gym.spaces.Box(0, 255, shape=space.shape[1:], dtype=np.uint8)
    enc = NatureCNN(per_env, features_dim=16, conv_dim=8, spatial_pool=False)
    conv1 = next(m for m in enc.modules() if isinstance(m, torch.nn.Conv2d))
    assert conv1.in_channels == 2
    obs, _ = chain.reset()
    assert enc(torch.as_tensor(obs["policy"]).float()).shape == (2, 16)


def test_fovea_label_keeps_three_channels_through_the_real_chain():
    from nett_skrl.body.wrappers.channels_first import ChannelsFirst
    from nett_skrl.brain.encoders.nature_cnn import NatureCNN

    frame = torch.randint(0, 256, (2, 64, 96, 3), generator=torch.Generator().manual_seed(3), dtype=torch.uint8)
    chain = ChannelsFirst(Fovea(_StubEnv(frame), color_radius=3, sharp_radius=12, sigma=2))
    space = chain.observation_space["policy"]
    assert tuple(space.shape) == (2, 3, 64, 96)
    per_env = gym.spaces.Box(0, 255, shape=space.shape[1:], dtype=np.uint8)
    enc = NatureCNN(per_env, features_dim=16, conv_dim=8, spatial_pool=False)
    conv1 = next(m for m in enc.modules() if isinstance(m, torch.nn.Conv2d))
    assert conv1.in_channels == 3
    obs, _ = chain.reset()
    assert enc(torch.as_tensor(obs["policy"]).float()).shape == (2, 16)


def test_xsp2_map35x56_is_l738s_xsp2_at_the_448x280_eye(monkeypatch):
    """§117's third arm: "the L738 model adapted to this". L738 is the "XSP2" label; XSP2-MAP35x56 is that encoder with
    features_dim set to the 448x280 eye's stride-8 map. Control: the plain XSP2 cfg refuses that eye, naming 1960."""
    from pathlib import Path
    import torch.nn as nn
    from nett_skrl.brain.config import EncoderCfg
    from nett_skrl.brain.registry import encoder_mapping

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train as campaign

    spec, base = campaign.MODELS["XSP2-MAP35x56"], campaign.MODELS["XSP2"]
    assert {k: v for k, v in spec.items() if k != "cfg"} == {k: v for k, v in base.items() if k != "cfg"}
    assert spec["cfg"] == {**base["cfg"], "features_dim": 1960}

    def kw(label):
        k = EncoderCfg(**campaign.MODELS[label]["cfg"]).as_kwargs()
        k.pop("trainable", None)
        return k

    eye = gym.spaces.Box(0, 255, shape=(6, 280, 448), dtype=np.uint8)
    with pytest.raises(ValueError, match="features_dim must be 1960 at the 448x280 eye"):
        encoder_mapping["xsp2"](eye, **kw("XSP2"))
    big = encoder_mapping["xsp2"](eye, **kw("XSP2-MAP35x56"))
    small = encoder_mapping["xsp2"](gym.spaces.Box(0, 255, shape=(6, 80, 128), dtype=np.uint8), **kw("XSP2"))
    assert big.map_hw == (35, 56) and big.features_dim == 1960 and big.map_stride == 8
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(big) == n(small) == 166_510
    convs = lambda e: [(m.kernel_size, m.stride, m.padding, m.in_channels, m.out_channels)
                       for m in e.modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))]
    assert convs(big) == convs(small)
    x = torch.rand(2, 6, 280, 448, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        assert big(x).shape == (2, 1960)
