"""The B&W eye must be BT.601 luminance, one channel end to end, on the input the env really supplies.

Fixtures use ``{"policy": uint8 NHWC tensor}`` on CUDA when present (test_lumnorm.py's lesson:
a numpy-only fixture cannot fail on the CUDA path). The chain test drives Grayscale ->
ChannelsFirst -> NatureCNN, so "3 -> 1 input channels" is checked where the encoder builds it,
not asserted from the wrapper's own space.
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers.channels_first import ChannelsFirst
from nett_skrl.body.wrappers.grayscale import Grayscale

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _StubEnv(gym.Env):
    metadata: dict = {}
    render_mode = None
    spec = None

    def __init__(self, n=4, h=28, w=44, c=3, device="cpu"):
        self.observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(0, 255, shape=(n, h, w, c), dtype=np.uint8)})
        self.action_space = gym.spaces.Box(-1, 1, shape=(n, 2), dtype=np.float32)
        g = torch.Generator().manual_seed(0)
        self.frame = torch.randint(0, 256, (n, h, w, c), generator=g, dtype=torch.uint8).to(device)
        self.n = n

    def reset(self, **kwargs):
        return {"policy": self.frame.clone()}, {}

    def step(self, action):
        z = torch.zeros(self.n, device=self.frame.device)
        return {"policy": self.frame.clone()}, z, z.bool(), z.bool(), {}


def _bt601(frame_nhwc):
    x = frame_nhwc.detach().cpu().numpy().astype(np.float64)
    return np.clip(np.round(x @ np.array([0.299, 0.587, 0.114])), 0, 255).astype(np.uint8)[..., None]


@pytest.mark.parametrize("device", DEVICES)
def test_luminance_is_bt601_one_channel_same_dtype_and_device(device):
    env = Grayscale(_StubEnv(device=device))
    obs, _ = env.reset()
    y = obs["policy"]
    assert y.dtype == torch.uint8 and y.device.type == device and tuple(y.shape) == (4, 28, 44, 1)
    assert np.abs(y.cpu().numpy().astype(int) - _bt601(env.env.frame).astype(int)).max() <= 1
    obs, *_ = env.step(None)
    assert tuple(obs["policy"].shape) == (4, 28, 44, 1)


def test_numpy_and_chw_inputs():
    env = Grayscale(_StubEnv())
    rgb = env.env.frame.numpy()
    out = env.observation(rgb)
    assert isinstance(out, np.ndarray) and out.shape == (4, 28, 44, 1)
    chw = env.observation(torch.as_tensor(rgb).permute(0, 3, 1, 2))
    assert tuple(chw.shape) == (4, 1, 28, 44)
    assert torch.equal(chw[:, 0], torch.as_tensor(out)[..., 0])


def test_pure_hues_map_to_their_weights():
    """A saturated cyan (background C's hue) and a grey of equal luminance become identical."""
    env = Grayscale(_StubEnv(n=1, h=4, w=4))
    cyan = torch.tensor([0, 200, 200], dtype=torch.uint8).expand(1, 4, 4, 3).clone()
    grey_level = round(0.587 * 200 + 0.114 * 200)
    grey = torch.full((1, 4, 4, 3), grey_level, dtype=torch.uint8)
    assert torch.equal(env.observation(cyan), env.observation(grey))


def test_space_is_one_channel_and_refuses_stacked_input():
    env = Grayscale(_StubEnv())
    assert env.observation_space["policy"].shape == (4, 28, 44, 1)
    with pytest.raises(ValueError, match="6 channels"):
        Grayscale(_StubEnv(c=6))


def test_first_conv_is_built_for_one_channel_through_the_real_chain():
    from nett_skrl.brain.encoders.nature_cnn import NatureCNN

    chain = ChannelsFirst(Grayscale(_StubEnv(h=64, w=96)))
    space = chain.observation_space["policy"]
    assert tuple(space.shape) == (4, 1, 64, 96)
    per_env = gym.spaces.Box(0, 255, shape=space.shape[1:], dtype=np.uint8)
    enc = NatureCNN(per_env, features_dim=16, conv_dim=8, spatial_pool=False)
    conv1 = next(m for m in enc.modules() if isinstance(m, torch.nn.Conv2d))
    assert conv1.in_channels == 1
    obs, _ = chain.reset()
    assert enc(obs["policy"].float()).shape == (4, 16)


def test_registry_and_labels_add_one_wrapper_and_nothing_else(monkeypatch):
    from pathlib import Path
    from nett_skrl.body.wrappers.registry import _load_wrapper

    assert _load_wrapper("grayscale") is Grayscale
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import campaign_train as campaign
    base = campaign.MODELS["CNN-UnityRecipe"]
    for label, name in (("CNN-UnityRecipe+Gray", "grayscale"), ("CNN-UnityRecipe+LumNorm", "lumnorm")):
        spec = campaign.MODELS[label]
        assert campaign.segmentation_wrappers(spec) == [name]
        assert {k: v for k, v in spec.items() if k != "pre"} == base
