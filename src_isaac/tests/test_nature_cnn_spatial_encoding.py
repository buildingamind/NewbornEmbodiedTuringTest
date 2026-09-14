"""Spatial host features and single-frame policy input on stacked observations."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from nett_skrl.brain.encoders.base import NETTFeatureExtractor
from nett_skrl.brain.encoders.nature_cnn import NatureCNN
from nett_skrl.brain.encoders.utils.pool import DeterministicAvgPool2d


@pytest.fixture(autouse=True)
def cpu_threads():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    torch.set_num_threads(threads)


def space(channels=6):
    return gym.spaces.Box(0, 255, (80, 128, channels), dtype=np.uint8)


def test_base_spatial_contract_raises_with_class_name():
    with pytest.raises(NotImplementedError, match="NETTFeatureExtractor"):
        NETTFeatureExtractor(space()).encode_spatial(torch.zeros(2, 3, 80, 128))


@pytest.mark.parametrize("pooled", [True, False])
def test_spatial_readout_and_forward_are_bit_identical_to_original(pooled):
    host = NatureCNN(space(), features_dim=32, spatial_pool=pooled)
    obs = torch.randint(0, 256, (2, 80, 128, 6), dtype=torch.uint8)
    # Original forward expression, with unchanged initialization and state keys.
    expected = host.linear(host.cnn(host._prepare_image(obs)))
    spatial = host.encode_spatial(obs)
    assert spatial.shape == (2, 64, 6, 12)
    readout = spatial
    for layer in host.cnn:
        if isinstance(layer, (DeterministicAvgPool2d, nn.Flatten)):
            readout = layer(readout)
    torch.testing.assert_close(host.linear(readout), expected, rtol=0, atol=0)
    torch.testing.assert_close(host(obs), expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["hwc", "chw", "flat_hwc", "flat_chw"])
@pytest.mark.parametrize("input_frames", [1, 2])
def test_policy_and_spatial_map_use_only_last_frames(layout, input_frames):
    obs_space = space(9)
    if "chw" in layout:
        obs_space = gym.spaces.Box(0, 255, (9, 80, 128), dtype=np.uint8)
    host = NatureCNN(obs_space, features_dim=32, input_frames=input_frames)
    assert next(m for m in host.cnn if isinstance(m, nn.Conv2d)).in_channels == 3 * input_frames
    obs = torch.randint(0, 256, (2, 80, 128, 9), dtype=torch.uint8)
    altered = obs.clone()
    altered[..., :-3 * input_frames] = 255 - altered[..., :-3 * input_frames]

    def arrange(x):
        if "chw" in layout:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x.flatten(1) if layout.startswith("flat") else x

    for method in (host.forward, host.encode_spatial):
        torch.testing.assert_close(method(arrange(obs)), method(arrange(altered)), rtol=0, atol=0)
        current = obs[..., -3 * input_frames:].permute(0, 3, 1, 2)
        torch.testing.assert_close(method(arrange(obs)), method(current), rtol=0, atol=0)
    prepared = obs[..., -3 * input_frames:].permute(0, 3, 1, 2).float() / 255
    torch.testing.assert_close(host.encode_spatial_prepared(prepared),
                               host.encode_spatial(arrange(obs)), rtol=0, atol=0)
    assert not host._skip_prepare


@pytest.mark.parametrize("channels,input_frames", [(2, 1), (4, 1), (3, 2), (7, 2)])
def test_invalid_frame_channels_raise_at_construction_and_encoding(channels, input_frames):
    with pytest.raises(ValueError, match="RGB channels"):
        NatureCNN(space(channels), input_frames=input_frames)
    host = NatureCNN(space(), input_frames=input_frames)
    for method in (host.forward, host.encode_spatial):
        with pytest.raises(ValueError, match="RGB channels"):
            method(torch.zeros(2, channels, 80, 128))


@pytest.mark.parametrize("input_frames", [0, -1, 1.5, True, "1"])
def test_input_frames_requires_positive_integer(input_frames):
    with pytest.raises(ValueError, match="positive integer"):
        NatureCNN(space(), input_frames=input_frames)
