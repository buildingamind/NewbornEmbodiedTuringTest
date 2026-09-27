from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.body.wrappers import Video
from nett_skrl.body.wrappers.framestack import FrameStack


class _DictEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(0, 255, (4, 4, 3), dtype=np.uint8)}
        )

    def reset(self, *, seed=None, options=None):
        return {"policy": np.zeros((4, 4, 3), dtype=np.uint8)}, {}

    def step(self, action):
        return {"policy": np.ones((4, 4, 3), dtype=np.uint8)}, 0.0, False, False, {}


def test_video_stacks_policy_frames_for_dict_obs():
    env = Video(_DictEnv(), frames=3)
    obs, _ = env.reset()
    assert obs["policy"].shape == (4, 4, 9)
    assert env.observation_space["policy"].shape == (4, 4, 9)
    obs, *_ = env.step(np.zeros(1, dtype=np.float32))
    assert obs["policy"].shape == (4, 4, 9)


class _TensorDictEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.observation_space = gym.spaces.Dict(
            {"policy": gym.spaces.Box(0, 255, (4, 4, 3), dtype=np.uint8)}
        )

    def reset(self, *, seed=None, options=None):
        return {
            "policy": torch.zeros((4, 4, 3), dtype=torch.uint8, device=self.device),
            "extra": "kept",
        }, {}

    def step(self, action):
        return {
            "policy": torch.ones((4, 4, 3), dtype=torch.uint8, device=self.device),
            "extra": "kept",
        }, 0.0, False, False, {}


def test_video_stacks_tensor_policy_without_numpy_conversion():
    env = Video(_TensorDictEnv(), frames=2)
    obs, _ = env.reset()
    assert isinstance(obs["policy"], torch.Tensor)
    assert obs["policy"].device.type == "cpu"
    assert obs["policy"].shape == (4, 4, 6)
    assert obs["extra"] == "kept"
    obs, *_ = env.step(torch.zeros(1))
    assert isinstance(obs["policy"], torch.Tensor)
    assert obs["policy"].shape == (4, 4, 6)


def test_video_stacks_cuda_tensor_policy_without_cpu_transfer():
    if not torch.cuda.is_available():
        return
    env = Video(_TensorDictEnv("cuda"), frames=2)
    obs, _ = env.reset()
    assert isinstance(obs["policy"], torch.Tensor)
    assert obs["policy"].device.type == "cuda"
    assert obs["policy"].shape == (4, 4, 6)


class _BatchedDictEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, (2, 1), dtype=np.float32)

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(0, 255, (2, 4, 4, 3), dtype=np.uint8),
                "critic": gym.spaces.Box(-1.0, 1.0, (2, 5), dtype=np.float32),
            }
        )

    def reset(self, *, seed=None, options=None):
        return {
            "policy": np.zeros((2, 4, 4, 3), dtype=np.uint8),
            "critic": np.zeros((2, 5), dtype=np.float32),
        }, {}

    def step(self, action):
        return {
            "policy": np.ones((2, 4, 4, 3), dtype=np.uint8),
            "critic": np.ones((2, 5), dtype=np.float32),
        }, np.zeros(2), np.array([True, False]), np.array([False, False]), {}


def test_framestack_stacks_batched_dict_policy_and_preserves_critic():
    env = FrameStack(_BatchedDictEnv(), n_stack=2)

    obs, _ = env.reset()

    assert env.observation_space["policy"].shape == (2, 4, 4, 6)
    assert env.observation_space["critic"].shape == (2, 5)
    assert obs["policy"].shape == (2, 4, 4, 6)
    assert obs["critic"].shape == (2, 5)

    obs, *_ = env.step(np.zeros((2, 1), dtype=np.float32))

    assert obs["policy"].shape == (2, 4, 4, 6)
    assert obs["critic"].shape == (2, 5)


class _BatchedTensorDoneEnv(_BatchedDictEnv):
    """Like _BatchedDictEnv but returns terminated/truncated as torch tensors on `device`."""

    def __init__(self, device):
        super().__init__()
        self.device = device

    def step(self, action):
        obs, rew, term, trunc, info = super().step(action)
        return (obs, rew, torch.as_tensor(term, device=self.device),
                torch.as_tensor(trunc, device=self.device), info)


def _check_scrub(device):
    env = FrameStack(_BatchedTensorDoneEnv(device), n_stack=2)
    env.reset()                                  # frames: zeros
    obs, *_ = env.step(np.zeros((2, 1), dtype=np.float32))  # env 0 done, env 1 not; new obs = ones
    pol = obs["policy"]
    # env 0 ended: BOTH stacked frames are the new episode's first obs (no straddle)
    assert (pol[0] == 1).all()
    # env 1 continues: old frame (zeros) then new frame (ones)
    assert (pol[1, ..., :3] == 0).all() and (pol[1, ..., 3:] == 1).all()


def test_framestack_scrubs_on_cpu_tensor_done_flags():
    _check_scrub("cpu")


def test_framestack_scrubs_on_cuda_tensor_done_flags():
    # The live Isaac case: CUDA done flags. Before the fix _done_mask returned None here
    # and the scrub silently never ran.
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device: the live-Isaac case was NOT checked")
    _check_scrub("cuda")
