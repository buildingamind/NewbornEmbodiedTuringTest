from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from nett_skrl.wrappers import Video


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
