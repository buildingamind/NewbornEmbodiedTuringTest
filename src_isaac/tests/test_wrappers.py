from __future__ import annotations

import gymnasium as gym
import numpy as np

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
