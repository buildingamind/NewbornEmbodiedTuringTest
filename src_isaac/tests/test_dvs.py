"""Sanity tests for the DVS observation wrapper — both raw and dict obs spaces."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from nett_skrl.wrappers import DVS


def _box(h: int = 32, w: int = 32, c: int = 3) -> gym.spaces.Box:
    return gym.spaces.Box(low=0, high=255, shape=(h, w, c), dtype=np.uint8)


class _RawEnv(gym.Env):
    """Returns a raw (H, W, C) uint8 obs per step — legacy gym shape."""
    metadata = {}

    def __init__(self) -> None:
        self.observation_space = _box()
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        return self._rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8), {}

    def step(self, action):
        return self._rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8), 0.0, False, False, {}


class _DictEnv(gym.Env):
    """Returns ``{"policy": (H, W, C)}`` — Isaac Lab `NETTEnv` shape."""
    metadata = {}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Dict({"policy": _box()})
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self._rng = np.random.default_rng(1)

    def reset(self, *, seed=None, options=None):
        return {"policy": self._rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)}, {}

    def step(self, action):
        return ({"policy": self._rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)},
                0.0, False, False, {})


def test_dvs_raw_obs_returns_3d_array():
    """Legacy raw-obs path still returns a 3D uint8 array.

    NETTEnv produces dict observations, so this path is vestigial; only
    asserting that the wrapper doesn't crash on the gym-style input.
    """
    env = DVS(_RawEnv())
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 3
    assert obs.dtype == np.uint8


def test_dvs_dict_obs_returns_dict_with_same_keys():
    env = DVS(_DictEnv())
    obs, _ = env.reset()
    assert isinstance(obs, dict)
    assert set(obs.keys()) == {"policy"}
    assert obs["policy"].ndim == 3
    assert obs["policy"].dtype == np.uint8


def test_dvs_dict_obs_step():
    env = DVS(_DictEnv())
    env.reset()
    obs, *_ = env.step(env.action_space.sample())
    assert isinstance(obs, dict)
    assert obs["policy"].shape[0] in (1, 3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
