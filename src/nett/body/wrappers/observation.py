#!/usr/bin/env python3
import gym
import numpy as np 

class ObservationWrapper(gym.ObservationWrapper):
    """
    Gym env wrapper that transpose visual observations to (C,H,W).
    """
    def __init__(self, env):
        super().__init__(env)

        # Assumes visual observation space with shape (H, W, C)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert len(env.observation_space.shape) == 3

        self.observation_space = gym.spaces.Box(low=0.0, high=255, shape=env.observation_space.shape, dtype=np.uint8)
