#!/usr/bin/env python3
import gym
import numpy as np 

class ObservationWrapper(gym.ObservationWrapper):
    """
    Gym env wrapper that transpose visual observations to (H, W, C).
    """
    def __init__(self, env):
        super().__init__(env)

        # Assumes visual observation space with shape (H, W, C)
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        assert len(self.env.observation_space.shape) == 3

        #h, w, c = env.observation_space.shape
        #self.observation_space = gym.spaces.Box(0, 1, dtype=np.float32, shape=(c, h, w))
        
        channels, width, height = self.env.observation_space.shape
        new_shape = (width, height, channels)
        self.observation_space = gym.spaces.Box(low=0.0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return np.swapaxes(observation, 2, 0)