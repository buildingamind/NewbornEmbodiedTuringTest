#!/usr/bin/env python3

import gym
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Binocular(gym.ObservationWrapper):
    """
    A gym observation wrapper for Binocular Vision.

    Args:
        env (gym.Env): The environment to wrap.

    Attributes:

        env (gym.Env): The wrapped environment.
        shape (tuple): The shape of the observation space.
        observation_space (gym.spaces.Box): The modified observation space.

    Methods:
        observation(obs): Performs the DVS transformation on the observation.
        reset(**kwargs): Resets the environment and returns the initial observation.

    """

    def __init__(self, env):
        super().__init__(env)

        try:
            _, _, width, height = self.env.observation_space.shape # stack, channels,
            self.shape=(6, width, height)
            self.observation_space = gym.spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)
        except Exception as e:
            raise e
    
    def observation(self, obs):
        """
        Performs the DVS transformation on the observation.

        Args:
            obs (list): The list of stacked frames.

        Returns:
            numpy.ndarray: The transformed observation.

        """
        left, right = obs

        # Concatenate the low and high bounds of the two spaces
        new_low = np.concatenate([left.low, right.low])
        new_high = np.concatenate([left.high, right.high])

        # Create a new Box space that combines both
        combined_obs = gym.spaces.Box(low=new_low, high=new_high, dtype=np.uint8)

        return combined_obs
    
    def reset(self, **kwargs):
        initial_obs = self.env.reset(**kwargs)
        return self.observation(initial_obs)
