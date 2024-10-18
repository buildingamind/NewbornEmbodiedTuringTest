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
            channels, width, height = self.env.observation_space[0].shape
            self.shape=(2*channels, width, height)
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

        # Combine two observations into a single observation
        combined_obs = np.concatenate([left, right])

        return combined_obs
    
    def reset(self, **kwargs):
        initial_obs = self.env.reset(**kwargs)
        return self.observation(initial_obs)
    
    def seed(self, seed: int) -> None:
        """
        Seed the environment.

        Args:
            seed (int): The seed value.
        """
        self.env.seed(seed)
