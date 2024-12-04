#!/usr/bin/env python3

import gymnasium as gym

# import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MultiObs(gym.ObservationWrapper):
    """
    A gym observation wrapper for using SB3 MultiInputPolicy.

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
        # convert Tuple to Dict
        self.observation_space = gym.spaces.Dict(dict(enumerate(env.observation_space)))

    def observation(self, obs):
        """
        Performs the DVS transformation on the observation.

        Args:
            obs (list): The list of stacked frames.

        Returns:
            numpy.ndarray: The transformed observation.

        """
        return gym.spaces.Dict(dict(enumerate(obs)))
