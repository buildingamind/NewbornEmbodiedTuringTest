"""Wrapper for converting tuple observations to dictionary format for multi-input policies."""

#!/usr/bin/env python3

import gymnasium as gym

# import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _tuple_to_dict(tup: tuple):
    """
    Converts a tuple to a dictionary.
    """
    return {f"obs {k}": v for k, v in enumerate(tup)}


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
        observation(obs): Converts tuple observation to dictionary observation.
        reset(**kwargs): Resets the environment and returns the initial observation.

    """

    def __init__(self, env, *args, **kwargs): # pylint: disable=unused-argument
        super().__init__(env)
        # convert Tuple to Dict
        self.observation_space = gym.spaces.Dict(_tuple_to_dict(env.observation_space))

    def observation(self, obs):
        """
        Converts tuple observation to dictionary observation.

        Args:
            obs (list): The list of stacked frames.

        Returns:
            dict: The transformed observation.

        """
        return _tuple_to_dict(obs)

    def reset(self, **kwargs):
        """
        Reset the environment and return the transformed dictionary observation.

        Args:
            **kwargs: Additional arguments passed to the environment reset.

        Returns:
            tuple: Dictionary observation and info dictionary.
        """
        initial_obs, initial_info = self.env.reset(**kwargs)
        return self.observation(initial_obs), initial_info
