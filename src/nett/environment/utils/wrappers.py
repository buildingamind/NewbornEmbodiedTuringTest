import numpy as np
from typing import Optional, Any
import gymnasium as gym

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper

# checks to see if ml-agents tmp files have the proper permissions
try:
    from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
except PermissionError as _:
    raise PermissionError(
        "Directory '/tmp/ml-agents-binaries' is not accessible. Please change permissions of the directory and its subdirectories ('tmp' and 'binaries') to 1777 or delete the entire directory and try again."
    )

class BaseWrapper:
    """Base Wrapper for Unity environment"""

    # Converts the (c, w, h) frame returned by mlagents v1.0.0 and Unity 6 to (w, h, c) as expected by gym
    # TODO: See if this is still necessary
    def render(self, mode="rgb_array") -> np.ndarray:  # pylint: disable=unused-argument
        """
        Renders the current frame of the environment.

        Args:
            mode (str, optional): The rendering mode. Defaults to "rgb_array".

        Returns:
            np.ndarray: The rendered frame as a numpy array.
        """
        # The axis move is to convert from PyTorch's CHW format to TensorFlow/OpenCV's HWC format.
        return np.moveaxis(self.env.render(), [0, 1, 2], [2, 0, 1])

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> None | list[np.ndarray] | np.ndarray:  # pylint: disable=unused-argument
        """
        Resets the environment.

        Args:
            seed (Optional[int], optional): The random seed. Defaults to None.

        Returns:
            None | list[np.ndarray] | np.ndarray: The initial observation after reset.
        """
        # The wrapped environment's reset is called. Nothing to do here if it doesn't accept a seed.
        return self.env.reset(**kwargs)


class GymWrapper(BaseWrapper, gym.Wrapper):
    """Wrapper to adapt Unity environment to Gymnasium"""

    def __init__(self, env: UnityEnvironment, seed: int, multiobs: bool):
        """
        Initializes the GymWrapper.

        Args:
            env (UnityEnvironment): The Unity environment to wrap.
            seed (int): The random seed.
            multiobs (bool): Flag to indicate if multiple observations are used.
        """

        # Wrap the environment for ML-Agents to work with Gymnasium
        self.env = UnityToGymWrapper(
            env,
            uint8_visual=True,
            allow_multiple_obs=multiobs,
            seed=seed,
        )
        # Initialize the Gym Wrapper instance
        gym.Wrapper.__init__(self, self.env)

        # Set the render mode based on whether multiple observations are used
        self.env.render_mode = "rgb_array_list" if multiobs else "rgb_array"

    def step(self, action: list[Any]) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes a step in the environment with the given action.

        Args:
            action (list[Any]): The action to take.

        Returns:
            tuple[np.ndarray, float, bool, bool, dict]: A tuple containing the next state, reward, terminated flag, truncated flag, and info dictionary.
        """
        # Take a step in the wrapped environment
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # Ensure reward is a float
        return next_state, float(reward), terminated, truncated, info

    # def kill(self):
    #     # immediately kill the environment rather than waiting
    #     self.env._env._close(0)


class ZooWrapper(BaseWrapper, BaseParallelWrapper):
    """Wrapper to adapt Unity environment to PettingZoo for multi-agent scenarios."""

    def __init__(self, env: UnityEnvironment, seed: int):
        """
        Initializes the ZooWrapper.

        Args:
            env (UnityEnvironment): The Unity environment to wrap.
            seed (int): The random seed.
        """
        # Wrap the environment for ML-Agents to work with PettingZoo
        self.env = UnityParallelEnv(env, uint8_visual=True, seed=seed)

        # Initialize the PettingZoo BaseParallelWrapper instance
        BaseParallelWrapper.__init__(self, self.env)
