"""Module for the Environment class."""

from __future__ import annotations

import logging
from pathlib import Path
import os
import subprocess
from typing import Optional, Any

import numpy as np

from gymnasium import Wrapper
from mlagents_envs.exception import UnityWorkerInUseException
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

from nett.utils.task import Task
from .utils import random_port, validate_executable_path

# Types
SINGLE_AGENT_STEP_RETURN = tuple[np.ndarray, float, bool, bool, dict]
MULTI_AGENT_STEP_RETURN = tuple[dict, dict, dict, dict, dict]


class Environment:
    """
    Represents the environment where the agent lives.

    The environment is the source of all input data streams to train the brain of the agent.
    It accepts a Unity Executable and wraps it around as a Gym environment by leveraging the UnityEnvironment
    class from the mlagents_envs library.

    It provides a convenient interface for interacting with the Unity environment and includes methods for initializing the environment, rendering frames, taking steps, resetting the environment, and logging messages.

    Args:
        executable_path (str): The path to the Unity executable file.
        display (int, optional): The display number to use for the Unity environment. Defaults to 0.

    Example:

        >>> from nett import Environment
        >>> env = Environment(executable_path="path/to/executable")
    """

    # Class Variables
    input_params: dict = {} # keeps track of the parameters that were fed into the class during initialization
    executable_path: Path  # the path to the Unity executable file
    multiagent: bool  # whether the environment is multiagent
    base_args: dict  # the base arguments to pass to the Unity environment
    initialized: bool = False  # keeps track of whether the class has been initialized
    multiobs: bool  # whether the environment passes multiple observations to the agent

    @classmethod
    def initialize(
        cls,
        executable_path: str,  # env
        record_eps: dict = {"train": 0, "test": 0},
        multiagent: bool = False,  # env
        display: Optional[int] = None,  # env
    ):
        # save all of the input parameters (executable_path etc) in case they are needed later
        cls.input_params.update({k: v for k, v in locals().items() if k != "cls"})

        cls.executable_path = validate_executable_path(executable_path)
        cls.multiagent = multiagent

        # set the correct permissions on the executable
        subprocess.run(["chmod", "-R", "755", executable_path], check=True)

        # Create a list of arguments to pass to the Unity environment
        args = []

        if display is None:
            # enable batchmode for headless servers
            args.append("-batchmode")
        else:
            # set the display for Unity environment
            os.environ["DISPLAY"] = str(f":{display}")

        # split into train and test args
        cls.base_args = {"train": args[:], "test": args[:]}

        # initialize as a random position in train
        cls.base_args["train"].extend(["--random-pos", "true"])

        # specify what to record
        for mode in ["train", "test"]:
            if record_eps.get(mode, 0) > 0:
                cls.base_args[mode].extend(
                    ["--record-chamber", "true", "--recording-steps", record_eps[mode]]
                )

        cls.initialized = True

    @classmethod
    def adjust_to_agent(
        cls,
        steps_per_episode: int,
        supervised_reward: bool,
        multiobs: bool,
    ):
        cls.multiobs = multiobs

        args = ["--episode-steps", str(steps_per_episode)]

        # add supervised reward
        if supervised_reward:
            args.extend(["--rewarded", "true"])

        for mode in ["train", "test"]:
            cls.base_args[mode].extend(args)

    def __init__(
        self, task: Task, validation_mode: bool, seed: Optional[int] = None
    ) -> None:
        # constructor method, opens the Unity environment

        # check if the class has been initialized
        if not self.initialized:
            raise RuntimeError(
                "Environment class must be initialized before creating an instance"
            )

        # set up logger
        self.logger = task.logger

        args = self.base_args[task.mode]

        # create record path
        recording_path = task.path / "env_recs"
        recording_path.mkdir(exist_ok=True, parents=True)

        if validation_mode:
            args.extend(["--validation-mode", "true"])

        # TODO: Figure out a way to run on multiple GPUs
        args.extend(
            [
                "--mode",
                f"{task.mode}-{task.condition}",  # set mode
                "--log-dir",
                str(recording_path),  # set log path
                "-force-device-index",
                str(task.device),  # set GPU
                "-gpu",
                str(task.device),  # set GPU
            ]
        )

        self.seed = seed if seed is not None else task.brain_id

        # create environment and connect it to logger
        complete = False
        while not complete:
            try:
                self.env = UnityEnvironment(
                    str(self.executable_path),
                    additional_args=args,
                    base_port=random_port(),
                    seed=self.seed,
                )
                complete = True
            except UnityWorkerInUseException as e:
                continue
            except Exception as e:
                self.logger.exception(f"Error initializing environment: {e}")
                raise e

    # converts the (c, w, h) frame returned by mlagents v1.0.0 and Unity 2022.3 to (w, h, c) as expected by gym
    # TODO: See if this is still necessary
    def render(self, mode="rgb_array") -> np.ndarray:  # pylint: disable=unused-argument
        # Renders the current frame of the environment.
        return np.moveaxis(self.env.render(), [0, 1, 2], [2, 0, 1])  # TODO: Why?

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> None | list[np.ndarray] | np.ndarray:  # pylint: disable=unused-argument
        # nothing to do if the wrapped env does not accept `seed`
        """
        Resets the environment with the given seed and arguments.

        Args:
            seed (int, optional): The seed to use for the environment. Defaults to None.
            **kwargs: The arguments to pass to the environment.

        Returns:
            numpy.ndarray: The initial state of the environment.
        """
        return self.env.reset(**kwargs)

    def step(
        self, action: list[Any]
    ) -> SINGLE_AGENT_STEP_RETURN | MULTI_AGENT_STEP_RETURN:
        # Takes a step in the environment with the given action.
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return next_state, reward, terminated, truncated, info


class GymEnvironment(Environment, Wrapper):
    # used for single-agent environments
    def __init__(
        self, task: Task, validation_mode: bool, seed: Optional[int] = None
    ) -> None:
        # init the Environment instance
        Environment.__init__(self, task, validation_mode, seed)

        # wrap the environment for ML Agents to work with Gym
        self.env = UnityToGymWrapper(
            self.env,
            uint8_visual=True,
            allow_multiple_obs=self.multiobs,
            action_space_seed=self.seed,
        )

        # init the Gym Wrapper instance
        Wrapper.__init__(self, self.env)

    def step(self, action: list[Any]) -> SINGLE_AGENT_STEP_RETURN:
        # step
        next_state, reward, terminated, truncated, info = super().step(action)
        # convert reward to float
        return next_state, float(reward), terminated, truncated, info


class ZooEnvironment(Environment, BaseParallelWrapper):
    # used for multi-agent environments
    def __init__(self, task: Task) -> None:
        # init the Environment instance
        Environment.__init__(self, task, False)

        # wrap the environment for ML Agents to work with PettingZoo
        self.env = UnityParallelEnv(self.env, uint8_visual=True, seed=self.seed)

        # init the PettingZoo BaseParallelWrapper instance
        BaseParallelWrapper.__init__(self, self.env)
