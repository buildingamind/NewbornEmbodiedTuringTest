"""
Represents the environment where the agent lives.

The environment is the source of all input data streams to train the brain of the agent.
It accepts a Unity Executable and wraps it around as a Gym environment by leveraging the UnityEnvironment
class from the mlagents_envs library.

It provides a convenient interface for interacting with the Unity environment and includes methods for initializing the environment, rendering frames, taking steps, resetting the environment, and logging messages.
"""

import copy
from pathlib import Path
import os
import subprocess
from typing import Optional, Any

import numpy as np
import gymnasium as gym

from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper

from .utils.logger import Logger
from ..utils.task import TaskConfig

# checks to see if ml-agents tmp files have the proper permissions
try:
    from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
except PermissionError as _:
    raise PermissionError(
        "Directory '/tmp/ml-agents-binaries' is not accessible. Please change permissions of the directory and its subdirectories ('tmp' and 'binaries') to 1777 or delete the entire directory and try again."
    )

from .utils import (
    validate_executable_path,
    validate_conditions,
    get_experiment_design,
    random_port,
)


class Environment:
    """
    Args:
        executable_path (str): The path to the Unity executable file.
        conditions (list[str], optional): A list of imprinting conditions to run. If None, all available imprinting conditions will be run. For a list of available imprinting condtions from an executable, run `nett.list_conditions` :func:`~nett.nett.list_conditions`. Defaults to `None`.
        record_eps (dict[str, int]): Dictionary specifying the number of episodes to record the entire chamber for each mode (train and test). Defaults to `{"train": 0, "test": 0}`
        timescale (float): The timescale to run the Unity environment at. This multiplies the speed of the simulation from realtime. A value below 1 slows down the simulation. A value above 1 speeds up the simulation. Setting the value too high may result in skipping steps within Unity environment. Defaults to `20.0`.
        multiagent (bool): Flag to indicate if environment is a multiagent environment (beta). Defaults to `False`.
        display (int, optional): The display number to use for the Unity environment. If None, the environment will be run headless. Defaults to `None`.
    """

    # Class Variables
    executable_path: Path  # the path to the Unity executable file
    num_test_conditions: int  # the number of test conditions
    conditions: list[str]  # the imprinting conditions to run
    multiagent: bool  # whether the environment is multiagent
    base_args: dict[str, list]  # the base arguments to pass to the Unity environment
    multiobs: bool  # whether the environment passes multiple observations to the agent
    env: UnityEnvironment  # the Unity environment

    def __init__(
        self,
        executable_path: str,
        conditions: Optional[list[str]] = None,
        record_eps: dict = {"train": "0", "test": "0"},
        timescale: float = 20.0, # TODO Change this to decision period
        multiagent: bool = False,
        display: Optional[int] = None,
    ):
        self.executable_path = validate_executable_path(executable_path)

        # get experiment design
        self.num_test_conditions, valid_imprinting_conditions = get_experiment_design(
            self.executable_path
        )

        # validate conditions
        self.conditions = validate_conditions(
            valid_imprinting_conditions, conditions
        )  # multi

        self.multiagent = multiagent

        # set the correct permissions on the executable
        subprocess.run(["chmod", "-R", "755", executable_path], check=True)

        # Create a list of arguments to pass to the Unity environment
        args = ["timescale", str(timescale)]

        if display is None:
            # enable batchmode for headless servers
            args.append("-batchmode")
            os.environ["DISPLAY"] = str(f":0")
        else:
            # set the display for Unity environment
            os.environ["DISPLAY"] = str(f":{display}")

        # split into train and test args
        self.base_args = {"train": args[:], "test": args[:]}

        # specify what to record
        for mode in ["train", "test"]:
            self.base_args[mode].extend(
                [
                    "--record-episodes",
                    record_eps.get(mode, "0"),
                ]
            )

    def adjust_to_agent(
        self,
        steps_per_episode: int,
        reward: str,
        multiobs: bool,
        panini: bool = False,
    ):
        self.multiobs = multiobs

        args = ["--episode-steps", str(steps_per_episode)]

        if multiobs:  # TODO: Make this so it is binocular specific
            args.append("--binocular")
        if panini:
            args.append("--panini-projection")

        if reward in {
            "closeness",
            "completeness",
            "closeness,completeness",
        }:  # TODO: Clean this up
            args.extend(["--reward", reward])

        for mode in ["train", "test"]:
            self.base_args[mode].extend(args)

    def load(
        self, config: TaskConfig, validation_mode: bool, seed: Optional[int] = None
    ) -> gym.Env:  # logger, brain_id, path, current_mode, device, condition
        # constructor method, opens the Unity environment

        # check if vec env is being run in parallel
        if seed is not None:
            logger = config.logger.getChild(str(seed))
        else:
            logger = config.logger
            seed = config.brain_id

        # create record path
        recording_path = config.path / "recordings"
        recording_path.mkdir(exist_ok=True, parents=True)

        # create Unity args
        args = copy.deepcopy(self.base_args[config.current_mode])

        if validation_mode:
            args.append("--validation-mode")

        # TODO: Figure out a way to run on multiple GPUs
        args.extend(
            [
                "--phase",
                config.current_mode,  # set mode
                "--imprint-condition",
                config.condition,  # set mode
                "--record-path",
                str(recording_path),  # set log path
                "-force-device-index",
                str(config.device),  # set GPU
                "-gpu",
                str(config.device),  # set GPU
            ]
        )

        # create environment and connect it to logger
        # create logger
        # create log path
        log_path = config.path / "logs"
        log_dir = (
            log_path
            / f"{config.current_mode}_{config.condition}_{config.brain_id}_{seed or ''}.csv"
        )
        args.extend(["--log-path", str(log_dir)])
        # side_channels = [
        #     Logger(
        #         f"{config.current_mode}_{config.condition}_{config.brain_id}_{seed}",
        #         log_dir=str(log_path),
        #     )
        # ]

        complete = False
        while not complete:
            try:
                env = UnityEnvironment(
                    str(self.executable_path),
                    additional_args=args,
                    base_port=random_port(),
                    seed=seed,
                    # side_channels=side_channels,
                )
                env.render_mode = "rgb_array_list" if self.multiobs else "rgb_array"
                complete = True
            except UnityWorkerInUseException as e:
                continue
            except Exception as e:
                logger.exception(f"Error initializing environment: {e}")
                raise e

        return (
            ZooWrapper(env, seed)
            if self.multiagent
            else GymWrapper(env, seed, self.multiobs)
        )


class BaseWrapper:
    """Base Wrapper for Unity environment"""

    # converts the (c, w, h) frame returned by mlagents v1.0.0 and Unity 2022.3 to (w, h, c) as expected by gym
    # TODO: See if this is still necessary
    def render(self, mode="rgb_array") -> np.ndarray:  # pylint: disable=unused-argument
        # Renders the current frame of the environment.
        return np.moveaxis(self.env.render(), [0, 1, 2], [2, 0, 1])  # TODO: Why?

    def reset(
        self, seed: Optional[int] = None, **kwargs
    ) -> None | list[np.ndarray] | np.ndarray:  # pylint: disable=unused-argument
        # nothing to do if the wrapped env does not accept `seed`
        return self.env.reset(**kwargs)


class GymWrapper(BaseWrapper, gym.Wrapper):
    """Wrapper to adapt Unity environment to Gymnasium"""

    def __init__(self, env: UnityEnvironment, seed: int, multiobs: bool):

        # wrap the environment for ML Agents to work with Gym
        self.env = UnityToGymWrapper(
            env,
            uint8_visual=True,
            allow_multiple_obs=multiobs,
            seed=seed,
        )
        # init the Gym Wrapper instance
        gym.Wrapper.__init__(self, self.env)

        self.env.render_mode = "rgb_array_list" if multiobs else "rgb_array"

    def step(self, action: list[Any]) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Takes a step in the environment with the given action.
        next_state, reward, terminated, truncated, info = self.env.step(action)
        return next_state, float(reward), terminated, truncated, info

    # def kill(self):
    #     # immediately kill the environment rather than waiting
    #     self.env._env._close(0)


class ZooWrapper(BaseWrapper, BaseParallelWrapper):
    """Wrapper to adapt Unity environment to PettingZoo"""

    def __init__(self, env: UnityEnvironment, seed: int):
        # wrap the environment for ML Agents to work with PettingZoo
        self.env = UnityParallelEnv(env, uint8_visual=True, seed=seed)

        # init the PettingZoo BaseParallelWrapper instance
        BaseParallelWrapper.__init__(self, self.env)
