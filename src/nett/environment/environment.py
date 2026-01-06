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
from typing import Optional

import gymnasium as gym

from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.environment import UnityEnvironment
import torch

from ..utils.task import TaskConfig
from .utils import GymWrapper, ZooWrapper

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
        multiagent (bool): Flag to indicate if environment is a multiagent environment (beta). Defaults to `False`.
        random_first_frame (bool): Flag to randomize the first frame of episodes. Defaults to `False`.
        display (int, optional): The display number to use for the Unity environment. If None, the environment will be run headless. Defaults to `None`.
    """

    # Class Variables
    executable_path: Path  # the path to the Unity executable file
    num_test_conditions: int  # the number of test conditions
    conditions: list[str]  # the imprinting conditions to run
    multiagent: bool  # whether the environment is multiagent
    base_args: dict[str, list]  # the base arguments to pass to the Unity environment
    binocular_vision: (
        bool  # whether the environment passes multiple observations to the agent
    )
    env: UnityEnvironment  # the Unity environment

    def __init__(
        self,
        executable_path: str,
        conditions: Optional[list[str]] = None,
        record_eps: dict = {"train": "0", "test": "0"},
        multiagent: bool = False,
        random_first_frame: bool = False,
        display: Optional[int] = None,
    ):
        """
        Initializes the Environment object.
        """
        # Validate the executable path
        self.executable_path = validate_executable_path(executable_path)

        # Get experiment design from the Unity executable
        # self.num_test_conditions, valid_imprinting_conditions = get_experiment_design(
        valid_imprinting_conditions = get_experiment_design(self.executable_path)

        # Validate the provided imprinting conditions against the valid conditions
        self.conditions: list[str] = validate_conditions(
            valid_imprinting_conditions, conditions
        )  # multi

        # Store the number of iterations for each test episode
        self.iterations_per_test_episode: dict[str, int] = {
            k: v for k, v in valid_imprinting_conditions.items() if k in self.conditions
        }

        self.multiagent = multiagent
        self.random_first_frame = random_first_frame

        # Set the correct permissions on the executable to ensure it can be run
        subprocess.run(["chmod", "-R", "755", executable_path], check=True)

        # Create a list of base arguments to pass to the Unity environment
        args = []

        if display is None:
            # Enable batchmode for headless servers (no display)
            args.append("-batchmode")
            os.environ["DISPLAY"] = str(f":0")
        else:
            # Set the display for the Unity environment
            os.environ["DISPLAY"] = str(f":{display}")

        if self.random_first_frame:
            args.append("--random-first-frame")

        # Split arguments for training and testing modes
        self.base_args = {"train": args[:], "test": args[:]}

        # Add recording arguments for each mode
        for mode in ["train", "test"]:
            self.base_args[mode].extend(
                [
                    "--record-episodes",
                    record_eps.get(mode, "0")
                ]
            )

    def adjust_to_agent(
        self,
        steps_per_episode: int,
        reward: str,
        binocular_vision: bool,
        panini: bool = False,
        input_resolution: Optional[int] = None,
    ):
        """
        Adjusts the environment settings based on the agent's configuration.
        """
        self.binocular_vision = binocular_vision

        # Add arguments based on agent configuration
        args = ["--episode-steps", str(steps_per_episode)]

        if input_resolution is not None:
            args.extend(["--input-resolution", str(input_resolution)])

        if binocular_vision:
            args.append("--binocular")
        if panini:
            args.append("--panini-projection")

        # Add reward function argument
        if reward in {
            "closeness",
            "completeness",
            "closeness,completeness",
        }:  # TODO: Clean this up
            args.extend(["--reward", reward])

        # Extend base arguments for both train and test modes
        for mode in ["train", "test"]:
            self.base_args[mode].extend(args)

    def load(
        self, config: TaskConfig, validation_mode: bool, seed: Optional[int] = None
    ) -> gym.Env:  # logger, brain_id, path, current_mode, device, condition
        """
        Loads the Unity environment with the specified configuration.
        """
        # This method opens the Unity environment.

        # Check if running in parallel and adjust logger and seed accordingly
        if seed is not None:
            logger = config.logger.getChild(str(seed))
        else:
            logger = config.logger
            seed = config.brain_id

        # Set the random seed for reproducibility
        torch.manual_seed(seed)

        # Create path for recordings
        recording_path = config.path / "recordings"
        recording_path.mkdir(exist_ok=True, parents=True)

        # Create a deep copy of the base arguments for the current mode
        args = copy.deepcopy(self.base_args[config.current_mode])

        if validation_mode:
            args.append("--validation-mode")

        # Extend arguments with task-specific configurations
        # TODO: Figure out a way to run on multiple GPUs
        args.extend(
            [
                "--phase",
                config.current_mode,  # set mode (train/test)
                "--imprint-condition",
                config.condition,  # set imprinting condition
                "--record-path",
                str(recording_path),  # set recording path
                "-force-device-index",
                str(config.device),  # set GPU device
                "-gpu",
                str(config.device),  # set GPU device
            ]
        )

        # Create log path and extend arguments
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
        # Loop to handle UnityWorkerInUseException, which can occur during parallel environment initialization
        while not complete:
            try:
                # Initialize the Unity environment
                env = UnityEnvironment(
                    str(self.executable_path),
                    additional_args=args,
                    base_port=random_port(),  # Use a random port to avoid conflicts
                    seed=seed,
                    # side_channels=side_channels,
                )
                # Set render mode based on whether multiple observations are expected
                env.render_mode = (
                    "rgb_array_list" if self.binocular_vision else "rgb_array"
                )
                complete = True
            except UnityWorkerInUseException as e:
                # If the worker is in use, try again
                continue
            except Exception as e:
                logger.exception(f"Error initializing environment: {e}")
                raise e

        # Return the appropriate wrapper for the environment (multi-agent or single-agent)
        return (
            ZooWrapper(env, seed)
            if self.multiagent
            else GymWrapper(env, seed, self.binocular_vision)
        )
