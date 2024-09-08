"""Module for the Environment class."""

from __future__ import annotations

import importlib
import os
import subprocess
from typing import Optional, Any

import numpy as np
import yaml

from gym import Wrapper
import mlagents_envs
from mlagents_envs.environment import UnityEnvironment

# checks to see if ml-agents tmp files have the proper permissions
try :
    from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
except PermissionError as _:
     raise PermissionError("Directory '/tmp/ml-agents-binaries' is not accessible. Please change permissions of the directory and its subdirectories ('tmp' and 'binaries') to 1777 or delete the entire directory and try again.")

from nett.utils.environment import Logger

class Environment(Wrapper):
    """
    Represents the environment where the agent lives.

    The environment is the source of all input data streams to train the brain of the agent. 
    It accepts a Unity Executable and wraps it around as a Gym environment by leveraging the UnityEnvironment 
    class from the mlagents_envs library.

    It provides a convenient interface for interacting with the Unity environment and includes methods for initializing the environment, rendering frames, taking steps, resetting the environment, and logging messages.

    Args:
        executable_path (str): The path to the Unity executable file.
        display (int, optional): The display number to use for the Unity environment. Defaults to 0.
        record_chamber (bool, optional): Whether to record the chamber. Defaults to False.
        record_agent (bool, optional): Whether to record the agent. Defaults to False.
        recording_frames (int, optional): The number of frames to record. Defaults to 1000.

    Example:

        >>> from nett import Environment
        >>> env = Environment(executable_path="path/to/executable")
    """
    def __init__(self,
                 executable_path: str,
                 display: int = 0,
                 record_chamber: bool = False,
                 record_agent: bool = False,
                 recording_frames: int = 1000) -> None:
        """Constructor method
        """
        from nett import logger
        self.logger = logger.getChild(__class__.__name__)

        self.executable_path = self._validate_executable_path(executable_path)
        self.record_chamber = record_chamber
        self.record_agent = record_agent
        self.recording_frames = recording_frames
        self.display = display
        # grab the experiment design from the executable directory
        self.num_test_conditions, self.imprinting_conditions = self._get_experiment_design(self.executable_path)

        # set the correct permissions on the executable
        self._set_executable_permission()
        # set the display for Unity environment
        self._set_display()

    # copied from __init__() of chickai_env_wrapper.py (legacy)
    # TODO (v0.4) Critical refactor, don't like how this works, extremely error prone.
    # how can we build + constraint arguments better? something like an ArgumentParser sounds neat
    # TODO (v0.4) fix random_pos logic inside of Unity code
    def initialize(self, mode: str, port: int, **kwargs) -> None:
        """
        Initializes the environment with the given mode and arguments.

        Args:
            mode (str): The mode to set the environment for training or testing or both.
            **kwargs: The arguments to pass to the environment.
        """
        importlib.reload(mlagents_envs)

        args = []

        # from environment arguments
        if self.recording_frames:
            args.extend(["--recording-steps", str(self.recording_frames)])
        if self.record_chamber:
            args.extend(["--record-chamber", "true"])
        if self.record_agent:
            args.extend(["--record-agent", "true"])

        # from runtime
        args.extend(["--mode", f"{mode}-{kwargs['condition']}"])
        if kwargs.get("rec_path", None):
            args.extend(["--log-dir", f"{kwargs['rec_path']}/"])
        # needs to fixed in Unity code where the default is always false
        if mode == "train":
            args.extend(["--random-pos", "true"])
        if kwargs.get("rewarded", False):
            args.extend(["--rewarded", "true"])
        self.step_per_episode = kwargs.get("episode_steps", 1000)
        args.extend(["--episode-steps", str(self.step_per_episode)])

        if kwargs["batch_mode"]:
            args.append("-batchmode")
        
        
        # TODO: Figure out a way to run on multiple GPUs
        if ("device" in kwargs):
            args.extend(["-force-device-index", str(kwargs["device"])])
            args.extend(["-gpu", str(kwargs["device"])])

        # create logger
        self.log = Logger(f"{kwargs['condition'].replace('-', '_')}{kwargs['brain_id']}-{mode}",
                          log_dir=f"{kwargs['log_path']}/")

        # create environment and connect it to logger
        self.env = UnityEnvironment(self.executable_path, side_channels=[self.log], additional_args=args, base_port=port)
        self.env = UnityToGymWrapper(self.env, uint8_visual=True, allow_multiple_obs=True) #TODO: Change this to vary base on Binocular Wrapper

        # initialize the parent class (gym.Wrapper)
        super().__init__(self.env)

    def log(self, msg: str) -> None:
        """
        Logs a message to the environment.

        Args:
            msg (str): The message to log.
        """
        self.log.log_str(msg)

    # converts the (c, w, h) frame returned by mlagents v1.0.0 and Unity 2022.3 to (w, h, c)
    # as expected by gym==0.21.0
    # HACK: mode is not used, but is required by the gym.Wrapper class (might be unnecessary but keeping for now)
    def render(self, mode="rgb_array") -> np.ndarray: # pylint: disable=unused-argument
        """
        Renders the current frame of the environment.

        Args:
            mode (str, optional): The mode to render the frame in. Defaults to "rgb_array".

        Returns:
            numpy.ndarray: The rendered frame of the environment.
        """
        return np.moveaxis(self.env.render(), [0, 1, 2], [2, 0, 1]) #TODO: Why?
    
    def reset(self, seed: Optional[int] = None, **kwargs) -> None | list[np.ndarray] | np.ndarray: # pylint: disable=unused-argument
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

    def step(self, action: list[Any]) -> tuple[np.ndarray, float, bool, dict]:
        """
        Takes a step in the environment with the given action.

        Args:
            action (list[Any]): The action to take in the environment.

        Returns:
            tuple[numpy.ndarray, float, bool, dict]: A tuple containing the next state, reward, done flag, and info dictionary.
        """
        next_state, reward, done, info = self.env.step(action)
        return next_state, float(reward), done, info

    def _set_executable_permission(self) -> None:
        """
        Sets the executable permission for the Unity executable file.
        """
        subprocess.run(["chmod", "-R", "755", self.executable_path], check=True)
        self.logger.info("Executable permission is set")

    def _set_display(self) -> None:
        """
        Sets the display environment variable for the Unity environment.
        """
        os.environ["DISPLAY"] = str(f":{self.display}")
        self.logger.info("Display is set")

    @staticmethod
    def _get_experiment_design(executable_path: str) -> tuple[int, list[str]]:
        """
        Gets the experiment design from the executable directory.

        Args:
            executable_path (str): The path to the Unity executable file.

        Returns:
            tuple[int, list[str]]: A tuple containing the number of test conditions and the list of imprinting conditions.

        Raises:
            FileNotFoundError: If the experiment configuration file is not found.
            KeyError: If the experiment configuration file is not properly formatted.
        """
        # get the experiment design from the executable directory
        parent_dir = os.path.dirname(executable_path)
        yaml_files: str = [file for file in os.listdir(parent_dir) if file.endswith(".yaml")]
        if not yaml_files:
            raise FileNotFoundError("No experiment configuration file found in the executable directory. You may be using a Unity executable meant for nett versions prior to v0.5.0. Please update the Unity executable to the latest version or use nett v0.4.1 or older.")

        yaml_file: str = os.path.join(parent_dir, yaml_files[0])

        # read the yaml file
        with open(yaml_file, "r") as file:
            yaml_data = yaml.safe_load(file)

            try:
                num_test_conditions: int = yaml_data["num_test_conditions"]
                imprinting_conditions: list[str] = yaml_data["imprinting_conditions"]
            except KeyError:
                raise KeyError("Experiment configuration file is not properly formatted. It should contain 'num_test_conditions' and 'imprinting_conditions' keys.")

        return num_test_conditions, imprinting_conditions

    @staticmethod
    def _validate_executable_path(executable_path: str) -> str:
        """
        Validates the Unity executable path.

        Args:
            executable_path (str): The path to the Unity executable file.

        Returns:
            str: The validated path to the Unity executable file.

        Raises:
            ValueError: If the executable path is not a string.
            FileNotFoundError: If the executable path does not exist.
            ValueError: If the executable path is not a valid Unity executable file.
            FileNotFoundError: If the directory does not contain the 'UnityPlayer.so' file.
            FileNotFoundError: If the data directory does not exist.
        """
        if not isinstance(executable_path, str):
            raise ValueError("executable_path should be a string. Instead, it is of type {type(executable_path)}")

        # check if the executable path exists
        if not os.path.exists(executable_path):
            raise FileNotFoundError(f"{executable_path} does not exist")
        
        # get the filename of the executable without '.x86_64'
        filename, extension = os.path.splitext(os.path.basename(executable_path))

        # check if executable is correct filetype
        if extension != ".x86_64" and extension != ".x86":
            raise ValueError(f"{executable_path} is not a valid Unity executable file")

        # check if the directory contains the 'UnityPlayer.so' file
        if not os.path.isfile(os.path.join(os.path.dirname(executable_path), "UnityPlayer.so")):
            raise FileNotFoundError(f"The directory {os.path.dirname(executable_path)} does not contain the file 'UnityPlayer.so'. This may not be a valid Unity executable.")

        # check if the data directory exists
        data_directory = filename + '_Data'
        if not os.path.isdir(os.path.join(os.path.dirname(executable_path), data_directory)):
            raise FileNotFoundError(f"Expected {data_directory} to exist in executable directory, but it does not exist. Please check that the path to the Unity executable is correct and that the data directory and executable use the same naming convention.")

        return executable_path

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"

    def __str__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"
