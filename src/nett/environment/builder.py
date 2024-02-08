"""Module for the Environment class."""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import numpy as np
from gym import Wrapper
from mlagents_envs.environment import UnityEnvironment

# checks to see if ml-agents tmp files have the proper permissions
try :
    from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
except PermissionError as _:
     raise PermissionError("Directory '/tmp/ml-agents-binaries' is not accessible. Please change permissions of the directory and its subdirectories ('tmp' and 'binaries') to 1777 or delete the entire directory and try again.")
# NOTE: Import was causing circular import error (nett.environment -> environment) (nett.utils -> utils)
from nett.environment.configs import NETTConfig, list_configs
from nett.environment import configs
from nett.utils.environment import Logger, port_in_use

class Environment(Wrapper):
    """
    A wrapper around the UnityEnvironment class from the mlagents_envs library.

    This class provides a convenient interface for interacting with the Unity environment and includes methods for initializing the environment, rendering frames, taking steps, resetting the environment, and logging messages.

    The Environment class inherits from the gym.Wrapper class, allowing it to be used as a gym environment.

    :param config: The configuration for the environment. It can be either a string representing the name of a pre-defined configuration, or an instance of the NETTConfig class.
    :type config: str | NETTConfig
    :param executable_path: The path to the Unity executable file.
    :type executable_path: str
    :param display: The display number to use for the Unity environment. Defaults to 0.
    :type display: int, optional
    :param base_port: The base port number to use for communication with the Unity environment. Defaults to 5004.
    :type base_port: int, optional
    :param record_chamber: Whether to record the chamber. Defaults to False.
    :type record_chamber: bool, optional
    :param record_agent: Whether to record the agent. Defaults to False.
    :type record_agent: bool, optional
    :param recording_frames: The number of frames to record. Defaults to 1000.
    :type recording_frames: int, optional

    :raises ValueError: If the configuration is not a valid string or an instance of NETTConfig.

    Example:

    >>> from nett import Environment
    >>> env = Environment(config="identityandview", executable_path="path/to/executable")
    """
    def __init__(self,
                 config: str | NETTConfig,
                 executable_path: str,
                 display: int = 0,
                 base_port: int = 5004,
                 record_chamber: bool = False,
                 record_agent: bool = False,
                 recording_frames: int = 1000) -> None:
        """Constructor method
        """
        from nett import logger
        self.logger = logger.getChild(__class__.__name__)
        self.config = self._validate_config(config)
        # TODO (v0.4) what might be a way to check if it is a valid executable path?
        self.executable_path = executable_path
        self.base_port = base_port
        self.record_chamber = record_chamber
        self.record_agent = record_agent
        self.recording_frames = recording_frames
        self.display = display

        # set the correct permissions on the executable
        self._set_executable_permission()
        # set the display for Unity environment
        self._set_display()

    def _validate_config(self, config: str | NETTConfig) -> NETTConfig:
        """
        Validates the configuration for the environment.

        :param config: The configuration to validate.
        :type config: str | NETTConfig
        :return: The validated configuration.
        :rtype: NETTConfig
        :raises ValueError: If the configuration is not a valid string or an instance of NETTConfig.
        """
        # for when config is a str
        if isinstance(config, str):
            config_dict = {config_str.lower(): config_str for config_str in list_configs()}
            if config not in config_dict.keys():
                raise ValueError(f"Should be one of {config_dict.keys()}")

            config = getattr(configs, config_dict[config])()

        # for when config is a NETTConfig
        elif isinstance(config, NETTConfig):
            pass

        else:
            raise ValueError(f"Should either be one of {list(config_dict.keys())} or a subclass of NETTConfig")

        return config

    def _set_executable_permission(self) -> None:
        """
        Sets the executable permission for the Unity executable file.

        :return: None
        """
        subprocess.run(["chmod", "-R", "755", self.executable_path], check=True)
        self.logger.info("Executable permission is set")

    def _set_display(self) -> None:
        """
        Sets the display environment variable for the Unity environment.

        :return: None
        """
        os.environ["DISPLAY"] = str(f":{self.display}")
        self.logger.info("Display is set")

    
    # copied from __init__() of chickai_env_wrapper.py (legacy)
    # TODO (v0.3) Critical refactor, don"t like how this works, extremely error prone.
    # how can we build + constraint arguments better? something like an ArgumentParser sounds neat
    # TODO (v0.3) fix random_pos logic inside of Unity code
    def initialize(self, mode: str, **kwargs) -> Environment:
        """
        Initializes the environment with the given mode and arguments.

        :param mode: The mode to initialize the environment in.
        :type mode: str
        :param kwargs: The arguments to pass to the environment.
        :type kwargs: Any
        :return: The initialized environment.
        :rtype: Environment
        """
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
        # TODO: Discuss this with Manju, may be a MAJOR bug
        self.step_per_episode = kwargs.get("episode_steps", 200)
        if kwargs.get("episode_steps", False):
            args.extend(["--episode-steps", str(kwargs["episode_steps"])])

        # find unused port
        while port_in_use(self.base_port):
            self.base_port += 1

        # create logger
        self.log = Logger(f"{kwargs['condition'].replace('-', '_')}{kwargs['run_id']}-{mode}",
                          log_dir=f"{kwargs['log_path']}/")

        # create environment and connect it to logger
        self.env = UnityEnvironment(self.executable_path, side_channels=[self.log], additional_args=args, base_port=self.base_port)
        self.env = UnityToGymWrapper(self.env, uint8_visual=True)

        # initialize the parent class (gym.Wrapper)
        super().__init__(self.env)

    # converts the (c, w, h) frame returned by mlagents v1.0.0 and Unity 2022.3 to (w, h, c)
    # as expected by gym==0.21.0
    def render(self, mode="rgb_array"): # pylint: disable=unused-argument
        """
        Renders the current frame of the environment.

        :param mode: The mode to render the frame in. Defaults to "rgb_array".
        :type mode: str, optional
        :return: The rendered frame of the environment.
        :rtype: np.ndarray
        """
        return np.moveaxis(self.env.render(), [0, 1, 2], [2, 0, 1])

    def step(self, action):
        """
        Takes a step in the environment with the given action.

        :param action: The action to take in the environment.
        :type action: Any

        :return: A tuple containing the next state, reward, done flag, and info dictionary.
        :rtype: tuple[np.ndarray, float, bool, dict]
        """
        next_state, reward, done, info = self.env.step(action)
        return next_state, float(reward), done, info

    def log(self, msg: str) -> None:
        """
        Logs a message to the environment.

        :param msg: The message to log.
        :type msg: str
        """
        self.log.log_str(msg)

    def reset(self, seed: Optional[int] = None, **kwargs): # pylint: disable=unused-argument
        # nothing to do if the wrapped env does not accept `seed`
        """
        Resets the environment with the given seed and arguments.

        :param seed: The seed to use for the environment. Defaults to None.
        :type seed: int, optional
        :param kwargs: The arguments to pass to the environment.
        :type kwargs: Any

        :return: The initial state of the environment.
        :rtype: np.ndarray
        """
        self.log.info("Reset environment")
        return self.env.reset(**kwargs)

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"

    def __str__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"
