"""
This module contains the definition of the Environment class, which is a wrapper around the UnityEnvironment
class from the mlagents_envs library. It provides a convenient interface for interacting with the Unity environment
and includes methods for initializing the environment, rendering frames, taking steps, resetting the environment,
and logging messages.

The Environment class inherits from the gym.Wrapper class, allowing it to be used as a gym environment.
"""

from __future__ import annotations

import os
import subprocess
from typing import Optional

import numpy as np
from gym import Wrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from nett.environment.configs import NETTConfig, list_configs
from nett.environment import configs
from nett.utils.environment import Logger, port_in_use

class Environment(Wrapper):
    def __init__(self,
                 config: str | NETTConfig,
                 executable_path: str,
                 display: int = 0,
                 base_port: int = 5004,
                 record_chamber: bool = False,
                 record_agent: bool = False,
                 recording_frames: int = 1000) -> None:
        """
        Initializes the Environment object.

        Args:
            config (str | NETTConfig): The configuration for the environment. It can be either a string representing
                the name of a pre-defined configuration, or an instance of the NETTConfig class.
            executable_path (str): The path to the Unity executable file.
            display (int, optional): The display number to use for the Unity environment. Defaults to 0.
            base_port (int, optional): The base port number to use for communication with the Unity environment.
                Defaults to 5004.
            record_chamber (bool, optional): Whether to record the chamber. Defaults to False.
            record_agent (bool, optional): Whether to record the agent. Defaults to False.
            recording_frames (int, optional): The number of frames to record. Defaults to 1000.
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

        Args:
            config (str | NETTConfig): The configuration to validate.

        Returns:
            NETTConfig: The validated configuration.

        Raises:
            ValueError: If the configuration is not a valid string or an instance of NETTConfig.
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
        """
        subprocess.run(["chmod", "-R", "755", self.executable_path], check=True)
        self.logger.info("Executable permission is set")

    def _set_display(self) -> None:
        """
        Sets the display environment variable for the Unity environment.
        """
        os.environ["DISPLAY"] = str(f":{self.display}")
        self.logger.info("Display is set")

    
    # copied from __init__() of chickai_env_wrapper.py (legacy)
    # TODO (v0.3) Critical refactor, don"t like how this works, extremely error prone.
    # how can we build + constraint arguments better? something like an ArgumentParser sounds neat
    # TODO (v0.3) fix random_pos logic inside of Unity code
    def initialize(self, mode: str, **kwargs) -> Environment:
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
    def render(self, mode="rgb_array"):
        return np.moveaxis(self.env.render(), [0, 1, 2], [2, 0, 1])

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, float(reward), done, info

    def log(self, msg: str) -> None:
        self.log.log_str(msg)

    def reset(self, seed: Optional[int] = None, **kwargs):
        # nothing to do if the wrapped env does not accept `seed`
        return self.env.reset(**kwargs)

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"

    def __str__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"
