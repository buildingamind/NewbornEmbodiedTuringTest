from __future__ import annotations

import os
import numpy as np
import subprocess

from typing import Any, Optional
from netts.environment.configs import NETTConfig, list_configs
from netts.environment import configs
from netts.utils.environment import Logger, port_in_use
from gym import Wrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

class Environment(Wrapper):
    def __init__(self, 
                 config: str | NETTConfig, 
                 executable_path: str, 
                 display: int = 0,
                 base_port: int = 5004,
                 record_chamber: bool = False, 
                 record_agent: bool = False, 
                 recording_frames: int = 1000) -> None:
        from netts import logger
        self.logger = logger.getChild(__class__.__name__)
        self.config = self._validate_config(config)
        # TO DO (v0.4) what might be a way to check if it is a valid executable path? 
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
        # for when config is a str
        if isinstance(config, str):
            config_dict = {config_str.lower(): config_str for config_str in list_configs()}
            if config not in config_dict.keys():
                raise ValueError(f"Should be one of {config_dict.keys()}") 
            else:
                config = getattr(configs, config_dict[config])()

        # for when config is a NETTConfig
        elif isinstance(config, NETTConfig):
            pass

        else:
            raise ValueError(f"Should either be one of {list(config_dict.keys())} or a subclass of NETTConfig")
        
        return config

    def _set_executable_permission(self) -> None:
        subprocess.run(['chmod', '-R', '755', self.executable_path])
        self.logger.info("Executable permission is set")

    def _set_display(self) -> None:
        os.environ["DISPLAY"] = str(f":{self.display}")
        self.logger.info("Display is set")


    # copied from __init__() of chickai_env_wrapper.py (legacy)
    # TO DO (v0.3) Critical refactor, don't like how this works, extremely error prone.  
    # how can we build + constraint arguments better? something like an ArgumentParser sounds neat
    # TO DO (v0.3) fix random_pos logic inside of Unity code
    def initialize(self, mode: str, **kwargs) -> Environment:
        # from environment arguments
        args = []
        if self.recording_frames:
            args.extend(["--recording-steps", str(self.recording_frames)])
        if self.record_chamber:
            args.extend(["--record-chamber", "true"])
        if self.record_agent:
            args.extend(["--record-agent", "true"])

        # from runtime
        if kwargs.get('mode', None):
            args.extend(["--mode", f"{mode}-{kwargs['mode']}"])
        if kwargs.get('rec_path', None):
            args.extend(["--log-dir", f"{kwargs['rec_path']}/"])
        # needs to fixed in Unity code where the default is always false
        # if kwargs['mode'] != 'test':
        #     args.extend(["--random-pos", "true"])
        if kwargs.get("rewarded", False): 
            args.extend(["--rewarded", "true"])
        # TO DO: Discuss this with Manju, may be a MAJOR bug
        self.step_per_episode = kwargs.get("episode_steps", 200)
        if kwargs.get("episode_steps", False):
            args.extend(["--episode-steps", str(kwargs["episode_steps"])])
        
        # find unused port 
        while port_in_use(self.base_port):
            self.base_port += 1

        # create logger
        self.log = Logger(f"{kwargs['mode']}-{kwargs['run_id']}-{mode}", log_dir=f"{kwargs['log_path']}/")
        
        # create environment and connect it to logger
        self.env = UnityEnvironment(self.executable_path, side_channels=[self.log], additional_args=args, base_port=self.base_port)
        self.env = UnityToGymWrapper(self.env, uint8_visual=True)
        
        # body?
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
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return "{}({!r})".format(self.__class__.__name__, attrs)
    
    def __str__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return "{}({!r})".format(self.__class__.__name__, attrs)