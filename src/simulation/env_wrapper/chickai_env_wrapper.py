from abc import ABC, abstractmethod
from pprint import pprint
from typing import Optional

import gym
import numpy as np

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

from common.logger import Logger
from env_wrapper.dvs_wrapper import DVSWrapper
from utils import port_in_use
import pdb


class ChickAIEnvWrapper(gym.Wrapper):
    def __init__(self, run_id: str, env_path=None, base_port=5004, **kwargs):
        
        #Parse arguments and determine which version of the environment to use.
        args = []
        if "rec_path" in kwargs: args.extend(["--log-dir", kwargs["rec_path"]])
        if "recording_frames" in kwargs: args.extend(["--recording-steps", str(kwargs["recording_frames"])])
        if "record_chamber" in kwargs and kwargs["record_chamber"]: args.extend(["--record-chamber", "true"])
        if "record_agent" in kwargs and kwargs["record_agent"]: args.extend(["--record-agent", "true"])
        if "random_pos" in kwargs: args.extend(["--random-pos", "true"])
        if "rewarded" in kwargs: args.extend(["--rewarded", "true" if kwargs["rewarded"] else "false"])
        if "episode_steps" in kwargs: args.extend(["--episode-steps",str(kwargs['episode_steps'])])

        
        if "mode" in kwargs: 
            args.extend(["--mode", kwargs["mode"]])
            self.mode = kwargs["mode"]
        else: self.mode = "rest"

        
        #Find unused port 
        while port_in_use(base_port):
            base_port += 1

        #Create logger
        log_title = kwargs["log_title"] if "log_title" in kwargs else run_id
        self.log = Logger(log_title, log_dir=kwargs["log_path"])
        
        #Create environment and connect it to logger
        unity_env = UnityEnvironment(env_path, side_channels=[self.log], additional_args=args, \
            base_port=base_port)
        self.env = UnityToGymWrapper(unity_env, uint8_visual=True)
        
        if "dvs_wrapper" in kwargs and kwargs["dvs_wrapper"]:
            self.env = DVSWrapper(self.env)
        
        super().__init__(self.env)
        
    #Step the environment for one timestep
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state, float(reward), done, info
    
    #Write to the log file
    def log(self, msg: str) -> None:
        self.log.log_str(msg)
    
    #Close environment
    def close(self):
        self.env.close()
        del self.log

    def reset(self, seed: Optional[int] = None, **kwargs):
        # nothing to do if the wrapped env does not accept `seed`
        return self.env.reset(**kwargs)
    
    #This function is needed since episode lengths and the number of stimuli are determined in unity
    @abstractmethod
    def steps_from_eps(self, eps):
        pass

    # converts the (c, w, h) frame returned by mlagents v1.0.0 and Unity 2022.3 to (w, h, c) expected by gym==0.21.0
    def render(self, mode="rgb_array"):
        return np.swapaxes(self.env.render(), 2, 0)
    