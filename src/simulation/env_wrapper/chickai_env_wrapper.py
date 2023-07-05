from abc import ABC, abstractmethod
from typing import Optional

import  gym

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from src.simulation.common.logger import Logger
from src.simulation.utils import port_in_use
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
        if "rewarded" in kwargs: args.extend(["--rewarded", "true"])
        if "episode_steps" in kwargs: args.extend(["--episode-steps",str(kwargs['episode_steps'])])

        if "mode" in kwargs: 
            args.extend(["--mode", kwargs["mode"]])
            self.mode = kwargs["mode"]
        else: self.mode = "rest"

        #Find unused port 
        while port_in_use(base_port):
            base_port += 1

        #Create logger
        self.log = Logger(run_id, log_dir=kwargs["log_path"])
        
        #Create environment and connect it to logger
        env = UnityEnvironment(env_path, side_channels=[self.log], additional_args=args, \
            base_port=base_port)
        self.env = UnityToGymWrapper(env, uint8_visual=True)
        print(gym.__version__)
        
        
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
    