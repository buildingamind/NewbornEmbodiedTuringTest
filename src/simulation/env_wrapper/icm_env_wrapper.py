import gym
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper

from env_wrapper.chickai_env_wrapper import ChickAIEnvWrapper

from common.logger import Logger
from utils import port_in_use


#Gym wrapper for the viewpoint environment. New gym wrapper (specifically for argument parsing) needs to be made for
#a different experiment setup.
class ICMEnv(ChickAIEnvWrapper):
    def __init__(self, run_id: str, env_path=None, base_port=5004, **kwargs):
        super().__init__(run_id, env_path, base_port, **kwargs)
    
    #This function is needed since episode lengths and the number of stimuli are determined in unity
    def steps_from_eps(self, eps):
        step_per_episode = 200
        return step_per_episode * eps
        