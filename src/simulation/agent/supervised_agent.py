
#!/usr/bin/env python3

import logging
import pdb
import os
from src.simulation.common.base_agent import BaseAgent
import torch

from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env


import matplotlib.pyplot as plt
import pandas as pd


from src.simulation.callback.supervised_save_bestmodel_callback import SupervisedSaveBestModelCallback
from src.simulation.networks.resnet10 import CustomResnet10CNN
from src.simulation.networks.resnet10 import CustomResnet10CNN
from src.simulation.networks.resnet18 import CustomResnet18CNN


class SupervisedAgent(BaseAgent):
    def __init__(self, agent_id="Default Agent", \
        log_path="./Brains",
        **kwargs):
        super().__init__(agent_id, log_path, **kwargs)
        
        self.callback = SupervisedSaveBestModelCallback(summary_freq=self.summary_freq,\
            log_dir=self.path, \
            env_log_path = self.env_log_path, agent_id = self.id)

        
    #Train an agent. Still need to allow exploration wrappers and non PPO rl algos.
    def train(self, env, eps):
        
        steps = env.steps_from_eps(eps)
        env = Monitor(env, self.path)
        try:
            self.check_env(env)
        except Exception as ex:
            print("Failed training env check",str(ex))
            return
        
        e_gen = lambda : env
        envs = make_vec_env(env_id=e_gen, n_envs=1)
        
        ## setup tensorboard logger
        new_logger = configure(self.path, ["csv", "tensorboard"])
                
        
        ## Add small, medium and large network
        policy = "CnnPolicy"
        policy_kwargs = dict(features_extractor_kwargs=dict(features_dim=128))
        print(self.encoder_type)
        
        if self.encoder_type == "small":
            self.model = PPO(policy, envs, tensorboard_log=self.path,\
                device=self.device)
        elif self.encoder_type == "medium":
            policy_kwargs["features_extractor_class"] = CustomResnet10CNN
            self.model = PPO(policy, envs, tensorboard_log=self.path,\
                policy_kwargs=policy_kwargs, device=self.device)
        elif self.encoder_type == "large":
            policy_kwargs["features_extractor_class"] = CustomResnet18CNN
            self.model = PPO(policy, envs, tensorboard_log=self.path, \
                policy_kwargs=policy_kwargs, device=self.device)
        else:
            raise Exception(f"unknown network size: {self.encoder_type}")
    
    
        self.model.set_logger(new_logger)
        print(f"Total training steps:{steps}")
        
        self.model.learn(total_timesteps=steps,\
                         progress_bar=True,\
                         callback=[self.callback])
        
        self.save()
        del self.model
        self.model = None
        
        ## plot reward graph
        self.plot_results(steps, \
            plot_name=f"reward_graph_{self.id}")
