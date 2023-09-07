
#!/usr/bin/env python3

import logging
import pdb
import os
from sb3_contrib import RecurrentPPO
import torch

from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecFrameStack

import matplotlib.pyplot as plt
import pandas as pd


from callback.supervised_save_bestmodel_callback import SupervisedSaveBestModelCallback
from networks.resnet10 import CustomResnet10CNN
from networks.resnet10 import CustomResnet10CNN
from networks.resnet18 import CustomResnet18CNN
from utils import to_dict, write_to_file

from callback.hyperparam_callback import HParamCallback
from common.base_agent import BaseAgent
from networks.lstm import CustomCNNLSTM

class SupervisedAgent(BaseAgent):
    def __init__(self, agent_id="Default Agent", \
        log_path="./Brains",
        **kwargs):
        super().__init__(agent_id, log_path, **kwargs)
        
        self.callback = SupervisedSaveBestModelCallback(summary_freq=self.summary_freq,\
            log_dir=self.path, \
            env_log_path = self.env_log_path, agent_id = self.id)
        
        self.hparamcallback = HParamCallback()
        self.checkpoint_callback = CheckpointCallback(save_freq=self.summary_freq,
                                                      save_path=os.path.join(self.path, "checkpoints"),
                                                      name_prefix="supervised_model",
                                                      save_replay_buffer=True,
                                                      save_vecnormalize=True)
        
        self.callback_list = CallbackList([self.callback, self.hparamcallback, self.checkpoint_callback])
        
        
        

        
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
        
        ## if dvs_wrapper is enabled
        #if self.frame_stack:
        #    envs = VecFrameStack(envs, n_stack=self.n_stack)
           
        
        ## setup tensorboard logger
        new_logger = configure(self.path, ["stdout", "csv", "tensorboard"])
        
        ## Add small, medium and large network
        if self.policy.lower() == "ppo":
            policy_model = "CnnPolicy"
            policy_kwargs = dict(features_extractor_kwargs=dict(features_dim=128))
            
            if self.encoder_type == "small":
                policy_kwargs = {}
            elif self.encoder_type == "medium":
                policy_kwargs["features_extractor_class"] = CustomResnet10CNN
            elif self.encoder_type == "large":
                policy_kwargs["features_extractor_class"] = CustomResnet18CNN
                
            elif self.encoder_type == "lstm":
                policy_kwargs["features_extractor_class"] = CustomCNNLSTM
                
            else:
                raise Exception(f"unknown network size: {self.encoder_type}")

            
            self.model = PPO(policy_model, envs, 
                             batch_size = self.batch_size, ## minibatch size for one gradient update - https://github.com/gzrjzcx/ML-agents/blob/master/docs/Training-PPO.md
                             n_steps = self.buffer_size, # rollout buffer size
                             tensorboard_log=self.path,
                             verbose = 0,
                    policy_kwargs=policy_kwargs, device=self.device)
            

        else:
            policy = "CnnLstmPolicy"
            policy_kwargs = dict(features_extractor_kwargs=dict(features_dim=128))
            
            if self.encoder_type == "small":
                policy_kwargs = {}
            elif self.encoder_type == "medium":
                policy_kwargs["features_extractor_class"] = CustomResnet10CNN
            elif self.encoder_type == "large":
                policy_kwargs["features_extractor_class"] = CustomResnet18CNN
                
            elif self.encoder_type == "lstm":
                policy_kwargs["features_extractor_class"] = CustomCNNLSTM
                
            else:
                raise Exception(f"unknown network size: {self.encoder_type}")

            self.model = RecurrentPPO(policy,\
                envs,
                batch_size=self.batch_size,
                n_steps = self.buffer_size,
                tensorboard_log=self.path,
                device=self.device, 
                verbose=0,
                policy_kwargs = policy_kwargs)
            print(self.model)
        
        self.model.set_logger(new_logger)
        print(f"Total training steps:{steps}")
        
        ## write model properties to the file
        d = to_dict(self.model.__dict__)
        x = {"encoder_type":self.encoder_type, "batch_size":self.batch_size,
                  "buffer_size":self.buffer_size,
                  "policy":self.policy,
                  "total_timesteps":steps,
                  "tensorboard_path":self.path,
                  "frame_stack": "true" if self.frame_stack else "false",
                  "logpath":self.path, "env_log_path":self.env_log_path, "agent_id":self.id}
        d.update(x)  
        write_to_file(os.path.join(self.path, "model_agent_dump.json"), d)
        
        
        self.model.learn(total_timesteps=steps, tb_log_name=f"{self.id}",\
                         progress_bar=True,\
                         callback=[self.callback_list])
        
        self.save()
        del self.model
        self.model = None
        
        ## plot reward graph
        self.plot_results(steps, plot_name=f"reward_graph_{self.id}")
