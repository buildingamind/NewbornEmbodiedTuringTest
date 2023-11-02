
#!/usr/bin/env python3

import logging
import pdb
import os
from sb3_contrib import RecurrentPPO
from src.simulation.networks.custom_network import CustomActorCriticPolicy
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
from GPUtil import getFirstAvailable

from networks.frozen_encoder import CustomFrozenNetwork


class SupervisedAgent(BaseAgent):
    def __init__(self, agent_id="Default Agent", \
        log_path="./Brains",
        **kwargs):
        super().__init__(agent_id, log_path, **kwargs)
        
        self.initialize_callbacks()
    
    def initialize_callbacks(self):
        self.callback = SupervisedSaveBestModelCallback(summary_freq=self.summary_freq, log_dir=self.path, env_log_path=self.env_log_path, agent_id=self.id)
        self.hparamcallback = HParamCallback()
        self.checkpoint_callback = CheckpointCallback(save_freq=self.summary_freq, save_path=os.path.join(self.path, "checkpoints"), name_prefix="supervised_model", save_replay_buffer=True, save_vecnormalize=True)
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
        # make a vector environment
        e_gen = lambda : env
        envs = make_vec_env(env_id=e_gen, n_envs=1)
        
        ## setup tensorboard logger
        new_logger = configure(self.path, ["stdout", "csv", "tensorboard"])

        feature_dims = self.get_feature_dimensions()
        policy_kwargs = self.get_policy_args(feature_dims)
        policy_kwargs['features_extractor_kwargs']['train'] = self.retrain_encoder
        policy_kwargs['features_extractor_kwargs']['encoder_type'] = self.encoder_type
        
        # policy network
        policy_model = "CnnPolicy" if self.policy.lower() == "ppo" else "CnnLstmPolicy"
        self.model = self.create_policy(envs, policy_model, policy_kwargs)
        print(self.model.policy.features_extractor)
        
        self.model.set_logger(new_logger)
        print(f"Total training steps:{steps}")
        
        # write model properties to the file
        self.write_model_properties(self.model, steps)
        
        # set the feature network to require_grad = False
        
        if self.use_frozen_encoder and not self.retrain_encoder:
            self.model = self.set_feature_extractor_require_grad(self.model)
        
        print(self.model.policy)
        requires_grad_str = ""
        for param in self.model.policy.features_extractor.parameters():
            requires_grad_str+=str(param.requires_grad)
        
        print(requires_grad_str)
        
        #pdb.set_trace()
        # start training
        self.model.learn(total_timesteps=steps, tb_log_name=f"{self.id}", progress_bar=True, callback=[self.callback_list])
        self.save()
        
        del self.model
        self.model = None
        
        # plot reward graph
        self.plot_results(steps, plot_name=f"reward_graph_{self.id}")
        
        # save encoder and policy network state dict - to perform model analysis
        self.save_encoder_policy_network()
        
        
    def get_feature_dimensions(self):
        feature_dims = 512
        if self.use_frozen_encoder:
            if self.encoder_type == "resnet50":
                feature_dims = 2048
            elif self.encoder_type == "retinalwaves":
                feature_dims = 512
        else:
            if self.encoder_type in ["medium","large"]:
                feature_dims = 128
        return feature_dims

    def get_policy_args(self, feature_dims):
        policy_kwargs = {}
        if not self.use_frozen_encoder and self.encoder_type.lower() == "small":
            policy_kwargs = {} 
        else:
            feature_class = self.get_feature_class()
            policy_kwargs = dict(features_extractor_kwargs=dict(features_dim= feature_dims), features_extractor_class=feature_class)
        return policy_kwargs
    
    def get_feature_class(self):
        feature_class = ''
        if self.encoder_type == "small":
            return CustomFrozenNetwork
        elif self.encoder_type == "simclr":
            return CustomFrozenNetwork
        elif self.encoder_type in ["resnet50","resnet18"]:
            return CustomFrozenNetwork
        elif self.encoder_type == "random":
            return CustomFrozenNetwork
        elif self.encoder_type == "retinalwaves":
            return CustomFrozenNetwork
        elif self.encoder_type == "untrained_r18_2b":
            return CustomFrozenNetwork
        elif self.encoder_type == "medium":
            feature_class = CustomResnet10CNN
        elif self.encoder_type == "large":
            feature_class = CustomResnet18CNN
        elif self.encoder_type == "vit":
            feature_class = CustomFrozenNetwork
        else:
            print("feature_class not found!")
        return feature_class
