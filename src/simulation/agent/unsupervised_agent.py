import logging
import pdb
import os
from src.simulation.common.base_agent import BaseAgent


import torch
from gym.wrappers import RecordEpisodeStatistics

from gym.wrappers.monitoring.video_recorder import VideoRecorder
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import VecMonitor

from collections import deque
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from src.simulation.algorithms.icm import ICM


#Agent class as specified in the config file. Models are stored as files rather than
#being kept in memory for performance reasons.
class ICMAgent(BaseAgent):
    def __init__(self, agent_id="Default Agent", \
        reward="icm", log_path="./Brains", **kwargs):
        
        self.reward = reward
        self.id = agent_id
        self.model = None
        summary_freq = 30000
        self.rec_path = kwargs['rec_path'] if 'rec_path' in kwargs else ""
        self.encoder_type = kwargs['encoder']
        
        #If path does not exist, create it as a directory
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        self.log_dir = log_path
        
        #If path is a saved model assign to path
        if os.path.isfile(log_path):
            self.path = log_path
        else:
            #If path is a directory create a file in the directory name after the agent
            self.path = os.path.join(log_path, self.id)
        
        self.plots_path = os.path.join(self.path , "plots")
        os.makedirs(self.plots_path, exist_ok = True)
        self.env_log_path = os.path.join(kwargs['env_log_path'])
        self.model_save_path = os.path.join(self.path, "icm_agent")
        
        ## record video for rest
        self.video_record_path = os.path.join(self.rec_path,"test")
        os.makedirs(self.video_record_path, exist_ok=True)
        
        ## set cuda device if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## icm stats
        self.global_episode = 0
        self.global_step = 0
        
        
        
    #Train an agent. Still need to allow exploration wrappers and non PPO rl algos.
    def train(self, env, eps):
        steps = env.steps_from_eps(eps)
        env = Monitor(env, self.path)
        
        e_gen = lambda : env
        train_env = make_vec_env(env_id=e_gen, n_envs=1)
        train_env = RecordEpisodeStatistics(train_env)
        train_env = VecTransposeImage(train_env)
        
        ## setup tensorboard logger
        new_logger = configure(self.path, ["csv", "tensorboard"])
        icm = ICM(train_env.observation_space, train_env.action_space,self.device)
        
        
        policy = "CnnPolicy"
        self.model = PPO(policy, train_env, tensorboard_log=self.path,n_steps=200, device=self.device)
        self.model.set_logger(new_logger)
        print(f"Total training steps:{steps}")
        
        # Set info buffer
        all_eps_rewards = list()
        eps_rewards = deque([0.] * 10, maxlen=10)
        mean_eps_rewards = list()
        
        _ = train_env.reset()
        # Number of updates
        num_train_steps = steps
        self.num_envs = 1
        self.num_steps = 200
        
        num_updates = num_train_steps // self.num_envs // self.num_steps
        
        self.model.ep_info_buffer = deque(maxlen=10)
        _, callback = self.model._setup_learn(total_timesteps=steps)


        for update in range(num_updates):
            self.model.collect_rollouts(
                env=train_env,
                rollout_buffer=self.model.rollout_buffer,
                n_rollout_steps=200,
                callback=callback
            )
            
            intrinsic_rewards = icm.compute_irs(samples={
                        "obs": self.model.rollout_buffer.observations,
                        "actions": self.model.rollout_buffer.actions,
                        "next_obs": self.model.rollout_buffer.observations[1:]
                    },
                step = self.global_episode * self.num_envs * self.num_steps)
            
            
            self.model.rollout_buffer.rewards += intrinsic_rewards[:,:].numpy()
            
            # Update policy using the currently gathered rollout buffer.
            self.model.train()
            self.global_episode+=1
            self.global_step +=self.num_envs * self.num_steps
            
            
            eps_rewards.extend([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            all_eps_rewards.append(list(eps_rewards.copy()))
            mean_eps_rewards.append(np.mean(eps_rewards))
            
            print('ENV:steps/total timesteps {}/{}, \n \
                MEAN|MEDIAN REWARDS {:.2f}|{:.2f}, MIN|MAX REWARDS {:.2f}|{:.2f}\n'.format(
                steps, self.global_step,
                np.mean(eps_rewards), 
                np.median(eps_rewards),
                np.min(eps_rewards), 
                np.max(eps_rewards)
            ))
         
        #np.save(os.path.join(self.plots_path, 'episode_rewards.npy'), eps_rewards)
        #np.save(os.path.join(self.plots_path, 'mean_episode_rewards.npy'), mean_eps_rewards)
        
        self.save()
        del self.model
        self.model = None
        
        ## plot reward graph
        self.plot_results(steps, \
            plot_name=f"reward_graph_{self.id}")