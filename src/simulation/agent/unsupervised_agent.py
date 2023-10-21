import logging
import pdb
import os
from algorithms.rnd import RND
from callback.hyperparam_callback import HParamCallback
from common.base_agent import BaseAgent
from utils import to_dict, write_to_file


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
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback


from collections import deque
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from algorithms.icm import ICM

from tqdm import tqdm

class ICMAgent(BaseAgent):
    """ICM agent with callback to implement intrinsic curiosity reward
    
    Args:
        BaseAgent (_type_): _description_
    """
    def __init__(self, agent_id="Default Agent", \
        reward="icm", log_path="./Brains", **kwargs):
        super().__init__(agent_id, log_path, **kwargs)
        
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
        
        
        self.hparamcallback = HParamCallback()
        self.checkpoint_callback = CheckpointCallback(save_freq=summary_freq,save_path=os.path.join(self.path, "checkpoints"),
                                                      name_prefix="unsupervised_model",
                                                      save_replay_buffer=True,
                                                      save_vecnormalize=True)
        
        self.callback_list = CallbackList([self.hparamcallback, self.checkpoint_callback])
        
        
        
    def train(self, env, eps):
        steps = env.steps_from_eps(eps)
        print(steps)
        
        env = Monitor(env, self.path)
        
        e_gen = lambda : env
        train_env = make_vec_env(env_id=e_gen, n_envs=1)
        train_env = RecordEpisodeStatistics(train_env)
        train_env = VecTransposeImage(train_env)
        print(train_env.observation_space.shape)
        
        
        ## setup tensorboard logger
        new_logger = configure(self.path, ["csv", "tensorboard"])
        
        print(self.reward)
        explore_reward = self.initialize_reward_algo(train_env)
                
        print(self.buffer_size)
        policy = "CnnPolicy"
        self.model = PPO(policy, train_env, tensorboard_log=self.path,\
            n_steps=self.buffer_size, device=self.device)
        self.model.set_logger(new_logger)
        print(f"Total training steps:{steps}")
        
        # Set info buffer
        # reset the env
        episode_rewards = deque(maxlen=10)
        episode_steps = deque(maxlen=10)
        
        
        # Number of updates
        num_train_steps = steps
        self.num_envs = 1
        self.num_steps = self.buffer_size
        num_updates = num_train_steps // self.num_envs // self.num_steps

        
        ## write model properties to the file
        d = to_dict(self.model.__dict__)
        write_to_file(os.path.join(self.path, "model_dump.json"), d)
        
        
        _ = train_env.reset()
        
        
        self.model.ep_info_buffer = deque(maxlen=10)
        _, callback = self.model._setup_learn(total_timesteps=num_train_steps,callback=[self.callback_list])
        eps_rewards = deque([0.] * 10, maxlen=10)
        all_eps_rewards = list()
        
        for update in tqdm(range(num_updates)):
            self.model.collect_rollouts(
                env=train_env,
                rollout_buffer=self.model.rollout_buffer,
                n_rollout_steps=self.buffer_size,
                callback=callback
            )
            
            intrinsic_rewards = explore_reward.compute_irs(samples={
                        "obs": self.model.rollout_buffer.observations,
                        "actions": self.model.rollout_buffer.actions
                    },
                step = self.global_episode * self.num_envs * self.num_steps)
            
            #pdb.set_trace()
            self.model.rollout_buffer.rewards += intrinsic_rewards
            self.model.train()
            
            # Update policy using the currently gathered rollout buffer.
            self.global_episode+=1
            self.global_step +=self.num_envs * self.num_steps
            eps_rewards.extend([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            all_eps_rewards.append(list(eps_rewards.copy()))
            times_steps = update * self.num_steps 
            
            print('TOTAL TIME STEPS {} \n \
                MEAN|MEDIAN REWARDS {:.2f}|{:.2f}, MIN|MAX REWARDS {:.2f}|{:.2f}\n'.format(
                times_steps,
                np.mean(eps_rewards), np.median(eps_rewards), np.min(eps_rewards), np.max(eps_rewards)
            ))
            
        
        self.save()
        del self.model
        self.model = None
        
        np.save(os.path.join(self.log_dir, 'episode_rewards.npy'), all_eps_rewards)
        
        ## plot reward graph
        self.plot_results(steps, plot_name=f"reward_graph_{self.id}")
        

    def initialize_reward_algo(self, train_env):
        if self.reward.lower() == "icm":
            explore_reward = ICM(train_env.observation_space, train_env.action_space,\
                self.device, batch_size=self.batch_size)
        elif self.reward.lower() == "rnd":
            explore_reward = RND(train_env.observation_space, train_env.action_space,\
                self.device, batch_size=self.batch_size)
        else:
            explore_reward = ICM(train_env.observation_space, train_env.action_space,\
                self.device, batch_size=self.batch_size)
                
        print(explore_reward)
        return explore_reward