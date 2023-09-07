from abc import ABC, abstractmethod
import logging
import pdb
import os
from typing import Optional

import torch
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import VecFrameStack

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class BaseAgent(ABC):
    def __init__(self, agent_id="Default Agent", \
        log_path="./Brains",
        **kwargs):
        
        self.id = agent_id
        self.model = None
        self.summary_freq = 30000 
        self.rec_path = kwargs['rec_path'] if 'rec_path' in kwargs else ""
        self.encoder_type = kwargs['encoder']
        self.batch_size = kwargs['mini_batchsize']
        self.buffer_size = kwargs['buffer_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_stack = kwargs["frame_stack"]
        self.n_stack = kwargs["n_stack"]
        self.policy = kwargs["policy"]
        
        
        #If path does not exist, create it as a directory
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_dir = log_path
        
        if os.path.isfile(log_path):
            self.path = log_path
        else:
            self.path = os.path.join(log_path, self.id)
        self.plots_path = os.path.join(self.path , "plots")
        os.makedirs(self.plots_path, exist_ok = True)
        self.model_save_path = os.path.join(self.path, "model")
        
        
        ## env logs - train and test logs
        self.env_log_path = os.path.join(kwargs['env_log_path'])
        
        ## recordings path - recordings
        self.video_record_path = os.path.join(self.rec_path,"Test")
        os.makedirs(self.video_record_path, exist_ok=True)
        
    @abstractmethod   
    def train(self, env, eps)->None:
        pass

    def test(self, env, eps, record_prefix = "rest"):
        """
        Test the agent in the given environment for the set number of steps

        Args:
            env : gym environment wrapper
            eps : number of test episodes
            record_prefix (str, optional): recording file name prefix
        """
        self.load()
        if self.model == None:
            print("Usage Error: model is not specified either train a new model or load a trained model")
            return
        
        #Run the testing
        steps = env.steps_from_eps(eps)
        
        e_gen = lambda : env
        envs = make_vec_env(env_id=e_gen, n_envs=1)
        
        ## no need for another framestack
        #if self.frame_stack:
        #    env = VecFrameStack(env, n_stack=self.n_stack)
        
        
        
        ## record - test video
        vr = VideoRecorder(env=envs,
        path="{}/{}_{}.mp4".format(self.video_record_path, \
            str(self.id), record_prefix), enabled=True)
        
        
        if self.policy.lower()=="ppo":
            obs = envs.reset()
            for i in range(steps):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = envs.step(action)
                
                if done:
                    env.reset()
                
                env.render(mode="rgb_array")
                vr.capture_frame()    

            vr.close()
            vr.enabled = False
        else:
            obs = envs.reset()
            for c in range(28):
                for i in range(eps):
                    ## cell and hidden states for lSTM
                    lstm_states = None
                    num_envs = 1
                    max_episode_steps = 1000
                    # Episode start signals are used to reset the lstm states
                    episode_starts = np.ones((num_envs,), dtype=bool)
                    done = False
                    episode_rewards, episode_steps = 0, 0
                    while not done and episode_steps < max_episode_steps:
                        action, lstm_states = self.model.predict(obs, state=lstm_states,
                                                            episode_start=episode_starts, 
                                                            deterministic=True)
                        
                        next_obs, rewards, done, info = env.step(action)
                        env.render(mode="rgb_array")
                        vr.capture_frame()
                        obs = next_obs
                        episode_rewards += rewards
                        episode_steps += 1
                    
                    if done:
                        env.reset()
                        
                    #print('Reward = %.3f | Step = %d | episode_cnt = %d' % (episode_rewards, episode_steps,i))
                
                
            
               
        del self.model
        self.model = None
        

    def save(self, path: Optional[str] = None) -> None:
        """
        Save agent prains to the specified path

        Args:
            path (str): Path value to save the model
        """
        if path is None:
            path = self.model_save_path
        
        if self.model == None:
            self.load(path)
        else:
            self.model.save(path)

    
    def load(self, path=None)->None:
        """
        Load the model from the specified path

        Args:
            path (str): model saved path. Defaults to None.
        """
        print("load called")
        if path == None:
            path = self.model_save_path
        
        print(self.model_save_path)
        self.model = PPO.load(self.model_save_path, print_system_info=True)
        
    def check_env(self, env):
        """
        Check environment

        Args:
            env (vector environment): vector env check for correctness

        Raises:
            Exception: raise exception if env check fails

        Returns:
            bool: env check is successful or failed
        """
        env_check = check_env(env, warn=True)
        if env_check != None:
            logging.error(env_check)
            raise Exception(f"Failed env check")
        
        return True

    def plot_results(self, steps:int, plot_name="chickai-train") -> None:
        """
        Generate reward plot for training

        Args:
            steps (int): number of training steps
            plot_name (str, optional): Name of the reward plot. Defaults to "chickai-train".
        """
        results_plotter.plot_results([self.path], steps, 
                                     results_plotter.X_TIMESTEPS, plot_name)
        
        
        plt.savefig(self.plots_path + "/" + plot_name + ".png")
        plt.clf()
        
    
    