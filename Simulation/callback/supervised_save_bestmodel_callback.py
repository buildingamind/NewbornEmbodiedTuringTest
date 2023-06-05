#!/usr/bin/env pytho3
import os
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pdb
from stable_baselines3.common.results_plotter import load_results, ts2xy

from stable_baselines3.common.callbacks import BaseCallback

from Simulation.utils import compute_train_performance

class SupervisedSaveBestModelCallback(BaseCallback):
    def __init__(self, summary_freq:int, log_dir:str, env_log_path:str, agent_id:str) -> None:
        super().__init__(verbose= 1)
        self.summary_freq = summary_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(self.log_dir, "best_performance_model")
        self.env_log_path = env_log_path
        self.best_mean_performance = -np.inf
        self.best_mean_reward = -np.inf
        
    
    '''        
    def _on_step(self) -> None:
        if self.n_calls % self.summary_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # mean reward for last 100 episodes
                mean_reward  = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f" Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f}\
                        - Last mean reward per episode: {mean_reward:.2f}")
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose >0:
                        print(f"Saving the best model to:{self.save_path}.zip")
                    self.model.save(self.save_path)    
        return True
    '''
    
    def _on_step(self) -> None:
        if self.n_calls % self.summary_freq == 0:
            # Retrieve training reward
            x, y = compute_train_performance(self.env_log_path)
            if len(x) > 0:
                
                # mean performance for last 100 episodes
                mean_performance  = y[-1]
                
                x, y = ts2xy(load_results(self.log_dir), 'timesteps')
                if len(x) > 0:
                    # mean reward for last 100 episodes
                    mean_reward  = np.mean(y[-100:])
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f}\
                            - Last mean reward per episode: {mean_reward:.2f}")
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                
                
                if self.verbose > 0:
                    print(f"Best mean performance: {self.best_mean_performance:.2f}\
                        - Last mean performance per episode: {mean_performance:.2f}")
                if mean_performance > self.best_mean_performance:
                    self.best_mean_performance = mean_performance
                    if self.verbose >0:
                        print(f"Saving the best model to:{self.save_path}.zip")
                    self.model.save(self.save_path)    
        return True
    
    
    

            
        
        

            
        