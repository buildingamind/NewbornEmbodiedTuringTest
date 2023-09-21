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

from GPUtil import getFirstAvailable

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sb3_contrib import RecurrentPPO


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
        self.num_test_conditions = kwargs["num_conditions"]
        
        
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
        
        ## set cuda device if available
        print(torch.cuda.device_count())
        self.device_num = getFirstAvailable(attempts=5, interval=5, maxMemory=0.5, verbose=True)
        print(self.device_num)
        torch.cuda.set_device(self.device_num[0])
        assert torch.cuda.current_device() == self.device_num[0]
        
        
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
            
            for i in range(steps):
                obs = env.reset()
                done, lstm_states = False, None
                while not done:
                    action, lstm_states = self.model.predict(obs, state=lstm_states, deterministic=True)
                    obs, reward, done, _info = env.step(action)
                    env.render(mode="rgb_array")
                    vr.capture_frame()
                    
            vr.close()
            vr.enabled = False       
                        
                    
                
            
        
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
        if self.policy.lower() == "ppo":
            self.model = PPO.load(self.model_save_path, print_system_info=True)
        else:
            print("Loading recurrent agent:" + self.model_save_path)
            self.model = RecurrentPPO.load(self.model_save_path, print_system_info=True)
        
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
        
    def save_encoder_policy_network(self):
        self.load()
        if self.model == None:
            print("Usage Error: model is not specified either train a\
                new model or load a trained model")
            return
        
        
        base_path, model_name = os.path.split(self.model_save_path)
        
        ## save policy
        policy = self.model.policy
        policy.save(os.path.join(base_path, "policy.pkl"))
        
        
        ## save encoder
        encoder = self.model.policy.features_extractor.state_dict()
        save_path = os.path.join(base_path, "feature_extractor.pth")
        torch.save(encoder,save_path)
        
        print(f"Saved feature_extrator:{save_path}")
        return
    
    