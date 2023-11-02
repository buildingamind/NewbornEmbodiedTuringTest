from abc import ABC, abstractmethod
import logging
import pdb
import os
from typing import Optional
from src.simulation.env_wrapper.frozen_encoder_wrapper import FrozenEncoderWrapper

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

from src.simulation.utils import to_dict, write_to_file


class BaseAgent(ABC):
    def __init__(self, agent_id="Default Agent", \
        log_path="./Brains",
        **kwargs):
        
        self.id = agent_id
        self.model = None
        self.summary_freq = 30000 
        self.rec_path = kwargs.get('rec_path', "")

        
        ## encoder
        encoder = kwargs.get('encoder', {})
        self.encoder_type = encoder.get('name', '')
        self.use_frozen_encoder = encoder.get('frozen', False)
        self.retrain_encoder = encoder.get('train', False)
        
        
        self.batch_size = kwargs.get('mini_batchsize', 512)
        self.buffer_size = kwargs.get('buffer_size', 2048)

        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.frame_stack = kwargs.get("frame_stack", 1)
        self.n_stack = kwargs.get("n_stack", 1)
        self.policy = kwargs.get("policy", "PPO")
        self.num_test_conditions = kwargs.get("num_conditions", 29)

        ## log dir
        kwargs['log_path'] = log_path
        self.create_log_directory(log_path)
        self.set_paths(kwargs)
        
        ## device
        self.setup_cuda_device()
        
        
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
            obs = env.reset()
            max_steps_count = 1000
            for i in range(steps):
                # Cell and hidden state of the LSTM
                lstm_states = None
                num_envs = 1
                
                # Episode start signals are used to reset the lstm states
                episode_starts = np.ones((num_envs,), dtype=bool)
                num_steps = 0
                while num_steps < max_steps_count:
                    action, lstm_states = self.model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                    # Note: vectorized environment resets automatically
                    obs, rewards, dones, info = env.step(action)
                    episode_starts = dones
                    env.render(mode="rgb_array")
                    vr.capture_frame()  
                    num_steps+=1
                    if dones:
                        obs = env.reset()
                        break
                        
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
        
    def create_log_directory(self, log_path):
        os.makedirs(log_path, exist_ok=True)
        self.log_dir = log_path

    def set_paths(self,  kwargs):
        log_path = kwargs.get('log_path','')
        self.path = log_path if os.path.isfile(log_path) else os.path.join(log_path, self.id)
        self.plots_path = os.path.join(self.path, 'plots')
        os.makedirs(self.plots_path, exist_ok=True)
        
        self.model_save_path = os.path.join(self.path, 'model')
        self.env_log_path = kwargs.get('env_log_path', '')
        
        self.video_record_path = os.path.join(self.rec_path, 'Test')
        os.makedirs(self.video_record_path, exist_ok=True)

    def setup_cuda_device(self):
        print(torch.cuda.device_count())
        self.device_num = getFirstAvailable(attempts=5, interval=5, maxMemory=0.7, verbose=True)
        print(self.device_num)
        torch.cuda.set_device(self.device_num[0])
        assert torch.cuda.current_device() == self.device_num[0]
    
    
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
        
        print(f"loading the model from:{self.model_save_path}")
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
    
    def create_policy(self, envs, policy_model, policy_kwargs, **kwargs):
        if self.policy.lower() == "ppo":
            model = PPO(policy_model, envs, batch_size=self.batch_size, 
                            n_steps=self.buffer_size,
                            tensorboard_log=self.path,
                            verbose=0, policy_kwargs=policy_kwargs, device=self.device)
        else:
            model = RecurrentPPO(policy_model,\
                    envs,
                    batch_size=self.batch_size,
                    n_steps = self.buffer_size,
                    tensorboard_log=self.path,
                    device=self.device, 
                    verbose=0,
                    policy_kwargs = policy_kwargs)
                
        return model   

    def set_feature_extractor_require_grad(self, model, require_grad = False):
        model.policy.features_extractor.eval()
        for param in model.policy.features_extractor.parameters():
            param.require_grad = require_grad
            
        return model
    
    
    
    
    
    def write_model_properties(self, model, steps):
        model_props = {
            "encoder_type": self.encoder_type,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
            "policy": self.policy,
            "total_timesteps": steps,
            "tensorboard_path": self.path,
            "frame_stack": str(self.frame_stack).lower(),
            "logpath": self.path,
            "env_log_path": self.env_log_path,
            "agent_id": self.id
        }

        model_dict = to_dict(model.__dict__)
        model_dict.update(model_props)

        file_path = os.path.join(self.path, "model_agent_dump.json")
        write_to_file(file_path, model_dict)

    
    def wrap_frozen_encoder(self, env):
        if not self.use_frozen_encoder:
            return
        
        ## wrap the encoder
        env = FrozenEncoderWrapper(env,self.encoder_type)
        
        return env
    
    
    