import logging
import pdb
import os
from Simulation.networks.resnet18 import CustomResnet18CNN #Used for model saving and loading

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

import matplotlib.pyplot as plt
import pandas as pd
from Simulation.callback.supervised_save_bestmodel_callback import SupervisedSaveBestModelCallback
from Simulation.networks.resnet10 import CustomResnet10CNN
from Simulation.utils import compute_train_performance, get_train_performance_plot_data

#Agent class as specified in the config file. Models are stored as files rather than
#being kept in memory for performance reasons.
class Agent:
    def __init__(self, agent_id="Default Agent", \
        reward="supervised", log_path="./Brains", **kwargs):
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
        self.save_bestmodel_callback = SupervisedSaveBestModelCallback(summary_freq=summary_freq,\
            log_dir=self.path, \
            env_log_path = self.env_log_path, agent_id = self.id)
        
        self.model_save_path = os.path.join(self.path, "supervised_agent")
        
        ## record video for rest
        self.video_record_path = os.path.join(self.rec_path,"test")
        os.makedirs(self.video_record_path, exist_ok=True)
        
        ## set cuda device if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        

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
        #envs = VecMonitor(envs, self.path)
        
        ## setup tensorboard logger
        new_logger = configure(self.path, ["csv", "tensorboard"])
        
        
        if self.reward == "supervised":
            ## Add small, medium and large network
            policy = "CnnPolicy"
            policy_kwargs = dict(features_extractor_kwargs=dict(features_dim=128))
            print(self.encoder_type)
            if self.encoder_type == "small":
                self.model = PPO(policy, envs, tensorboard_log=self.path, device=self.device)
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
        else:
            print("Please use the supervised reward until I implement rlexplore correctly.")
            return
        
        
        self.model.set_logger(new_logger)
        print(f"Total training steps:{steps}")
        
        self.model.learn(total_timesteps=steps,\
                         progress_bar=True,\
                         callback=[self.save_bestmodel_callback])
        
        self.save()
        del self.model
        self.model = None
        
        ## plot reward graph
        self.plot_results(steps, \
            plot_name=f"reward_graph_{self.id}")
        
        ## plot train performance graph
        self.plot_train_performance()
        
    
    def train_intrinsic(self, env, eps):
        e_gen = lambda : env
        envs = make_vec_env(env_id=e_gen, n_envs=1)
        re3 = RE3(obs_shape=envs.observation_space.shape, 
                action_shape=envs.action_space.shape, 
                device=device, latent_dim=128, beta=1e-2, kappa=1e-5)
        #Need to figure out how to make this generic and use it.

    #Test the agent in the given environment for the set number of steps
    def test(self, env, eps, record_prefix = "rest"):
        self.load()
        if self.model == None:
            print("Usage Error: model is not specified either train a new model or load a trained model")
            return
        
        #Run the testing
        steps = env.steps_from_eps(eps)
        
        ## record - rest video
        vr = VideoRecorder(env=env,
        path="{}/{}_{}.mp4".format(self.video_record_path, str(self.id), record_prefix),
        enabled=True)

        
        obs = env.reset()
        for i in range(steps):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            if done:
                env.reset()
            vr.capture_frame()    
            
        
        vr.close()
        vr.enabled = False
               
        del self.model
        self.model = None
        

    #Saves brains to the specified path
    def save(self, path=None):
        if path is None:
            path = self.model_save_path
        
        if self.model == None:
            self.load(path)
        else:
            self.model.save(path)

    #Load brains from the file
    def load(self, path=None):
        print("load called")
        if path == None:
            path = self.model_save_path
        self.model = PPO.load(self.model_save_path, print_system_info=True)
        
    def check_env(self, env):
        env_check = check_env(env, warn=True)
        if env_check != None:
            logging.error(env_check)
            raise Exception(f"Failed env check")
        
        return True

    ## plot results - train graph and reward graph
    def plot_results(self, steps, plot_name="chickai-train"):
        results_plotter.plot_results([self.path], steps, 
                                     results_plotter.X_TIMESTEPS, plot_name)
        
        
        plt.savefig(self.plots_path + "/" + plot_name + ".png")
        plt.clf()
        
    def plot_train_performance(self):
        val = get_train_performance_plot_data(self.env_log_path)    
        plt.ylim([0,1])
        plt.plot(val,alpha=0.3)
        plt.savefig(self.plots_path + "/train_performance_plt.png")
        plt.clf()
