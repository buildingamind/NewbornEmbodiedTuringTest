#!/usr/bin/env python3

## load imports
from logging import Logger
import logging
import pytest
import sys
import random
import gym
from itertools import count
import traceback
import os

from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv,VecTransposeImage
import numpy as np
import json
from mlagents_envs.exception import (UnityEnvironmentException,
                                     UnityWorkerInUseException)
from omegaconf import DictConfig, OmegaConf, open_dict

from src.simulation.env_wrapper.parsing_env_wrapper import ParsingEnv
from src.simulation.env_wrapper.viewpoint_env_wrapper import ViewpointEnv

from hydra import initialize, compose
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import warnings; 
warnings.filterwarnings("ignore")

## test methods

def get_key(value):
    if value == 3:
        return 'w'
    if value == 6:
        return 's'
    if value == 2:
        return 'a'
    if value == 1:
        return 'd'

    return ""

def get_random_actions():
    # w- forward, s - backward, a - rotate left, d - rotate right
    keyboard_inputs = {'w':1,'s':2,'a':3,'d':4, 'r':5,'f':6}
    n = 500
    values = list(keyboard_inputs.keys())
    random_actions = random.choices(values, k=n)
    return random_actions


@pytest.mark.parametrize(
    "env_cls", [ParsingEnv,ViewpointEnv]
)
def test_cont_env_parsing(env_cls):
    with initialize(version_base=None, config_path="../src/simulation/conf"):
        # config is relative to a module
        config_name = "config_parsing" if env_cls in [ParsingEnv] else "config"
        cfg = compose(config_name=config_name)
        
    random_actions = get_random_actions()
    
    ## generate env
    if env_cls in [ParsingEnv]:
        env_config = cfg["Environment"]
        with open_dict(env_config):
                object =  "ship" if env_config["use_ship"] else "fork"
                env_config["mode"] = "rest" + "-"+ object +"-"+env_config["background"]
                env_config["random_pos"] = True
                env_config["rewarded"] = True
                env_config["run_id"] = cfg["run_id"] + "_" + "test"
                env_config["rec_path"] = os.path.join(env_config["rec_path"] , f"agent_0/")   
        env = env_cls(**env_config)
    elif env_cls in [ViewpointEnv]:
        env_config = cfg["Environment"]
        with open_dict(env_config):
            object =  "ship" if env_config["use_ship"] else "fork"
            mode = "rest"
            
            side_view= "side" if env_config["side_view"] else "front"
            env_config["mode"] = mode + "-"+ object +"-"+side_view
            env_config["random_pos"] = True
            env_config["rewarded"] = True
            env_config["run_id"] = cfg["run_id"] + "_" + "test"    
            print(env_config)
        env = env_cls(**env_config)   
    else:
        env = None
    obs = env.reset()
    print(env.observation_space, env.action_space)

    video_folder = "test_videos"
    env_class_name = env_cls.__name__
    os.makedirs("test_videos",exist_ok=True)
    video = VideoRecorder(env, f"{video_folder}/{env_class_name}.mp4")
    #gym.logger.set_level(gym.logger.DEBUG)

    try:    
        state = env.reset()
        episode_reward = 0
        for t in count():
            video.capture_frame()
            if len(random_actions) > 0:
                key = random_actions.pop() 
                if key == 'w':
                    action  = [0,0,1]
                elif key == 'a':
                    action  = [0,1,0]
                elif key == 'd':
                    action  = [0,-1,0]
                elif key == 's':
                    action  = [0,0,-1]
                elif key == 'r':
                    action  = [1,0,0]
                elif key == 'f':
                    action  = [-1,0,0]
                   
            else:
                break
            
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward+=reward
            logging.info("reward:{:.4f}".format(reward))
            logging.info(info)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                state = env.reset()
                episode_reward = 0
            env.render(mode="rgb_array")
            video.capture_frame()
            
        print("Saved video.")
        
        video.close()
        video.enabled = False
        assert True
            
    except UnityEnvironmentException as ex:
        logging.error(str(ex))
        print(traceback.print_tb())
        raise ex
    finally:
        video.close()
        env.close()


