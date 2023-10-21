#!/usr/bin/env python
import argparse
from itertools import count
import random
import sys
import warnings
import logging
import yaml


import sys

sys.path.append("/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/")
from networks.resnet10 import CustomResnet10CNN
from networks.resnet18 import CustomResnet18CNN
from common.logger import Logger
from env_wrapper.parsing_env_wrapper import ParsingEnv

import torch
from stable_baselines3 import PPO
import os

import pdb

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)

from mlagents_envs.exception import (UnityEnvironmentException,
                                     UnityWorkerInUseException)
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf, open_dict
import sys
import hydra
import traceback

# setting path

import cv2
import matplotlib.pyplot as plt 
import numpy as np
from PIL  import Image
from visualize_encoder import VisualizeEncoder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv,VecTransposeImage
import shutil

warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="path to the model", type=str)
parser.add_argument("--encoder_type", help="type of the encoder - small, medium, large", type=str)



def save_load_encoder_statedict(observation_space, model_path, encoder_type = "small"):
    """
    {'use_ship': True, 'background': 'A', 'base_port': 5100, 
    'env_path': '/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/data/executables/parsing_benchmark/parsing.x86_64', 
    'log_path': '${log_path}/Env_Logs', 'rec_path': '/data/mchivuku/embodiedai/benchmark_experiments/parsing_ship_A_med_exp1/Recordings/parsing_ship_A_med_exp1_Agent_0/', 'record_chamber': False, 'record_agent': False, 'recording_frames': 1000, 'dvs_wrapper': False, 'mode': 'rest-ship-A', 'random_pos': True, 'rewarded': True, 
    'run_id': 'parsing_ship_A_med_exp1_Agent_0_train', 
    'log_title': 'ship_A-parsing_ship_A_med_exp1_Agent_0_train'}

    Args:
        model_path (_type_): _description_
        encoder_type (str, optional): _description_. Defaults to "medium".

    Returns:
        _type_: _description_
    """
    ## env
    base_path, model_name = os.path.split(model_path)
    feature_vector_pth = os.path.join(base_path, "feature_extractor.pth")
    
    ## Save feature vector if doesnt exist
    if(not os.path.exists(feature_vector_pth)):
        model = PPO.load(model_path, device="cpu",print_system_info=True)        
        ## save policy
        policy = model.policy
        policy.save(os.path.join(base_path, "policy.pkl"))
        ## save encoder
        encoder = model.policy.features_extractor.state_dict()
        torch.save(encoder,os.path.join(base_path, "feature_extractor.pth"))
        
    
    print(observation_space.shape)
    
    if encoder_type.lower()=="small":
        trained_model = NatureCNN(observation_space)
        trained_model.load_state_dict(torch.load(os.path.join(base_path, "feature_extractor.pth")))
    elif encoder_type.lower() =="medium":
        trained_model = CustomResnet10CNN(observation_space,128)
        trained_model.load_state_dict(torch.load(os.path.join(base_path, "feature_extractor.pth")))
    elif encoder_type.lower()=="large":
        trained_model = CustomResnet18CNN(observation_space,128)
        trained_model.load_state_dict(torch.load(os.path.join(base_path, "feature_extractor.pth")))
    else:
        trained_model = NatureCNN(observation_space)
        trained_model.load_state_dict(torch.load(os.path.join(base_path, "feature_extractor.pth")))
    
    ## save as onnx format 
    observation_shape = observation_space.shape
    return trained_model, observation_shape

def get_env_observation_space():
    with open('config.yaml', 'r') as f:
        env_config = yaml.full_load(f)
        print(env_config)
        env_config["mode"] = "ship-A-rest"
        env_config["random_pos"] = True
        env_config["rewarded"] = True
        env_config["run_id"] = "encoder_analysis"
        env_config["rec_path"] = os.path.join(env_config["rec_path"] , f"agent_0/")   
        env_config["log_path"] =env_config["rec_path"]
        
        env = ParsingEnv(**env_config)
        
    e_gen = lambda : env
    envs = make_vec_env(env_id=e_gen, n_envs=1)
    print(envs.observation_space.shape)
    #envs = VecTransposeImage(envs)
    observation_space = envs.observation_space
    return observation_space


def main():
    args = parser.parse_args()
    observation_space = get_env_observation_space()

    # 2. save encoder state dictionary
    #model_base_dir = "/data/mchivuku/embodiedai/benchmark_experiments/ship-new/"
    model_base_dir = "/data/mchivuku/samantha_output/dvs_wrapper_new"
    model_paths =[os.path.join(model_base_dir,f) for f in os.listdir(model_base_dir) if not os.path.isfile(os.path.join(model_base_dir, f))]
    
    for model_path in model_paths:
        name = model_path.split("/")[-1]+"_Agent_0"
        
        model_path = os.path.join(model_path, "Brains", name,"model")
        base_path, _ = os.path.split(model_path) 
        
        if 'med' in name:
            encoder_type = "medium"
            vector_length = 128
        elif 'large' in name:
            encoder_type = "large"
            vector_length = 128
        else:
            encoder_type = "small"
            vector_length = 512
        encoder, observation_shape = save_load_encoder_statedict(observation_space, model_path,encoder_type)

        copy_image_path = os.path.join("tSNE_enc_plots", "dvs_wrapper")
        os.makedirs(copy_image_path,exist_ok=True)
        
        ## 3. save and visualize gradcam images
        if os.path.exists(os.path.join(base_path, "plots","encoder_analysis.png")):
            print(os.path.join(copy_image_path, name +"_enc_analysis.png"))
            shutil.copy(os.path.join(base_path, "plots","encoder_analysis.png"), 
                        os.path.join(copy_image_path, name +"_enc_analysis.png"))
        else:
            viz = VisualizeEncoder(encoder, os.path.join(base_path, "plots"),vector_length,True)
            viz.visualize_tsne()
        
    
if __name__=="__main__":
    main()
    