#!/usr/bin/env python3
import argparse
from itertools import count
import random
import sys
import warnings
import logging
import yaml


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
import argparse

import cv2
import matplotlib.pyplot as plt 
import numpy as np
from PIL  import Image
from visualize_encoder import VisualizeEncoder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv,VecTransposeImage
import shutil
import pytorch_lightning as pl
import torch
import torch.nn as nn
import pdb
sys.path.append("/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/src/simulation/")
from networks.disembodied_models.models.simclr import SimCLR    
    
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="path to the model", type=str)
parser.add_argument("--model", help="type of the encoder - small, medium, large", type=str)

def init_model(args):
    if args.model == 'pixels':
        model = nn.Flatten()
    elif args.model == 'simclr':
        if args.model_path.endswith(".pth"):
            kwargs = {}
            model = SimCLR(gpus=1,num_samples=500, batch_size = 200)
            model.load_state_dict(torch.load(args.model_path))
            model.eval()      
        else:
            model = SimCLR.load_from_checkpoint(args.model_path)
    elif args.model == 'untrained':
        model = resnet18()
        model.fc = nn.Identity()
    pdb.set_trace()
    return model




def main():
    args = parser.parse_args()
    try:
        model = init_model(args)
    except Exception as ex:
        print(str(ex))
        print("Exception occurred while loading the model")
    
    ## if model
    base_path = "tSNE_enc_plots"
    os.makedirs(os.path.join(base_path, args.model), exist_ok=True)
    
    viz = VisualizeEncoder(model, os.path.join(base_path, args.model),512,False)
    viz.visualize_tsne()
        

if __name__=="__main__":
    main()