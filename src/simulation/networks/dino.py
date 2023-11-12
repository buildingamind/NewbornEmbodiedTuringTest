#!/usr/bin/env python3
import pdb
import gym


import torch as th
import torch.nn as nn
import torchvision
import timm
from torchvision.transforms import Compose
from torchvision.transforms import Resize, CenterCrop, Normalize, InterpolationMode

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DinoV1(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384):
        super(DinoV1, self).__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]
        self.transforms = Compose([Resize(size=248, 
                                          interpolation=InterpolationMode.BICUBIC, 
                                          max_size=None, 
                                          antialias=True),
                                   CenterCrop(size=(224, 224)),
                                   Normalize(mean=th.tensor([0.4850, 0.4560, 0.4060]), 
                                             std=th.tensor([0.2290, 0.2240, 0.2250]))])
        
        n_input_channels = observation_space.shape[0]
        print("N_input_channels", n_input_channels)
        
        self.model = timm.create_model('vit_small_patch8_224.dino', 
                                       in_chans=self.n_input_channels, 
                                       num_classes=0, 
                                       pretrained=True)
        
        ## loading pretrained model
        #self.model.eval()
        #for param in self.model.parameters():
        #    param.requires_grad = False
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Cut off image
        # reshape to from vector to W*H
        # gray to color transform
        # application of ResNet
        # Concat features to the rest of observation vector
        # return
        return self.model(self.transforms(observations))
    
class DinoV2(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384):
        super(DinoV2, self).__init__(observation_space, features_dim)
        self.n_input_channels = observation_space.shape[0]
        self.transforms = Compose([Resize(size=256, 
                                          interpolation=InterpolationMode.BICUBIC, 
                                          max_size=None, 
                                          antialias=True),
                                   CenterCrop(size=(224, 224)),
                                   Normalize(mean=th.tensor([0.485, 0.456, 0.406]), 
                                             std=th.tensor([0.229, 0.224, 0.225]))])
        
        n_input_channels = observation_space.shape[0]
        print("N_input_channels", n_input_channels)
        
        
        self.model = th.hub.load('facebookresearch/dinov2', 'dinov2_vits14',pretrained=True)
        
        # get model specific transforms (normalization, resize)
        #data_config = timm.data.resolve_model_data_config(self.model)
        #self.transforms = timm.data.create_transform(**data_config, is_training=False)
        ## loading pretrained model
        
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Cut off image
        # reshape to from vector to W*H
        # gray to color transform
        # application of ResNet
        # Concat features to the rest of observation vector
        # return
        return self.model(self.transforms(observations))