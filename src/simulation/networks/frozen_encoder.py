#!/usr/bin/env python3

import numpy as np
import gym
from gym.spaces.box import Box

import os
import pdb

import torch 
import torch.nn as nn

from torch.nn.modules.linear import Identity

import torchvision.models as models
import torchvision.transforms as T

from stable_baselines3.common.torch_layers import (
    NatureCNN
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from networks.disembodied_models.models.simclr import SimCLR
from networks.disembodied_models.models.archs.resnet_2b import resnet_2blocks
from networks.disembodied_models.models.vit_contrastive import VisionTransformer,LitClassifier,ViTConfigExtended,Backbone


from networks.resnet10 import BasicBlock, CustomResnet10CNN, _resnet
from networks.resnet18 import CustomResnet18CNN, ResNet_18

checkpoint_path = "/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/data/checkpoints"


# initialize the model
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data,  mode='fan_out', nonlinearity='relu')
    bias_init(module.bias.data)
    return module

def _get_frozen_encoder(observation_space, encoder_name = "natureCNN", train = False):
    ## NatureCNN
    ## resnet10
    ## resnet18
    ## sim_clr
    global checkpoint_path
    in_channels = 3
    dummy_x = torch.zeros((1, 3, 64, 64))
    
    ## resnet model transforms - Default transforms: https://pytorch.org/vision/stable/models.html
    transforms = nn.Sequential(
        T.Resize(256),
        T.CenterCrop(224),
        T.ConvertImageDtype(torch.float),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    )

    
    ## if encoder is random - 5 layer convolution network with relu activation and initialization
    if encoder_name == "random":
        init_ = lambda m: init(m, nn.init.kaiming_normal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))
        
        model = nn.Sequential(
            init_(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_out_size = np.prod(model(dummy_x).shape)
        
        # Linear layers
        model.fc = nn.Sequential(
            init_(nn.Linear(conv_out_size, 512, bias=True)),
            nn.ReLU()
        )
        
        print(model)
    elif encoder_name == "small":
        checkpoint_path = os.path.join(checkpoint_path,"encoder_small.pth")
        model = NatureCNN(observation_space)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    elif encoder_name == "medium":
        checkpoint_path = "encoder_medium.pth"
        model = CustomResnet10CNN(observation_space)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    elif encoder_name == "large":
        checkpoint_path = "encoder_large.pth"
        model = CustomResnet18CNN(observation_space)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    elif encoder_name == 'resnet50':
        # resnet50(weights=ResNet50_Weights.DEFAULT)
        model = models.resnet50(pretrained=True, progress=False)
        model.fc = Identity()
        
    elif encoder_name == 'resnet18':
        # resnet50(weights=ResNet50_Weights.DEFAULT)
        model = models.resnet18(pretrained=True, progress=False)
        model.fc = Identity()
    
    elif encoder_name == "retinalwaves":
        model = SimCLR.load_from_checkpoint(os.path.join(checkpoint_path, "retinalwaves/epoch=97-step=29106.ckpt"))
    
    elif encoder_name == "simclr":
        model = SimCLR.load_from_checkpoint(os.path.join(checkpoint_path, "sim_clr/epoch=98-step=15542.ckpt"))
    
    elif encoder_name == "untrained_r18_2b":
        
        model = resnet_2blocks(pretrained=False)
        model.fc = nn.Identity()
        
    
    elif encoder_name == "vit":
        #model = VisionTransformer.load_from_checkpoint(os.path.join(checkpoint_path, "vit/retinalwaves/epoch=94-step=14154.ckpt"))
        #pdb.set_trace()
        #model = LitClassifier.load_from_checkpoint(os.path.join(checkpoint_path, "vit/v101trained/epoch=96-step=14258.ckpt"))
        #pdb.set_trace()
        #model.fc = nn.Identity()
        configuration = ViTConfigExtended()
        configuration.image_size = 64
        configuration.patch_size = 8
        configuration.num_hidden_layers = 3
        configuration.num_attention_heads = 3
        # print configuration parameters of ViT
        print('image_size - ', configuration.image_size)
        print('patch_size - ', configuration.patch_size)
        print('num_classes - ', configuration.num_classes)
        print('hidden_size - ', configuration.hidden_size)
        print('intermediate_size - ', configuration.intermediate_size)
        print('num_hidden_layers - ', configuration.num_hidden_layers)
        print('num_attention_heads - ', configuration.num_attention_heads)
        backbone = Backbone('vit', configuration)
        model = LitClassifier(backbone).backbone
        model.fc = nn.Identity()
        
    else:
        print("name not found!")
    
    if not train:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        
    else:
        model.train()
    
    
    return model,transforms
    

class FrozenEncoderNetwork(nn.Module):
    """
    Frozen encoder network to encoder the observations before passing into policy network

    Args:
        nn (_type_): _description_
    """
    def __init__(self, observation_space, encoder_name, train = False):
        super().__init__()
        
        self.encoder_name = encoder_name
        
        self.in_channels = 3
        self.encoder, self.transforms = _get_frozen_encoder(observation_space, encoder_name, train)
        
        if self.encoder == None:
            print("Encoder not found!")
            return 
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.encoder = self.encoder.to(device=self.device)
        dummy_in = torch.zeros(1, self.in_channels, 64, 64)
        
        ## tranform for resnet50
        if self.encoder_name == ["resnet50","resnet18"]:
            dummy_in = self.transforms(dummy_in)
        
        self.in_shape = dummy_in.shape[1:]
        dummy_in = dummy_in.to(self.device)
        
        dummy_out = self.encoder(dummy_in)
        self.out_size = np.prod(dummy_out.shape)
        
    def forward(self, observation):
        # observation.shape -> (N, H, W, 3)
        observation = observation.to(device=self.device)
        
        ## apply transform on the image
        if self.encoder_name in ["resnet50","resnet18"]:
            #observation = observation.transpose(1, 3).contiguous()
            observation = self.transforms(observation)
        
        #observation = observation.reshape(-1, *self.in_shape)
        
        #pdb.set_trace()
        #with torch.no_grad():
        out = self.encoder(observation)
        return out
        #return out.view(-1, self.out_size).squeeze()

class CustomFrozenNetwork(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param feature_dim: (int) Number of feature extractors
    

    Args:
        BaseFeaturesExtractor (_type_): _description_
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, encoder_type = 'random', train = False):
        super(CustomFrozenNetwork, self).__init__(observation_space, features_dim)
        self.encoder_type = encoder_type
        self.model = FrozenEncoderNetwork(observation_space, encoder_type, train)
    
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)


