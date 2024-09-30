#!/usr/bin/env python3

from typing import Tuple
from torch import nn, optim
import torch as th
import pdb
import torch.nn.functional as F
import math
import numpy as np


def orthogonal_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

def default_layer_init(layer):
    stdv = 1. / math.sqrt(layer.weight.size(1))
    layer.weight.data.uniform_(-stdv, stdv)
    if layer.bias is not None:
        layer.bias.data.uniform_(-stdv, stdv)
    return layer

class Encoder(nn.Module):
    """Encoder for encoding observations

    Args:
        obs_shape (Tuple): data shape of observations
        action_shape: data shape for action space
        latent_dim: dimension for encoding vectors
        
    Returns:
        instance of the encoder
    
    """
    def __init__(self, obs_shape: Tuple, action_dim: int, latent_dim: int,
                 encoder_model:str = "mnih", weight_init="default") -> None:
        super().__init__()
        
        
        if weight_init == "orthogonal":
            init_ = orthogonal_layer_init
        elif weight_init == "default":
            init_ = default_layer_init
        else:
            raise ValueError("Invalid weight_init")

        if encoder_model == "mnih" and len(obs_shape) > 2:

            self.main = nn.Sequential(
                    init_(nn.Conv2d(obs_shape[0], 32, 8, stride=4)),
                    nn.ReLU(),
                    init_(nn.Conv2d(32, 64, 4, stride=2)),
                    nn.ReLU(),
                    init_(nn.Conv2d(64, 64, 3, stride=1)),
                    nn.ReLU(),
                    nn.Flatten(),
                )
            
            
            with th.no_grad():
                sample = th.ones(size = tuple(obs_shape))
                n_flatten = self.main(sample.unsqueeze(0)).shape[1]
                
            self.main.append(init_(nn.Linear(n_flatten, latent_dim)))
            self.main.append(nn.ReLU())
        else:
            self.main = nn.Sequential(
                init_(nn.Linear(obs_shape[0], 256)), 
                nn.ReLU()
            )
            self.main.append(init_(nn.Linear(256, latent_dim)))
            
        
    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Encode input observations

        Args:
            obs (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        
        return self.main(obs)
    
    def feature_size(self,device):
        with th.no_grad():
            x = self.forward(th.zeros(1,64,64,3).to(device))
            return x.view(1, -1).size(1)
        
    

class InverseDynamicsModel(nn.Module):
    """Inverse model for reconstructing transition process.

    Args:
        latent_dim (int): The dimension of encoding vectors of the observations.
        action_dim (int): The dimension of predicted actions.

    Returns:
        Model instance.
    """
    
    def __init__(self, latent_dim, action_dim, encoder_model="mnih", weight_init="default") -> None:
        super(InverseDynamicsModel, self).__init__()
        #obs_shape: Tuple, action_dim: int, latent_dim: int,
        self.trunk = Encoder(obs_shape=(latent_dim * 2,), 
                             latent_dim=action_dim, 
                             action_dim = action_dim,
                             encoder_model=encoder_model, 
                             weight_init=weight_init)

        
        
    
    def forward(self, obs: th.Tensor, next_obs:th.Tensor) -> th.Tensor:
        """Forward functions to predict actions

        Args:
            obs (th.Tensor): current observations
            next_obs (th.Tensor): next observations

        Returns:
            th.Tensor: Predicted observations
        """
        
        return self.trunk(th.cat([obs, next_obs], dim=1))
    
class ForwardDynamicsModel(nn.Module):
    """Forward model for reconstructing transistion process

    Args:
        latent_dim (int): The dimension of encoding vectors of the observations.
        action_dim (int): The dimension of predicted actions.

    Returns:
        Model instance.
    """
    
    def __init__(self, latent_dim, action_dim, encoder_model="mnih", weight_init="default") -> None:
        super(ForwardDynamicsModel, self).__init__()
        self.action_dim = action_dim
        
        self.trunk = Encoder(obs_shape=(latent_dim + action_dim,), 
                             action_dim = action_dim,
                             latent_dim=latent_dim, 
                             encoder_model=encoder_model, weight_init=weight_init)

        
    def forward(self, obs: th.Tensor, pred_actions: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted next-obs.

        Args:
            obs (th.Tensor): Current observations.
            pred_actions (th.Tensor): Predicted observations.

        Returns:
            Predicted next-obs.
        """
        
        return self.trunk(th.cat([obs, pred_actions], dim=1))