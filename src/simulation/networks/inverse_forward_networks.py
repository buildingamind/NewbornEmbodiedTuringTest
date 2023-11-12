#!/usr/bin/env python3

from typing import Tuple
from torch import nn, optim
import torch as th
import pdb
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder for encoding observations

    Args:
        obs_shape (Tuple): data shape of observations
        action_shape: data shape for action space
        latent_dim: dimension for encoding vectors
        
    Returns:
        instance of the encoder
    
    """
    def __init__(self, obs_shape: Tuple, action_dim: int, latent_dim: int) -> None:
        super().__init__()
        
        self.main = nn.Sequential(
                nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
                nn.ELU(),
                nn.Flatten(),
            )
        with th.no_grad():
            sample = th.ones(size = tuple(obs_shape))
            n_flatten = self.main(sample.unsqueeze(0)).shape[1]
            
        self.linear = nn.Linear(n_flatten, latent_dim)
        
    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Encode input observations

        Args:
            obs (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
            
        return self.linear(self.main(obs))
    
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
    
    def __init__(self, latent_dim, action_dim)-> None:
        super(InverseDynamicsModel, self).__init__()
        
        dim = 2 * latent_dim
        
        self.main = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )
        
        
    
    def forward(self, obs: th.Tensor, next_obs:th.Tensor) -> th.Tensor:
        """Forward functions to predict actions

        Args:
            obs (th.Tensor): current observations
            next_obs (th.Tensor): next observations

        Returns:
            th.Tensor: Predicted observations
        """
        return self.main(th.cat([obs,next_obs],dim=1))
    
class ForwardDynamicsModel(nn.Module):
    """Forward model for reconstructing transistion process

    Args:
        latent_dim (int): The dimension of encoding vectors of the observations.
        action_dim (int): The dimension of predicted actions.

    Returns:
        Model instance.
    """
    
    def __init__(self, latent_dim, action_dim) -> None:
        super(ForwardDynamicsModel, self).__init__()
        self.action_dim = action_dim
        
        self.main = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        
    def forward(self, obs: th.Tensor, action: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted next-obs.

        Args:
            obs (th.Tensor): Current observations.
            pred_actions (th.Tensor): Predicted observations.

        Returns:
            Predicted next-obs.
        """
        #pdb.set_trace()
        return self.main(th.cat([obs, action], dim=1))