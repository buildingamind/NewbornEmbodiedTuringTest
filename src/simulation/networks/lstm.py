#!/usr/bin/env python3

import pdb
import gym


import torch as th
import torch.nn as nn
import torchvision

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNNLSTM(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, 
                  features_dim: int = 256):
        rnn_hidden_size = 100
        rnn_num_layers = 1
        super(CustomCNNLSTM, self).__init__(observation_space, features_dim)
        
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

         # define LSTM layer
        hidden_size = 512
        self.lstm = nn.LSTM(input_size = n_flatten, hidden_size = hidden_size,
                            num_layers = 1)
        
        self.linear = nn.Sequential(nn.Linear(hidden_size, features_dim), nn.ReLU())
        
    def forward(self, observations: th.Tensor) :
        observations = observations.unsqueeze(0)
        batch_size, seq_length, c, h, w = observations.shape
        ii = 0
        y = self.cnn((observations[:,ii]))
        
        #outputs = [sen_len, batch_size, hid_dim * n_directions]
        out, (hn, cn) = self.lstm(y.unsqueeze(1))
        out = self.linear(out[:,-1]) 
        
        return out 

    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x