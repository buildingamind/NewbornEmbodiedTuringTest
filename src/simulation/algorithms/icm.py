#!/usr/bin/env python3

import pdb
from typing import Dict, Tuple

import  gym
import numpy as np
from networks.inverse_forward_networks import Encoder, ForwardDynamicsModel, InverseDynamicsModel
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class ICM(object):
    """Curiosity-Driven Exploration by Self-Supervised Prediction.
        See paper: http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf

    Args:
        observation_space (Space): The observation space of environment.
        action_space (Space): The action space of environment.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.
        latent_dim (int): The dimension of encoding vectors.
        lr (float): The learning rate.
        batch_size (int): The batch size for update.

    Returns:
        Instance of ICM.
    """
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: str = "cpu",
        beta: float = 0.01,
        kappa: float = 0.000025,
        latent_dim: int = 128,
        lr: float = 3e-4,
        batch_size: int = 500,
    ) -> None:
        
        self._obs_shape = observation_space.shape
        self._action_shape = action_space.shape
        self._action_dim = action_space.shape[0]
        self._action_type = "Box"
        self._device = device
        
        self.encoder = Encoder(
            obs_shape=self._obs_shape,
            action_dim=self._action_dim,
            latent_dim=latent_dim,
        ).to(self._device)

        self._device = th.device(device)
        self._beta = beta
        self._kappa = kappa
        
        
        self.im = InverseDynamicsModel(latent_dim=latent_dim, action_dim=self._action_dim).to(self._device)
        self.im_loss = nn.MSELoss()

        self.fm = ForwardDynamicsModel(latent_dim=latent_dim, action_dim=self._action_dim).to(self._device)

        self.encoder_opt = th.optim.Adam(self.encoder.parameters(), lr=lr)
        self.im_opt = th.optim.Adam(self.im.parameters(), lr=lr)
        self.fm_opt = th.optim.Adam(self.fm.parameters(), lr=lr)
        self.batch_size = batch_size
        
        
    def compute_irs(self, samples: Dict, step: int = 0) -> th.Tensor:
        """Compute the intrinsic rewards for current samples.

        Args:
            samples (Dict): The collected samples. A python dict like
                {obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                rewards (n_steps, n_envs) <class 'th.Tensor'>,
                next_obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.
            step (int): The global training step.

        Returns:
            The intrinsic rewards.
        """
        # compute the weighting coefficient of timestep t
        beta_t = self._beta * np.power(1.0 - self._kappa, step)
        
        obs = th.from_numpy(samples["obs"])
        actions = th.from_numpy(samples["actions"])
        
        
        num_steps = obs.size()[0]
        num_envs = obs.size()[1]
        
        obs_tensor =  obs.to(self._device)
        actions_tensor = actions.to(self._device)
        
        
        
        
        intrinsic_rewards = th.zeros(size=(num_steps, num_envs))
        
        with th.no_grad():
            for i in range(num_envs):
                encoded_obs = self.encoder(obs_tensor[:, i])
                encoded_next_obs = encoded_obs[:-1]
                pred_next_obs = self.fm(encoded_obs[:-1], actions_tensor[:-1, i])
                dist = th.linalg.vector_norm(encoded_next_obs - pred_next_obs, ord=2, dim=1)
                intrinsic_rewards[:-1,i] = dist.cpu()


        self.update(samples)

        return intrinsic_rewards * beta_t

    def update(self, samples) -> None:
        """Update the intrinsic reward module if necessary.

        Args:
            samples: The collected samples. A python dict like
                {obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                rewards (n_steps, n_envs) <class 'th.Tensor'>,
                next_obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.

        Returns:
            None
        """
        
        num_steps = samples["obs"].shape[0]
        num_envs = samples["obs"].shape[1]
        
        
        obs_tensor = th.from_numpy(samples["obs"]).view((num_envs * num_steps, *self._obs_shape)).to(self._device)
        actions_tensor = th.from_numpy(samples["actions"]).to(self._device)

        encoded_obs = self.encoder(obs_tensor)
        
        dataset = TensorDataset(encoded_obs[:-1], actions_tensor[:-1], encoded_obs[1:])
        loader = DataLoader(dataset = dataset, batch_size = self.batch_size)
        
        for _idx, batch in enumerate(loader):
            obs, actions, next_obs = batch
            actions = actions.squeeze(1)
            
            #db.set_trace()
            pred_actions = self.im(obs, next_obs)
            im_loss = self.im_loss(pred_actions, actions)
            
            pred_next_obs = self.fm(obs,actions )
            
            fm_loss = F.mse_loss(pred_next_obs, next_obs)
            loss = (im_loss + fm_loss)
            
            self.im_opt.zero_grad()
            self.fm_opt.zero_grad()

            loss.backward(retain_graph = True)
            self.im_opt.step()
            self.fm_opt.step()
    
        

    