#!/usr/bin/env python3
import pdb
from typing import Dict, Tuple

import  gym
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


from .networks.target_predict_networks import RNDEncoder


class RND(object):
    """Exploration by Random Network Distillation (RND).
        See paper: https://arxiv.org/pdf/1810.12894.pdf

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
        Instance of RND.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: int,
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
        self._device = th.device("cuda", device)
        
        self.predictor = RNDEncoder(
            obs_shape=self._obs_shape,
            action_dim=self._action_dim,
            latent_dim=latent_dim,
        ).to(self._device)
        
        self.target = RNDEncoder(
            obs_shape=self._obs_shape,
            action_dim=self._action_dim,
            latent_dim=latent_dim,
        ).to(self._device)

        self._device = th.device(device)
        self._beta = beta
        self._kappa = kappa
        
        self.opt = th.optim.Adam(self.predictor.parameters(), lr=lr)
        self.batch_size = batch_size
        
        # freeze the network parameters
        for p in self.target.parameters():
            p.requires_grad = False
            
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
        next_obs = th.from_numpy(samples["next_obs"])
        
        
        num_steps = obs.size()[0]
        num_envs = obs.size()[1]
        
        obs_tensor =  obs.to(self._device)
        actions_tensor = actions.to(self._device)
        next_obs_tensor = next_obs.to(self._device)
        
        intrinsic_rewards = th.zeros(size=(num_steps, num_envs))
        
        with th.no_grad():
            for i in range(num_envs):
                src_feats = self.predictor(next_obs_tensor[:, i])
                target_feats = self.target(next_obs_tensor[:, i])
                dist = F.mse_loss(src_feats, target_feats, reduction="none").mean(dim=1)
                dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-11)
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
        
        
        dataset = TensorDataset(obs_tensor)
        loader = DataLoader(dataset = dataset, batch_size = self.batch_size)
        
        for _idx, batch_data in enumerate(loader):
            obs = batch_data[0]
            src_feats = self.predictor(obs)
            with th.no_grad():
                tgt_feats = self.target(obs)

            self.opt.zero_grad()
            loss = F.mse_loss(src_feats, tgt_feats)
            loss.backward()
            self.opt.step()
    