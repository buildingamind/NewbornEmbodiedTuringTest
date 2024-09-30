#!/usr/bin/env python3

from typing import Dict, Optional, Tuple

import  gym
import numpy as np
from .networks.inverse_forward_networks import Encoder, ForwardDynamicsModel, InverseDynamicsModel
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
        envs,
        observation_space: gym.Space,
        action_space: gym.Space,
        device: int,
        beta: float = 1.0,
        kappa: float = 0.000025,
        gamma: Optional[float] = None,
        rwd_norm_type: str = "rms",
        obs_norm_type: str = "none",
        latent_dim: int = 128,
        lr: float = 0.001,
        batch_size: int = 256,
        update_proportion: float = 1.0,
        encoder_model: str = "mnih",
        weight_init: str = "orthogonal"
    ) -> None:
        
        self._obs_shape = observation_space.shape
        self._action_shape = action_space.shape
        self._action_dim = action_space.shape[0]
        self._action_type = "Box"
        self.device = th.device("cuda", device)
        self.rwd_norm_type=rwd_norm_type
        self.obs_norm_type=obs_norm_type
        self.gamma = gamma
        self.envs = envs
        self.obs_norm = None
        self.encoder = Encoder(
            obs_shape=self._obs_shape,
            action_dim=self._action_dim,
            latent_dim=latent_dim,
            encoder_model =encoder_model,
            weight_init = weight_init
        ).to(self.device)

        self._beta = beta
        self._kappa = kappa
        
        
        self.im = InverseDynamicsModel(latent_dim=latent_dim, 
                                       action_dim=self._action_dim,encoder_model=encoder_model,
                                       weight_init=weight_init).to(self.device)
        self.im_loss = nn.MSELoss()

        self.fm = ForwardDynamicsModel(latent_dim=latent_dim, action_dim=self._action_dim,
                                       encoder_model=encoder_model,weight_init=weight_init,).to(self.device)

        self.encoder_opt = th.optim.Adam(self.encoder.parameters(), lr=lr)
        self.im_opt = th.optim.Adam(self.im.parameters(), lr=lr)
        self.fm_opt = th.optim.Adam(self.fm.parameters(), lr=lr)
        self.batch_size = batch_size
        self.update_proportion = update_proportion
        self.latent_dim = latent_dim
        
        self.beta = beta
        self.kappa = kappa
        
        ## training tracker
        self.global_step = 0
        self.metrics = {"loss": [], "intrinsic_rewards": []}
        
        
    def compute_irs(self, samples: Dict[str, th.Tensor], sync: bool = True) -> th.Tensor:
        """
        Compute the intrinsic rewards for current samples.

        Args:
            samples (Dict): The collected samples. A python dict like
                {obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                rewards (n_steps, n_envs) <class 'th.Tensor'>,
                next_obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.
            sync (bool): Whether to update the reward module after the `compute` function, default is `True`.


        Returns:
            The intrinsic rewards.
        """
        self.global_step+=1
        
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        
        # get the observations, actions and next observations
        obs_tensor = samples.get("observations").to(self.device)
        actions_tensor = samples.get("actions").to(self.device)
        next_obs_tensor = samples.get("next_observations").to(self.device)
        
        ## normalize observations
        obs_tensor = self.normalize(obs_tensor)
        next_obs_tensor = self.normalize(next_obs_tensor)
        
        # initialize ICM
        intrinsic_rewards = th.zeros(size=(n_steps, n_envs)).to(self.device)
        self.rwd_norm = TorchRunningMeanStd()
        
        
        with th.no_grad():
            for i in range(n_envs):
                encoded_obs = self.encoder(obs_tensor[:, i])
                encoded_next_obs = self.encoder(next_obs_tensor[:, i])
                pred_next_obs = self.fm(encoded_obs, actions_tensor[:, i])
                dist = F.mse_loss(pred_next_obs, encoded_next_obs, reduction="none").mean(dim=1)
                intrinsic_rewards[:, i] = dist.cpu()
        
        ## update the reward module
        if sync:
            self.update(samples)
        
        ## scale intrinsic rewards
        return self.scale(intrinsic_rewards)
    
    @property
    def weight(self) -> float:
        """Get the weighting coefficient of the intrinsic rewards."""
        return self.beta * np.power(1.0 - self.kappa, self.global_step)
    
    def scale(self, intrinsic_rewards: th.Tensor) -> th.Tensor:
        """
        Scale the intrinsic rewards.

        Args:
            intrinsic_rewards (th.Tensor): The intrinsic rewards.

        Returns:
            The scaled intrinsic rewards.
        """
        if self.rwd_norm:
            self.rwd_norm.update(intrinsic_rewards.ravel())
            return (intrinsic_rewards / self.rwd_norm.std) * self.weight
        
        return intrinsic_rewards
    
    
    def normalize(self, x: th.Tensor) -> th.Tensor:
        """
        Normalize the observations data, especially useful for images-based observations."""
        if self.obs_norm:
            x = (
                ((x - self.obs_norm.mean.to(self.device)))
                / th.sqrt(self.obs_norm.var.to(self.device))
            ).clip(-5, 5)
        else:
            x = x / 255.0 if len(self._obs_shape) > 2 else x
        return x
    
    def scale(self, intrinsic_rewards: th.Tensor) -> th.Tensor:
        """
        Scale the intrinsic rewards.

        Args:
            intrinsic_rewards (th.Tensor): The intrinsic rewards.

        Returns:
            The scaled intrinsic rewards.
        """
        if self.rwd_norm:
            intrinsic_rewards = (
                (intrinsic_rewards - self.rwd_norm.mean.to(self.device))
                / th.sqrt(self.rwd_norm.var.to(self.device))
            ).clip(-5, 5)
        print(f"Timestep: {self.global_step} + Intrinsic Reward: {(intrinsic_rewards)}")
        return intrinsic_rewards
    
    
    
    def update(self, samples) -> None:
        """
        Update the intrinsic reward module if necessary.

        Args:
            samples: The collected samples. A python dict like
                {obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>,
                actions (n_steps, n_envs, *action_shape) <class 'th.Tensor'>,
                rewards (n_steps, n_envs) <class 'th.Tensor'>,
                next_obs (n_steps, n_envs, *obs_shape) <class 'th.Tensor'>}.

        Returns:
            None
        """
        
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        
        ## get observations and next observations
        obs_tensor = (
            samples.get("observations").to(self.device).view(-1, *self._obs_shape)
        )
        next_obs_tensor = (
            samples.get("next_observations").to(self.device).view(-1, *self._obs_shape)
        )
        
        
        ## normalize
        obs_tensor = self.normalize(obs_tensor)
        next_obs_tensor = self.normalize(next_obs_tensor)
        # sample actions
        actions_tensor = samples.get("actions").view(n_steps * n_envs, -1)
        
        ## build dataset and dataloader
        dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        avg_im_loss = []
        avg_fm_loss = []
        
        for _idx, batch in enumerate(loader):
            # get batch
            obs, actions, next_obs = batch
            # convert to device
            obs, actions, next_obs = (
                obs.to(self.device),
                actions.to(self.device),
                next_obs.to(self.device),
            )
            ## set the gradients to zero
            self.encoder_opt.zero_grad()
            self.im_opt.zero_grad()
            self.fm_opt.zero_grad()
            
            ## encoder observations and next observations
            encoded_obs = self.encoder(obs)
            encoded_next_obs =  self.encoder(next_obs)
            ## pass it to the inverse model
            pred_actions = self.im(encoded_obs, encoded_next_obs)
            ## compute loss
            im_loss = self.im_loss(pred_actions, actions)
            
            ## forward dyanmics model
            pred_next_obs = self.fm(encoded_obs, actions)
            ## compute loss
            fm_loss = F.mse_loss(pred_next_obs, encoded_next_obs, reduction="none").mean(dim=-1)
            
            ## random mask to select subset of the batch
            x = th.tensor([im_loss.clone()])
            mask = th.rand(len(x), device=self.device)
            
            mask = (mask < self.update_proportion).type(th.FloatTensor).to(self.device)

            # get masked losss
            # get the masked losses
            im_loss = (im_loss * mask).sum() / th.max(
                mask.sum(), th.tensor([1], device=self.device, dtype=th.float32)
            )
            fm_loss = (fm_loss * mask).sum() / th.max(
                mask.sum(), th.tensor([1], device=self.device, dtype=th.float32)
            )
            
            ## backward and update
            
            #db.set_trace()
            (im_loss + fm_loss).backward()
            self.encoder_opt.step()
            self.im_opt.step()
            self.fm_opt.step()
            
            avg_im_loss.append(im_loss.item())
            avg_fm_loss.append(fm_loss.item())
        
        # save the loss
        self.metrics["loss"].append(
            [self.global_step, np.mean(avg_im_loss) + np.mean(avg_fm_loss)]
        )
        
    
        
class TorchRunningMeanStd:
    """Running mean and std for torch tensor."""

    def __init__(self, epsilon=1e-4, shape=(), device=None) -> None:
        self.mean = th.zeros(shape, device=device)
        self.var = th.ones(shape, device=device)
        self.count = epsilon

    def update(self, x) -> None:
        """Update mean and std with batch data."""
        with th.no_grad():
            batch_mean = th.mean(x, dim=0)
            batch_var = th.var(x, dim=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        """Update mean and std with batch moments."""
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self) -> th.Tensor:
        return th.sqrt(self.var)

    def update_mean_var_count_from_moments(
        self, mean, var, count, batch_mean, batch_var, batch_count
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta + batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + th.pow(delta, 2) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count
    