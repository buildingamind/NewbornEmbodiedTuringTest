# =============================================================================
# MIT License

# Copyright (c) 2024 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


from typing import Dict

import numpy as np
import torch as th
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from rllte.xplore.reward import E3B as BaseE3B


class E3B(BaseE3B):
    """Exploration via Elliptical Episodic Bonuses (E3B).
        See paper: https://proceedings.neurips.cc/paper_files/paper/2022/file/f4f79698d48bdc1a6dec20583724182b-Paper-Conference.pdf

    Args:
        envs (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        gamma (Optional[float]): Intrinsic reward discount rate, default is `None`.
        rwd_norm_type (str): Normalization type for intrinsic rewards from ['rms', 'minmax', 'none'].
        obs_norm_type (str): Normalization type for observations data from ['rms', 'none'].

        ridge (float): The ridge parameter for the covariance matrix.
        latent_dim (int): The dimension of encoding vectors.
        lr (float): The learning rate.
        batch_size (int): The batch size for training.
        update_proportion (float): The proportion of the training data used for updating the forward dynamics models.
        encoder_model (str): The network architecture of the encoder from ['mnih', 'pathak'].
        weight_init (str): The weight initialization method from ['default', 'orthogonal'].

    Returns:
        Instance of E3B.
    """

    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.

        Returns:
            None.
        """
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        # get the observations
        obs_tensor = (
            samples.get("observations").to(self.device).view(-1, *self.obs_shape)
        )
        next_obs_tensor = (
            samples.get("next_observations").to(self.device).view(-1, *self.obs_shape)
        )
        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        next_obs_tensor = self.normalize(next_obs_tensor)

        # apply one-hot encoding if the action type is discrete
        if self.action_type == "Discrete":
            actions_tensor = samples.get("actions").view(n_steps * n_envs)
            actions_tensor = F.one_hot(
                actions_tensor.long(), self.policy_action_dim
            ).float()
        else:
            actions_tensor = samples.get("actions").view(n_steps * n_envs, -1)

        # build the dataset and loader
        dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        avg_im_loss = []
        # update the encoder and inverse dynamics model
        for _idx, batch_data in enumerate(loader):
            # get the batch data
            obs, actions, next_obs = batch_data
            obs, actions, next_obs = (
                obs.to(self.device),
                actions.to(self.device),
                next_obs.to(self.device),
            )
            # zero the gradients
            self.encoder_opt.zero_grad()
            self.im_opt.zero_grad()
            # encode the observations
            encoded_obs = self.encoder(obs)
            encoded_next_obs = self.encoder(next_obs)
            # get the predicted actions
            pred_actions = self.im(encoded_obs, encoded_next_obs)
            # compute the inverse dynamics loss
            im_loss = self.im_loss(pred_actions, actions)
            # use a random mask to select a subset of the training data
            mask = th.rand(len(im_loss), device=self.device)
            mask = (mask < self.update_proportion).type(th.FloatTensor).to(self.device)
            # expand the mask to match action spaces > 1
            mask = mask.unsqueeze(1).expand_as(im_loss)
            # get the masked loss
            im_loss = (im_loss * mask).sum() / th.max(
                mask.sum(), th.tensor([1], device=self.device, dtype=th.float32)
            )
            # backward and update
            im_loss.backward()
            self.encoder_opt.step()
            self.im_opt.step()
            # record the loss
            avg_im_loss.append(im_loss.item())
        # save the average loss
        self.metrics["loss"].append([self.global_step, np.mean(avg_im_loss)])
