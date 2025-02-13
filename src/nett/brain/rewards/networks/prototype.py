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
# copied from:
# https://github.com/RLE-Foundation/rllte/main/rllte/common/prototype/base_reward.py
# https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/utils.py
# https://github.com/RLE-Foundation/rllte/blob/main/rllte/common/preprocessing.py

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from stable_baselines3.common.vec_env import VecEnv as VectorEnv


import numpy as np
import torch as th


from typing import Dict, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

ObsShape = Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]


class BaseReward(ABC):
    """Base class of reward module.

    Args:
        envs (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        gamma (Optional[float]): Intrinsic reward discount rate, default is `None`.
        rwd_norm_type (str): Normalization type for intrinsic rewards from ['rms', 'minmax', 'none'].
        obs_norm_type (str): Normalization type for observations data from ['rms', 'none'].

    Returns:
        Instance of the base reward module.
    """

    def __init__(
        self,
        envs: VectorEnv,
        device: str,
        beta: float = 1.0,
        kappa: float = 0.0,
        gamma: Optional[float] = None,
        rwd_norm_type: str = "rms",
        obs_norm_type: str = "rms",
    ) -> None:
        # get environment information
        self.action_space = envs.action_space
        self.n_envs = envs.num_envs
        ## process the observation and action space
        observation_space = envs.observation_space
        self.obs_shape: Tuple = observation_space.shape

        action_shape = self.action_space.shape
        self.policy_action_dim = int(np.prod(self.action_space.shape))
        action_dim = self.policy_action_dim
        action_type = "Box"

        # set device and parameters
        self.device = device
        self.beta = beta
        self.kappa = kappa
        self.rwd_norm_type = rwd_norm_type
        self.obs_norm_type = obs_norm_type
        # build the running mean and std for normalization
        self.rwd_norm = (
            TorchRunningMeanStd(device=device) if self.rwd_norm_type == "rms" else None
        )

        if self.obs_norm_type == "rms":
            self.obs_norm = TorchRunningMeanStd(shape=self.obs_shape, device=device)
            # initialize the normalization parameters if necessary
            self.envs = envs
            self.init_normalization()
        else:
            self.obs_norm = None
        # build the reward forward filter
        self.rff = RewardForwardFilter(gamma) if gamma is not None else None
        # training tracker
        self.global_step = 0
        self.metrics = {"loss": [], "intrinsic_rewards": []}

    @property
    def weight(self) -> float:
        """Get the weighting coefficient of the intrinsic rewards."""
        return self.beta * np.power(1.0 - self.kappa, self.global_step)

    def scale(self, rewards: th.Tensor) -> th.Tensor:
        """Scale the intrinsic rewards.

        Args:
            rewards (th.Tensor): The intrinsic rewards with shape (n_steps, n_envs).

        Returns:
            The scaled intrinsic rewards.
        """
        # update reward forward filter if necessary
        if self.rff is not None:
            for step in range(rewards.size(0)):
                rewards[step] = self.rff.update(rewards[step])
        # scale the intrinsic rewards
        if self.rwd_norm_type == "rms":
            self.rwd_norm.update(rewards.ravel())
            return (rewards / self.rwd_norm.std) * self.weight
        elif self.rwd_norm_type == "minmax":
            return (
                (rewards - rewards.min())
                / (rewards.max() - rewards.min())
                * self.weight
            )
        else:
            return rewards * self.weight

    def normalize(self, x: th.Tensor) -> th.Tensor:
        """Normalize the observations data, especially useful for images-based observations."""
        if self.obs_norm:
            x = (
                ((x - self.obs_norm.mean.to(self.device)))
                / th.sqrt(self.obs_norm.var.to(self.device))
            ).clip(-5, 5)
        else:
            x = x / 255.0 if len(self.obs_shape) > 2 else x
        return x

    def init_normalization(self) -> None:
        """Initialize the normalization parameters for observations if the RMS is used."""
        # TODO: better initialization parameters?
        num_steps, num_iters = 128, 20
        _ = self.envs.reset()
        if self.obs_norm_type == "rms":
            all_next_obs = []
            for _ in range(num_steps * num_iters):
                actions = [self.action_space.sample() for _ in range(self.n_envs)]
                actions = np.stack(actions)

                next_obs, _, _, _ = self.envs.step(actions)

                all_next_obs += th.as_tensor(next_obs).view(-1, *self.obs_shape)
                # update the running mean and std
                if len(all_next_obs) % (num_steps * self.n_envs) == 0:
                    all_next_obs = th.stack(all_next_obs).float()
                    self.obs_norm.update(all_next_obs)
                    all_next_obs = []

    def watch(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        terminateds: th.Tensor,
        truncateds: th.Tensor,
        next_observations: th.Tensor,
    ) -> Optional[Dict[str, th.Tensor]]:
        """Watch the interaction processes and obtain necessary elements for reward computation.

        Args:
            observations (th.Tensor): Observations data with shape (n_envs, *obs_shape).
            actions (th.Tensor): Actions data with shape (n_envs, *action_shape).
            rewards (th.Tensor): Extrinsic rewards data with shape (n_envs).
            terminateds (th.Tensor): Termination signals with shape (n_envs).
            truncateds (th.Tensor): Truncation signals with shape (n_envs).
            next_observations (th.Tensor): Next observations data with shape (n_envs, *obs_shape).

        Returns:
            Feedbacks for the current samples.
        """

    @abstractmethod
    def compute(self, samples: Dict[str, th.Tensor], sync: bool = True) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors,
                whose keys are ['observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'].
                For example, the data shape of 'observations' is (n_steps, n_envs, *obs_shape).
            sync (bool): Whether to update the reward module after the `compute` function, default is `True`.

        Returns:
            The intrinsic rewards.
        """
        for key in [
            "observations",
            "actions",
            "rewards",
            "terminateds",
            "truncateds",
            "next_observations",
        ]:
            assert key in samples.keys(), f"Key {key} is not in samples."

        # update the obs RMS if necessary
        if self.obs_norm_type == "rms" and sync:
            self.obs_norm.update(
                samples["observations"].reshape(-1, *self.obs_shape).cpu()
            )
        # update the global step
        self.global_step += 1

    @abstractmethod
    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.

        Returns:
            None.
        """


class RewardForwardFilter:
    """Reward forward filter."""

    def __init__(self, gamma: float = 0.99) -> None:
        self.rewems = None
        self.gamma = gamma

    def update(self, rews: th.Tensor) -> th.Tensor:
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class TorchRunningMeanStd:
    """Running mean and std for torch tensor."""

    def __init__(self, epsilon=1e-4, shape=(), device=None) -> None:
        self.mean: th.Tensor = th.zeros(shape, device=device)
        self.var: th.Tensor = th.ones(shape, device=device)
        self.count: th.Tensor = epsilon

    def update(self, x) -> None:
        """Update mean and std with batch data."""
        with th.no_grad():
            batch_mean = th.mean(x, dim=0)
            batch_var = th.var(x, dim=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        """Update mean and std with batch moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean += delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + th.pow(delta, 2) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    @property
    def std(self) -> th.Tensor:
        return th.sqrt(self.var)
