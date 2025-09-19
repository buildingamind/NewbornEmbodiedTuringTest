"""
RLlib compatibility layer to replace stable-baselines3 functionality.

This module provides base classes and utilities to maintain the existing API
while using RLlib as the underlying RL framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type, Union, Callable
from pathlib import Path

import numpy as np
import torch as th
import gymnasium as gym

from ray import tune
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()


class BaseAlgorithm(ABC):
    """Base class to mimic stable-baselines3 BaseAlgorithm interface."""
    
    def __init__(
        self,
        policy: str,
        env,
        learning_rate: float = 3e-4,
        batch_size: int = 512,
        n_steps: int = 2048,
        verbose: int = 1,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        seed: Optional[int] = None,
        tensorboard_log: Optional[str] = None,
        **kwargs
    ):
        self.policy = policy
        self.env = env
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.verbose = verbose
        self.policy_kwargs = policy_kwargs or {}
        self.device = device
        self.seed = seed
        self.tensorboard_log = tensorboard_log
        self.custom_kwargs = kwargs
        
        # Initialize RLlib algorithm
        self._algorithm: Optional[Algorithm] = None
        self._setup_algorithm()
    
    @abstractmethod
    def _get_algorithm_class(self) -> Type[Algorithm]:
        """Return the RLlib algorithm class."""
        pass
    
    @abstractmethod
    def _get_default_config(self) -> AlgorithmConfig:
        """Return the default configuration for the algorithm."""
        pass
    
    def _setup_algorithm(self):
        """Setup the RLlib algorithm with configuration."""
        config = self._get_default_config()
        
        # Configure basic settings
        config = config.environment(env=self.env)
        config = config.training(
            lr=self.learning_rate,
            train_batch_size=self.batch_size,
        )
        config = config.framework("torch")
        
        if self.seed is not None:
            config = config.environment(env_config={"seed": self.seed})
        
        # Handle custom model if specified in policy_kwargs
        if "features_extractor_class" in self.policy_kwargs:
            model_name = f"custom_model_{id(self)}"
            ModelCatalog.register_custom_model(
                model_name,
                self._create_custom_model_wrapper(
                    self.policy_kwargs["features_extractor_class"],
                    self.policy_kwargs.get("features_extractor_kwargs", {})
                )
            )
            config = config.training(model={"custom_model": model_name})
        
        self._algorithm = config.build()
    
    def _create_custom_model_wrapper(self, extractor_class, extractor_kwargs):
        """Create a wrapper to adapt SB3 feature extractor to RLlib model."""
        class CustomModelWrapper(TorchModelV2, nn.Module):
            def __init__(self, obs_space, action_space, num_outputs, model_config, name):
                TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
                nn.Module.__init__(self)
                
                # Create the feature extractor
                self.features_extractor = extractor_class(obs_space, **extractor_kwargs)
                
                # Create value and policy heads
                self.value_head = nn.Linear(self.features_extractor.features_dim, 1)
                self.policy_head = nn.Linear(self.features_extractor.features_dim, num_outputs)
                
                self._value_out = None
            
            @override(TorchModelV2)
            def forward(self, input_dict, state, seq_lens):
                obs = input_dict["obs"]
                features = self.features_extractor(obs)
                
                # Store value for value_function call
                self._value_out = self.value_head(features).squeeze(-1)
                
                logits = self.policy_head(features)
                return logits, state
            
            @override(TorchModelV2)
            def value_function(self):
                return self._value_out
        
        return CustomModelWrapper
    
    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False
    ):
        """Train the algorithm for the specified number of timesteps."""
        if self._algorithm is None:
            raise RuntimeError("Algorithm not initialized")
        
        # Convert timesteps to iterations (approximate)
        iterations = max(1, total_timesteps // self.n_steps)
        
        for i in range(iterations):
            result = self._algorithm.train()
            
            if callback is not None:
                # Execute callbacks if provided
                if hasattr(callback, '__iter__'):
                    for cb in callback:
                        if hasattr(cb, '_on_step'):
                            cb._on_step()
                elif hasattr(callback, '_on_step'):
                    callback._on_step()
            
            if self.verbose >= 1 and i % log_interval == 0:
                print(f"Iteration {i}: {result}")
    
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Predict action given observation."""
        if self._algorithm is None:
            raise RuntimeError("Algorithm not initialized")
        
        action = self._algorithm.compute_single_action(
            observation,
            explore=not deterministic
        )
        return action, state
    
    def save(self, path: str):
        """Save the trained model."""
        if self._algorithm is None:
            raise RuntimeError("Algorithm not initialized")
        
        checkpoint_path = self._algorithm.save(path)
        return checkpoint_path
    
    @classmethod
    def load(cls, path: str, env=None, device: str = "auto", **kwargs):
        """Load a trained model."""
        # This is a simplified implementation
        # In practice, you'd need to restore the algorithm configuration
        instance = cls(policy="DummyPolicy", env=env, device=device, **kwargs)
        if instance._algorithm is not None:
            instance._algorithm.restore(path)
        return instance


class BaseFeaturesExtractor(nn.Module, ABC):
    """
    Base class for feature extractors to replace SB3's BaseFeaturesExtractor.
    
    This maintains compatibility with existing encoder implementations.
    """
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__()
        self.observation_space = observation_space
        self.features_dim = features_dim
    
    @abstractmethod
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Extract features from observations."""
        pass


class BasePolicy(ABC):
    """Base class for policies to replace SB3's BasePolicy interface."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def save(self, path: Path):
        """Save policy (placeholder for compatibility)."""
        pass


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from Nature paper - replacement for SB3's NatureCNN.
    
    :param observation_space:
    :param features_dim: Number of features extracted.
        Corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or if normalization should be applied (default: False)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim)
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
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


# Specific algorithm implementations
class PPO(BaseAlgorithm):
    """PPO algorithm wrapper for RLlib."""
    
    def _get_algorithm_class(self):
        from ray.rllib.algorithms.ppo import PPO as RLlibPPO
        return RLlibPPO
    
    def _get_default_config(self):
        from ray.rllib.algorithms.ppo import PPOConfig
        return PPOConfig()


class SAC(BaseAlgorithm):
    """SAC algorithm wrapper for RLlib."""
    
    def _get_algorithm_class(self):
        from ray.rllib.algorithms.sac import SAC as RLlibSAC
        return RLlibSAC
    
    def _get_default_config(self):
        from ray.rllib.algorithms.sac import SACConfig
        return SACConfig()


class DQN(BaseAlgorithm):
    """DQN algorithm wrapper for RLlib."""
    
    def _get_algorithm_class(self):
        from ray.rllib.algorithms.dqn import DQN as RLlibDQN
        return RLlibDQN
    
    def _get_default_config(self):
        from ray.rllib.algorithms.dqn import DQNConfig
        return DQNConfig()


class A2C(BaseAlgorithm):
    """A2C algorithm wrapper for RLlib."""
    
    def _get_algorithm_class(self):
        from ray.rllib.algorithms.a2c import A2C as RLlibA2C
        return RLlibA2C
    
    def _get_default_config(self):
        from ray.rllib.algorithms.a2c import A2CConfig
        return A2CConfig()


# Policy aliases for compatibility
POLICY_ALIASES = {
    'CnnPolicy': 'CNN_POLICY',
    'MlpPolicy': 'MLP_POLICY', 
    'MultiInputPolicy': 'MULTI_INPUT_POLICY',
    'CnnLstmPolicy': 'CNN_LSTM_POLICY',
    'MlpLstmPolicy': 'MLP_LSTM_POLICY',
    'MultiInputLstmPolicy': 'MULTI_INPUT_LSTM_POLICY'
}


def make_vec_env(env_id, n_envs: int = 1, seed: Optional[int] = None, **kwargs):
    """Create vectorized environment (simplified compatibility function)."""
    # This is a simplified implementation
    # In practice, you might want to use RLlib's vectorization or other approaches
    if callable(env_id):
        return env_id()
    return env_id


def get_flattened_obs_dim(observation_space: gym.Space) -> int:
    """
    Get the dimension of a flattened observation space.
    Compatible replacement for SB3's get_flattened_obs_dim.
    """
    if isinstance(observation_space, gym.spaces.Box):
        return int(np.prod(observation_space.shape))
    elif isinstance(observation_space, gym.spaces.Discrete):
        return 1
    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
        return int(len(observation_space.nvec))
    elif isinstance(observation_space, gym.spaces.MultiBinary):
        return int(observation_space.n)
    elif isinstance(observation_space, gym.spaces.Dict):
        return sum(get_flattened_obs_dim(subspace) for subspace in observation_space.spaces.values())
    else:
        raise NotImplementedError(f"Unsupported observation space: {observation_space}")


def is_image_space(observation_space: gym.Space) -> bool:
    """
    Check if observation space is an image space.
    Compatible replacement for SB3's is_image_space.
    """
    if isinstance(observation_space, gym.spaces.Box):
        # Check if it's a 3D space (H, W, C) or (C, H, W)
        return len(observation_space.shape) == 3 and (
            observation_space.shape[0] in [1, 3, 4] or  # Channels first
            observation_space.shape[-1] in [1, 3, 4]     # Channels last
        )
    return False