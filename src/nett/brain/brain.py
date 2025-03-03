"""
Represents the brain of an agent.

The brain is made up of an encoder, policy, algorithm, reward function, and the hyperparameters determined for these components such as the batch and buffer sizes. It produces a trained model based on the environment data and the inputs received by the brain through the body.
"""

import inspect
from math import ceil
import torch
import numpy as np

from typing import Any, Optional
from pathlib import Path

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rllte.common.prototype import BaseReward

from ..utils.task import TaskConfig

from .utils import callbacks as cb
from .utils.validate import (
    validate_algorithm,
    validate_encoder,
    validate_reward,
    validate_policy,
)


def _set_encoder_as_eval(model: BaseAlgorithm) -> BaseAlgorithm:
    """Set the encoder as evaluation mode and freeze its parameters."""
    model.policy.features_extractor.eval()

    for param in model.policy.features_extractor.parameters():
        param.requires_grad = False
    return model


def _save_model(model: BaseAlgorithm, path: Path) -> None:
    """
    Saves the policy and feature extractor of the agent's model.

    This method saves the policy and feature extractor of the agent's model
    to the specified paths. It first checks if the model is loaded, and if not,
    it prints an error message and returns. Otherwise, it saves the policy as
    a pickle file and the feature extractor as a PyTorch state dictionary.
    """
    ## save policy
    path.mkdir(parents=True, exist_ok=True)
    model.policy.save(path / "policy.pkl")

    ## save encoder
    encoder = model.policy.features_extractor.state_dict()
    torch.save(encoder, path / "feature_extractor.pth")

    print("Saved feature extractor")

    save_path = path / "latest_model.zip"
    model.save(save_path)


class Brain:
    """
    Args:
        encoder (Any | str, optional): The network used to extract features from the observations. Must be either a string pointing to an already implemented encoder or a custom encoder class that inherits from Stable-Baselines3 BaseFeaturesExtractor. For a list of available extractors, run `nett.list_encoders` :func:`~nett.nett.list_encoders`. Defaults to "small".
        policy (str | BasePolicy): The network used for defining the value and action networks. Must be either a string pointing to an already implemented policy or a custom policy class that inherits from Stable-Baselines3 BasePolicy. For a list of available policy strings, run `nett.list_policies` :func:`~nett.nett.list_policies`. Defaults to "CnnPolicy".
        algorithm (str | BaseAlgorithm): The optimization algorithm used for training the model. Must be either a string pointing to an already implemented algorithm or a custom algorithm class that inherits from Stable-Baselines3 BaseAlgorithm. For a list of available algorithms, run `nett.list_algorithms` :func:`~nett.nett.list_algorithms`. Defaults to "PPO".
        reward (str): The type of reward used for training the brain. For a list of available reward strings, run `nett.list_rewards` :func:`~nett.nett.list_rewards`. Defaults to "supervised".
        embedding_dim (int, optional): The dimension of the embedding space of the encoder. If None, default embedding dim defined by encoder is used. Defaults to None.
        batch_size (int): The batch size used for training. Defaults to 512.
        buffer_size (int): The buffer size used for training. Defaults to 2048.
        ent_coef (int): Entropy coefficient. Defaults to 0.
        checkpoint_freq (int, optional): Number of steps to save checkpoints of the model. If None, no checkpoints are saved. Defaults to None.
        train_encoder (bool, optional): Whether to train the encoder or not. Defaults to True.
        custom_encoder_args (dict[str, str], optional): Custom arguments for the encoder. Defaults to {}.
        custom_policy_arch (list[int|dict[str,list[int]]], optional): Custom architecture for the policy. Takes the form of a list of integers with each integer representing the number of neurons in that layer. The first member defines the number of neurons in the first hidden layer after the encoder and the last member defines the number of neurons in the final hidden layer before the output layer. If set to None, the policy arch will be the defaults defined in SB3, which is the equivalent of [] when the encoder is NatureCNN (small) and [64, 64] for any other encoder. Defaults to None.
        reward_args (dict[str, Any]): Arguments for the encoder. Defaults to {"beta": 0.2, "kappa": 0.0, "gamma": 0.99}.
    """

    def __init__(
        self,
        encoder: str | type[BaseFeaturesExtractor] = "small",
        policy: str | type[BasePolicy] = "CnnPolicy",
        algorithm: str | type[BaseAlgorithm] = "PPO",
        reward: str | type[BaseReward] = "supervised",
        embedding_dim: Optional[int] = None,
        batch_size: int = 512,
        buffer_size: int = 2048,
        learning_rate: float = 3e-4,
        ent_coef: float = 0,
        checkpoint_freq: Optional[int] = None,
        train_encoder: bool = True,
        custom_encoder_args: dict[str, Any] = {},
        custom_policy_arch: Optional[list[int | dict[str, list[int]]]] = None,
        reward_args: dict[str, Any] = {"beta": 0.2, "kappa": 0.0, "gamma": 0.99},
    ):
        # Set attributes
        self.encoder = validate_encoder(encoder)
        self.policy = validate_policy(policy)
        self.algorithm = validate_algorithm(algorithm)

        self.supervised: bool = reward == "supervised"
        self.reward: Optional[type[BaseReward]] = validate_reward(reward)

        self.embedding_dim = int(embedding_dim) if embedding_dim is not None else None
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.learning_rate = float(learning_rate)
        self.ent_coef = float(ent_coef)

        self.checkpoint_freq = (
            int(checkpoint_freq) if checkpoint_freq is not None else None
        )
        self.train_encoder = bool(train_encoder)

        self.custom_encoder_args = custom_encoder_args
        self.custom_policy_arch = custom_policy_arch
        self.reward_args = reward_args

    def calc_iterations(
        self,
        num_brains: int,
        num_threads: int,
        num_imprinting_conditions: int,
        num_test_conditions: int,
        episodes: dict[str, int],
        steps_per_episode: int,
    ):
        """Calculate the total number of iterations for training and testing."""
        # Calculate the total number of tasks to be run
        self.n_tasks = num_imprinting_conditions * num_brains
        if "train" in episodes:
            self.train_iterations = episodes["train"] * steps_per_episode
        if "test" in episodes:
            # calculate number of environments that can be run at once per job (using SubProcVecEnv)
            # TODO: Determine the number of threads used per brain and per env
            n_threads_per_task = 4

            max_envs = num_threads / (n_threads_per_task * self.n_tasks)

            if max_envs <= 1:
                self.n_parallel_envs = 1
                self.test_iterations = num_test_conditions * episodes["test"]
            elif max_envs >= episodes["test"]:
                self.n_parallel_envs = episodes["test"]
                self.test_iterations = num_test_conditions
            else:  # max_envs is between 1 and test_eps
                self.n_parallel_envs = int(max_envs)
                self.test_iterations = num_test_conditions * ceil(
                    episodes["test"] / max_envs
                )

    def train(self, envs: VecEnv, config: TaskConfig):
        """Train the brain."""
        config.logger.info(
            f"Training {self.encoder.__name__} with {self.algorithm.__name__}"
        )

        # build model
        policy_kwargs = (
            {
                "features_extractor_class": self.encoder,
                "features_extractor_kwargs": {
                    "features_dim": self.embedding_dim
                    or inspect.signature(self.encoder)
                    .parameters["features_dim"]
                    .default,
                    **self.custom_encoder_args,
                },
            }
            if self.encoder is not None
            else {}
        )

        if self.custom_policy_arch:
            policy_kwargs["net_arch"] = self.custom_policy_arch

        try:
            model = self.algorithm(
                self.policy,
                envs,
                batch_size=self.batch_size,
                n_steps=self.buffer_size,  # TODO: Will need to be adjusted if running parallel envs
                learning_rate=self.learning_rate,
                ent_coef=self.ent_coef,
                verbose=1,  # 0,  # TODO: Incorporate this into options
                policy_kwargs=policy_kwargs,
                device=f"cuda:{config.device}",
                seed=config.brain_id,  # env.seed() function is expected in sb3 but does not exist in the ss.SB3VecEnvWrapper
                tensorboard_log=config.path / "logs",
            )

            # set encoder as eval only if train_encoder is not True
            if not self.train_encoder:
                model = _set_encoder_as_eval(model)
                config.logger.warning(
                    f"Encoder training is set to {str(self.train_encoder).upper()}"
                )

        except Exception as e:
            config.logger.exception(f"Failed to initialize model with error: {str(e)}")
            raise e

        # initialize callbacks
        config.logger.info("Initializing Callbacks")
        callback_list = self._init_callbacks(envs, config)

        # train
        total_timesteps = (
            self.buffer_size if config.memory is None else self.train_iterations
        )
        config.logger.info(f"Total number of training steps: {total_timesteps}")
        try:
            model.learn(
                total_timesteps=total_timesteps,
                tb_log_name=self.algorithm.__name__,
                progress_bar=False,
                callback=callback_list,
                log_interval=None,
                # tb_log_name="train",
            )
        except Exception as e:
            if "CUDA out of memory" in str(e):
                config.logger.error("CUDA out of memory. Try reducing batch size.")
                raise
            else:
                config.logger.exception(f"Failed to train model with error: {str(e)}")
                raise e
        config.logger.info("Training Complete")

        # nothing else is needed for memory estimation
        if config.memory is not None:
            # save
            ## create save directory
            config.logger.info(f"Saving model...")
            model_path = config.path / "model"
            _save_model(model, model_path)
            config.logger.info(f"Saved model at {model_path}")

    def test(self, envs: VecEnv, config: TaskConfig):
        """Test the brain."""
        try:
            config.logger.info(f"Testing with {self.algorithm.__name__}")

            # load previously trained model from save_dir, if it exists
            model: BaseAlgorithm = self.algorithm.load(
                config.path / "model" / "latest_model.zip",
                device=f"cuda:{config.device}",
            )

            # loop over episodes
            for _ in range(self.test_iterations):
                # reset environment and get initial obs
                obs = envs.reset()
                # reset states for recurrentPPO
                states = None
                # dones need to start True for episode_start for recurrentPPO
                dones = np.ones((self.n_parallel_envs,), dtype=bool)

                while True:
                    # predict an action
                    action, states = model.predict(
                        obs,
                        state=states,  # used only for recurrentPPO
                        episode_start=dones,  # used only for recurrentPPO
                        deterministic=True,
                    )
                    # perform the action
                    obs, _, dones, _ = envs.step(action)  # obs, rewards, done, info
                    # update the loading bar
                    config.queue.put((config.name, 1))

                    if not all(dones):
                        # episode is done
                        break
        except Exception as e:
            config.logger.exception(f"Failed to test model with error: {str(e)}")
            raise e

    def _init_callbacks(self, envs: VecEnv, config: TaskConfig) -> CallbackList:
        """Initialize the callbacks for training."""

        if config.memory is None:
            callback_list = [
                cb.LoadingBarCallback(
                    f"Estimating Memory Usage for {config.name}", config.queue
                ),  # , self.buffer_size),
                cb.MemoryCallback(config.device, save_path=config.path),
            ]
        else:
            # creates the parallel progress bars
            callback_list = [
                cb.HParamCallback(),
                cb.LoadingBarCallback(config.name, config.queue),
            ]

            if self.checkpoint_freq is not None:
                callback_list.append(
                    CheckpointCallback(
                        save_freq=self.checkpoint_freq,  # defaults to 30_000 steps
                        save_path=config.path / "checkpoints",
                        save_replay_buffer=True,
                        save_vecnormalize=True,
                    )
                )

        # create reward function
        if self.reward is not None:
            reward_func: BaseReward = self.reward(
                envs,
                device=f"cuda:{config.device}",
                batch_size=self.batch_size,
                lr=self.learning_rate,
                **self.reward_args,
            )
            if issubclass(self.algorithm, OnPolicyAlgorithm):
                # brain.algorithm is instance of OnPolicyAlgorithn
                callback_list.append(cb.IntrinsicRewardWithOnPolicyRL(reward_func))
            elif issubclass(self.algorithm, OffPolicyAlgorithm):
                callback_list.append(cb.IntrinsicRewardWithOffPolicyRL(reward_func))
            else:
                config.logger.warning(
                    f"Instrinsic rewards do not support selected algorithm {self.algorithm}"
                )

        return CallbackList(callback_list)
