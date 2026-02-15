"""
Represents the brain of an agent.

The brain is made up of an encoder, policy, algorithm, reward function, and the hyperparameters determined for these components such as the batch and buffer sizes. It produces a trained model based on the environment data and the inputs received by the brain through the body.
"""

import inspect
from math import ceil
import torch
import numpy as np

from typing import Any, Optional, Callable
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
    # Set the encoder as evaluation mode and freeze its parameters.

    # Set the feature extractor to evaluation mode
    model.policy.features_extractor.eval()

    # Freeze the parameters of the feature extractor
    for param in model.policy.features_extractor.parameters():
        param.requires_grad = False
    return model


def _save_model(model: BaseAlgorithm, path: Path) -> None:
    # Saves the policy and feature extractor of the agent's model.

    # This method saves the policy and feature extractor of the agent's model
    # to the specified paths. It first checks if the model is loaded, and if not,
    # it prints an error message and returns. Otherwise, it saves the policy as
    # a pickle file and the feature extractor as a PyTorch state dictionary.

    # Create the directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)
    # Save the policy
    # model.policy.save(path / "policy.pkl")

    # Save the feature extractor's state dictionary
    # encoder = model.policy.features_extractor.state_dict()
    # torch.save(encoder, path / "feature_extractor.pth")

    # print("Saved feature extractor")

    # Save the full model
    save_path = path / "latest_model.zip"
    model.save(save_path)


class Brain:
    """
    Args:
        encoder (Any | str, optional): The network used to extract features from the observations. Must be either a string pointing to an already implemented encoder or a custom encoder class that inherits from Stable-Baselines3 BaseFeaturesExtractor. For a list of available extractors, run `nett.list_encoders` :func:`~nett.nett.list_encoders`. Defaults to "small".
        policy (str | BasePolicy): The network used for defining the value and action networks. Must be either a string pointing to an already implemented policy or a custom policy class that inherits from Stable-Baselines3 BasePolicy. For a list of available policy strings, run `nett.list_policies` :func:`~nett.nett.list_policies`. Defaults to "CnnPolicy".
        algorithm (str | BaseAlgorithm): The optimization algorithm used for training the model. Must be either a string pointing to an already implemented algorithm or a custom algorithm class that inherits from Stable-Baselines3 BaseAlgorithm. For a list of available algorithms, run `nett.list_algorithms` :func:`~nett.nett.list_algorithms`. Defaults to "PPO".
        reward (str): The type of reward used for training the brain. For a list of available reward strings, run `nett.list_rewards` :func:`~nett.nett.list_rewards`. Defaults to "closeness".
        embedding_dim (int, optional): The dimension of the embedding space of the encoder. If None, default embedding dim defined by encoder is used. Defaults to None.
        batch_size (int): The batch size used for training. Defaults to 512.
        buffer_size (int): The buffer size used for training. Defaults to 2048.
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
        reward: str | type[BaseReward] = "closeness",
        embedding_dim: Optional[int] = None,
        batch_size: int = 512,
        buffer_size: int = 2048,
        learning_rate: float | Callable = 3e-4,
        checkpoint_freq: Optional[int] = None,
        train_encoder: bool = True,
        deterministic: bool = True,
        custom_encoder_args: dict[str, Any] = {},
        custom_algorithm_args: dict[str, Any] = {},
        custom_policy_args: Optional[dict[str, Any]] = {},
        custom_policy_arch: Optional[list[int | dict[str, list[int]]]] = None,
        reward_args: dict[str, Any] = {"beta": 0.2, "kappa": 0.0, "gamma": 0.99},
        continual_learning: bool = False,
    ):
        # --- Validate and set attributes ---
        self.encoder = validate_encoder(encoder)
        self.policy = validate_policy(policy)
        self.algorithm = validate_algorithm(algorithm)
        self.reward: Optional[type[BaseReward]] = validate_reward(reward)

        # --- Hyperparameters ---
        self.embedding_dim = int(embedding_dim) if embedding_dim is not None else None
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.learning_rate = float(learning_rate)

        # --- Training configuration ---
        self.checkpoint_freq = (
            int(checkpoint_freq) if checkpoint_freq is not None else None
        )
        self.train_encoder = bool(train_encoder)
        self.deterministic = bool(deterministic)

        # --- Continual Learning Test Mode ---
        self.continual_learning = bool(continual_learning)

        # --- Custom arguments ---
        # used for extractors that wrap other extractors e.g. multiinput
        if "extractor_class" in custom_encoder_args:
            custom_encoder_args["extractor_class"] = validate_encoder(
                custom_encoder_args["extractor_class"]
            )

        self.custom_encoder_args = custom_encoder_args
        self.custom_algorithm_args = custom_algorithm_args
        self.custom_policy_arch = custom_policy_arch
        self.custom_policy_args = custom_policy_args
        self.reward_args = reward_args

        # --- Reward arguments ---
        if reward != "RE3":
            self.reward_args["batch_size"] = self.batch_size
            self.reward_args["lr"] = self.learning_rate

    def calc_iterations(
        self,
        num_brains: int,
        num_threads: int,
        iterations_per_episode: dict[str, int],
        episodes: dict[str, int],
        steps_per_episode: int,
    ):
        """Calculate the total number of iterations for training and testing."""
        self.steps_per_episode = steps_per_episode
        # Calculate the total number of tasks to be run
        self.n_tasks = len(iterations_per_episode) * num_brains

        # Calculate total training iterations if in 'train' mode
        if "train" in episodes:
            self.train_iterations = episodes["train"] * steps_per_episode

        # Calculate testing iterations if in 'test' mode
        if "test" in episodes:
            # calculate number of environments that can be run at once per job (using SubProcVecEnv)
            # TODO: Determine the number of threads used per brain and per env
            n_threads_per_task = 4

            max_envs = num_threads / (n_threads_per_task * self.n_tasks)

            self.n_parallel_envs = 1
            self.test_iterations = {
                k: v * episodes["test"] for k, v in iterations_per_episode.items()
            }

    def train(self, envs: VecEnv, config: TaskConfig):
        """Train the brain."""
        # --- Build model ---
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

        policy_kwargs.update(self.custom_policy_args)

        try:
            model = self.algorithm(
                self.policy,
                envs,
                batch_size=self.batch_size,
                n_steps=self.buffer_size,  # TODO: Will need to be adjusted if running parallel envs
                learning_rate=self.learning_rate,
                verbose=1,  # 0,  # TODO: Incorporate this into options
                policy_kwargs=policy_kwargs,
                device=f"cuda:{config.device}",
                seed=(config.brain_id * 7919) % (2**31 - 1),  # Diversified seed: avoids systematic failures from sequential brain_id seeds (e.g. seeds 4,5 cause middle-dwelling). Original: seed=config.brain_id
                tensorboard_log=config.path / "tensorboard",
                **self.custom_algorithm_args,
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

        # --- Initialize callbacks ---
        callback_list = self._init_callbacks(envs, config)

        # --- Train model ---
        total_timesteps = (
            self.buffer_size if config.memory is None else self.train_iterations
        )
        try:
            model.learn(
                total_timesteps=total_timesteps,
                tb_log_name=self.algorithm.__name__,
                progress_bar=False,
                callback=callback_list,
                # reset_num_timesteps=True,
                # log_interval=None, #TODO: Reocrd to tb and not stdout
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

        # --- Save model if not estimating memory ---
        if config.memory is not None:
            # save
            ## create save directory
            config.logger.info(f"Saving model...")
            model_path = config.path / "model"
            _save_model(model, model_path)
            config.logger.info(f"Saved model at {model_path}")

        del model  # free memory

    def test(self, envs: VecEnv, config: TaskConfig):
        """Test the brain."""
        # --- Continual Learning Test Mode ---
        if self.continual_learning:
            try:
                # Load the trained model with test envs attached so SB3 rebinds
                model: BaseAlgorithm = self.algorithm.load(
                    config.path / "model" / "latest_model.zip",
                    env=envs,
                    device=f"cuda:{config.device}",
                )

                # Initialize the full callback list (intrinsic reward, TB, video, etc.)
                callback_list = self._init_callbacks(envs, config)

                # Continue learning in the test environment
                total_timesteps = (
                    self.test_iterations[config.condition] * self.steps_per_episode
                )
                model.learn(
                    total_timesteps=total_timesteps,
                    tb_log_name=f"{self.algorithm.__name__}_test",
                    progress_bar=False,
                    callback=callback_list,
                    reset_num_timesteps=False,  # continue TB step counter from training
                )
                config.logger.info("Continual-learning test phase complete")

                # Save updated model to a separate path (preserve original trained model)
                test_model_path = config.path / "model_test" / config.condition
                _save_model(model, test_model_path)
                config.logger.info(
                    f"Saved continual-learning test model at {test_model_path}"
                )

                del model  # free memory
            except Exception as e:
                config.logger.exception(
                    f"Failed continual-learning test with error: {str(e)}"
                )
                raise e
        else:
            # --- Standard (inference-only) Test Mode ---
            try:
                # load previously trained model from save_dir, if it exists
                model: BaseAlgorithm = self.algorithm.load(
                    config.path / "model" / "latest_model.zip",
                    device=f"cuda:{config.device}",
                )

                # reset environment and get initial obs
                obs = envs.reset()
                # reset states for recurrent policies
                states = None
                # dones need to start True for episode_start for recurrent policies
                dones = np.ones((self.n_parallel_envs,), dtype=bool)

                # loop over episodes
                for _ in range(self.test_iterations[config.condition]):
                    while True:
                        # predict an action
                        action, states = model.predict(
                            obs,
                            state=states,  # used only for recurrent policies
                            episode_start=dones,  # used only for recurrent policies
                            deterministic=self.deterministic,
                        )
                        # perform the action
                        obs, _, dones, _ = envs.step(action)  # obs, rewards, done, info
                        # update the loading bar
                        config.queue.put((config.name, 1))

                        if all(dones):
                            # episode is done
                            break

                    # Convert recorded frames to video
                    cb.img2video(
                        config.path / "recordings" / "chamber" / config.current_mode,
                        self.steps_per_episode,
                    )

                del model  # free memory
            except Exception as e:
                config.logger.exception(f"Failed to test model with error: {str(e)}")
                raise e

    def _init_callbacks(self, envs: VecEnv, config: TaskConfig) -> CallbackList:
        # Initialize the callbacks for training.

        # Callbacks for memory estimation mode
        if config.memory is None:
            callback_list = [
                cb.LoadingBarCallback(
                    f"Estimating Memory Usage for {config.name}", config.queue
                ),  # , self.buffer_size),
                cb.MemoryCallback(config.device, save_path=config.path),
            ]
        else:
            # Callbacks for regular training
            # creates the parallel progress bars
            callback_list = [
                cb.HParamCallback(),
                cb.LoadingBarCallback(config.name, config.queue),
            ]

            # Add checkpoint callback if specified
            if self.checkpoint_freq is not None:
                callback_list.append(
                    CheckpointCallback(
                        save_freq=self.checkpoint_freq,  # defaults to 30_000 steps
                        save_path=config.path / "checkpoints",
                    )
                )

        # create and add intrinsic reward callback
        if self.reward is not None:
            reward_func: BaseReward = self.reward(
                envs,
                device=f"cuda:{config.device}",
                **self.reward_args,
            )
            if issubclass(self.algorithm, OnPolicyAlgorithm):
                # brain.algorithm is instance of OnPolicyAlgorithn
                callback_list.append(cb.IntrinsicRewardWithOnPolicyRL(reward_func))
            elif issubclass(self.algorithm, OffPolicyAlgorithm):
                callback_list.append(cb.IntrinsicRewardWithOffPolicyRL(reward_func))
            else:
                config.logger.warning(
                    f"Intrinsic rewards do not support selected algorithm {self.algorithm}"
                )

        # Add video recording callback
        callback_list.append(
            cb.PngToMp4Callback(
                config.path / "recordings" / "chamber" / config.current_mode,
                self.steps_per_episode,
            )
        )

        return CallbackList(callback_list)
