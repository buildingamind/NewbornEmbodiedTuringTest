"""
Module for the Brain class.

policy: Any | str = "CnnPolicy",
algorithm: str | BaseAlgorithm = "PPO",
encoder: Any | str = "small",
embedding_dim: Optional[int] = None,
reward: str | type[BaseReward] = "supervised",
batch_size: int = 512,
buffer_size: int = 2048,
learning_rate: float = 3e-4,
ent_coef: float = 0,
train_eps: int = 5000,  # 1000
test_eps: int = 100,  # 20
steps_per_episode: int = 200,  # 1000
checkpoint_freq: Optional[int] = None,
train_encoder: bool = True,
custom_encoder_args: dict[str, Any] = {},
custom_policy_arch: Optional[list[int | dict[str, list[int]]]] = None,
reward_args: dict[str, Any] = {"beta": 0.2, "kappa": 0.0, "gamma": 0.99},

The brain is made up of an encoder, policy, algorithm, reward function, and the hyperparameters determined for these components such as the batch and buffer sizes. It produces a trained model based on the environment data and the inputs received by the brain through the body.

Args:
    policy (Any | str): The network used for defining the value and action networks. Defaults to "CnnPolicy". Must be either a string pointing to an already implemented policy or a custom policy class that inherits from Stable-Baselines3 BasePolicy. For a list of available policy strings, run `Brain.list_policies` :func:`~nett.nett.list_policies`.
    algorithm (str | BaseAlgorithm): The optimization algorithm used for training the model.
    encoder (Any | str, optional): The network used to extract features from the observations. Defaults to None.
    embedding_dim (int, optional): The dimension of the embedding space of the encoder. Defaults to None.
    reward (str, optional): The type of reward used for training the brain. Defaults to "supervised".
    batch_size (int, optional): The batch size used for training. Defaults to 512.
    buffer_size (int, optional): The buffer size used for training. Defaults to 2048.
    train_encoder (bool, optional): Whether to train the encoder or not. Defaults to True.
    seed (int, optional): The random seed used for training. Defaults to 12.
    custom_encoder_args (dict[str, str], optional): Custom arguments for the encoder. Defaults to {}.
    custom_policy_arch (Optional[list[int|dict[str,list[int]]]], optional): Custom architecture for the policy. Takes the form of a list of integers with each integer representing the number of neurons in that layer. The first member defines the number of neurons in the first hidden layer after the encoder and the last member defines the number of neurons in the final hidden layer before the output layer. If set to None, the policy arch will be the defaults defined in SB3, which is the equivalent of [] when the encoder is NatureCNN (small) and [64, 64] for any other encoder. Defaults to None.

Example:

    >>> from nett import Brain
    >>> brain = Brain(policy='CnnPolicy', algorithm='PPO')
"""

import logging
import os
import inspect
from math import ceil
import torch
import numpy as np

from tqdm import tqdm
from typing import Any, Optional
from pathlib import Path

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from rllte.common.prototype import BaseReward

from ..utils.task import Task

from .utils import callbacks as cb
from .utils.validate import (
    validate_algorithm,
    validate_encoder,
    validate_reward,
    validate_policy,
)

def _set_encoder_as_eval(model: BaseAlgorithm) -> BaseAlgorithm:
    """
    Set the encoder as evaluation mode and freeze its parameters.

    Args:
        model (BaseAlgorithm): The model containing the encoder.

    Returns:
        BaseAlgorithm: The model with the encoder set as evaluation mode.
    """
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

    Returns:
        None
    """
    ## save policy
    path.mkdir(parents=True, exist_ok=True)
    model.policy.save(path / "policy.pkl")

    ## save encoder
    encoder = model.policy.features_extractor.state_dict()
    torch.save(encoder, path / "feature_extractor.pth")

    print("Saved feature extractor")

    save_path = path / "model" / "latest_model.zip"
    model.save(save_path)


class Brain:
    """Represents the brain of an agent.

    The brain is made up of an encoder, policy, algorithm, reward function, and the hyperparameters determined for these components such as the batch and buffer sizes. It produces a trained model based on the environment data and the inputs received by the brain through the body.

    Args:
        policy (Any | str): The network used for defining the value and action networks.
        algorithm (str | BaseAlgorithm): The optimization algorithm used for training the model.
        encoder (Any | str, optional): The network used to extract features from the observations. Defaults to None.
        embedding_dim (int, optional): The dimension of the embedding space of the encoder. Defaults to None.
        reward (str, optional): The type of reward used for training the brain. Defaults to "supervised".
        batch_size (int, optional): The batch size used for training. Defaults to 512.
        buffer_size (int, optional): The buffer size used for training. Defaults to 2048.
        train_encoder (bool, optional): Whether to train the encoder or not. Defaults to True.
        seed (int, optional): The random seed used for training. Defaults to 12.
        custom_encoder_args (dict[str, str], optional): Custom arguments for the encoder. Defaults to {}.
        custom_policy_arch (Optional[list[int|dict[str,list[int]]]], optional): Custom architecture for the policy. Takes the form of a list of integers with each integer representing the number of neurons in that layer. The first member defines the number of neurons in the first hidden layer after the encoder and the last member defines the number of neurons in the final hidden layer before the output layer. If set to None, the policy arch will be the defaults defined in SB3, which is the equivalent of [] when the encoder is NatureCNN (small) and [64, 64] for any other encoder. Defaults to None.

    Example:

        >>> from nett import Brain
        >>> brain = Brain(policy='CnnPolicy', algorithm='PPO')
    """

    def __init__(
        self,
        policy: Any | str = "CnnPolicy",
        algorithm: str | BaseAlgorithm = "PPO",
        encoder: Any | str = "small",
        embedding_dim: Optional[int] = None,
        reward: str | type[BaseReward] = "supervised",
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
        self.algorithm = validate_algorithm(algorithm)
        self.policy = validate_policy(policy)
        self.encoder = validate_encoder(encoder)

        self.supervised: bool = reward == "supervised"
        self.reward: Optional[type[BaseReward]] = validate_reward(reward)

        self.embedding_dim = int(embedding_dim) if embedding_dim is not None else None
        self.batch_size = int(batch_size)
        self.buffer_size = int(buffer_size)
        self.learning_rate = float(learning_rate)
        self.ent_coef = float(ent_coef)

        self.checkpoint_freq = int(checkpoint_freq) if checkpoint_freq is not None else None
        self.train_encoder = bool(train_encoder)

        self.custom_encoder_args = custom_encoder_args
        self.custom_policy_arch = custom_policy_arch
        self.reward_args = reward_args

    def calc_iterations(
        self,
        num_brains: int,
        num_threads: int, # test, test_iterations
        num_imprinting_conditions: int,
        num_test_conditions: int, # test_iterations
        episodes: dict[str,int], # train_iterations
        steps_per_episode: int, # train_iterations
    ):
        self.n_tasks = num_imprinting_conditions * num_brains
        if "train" in episodes:
            self.train_iterations = episodes['train'] * steps_per_episode
        if "test" in episodes:
            # calculate number of environments that can be run at once per job (using SubProcVecEnv)
            n_threads_per_task = 4 # TODO: Determine the number of threads used per brain and per env

            max_envs = num_threads / (n_threads_per_task * self.n_tasks)

            if max_envs <= 1:
                self.n_parallel_envs = 1
                self.test_iterations = num_test_conditions * episodes["test"]
            elif max_envs >= episodes["test"]:
                self.n_parallel_envs = episodes["test"]
                self.test_iterations = num_test_conditions
            else:  # max_envs is between 1 and test_eps
                self.n_parallel_envs = int(max_envs)
                self.test_iterations = num_test_conditions * ceil(episodes["test"] / max_envs)

    def train(self, envs: VecEnv, task: Task):
        """
        Train the brain.

        Args:
            job(Job): The job object containing the environment, paths, and training parameters.

        Raises:
            ValueError: If the environment fails the validation check.
        """
        task.logger.info(
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
                device=f"cuda:{task.device}",
                seed=task.seed,  # env.seed() function is expected in sb3 but does not exist in the ss.SB3VecEnvWrapper
                tensorboard_log=task.path / "logs",
            )

            # set encoder as eval only if train_encoder is not True
            if not self.train_encoder:
                model = _set_encoder_as_eval(model)
                task.logger.warning(
                    f"Encoder training is set to {str(self.train_encoder).upper()}"
                )

        except Exception as e:
            task.logger.exception(f"Failed to initialize model with error: {str(e)}")
            raise e

        # initialize callbacks
        task.logger.info("Initializing Callbacks")
        callback_list = self._init_callbacks(envs, task)

        # train
        task.logger.info(f"Total number of training steps: {self.train_iterations}")
        try:
            model.learn(
                total_timesteps=self.train_iterations,
                tb_log_name=self.algorithm.__name__,
                progress_bar=False,
                callback=callback_list,
                # tb_log_name="train",
            )
        except Exception as e:
            if "CUDA out of memory" in str(e):
                task.logger.error("CUDA out of memory. Try reducing batch size.")
                raise
            else:
                task.logger.exception(f"Failed to train model with error: {str(e)}")
                raise e
        task.logger.info("Training Complete")

        # nothing else is needed for memory estimation
        if task.estimate_memory:
            return

        # save
        ## create save directory
        model_path = task.path / "model"
        _save_model(model, model_path)
        task.logger.info(f"Saved model at {model_path}")

    def test(self, envs: VecEnv, task: Task, base_env, base_body):
        """
        Test the brain.

        Args:
            env (gym.Env): The environment used for testing.
            job (Job): The job object containing the environment, paths, and training parameters.
        """
        try:
            # load previously trained model from save_dir, if it exists
            model: BaseAlgorithm = self.algorithm.load(
                task.path / "model" / "latest_model.zip", device=f"cuda:{task.device}"
            )

            task.logger.info(f"Testing with {self.algorithm.__name__}")

            vec_env = ZooEnv if base_env.muliagent else MultiEnv

            with envs as vec_env(task, base_env, base_body, self.n_parallel_envs):
                # progress bar
                t = tqdm(
                    total=self.test_iterations * self.n_parallel_envs,
                    desc=f"Test Progress",
                    position=0,
                    leave=True,
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
                        t.update(1)

                        if not all(dones):
                            # episode is done
                            break
        except Exception as e:
            task.logger.exception(f"Failed to test model with error: {str(e)}")
            raise e

        t.close()

    def _init_callbacks(self, envs, task: Task) -> CallbackList:
        """Initialize the callbacks for training.
        Args:
            envs (VecEnv): The environments for which to initialize the callbacks.
            task (Task): The task for which to initialize the callbacks.
        Returns:
            CallbackList: The list of callbacks for training.
        """
        callback_list = [cb.HParamCallback()]

        if task.estimate_memory:
            callback_list.extend(
                [
                    cb.LoadingBarCallback("Estimating Memory Usage", self.buffer_size),
                    cb.MemoryCallback(task.device, save_path=task.path),
                ]
            )
        else:
            # creates the parallel progress bars
            callback_list.append(
                cb.LoadingBarCallback(
                    f"Training: ", self.n_tasks * self.train_iterations
                )
            )

            if self.checkpoint_freq is not None:
                callback_list.append(
                    CheckpointCallback(
                        save_freq=self.checkpoint_freq,  # defaults to 30_000 steps
                        save_path=task.path / "checkpoints",
                        save_replay_buffer=True,
                        save_vecnormalize=True,
                    )
                )

        # create reward function
        if self.reward is not None:
            reward_func: BaseReward = self.reward(
                envs,
                device=f"cuda:{task.device}",
                batch_size=self.batch_size,
                lr=self.learning_rate,
                **self.reward_args,
            )
            if self.algorithm.issubclass(OnPolicyAlgorithm):
                # brain.algorithm is instance of OnPolicyAlgorithn
                callback_list.append(cb.IntrinsicRewardWithOnPolicyRL(reward_func))
            elif self.algorithm.issubclass(OffPolicyAlgorithm):
                callback_list.append(cb.IntrinsicRewardWithOffPolicyRL(reward_func))
            else:
                task.logger.warning(
                    f"Instrinsic rewards do not support selected algorithm {self.algorithm}"
                )

        return CallbackList(callback_list)
