"""Module for the Brain class."""

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
from .params import (
    validate_algorithm,
    validate_encoder,
    validate_reward,
    validate_policy,
)


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

    @classmethod
    def initialize(
        cls,
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
    ):

        # Initialize logger
        cls.logger = logging.getLogger("nett.Brain")

        # Set attributes
        cls.algorithm = validate_algorithm(algorithm)
        cls.policy = validate_policy(policy)
        cls.train_encoder = train_encoder
        cls.encoder = validate_encoder(encoder)
        cls.supervised: bool = reward == "supervised"
        cls.reward: Optional[type[BaseReward]] = validate_reward(reward)

        cls.embedding_dim = embedding_dim
        cls.batch_size = batch_size
        cls.buffer_size = buffer_size
        cls.learning_rate = learning_rate
        cls.ent_coef = ent_coef

        cls.checkpoint_freq = checkpoint_freq

        cls.custom_encoder_args = custom_encoder_args
        cls.custom_policy_arch = custom_policy_arch
        cls.reward_args = reward_args

        # train
        # train_eps
        cls.train_iterations = train_eps * steps_per_episode
        # test
        # test_eps, n_tasks (brains*conditions)
        cls.test_eps = test_eps

    @classmethod
    def calc_run_info(
        cls,
        num_brains: int,
        num_test_conditions: int,
        num_imprinting_conditions: int,
    ):

        cls.n_tasks = num_imprinting_conditions * num_brains

        # calculate number of environments that can be run at once per job (using SubProcVecEnv)
        # TODO: Determine the number of threads used per brain and per env
        n_threads_per_task = 4

        max_envs = os.cpu_count() / (n_threads_per_task * cls.n_tasks)
        if max_envs <= 1:
            cls.n_parallel_envs = 1
            cls.test_iterations = num_test_conditions * cls.test_eps
        elif max_envs >= cls.test_eps:
            cls.n_parallel_envs = cls.test_eps
            cls.test_iterations = num_test_conditions
        else:  # max_envs is between 1 and test_eps
            cls.n_parallel_envs = int(max_envs)
            cls.test_iterations = num_test_conditions * ceil(cls.test_eps / max_envs)

    def __init__(self, device: int, seed: int) -> None:
        """Constructor method"""
        self.device = f"cuda:{device}"
        self.seed = seed

    def train(self, envs: VecEnv, task: Task):
        """
        Train the brain.

        Args:
            job(Job): The job object containing the environment, paths, and training parameters.

        Raises:
            ValueError: If the environment fails the validation check.
        """
        self.logger.info(
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
                device=self.device,
                seed=self.seed,  # env.seed() function is expected in sb3 but does not exist in the ss.SB3VecEnvWrapper
                tensorboard_log=task.path / "logs",
            )

            # set encoder as eval only if train_encoder is not True
            if not self.train_encoder:
                model = self._set_encoder_as_eval(model)
                self.logger.warning(
                    f"Encoder training is set to {str(self.train_encoder).upper()}"
                )

        except Exception as e:
            self.logger.exception(f"Failed to initialize model with error: {str(e)}")
            raise e

        # initialize callbacks
        self.logger.info("Initializing Callbacks")
        callback_list = self.initialize_callbacks(envs, task)

        # train
        self.logger.info(f"Total number of training steps: {self.train_iterations}")
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
                self.logger.error("CUDA out of memory. Try reducing batch size.")
                raise
            else:
                self.logger.exception(f"Failed to train model with error: {str(e)}")
                raise e
        self.logger.info("Training Complete")

        # nothing else is needed for memory estimation
        if task.estimate_memory:
            return

        # save
        ## create save directory
        model_path = task.path / "model"
        self.save(model, model_path)
        self.logger.info(f"Saved model at {model_path}")

    def test(self, envs: VecEnv, task: Task):
        """
        Test the brain.

        Args:
            env (gym.Env): The environment used for testing.
            job (Job): The job object containing the environment, paths, and training parameters.
        """
        try:
            # load previously trained model from save_dir, if it exists
            model: BaseAlgorithm = self.algorithm.load(
                task.path / "model" / "latest_model.zip", device=self.device
            )

            self.logger.info(f"Testing with {self.algorithm.__name__}")

            num_envs = envs.num_envs

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
                dones = np.ones((num_envs,), dtype=bool)

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
            self.logger.exception(f"Failed to test model with error: {str(e)}")
            raise e

        t.close()

    @staticmethod
    def save(model: BaseAlgorithm, path: Path) -> None:
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

    @staticmethod
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

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"

    def __str__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"

    def initialize_callbacks(self, envs, task: Task) -> CallbackList:
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
                device=self.device,
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
                self.logger.warning(
                    f"Instrinsic rewards do not support selected algorithm {self.algorithm}"
                )

        return CallbackList(callback_list)
