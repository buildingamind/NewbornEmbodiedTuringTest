"""Module for the Brain class."""

from typing import Any, Optional
from pathlib import Path
import inspect

import torch
import stable_baselines3
import sb3_contrib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common import results_plotter

from nett.brain import algorithms, policies, encoder_dict
from nett.brain import encoders
from nett.utils.callbacks import SupervisedSaveBestModelCallback, HParamCallback, multiBarCallback

# TODO (v0.3): Extend with support for custom policy models
# TODO (v0.3): should we move validation checks to utils under validations.py?

class Brain:
    """Represents the brain of an agent. 

    The brain is made up of an encoder, policy, algorithm, reward function, and the hyperparameters determined for these components such as the batch and buffer sizes. It produces a trained model based on the environment data and the inputs received by the brain through the body.

    Args:
        policy (Any | str): The network used for defining the value and action networks.
        algorithm (str | OnPolicyAlgorithm | OffPolicyAlgorithm): The optimization algorithm used for training the model.
        encoder (Any | str, optional): The network used to extract features from the observations. Defaults to None.
        embedding_dim (int, optional): The dimension of the embedding space of the encoder. Defaults to None.
        reward (str, optional): The type of reward used for training the brain. Defaults to "supervised".
        batch_size (int, optional): The batch size used for training. Defaults to 512.
        buffer_size (int, optional): The buffer size used for training. Defaults to 2048.
        train_encoder (bool, optional): Whether to train the encoder or not. Defaults to False.
        seed (int, optional): The random seed used for training. Defaults to 12.

    Example:

        >>> from nett import Brain
        >>> brain = Brain(policy='CnnPolicy', algorithm='PPO')
    """

    def __init__(
        self,
        policy: Any | str,
        algorithm:  str | OnPolicyAlgorithm | OffPolicyAlgorithm,
        encoder: Any | str = None,
        embedding_dim: Optional[int] = None,
        reward: str = "supervised",
        batch_size: int = 512,
        buffer_size: int = 2048,
        train_encoder: bool = False,
        seed: int = 12
    ) -> None:
        """Constructor method
        """
        # Initialize logger
        from nett import logger

        self.logger = logger.getChild(__class__.__name__)

        # Set attributes
        self.algorithm = self._validate_algorithm(algorithm)
        self.policy = self._validate_policy(policy)
        self.train_encoder = train_encoder
        self.encoder = self._validate_encoder(encoder) if encoder else None
        self.reward = self._validate_reward(reward) if reward else None
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seed = seed

    def train(
        self,
        env: "nett.Body",
        iterations: int,
        device_type: str,
        device: int,
        index: int,
        paths: dict[str, Path]):
        """
        Train the brain.

        Args:
            env (nett.Body): The environment used for training.
            iterations (int): The number of training iterations.
            device_type (str): The type of device used for training.
            device (int): The device index used for training.
            index (int): The index of the model to test, needed for tracking bar.
            paths (dict[str, Path]): The paths for saving logs, models, and plots.

        Raises:
            ValueError: If the environment fails the validation check.
        """
        # validate environment
        env = self._validate_env(env)

        # initialize environment
        log_path = paths["env_logs"]
        env = Monitor(env, str(log_path))
        envs = make_vec_env(env_id=lambda: env, n_envs=1, seed=self.seed)

        # build model
        policy_kwargs = {
            "features_extractor_class": self.encoder,
            "features_extractor_kwargs": {
                "features_dim": inspect.signature(self.encoder).parameters["features_dim"].default
            }
        } if self.encoder else {}

        self.model = self.algorithm(
            self.policy,
            envs,
            batch_size=self.batch_size,
            n_steps=self.buffer_size,
            verbose=0,
            policy_kwargs=policy_kwargs,
            device=torch.device(device_type, device))

        # setup tensorboard logger and attach to model
        tb_logger = configure(str(paths["logs"]), ["stdout", "csv", "tensorboard"])
        self.model.set_logger(tb_logger)

        # set encoder as eval only if train_encoder is not True
        if not self.train_encoder:
            self.model = self._set_encoder_as_eval(self.model)
            self.logger.info(f"Encoder training is set to {str(self.train_encoder).upper()}")

        # initialize callbacks
        callback_list = self._initialize_callbacks(paths, index)

        # train
        self.logger.info(f"Total number of training steps: {iterations}")
        self.model.learn(
            total_timesteps=iterations,
            tb_log_name=self.algorithm.__name__,
            progress_bar=False,
            callback=[callback_list])
        self.logger.info("Training Complete")

        # save
        save_path = f"{paths['model'].joinpath('latest_model.zip')}"
        self.save(save_path)
        self.logger.info(f"Saved model at {save_path}")
        # delete model to free up space
        del self.model
        # plot reward graph
        self.plot_results(iterations=iterations,
                          model_log_dir=paths["env_logs"],
                          plots_dir=paths["plots"],
                          name="reward_graph")

    def test(
        self,
        env,
        iterations,
        model_path: str,
        index: int,
        record_prefix: Optional[str] = None): # pylint: disable=unused-argument
        """
        Test the brain.

        Args:
            env (gym.Env): The environment used for testing.
            iterations (int): The number of testing iterations.
            model_path (str): The path to the trained model.
            index (int): The index of the model to test, needed for tracking bar.
            record_prefix (str, optional): The prefix for recording videos of the testing process. Defaults to None.
        """
        # load previously trained model from save_dir, if it exists
        self.model = self.load(model_path)

        # validate environment
        env = self._validate_env(env)

        # initialize environment
        envs = make_vec_env(env_id=lambda: env, n_envs=1, seed=self.seed)

        self.logger.info(f'Testing with {self.algorithm.__name__}')

        # for when algorithm is RecurrentPPO
        if issubclass(self.algorithm, RecurrentPPO):
            self.logger.info(f"Total number of episodes: {iterations}")
            num_envs = 1
            t = tqdm(total=iterations, desc=f"Condition {index}", position=index)
            for _ in range(iterations):
                obs = env.reset()
                # cell and hidden state of the LSTM
                done, lstm_states = False, None
                # episode start signals are used to reset the lstm states
                episode_starts = np.ones((num_envs,), dtype=bool)
                episode_length = 0
                while not done:
                    action, lstm_states = self.model.predict(
                        obs,
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True)
                    obs, _, done, _ = env.step(action) # obs, rewards, done, info
                    t.update(1)
                    # t.refresh()
                    episode_starts = done
                    episode_length += 1
                    env.render(mode="rgb_array")

        # for all other algorithms
        else:
            self.logger.info(f"Total number of testing steps: {iterations}")
            obs = envs.reset()
            t = tqdm(total=iterations, desc=f"Condition {index}", position=index)
            for _ in range(iterations):
                action, _ = self.model.predict(obs, deterministic=True) # action, states
                obs, _, done, _ = envs.step(action) # obs, reward, done, info
                t.update(1)
                # t.refresh()
                if done:
                    env.reset()
                env.render(mode="rgb_array")
        
        t.close()

    def save(self, path: str) -> None:
        """
        Save the trained model.

        Args:
            path (str): The path to save the model.
        """
        self.model.save(path)

    def load(self, model_path: str | Path) -> OnPolicyAlgorithm | OffPolicyAlgorithm:
        """
        Load a trained model.

        Args:
            model_path (str | Path): The path to the trained model.

        Returns:
            OnPolicyAlgorithm | OffPolicyAlgorithm: The loaded model.
        """
        return self.algorithm.load(model_path)

    def plot_results(self,
        iterations: int,
        model_log_dir: Path,
        plots_dir: Path,
        name: str
    ) -> None:
        """
        Plot the training results.

        Args:
            iterations (int): The number of training iterations.
            model_log_dir (Path): The directory containing the model logs.
            plots_dir (Path): The directory to save the plots.
            name (str): The name of the plot.
        """
        results_plotter.plot_results([str(model_log_dir)],
            iterations,
            results_plotter.X_TIMESTEPS,
            name)
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir.joinpath(f"{name}.png"))
        plt.clf()

    def _validate_encoder(self, encoder: Any | str) -> BaseFeaturesExtractor:
        """
        Validate the encoder.

        Args:
            encoder (Any | str): The encoder to validate.

        Returns:
            BaseFeaturesExtractor: The validated encoder.
        """
        # for when encoder is a string
        if isinstance(encoder, str):
            if encoder not in encoder_dict.keys():
                raise ValueError(f"If a string, should be one of: {encoder_dict.keys()}")
            encoder = getattr(encoders, encoder_dict[encoder])

        # for when encoder is a custom PyTorch encoder
        if isinstance(encoder, BaseFeaturesExtractor):
            # TODO (v0.3) pass dummy torch.tensor on "meta" device to validate embedding dim
            pass

        if encoder and not hasattr(self, 'train_encoder'):
            raise ValueError("encoder passed without setting train_encoder, should be one of: [True, False]")

        return encoder

    def _validate_algorithm(self, algorithm: str | OnPolicyAlgorithm | OffPolicyAlgorithm) -> OnPolicyAlgorithm | OffPolicyAlgorithm:
        """
        Validate the optimization algorithm.

        Args:
            algorithm (str | OnPolicyAlgorithm | OffPolicyAlgorithm): The algorithm to validate.

        Returns:
            OnPolicyAlgorithm | OffPolicyAlgorithm: The validated algorithm.

        Raises:
            ValueError: If the algorithm is not valid.
        """
        # for when policy is a string
        if isinstance(algorithm, str):
            if algorithm not in algorithms:
                raise ValueError(f"If a string, should be one of: {algorithms}")
            # check for the passed policy in stable_baselines3 as well as sb3-contrib
            # at this point in the code, it is guaranteed to be in either of the two
            try:
                algorithm = getattr(stable_baselines3, algorithm)
            except:
                algorithm = getattr(sb3_contrib, algorithm)

        # for when policy algorithm is custom
        elif isinstance(algorithm, OnPolicyAlgorithm) or isinstance(algorithm, OffPolicyAlgorithm):
            # TODO (v0.4) determine appropriate validation checks to be performed before passing
            pass

        else:
            raise ValueError(f"Policy Algorithm should be either one of {algorithms} or a subclass of [{OnPolicyAlgorithm}, {OffPolicyAlgorithm}]")

        return algorithm

    def _validate_policy(self, policy: str | BasePolicy) -> str | BasePolicy:
        """
        Validate the policy model.

        Args:
            policy (str | BasePolicy): The policy model to validate.

        Returns:
            str | BasePolicy: The validated policy model.

        Raises:
            ValueError: If the policy is a string and not one of the supported policies.
            ValueError: If the policy is not a string or a subclass of BasePolicy.
        """
        # for when policy is a string
        if isinstance(policy, str):
            # support tested for PPO and RecurrentPPO only
            if policy not in policies:
                raise ValueError(f"If a string, should be one of: {policies}")

        # for when policy is custom
        elif isinstance(policy, BasePolicy):
            # TODO (v0.4) determine appropriate validation checks to be performed before passing
            pass

        else:
            raise ValueError(f"Policy Model should be either one of {policies} or a subclass of [{BasePolicy}]")

        return policy

    def _validate_reward(self, reward: str) -> str:
        """
        Validate the reward type.

        Args:
            reward (str): The reward type to validate.

        Returns:
            str: The validated reward type.

        Raises:
            ValueError: If the reward is a string and not one of the supported reward types.
        """
        # for when reward is a string
        if not isinstance(reward, str) or reward not in ['supervised', 'unsupervised']:
            raise ValueError("If a string, should be one of: ['supervised', 'unsupervised']")
        return reward

    def _validate_env(self, env: "gym.Env") -> "gym.Env":
        """
        Validate the environment.

        Args:
            env (gym.Env): The environment to validate.

        Returns:
            gym.Env: The validated environment.

        Raises:
            ValueError: If the environment fails the validation check.
        """
        try:
            check_env(env)
        except Exception as ex:
            raise ValueError(f"Failed training env check with {str(ex)}")
        return env

    def _set_encoder_as_eval(self, model: OnPolicyAlgorithm | OffPolicyAlgorithm) -> OnPolicyAlgorithm | OffPolicyAlgorithm:
        """
        Set the encoder as evaluation mode and freeze its parameters.

        Args:
            model (OnPolicyAlgorithm | OffPolicyAlgorithm): The model containing the encoder.

        Returns:
            OnPolicyAlgorithm | OffPolicyAlgorithm: The model with the encoder set as evaluation mode.
        """
        model.policy.features_extractor.eval()
        for param in model.policy.features_extractor.parameters():
            param.requires_grad = False
        return model

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return f"{self.__class__.__name__}({attrs!r})"

    def __str__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return f"{self.__class__.__name__}({attrs!r})"

    def _initialize_callbacks(self, paths: dict[str, Path], index: int) -> CallbackList:
        """
        Initialize the callbacks for training.

        Args:
            paths (dict[str, Path]): The paths for saving logs, models, and plots.
            index (int): The index of the model to test, needed for tracking bar.
        
        Returns:
            CallbackList: The list of callbacks for training.
        """
        save_best_model_callback = SupervisedSaveBestModelCallback(
            summary_freq=30000, 
            save_dir=paths["model"], 
            env_log_path=paths["env_logs"])
        hparam_callback = HParamCallback()
        checkpoint_callback = CheckpointCallback(
            save_freq=30000,
            save_path=paths["checkpoints"],
            name_prefix=self.algorithm.__name__,
            save_replay_buffer=True,
            save_vecnormalize=True)
        bar_callback = multiBarCallback(index)
        return CallbackList([save_best_model_callback, hparam_callback, checkpoint_callback, bar_callback])