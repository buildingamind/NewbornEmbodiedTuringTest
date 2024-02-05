import torch
import gym
import inspect
import stable_baselines3
import sb3_contrib
import numpy as np
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from typing import Any
from pathlib import Path
from nett.brain import algorithms, policies, encoders_list, encoder_dict
from nett.brain import encoders
from nett.utils.callbacks import SupervisedSaveBestModelCallback, HParamCallback
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


# TO DO (v0.2): Extend with support for custom policy models
# TO DO (v0.2): should we move validation checks to utils under validations.py?
class Brain:
    def __init__(self,
                 encoder: Any | str = None,
                 embedding_dim: int | None = None,
                 policy: Any | str | None = None,
                 algorithm: Any | str | None = None,
                 reward: Any | str = "supervised",
                 batch_size: int = 512,
                 buffer_size: int = 2048,
                 train_encoder: bool | None = False,
                 seed: int = 12
                 ) -> None:
        from nett import logger
        self.logger = logger.getChild(__class__.__name__)
        self.train_encoder = train_encoder
        self.encoder = self._validate_encoder(encoder) if encoder else None
        self.algorithm = self._validate_algorithm(algorithm) if algorithm else None
        self.policy = self._validate_policy(policy) if policy else None
        self.reward = self._validate_reward(reward) if reward else None
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seed = seed

    def train(self, env, iterations, device_type: str, device: int, paths: dict[str, Path]):
        # validate environment
        env = self._validate_env(env)

        # initialize environment
        log_path = paths['env_logs']
        env = Monitor(env, str(log_path))
        envs = make_vec_env(env_id=lambda : env, n_envs=1, seed=self.seed)

        # build model
        if self.encoder:
            policy_kwargs = dict(features_extractor_class=self.encoder,
                                 features_extractor_kwargs=dict(features_dim=inspect.signature(self.encoder).parameters['features_dim'].default))
        else:
            policy_kwargs = {}
        self.model = self.algorithm(self.policy,
                                    envs,
                                    batch_size=self.batch_size,
                                    n_steps=self.buffer_size,
                                    verbose=0,
                                    policy_kwargs=policy_kwargs,
                                    device=torch.device(device_type, device))

        # setup tensorboard logger and attach to model
        tb_logger = configure(str(paths['logs']), ["stdout", "csv", "tensorboard"])
        self.model.set_logger(tb_logger)

        # set encoder as eval only if train_encoder is not True
        if not self.train_encoder:
            self.model = self._set_encoder_as_eval(self.model)
            self.logger.info(f'Encoder training is set to {str(self.train_encoder).upper()}')

        # initialize callbacks
        save_best_model_callback = SupervisedSaveBestModelCallback(summary_freq=30000,
                                                                   save_dir=paths['model'],
                                                                   env_log_path=paths['env_logs'])
        hparam_callback = HParamCallback()
        checkpoint_callback = CheckpointCallback(save_freq=30000,
                                                 save_path=paths["checkpoints"],
                                                 name_prefix=self.algorithm.__name__,
                                                 save_replay_buffer=True,
                                                 save_vecnormalize=True)
        callback_list = CallbackList([save_best_model_callback, hparam_callback, checkpoint_callback])

        # train
        self.logger.info(f"Total number of training steps: {iterations}")
        self.model.learn(total_timesteps=iterations,
                         tb_log_name=self.algorithm.__name__,
                         progress_bar=True,
                         callback=[callback_list])
        self.logger.info(f"Training Complete")

        # save
        save_path = f"{paths['model'].joinpath('latest_model.zip')}"
        self.save(save_path)
        self.logger.info(f"Saved model at {save_path}")
        # delete model to free up space
        del self.model
        # plot reward graph
        self.plot_results(iterations=iterations,
                          model_log_dir=paths['env_logs'],
                          plots_dir=paths['plots'],
                          name="reward_graph")

    def test(self, env, iterations, model_path: str, record_prefix: str | None = None):
        # load previously trained model from save_dir, if it exists
        self.model = self.load(model_path)

        # validate environment
        env = self._validate_env(env)

        # initialize environment
        envs = make_vec_env(env_id=lambda : env, n_envs=1, seed=self.seed)

        # for when algorithm is RecurrentPPO
        if issubclass(self.algorithm, RecurrentPPO):
            self.logger.info(f'Testing with {self.algorithm.__name__}')
            self.logger.info(f"Total number of episodes: {iterations}")
            num_envs = 1
            for episode in tqdm(range(iterations)):
                obs = env.reset()
                # cell and hidden state of the LSTM
                done, lstm_states = False, None
                # episode start signals are used to reset the lstm states
                episode_starts = np.ones((num_envs,), dtype=bool)
                episode_length = 0
                while not done:
                    action, lstm_states = self.model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                    obs, rewards, done, info = env.step(action)
                    episode_starts = done
                    episode_length += 1
                    env.render(mode="rgb_array")

        # for all other algorithms
        else:
            self.logger.info(f'Testing with {self.algorithm.__name__}')
            self.logger.info(f"Total number of testing steps: {iterations}")
            obs = envs.reset()
            for step in tqdm(range(iterations)):
                action, states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = envs.step(action)
                if done:
                    env.reset()
                env.render(mode="rgb_array")

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, model_path: str | Path):
        return self.algorithm.load(model_path)

    def plot_results(self, iterations: int, model_log_dir: Path, plots_dir: Path, name: str) -> None:
        results_plotter.plot_results([str(model_log_dir)],
                                     iterations,
                                     results_plotter.X_TIMESTEPS,
                                     name)
        Path.mkdir(plots_dir)
        plt.savefig(plots_dir.joinpath(f"{name}.png"))
        plt.clf()

    def _validate_encoder(self, encoder: Any | str) -> BaseFeaturesExtractor:
        # for when encoder is a string
        if isinstance(encoder, str):
            if encoder not in encoder_dict.keys():
                raise ValueError(f"If a string, should be one of: {encoder_dict.keys()}")
            else:
                encoder = getattr(encoders, encoder_dict[encoder])

        # for when encoder is a custom PyTorch encoder
        if isinstance(encoder, BaseFeaturesExtractor):
            # TO DO (v0.2) pass dummy torch.tensor on "meta" device to validate embedding dim
            pass

        if encoder and not hasattr(self, 'train_encoder'):
            raise ValueError("encoder passed without setting train_encoder, should be one of: [True, False]")

        return encoder

    def _validate_algorithm(self, algorithm: Any | str) -> OnPolicyAlgorithm | OffPolicyAlgorithm:
        # for when policy is a string
        if isinstance(algorithm, str):
            if algorithm not in algorithms:
                raise ValueError(f"If a string, should be one of: {algorithms}")
            else:
                # check for the passed policy in stable_baselines3 as well as sb3-contrib
                # at this point in the code, it is guaranteed to be in either of the two
                try:
                    algorithm = getattr(stable_baselines3, algorithm)
                except:
                    algorithm = getattr(sb3_contrib, algorithm)

        # for when policy algorithm is custom
        elif isinstance(algorithm, OnPolicyAlgorithm) or isinstance(algorithm, OffPolicyAlgorithm):
            # TO DO (v0.3) determine appropriate validation checks to be performed before passing
            pass

        else:
            raise ValueError(f"Policy Algorithm should be either one of {algorithms} or a subclass of [{OnPolicyAlgorithm}, {OffPolicyAlgorithm}]")

        return algorithm

    def _validate_policy(self, policy: Any | str) -> str | BasePolicy:
        # for when policy is a string
        if isinstance(policy, str):
            # support teseted for PPO and RecurrentPPO only
            if policy not in policies:
                raise ValueError(f"If a string, should be one of: {policies}")

        # for when policy is custom
        elif isinstance(policy, BasePolicy):
            # TO DO (v0.3) determine appropriate validation checks to be performed before passing
            pass

        else:
            raise ValueError(f"Policy Model should be either one of {policies} or a subclass of [{BasePolicy}]")

        return policy

    def _validate_reward(self, reward: Any | str) -> Any | str:
        # for when reward is a string
        if isinstance(reward, str) and reward not in ['supervised', 'unsupervised']:
            raise ValueError(f"If a string, should be one of: ['supervised', 'unsupervised']")
        return reward

    # TO DO (v0.2) add typehinting for gym environments
    def _validate_env(self, env) -> Any:
        try:
            check_env(env)
        except Exception as ex:
            raise ValueError(f"Failed training env check with {str(ex)}")
        return env

    def _set_encoder_as_eval(self, model):
        model.policy.features_extractor.eval()
        for param in model.policy.features_extractor.parameters():
            param.requires_grad = False
        return model

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return "{}({!r})".format(self.__class__.__name__, attrs)

    def __str__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return "{}({!r})".format(self.__class__.__name__, attrs)


