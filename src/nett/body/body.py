"""
Represents the body of an agent in an environment.

The body determines how observations from the environment are processed before they reach the brain.
It can apply wrappers to modify the observations and provide a different perception to the brain.

Args:
    wrappers (list[Wrapper | str], optional): List of wrappers to be applied to the environment. Defaults to [].
    record_eps (dict[str, int]): Dictionary specifying the number of episodes to record the agent's perspective for each mode (train and test). Defaults to {"train": 0, "test": 0}.
"""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from time import sleep
from typing import Optional

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import supersuit as ss
from supersuit.vector.concat_vec_env import ConcatVecEnv
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.env_checker import check_env

from ..utils.task import TaskConfig


import gymnasium as gym

from .utils import validate_wrappers


class Body:
    multiobs: bool
    wrappers: list[gym.Wrapper]
    record_eps: dict

    def __init__(
        self,
        wrappers: list[gym.Wrapper | str] = [],
        record_eps: dict[str, int] = {"train": 0, "test": 0},
    ):
        self.multiobs = "binocular" in wrappers
        self.wrappers = validate_wrappers(wrappers)
        self.record_eps = record_eps

    def _load_env(
        self, env: gym.Env, config: TaskConfig, validation_mode: bool, seed: Optional[int] = None
    ) -> gym.Env:
        env = env.load(config, validation_mode, seed)

        try:
            for wrapper in self.wrappers:
                env = wrapper(env)
        except Exception as e:
            config.logger.getChild(seed).exception(
                f"Failed to apply wrappers to environment"
            )
            raise e

        return env

    def _validate_env(self, env: gym.Env, config: TaskConfig):
        try:
            test_env = self._load_env(env, config, True)
            check_env(test_env)
        finally:
            if "test_env" in locals() and getattr(test_env, "close", None) is not None:
                test_env.close()

    def embed(self, env: gym.Env, config: TaskConfig):

        if env.multiagent:
            wrapped_env = self._zoo_wrapper(env, config)
        else:
            self._validate_env(env, config)
            # validate
            if config.current_mode == "train":
                wrapped_env = self._single_gym_wrapper(env, config)
            else:  # test
                wrapped_env = self._multi_gym_wrapper(env, config)

        self.env = wrapped_env
        return self

    def _record_wrapper(self, env: gym.Env, config: TaskConfig, seed: int = 0) -> gym.Env:
        record_episodes = self.record_eps.get(config.current_mode, 0)
        if config.current_mode == "test":
            record_episodes /= env.n_parallel_envs
        if record_episodes > 0:
            record_ep_cb = lambda t: t < record_episodes
            return RecordVideo(
                env,
                config.path / "env_recs" / "agent",
                episode_trigger=record_ep_cb,
                name_prefix=f"agent{seed}_",
            )
        config.logger.info('Finished recording wrapper')
        return env

    def _zoo_wrapper(self, env: gym.Env, config: TaskConfig) -> SB3VecEnvWrapper:
        env = self._load_env(env, config, False)
        # TODO: Add support for wrapping ZooEnvironments
        # TODO: Add support for recording agents in ZooEnvironments
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ConcatVecEnv([lambda: env])
        return SB3VecEnvWrapper(env)

    def _single_gym_wrapper(self, env: gym.Env, config: TaskConfig) -> DummyVecEnv:
        def callback():
            loaded_env = self._load_env(env, config, False)
            return self._record_wrapper(loaded_env, config)

        return DummyVecEnv([callback])

    def _multi_gym_wrapper(self, env: gym.Env, config: TaskConfig) -> SubprocVecEnv:
        def seed_callback(seed):
            def callback():
                sleep(seed)
                loaded_env = self._load_env(env, config, False, seed)
                return self._record_wrapper(loaded_env, config, seed)

            return callback

        # create n_envs environments
        return SubprocVecEnv([seed_callback(seed) for seed in range(env.n_parallel_envs)])

    def __enter__(self) -> VecEnv:
        """return env at beginning of `with` statement"""
        return self.env

    def __exit__(self, *args):
        """close env outside of `with` statement"""
        return self.env.close()
