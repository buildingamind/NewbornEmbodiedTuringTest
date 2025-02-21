"""
Represents the body of an agent in an environment.

The body determines how observations from the environment are processed before they reach the brain.
It can apply wrappers to modify the observations and provide a different perception to the brain.
"""

from ..environment.environment import Environment
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
from .utils import validate_wrappers


def _load_env(
    env: Environment,
    config: TaskConfig,
    wrappers: list,
    validation_mode: bool,
    seed: Optional[int] = None,
) -> gym.Env:
    loaded_env = env.load(config, validation_mode, seed)

    try:
        for wrapper in wrappers:
            loaded_env = wrapper(env)
    except Exception as e:
        config.logger.getChild(seed).exception(
            f"Failed to apply wrappers to environment"
        )
        raise e

    return loaded_env


def _record_wrapper(
    env: gym.Env, config: TaskConfig, record_eps: dict, seed: int = 0
) -> gym.Env:
    record_episodes = record_eps.get(config.current_mode, 0)
    if config.current_mode == "test":
        record_episodes /= config.n_parallel_envs
    if record_episodes > 0:
        record_ep_cb = lambda t: t < record_episodes
        return RecordVideo(
            env,
            config.path / "env_recs" / "agent",
            episode_trigger=record_ep_cb,
            name_prefix=f"agent{seed}_",
        )
    config.logger.info("Finished recording wrapper")
    return env


class Body:
    """
    Args:
        wrappers (list[Wrapper | str], optional): List of wrappers to be applied to the environment. Defaults to `[]`.
        record_eps (dict[str, int]): Dictionary specifying the number of episodes to record the agent's perspective for each mode (train and test). Defaults to `{"train": 0, "test": 0}`.
    """

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

    def _validate_env(self, env: gym.Env, config: TaskConfig):
        try:
            test_env = _load_env(env, config, self.wrappers, True)
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

    def _zoo_wrapper(self, env: gym.Env, config: TaskConfig) -> SB3VecEnvWrapper:
        env = _load_env(env, config, self.wrappers, False)
        # TODO: Add support for wrapping ZooEnvironments
        # TODO: Add support for recording agents in ZooEnvironments
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ConcatVecEnv([lambda: env])
        return SB3VecEnvWrapper(env)

    def _single_gym_wrapper(self, env: gym.Env, config: TaskConfig) -> DummyVecEnv:
        def callback():
            loaded_env = _load_env(env, config, self.wrappers, False)
            return _record_wrapper(loaded_env, config, self.record_eps)

        return DummyVecEnv([callback])

    def _multi_gym_wrapper(self, env: gym.Env, config: TaskConfig) -> SubprocVecEnv:
        def seed_callback(env, seed, wrappers, record_eps):
            def callback():
                sleep(seed)
                loaded_env = _load_env(env, config, wrappers, False, seed)
                return _record_wrapper(loaded_env, config, record_eps, seed)

            return callback

        # create n_envs environments
        wrappers = self.wrappers
        record_eps = self.record_eps
        seed_list = range(config.n_parallel_envs)
        return SubprocVecEnv(
            [seed_callback(env, seed, wrappers, record_eps) for seed in seed_list]
        )

    def __enter__(self) -> VecEnv:
        """return env at beginning of `with` statement"""
        return self.env

    def __exit__(self, exc_type, exc_val, exc_tb):
        """close env outside of `with` statement"""
        self.env.close()
        if exc_type is None:
            return False
        # An exception occurred
        print(f"Exception type: {exc_type}")
        print(f"Exception value: {exc_val}")
        # Optionally print traceback using traceback module
        import traceback

        traceback.print_tb(exc_tb)
        return True  # Suppress the exception in this example
