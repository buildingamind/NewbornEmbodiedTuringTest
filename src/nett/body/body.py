"""
Represents the body of an agent in an environment.

The body determines how observations from the environment are processed before they reach the brain.
It can apply wrappers to modify the observations and provide a different perception to the brain.
"""

from math import ceil
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
    record_eps: int = 0,
    seed: Optional[int] = None,
) -> gym.Env:
    loaded_env = env.load(config, validation_mode, seed)
    # Record Video only if not in validation mode and not estimating memory
    if not (validation_mode or config.memory is None):
        loaded_env = _record_wrapper(loaded_env, config, record_eps, seed)

    try:
        for wrapper in wrappers:
            loaded_env = wrapper(loaded_env)
    except Exception as e:
        config.logger.getChild(seed).exception(
            f"Failed to apply wrappers to environment"
        )
        raise e

    return loaded_env


def _record_wrapper(  # TODO: Capture both eyes rather than just one
    env: gym.Env, config: TaskConfig, record_eps: int, seed: Optional[int] = None
) -> gym.Env:
    if seed is None:
        seed = 0
    record_episodes = record_eps
    if config.current_mode == "test":
        record_episodes = ceil(record_episodes / config.n_parallel_envs)
    if (
        record_episodes > 0
    ):  #####TODO: Add support for recording multiple agents and multiobs
        record_ep_cb = lambda t: t < record_episodes
        return RecordVideo(
            env,
            config.path / "recordings" / "agent" / config.current_mode,
            episode_trigger=record_ep_cb,
            name_prefix=f"agent{seed}",
            disable_logger=True,
        )
    return env


class Body:
    """
    Args:
        wrappers (list[Wrapper | str], optional): List of wrappers to be applied to the environment. Defaults to `[]`.
        record_eps (dict[str, int]): Dictionary specifying the number of episodes to record the agent's perspective for each mode (train and test). Defaults to `{"train": 0, "test": 0}`.
        panini_projection (bool): Whether to apply a Panini projection to the environment observations. Defaults to `False`.
    """

    multiobs: bool
    wrappers: list[gym.Wrapper]
    record_eps: dict
    panini_projection: bool

    def __init__(
        self,
        wrappers: list[gym.Wrapper | str] = [],
        record_eps: dict[str, int] = {"train": 0, "test": 0},
        panini_projection: bool = False,
    ):
        self.multiobs = "binocular" in wrappers
        self.wrappers = validate_wrappers(wrappers)
        self.record_eps = record_eps
        self.panini_projection = panini_projection

    def validate_env(self, env: gym.Env, config: TaskConfig):
        test_env = _load_env(env, config, self.wrappers, True)
        try:
            check_env(test_env)
        except Exception as e:
            self.logger.error("Failed to Validate Environment")
            raise e
        finally:
            if getattr(test_env, "close", None) is not None:
                test_env.close()

    def embed(self, env: gym.Env, config: TaskConfig):
        """Embed the environment in the body."""
        wrapper: callable

        if env.multiagent:
            wrapper = self._zoo_wrapper
        elif config.current_mode == "train":
            wrapper = self._single_gym_wrapper
        else:  # test
            wrapper = self._multi_gym_wrapper

        self.env = wrapper(env, config)

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
            record_eps = self.record_eps.get(config.current_mode, 0)
            return _load_env(env, config, self.wrappers, False, record_eps)

        return DummyVecEnv([callback])

    def _multi_gym_wrapper(self, env: gym.Env, config: TaskConfig) -> SubprocVecEnv:
        def seed_callback(env, seed, wrappers, record_eps):
            def callback():
                sleep(seed)
                return _load_env(env, config, wrappers, False, record_eps, seed)

            return callback

        # create n_envs environments
        wrappers = self.wrappers
        record_eps = self.record_eps.get(config.current_mode, 0)
        seed_list = range(config.n_parallel_envs)
        return SubprocVecEnv(
            [seed_callback(env, seed, wrappers, record_eps) for seed in seed_list]
        )

    def __enter__(self) -> VecEnv:
        """return env at beginning of `with` statement"""
        return self.env

    def __exit__(self, exc_type, exc_val, exc_tb):
        """close env outside of `with` statement"""
        # TODO: Add a way to close Unity Environment after episodes are complete (in Unity)
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
