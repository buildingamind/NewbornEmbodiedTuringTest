"""
Represents the body of an agent in an environment.

The body determines how observations from the environment are processed before they reach the brain.
It can apply wrappers to modify the observations and provide a different perception to the brain.
"""

from ..environment.environment import Environment
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import sys
from time import sleep
from typing import Optional

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
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
    record_eps: str = "0:0:1",
    seed: Optional[int] = None,
) -> gym.Env:
    # Loads and wraps a single environment instance.
    #
    # This function takes an environment, a task configuration, a list of wrappers,
    # and other parameters to prepare an environment for use. It applies the
    # specified wrappers and, if not in validation mode, sets up monitoring and
    # video recording.
    loaded_env = env.load(config, validation_mode, seed)
    # Record Video only if not in validation mode and not estimating memory

    try:
        for wrapper in wrappers:
            loaded_env = wrapper(loaded_env, device=config.device)
    except Exception as e:
        config.logger.getChild(str(seed)).exception(
            f"Failed to apply wrappers to environment"
        )
        raise e

    # If not in validation mode and memory estimation is not the goal, add monitoring and recording
    if not (validation_mode or config.memory is None):
        if config.current_mode == "train":
            (config.path / "monitor").mkdir(exist_ok=True)
            loaded_env = Monitor(
                loaded_env,
                str(
                    config.path
                    / "monitor"
                    / f"{config.brain_id}_{seed if seed is not None else 0}.csv"
                ),
            )
        loaded_env = _record_wrapper(loaded_env, config, record_eps, seed)

    return loaded_env


def _record_wrapper(  # TODO: Capture both eyes rather than just one
    env: gym.Env, config: TaskConfig, record_eps: str, seed: Optional[int] = None
) -> gym.Env:
    # Wraps an environment to record videos of episodes.
    if seed is None:
        seed = 0
    record_episodes = record_eps.split(":")
    record_start = 0
    record_step = 1
    if len(record_episodes) == 1:
        record_stop = int(record_episodes[0] or 0)
    else:
        record_start = int(record_episodes[0] or 0)
        record_stop = int(record_episodes[1] or sys.maxsize)
        if len(record_episodes) == 3:
            record_step = int(record_episodes[2] or 1)

    if (
        record_stop > 0 and record_step > 0
    ):  #####TODO: Add support for recording multiple agents and multiobs
        # Define a callback to determine which episodes to record
        record_ep_cb = (
            lambda t: t >= record_start
            and t < record_stop
            and (t - record_start) % record_step == 0
        )
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
    The Body acts as an interface between the Environment and the Brain.

    It is responsible for processing observations from the environment before they
    are passed to the brain. This can include applying various wrappers to modify
    the observations, such as changing the resolution, adding binocular vision,
    or applying projections. The body also handles environment setup, including
    vectorization for parallel processing.

    Args:
        wrappers (list[Wrapper | str], optional): List of wrappers to be applied
            to the environment. These can be specified as strings (e.g., "binocular")
            or as gym.Wrapper classes. Defaults to `[]`.
        record_eps (dict[str, str]): Dictionary specifying the episodes to record
            for 'train' and 'test' modes. The format is "start:stop:step".
            Defaults to `{"train": "0:0:1", "test": "0:0:1"}`.
        panini_projection (bool): Whether to apply a Panini projection to the
            environment observations. Defaults to `False`.
        input_resolution (Optional[int]): The resolution to which the
            observations should be resized. Defaults to `None`.
        binocular_vision (bool): Whether the environment provides multiple observations.
            If the "binocular" or "multiobs" wrapper is included, this will be overridden to `True`. This is useful for implementing custom wrappers that handle multiple observations. Defaults to `False`.
    """

    binocular_vision: bool
    wrappers: list[gym.Wrapper]
    record_eps: dict
    panini_projection: bool
    input_resolution: Optional[int]

    def __init__(
        self,
        wrappers: Optional[list[gym.Wrapper | str]] = None,
        record_eps: Optional[dict[str, str]] = None,
        panini_projection: bool = False,
        input_resolution: Optional[int] = None,
        binocular_vision: bool = False,
    ):
        if wrappers is None:
            wrappers = []
        if record_eps is None:
            record_eps = {"train": "0:0:1", "test": "0:0:1"}
        self.binocular_vision = (
            "binocular" in wrappers or "multiobs" in wrappers or binocular_vision
        )
        self.wrappers = validate_wrappers(wrappers)
        self.record_eps = record_eps
        self.panini_projection = panini_projection
        self.input_resolution = input_resolution

    def validate_env(self, env: gym.Env, config: TaskConfig):
        """
        Validates the wrapped environment using Stable Baselines3's environment checker.

        Args:
            env (gym.Env): The base environment.
            config (TaskConfig): The task configuration.

        Raises:
            Exception: If the environment check fails.
        """
        test_env = _load_env(env, config, self.wrappers, True)
        try:
            check_env(test_env)
        except Exception as e:
            config.logger.error("Failed to Validate Environment")
            raise e
        finally:
            if getattr(test_env, "close", None) is not None:
                test_env.close()

    def embed(self, env: gym.Env, config: TaskConfig):
        """
        Embeds the environment in the body, applying necessary wrappers and vectorization.

        This method prepares the environment for interaction with the agent's brain.
        It selects the appropriate wrapper (_zoo_wrapper for multi-agent, _gym_wrapper
        for single-agent) to create a vectorized environment.

        Args:
            env (gym.Env): The environment to embed.
            config (TaskConfig): The task configuration.

        Returns:
            Body: The Body instance with the embedded environment.
        """
        wrapper: callable = self._zoo_wrapper if env.multiagent else self._gym_wrapper

        self.env = wrapper(env, config)

        return self

    def _zoo_wrapper(self, env: gym.Env, config: TaskConfig) -> SB3VecEnvWrapper:
        # Wraps a PettingZoo (multi-agent) environment for use with Stable Baselines3.
        env = _load_env(env, config, self.wrappers, False)
        # TODO: Add support for wrapping ZooEnvironments
        # TODO: Add support for recording agents in ZooEnvironments
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ConcatVecEnv([lambda: env])
        return SB3VecEnvWrapper(env)

    def _gym_wrapper(self, env: gym.Env, config: TaskConfig) -> VecEnv:
        # Wraps a Gymnasium (single-agent) environment in a vectorized env.
        record_eps = self.record_eps.get(config.current_mode, "0:0:1")

        def _init():
            return _load_env(env, config, self.wrappers, False, record_eps)

        return DummyVecEnv([_init])

    def __enter__(self) -> VecEnv:
        # Enter the runtime context related to this object.

        return self.env

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exit the runtime context, closing the environment and handling exceptions.

        # TODO: Add a way to close Unity Environment after episodes are complete (in Unity)
        self.env.close()
        del self.env  # free memory
        return False  # Don't suppress exceptions
