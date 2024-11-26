"""The body of the agent in the environment."""

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from nett.utils.task import Task
from nett import logger

from .wrappers import validate_wrappers


class Body(gym.Wrapper):
    """Represents the body of an agent in an environment.

    The body determines how observations from the environment are processed before they reach the brain.
    It can apply wrappers to modify the observations and provide a different perception to the brain.

    Args:
        type (str, optional): The type of the agent's body. Defaults to "basic".
        wrappers (list[Wrapper | str], optional): List of wrappers to be applied to the environment. Defaults to [].

    Raises:
        ValueError: If the agent type is not valid.
        TypeError: If dvs is not a boolean.

    Example:

        >>> from nett import Body
        >>> body = Body(type="basic", wrappers=None, dvs=False)
    """

    @classmethod
    def initialize(
        cls,
        wrappers: list[gym.Wrapper | str] = [],
        record_eps: dict = {"train": 0, "test": 0},
    ) -> None:
        """
        Constructor method
        """
        cls.logger = logger.getChild(__class__.__name__)

        cls.multiobs = "binocular" in wrappers
        cls.wrappers = validate_wrappers(wrappers)
        cls.record_eps = record_eps

    def __init__(self, env: gym.Env, task: Task) -> None:
        try:
            for wrapper in self.wrappers:
                env = wrapper(env)
        except Exception as e:
            self.logger.exception(f"Failed to apply wrappers to environment")
            raise e

        env = self.record_wrapper(env, task)

        super().__init__(env)
        self.env = env

    def record_wrapper(self, env: gym.Env, task: Task) -> gym.Env:
        record_episodes = self.record_eps.get(task.mode, 0)
        if record_episodes > 0:
            record_ep_cb = lambda t: t < record_episodes
            return RecordVideo(
                env,
                task.path / "env_recs" / "agent",
                episode_trigger=record_ep_cb,
                name_prefix="agent",
            )
        return env
