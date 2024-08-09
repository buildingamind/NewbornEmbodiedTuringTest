"""The body of the agent in the environment."""
from gym import Env, Wrapper
from stable_baselines3.common.env_checker import check_env

from nett.body import types
from nett.body.wrappers.dvs import DVSWrapper
# from nett.body import ascii_art

class Body:
    """Represents the body of an agent in an environment.

    The body determines how observations from the environment are processed before they reach the brain.
    It can apply wrappers to modify the observations and provide a different perception to the brain.

    Args:
        type (str, optional): The type of the agent's body. Defaults to "basic".
        wrappers (list[Wrapper], optional): List of wrappers to be applied to the environment. Defaults to [].
        dvs (bool, optional): Flag indicating whether the agent uses dynamic vision sensors. Defaults to False.

    Raises:
        ValueError: If the agent type is not valid.
        TypeError: If dvs is not a boolean.

    Example:

        >>> from nett import Body
        >>> body = Body(type="basic", wrappers=None, dvs=False)
    """

    def __init__(self, type: str = "basic",
                    wrappers: list[Wrapper] = [],
                    dvs: bool = False) -> None:
        """
        Constructor method
        """
        from nett import logger
        self.logger = logger.getChild(__class__.__name__)
        self.type = self._validate_agent_type(type)
        self.wrappers = self._validate_wrappers(wrappers)
        self.dvs = self._validate_dvs(dvs)

    @staticmethod
    def _validate_agent_type(type: str) -> str:
        """
        Validate the agent type.

        Args:
            type (str): The type of the agent's body.

        Returns:
            str: The validated agent type.

        Raises:
            ValueError: If the agent type is not valid.
        """
        if type not in types:
            raise ValueError(f"agent type must be one of {types}")
        return type

    @staticmethod
    def _validate_dvs(dvs: bool) -> bool:
        """
        Validate the dvs flag.

        Args:
            dvs (bool): The dvs flag.

        Returns:
            bool: The validated dvs flag.

        Raises:
            TypeError: If dvs is not a boolean.
        """
        if not isinstance(dvs, bool):
            raise TypeError("dvs should be a boolean [True, False]")
        return dvs

    @staticmethod
    def _validate_wrappers(wrappers: list[Wrapper]) -> list[Wrapper]:
        """
        Validate the wrappers.

        Args:
            wrappers (list[Wrapper]): The list of wrappers.

        Returns:
            list[Wrapper]: The validated list of wrappers.

        Raises:
            ValueError: If any wrapper is not an instance of gym.Wrapper.
        """
        for wrapper in wrappers:
            if not issubclass(wrapper, Wrapper):
                raise ValueError("Wrappers must inherit from gym.Wrapper")
        return wrappers

    @staticmethod
    def _wrap(env: Env, wrapper: Wrapper) -> Env:
        """
        Wraps the environment with the registered wrappers.

        Args:
            env (Env): The environment to wrap.
            wrapper (Wrapper): The wrapper to apply.

        Returns:
            Env: The wrapped environment.

        Raises:
            Exception: If the environment does not follow the Gym API.
        """
        # wrap env
        env = wrapper(env)
        # check that the env follows Gym API
        env_check = check_env(env, warn=True)
        if env_check != None:
            raise Exception(f"Failed env check")

        return env

    def __call__(self, env: Env) -> Env:
        """
        Apply the registered wrappers to the environment.

        Args:
            env (Env): The environment.

        Returns:
            Env: The modified environment.
        """
        try:
            # apply DVS wrapper
            if self.dvs:
                env = self._wrap(env, DVSWrapper)
            # apply all custom wrappers
            if self.wrappers:
                for wrapper in self.wrappers:
                    env = self._wrap(env, wrapper)
        except Exception as e:
            self.logger.exception(f"Failed to apply wrappers to environment")
            raise e
        self.env = env
        return self.env
    
    def __enter__(self):
        return self.env

    def __exit__(self):
        self.env.close()
    

    def __repr__(self) -> str:
        """
        Return a string representation of the Body object.

        Returns:
            str: The string representation of the Body object.
        """
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"


    def __str__(self) -> str:
        """
        Return a string representation of the Body object.

        Returns:
            str: The string representation of the Body object.
        """
        attrs = {k: v for k, v in vars(self).items() if k != "logger"}
        return f"{self.__class__.__name__}({attrs!r})"


    def _register(self) -> None:
        """
        Register the body with the environment.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError
