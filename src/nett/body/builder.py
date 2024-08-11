"""The body of the agent in the environment."""
from typing import Any
from gym import Wrapper, Env

from nett.body import types
# from nett.body import ascii_art

# this will have the necessary wrappers before the observations interact with the brain,
# because it is the body that determines how the the observations will be processed.
# specifically, in the case of the two-eyed agent, because the agent has two eyes, the observations are stereo
# and need to be processed differently before they make it to the brain.
# the body is the medium through which information travels from the environment to the brain.
# the brain is limited by what the body can percieve and no information is objective.
# NO INFORMATION IS OBJECTIVE (!!!!!!)
class Body:
    """Represents the body of an agent in an environment.

    The body determines how observations from the environment are processed before they reach the brain.
    It can apply wrappers to modify the observations and provide a different perception to the brain.

    Args:
        type (str, optional): The type of the agent's body. Defaults to "basic".
        wrappers (list[Wrapper] | None, optional): List of wrappers to be applied to the environment. Defaults to None.
        dvs (bool, optional): Flag indicating whether the agent uses dynamic vision sensors. Defaults to False.

    Raises:
        ValueError: If the agent type is not valid.
        TypeError: If dvs is not a boolean.

    Example:

        >>> from nett import Body
        >>> body = Body(type="basic", wrappers=None, dvs=False)
    """

    def __init__(self, type: str = "basic",
                    wrappers: list[Wrapper] | None = None,
                    dvs: bool = False) -> None:
        """
        Constructor method
        """
        from nett import logger
        self.logger = logger.getChild(__class__.__name__)
        self.type = self._validate_agent_type(type)
        self.wrappers = self._validate_wrappers(wrappers)
        self.dvs = self._validate_dvs(dvs)

    def _validate_agent_type(self, type: str) -> str:
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


    def _validate_dvs(self, dvs: bool) -> bool:
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

    def _validate_wrappers(self, wrappers: list[Wrapper] | None) -> list[Any] | None:
        """
        Validate the wrappers.

        Args:
            wrappers (list[Wrapper] | None): The list of wrappers.

        Returns:
            list[Wrapper] | None: The validated list of wrappers.

        Raises:
            ValueError: If any wrapper is not an instance of gym.Wrapper.
        """
        if wrappers is not None:
            for wrapper in wrappers:
                if not isinstance(wrapper, Wrapper):
                    raise ValueError("Wrappers must inherit from gym.Wrapper")
        return wrappers


    def __call__(self, env: Env) -> Env:
        """
        Apply the registered wrappers to the environment.

        Args:
            env (Env): The environment.

        Returns:
            Env: The modified environment.
        """
        if self.wrappers:
            for wrapper in self.wrappers:
                env = wrapper(env)
        return env

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
