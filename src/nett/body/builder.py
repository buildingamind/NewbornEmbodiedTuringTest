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
    def __init__(self, type: str = "basic",
                 wrappers: list[Any] | None = None,
                 dvs: bool = False) -> None:
        from nett import logger
        self.logger = logger.getChild(__class__.__name__)
        self.type = type
        self.wrappers = self._validate_wrappers(wrappers)
        self.dvs = self._validate_dvs(dvs)

    def _validate_type(self, type: str) -> str:
        if type not in types:
            raise ValueError(f"type must be one of {types}")

    def _validate_dvs(self, dvs: bool) -> bool:
        if not isinstance(dvs, bool):
            raise TypeError("dvs should be a boolean [True, False]")
        return dvs

    def _validate_wrappers(self, wrappers: list[Any] | None) -> list[Any] | None:
        if wrappers is not None:
            for wrapper in wrappers:
                if not isinstance(wrapper, Wrapper):
                    raise ValueError("Wrappers must inherit from gym.Wrapper")
        return wrappers

    def __call__(self, env: Env) -> Env:
        if self.wrappers:
            for wrapper in self.wrappers:
                env = wrapper(env)
        return env

    def __repr__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return f"{self.__class__.__name__}({attrs!r})"

    def __str__(self) -> str:
        attrs = {k: v for k, v in vars(self).items() if k != 'logger'}
        return f"{self.__class__.__name__}({attrs!r})"

    def _register(self) -> None:
        raise NotImplementedError
