"""Isaac-aware body component for NETT-skrl.

The body owns observation wrappers and body-side perception settings before
the loaded environment is handed to the brain.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym

from .utils import validate_wrappers
from .wrappers.channels_first import ChannelsFirst


class Body:
    """Interface between a NETT-skrl environment and brain.

    Args:
        wrappers: Observation wrappers to apply after the Isaac environment is
            loaded. Entries may be registry names or ``gym.Wrapper`` classes.
        binocular_vision: Optional override for the environment's binocular
            observation setting.
        input_resolution: Optional override for the per-eye input resolution.
    """

    wrappers: list[type[gym.Wrapper]]
    binocular_vision: Optional[bool]
    input_resolution: Optional[int]

    def __init__(
        self,
        wrappers: Optional[list[str | type[gym.Wrapper]]] = None,
        *,
        binocular_vision: Optional[bool] = None,
        input_resolution: Optional[int] = None,
    ) -> None:
        self.wrappers = validate_wrappers(wrappers)
        self.binocular_vision = binocular_vision
        self.input_resolution = input_resolution
        self._channels_first: ChannelsFirst | None = None

    def adjust_to_agent(self, env, *, num_brains: int, **kwargs) -> None:
        """Apply body-side settings to an environment before loading it."""
        body_kwargs = dict(kwargs)
        if self.binocular_vision is not None:
            body_kwargs["binocular_vision"] = self.binocular_vision
        if self.input_resolution is not None:
            body_kwargs["input_resolution"] = self.input_resolution
        env.adjust_to_agent(num_brains=num_brains, **body_kwargs)

    def embed(self, env, config):
        """Load ``env`` for ``config`` and apply body wrappers."""
        loaded = env.load(config)
        return self.wrap(loaded)

    def wrap(self, loaded_env):
        """Apply configured observation wrappers, then ChannelsFirst as the terminal step."""
        for wrapper in self.wrappers:
            loaded_env = wrapper(loaded_env)
        self._channels_first = ChannelsFirst(loaded_env)
        return self._channels_first
