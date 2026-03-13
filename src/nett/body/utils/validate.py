import gymnasium as gym
from functools import partial

from ..wrappers import Binocular, DVS, MultiObs, Retina, Video

# dictionary of all available wrappers. Key is the case-insensitive key for passing to Body.wrappers. Value is the resulting wrapper class.
wrapper_mapping: dict[str, type[gym.Wrapper]] = {
    "binocular": Binocular,
    "dvs": DVS,
    "multiobs": MultiObs,
    "retina": Retina,
    "video": Video,
    # 'trace': trace_recording.TraceRecordingWrapper
}

wrapper_list: list[str] = list(wrapper_mapping.keys())


def validate_wrappers(
    wrappers: list[gym.Wrapper | str | partial],
) -> list[gym.Wrapper | partial]:
    """
    Validate the wrappers.

    Args:
        wrappers (list[Wrapper] | str | partial): The list of wrappers.

    Returns:
        list[Wrapper | partial]: The validated list of wrappers.

    Raises:
        KeyError: If the wrapper is a string that does not correspond to a wrapper.
        TypeError: If any wrapper is not a string, wrapper subclass,
            or functools.partial of a wrapper subclass.
    """
    validated = []
    for wrapper in wrappers:
        if isinstance(wrapper, str):
            try:
                validated.append(wrapper_mapping[wrapper])
            except KeyError:
                raise KeyError(
                    f"If string, wrapper should be one of: {wrapper_list}. Provided wrapper {wrapper} is not one of them."
                )
        elif isinstance(wrapper, type) and issubclass(wrapper, gym.Wrapper):
            validated.append(wrapper)
        elif (
            isinstance(wrapper, partial)
            and isinstance(wrapper.func, type)
            and issubclass(wrapper.func, gym.Wrapper)
        ):
            validated.append(wrapper)
        else:
            raise TypeError(
                "Wrapper should only be either a string, a subclass of gym.Wrapper, "
                "or functools.partial wrapping a subclass of gym.Wrapper"
            )

    return validated
