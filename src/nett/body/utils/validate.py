import gymnasium as gym

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


def validate_wrappers(wrappers: list[gym.Wrapper | str]) -> list[gym.Wrapper]:
    """
    Validate the wrappers.

    Args:
        wrappers (list[Wrapper] | str): The list of wrappers.

    Returns:
        list[Wrapper]: The validated list of wrappers.

    Raises:
        KeyError: If the wrapper is a string that does not correspond to a wrapper.
        TypeError: If any wrapper is not an instance of str or gym.Wrapper.
    """
    for i, wrapper in enumerate(wrappers):
        if type(wrapper) == str:
            try:
                wrappers[i] = wrapper_mapping[wrapper]
            except KeyError:
                raise KeyError(
                    f"If string, wrapper should be one of: {wrapper_list}. Provided wrapper {wrapper} is not one of them."
                )
        elif isinstance(wrapper, type) and issubclass(wrapper, gym.Wrapper):
            wrappers[i] = wrapper
        else:
            raise TypeError(
                "Wrapper should only be either a string or a subclass of gym.Wrapper"
            )

    return wrappers
