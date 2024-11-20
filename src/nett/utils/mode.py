from enum import Enum


class MODES(Enum):
    TRAIN = ["train"]
    TEST = ["test"]
    FULL = ["train", "test"]


def validate_mode(mode: str) -> str:
    """Validate the mode

    Args:
    mode (str): mode to validate

    Returns:
    str: mode
    """
    try:
        return MODES(mode.upper())
    except KeyError:
        all_modes = [mode.name for mode in MODES]
        raise ValueError(f"Unknown mode type {mode}, should be one of {all_modes}")
