from enum import Enum

from typing import Optional


class MODES(Enum):
    TRAIN = ["train"]
    TEST = ["test"]
    FULL = ["train", "test"]


def validate_mode(mode: str) -> list[str]:
    """Validate the mode

    Args:
    mode (str): mode to validate

    Returns:
    str: mode
    """
    try:
        return MODES[mode.upper()].value
    except KeyError:
        all_modes = [mode.name for mode in MODES]
        raise ValueError(f"Unknown mode type {mode}, should be one of {all_modes}")


def validate_conditions(all_conditions: list[str], conditions: Optional[list[str]]):
    # check if user-defined their own conditions
    if conditions is None:
        # default to all conditions
        return all_conditions
    elif not set(conditions).issubset(all_conditions):
        raise ValueError(
            f"Unknown conditions: {conditions}. Available conditions are: {all_conditions}"
        )
    else:
        return conditions
