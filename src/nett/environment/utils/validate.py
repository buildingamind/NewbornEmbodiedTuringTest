"""Validation utilities for the environment component."""

from typing import Optional
from pathlib import Path


def validate_executable_path(executable_path: str) -> Path:
    """
    Validates the Unity executable path.

    Args:
        executable_path (str): The path to the Unity executable file.

    Returns:
        str: The validated path to the Unity executable file.

    Raises:
        ValueError: If the executable path is not a string.
        FileNotFoundError: If the executable path does not exist.
        ValueError: If the executable path is not a valid Unity executable file.
        FileNotFoundError: If the directory does not contain the 'UnityPlayer.so' file.
        FileNotFoundError: If the data directory does not exist.
    """
    if not isinstance(executable_path, str):
        raise ValueError(
            f"executable_path should be a string. Instead, it is of type {type(executable_path)}"
        )

    executable_path: Path = Path(executable_path)
    unityplayer_path: Path = executable_path.with_name("UnityPlayer.so")
    datadir_path: Path = executable_path.with_name(executable_path.stem + "_Data")

    # check if executable is correct filetype
    if executable_path.suffix not in [".x86_64", ".x86"]:
        raise ValueError(f"{executable_path} is not a valid Unity executable file")

    # check if the executable path exists
    if not executable_path.is_file():
        raise FileNotFoundError(f"{executable_path} does not exist")

    # check if the directory contains the 'UnityPlayer.so' file
    if not unityplayer_path.is_file():
        raise FileNotFoundError(
            f"The directory {executable_path} does not contain the file 'UnityPlayer.so'. This may not be a valid Unity executable."
        )

    # check if the data directory exists
    if not datadir_path.is_dir():
        raise FileNotFoundError(
            f"Expected {datadir_path} to exist in executable directory, but it does not exist. Please check that the path to the Unity executable is correct and that the data directory and executable use the same naming convention."
        )

    return executable_path


def validate_conditions(all_conditions: list[str] | dict[str, int], conditions: Optional[list[str]]):
    """
    Validates the imprinting conditions.
    
    Args:
        all_conditions (list[str] | dict[str, int]): The list or dictionary of all available imprinting conditions.
        conditions (Optional[list[str]]): The list of user-specified imprinting conditions.

    Returns:
        list[str]: The validated list of imprinting conditions.

    Raises:
        ValueError: If any of the user-specified conditions are not in the list of available conditions.
    """

    if isinstance(all_conditions, dict):
        all_conditions = list(all_conditions.keys())

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
