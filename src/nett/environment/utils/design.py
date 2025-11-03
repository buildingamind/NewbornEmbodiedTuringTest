"""Grabs experiment design from the executable directory."""

from pathlib import Path
import yaml


def get_experiment_design(executable_path: Path) -> dict[str, int]:
    """
    Gets the experiment design from the executable directory.

    Args:
        executable_path (str): The path to the Unity executable file.

    Returns:
        tuple[int, list[str]]: A tuple containing the number of test conditions and the list of imprinting conditions.

    Raises:
        FileNotFoundError: If the experiment configuration file is not found.
        KeyError: If the experiment configuration file is not properly formatted.
    """
    # get the experiment design from the executable directory
    parent_dir = executable_path.parent
    yaml_files: str = [file for file in parent_dir.glob("*.yaml")]

    if not yaml_files:
        raise FileNotFoundError(
            "No experiment configuration file found in the executable directory. You may be using a Unity executable meant for nett versions prior to v0.5.0. Please update the Unity executable to the latest version or use nett v0.4.1 or older."
        )

    yaml_file: Path = yaml_files[0]

    try:
        # read the yaml file
        with open(yaml_file, "r") as file:
            valid_imprinting_conditions: dict[str, int] = yaml.safe_load(file)
    except KeyError:
        raise KeyError(
            "Experiment configuration file is not properly formatted. It should contain 'num_test_conditions' and 'imprinting_conditions' keys."
        )

    return valid_imprinting_conditions
