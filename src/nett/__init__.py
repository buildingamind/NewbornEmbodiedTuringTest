import os
import logging
from pathlib import Path
import gymnasium as gym

# release version
from ._version import __version__

# set up logging
logging.basicConfig(format="[%(name)s] %(levelname)s:  %(message)s", level=logging.INFO)
logger = logging.getLogger("nett")

# Alter permissions for ml-agents binaries which are shared between users
for tmp_dir in [
    "/tmp/ml-agents-binaries",
    "/tmp/ml-agents-binaries/binaries",
    "/tmp/ml-agents-binaries/tmp",
]:
    # Check if directory exists
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, mode=0o777)
    # check if directory is correct permission
    elif os.stat(tmp_dir).st_mode % 0o1000 != 0o777:
        # check if directory is owned by user
        if os.stat(tmp_dir).st_uid == os.getuid():
            # change permission of directory
            os.chmod(tmp_dir, 0o1777)
        else:
            logger.error(
                f"Error: '{tmp_dir}' does not have correct permissions and cannot be changed. If you have superuser access, you can run the following command to change the permissions: 'sudo chmod 1777 {tmp_dir}'. Otherwise, request {os.stat(tmp_dir).st_uid} to run 'chmod 1777 {tmp_dir}'."
            )

# Simplify Imports

from .environment.utils import get_experiment_design
from .body.utils import wrapper_list
from .brain.utils import (
    algorithms_list,
    encoders_list,
    policies_list,
    rewards_list,
)


def list_conditions(executable_dir: str | Path) -> list[str]:
    """
    Lists the possible imprinting conditions for the experiment.

    Args:
        executable_dir (str | Path): The path to the Unity executable directory.

    Returns:
        list[str]: A list of imprinting conditions for the experiment.
    """
    return get_experiment_design(Path(executable_dir))[1]


def list_wrappers() -> list[type[gym.Wrapper]]:
    """
    List all available wrappers.

    Returns:
        list[type[gym.Wrapper]]: A list of all available wrappers.
    """
    return wrapper_list


def list_algorithms() -> list[str]:
    """
    List all available algorithms.

    Returns:
        list[str]: A list of all available algorithms.
    """
    return algorithms_list


def list_encoders() -> list[str]:
    """
    List all available encoders.

    Returns:
        list[str]: A list of all available encoders.
    """
    return encoders_list


def list_policies() -> list[str]:
    """
    List all available policies.

    Returns:
        list[str]: A list of all available policies.
    """
    return policies_list


def list_rewards() -> list[str]:
    """
    List all available rewards.

    Returns:
        list[str]: A list of all available rewards.
    """
    return rewards_list


from .nett import NETT

from .analysis import analyze, map_trajectories

__all__ = [
    "NETT",
    "analyze",
    "map_trajectories",
    "feature_visualization",
    "generate_tSNEs",
    "list_algorithms",
    "list_conditions",
    "list_encoders",
    "list_policies",
    "list_rewards",
    "list_wrappers",
]
