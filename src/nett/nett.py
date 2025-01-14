"""
This module contains the NETT class, which is the main class for training, testing and analyzing brains in environments.

.. module:: nett
   :synopsis: Main class for training, testing and analyzing brains in environments.
"""

import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import Future

import yaml
from .brain.brain import Brain
from .body.body import Body
from .environment.environment import Environment
from .utils.tasklist import TaskList
from .utils.task_manager import TaskManager
from .utils.mode import validate_mode


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


def validate_executable_path(executable_path: str) -> str:
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


def get_experiment_design(executable_path: Path) -> tuple[int, list[str]]:
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

    # read the yaml file
    with open(yaml_file, "r") as file:
        yaml_data = yaml.safe_load(file)

    try:
        num_test_conditions: int = yaml_data["num_test_conditions"]
        valid_imprinting_conditions: list[str] = yaml_data["imprinting_conditions"]
    except KeyError:
        raise KeyError(
            "Experiment configuration file is not properly formatted. It should contain 'num_test_conditions' and 'imprinting_conditions' keys."
        )

    return num_test_conditions, valid_imprinting_conditions


class NETT:
    """
    The NETT class is the main class for training, testing, and analyzing brains in environments.

    Args:
        config

    Example:
        >>> from nett import NETT
        >>> # create a brain, body, and environment
        >>> benchmarks = NETT(brain, body, environment)
    """

    def __init__(
        self, config: Path | str | list[Path | str] | dict | list[dict] = None
    ) -> None:
        """
        Initialize the NETT class.
        """
        # initialize logger
        self.logger = logging.getLogger("nett.NETT")

        try:
            if isinstance(config, list):
                raise NotImplementedError(
                    "Multiple config files are not supported yet."
                )
            elif isinstance(config, (str, Path)):
                with open(config, "r") as file:
                    config_text = yaml.safe_load(file)

            self.brain_config = config_text.get("Brain", {})
            self.body_config = config_text.get("Body", {})
            self.environment_config = config_text.get("Environment", {})

            run_config = config_text.get("Run", {})
            self.run(**run_config)
        except Exception as e:
            self.logger.exception("Error in loading config")
            raise e

    def run(
        self,
        output_dir: Path | str,
        mode: str = "full",
        conditions: Optional[list[str]] = None,
        devices: Optional[list[int]] = None,
        task_memory: str | int = 4,
        verbose: int = True,
        synchronous: bool = False,
    ) -> list[Future]:
        """
        Run the training and testing of the brains in the environment.

        Args:
            output_dir (Path | str): The directory where the run results will be stored.
            num_brains (int, optional): The number of brains to be trained and tested. Defaults to 1.
            mode (str, optional): The mode in which the brains are to be trained and tested. It can be "train", "test", or "full". Defaults to "full".
            train_eps (int, optional): The number of episodes the brains are to be trained for. Defaults to 1000.
            test_eps (int, optional): The number of episodes the brains are to be tested for. Defaults to 20.
            batch_mode (bool, optional): Whether to run in batch mode, which will not display Unity windows. Good for headless servers. Defaults to True.
            devices (list[int], optional): The list of devices to be used for training and testing. If None, all available devices will be used. Defaults to None.
            job_memory (int, optional): The memory allocated, in Gigabytes, for a single job. Defaults to 4.
            steps_per_episode (int, optional): The number of steps per episode. Defaults to 1000.
            verbose (int, optional): Whether or not to print info statements. Defaults to True.
            synchronous (bool, optional): Whether to keep code running in the foreground until completion. Defaults to False.
            save_checkpoints (bool, optional): Whether to save checkpoints during training. Defaults to False.
            checkpoint_freq (int, optional): The frequency at which checkpoints are saved. Defaults to 30_000.
            record_training (list[str], optional): The list of what record options to use for training. Can include "agent" for recording the agent's view, "chamber" for recording the top-down view of the chamber, and "state" for recording the observations, actions, and states.
            record_testing (list[str], optional): The list of what record options to use for training. Can include "agent" for recording the agent's view, "chamber" for recording the top-down view of the chamber, and "state" for recording the observations, actions, and states.
            recording_eps (int, optional): Number of episodes to record for. Defaults to 10.

        Returns:
            list[Future]: A list of futures representing the jobs that have been launched.

        Example:
            >>> task_sheet = benchmarks.run(output_dir="./test_run", num_brains=2, train_eps=100, test_eps=10) # benchmarks is an instance of NETT
        """

        Body.initialize(**self.body_config)

        ## Input Validation ##

        # validate executable path
        executable_path: Path = validate_executable_path(
            self.environment_config["executable_path"]
        )

        # get experiment design
        num_test_conditions, valid_imprinting_conditions = get_experiment_design(
            executable_path
        )

        # validate conditions
        conditions = validate_conditions(valid_imprinting_conditions, conditions)

        # validate mode
        modes = validate_mode(mode)


        Brain.initialize(
            num_test_conditions=num_test_conditions,
            num_imprinting_conditions=len(conditions),
            **self.brain_config,
        )

        Environment.initialize(
            steps_per_episode=Brain.steps_per_episode,
            supervised_reward=Brain.supervised,
            multiobs=Body.multiobs,
            **self.environment_config,
        )

        # set up the output_dir (wherever the user specifies, REQUIRED, NO DEFAULT)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Set up run directory at: {output_dir.resolve()}")

        task_manager = TaskManager(devices, verbose)

        tasklist = TaskList(Brain.num_brains, conditions, output_dir)

        self.logger.info("Launching")

        task_manager.run(modes, tasklist, task_memory, synchronous)
