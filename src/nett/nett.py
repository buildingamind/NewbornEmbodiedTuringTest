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
from .utils.condition import validate_conditions
from .utils.tasklist import TaskList
from .utils.taskmanager import TaskManager
from .utils.mode import validate_mode
from .utils.design import get_experiment_design


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

    def __init__(self, config: Path | str | dict) -> None:
        """
        Initialize the NETT class.
        """
        # initialize logger
        self.logger = logging.getLogger("nett.NETT")

        self.config: dict

        try:
            if isinstance(config, dict):
                self.config = config
            elif isinstance(config, (str, Path)):
                with open(config, "r") as file:
                    self.config = yaml.safe_load(file)
        except Exception as e:
            self.logger.exception("Error in loading config")
            raise e

        if "Run" in self.config:
            self.run(**self.config["Run"])

    def run(
        self,
        output_dir: Path | str,
        mode: str = "full",
        conditions: Optional[list[str]] = None,
        num_brains: int = 1,
        devices: Optional[list[int]] = None,
        task_memory: str | int = 4,
        verbose: int = True,
        synchronous: bool = True,
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

        ## Initialization ##

        Brain.initialize(**self.config.get("Brain", {}))
        Body.initialize(**self.config.get("Body", {}))
        Environment.initialize(**self.config["Environment"])

        ## Validation ##

        # get experiment design
        num_test_conditions, valid_imprinting_conditions = get_experiment_design(
            Environment.executable_path
        )

        # validate conditions
        conditions = validate_conditions(valid_imprinting_conditions, conditions)

        # validate mode
        modes = validate_mode(mode)

        ## Setup ##

        # set up the output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Set up output directory at: {output_dir.resolve()}")

        # calculate run info for Brain
        Brain.calc_run_info(num_brains, num_test_conditions, conditions)

        # adjust environment to agent settings
        Environment.adjust_to_agent(
            steps_per_episode=Brain.steps_per_episode,
            supervised_reward=Brain.supervised,
            multiobs=Body.multiobs,
        )

        ## Run ##

        # create task manager
        task_manager = TaskManager(devices, verbose)

        # create task list
        tasklist = TaskList(num_brains, conditions, output_dir)

        # run tasks
        self.logger.info("Launching...")
        task_manager.run(modes, tasklist, task_memory, synchronous)

    def update(self, supplementary_config: Path | str | dict):
        """
        Update the current configuration with a supplementary configuration.

        Args:
            supplementary_config (Path | str | dict): The supplementary configuration to update the current configuration with.

        Example:
            >>> benchmarks.update(supplementary_config="./supplementary_config.yaml")
        """
        try:
            if isinstance(supplementary_config, dict):
                supplementary_config = supplementary_config
            elif isinstance(supplementary_config, (str, Path)):
                with open(supplementary_config, "r") as file:
                    supplementary_config = yaml.safe_load(file)
        except Exception as e:
            self.logger.exception("Error in loading supplementary config")
            raise e

        self.config.update(supplementary_config)
        self.logger.info(
            "Extended the current configuration with the supplementary configuration."
        )
