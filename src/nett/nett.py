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
        num_brains: int = 1,
        mode: str = "full",
        train_eps: int = 1000,
        test_eps: int = 20,
        batch_mode: bool = True,
        devices: Optional[list[int]] = None,
        job_memory: str | int = 4,
        steps_per_episode: int = 1000,
        conditions: Optional[list[str]] = None,
        verbose: int = True,
        synchronous: bool = False,
        record_training: Optional[list[str]] = [],
        record_testing: Optional[list[str]] = [],
        recording_eps: int = 10,
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
        # check if environment should use supervised reward or not. Defaults to True
        supervised_reward: bool = (
            self.brain_config.get("reward", "supervised") == "supervised"
        )

        Body.initialize(**self.body_config)

        Environment.initialize(
            steps_per_episode=steps_per_episode,
            supervised_reward=supervised_reward,
            multiobs=Body.multiobs,
            **self.environment_config,
        )

        conditions = validate_conditions(
            Environment.valid_imprinting_conditions, conditions
        )

        Brain.initialize(
            steps_per_episode=steps_per_episode,
            num_test_conditions=Environment.num_test_conditions,
            num_imprinting_conditions=len(conditions),
            num_brains=num_brains,
            **self.brain_config,
        )

        # set up the output_dir (wherever the user specifies, REQUIRED, NO DEFAULT)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Set up run directory at: {output_dir.resolve()}")

        tasklist = TaskList(num_brains, conditions, output_dir)

        modes = validate_mode(mode)
        self.logger.info("Launching")

        self.task_manager = TaskManager(
            modes, tasklist, devices, job_memory, verbose, synchronous
        )

        # launch jobs

        # task_sheet = self.task_manager.run(job_memory, synchronous)

        # return control back to the user after launching jobs, do not block
        # return task_sheet # should we continue to have this. Needs discussion

    # def status(self, task_sheet: dict[Future, Job]) -> pd.DataFrame:
    #     """
    #     Get the status of the jobs in the job sheet.

    #     Args:
    #         task_sheet (dict[Future, Job]): The job sheet returned by the .launch_jobs() method.

    #     Returns:
    #         pd.DataFrame: A dataframe containing the status of the jobs in the job sheet.

    #     Example:
    #         >>> status = benchmarks.status(task_sheet)
    #         >>> # benchmarks is an instance of NETT, task_sheet is the job sheet returned by the .run() method
    #     """
    #     selected_columns = ["brain_id", "condition", "device"]
    #     filtered_task_sheet = self._filter_task_sheet(task_sheet, selected_columns)
    #     return pd.json_normalize(filtered_task_sheet)

    # @staticmethod
    # def _filter_task_sheet(task_sheet: dict[Future, dict[str,Any]], selected_columns: list[str]) -> list[dict[str,bool|str]]:
    #     # TODO include waitlisted jobs
    #     runStatus = lambda job_future: {'running': job_future.running()}
    #     jobInfo = lambda job: {k: getattr(job, k) for k in selected_columns}

    #     return [runStatus(job_future) | jobInfo(job) for job_future, job in task_sheet.items()]


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
