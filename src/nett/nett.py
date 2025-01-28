"""
This module contains the NETT class, which is the main class for training, testing and analyzing brains in environments.

.. module:: nett
   :synopsis: Main class for training, testing and analyzing brains in environments.
"""

import logging
from pathlib import Path
from typing import Optional
import yaml
import shutil
from concurrent.futures import Future, wait as future_wait

from .brain.brain import Brain
from .body.body import Body
from .environment.environment import Environment
from .utils.executor import Executor
from .utils.tasklist import TaskList
from .utils.validate import validate_conditions, validate_mode
from .utils.design import get_experiment_design
from .utils.task import Task
from .utils.memory import MemoryManager


JobTooBigError = ValueError(
    "No jobs could be scheduled. Job size too large for GPUs. Consider setting job_memory to a value less than or equal to total free GPU memory."
)


class NETT:
    """
    The NETT class is the main class for training, testing, and analyzing brains in environments. It provides an interface for running the training and testing of the brains in the environment. A configuration is needed prior to running the benchmark. The configuration can be provided as a dictionary or as a path to a YAML file containing the configuration.

    A NETT configuration consists of the following sections:
    - Brain: Defines the agent brain, such as the policy and encoder networks and the training/testing parameters. A valid parameters are those for :func:`~nett.brain.brain`
    - Body: Defines the agent body.
    - Environment: Defines the Unity environment.
    - Run: Contains the run settings.

    Args:
        config

    Examples:
        >>> from nett import NETT
        >>> NETT('./config.yaml')

        >>> from nett import NETT
        >>> benchmark = NETT(config={
        >>>     "Environment": {
        >>>         "executable_path": "path/to/executable.x86_64",
                    "record_eps": {"train": 10, "test": 10}
                }
        >>> })
        >>>
        >>> benchmark.update({
        >>>     "Brain": {
        >>>         "policy": "CnnPolicy",
        >>>         "algorithm": "PPO",
        >>>         "encoder": "small",
        >>>         "reward": "supervised"
        >>>     },
        >>>     "Body": {
        >>>         "wrappers": ["dvs"],
        >>>         "record_eps": {"train": 10, "test": 10}
        >>>     }
        >>> })
        >>> # run the benchmark
        >>> benchmark.run(output_dir="path/to/output/directory", num_brains=2, mode="full")
    """

    logger: logging.Logger
    config: dict
    logger = logging.getLogger("nett.NETT")

    def __init__(self, config: Path | str | dict) -> None:
        """
        Initialize the NETT class.
        """

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
        num_threads: Optional[int] = None,
        devices: Optional[list[int]] = None,
        task_memory: str | int = 4,
        verbose: int = True,
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
        input_params = {k: v for k, v in locals().items() if k != 'self'}

        ########## Initialization ##########

        if not Brain.initialized:
            Brain.initialize(**self.config.get("Brain", {}))
        if not Body.initialized:
            Body.initialize(**self.config.get("Body", {}))
        if not Environment.initialized:
            Environment.initialize(**self.config["Environment"])

        ############ Validation ############

        # get experiment design
        num_test_conditions, valid_imprinting_conditions = get_experiment_design(
            Environment.executable_path
        )

        # validate conditions
        self.conditions = validate_conditions(valid_imprinting_conditions, conditions)

        # validate mode
        modes = validate_mode(mode)

        ############## Setup ###############

        # set up the output directory
        self.output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Set up output directory at: {output_dir.resolve()}")

        # save a copy of the config
        with open(Path.joinpath(self.output_dir, "config.yaml"), "w") as f:
            f.write(yaml.dump({
                "Brain": Brain.input_params,
                "Body": Body.input_params,
                "Environment": Environment.input_params,
                "Run": input_params
            }))

        # calculate run info for Brain
        Brain.calc_run_info(num_threads, num_brains, num_test_conditions, conditions)

        # adjust environment to agent settings
        Environment.adjust_to_agent(
            steps_per_episode=Brain.steps_per_episode,
            supervised_reward=Brain.supervised,
            multiobs=Body.multiobs,
        )

        ############### Run ################

        # run tasks
        self.logger.info("Launching...")

        # initialize executor
        self.executor = Executor(verbose)

        # initialize NVIDIA memory management
        self.memory_manager = MemoryManager()

        # validate devices
        self.devices: list[int] = self.memory_manager.validate_devices(devices)
        self.logger.info(f"Devices that will be used: {devices}")

        # estimate memory for a single task
        self._calculate_task_memory(task_memory)

        # run tasks
        for mode in modes:
            # initialize task sheet
            self.task_sheet: dict[Future, int] = {}

            # get the free memory status for each device
            free_device_memory: list[dict[str, int]] = [
                {
                    "device": device,
                    "memory": self.memory_manager.get_free_memory(device),
                }
                for device in self.devices
            ]

            # assign devices based on memory availability
            try:
                # assign tasks to devices and run them
                for task in TaskList(num_brains, conditions, output_dir, mode):
                    self._assign_task(task, free_device_memory)

                future_wait(self.task_sheet, return_when="ALL_COMPLETED")

            except Exception as e:
                self.logger.exception(f"Error in launching jobs: {e}")
                raise e
            finally:
                self._close()

    def _close(self) -> None:
        # close memory manager
        self.memory_manager.close()
        # close processes and free up resources on completion
        self.logger.info("Shutting down executor")
        self.executor.close()  # TODO: does future wait and this both need to be here?

    def _waitlist(self, task):
        # wait until there is GPU space to run task
        self.logger.warning(
            "Insufficient GPU Memory. Waiting for running tasks to complete."
        )
        done, _ = future_wait(self.task_sheet, return_when="FIRST_COMPLETED")
        done_future = done[0]
        free_device: int = self.task_sheet.pop(done_future)
        task_future: Future = self.executor.submit(task, free_device)
        self.task_sheet[task_future] = free_device

    def _assign_task(self, task: Task, free_device_memory: list[dict[str, int]]):
        # waitlist remaining tasks if no free memory
        if not free_device_memory:
            self._waitlist(task)
        # remove devices without enough remaining memory
        elif free_device_memory[-1]["memory"] < self.job_memory:
            free_device_memory.pop()
        # run the task
        else:
            # create task
            device = free_device_memory[-1]["device"]
            task_future = self.executor.submit(task, free_device_memory[-1]["device"])
            self.task_sheet[task_future] = device
            # allocate memory
            free_device_memory[-1]["memory"] -= self.job_memory
            # rotate devices
            free_device_memory = [free_device_memory[-1]] + free_device_memory[:-1]

    def _calculate_task_memory(self, job_memory: str | int) -> None:
        most_free_gpu, gpu_max_capacity = self.memory_manager.get_most_free_gpu(
            self.devices
        )

        if job_memory == "auto":
            self.logger.info("Estimating memory for a single task")
            # calculate current memory usage for baseline for comparison

            try:
                # create a test task to estimate memory
                # TODO: Allow mem estimation to accurately estimate for test
                task = Task(
                    "train",
                    0,
                    self.conditions[0],
                    self.output_dir,
                    estimate_memory=True,
                )
                task_future = self.executor.submit(task, most_free_gpu)
                future_wait({task_future: task}, return_when="ALL_COMPLETED")

                with open(task.path / "mem.txt", "r") as file:
                    post_memory: int = int(file.readline())
            except Exception as e:
                self.logger.exception(f"Error in estimating memory: {e}")
                raise e
            finally:
                if task.path.exists():
                    shutil.rmtree(task.path)

            # estimate memory allocated
            # TODO: Mem size can be larger than GPU allows but not big enough to cause problems when running it
            self.job_memory = gpu_max_capacity - post_memory
        else:
            self.job_memory = job_memory * (1024**3)
            # check to see if GPUs can run a single job
            if self.job_memory > gpu_max_capacity:
                raise JobTooBigError

    #####################

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
