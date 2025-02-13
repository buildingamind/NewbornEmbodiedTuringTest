"""
This module contains the NETT class, which is the main class for training, testing and analyzing brains in environments.

.. module:: nett
   :synopsis: Main class for training, testing and analyzing brains in environments.
"""

import logging
import json
import os
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
from .utils.task import Task, TaskConfig, run_task
from .utils.memory import MemoryManager
from .utils.validate import validate_config


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
        >>> benchmark = NETT('./config.yaml')
        >>> benchmark.run(output_dir="path/to/output/directory", num_brains=2, mode="full")

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
        >>>
        >>> # run the benchmark
        >>> benchmark.run(output_dir="path/to/output/directory", num_brains=2, mode="full")
    """

    logger: logging.Logger
    configs: list[dict] = []
    logger = logging.getLogger("nett.NETT")

    def __init__(self, config: Path | str | dict | list[Path | str | dict]) -> None:
        """Initialize the NETT class."""

        try:
            if not isinstance(config, list):
                config = [config]

            self.logger.info("Validating configs")
            with open(Path(__file__).resolve().parent / "schema.json", "r") as file:
                schema: dict = json.load(file)

            for config_instance in config:
                valid_config = validate_config(config_instance, schema)
                self.configs.append(valid_config)
        except Exception as e:
            self.logger.exception("Error in loading config")
            raise e

    def run(
        self,
        output_path: Path | str = ".",
        devices: Optional[list[int]] = None,
        num_threads: Optional[int] = None,
        verbose: int = True,
    ) -> list[Future]:
        # get the output directory
        self.output_path = Path(output_path).resolve()
        self.logger.info(f"Set up output directory at: {self.output_path.resolve()}")
        self.num_threads = os.cpu_count() if num_threads is None else num_threads

        test_config_count: int = 0
        for config in self.configs:
            if "episodes" in config:
                test_config_count += bool(config["episodes"].get("test", 0))

        if test_config_count > 0:
            self.num_threads = int(self.num_threads / test_config_count)

        # initialize task sheet
        self.task_sheet: dict[Future, TaskConfig] = {}
        self.waitlist: list[Task] = []

        # initialize NVIDIA memory management
        # self.memory_manager = MemoryManager()
        with MemoryManager() as self.memory_manager:

            # validate devices
            self.devices: list[int] = self.memory_manager.validate_devices(devices)
            self.logger.info(f"Devices that will be used: {self.devices}")

            # get the free memory status for each device
            self.free_device_memory: dict[int, float] = {
                device: self.memory_manager.get_free_memory(device)
                for device in self.devices
            }

            # initialize executor
            # self.executor = Executor(verbose)
            with Executor(verbose) as self.executor:
                # run tasks
                self.logger.info("Launching...")

                try:
                    for config in self.configs:
                        self._single_run(**config)

                    self.task_waiter()
                    future_wait(self.task_sheet, return_when="ALL_COMPLETED")  # mutli
                    for future in self.task_sheet:
                        try:
                            result = future.result()
                        except Exception as e:
                            print("exception = ", e)
                        else:
                            print("result = ", result)
                            print("state=", future._state)
                except Exception as e:
                    self.logger.exception(f"Error in launching jobs: {e}")
                    raise e

    def _single_run(
        self,
        name: str,
        environment: dict,
        body: dict = {},
        brain: dict = {},
        episodes: {str, int} = {"train": 5000, "test": 100},
        steps_per_episode: int = 200,
        num_brains: int = 1,
        task_memory: str | int = "auto",
        **kwargs,
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
        input_params = {k: v for k, v in locals().items() if k != "self"}

        ########## Initialization ##########

        base_brain = Brain(**brain)

        base_body = Body(**body)

        base_env = Environment(**environment)

        ############ Validation ############

        # check if episodes contains train and/or test only
        if len(episodes) == 0 or not set(episodes.keys()).issubset({"train", "test"}):
            raise ValueError(
                "Episodes should be a dictionary with keys 'train' and/or 'test'"
            )

        ############## Setup ###############

        # set up the output directory
        output_dir: Path = self.output_path / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # save a copy of the config
        with open(output_dir / "config.yaml", "w") as f:
            f.write(yaml.dump(input_params))

        # calculate run info for Brain
        base_brain.calc_iterations(
            num_brains,
            self.num_threads,
            len(base_env.conditions),
            base_env.num_test_conditions,
            episodes,
            steps_per_episode,
        )

        # adjust environment to agent settings
        base_env.adjust_to_agent(
            steps_per_episode,
            base_brain.supervised,
            base_body.multiobs,
            getattr(base_brain, "n_parallel_envs", 1),
        )

        ############### Run ################

        # estimate memory for a single task
        memory = self._calculate_task_memory(
            base_brain,
            base_body,
            base_env,
            task_memory,
            base_env.conditions[0],
            output_dir,
        )  # multi

        # run tasks

        self.logger.info("Running tasks...")
        # assign devices based on memory availability
        modes: list[str] = []
        for mode in episodes.keys():
            if episodes[mode] > 0:
                modes.append(mode)

        # assign tasks to devices and run them
        self.logger.info("Creating tasks...")
        tasklist = TaskList(
            base_brain,
            base_body,
            base_env,
            num_brains,
            base_env.conditions,
            output_dir,
            modes,
            memory,
        )

        for task in tasklist:  # multi
            self._assign_task(task)


    """
    waitlist = [(task, memory_use), ...]
    executor.submit (task_waiter)

    def task_waiter():
    memory_available = {device: 0 for device in devices}
    wait task_sheet FIRST_COMPLETED

    UnLock self.task_sheet
    free_device, free_memory = self.task_sheet.pop(done_future)
    memory_available[free_device] += free_memory
    complete = False
    for task, memory_use in waitlist:
        if memory_use <= memory_available[free_device]:
            self.task_sheet[task_future] = (free_device, memory_use)
            memory_available[free_device] -= memory_use
            break

    Lock self.task_sheet
    """

    def task_waiter(self):
        if len(self.waitlist) > 0:
            self.logger.warning(
                "Insufficient GPU Memory. Waiting for running tasks to complete."
            )
        while len(self.waitlist) > 0:
            done, _ = future_wait(self.task_sheet, return_when="FIRST_COMPLETED")
            for done_future in done:
                done_config: TaskConfig = self.task_sheet.pop(done_future)
                free_device: int = done_config.device
                self.free_device_memory[free_device] += done_config.memory

                for i, task in enumerate(self.waitlist):
                    if task.config.memory <= self.free_device_memory[free_device]:
                        self.free_device_memory[free_device] -= task.config.memory
                        task.set_device(free_device)
                        task_future: Future = self.executor.submit(run_task, task)
                        self.task_sheet[task_future] = task.config
                        self.waitlist.pop(i)
                        break

    def _assign_task(self, task: Task) -> None:
        # waitlist remaining tasks if no free memory
        self.logger.info(f"Assigning task...")
        assigned = False
        for device, memory in self.free_device_memory.items():
            # check if enough memory is available on the device
            if memory >= task.config.memory:
                assigned = True
                task.set_device(device)
                task_future = self.executor.submit(run_task, task)
                self.task_sheet[task_future] = task.config
                # allocate memory
                self.free_device_memory[device] -= task.config.memory
                break

        if not assigned:
            self.waitlist.append(task)

    def _calculate_task_memory(
        self,
        brain: Brain,
        body: Body,
        env: Environment,
        task_memory: str | int,
        example_condition: str,
        output_dir: Path,
    ) -> float:
        most_free_gpu, gpu_max_capacity = self.memory_manager.get_most_free_gpu(
            self.devices
        )

        if task_memory == "auto":
            self.logger.info("Estimating memory for a single task")
            # calculate current memory usage for baseline for comparison

            try:
                # create a test task to estimate memory
                # TODO: Allow mem estimation to accurately estimate for test
                task = Task(
                    brain,
                    body,
                    env,
                    0,
                    example_condition,
                    output_dir,
                    ["train"],
                )
                task.set_device(most_free_gpu)
                task_future = self.executor.submit(run_task, task)
                future_wait({task_future: task.config}, return_when="ALL_COMPLETED")

                with open(task.config.path / "mem.txt", "r") as file:
                    post_memory: int = int(file.readline())
            except Exception as e:
                self.logger.exception(f"Error in estimating memory: {e}")
                raise e
            finally:
                if "task" in locals() and task.config.path.exists():
                    shutil.rmtree(task.config.path)

            # estimate memory allocated
            # TODO: Mem size can be larger than GPU allows but not big enough to cause problems when running it
            memory_use = gpu_max_capacity - post_memory
            self.logger.info("Estimated Memory: " + str(memory_use / 1024**3) + " GB")
        else:
            memory_use = task_memory * (1024**3)

        # check to see if GPUs can run a single job
        if memory_use > gpu_max_capacity:
            raise JobTooBigError

        return memory_use

    #####################

    def update(self, supplementary_config: Path | str | dict | list[Path | str | dict]):
        """
        Update the current configuration with a supplementary configuration.

        Args:
            supplementary_config (Path | str | dict): The supplementary configuration to update the current configuration with.

        Example:
            >>> benchmarks.update(supplementary_config="./supplementary_config.yaml")
        """
        try:

            if not isinstance(supplementary_config, list):
                supplementary_config = [supplementary_config]

            for conf in supplementary_config:
                if isinstance(conf, dict):
                    self.configs.append(supplementary_config)
                elif isinstance(supplementary_config, (str, Path)):
                    with open(supplementary_config, "r") as file:
                        self.configs.append(yaml.safe_load(file))
        except Exception as e:
            self.logger.exception("Error in loading supplementary config")
            raise e

        self.logger.info(
            "Extended the current configuration with the supplementary configuration."
        )
