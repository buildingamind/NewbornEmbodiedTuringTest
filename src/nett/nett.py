"""
This module contains the NETT class, which is the main class for training, testing and analyzing brains in environments.

.. module:: nett
    :synopsis: Main class for training, testing and analyzing brains in environments.
"""

import logging
import json
import os
from pathlib import Path
import time
from typing import Optional
import yaml
import shutil
from concurrent.futures import Future, as_completed, wait as future_wait
from nett.utils.tasklist import validate_tasklist

from .brain import Brain
from .body import Body
from .environment import Environment
from .utils import (
    Executor,
    LoadingBarQueue,
    TaskList,
    Task,
    TaskConfig,
    run_task,
    MemoryManager,
    validate_config,
)


JobTooBigError = ValueError(
    "No jobs could be scheduled. Job size too large for GPUs. Consider setting job_memory to a value less than or equal to total free GPU memory."
)


class NETT:
    """
    The NETT class is the main class for training, testing, and analyzing brains in environments. It provides an interface for running the training and testing of the brains in the environment. A configuration is needed prior to running the benchmark. The configuration can be provided as a dictionary or as a path to a JSON or YAML file containing the configuration.

    Args:
        configs (list[Path | str | dict]): The configurations for each benchmark. Each member can be a path to a JSON or YAML file or a dictionary. The configuration should match the arguments for :func:`~nett.nett.NETT.single_run`.

    Example:
        >>> from nett import NETT
        >>>
        >>> experiment1_config = {
        >>>     "name": "Experiment1",
        >>>     "episodes": {
        >>>         "train": 5000,
        >>>         "test": 20
        >>>     }
        >>>     "steps_per_episode": 200,
        >>>     "num_brains": 5,
        >>>     "brain": {
        >>>         "policy": "CnnPolicy",
        >>>         "algorithm": "PPO",
        >>>         "encoder": "small",
        >>>         "reward": "closeness"
        >>>     },
        >>>     "body": {
        >>>         "wrappers": ["dvs"],
        >>>         "record_eps": {"train": 10, "test": 10}
        >>>     }
        >>>     "environment": {
        >>>         "executable_path": "path/to/executable.x86_64",
        >>>         "record_eps": {"train": 10, "test": 10}
        >>>     }
        >>> })
        >>>
        >>> experiment2_config = './experiment2_config.json'
        >>> experiment3_config = './experiment3_config.yaml'
        >>>
        >>> # run the benchmark
        >>> benchmark = NETT(config=[experiment1_config, experiment2_config, experiment3_config])
        >>> benchmark.run(output_dir="path/to/output/directory", devices=[0,1,2], num_threads=32)
    """

    output_path: Path
    num_threads: int
    devices: list[int]
    task_sheet: dict[Future, TaskConfig]
    waitlist: list[Task]
    memory_manager: MemoryManager
    loading_bar: LoadingBarQueue
    free_device_memory: dict[int, float]
    configs: list[dict] = []
    logger: logging.Logger = logging.getLogger("nett.NETT")

    def __init__(self, configs: list[Path | str | dict]) -> None:
        """Initialize the NETT class."""

        try:
            with open(Path(__file__).resolve().parent / "schema.json", "r") as file:
                schema: dict = json.load(file)

            self.configs = [validate_config(conf, schema) for conf in configs]
        except Exception as e:
            self.logger.exception("Error in loading config")
            raise e

    def run(
        self,
        output_path: Path | str = ".",
        devices: Optional[list[int]] = None,
        num_threads: Optional[int] = None,
        verbose: int = True,
        asynchronous: bool = False,
    ) -> list[Future]:
        """
        Run the training and testing of the brains in the environment.

        Args:
            output_path (Path | str, optional): The directory where the run results will be stored. Defaults to `"."`.
            devices (list[int], optional): The list of the indices of CUDA GPUs to be used for training and testing. If None, all available devices will be used. Defaults to `None`.
            num_threads (int, optional): The number of threads to run in parallel for testing. Defaults to `None`. If None, the number of threads is equal to the number of cpu cores.
            verbose (int, optional): Whether or not to print info statements. Defaults to `True`.

        Returns:
            list[Future]: A list of futures representing the jobs that have been launched.

        Example:
            >>> task_sheet = benchmarks.run(output_dir="./test_run", devices=[0,1,2], num_threads=32, verbose=True) # benchmarks is an instance of NETT
        """
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
        self.task_sheet = {}
        self.waitlist = []

        # initialize NVIDIA memory management
        # self.memory_manager = MemoryManager()
        with MemoryManager() as self.memory_manager:

            # validate devices
            self.devices = self.memory_manager.validate_devices(devices)
            self.logger.info(f"Devices that will be used: {self.devices}")

            # get the free memory status for each device
            self.free_device_memory = {
                device: self.memory_manager.get_free_memory(device)
                for device in self.devices
            }

            # initialize executor
            with Executor(verbose) as (self.executor, self.loading_bar_queue):
                # run tasks
                self.logger.info("Launching...")
                try:
                    for config in self.configs:
                        self.single_run(**config)

                    # if asynchronous:
                    #     # TODO: Change to a thread?
                    #     self.waiter = Process(target=self.task_waiter).start()
                    # else:
                    self.task_waiter()

                except ConnectionResetError as e:
                    # TODO: Fix this to appear at the end of a run
                    self.logger.error(
                        "Failed to create SubprocVecEnv. Please ensure your script includes `if __name__ == '__main__':` (see LINK)"
                    )
                    raise e
                except Exception as e:
                    self.logger.exception(f"Error in launching tasks: {e}")
                    raise e

            # TODO: Add an analysis stage to the run

    def single_run(
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
        Non-public function for running a single benchmark. The parameters here should match the top-level parameters of a config file input into :func:`~nett.nett.NETT.run`.

        Args:
            name (str): The name of the run. This will be used to create a directory with the same name in the output path.
            environment (dict): The environment configuration. See :func:`~nett.environment` for valid parameters.
            body (dict): The body configuration. Defaults to `{}`. See :func:`~nett.body` for valid parameters.
            brain (dict): The brain configuration. Defaults to `{}`. See :func:`~nett.brain` for valid parameters.
            episodes (dict[str, int]): The number of episodes the brains are to be trained and tested for. Defaults to `{"train": 5000, "test": 100}`.
            steps_per_episode (int, optional): The number of steps per episode. Defaults to `200`.
            num_brains (int): The number of brains to be trained and tested. Defaults to `1`.
            task_memory (str | int, optional): The memory allocated, in Gigabytes, for a single job. Defaults to `"auto"`.

        Returns:
            list[Future]: A list of futures representing the jobs that have been launched.

        """

        ############ Validation ############

        # check if episodes contains train and/or test only
        if len(episodes) == 0 or not set(episodes.keys()).issubset({"train", "test"}):
            raise ValueError(
                "Episodes should be a dictionary with keys 'train' and/or 'test'"
            )

        ########### Record Input ###########
        input_params = {
            k: v for k, v in locals().items() if k not in {"self", "kwargs"}
        }

        # set up the output directory
        output_dir: Path = self.output_path / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # save a copy of the config
        with open(output_dir / "config.yaml", "w") as f:
            f.write(yaml.dump(input_params))

        ########## Initialization ##########

        base_brain = Brain(**brain)

        base_body = Body(**body)

        base_env = Environment(**environment)

        ############## Setup ###############

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
            getattr(brain, "reward", "closeness"),  # TODO Clean this up
            base_body.multiobs,
            base_body.panini_projection,
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
        # assign devices based on memory availability
        modes: list[str] = []
        for mode in episodes.keys():
            if episodes[mode] > 0:
                modes.append(mode)

        # assign tasks to devices and run them
        tasklist = TaskList(
            base_brain,
            base_body,
            base_env,
            num_brains,
            base_env.conditions,
            output_dir,
            modes,
            self.loading_bar_queue,
            memory,
        )

        # create loading bar
        num_steps = (
            steps_per_episode
            * len(tasklist.tasks)
            * (
                episodes.get("train", 0)
                + episodes.get("test", 0) * base_env.num_test_conditions
            )
        )
        self.executor.loading_bar.add(name, num_steps)
        # validate tasks
        if not base_env.multiagent:
            self.logger.info("Validating tasks...")
            task_future: Future = self.executor.submit(validate_tasklist, tasklist)
            future_wait({task_future: ""}, return_when="ALL_COMPLETED")

        self.logger.info(f"Assigning tasks...")
        for task in tasklist:
            time.sleep(1)
            self._assign_task(task)

    def task_waiter(self):
        if len(self.waitlist) > 0:
            self.logger.warning(
                f"Insufficient GPU Memory. Waiting for running tasks to complete. Number of Tasks in Waitlist: {len(self.waitlist)}"
            )

        for done_future in as_completed(self.task_sheet):
            self.logger.info(f"Task Completed: Waitlist Size: {len(self.waitlist)}")
            done_future.result()
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

    def status(self):
        return self.task_sheet

    def _assign_task(self, task: Task) -> None:
        most_free_gpu, gpu_max_capacity = (None, 0)
        for device, memory in self.free_device_memory.items():
            if memory > gpu_max_capacity:
                most_free_gpu = device
                gpu_max_capacity = memory

        # check if enough memory is available on the device
        if gpu_max_capacity >= task.config.memory:
            task.set_device(most_free_gpu)
            task_future = self.executor.submit(run_task, task)
            self.task_sheet[task_future] = task.config
            # allocate memory
            self.free_device_memory[most_free_gpu] -= task.config.memory
        else:
            # waitlist remaining tasks if no free memory
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
                    self.loading_bar_queue,
                )
                task.set_device(most_free_gpu)
                self.executor.loading_bar.add(
                    f"Estimating Memory Usage for {task.config.name}", brain.buffer_size
                )
                task_future: Future = self.executor.submit(run_task, task)
                future_wait({task_future: task.config}, return_when="ALL_COMPLETED")
                self.logger.info("Finished estimating memory")
                self.executor.loading_bar.remove(
                    f"Estimating Memory Usage for {task.config.name}"
                )

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
