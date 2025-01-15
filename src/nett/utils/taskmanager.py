import logging
import os
from pathlib import Path
import shutil
import sys
from typing import Optional

from concurrent.futures import ProcessPoolExecutor, Future, wait as future_wait

from stable_baselines3.common.env_checker import check_env

# from nett.brain import Brain
# from nett.environment import Environment
from ..brain import Brain
from ..environment import Environment
from .task import Task
from .tasklist import TaskList
from .vec_env import MultiEnv, SingleEnv, TestEnv, ZooEnv
from .memory import MemoryManager

JobTooBigError = ValueError(
    "No jobs could be scheduled. Job size too large for GPUs. Consider setting job_memory to a value less than or equal to total free GPU memory."
)


def run_task(task: Task, logger) -> None:
    # run environment
    # can be train or test mode and can be for validation or actual run
    try:
        brain: Brain = Brain(task.device, task.brain_id)

        log_path = task.path / "env_logs"
        log_path.mkdir(exist_ok=True, parents=True)

        if Environment.multiagent:
            with ZooEnv(task) as envs:
                getattr(brain, task.mode)(envs, task)  # brain.train or brain.test
        else:
            # validation run
            validate_env(task)
            if task.mode == "train":
                with SingleEnv(task) as envs:
                    brain.train(envs, task)
            else:
                with MultiEnv(task, brain.n_parallel_envs) as envs:
                    brain.test(envs, task)

        logger.info("Environments Closed")
    except Exception as e:
        logger.exception(f"{task.mode} env failed: {str(e)}")
        raise e


def validate_env(task: Task) -> None:
    try:
        with TestEnv(task) as environment:
            check_env(environment)
    except Exception as e:
        raise RuntimeError(f"{task.mode} env validation failed: {str(e)}")


def _init_executor(verbose: bool) -> None:
    # mute stdout if not verbose
    mute = lambda: setattr(sys, "stdout", open(os.devnull, "w"))
    initializer = mute if not verbose else None

    return ProcessPoolExecutor(
        initializer=initializer,  # TODO: too many workers
    )


class TaskManager:
    def __init__(
        self,
        devices: Optional[list[int]],
        verbose: bool,
    ) -> None:
        # initialize logger
        self.logger = logging.getLogger("nett.TaskManager")

        # initialize executor
        self.executor = _init_executor(verbose)

        # initialize NVIDIA memory management
        self.memory_manager = MemoryManager()

        # validate devices
        self.devices: list[int] = self.memory_manager.validate_devices(devices)
        self.logger.info(f"Devices that will be used: {devices}")

    def run(
        self,
        modes: list[str],
        tasklist: TaskList,
        task_memory: str | int,
        synchronous: bool,
    ) -> None:
        # estimate memory for a single task
        self._calculate_task_memory(task_memory, tasklist)

        # run tasks
        for mode in modes:
            tasks = tasklist(mode)
            self._run_tasks(tasks, synchronous)

    ##########################

    def close(self) -> None:
        # close memory manager
        self.memory_manager.close()
        # close processes and free up resources on completion
        self.logger.info("Shutting down executor")
        self.executor.shutdown()  # TODO: does future wait and this both need to be here?

    def submit_task(self, task: Task, device: int) -> None:
        task.device = device
        task_future = self.executor.submit(run_task, task, self.logger)
        self.task_sheet[task_future] = device

    def waitlist(self, task):
        # wait until there is GPU space to run task
        self.logger.warning(
            "Insufficient GPU Memory. Waiting for running tasks to complete."
        )
        done, _ = future_wait(self.task_sheet, return_when="FIRST_COMPLETED")
        done_future = done[0]
        free_device: int = self.task_sheet.pop(done_future)
        self.submit_task(task, free_device)

    def _assign_task(self, task: Task, free_device_memory: list[dict[str, int]]):
        # waitlist remaining tasks if no free memory
        if not free_device_memory:
            self.waitlist(task)
        # remove devices without enough remaining memory
        elif free_device_memory[-1]["memory"] < self.job_memory:
            free_device_memory.pop()
        # run the task
        else:
            # create task
            self.submit_task(task, free_device_memory[-1]["device"])
            # allocate memory
            free_device_memory[-1]["memory"] -= self.job_memory
            # rotate devices
            free_device_memory = [free_device_memory[-1]] + free_device_memory[:-1]

    def _calculate_task_memory(self, job_memory: str | int, tasklist: TaskList) -> None:
        most_free_gpu: int = self.memory_manager.get_most_free_gpu(self.devices)
        gpu_max_capacity: int = self.memory_manager.get_free_memory(most_free_gpu)

        if job_memory == "auto":
            self.job_memory = self._estimate_task_memory(
                tasklist.conditions[0],
                tasklist.output_dir,
                most_free_gpu,
                gpu_max_capacity,
            )  # TODO Clean this up
        else:
            self.job_memory = job_memory * (1024**3)
            # check to see if GPUs can run a single job
            if self.job_memory > gpu_max_capacity:
                raise JobTooBigError

    def _estimate_task_memory(
        self,
        example_condition: str,
        output_dir: Path,
        most_free_gpu: int,
        pre_memory: int,
    ) -> int:
        self.logger.info("Estimating memory for a single task")
        # calculate current memory usage for baseline for comparison

        try:
            # create a test task to estimate memory
            # TODO: Allow mem estimation to accurately estimate for test
            task = Task("train", 0, example_condition, output_dir, estimate_memory=True)
            task.device = most_free_gpu
            task_future = self.executor.submit(run_task, task, self.logger)
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
        return post_memory - pre_memory

    def _run_tasks(self, tasks: TaskList, synchronous: bool) -> None:
        self.task_sheet: dict[Future, int] = {}

        # get the free memory status for each device
        free_device_memory: list[dict[str, int]] = (
            self.memory_manager.get_free_memory_by_device(self.devices)
        )

        # assign devices based on memory availability
        try:
            for task in tasks:
                self._assign_task(task, free_device_memory)

            if synchronous:
                future_wait(self.task_sheet, return_when="ALL_COMPLETED")

        except Exception as e:
            self.logger.exception(f"Error in launching jobs: {e}")
            raise e
        finally:
            self.close()
