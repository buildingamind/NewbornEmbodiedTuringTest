import os
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
from .vec_env import MultiEnv, SingleEnv, ZooEnv
from .memory import MemoryManager

from nett import logger

JobTooBigError = ValueError(
    "No jobs could be scheduled. Job size too large for GPUs. Consider setting job_memory to a value less than or equal to total free GPU memory."
)


class TaskManager:
    def __init__(
        self,
        modes: list[str],
        tasklist: TaskList,
        devices: Optional[list[int]],
        job_memory: str | int,
        verbose: bool,
        synchronous: bool,
    ) -> None:
        # initialize logger
        self.logger = logger.getChild(__class__.__name__)

        # mute stdout if not verbose
        mute = lambda: setattr(sys, "stdout", open(os.devnull, "w"))
        initializer = mute if not verbose else None

        # for NVIDIA memory management
        self.memory_manager = MemoryManager()

        # validate devices
        self.devices: list[int] = self.memory_manager.validate_devices(devices)
        self.logger.info(f"Devices that will be used: {devices}")

        self.executor = ProcessPoolExecutor(
            initializer=initializer,  # TODO: too many workers
        )

        if job_memory == "auto":
            self.job_memory = self._estimate_job_memory(
                tasklist.conditions[0]
            )  # TODO Clean this up
        else:
            self.job_memory = job_memory * (1024**3)

        # check to see if GPUs can run a single job
        most_free_gpu = self.memory_manager.get_most_free_gpu(self.devices)
        gpu_max_capacity = self.memory_manager.get_free_memory(most_free_gpu)
        if self.job_memory > gpu_max_capacity:
            raise JobTooBigError

        for mode in modes:
            tasks = tasklist(mode)

            self.task_sheet: dict[Future, int] = {}

            # get the free memory status for each device
            free_device_memory: list[dict[str, int]] = (
                self.memory_manager.get_free_memory_by_device(self.devices)
            )

            # assign devices based on memory availability
            try:
                for task in tasks:
                    if not free_device_memory:
                        # waitlist task
                        self.waitlist(task)
                    elif free_device_memory[-1]["memory"] < job_memory:
                        free_device_memory.pop()
                    else:
                        # create job
                        self.submitTask(task, free_device_memory[-1]["device"])
                        # allocate memory
                        free_device_memory[-1]["memory"] -= job_memory
                        # rotate devices
                        free_device_memory = [
                            free_device_memory[-1]
                        ] + free_device_memory[:-1]

                if synchronous:
                    future_wait(self.task_sheet.keys(), return_when="ALL_COMPLETED")

            except Exception as e:
                self.logger.exception(f"Error in launching jobs: {e}")
                raise e
            finally:
                # close memory manager
                self.memory_manager.close()
                # close processes and free up resources on completion
                self.logger.info("Shutting down executor")
                self.executor.shutdown()  # TODO: does future wait and this both need to be here?

    def run(self, task: Task) -> None:
        # run environment
        # can be train or test mode and can be for validation or actual run
        try:
            brain: Brain = Brain(task.device, task.brain_id)

            log_path = task.path / "env_logs"
            log_path.mkdir(exist_ok=True)

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

            self.logger.info("Environments Closed")
        except Exception as e:
            self.logger.exception(f"{self.mode} env failed: {str(e)}")
            raise e

    def submitTask(self, task: Task, device: int) -> None:
        task.device = device
        task_future = self.executor.submit(self.run, task)
        self.task_sheet[task_future] = device

    def waitlist(self, task):
        # wait until there is GPU space to run task
        self.logger.warning(
            "Insufficient GPU Memory. Waiting for running tasks to complete."
        )
        done, _ = future_wait(self.task_sheet.keys(), return_when="FIRST_COMPLETED")
        done_future = done[0]
        free_device: int = self.task_sheet.pop(done_future)
        self.submitTask(task, free_device)

    def _estimate_task_memory(self, example_condition: str) -> int:
        self.logger.info("Estimating memory for a single task")
        # calculate current memory usage for baseline for comparison
        # find the GPU with the most free memory
        most_free_gpu: int = self.memory_manager.get_most_free_gpu(self.devices)
        pre_memory: int = self.memory_manager.get_free_memory(self.devices)
        try:
            # create a test task to estimate memory
            # TODO: Allow mem estimation to accurately estimate for test
            task = Task(
                "train", 0, example_condition, self.output_dir, estimate_memory=True
            )
            task.device = most_free_gpu
            task_future = self.executor.submit(self.run, task)
            future_wait(task_future, return_when="ALL_COMPLETED")

            with open(self.path / "mem.txt", "r") as file:
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


def validate_env(task: Task) -> None:
    try:
        with SingleEnv(task, validation_mode=True) as environment:
            check_env(environment)
    except Exception as e:
        raise RuntimeError(f"{task.mode} env validation failed: {str(e)}")
