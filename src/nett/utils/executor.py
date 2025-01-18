
from concurrent.futures import ProcessPoolExecutor, Future
import logging
import os
import sys

from nett.brain.brain import Brain
from nett.environment.environment import Environment
from nett.nett import _validate_env
from nett.utils.task import Task
from nett.utils.vec_env import MultiEnv, SingleEnv, TestEnv, ZooEnv

from stable_baselines3.common.env_checker import check_env

def _validate_env(task: Task) -> None:
    try:
        with TestEnv(task) as environment:
            check_env(environment)
    except Exception as e:
        raise RuntimeError(f"{task.mode} env validation failed: {str(e)}")


class Executor:

  def __init__(self, verbose: bool) -> None:
      self.logger = logging.getLogger('nett.executor')
      # mute stdout if not verbose
      mute = lambda: setattr(sys, "stdout", open(os.devnull, "w"))
      initializer = mute if not verbose else None

      self.executor = ProcessPoolExecutor(
          initializer=initializer,  # TODO: too many workers
      )

  def _run_task(self, task: Task) -> None:
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
              _validate_env(task)
              if task.mode == "train":
                  with SingleEnv(task) as envs:
                      brain.train(envs, task)
              else:
                  with MultiEnv(task, brain.n_parallel_envs) as envs:
                      brain.test(envs, task)

          self.logger.info("Environments Closed")
      except Exception as e:
          self.logger.exception(f"{task.mode} env failed: {str(e)}")
          raise e

  def submit(self, task: Task, device: int) -> Future:
      task.device = device
      task_future = self.executor.submit(self._run_task, task, self.logger)
      return task_future

  def close(self) -> None:
      self.executor.shutdown()