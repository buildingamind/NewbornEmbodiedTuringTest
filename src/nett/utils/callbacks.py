"""
Callbacks for training the agents.

Classes:
    HParamCallback(BaseCallback)
"""
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.logger import HParam

from nett.utils.train import compute_train_performance

from pynvml import nvmlInitWithFlags, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetComputeRunningProcesses, nvmlDeviceGetComputeRunningProcesses_v3, nvmlDeviceGetUtilizationRates
import os

# TODO (v0.4): refactor needed, especially logging
class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "batch_size": self.model.batch_size,
            "n_steps": self.model.n_steps
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

class multiBarCallback(ProgressBarCallback):
    """
    Display a progress bar when training SB3 agent
    using tqdm and rich packages.
    """

    def __init__(self, index) -> None: #, num_steps
        super().__init__()
        self.index = index

    def _on_training_start(self) -> None:
        # Initialize progress bar
        # Remove timesteps that were done in previous training sessions
        self.pbar = tqdm(total=self.model.n_steps, position=self.index)
        # self.pbar = tqdm(total=self.locals["total_timesteps"] - self.model.num_timesteps, position=self.index)

class MemoryCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        handle = nvmlDeviceGetHandleByIndex(7)
        print("MEMORY INFO 0: ", nvmlDeviceGetMemoryInfo(handle))
        print("PROCESSES 0: ", nvmlDeviceGetComputeRunningProcesses_v3(handle))
        

        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # Create the directory if it doesn't exist
        os.makedirs("./.tmp", exist_ok=True)
        used_memory = nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(0)).used
        # Write the used memory to a file
        with open("./.tmp/memory_use", "w") as f:
            f.write(str(used_memory))
        # Rest of the code...
        # self.logger.info("SNAPSHOT: ", torch.cuda.memory_snapshot())
        # self.logger.info("STATS: ", torch.cuda.memory_stats(device=None))
        pass

    def _on_step(self) -> bool:
        print('STEP!!!!!!')
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        return False

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass