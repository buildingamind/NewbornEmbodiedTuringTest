"""
Callbacks for training the agents.

Classes:
    HParamCallback(BaseCallback)
    SupervisedSaveBestModelCallback(BaseCallback)
"""
from pathlib import Path
from tqdm import tqdm
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback
from stable_baselines3.common.logger import HParam

from nett.utils.train import compute_train_performance

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

class SupervisedSaveBestModelCallback(BaseCallback):
    """
    Callback to save the best model based on the mean performance of the last 100 episodes."""
    def __init__(self, summary_freq: int, save_dir: Path, env_log_path: str) -> None:
        super().__init__(verbose= 1)
        self.summary_freq = summary_freq
        self.save_dir = save_dir
        self.env_log_path = env_log_path
        self.best_mean_performance = -np.inf
        self.best_mean_reward = -np.inf
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> None:
        if self.n_calls % self.summary_freq == 0:
            # Retrieve training reward
            x, y = compute_train_performance(self.env_log_path)
            if len(x) > 0:

                # mean performance for last 100 episodes
                mean_performance  = y[-1]

                x, y = ts2xy(load_results(self.env_log_path), "timesteps")
                if len(x) > 0:
                    # mean reward for last 100 episodes
                    mean_reward  = np.mean(y[-100:])
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(f"Best mean reward: {self.best_mean_reward:.2f}\
                            - Last mean reward per episode: {mean_reward:.2f}")
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward

                if self.verbose > 0:
                    print(f"Best mean performance: {self.best_mean_performance:.2f}\
                        - Last mean performance per episode: {mean_performance:.2f}")
                if mean_performance > self.best_mean_performance:
                    self.best_mean_performance = mean_performance
                    if self.verbose > 0:
                        save_path = f"{self.save_dir.joinpath('best_model.zip')}"
                        print(f"Saving the best model to: {save_path}")
                    self.model.save(save_path)
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