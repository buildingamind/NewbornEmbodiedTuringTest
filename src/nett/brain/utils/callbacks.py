"""
Callbacks for training the agents.

Classes:
    HParamCallback(BaseCallback)
"""

from multiprocessing import SimpleQueue
from pathlib import Path
import sys
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.base_class import BaseAlgorithm

from ...utils.memory import MemoryManager
from ...utils.loading_bar_queue import LoadingBarQueue
import re
import glob
import cv2

# import numpy as np

# from nett.utils.performance import compute_train_performance


def img2video(record_path: Path, expected_length: int, fps: int = 25):
    png_files = glob.glob(str(record_path / "*.png"))
    if not png_files:
        return

    # Group pngs by episode number
    episode_dict = {}
    pattern = re.compile(r"(\d+)_(\d+)\.png$")
    for png in png_files:
        match = pattern.search(png)
        if match:
            ep, frame = int(match.group(1)), int(match.group(2))
            episode_dict.setdefault(ep, []).append((frame, png))

    for ep, frames in episode_dict.items():
        # Sort frames by frame number
        frames_sorted = sorted(frames, key=lambda x: x[0])
        if len(frames_sorted) < expected_length:
            continue
        images = [cv2.imread(f[1]) for f in frames_sorted]
        if not images or images[0] is None:
            continue
        height, width, layers = images[0].shape
        mp4_path = record_path / f"{ep}.mp4"
        out = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for img in images:
            if img is not None:
                out.write(img)
        out.release()
        # Optionally, remove PNGs after conversion
        for _, png_path in frames_sorted:
            try:
                Path(png_path).unlink()
            except Exception:
                pass


# TODO (v0.4): refactor needed, especially logging
class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        lr = self.model.learning_rate
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": lr if isinstance(lr, (int, float)) else str(lr),
            "gamma": getattr(self.model, "gamma", 0.99),
            "batch_size": getattr(self.model, "batch_size", None),
            "n_steps": getattr(self.model, "n_steps", None),
        }
        # Remove None entries (e.g. off-policy algos don't have n_steps)
        hparam_dict = {k: v for k, v in hparam_dict.items() if v is not None}
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


class LoadingBarCallback(BaseCallback):
    """
    Display a progress bar when training SB3 agent using tqdm
    """

    def __init__(self, label: str, queue: SimpleQueue) -> None:
        super().__init__()
        # label to prefix the progress bar
        self.label = label

        # queue to communicate with the loading bar process
        self.bar_queue = queue

    def _on_step(self) -> bool:
        # Update progress bar, we do num_envs steps per call to `env.step()`
        # self.pbar.update(self.training_env.num_envs)
        self.bar_queue.put((self.label, 1))  # self.training_env.num_envs
        return True


class MemoryCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    """

    def __init__(self, device: int, save_path: str) -> None:
        super().__init__()
        self.device = device
        self.save_path = save_path
        self.closeEnv = False
        self.memory_manager = MemoryManager()

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        return not self.closeEnv

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.closeEnv = True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # Grab the memory being used by the GPU
        free_memory = self.memory_manager.get_free_memory(self.device)
        # Write the used memory to a file
        with open(Path.joinpath(self.save_path, "mem.txt"), "w") as f:
            f.write(str(free_memory))


class IntrinsicRewardWithOnPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and on-policy algorithms from SB3.
    """

    def __init__(self, irs, verbose=0):
        super().__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.rollout_buffer  #
        # Set the logger in the intrinsic reward module for tensorboard logging
        if hasattr(self.irs, "set_logger"):
            self.irs.set_logger(self.logger)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        observations = self.locals["obs_tensor"]  #
        device = observations.device  #
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_observations = th.as_tensor(self.locals["new_obs"], device=device)  # ~

        # ===================== watch the interaction ===================== #
        self.irs.watch(
            observations, actions, rewards, dones, dones, next_observations
        )  # ~
        # ===================== watch the interaction ===================== #
        return True

    def _on_rollout_end(self) -> None:  ####################################
        # ===================== compute the intrinsic rewards ===================== #
        # prepare the data samples
        obs = th.as_tensor(self.buffer.observations)
        # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = th.as_tensor(self.locals["new_obs"])
        actions = th.as_tensor(self.buffer.actions)
        rewards = th.as_tensor(self.buffer.rewards)
        dones = th.as_tensor(self.buffer.episode_starts)
        # compute the intrinsic rewards
        intrinsic_rewards = self.irs.compute(
            samples=dict(
                observations=obs,
                actions=actions,
                rewards=rewards,
                terminateds=dones,
                truncateds=dones,
                next_observations=new_obs,
            ),
            sync=True,
        )
        # add the intrinsic rewards to the buffer
        self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # ===================== compute the intrinsic rewards ===================== #


class IntrinsicRewardWithOffPolicyRL(BaseCallback):
    """
    A custom callback for combining RLeXplore and off-policy algorithms from SB3.
    """

    def __init__(self, irs, verbose=0):
        super().__init__(verbose)
        self.irs = irs
        self.buffer = None

    def init_callback(self, model: BaseAlgorithm) -> None:
        super().init_callback(model)
        self.buffer = self.model.replay_buffer  #
        # Set the logger in the intrinsic reward module for tensorboard logging
        if hasattr(self.irs, "set_logger"):
            self.irs.set_logger(self.logger)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        device = self.irs.device  #
        obs = th.as_tensor(self.locals["self"]._last_obs, device=device)  #
        actions = th.as_tensor(self.locals["actions"], device=device)
        rewards = th.as_tensor(self.locals["rewards"], device=device)
        dones = th.as_tensor(self.locals["dones"], device=device)
        next_obs = th.as_tensor(self.locals["new_obs"], device=device)  # ~

        # ===================== watch the interaction ===================== #
        self.irs.watch(obs, actions, rewards, dones, dones, next_obs)  # ~
        # ===================== watch the interaction ===================== #
        ####################################
        # ===================== compute the intrinsic rewards ===================== #
        intrinsic_rewards = self.irs.compute(
            samples={
                "observations": obs.unsqueeze(0),
                "actions": actions.unsqueeze(0),
                "rewards": rewards.unsqueeze(0),
                "terminateds": dones.unsqueeze(0),
                "truncateds": dones.unsqueeze(0),
                "next_observations": next_obs.unsqueeze(0),
            },
            sync=False,
        )
        # ===================== compute the intrinsic rewards ===================== #

        try:
            # add the intrinsic rewards to the original rewards
            self.locals["rewards"] += intrinsic_rewards.cpu().numpy().squeeze()
            # update the intrinsic reward module
            replay_data = self.buffer.sample(batch_size=self.irs.batch_size)
            self.irs.update(
                samples={
                    "observations": th.as_tensor(replay_data.observations)
                    .unsqueeze(1)
                    .to(device),  # (n_steps, n_envs, *obs_shape)
                    "actions": th.as_tensor(replay_data.actions)
                    .unsqueeze(1)
                    .to(device),
                    "rewards": th.as_tensor(replay_data.rewards).to(device),
                    "terminateds": th.as_tensor(replay_data.dones).to(device),
                    "truncateds": th.as_tensor(replay_data.dones).to(device),
                    "next_observations": th.as_tensor(replay_data.next_observations)
                    .unsqueeze(1)
                    .to(device),
                }
            )
        except (RuntimeError, ValueError) as e:
            import logging

            logging.getLogger("nett.Brain").warning(
                f"Failed to update intrinsic rewards: {e}"
            )
        ####################################
        return True

    def _on_rollout_end(self) -> None:
        pass


class PngToMp4Callback(BaseCallback):
    """
    Callback to convert episode PNG frames to MP4 videos after each rollout or episode.
    PNGs must be named as <episode>_<frame>.png.
    """

    def __init__(
        self, record_path: Path, expected_size: int, fps: int = 24, verbose: int = 0
    ):
        super().__init__(verbose)
        if not record_path.exists():
            record_path.mkdir(parents=True)

        self.record_path = record_path
        self.fps = fps
        self.expected_size = expected_size

    def _on_rollout_end(self) -> None:
        self._convert_pngs_to_mp4s()

    def _on_step(self) -> bool:
        # Optionally, you can also call conversion here if you want per-episode conversion
        return True

    def _convert_pngs_to_mp4s(self):
        return img2video(self.record_path, self.expected_size, self.fps)


# Spatial zone constants — must match src/nett/analysis/utils/merge.py
_X_LIMITS = (-33.15, 33.15)
_AGENT_RADIUS = 3.0
_AGENT_LIMITS = (_X_LIMITS[0] + _AGENT_RADIUS, _X_LIMITS[1] - _AGENT_RADIUS)
_ONE_THIRD = (_AGENT_LIMITS[1] - _AGENT_LIMITS[0]) / 3
_BOUNDS = [_AGENT_LIMITS[0] + _ONE_THIRD, _AGENT_LIMITS[1] - _ONE_THIRD]


class EvalScoreCallback(BaseCallback):
    """Reads Unity test logs after each eval, computes percent_correct per
    test.cond, and records the scores to the SB3 TensorBoard logger.

    Intended for use as ``callback_after_eval`` on SB3's ``EvalCallback``,
    so ``_on_step`` fires exactly once after each evaluation completes.
    """

    def __init__(self, eval_log_dir: Path, verbose: int = 0):
        super().__init__(verbose)
        self.eval_log_dir = Path(eval_log_dir)
        self._last_line_counts: dict[str, int] = {}

    def _on_step(self) -> bool:
        import pandas as pd
        import numpy as np
        import logging

        logger = logging.getLogger("nett.EvalScoreCallback")

        log_files = list(self.eval_log_dir.glob("test_*.csv"))
        if not log_files:
            return True

        all_frames = []
        for log_file in log_files:
            key = str(log_file)
            try:
                df = pd.read_csv(log_file, skipinitialspace=True, on_bad_lines="skip")
            except Exception:
                continue
            if df.empty:
                continue

            # Only process rows added since the last eval
            prev_count = self._last_line_counts.get(key, 0)
            self._last_line_counts[key] = len(df)
            if len(df) <= prev_count:
                continue
            df = df.iloc[prev_count:]

            # Drop rows with missing test.cond
            if "test.cond" not in df.columns or "agent.x" not in df.columns:
                continue
            df = df.dropna(subset=["test.cond"])
            if df.empty:
                continue

            # Coerce numeric columns
            df["agent.x"] = pd.to_numeric(df["agent.x"], errors="coerce")
            df["Episode"] = pd.to_numeric(df.get("Episode"), errors="coerce")

            all_frames.append(df)

        if not all_frames:
            return True

        data = pd.concat(all_frames, ignore_index=True)
        data = data.dropna(subset=["agent.x", "Episode"])
        if data.empty:
            return True

        # Classify spatial zones (same logic as merge.py)
        data["left"] = (data["agent.x"] < _BOUNDS[0]).astype(int)
        data["right"] = (data["agent.x"] > _BOUNDS[1]).astype(int)

        # Group by episode and test condition, sum zone steps
        group_cols = ["Episode", "test.cond"]
        if "correct.monitor" in data.columns:
            group_cols.append("correct.monitor")

        agg = (
            data.groupby(group_cols, dropna=False)
            .agg(left_steps=("left", "sum"), right_steps=("right", "sum"))
            .reset_index()
        )
        if "correct.monitor" not in agg.columns:
            return True

        # Compute correct / incorrect steps (same logic as test_viz.py)
        agg["correct_steps"] = np.where(
            agg["correct.monitor"] == "left",
            agg["left_steps"],
            agg["right_steps"],
        )
        agg["incorrect_steps"] = np.where(
            agg["correct.monitor"] == "right",
            agg["left_steps"],
            agg["right_steps"],
        )
        total = agg["correct_steps"] + agg["incorrect_steps"]
        agg["percent_correct"] = np.where(total != 0, agg["correct_steps"] / total, 0.0)

        # Average percent_correct per test.cond and record to TensorBoard
        scores = agg.groupby("test.cond")["percent_correct"].mean()
        for test_cond, score in scores.items():
            self.logger.record(f"test/{test_cond}", score)
            logger.debug(f"Eval score test/{test_cond} = {score:.4f}")

        return True
