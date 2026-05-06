import threading
from pathlib import Path

from stable_baselines3.common.callbacks import EvalCallback


_HEARTBEAT_INTERVAL_SECONDS = 30


# Spatial zone constants — must match src/nett/analysis/utils/merge.py
_X_LIMITS = (-33.15, 33.15)
_AGENT_RADIUS = 3.0
_AGENT_LIMITS = (_X_LIMITS[0] + _AGENT_RADIUS, _X_LIMITS[1] - _AGENT_RADIUS)
_ONE_THIRD = (_AGENT_LIMITS[1] - _AGENT_LIMITS[0]) / 3
_BOUNDS = [_AGENT_LIMITS[0] + _ONE_THIRD, _AGENT_LIMITS[1] - _ONE_THIRD]


class KeepAliveEvalCallback(EvalCallback):
    """EvalCallback that keeps the training Unity process alive during long evaluations.

    After a long evaluation period the training Unity process can become
    unresponsive because it has been idle while eval episodes ran.  Two
    mechanisms work together to prevent UnityTimeOutException:

    1. **Heartbeat thread**: When an evaluation is triggered, a background
       thread calls ``training_env.reset()`` every ``_HEARTBEAT_INTERVAL_SECONDS``
       (default 60 s) *while the evaluation is in progress*, keeping the
       Python↔Unity communication channel alive throughout the eval.  The thread
       stops as soon as ``super()._on_step()`` returns.

    2. **Post-eval reset**: After the evaluation completes, a final
       ``training_env.reset()`` re-establishes the channel before
       ``collect_rollouts`` tries to step it again.

    Both resets are best-effort — exceptions are caught and silenced so that a
    transient Unity hiccup does not abort the whole training run.

    After the eval has fully wrapped up (and the eval Unity has had a chance to
    flush its log buffer via a final ``eval_env.reset()``), the callback parses
    the Unity test CSV, computes ``percent_correct`` per ``test.cond``, and
    records the scores to TensorBoard.
    """

    def __init__(
        self,
        eval_env,
        task_path: Path,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        eval_log_path = task_path / "eval_logs"
        eval_log_path.mkdir(parents=True, exist_ok=True)
        self._eval_unity_log_dir = task_path / "_eval" / "logs"
        self._last_line_counts: dict[str, int] = {}
        super().__init__(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            best_model_save_path=str(task_path / "best_model"),
            log_path=str(eval_log_path),
            deterministic=deterministic,
            verbose=verbose,
        )

    def _on_step(self) -> bool:
        should_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0

        if should_eval and self.training_env is not None:
            stop_event = threading.Event()

            def _heartbeat():
                while not stop_event.wait(_HEARTBEAT_INTERVAL_SECONDS):
                    try:
                        self.training_env.reset()
                    except Exception:
                        pass

            t = threading.Thread(target=_heartbeat, daemon=True)
            t.start()
            result = super()._on_step()
            stop_event.set()
            t.join(timeout=5)
        else:
            result = super()._on_step()

        if should_eval and self.training_env is not None:
            try:
                self.training_env.reset()
            except Exception:
                pass

            # Force the eval Unity to flush the final episode's rows to disk.
            # evaluate_policy stops as soon as n_eval_episodes is hit without
            # resetting the env, so without this the last episode can still be
            # sitting in Unity's stdio buffer when we read the CSV.
            try:
                self.eval_env.reset()
            except Exception:
                pass

            self._record_eval_scores()

        return result

    def _record_eval_scores(self) -> None:
        import pandas as pd
        import numpy as np
        import logging

        logger = logging.getLogger("nett.KeepAliveEvalCallback")

        log_files = list(self._eval_unity_log_dir.glob("test_*.csv"))
        if not log_files:
            return

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

            if "test.cond" not in df.columns or "agent.x" not in df.columns:
                continue
            df = df.dropna(subset=["test.cond"])
            if df.empty:
                continue

            df["agent.x"] = pd.to_numeric(df["agent.x"], errors="coerce")
            df["Episode"] = pd.to_numeric(df.get("Episode"), errors="coerce")

            all_frames.append(df)

        if not all_frames:
            return

        data = pd.concat(all_frames, ignore_index=True)
        data = data.dropna(subset=["agent.x", "Episode"])
        if data.empty:
            return

        # Classify spatial zones (same logic as merge.py)
        data["left"] = (data["agent.x"] < _BOUNDS[0]).astype(int)
        data["right"] = (data["agent.x"] > _BOUNDS[1]).astype(int)

        group_cols = ["Episode", "test.cond"]
        if "correct.monitor" in data.columns:
            group_cols.append("correct.monitor")

        agg = (
            data.groupby(group_cols, dropna=False)
            .agg(left_steps=("left", "sum"), right_steps=("right", "sum"))
            .reset_index()
        )
        if "correct.monitor" not in agg.columns:
            return

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

        scores = agg.groupby("test.cond")["percent_correct"].mean()
        for test_cond, score in scores.items():
            self.logger.record(f"test/{test_cond}", score)
            logger.debug(f"Eval score test/{test_cond} = {score:.4f}")
