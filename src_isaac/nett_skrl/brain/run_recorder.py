"""Run output recording for :mod:`nett_skrl.brain.trainer`."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..recording.export import RecordingCfg, export_recordings

if TYPE_CHECKING:
    from .trainer import TrainCfg

logger = logging.getLogger("nett.recorder")


class RunRecorder:
    """Writes run outputs around skrl training without owning the train loop."""

    def __init__(self, agents: list, num_envs: int) -> None:
        self.agents = agents
        self.num_envs = int(num_envs)

    def before_train(self, cfg: TrainCfg, *, dry_run: bool = False) -> None:
        if cfg.hparams_dir and not dry_run:
            self._write_hparams(cfg)

    def after_train(
        self,
        cfg: TrainCfg,
        *,
        elapsed_s: float,
        record_cfg: RecordingCfg | None = None,
        dry_run: bool = False,
    ) -> None:
        if dry_run:
            return
        if cfg.hparams_dir:
            self._write_train_timing(cfg, elapsed_s)
        self.save_final_checkpoints()
        try:
            if record_cfg:
                self._export_and_log_recordings(record_cfg)
        finally:
            self.finish_wandb_runs()

    def after_rollout(self, record_cfg: RecordingCfg | None = None) -> None:
        """Finalize a non-training rollout while skrl/wandb runs are alive."""
        try:
            if record_cfg:
                self._export_and_log_recordings(record_cfg)
        finally:
            self.finish_wandb_runs()

    def finish_wandb_runs(self) -> None:
        from .experiment import finish_agent_wandb_runs

        finish_agent_wandb_runs(self.agents)

    def save_final_checkpoints(self) -> None:
        """Write ``{experiment_dir}/checkpoints/final_agent.pt`` per agent."""
        for agent in self.agents:
            exp_dir = getattr(agent, "experiment_dir", None)
            if not exp_dir:
                continue
            ckpt_dir = Path(exp_dir) / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(ckpt_dir / "final_agent.pt"))

    def _export_and_log_recordings(self, record_cfg: RecordingCfg) -> None:
        export_recordings(record_cfg)
        log_agent_recordings_to_tensorboard(self.agents, record_cfg)

    def _write_hparams(self, cfg: TrainCfg) -> None:
        out = Path(cfg.hparams_dir) / "logs"
        out.mkdir(parents=True, exist_ok=True)
        payload = dict(cfg.hparams or {})
        payload.setdefault("total_timesteps", cfg.total_timesteps)
        with (out / "hparams.json").open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _write_train_timing(self, cfg: TrainCfg, elapsed_s: float) -> None:
        out = Path(cfg.hparams_dir) / "logs"
        out.mkdir(parents=True, exist_ok=True)
        env_timesteps = int(cfg.total_timesteps)
        train_steps = env_timesteps * max(1, self.num_envs)
        payload = {
            "skrl_train_total_s": elapsed_s,
            "env_timesteps": env_timesteps,
            "train_steps": train_steps,
            "env_steps_per_second": float(env_timesteps) / elapsed_s if elapsed_s > 0 else 0.0,
            "train_steps_per_second": float(train_steps) / elapsed_s if elapsed_s > 0 else 0.0,
        }
        with (out / "train_timing.json").open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)


def log_agent_recordings_to_tensorboard(agents: list, cfg: RecordingCfg) -> None:
    """Write exported MP4 recordings to each live agent's skrl TensorBoard log."""
    for brain_id, agent in enumerate(agents, start=1):
        writer, close_writer = _tensorboard_writer(agent)
        _log_env_recordings(writer, close_writer, cfg, env_id=brain_id - 1)


def _log_env_recordings(writer, close_writer: bool, cfg: RecordingCfg, *, env_id: int) -> None:
    if writer is None:
        return
    try:
        for kind, mp4 in _iter_recording_mp4s(cfg.root, env_id=env_id):
            _add_video(writer, mp4, tag=f"video/{kind}/{mp4.parent.name}/{mp4.stem}", fps=cfg.fps)
        if hasattr(writer, "flush"):
            writer.flush()
    finally:
        if close_writer and hasattr(writer, "close"):
            writer.close()


def _tensorboard_writer(agent):
    writer = getattr(agent, "writer", None)
    if writer is not None and hasattr(writer, "add_video"):
        return writer, False

    exp_dir = getattr(agent, "experiment_dir", None)
    if not exp_dir:
        return None, False
    return _tensorboard_writer_for_dir(Path(exp_dir))


def _tensorboard_writer_for_dir(exp_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        logger.debug("torch TensorBoard SummaryWriter unavailable", exc_info=True)
        return None, False
    return SummaryWriter(log_dir=str(exp_dir)), True


def _iter_recording_mp4s(rec_dir: Path, *, env_id: int):
    env_prefix = f"env_{env_id}"
    for kind in ("egocentric", "chamber"):
        kind_dir = Path(rec_dir) / kind
        if not kind_dir.exists():
            continue
        for env_subdir in kind_dir.rglob("*"):
            if env_subdir.is_dir() and env_subdir.name.startswith(env_prefix):
                for mp4 in sorted(env_subdir.glob("*.mp4")):
                    yield kind, mp4


def _add_video(writer, mp4: Path, *, tag: str, fps: int) -> None:
    try:
        import imageio.v3 as iio
        import torch

        frames = iio.imread(mp4)
        tensor = torch.as_tensor(frames)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(-1).expand(-1, -1, -1, 3)
        if tensor.ndim != 4:
            logger.debug("skipping TensorBoard video with unsupported shape %s: %s", tuple(tensor.shape), mp4)
            return
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)
        writer.add_video(tag, tensor, fps=int(fps))
    except Exception:
        logger.exception("failed to write recording to TensorBoard: %s", mp4)
