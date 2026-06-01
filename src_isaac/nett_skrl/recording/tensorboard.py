"""TensorBoard video logging for exported NETT recordings."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .export import RecordingCfg

logger = logging.getLogger("nett.recording")


def log_recording_videos_to_tensorboard(agents: list, cfg: RecordingCfg) -> None:
    """Write exported MP4 videos to each agent's skrl TensorBoard run directory."""
    for brain_id, agent in enumerate(agents, start=1):
        videos = list(_iter_recording_mp4s(cfg.root, env_id=brain_id - 1))
        if not videos:
            continue
        writer = _recording_video_writer(agent)
        if writer is None:
            continue
        _add_recording_videos(writer, videos, fps=cfg.fps)


def _add_recording_videos(writer, videos: list[tuple[str, Path]], *, fps: int) -> None:
    try:
        for kind, mp4 in videos:
            _add_video(writer, mp4, tag=f"video/{kind}/{mp4.parent.name}/{mp4.stem}", fps=fps)
        writer.flush()
    finally:
        writer.close()


def _recording_video_writer(agent):
    exp_dir = getattr(agent, "experiment_dir", None)
    if not exp_dir:
        return None
    return _recording_video_writer_for_dir(Path(exp_dir))


def _recording_video_writer_for_dir(exp_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        logger.debug("torch TensorBoard SummaryWriter unavailable for video logging", exc_info=True)
        return None
    return SummaryWriter(log_dir=str(exp_dir))


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


def log_wrapper_egocentric_to_tensorboard(
    completed_episodes: list[tuple[str, list]],
    agents: list,
    *,
    fps: int = 24,
) -> None:
    """Write ChannelsFirst-buffered CHW frames directly to each agent's TensorBoard.

    ``completed_episodes`` is a list of ``(tag, frames)`` pairs where each
    frame is a ``(C, H, W)`` uint8 numpy array.  Frames are already CHW so no
    permute is required — this is the second (and last) explicit format
    conversion point in the pipeline (the first is ChannelsFirst itself).
    """
    import torch

    # Group episodes by env_id extracted from the tag ("env_N/episode_M").
    by_env: dict[int, list[tuple[str, list]]] = {}
    for tag, frames in completed_episodes:
        try:
            env_id = int(tag.split("/")[0].split("_")[1])
        except (IndexError, ValueError):
            env_id = 0
        by_env.setdefault(env_id, []).append((tag, frames))

    for brain_id, agent in enumerate(agents, start=1):
        env_id = brain_id - 1
        episodes = by_env.get(env_id, [])
        if not episodes:
            continue
        writer = _recording_video_writer(agent)
        if writer is None:
            continue
        try:
            for tag, frames in episodes:
                if not frames:
                    continue
                # stack (T, C, H, W) then unsqueeze batch dim → (1, T, C, H, W)
                tensor = torch.as_tensor(np.stack(frames)).unsqueeze(0)
                writer.add_video(f"video/egocentric/{tag}", tensor, fps=int(fps))
            writer.flush()
        except Exception:
            logger.exception("failed to write wrapper egocentric frames to TensorBoard")
        finally:
            writer.close()


def _add_video(writer, mp4: Path, *, tag: str, fps: int) -> None:
    try:
        import imageio.v3 as iio
        import torch

        frames = iio.imread(mp4)
        tensor = torch.as_tensor(frames)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(-1).expand(-1, -1, -1, 3)
        if tensor.ndim != 4:
            logger.debug(
                "skipping TensorBoard video with unsupported shape %s: %s",
                tuple(tensor.shape),
                mp4,
            )
            return
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0)
        writer.add_video(tag, tensor, fps=int(fps))
    except Exception:
        logger.exception("failed to write recording to TensorBoard: %s", mp4)
