"""Upload NETT-skrl per-run outputs to each brain's wandb Run.

skrl already mirrors training scalars + hparams into wandb via
``sync_tensorboard=True`` (see ``brain/experiment.py``). The artifacts
NETT produces *outside* skrl's tensorboard writer — checkpoints, per-step
CSV logs, profiling JSON, exported MP4 recordings, the saved config —
don't make it to wandb without an explicit upload. This module handles
that, keyed per brain so each brain's wandb Run gets only its own files.

Per-brain attribution rules:
    Checkpoints (`wandb_runs/brain_N/checkpoints/*.pt`)
        Owned by brain N exclusively.
    Recordings under `recordings/<kind>/<phase>/env_<N-1>/`
        ``env_id = brain_id - 1`` (NETT-skrl maps brain 1 → env 0).
        MP4s log as ``wandb.Video`` so they show in the W&B media panel;
        the raw files are also ``run.save``'d for download.
    Per-step CSV log (one file per condition × phase × seed)
        Shared across brains (LogChannel writes brain rows interleaved
        into one CSV). Uploaded to every brain's Run so each brain's
        wandb page can be self-contained.
    Profiling JSON, hparams.json, train_timing.json, config.yaml
        Same shared-upload treatment.

If wandb is disabled (`brain.wandb_cfg.mode == "disabled"`) the lookup
returns ``None`` for every brain and the whole function is a no-op.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

try:
    import wandb
except ImportError:
    wandb = None

from .experiment import get_wandb_run_by_id, wandb_run_id

logger = logging.getLogger("nett.wandb")


def sync_outputs_to_wandb(
    agents: list,
    *,
    output_dir: Path,
    condition: str,
    phase: str,
    run_name: str,
) -> None:
    """Upload per-run artifacts to each brain's wandb Run.

    Safe to call even when wandb is disabled or some artifacts are missing
    — every step is best-effort and exceptions are logged, not raised.
    """
    if wandb is None:
        logger.debug("wandb not installed; skipping output sync")
        return

    condition_dir = Path(output_dir) / condition

    for brain_id, agent in enumerate(agents, start=1):
        run_id = wandb_run_id(run_name, condition, brain_id, phase)
        run = get_wandb_run_by_id(run_id)
        if run is None:
            logger.debug(
                "no wandb Run for brain_%d (id=%s); skipping upload",
                brain_id,
                run_id,
            )
            continue

        _upload_checkpoints(run, agent)
        _upload_recordings(
            run,
            condition_dir / "recordings",
            phase=phase,
            env_id=brain_id - 1,
        )
        _upload_shared_files(
            run,
            condition_dir=condition_dir,
            phase=phase,
            condition=condition,
        )


def _safe_save(run: Any, path: Path, base_path: Path, policy: str = "now") -> None:
    """``run.save`` wrapper that swallows + logs failures.

    ``policy='now'`` uploads immediately rather than at the end of the
    run; for files that are still being written we'd want ``'live'``,
    but all of NETT's outputs are written-then-uploaded so ``now`` is
    correct.
    """
    try:
        run.save(str(path), base_path=str(base_path), policy=policy)
    except Exception:
        logger.exception("wandb run.save failed: %s", path)


def _save_existing(run: Any, paths: Iterable[Path], *, base_path: Path) -> None:
    for path in paths:
        if path.exists():
            _safe_save(run, path, base_path=base_path)


def _upload_checkpoints(run: Any, agent: Any) -> None:
    exp_dir = getattr(agent, "experiment_dir", None)
    if not exp_dir:
        return
    ckpt_dir = Path(exp_dir) / "checkpoints"
    if not ckpt_dir.is_dir():
        return
    _save_existing(
        run,
        sorted(ckpt_dir.glob("*.pt")),
        base_path=Path(exp_dir).parent,
    )


def _upload_recordings(run: Any, rec_dir: Path, *, phase: str, env_id: int) -> None:
    """Upload MP4 recordings under ``recordings/<kind>/<phase>/env_<env_id>``.

    Both per-frame PNGs and the exported MP4 live under the same
    directory after ``export_recordings``. We upload the MP4s only —
    they're a 100× size win over PNGs and the W&B media panel renders
    them inline.
    """
    if not rec_dir.exists():
        return
    for kind, mp4 in _iter_recording_mp4s(rec_dir, phase=phase, env_id=env_id):
        _log_video(run, mp4, kind=kind)
        _safe_save(run, mp4, base_path=rec_dir.parent)


def _iter_recording_mp4s(
    rec_dir: Path,
    *,
    phase: str,
    env_id: int,
) -> Iterable[tuple[str, Path]]:
    env_prefix = f"env_{env_id}"
    for kind in ("egocentric", "chamber"):
        kind_dir = rec_dir / kind / phase
        if not kind_dir.exists():
            continue
        for env_subdir in kind_dir.iterdir():
            if env_subdir.is_dir() and env_subdir.name.startswith(env_prefix):
                for mp4 in sorted(env_subdir.rglob("*.mp4")):
                    yield kind, mp4


def _log_video(run: Any, mp4: Path, *, kind: str) -> None:
    """Log an MP4 as ``wandb.Video`` so it renders in the W&B media panel."""
    if wandb is None:
        return
    key = f"video/{kind}/{mp4.parent.name}/{mp4.stem}"
    try:
        run.log({key: wandb.Video(str(mp4))})
    except Exception:
        logger.exception("wandb.Video log failed: %s", mp4)


def _upload_shared_files(
    run: Any,
    *,
    condition_dir: Path,
    phase: str,
    condition: str,
) -> None:
    """Upload files that aren't naturally per-brain (config, logs, profiles).

    Each brain's Run gets its own copy of these so a single Run page on
    wandb is self-contained — there's no global "run" view that aggregates
    across brains.
    """
    _save_existing(run, _iter_run_files(condition_dir), base_path=condition_dir.parent)
    _save_existing(
        run,
        _iter_log_files(condition_dir / "logs", phase=phase, condition=condition),
        base_path=condition_dir,
    )


def _iter_run_files(condition_dir: Path) -> Iterable[Path]:
    """Files saved relative to the run-name directory."""
    yield condition_dir.parent / "config.yaml"


def _iter_log_files(logs_dir: Path, *, phase: str, condition: str) -> Iterable[Path]:
    """Files saved relative to the condition directory."""
    yield from (
        *logs_dir.glob(f"{phase}_{condition}_*.csv"),
        *logs_dir.glob(f"profile_{phase}_{condition}_*.json"),
        logs_dir / "hparams.json",
        logs_dir / "train_timing.json",
    )
