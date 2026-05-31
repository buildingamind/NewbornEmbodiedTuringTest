"""Isaac Sim 5.1 per-mode subprocess runner.

Isaac Sim/Kit teardown is not reliably reusable inside one Python process for
NETT's train/test/record sequence. This module isolates the workaround: each
mode runs in a fresh ``multiprocessing.Process`` and the child exits with a
late ``os._exit(0)`` atexit hook after artifacts are flushed.
"""

from __future__ import annotations

import atexit
import csv
import json
import logging
import os
import shutil
from pathlib import Path

import torch

from nett_skrl.recording import RecordingCfg

from .task import Task, TaskConfig, set_seeds


def run_task(task: Task) -> None:
    """Run one (condition × N brains) task: train/test/record in fresh Isaac workers."""
    config = task.config
    set_seeds(config.seed)
    if config.dry_run:
        config.logger.info(
            "Spawning dry-run subprocess for condition %s", config.condition
        )
        _spawn_mode_subprocess(task, "train")
        config.logger.info("Dry-run subprocess complete")
        return

    (config.path / "logs").mkdir(exist_ok=True, parents=True)
    for mode in config.modes:
        if mode == "train" and config.eval_freq:
            _run_train_with_eval_milestones(task)
            continue
        config.logger.info(
            "Spawning %s subprocess for condition %s", mode, config.condition
        )
        _spawn_mode_subprocess(task, mode)
        config.logger.info("Mode %s subprocess complete", mode)


def _run_train_with_eval_milestones(task: Task) -> None:
    config = task.config
    total = int(getattr(task.agent.brain, "train_iterations", 0) or 0)
    eval_freq = int(config.eval_freq or 0)
    if total <= 0 or eval_freq <= 0:
        _spawn_mode_subprocess(task, "train")
        return

    boundaries = _training_boundaries(
        total,
        eval_freq=eval_freq,
        checkpoint_freq=getattr(task.agent.brain, "checkpoint_freq", None),
    )
    previous = 0
    for boundary in boundaries:
        chunk = boundary - previous
        if chunk > 0:
            config.logger.info(
                "Spawning train subprocess for condition %s: steps %d..%d",
                config.condition,
                previous,
                boundary,
            )
            _spawn_mode_subprocess(
                task,
                "train",
                train_timesteps=chunk,
                train_global_step=boundary,
            )
            _copy_final_checkpoints_to_global_step(task, boundary)
        if boundary % eval_freq == 0:
            config.logger.info(
                "Spawning metrics-only eval at train step %d for condition %s",
                boundary,
                config.condition,
            )
            _spawn_mode_subprocess(
                task,
                "test",
                eval_step=boundary,
                eval_metrics_only=True,
            )
        previous = boundary


def _copy_final_checkpoints_to_global_step(task: Task, step: int) -> None:
    freq = getattr(task.agent.brain, "checkpoint_freq", None)
    if not freq or step % int(freq) != 0:
        return
    for brain_id in range(1, int(task.config.num_brains) + 1):
        ckpt_dir = (
            task.config.path / "wandb_runs" / f"brain_{brain_id}" / "checkpoints"
        )
        src = ckpt_dir / "final_agent.pt"
        dst = ckpt_dir / f"agent_{step}.pt"
        if not src.exists():
            task.config.logger.warning("checkpoint alias skipped; missing %s", src)
            continue
        shutil.copy2(src, dst)


def _spawn_mode_subprocess(task: Task, mode: str, **overrides) -> None:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    p = ctx.Process(
        target=_run_single_mode, args=(task, mode, overrides), daemon=False
    )
    p.start()
    p.join()
    if p.exitcode != 0:
        if not _is_tolerated_isaac_teardown_exit(p.exitcode):
            raise RuntimeError(
                f"Mode {mode} subprocess failed with exit code {p.exitcode}"
            )
        task.config.logger.warning(
            "Mode %s subprocess exited with code %d (likely Isaac Sim teardown "
            "SIGSEGV; outputs on disk should still be intact)",
            mode,
            p.exitcode,
        )


def _run_single_mode(task: Task, mode: str, overrides: dict | None = None) -> None:
    """Child-process entry point for exactly one NETT mode."""
    # Kit's SimulationApp inspects sys.argv and aborts on flags it does not
    # recognize (e.g. pytest's ``-m``). Spawn forwards the parent's argv
    # verbatim, so scrub it down to the script name before any Isaac import.
    import sys

    sys.argv = sys.argv[:1] or ["nett-skrl"]

    config = task.config
    agent = task.agent
    set_seeds(config.seed)
    if not config.dry_run:
        (config.path / "logs").mkdir(exist_ok=True, parents=True)

    run_config = config.for_mode(mode, **(overrides or {}))
    agent.body.adjust_to_agent(
        agent.env,
        num_brains=config.num_brains,
        num_envs=config.num_envs,
    )
    loaded = agent.body.embed(agent.env, run_config)
    if mode == "train":
        agent.brain.train(
            loaded,
            run_config,
            record_cfg=(
                _make_record_cfg(agent.env, run_config) if not config.dry_run else None
            ),
        )
    elif mode == "record":
        agent.brain.record(
            loaded,
            run_config,
            record_cfg=_make_record_cfg(agent.env, run_config),
        )
    else:
        metrics = agent.brain.test(
            loaded,
            run_config,
            record_cfg=_make_record_cfg(agent.env, run_config),
        )
        if run_config.eval_step is not None:
            _write_eval_metrics(run_config, metrics, agent.brain)
    run_config.logger.info("Mode %s complete", mode)

    if config.dry_run:
        _write_dry_run_mem_report(run_config)

    torch.cuda.empty_cache()
    _exit_worker_cleanly(run_config.logger)


def _training_boundaries(
    total: int,
    *,
    eval_freq: int,
    checkpoint_freq: int | None,
) -> list[int]:
    boundaries = {int(total)}
    boundaries.update(range(eval_freq, total + 1, eval_freq))
    if checkpoint_freq:
        boundaries.update(range(int(checkpoint_freq), total + 1, int(checkpoint_freq)))
    return sorted(boundaries)


def _is_tolerated_isaac_teardown_exit(exitcode: int | None) -> bool:
    # multiprocessing reports Unix signals as negative numbers. Isaac/Kit can
    # segfault or abort during teardown after all Python-side artifacts have
    # flushed; ordinary positive exit codes are real failures and must stop
    # the run.
    return exitcode in {-6, -11, 134, 139}


def _write_eval_metrics(config: TaskConfig, metrics: dict[int, float], brain) -> None:
    logs = Path(config.path) / "logs"
    logs.mkdir(exist_ok=True, parents=True)
    total_timesteps = int(
        getattr(brain, "test_iterations", {}).get(config.condition, 1)
        * getattr(brain, "steps_per_episode", 1)
    )
    rows = [
        {
            "eval_step": int(config.eval_step or 0),
            "condition": config.condition,
            "brain_id": int(brain_id) + 1,
            "mean_reward": float(mean_reward),
            "timesteps": total_timesteps,
        }
        for brain_id, mean_reward in sorted(metrics.items())
    ]

    csv_path = logs / "eval_metrics.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "eval_step",
                "condition",
                "brain_id",
                "mean_reward",
                "timesteps",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    jsonl_path = logs / "eval_metrics.jsonl"
    with jsonl_path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_dry_run_mem_report(config: TaskConfig) -> None:
    """Write current free VRAM on this task's device to ``mem.txt``.

    Called from the dry-run subprocess after train+update has committed peak
    memory. The parent reads this and computes
    ``consumed = baseline_free_before_spawn - post_train_free``.
    """
    try:
        from .memory import MemoryManager

        with MemoryManager() as mm:
            free = mm.get_free_memory(int(config.device or 0))
    except Exception:
        config.logger.exception("dry-run: failed to read free memory")
        return
    config.path.mkdir(parents=True, exist_ok=True)
    (config.path / "mem.txt").write_text(str(int(free)))


def _make_record_cfg(env, config: TaskConfig) -> RecordingCfg | None:
    """Build the export-recordings config from ``env.recording`` if any
    camera has at least one phase configured. A camera is "enabled" by
    presence (a non-empty per-phase mapping) — there's no boolean flag.
    """
    if getattr(config, "eval_metrics_only", False):
        return None
    recording = getattr(env, "recording", {}) or {}
    ego_enabled = bool(recording.get("egocentric") or {})
    chamber_enabled = bool(recording.get("chamber") or {})
    if not ego_enabled and not chamber_enabled:
        return None
    return RecordingCfg(
        root=config.path / "recordings",
        fps=int(recording.get("fps", 24)),
        egocentric_enabled=ego_enabled,
        chamber_enabled=chamber_enabled,
    )


def _exit_worker_cleanly(logger: logging.Logger) -> None:
    """Flush Python-side outputs, then bypass Kit's fragile atexit teardown."""
    try:
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    for handler in list(logging.root.handlers):
        try:
            handler.flush()
        except Exception:
            pass
    try:
        import wandb

        for _ in range(16):
            if getattr(wandb, "run", None) is None:
                break
            wandb.finish()
    except Exception:
        logger.debug("wandb.finish skipped", exc_info=True)
    atexit.register(os._exit, 0)
