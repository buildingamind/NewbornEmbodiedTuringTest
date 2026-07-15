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
import math
import os
import shutil
from pathlib import Path

import torch

from nett_skrl.recording import RecordingCfg

from . import crash_guard
from .reap import (
    DeviceLostError,
    TaskReaper,
    TaskTimeoutError,
    is_device_lost_exit,
    join_with_reap,
)
from .task import Task, TaskConfig, recording_phase_map, set_seeds


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
        overrides = {}
        if mode == "test":
            overrides["num_envs"] = _compute_eval_num_envs(task)
        config.logger.info(
            "Spawning %s subprocess for condition %s", mode, config.condition
        )
        _spawn_mode_subprocess(task, mode, **overrides)
        config.logger.info("Mode %s subprocess complete", mode)


def _run_train_with_eval_milestones(task: Task) -> None:
    config = task.config
    total = int(getattr(task.agent.brain, "train_iterations", 0) or 0)
    eval_freq = int(config.eval_freq or 0)
    if total <= 0 or eval_freq <= 0:
        _spawn_mode_subprocess(task, "train")
        return

    eval_num_envs = _compute_eval_num_envs(task)
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
                train_start_step=previous,
            )
            _copy_final_checkpoints_to_global_step(task, boundary)
        if boundary % eval_freq == 0:
            config.logger.info(
                "Spawning metrics-only eval at train step %d for condition %s "
                "(eval_num_envs=%d)",
                boundary,
                config.condition,
                eval_num_envs,
            )
            _spawn_mode_subprocess(
                task,
                "test",
                eval_step=boundary,
                eval_metrics_only=True,
                num_envs=eval_num_envs,
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
    reaper = TaskReaper(
        task_key=f"{task.config.name}/{task.config.condition}/{mode}",
        device=task.config.device,
        logger=task.config.logger,
    )
    # launch_scope() pre-warms multiprocessing's resource_tracker and THEN stamps
    # the per-spawn ownership token into os.environ, so the child (and every
    # descendant it forks) inherits it and stays attributable after reparenting
    # to PPID=1. The pre-warm ordering is load-bearing: spawn's Popen.__init__
    # calls resource_tracker.ensure_running(), so without it the worker's SHARED
    # tracker would be born inside the scope, inherit the token, and be reaped as
    # if it were ours. The token is removed again as soon as start() returns, so
    # a later healthy spawn is never mis-attributed to this task.
    with reaper.launch_scope():
        p = ctx.Process(
            target=_run_single_mode, args=(task, mode, overrides), daemon=False
        )
        p.start()
    reaper.adopt(p.pid)

    # crash_evidence=None deliberately: crash_guard signals DEVICE_LOST by EXIT
    # CODE (75 / -14), not by a marker file, and its SIGALRM kernel backstop
    # already guarantees this join returns. A healthy run of any duration joins
    # normally; NETT_REAP_TIMEOUT defaults to 0 (disabled).
    outcome = join_with_reap(p, reaper, crash_evidence=None, logger=task.config.logger)

    # ORDER IS LOAD-BEARING: the reap outcome is checked BEFORE any exitcode
    # logic. A reaped child was SIGKILLed, so p.exitcode is -9 -- which IS in
    # _is_tolerated_isaac_teardown_exit()'s {-6, -9, -11, 134, 139} set and would
    # otherwise be mislabeled "outputs on disk should still be intact" and let
    # the run continue as though the mode had succeeded.
    if outcome == "reaped-crash":
        raise DeviceLostError(
            f"Mode {mode} subprocess reaped after DEVICE_LOST (exit {p.exitcode})"
        )
    if outcome == "reaped-timeout":
        raise TaskTimeoutError(
            f"Mode {mode} subprocess reaped on NETT_REAP_TIMEOUT (exit {p.exitcode})"
        )

    if p.exitcode != 0:
        if is_device_lost_exit(p.exitcode):
            # The child self-exited via crash_guard's os._exit(75) (or its
            # SIGALRM backstop, -14/142). os._exit runs no atexit hooks, so its
            # multiprocessing.spawn workers were NOT terminated and its CUDA/RTX
            # context was NOT released: they reparent to PPID=1 and keep holding
            # ~5GB of VRAM, which would OOM the next wave item on this device.
            # The join returned cleanly, so ONLY this reap frees the GPU.
            reaper.reap("device-lost-exit")
            raise DeviceLostError(
                f"Mode {mode} subprocess exited with code {p.exitcode} (DEVICE_LOST)"
            )
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

    # Pin torch/OpenMP to this cell's CPU budget BEFORE any torch work. torch
    # otherwise sizes its intra-op pool from os.cpu_count(), so every cell in a
    # wave claims all 64 cores during the PPO update. Must happen in the child:
    # spawn does not inherit the parent's torch thread settings.
    from .cpu_budget import apply_torch_thread_limits

    apply_torch_thread_limits()

    config = task.config
    agent = task.agent
    set_seeds(config.seed)
    if not config.dry_run:
        (config.path / "logs").mkdir(exist_ok=True, parents=True)

    run_config = config.for_mode(mode, **(overrides or {}))
    # Use run_config.num_envs: may differ from config.num_envs when eval_num_envs
    # is passed as an override for mid-training eval subprocesses.
    agent.body.adjust_to_agent(
        agent.env,
        num_brains=config.num_brains,
        num_envs=run_config.num_envs,
    )
    loaded = agent.body.embed(agent.env, run_config)
    # Kit is up now (embed builds AppLauncher/SimulationApp). Its startup resets
    # carb logging, so the device-lost guard MUST arm after this line. Pass the
    # scheduled GPU so crash forensics can label the telemetry snapshot.
    crash_guard.arm(config.path, device=config.device)
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


def _is_square_tile_grid(num_envs: int) -> bool:
    """Does ``num_envs`` tile into a SQUARE camera grid?

    Isaac Lab tiles N cameras into ``ceil(sqrt(N)) x ceil(N / cols)``. The chick eye
    is a fisheye, and at a NON-square tile grid the lens distortion is applied over
    the full non-square canvas aspect, so the rendered eye image is distorted --
    Isaac Sim #488. See isaac_lab/docs/known_issues.md.

    Perfect squares qualify; so do a few others whose grid still comes out square
    (8 -> 3x3, 13 -> 4x4, 80 -> 9x9). 52 -> 8x7 does NOT.
    """
    if num_envs < 1:
        return False
    cols = math.ceil(math.sqrt(num_envs))
    rows = math.ceil(num_envs / cols)
    return cols == rows


def _compute_eval_num_envs(task: Task) -> int:
    """Largest VALID test-phase num_envs: divides the test episodes, is a multiple of
    num_brains, and tiles into a SQUARE camera grid.

    Test is embarrassingly parallel in a way training is not: training's num_envs is
    load-bearing for learning (rollout composition), but test just replays a fixed
    schedule, so the only real caps are (a) not leaving empty NETTEnv episode slots
    and (b) VRAM. Every extra env removes a whole sequential batch.

    THE SQUARE-GRID FILTER IS A CORRECTNESS CONSTRAINT, NOT A PREFERENCE. Without it
    this function happily returns e.g. 52 (8x7) for 1040 test episodes -- which is
    exactly what ``NETT_TEST_ENVS=64`` used to resolve to -- and every test frame is
    then rendered through a distorted fisheye (#488). Silent, and it corrupts the
    measurement rather than crashing.
    """
    config = task.config
    num_brains = max(1, int(config.num_brains))
    episodes_test = int((config.episodes or {}).get("test", 1))
    num_test_tasks = max(1, task.agent.env.iterations_per_test_episode.get(config.condition, 1))
    total_test_episodes = num_test_tasks * episodes_test

    # NETT_TEST_ENVS: request a test-phase parallelism independent of training's
    # num_envs. Treated as a CEILING the caller controls (VRAM), never a target.
    forced = os.environ.get("NETT_TEST_ENVS")
    if forced:
        want = min(max(1, int(forced)), total_test_episodes)
    else:
        # Default: cap at training's max_parallel_envs. Test could go wider (it only
        # replays a fixed schedule) but that is a VRAM call, so it stays opt-in.
        max_envs = config.max_parallel_envs
        want = min(max_envs, total_test_episodes) if max_envs is not None else total_test_episodes
        want = max(1, want)

    def _square_multiple(n: int) -> bool:
        # HARD constraints, both of which BREAK something if violated:
        #   n % num_brains == 0  -> else BrainTrainer raises (each brain owns one
        #                           contiguous env scope; see brain_scope_sizes)
        #   square tile grid     -> else the fisheye render is distorted (#488)
        return n >= 1 and n % num_brains == 0 and _is_square_tile_grid(n)

    # Pass 1: also divide the episode total, so no env sits idle in the last batch.
    for n in range(want, 0, -1):
        if _square_multiple(n) and total_test_episodes % n == 0:
            return n

    # Pass 2: relax ONLY the divides-the-total preference (the original code did the
    # same). A partial final batch leaves some envs idle; that is survivable, a
    # crash or a distorted lens is not.
    for n in range(want, 0, -1):
        if _square_multiple(n):
            return n

    # Pass 3: nothing at or below the request qualifies (e.g. want < num_brains).
    # Go UP to the smallest square-grid multiple of num_brains rather than return
    # something that breaks BrainTrainer.
    n = num_brains
    while not _is_square_tile_grid(n):
        n += num_brains
    config.logger.warning(
        "no square-grid num_envs <= %d is a multiple of num_brains=%d for %d test "
        "episodes; using %d (above the request) so the tile grid stays square and "
        "each brain keeps a whole env scope.",
        want, num_brains, total_test_episodes, n,
    )
    return n


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
    # segfault, abort, or get SIGKILL'd (OOM killer or manual kill after
    # artifacts are flushed) during teardown; ordinary positive exit codes are
    # real failures and must stop the run.
    return exitcode in {-6, -9, -11, 134, 139}


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
    ego_enabled = bool(recording_phase_map(recording, "egocentric"))
    chamber_enabled = bool(recording_phase_map(recording, "chamber"))
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
    # This IS the clean path — retire the device-lost guard so teardown is
    # exactly what it was before crash_guard existed.
    crash_guard.disarm()
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
