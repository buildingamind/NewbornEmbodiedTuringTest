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
from pathlib import Path

import torch

from nett_skrl.recording import RecordingCfg

from . import crash_guard
from . import stall_guard
from .device import visible_device_scope
from .parallel_envs import select_test_num_envs
from .reap import (
    DeviceLostError,
    StallError,
    TaskReaper,
    TaskTimeoutError,
    VramOomError,
    is_device_lost_exit,
    is_stall_exit,
    is_teardown_wedge_exit,
    is_vram_oom_exit,
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
        # The probe's mode is the caller's: a TRAIN probe measures the optimizer and
        # rollout buffer, a TEST probe measures neither (and is far cheaper).
        _spawn_mode_subprocess(task, config.modes[0] if config.modes else "train")
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
    boundaries = _training_boundaries(total, eval_freq=eval_freq)
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


def _spawn_mode_subprocess(task: Task, mode: str, **overrides) -> None:
    """Run one mode in its own Isaac process, with the reaper visible to a signal handler.

    ★ THE REGISTRATION IS THE POINT OF THIS WRAPPER. Every other ``reaper.reap()`` call
    site is on a path THIS process chooses (vram-oom / stall / device-lost /
    absolute-timeout). Registering the reaper for the life of the task is what lets a
    SIGTERM or SIGINT arriving from OUTSIDE -- the shell ``timeout`` bounding a wave, a
    pkill, an operator, or the kernel's PDEATHSIG when the driver is ``kill -9``ed -- run
    the SAME ownership-checked reap instead of stranding the Isaac child at PPID=1 holding
    VRAM. The handler itself is installed once per pool worker in ``executor._worker_init``.

    Weakly held, and unregistered in ``finally``: a finished task must not keep its reaper
    -- or its process identities -- alive for the next one to trip over.
    """
    from .lifecycle import register_reaper, unregister_reaper

    reaper = TaskReaper(
        task_key=f"{task.config.name}/{task.config.condition}/{mode}",
        device=task.config.device,
        logger=task.config.logger,
    )
    register_reaper(reaper)
    try:
        _run_mode_subprocess(task, mode, reaper, overrides)
    finally:
        unregister_reaper(reaper)


def _run_mode_subprocess(task: Task, mode: str, reaper: TaskReaper,
                         overrides: dict) -> None:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    # launch_scope() pre-warms multiprocessing's resource_tracker and THEN stamps
    # the per-spawn ownership token into os.environ, so the child (and every
    # descendant it forks) inherits it and stays attributable after reparenting
    # to PPID=1. The pre-warm ordering is load-bearing: spawn's Popen.__init__
    # calls resource_tracker.ensure_running(), so without it the worker's SHARED
    # tracker would be born inside the scope, inherit the token, and be reaped as
    # if it were ours. The token is removed again as soon as start() returns, so
    # a later healthy spawn is never mis-attributed to this task.
    with visible_device_scope(task.config.device), reaper.launch_scope():
        p = ctx.Process(
            target=_run_single_mode, args=(task, mode, overrides), daemon=False
        )
        p.start()
    reaper.adopt(p.pid)

    # crash_evidence=None deliberately: crash_guard signals DEVICE_LOST by EXIT
    # CODE (75 / -14), not by a marker file, and its SIGALRM kernel backstop
    # already guarantees this join returns.
    #
    # absolute_timeout is set for DRY RUNS ONLY (TaskConfig.dry_run_timeout). None --
    # every real run -- keeps the unbounded join, since a healthy run of any duration
    # must join normally and NETT_REAP_TIMEOUT defaults to 0 (disabled). A probe is the
    # one caller that genuinely knows its own budget, which is exactly the opt-in case
    # the reap module documents; without it, a probe that OOMs outside the DEVICE_LOST
    # path wedges forever.
    outcome = join_with_reap(
        p,
        reaper,
        crash_evidence=None,
        absolute_timeout=getattr(task.config, "dry_run_timeout", None),
        logger=task.config.logger,
    )

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
        if is_vram_oom_exit(p.exitcode):
            # Same os._exit story as DEVICE_LOST below: no atexit hooks ran, so the
            # child's spawn workers and its RTX context still hold VRAM. ONLY this
            # reap frees the device for the search's next probe.
            reaper.reap("vram-oom-exit")
            raise VramOomError(
                f"Mode {mode} subprocess ran out of VRAM (exit {p.exitcode}); "
                f"num_envs is too large for this GPU"
            )
        if is_stall_exit(p.exitcode):
            # Same os._exit story as DEVICE_LOST: stall_guard hard-exits, so no atexit
            # hooks ran, the child's spawn workers are still alive and its RTX context
            # still holds ~5GB. ONLY this reap frees the device for the next wave item.
            reaper.reap("stall-exit")
            raise StallError(
                f"Mode {mode} subprocess made no env-step progress for the stall "
                f"budget and self-exited (exit {p.exitcode}); this is the Kit "
                f"render-pump wedge, which emits no DEVICE_LOST"
            )
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
        if is_teardown_wedge_exit(p.exitcode):
            # NOT a casualty: the mode's work COMPLETED and _finalize_env_artifacts had
            # already flushed before the guarded window even opened -- only Kit's
            # shutdown wedged. So the run continues as a success, loudly. The reap is
            # still mandatory: the child left via os._exit, so no atexit hooks ran and
            # anything it spawned is still holding VRAM.
            reaper.reap("teardown-wedge-exit")
            task.config.logger.warning(
                "Mode %s COMPLETED but its teardown wedged and was bounded (exit %d). "
                "Outputs on disk are intact -- artifacts are flushed before that window "
                "opens. The GPU has been reclaimed by the reap. Raise "
                "NETT_TEARDOWN_BUDGET_S if a legitimate shutdown here needs longer.",
                mode, p.exitcode,
            )
            return
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
    """Child-process entry point for exactly one NETT mode.

    ★ THE ERROR PATH MUST HARD-EXIT TOO. ``_exit_worker_cleanly`` -- whose whole job is
    ``atexit.register(os._exit, 0)``, bypassing Kit's fragile teardown -- was reached
    ONLY on success. Any exception therefore fell through into Kit's REAL teardown,
    which is exactly where this stack hangs, and the parent's join is unbounded
    (``absolute_timeout=None``, ``NETT_REAP_TIMEOUT`` defaults to 0). A failing mode
    hung the whole wave instead of failing it. It also lost every artifact, because
    ``_finalize_env_artifacts`` was on the success path only.

    So the body is wrapped: on ANY exception, flush the env's artifacts, retire both
    guards, and ``os._exit(1)``. NON-ZERO on purpose -- ``os._exit(0)`` would report a
    failed mode as a success, and 1 is outside both the tolerated-teardown set and every
    special code (75/76/77/78), so the parent raises for it as it always has.
    """
    state: dict = {}
    try:
        _run_single_mode_body(task, mode, overrides, state)
    except BaseException:  # noqa: BLE001 - the traceback IS the report; then we must exit
        if not _in_spawned_worker():
            # ⚠ IN-PROCESS CALLER (a test, or anything that has not spawned). os._exit
            # here would take the CALLER down -- pytest included -- turning one red test
            # into a suite that vanishes with no report. The hard exit exists to dodge
            # Kit's teardown in a throwaway child; in-process there is no such child, so
            # the exception is the right answer.
            raise
        _abort_worker(state, task, mode)          # never returns


def _in_spawned_worker() -> bool:
    """True when this really is the mode subprocess.

    ``multiprocessing`` names the interpreter's root process ``MainProcess`` and every
    spawned child something else, so this distinguishes "I am the disposable Isaac
    worker" from "someone called me directly". Kept as a named function so a test can
    force either answer.
    """
    import multiprocessing as mp

    return mp.current_process().name != "MainProcess"


def _abort_worker(state: dict, task: Task, mode: str) -> None:
    """Report the failure, salvage the artifacts, and leave WITHOUT Kit's teardown.

    Ordering is the whole content of this function:
      1. print the traceback while stdout/stderr still exist;
      2. flush the env's artifacts -- ``_finalize_env_artifacts`` is Kit-free and
         idempotent, and on the old error path it never ran at all, so a crashed mode
         lost its profiler JSON and its last grid-video manifest every time;
      3. retire both guards, so neither fires during the exit and mislabels it;
      4. ``os._exit(1)`` -- the same hard exit the success path takes, for the same
         reason: Kit's real teardown is where this stack hangs.
    """
    import sys
    import traceback

    logger = getattr(getattr(task, "config", None), "logger", None) or logging.getLogger(
        "nett.task_runner"
    )
    try:
        traceback.print_exc()
        logger.exception("Mode %s failed; exiting worker without Kit teardown", mode)
    except Exception:  # noqa: BLE001
        pass
    loaded = state.get("loaded")
    if loaded is not None:
        try:
            _finalize_env_artifacts(loaded, logger)
        except Exception:  # noqa: BLE001 - salvage is best effort
            pass
    try:
        crash_guard.disarm()
        stall_guard.disarm()
    except Exception:  # noqa: BLE001
        pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    for handler in list(logging.root.handlers):
        try:
            handler.flush()
        except Exception:  # noqa: BLE001
            pass
    os._exit(1)


def _run_single_mode_body(task: Task, mode: str, overrides: dict | None,
                          state: dict) -> None:
    """The actual mode. ``state`` carries what :func:`_abort_worker` needs to salvage."""
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

    # Die with the parent, enforced by the kernel. Every reaper.reap() call site is on a
    # path the PARENT chooses, and there is no SIGTERM handler -- so a parent killed from
    # OUTSIDE (the shell `timeout` bounding a wave, a pkill, an operator) leaves THIS
    # process, which owns the Isaac/Kit GPU context, orphaned at PPID=1 forever. Measured
    # 2026-07-30: 29 such orphans up to 8h old (52GB RAM) plus three Kit orphans holding
    # ~31GB of VRAM. A handler could not cover the SIGKILL case; PR_SET_PDEATHSIG does.
    from . import pdeathsig

    pdeathsig.arm()

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
    # ⚠ ARMED BEFORE embed() ON PURPOSE -- moved here 2026-07-31 after it demonstrably
    # failed to cover the case it exists for. embed() is where Kit boots, and a boot-time
    # Vulkan OOM WEDGES rather than crashing (no DEVICE_LOST, so crash_guard is blind).
    # Armed after embed, a process that wedged during boot never reached this line, so NO
    # guard existed at any budget: four such processes ran 41 MINUTES at ~128% CPU with no
    # output, ignored SIGTERM, and needed -9. Unlike crash_guard this needs no carb and no
    # Kit -- its signal is our own env-step counter -- so arming it early is safe, and the
    # startup grace (NETT_STALL_STARTUP_GRACE_S) exists precisely to span the boot it now
    # covers. Its exit is os._exit(), which works even when the process ignores signals.
    stall_guard.arm()
    loaded = agent.body.embed(agent.env, run_config)
    # Hand the env to _abort_worker: without this, a failure after embed() lost every
    # artifact, because _finalize_env_artifacts was on the success path only.
    state["loaded"] = loaded
    # Kit is up now (embed builds AppLauncher/SimulationApp). Its startup resets carb
    # logging, so the guard MUST arm after this line -- and must not arm EARLIER, inside
    # embed: the consumer is a synchronous Python callback on every carb message, and
    # arming it before the scene build wedges Kit outright (measured: a known-good
    # 16-env probe hung 30min). Pass the scheduled GPU so crash forensics can label the
    # telemetry snapshot.
    #
    # oom_fatal for DRY RUNS ONLY: a probe deliberately reaches for env counts that
    # cannot fit, so an out-of-VRAM there is its ANSWER, not a crash. It catches an OOM
    # raised from HERE on (e.g. the first optimizer allocation); a scene-build OOM
    # happens before this line and is caught by the probe timeout instead. A real run
    # leaves it off -- a steady-state OOM is not confirmed unrecoverable, and killing a
    # long training on a transient allocation failure would be worse than the hang.
    crash_guard.arm(config.path, device=config.device, oom_fatal=bool(config.dry_run))
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
    # Must precede _exit_worker_cleanly: that ends in os._exit, which discards
    # anything not already on disk (see _finalize_env_artifacts).
    _finalize_env_artifacts(loaded, run_config.logger)
    _exit_worker_cleanly(run_config.logger)


def _compute_eval_num_envs(task: Task) -> int:
    """Largest test-phase num_envs that tiles into a SQUARE camera grid and is a
    multiple of num_brains. Does NOT require dividing the episode total.

    Test is embarrassingly parallel in a way training is not: training's num_envs is
    load-bearing for learning (rollout composition), but test just replays a fixed
    schedule, so the cap is VRAM. Every extra env removes a whole sequential batch.

    TWO HARD CONSTRAINTS (each breaks something if violated):
      * multiple of num_brains -- else BrainTrainer raises (each brain owns one
        contiguous env scope; see brain_scope_sizes);
      * SQUARE tile grid (N in [k^2-k+1, k^2]) -- else the fisheye render is distorted
        (#488, measured). NETT_TEST_ENVS=64 used to resolve to 52 (8x7) and silently
        distorted every test frame.

    DIVISIBILITY IS NO LONGER REQUIRED. When num_envs does not divide the episode
    total the eval runs ceil(total/num_envs)*num_envs episodes; the surplus (index
    >= total) are OVERFLOW episodes that run to fill the fixed eval budget but are not
    logged (nett_env_cfg.test_total_episodes / LogChannel mute), so every design row
    still contributes exactly episodes_test episodes -- no skips, no over-sampling.
    A divisor is still preferred WITHIN the chosen tile-grid band (same parallelism)
    so we avoid overflow for free when we can.
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
        # Prefer the test-phase ceiling the orchestrator measured for THIS phase
        # (max_parallel_envs is training's, and training is deliberately pinned to the
        # recipe's count, which is usually far below what test can run). Falls back to
        # training's ceiling when no test search ran.
        max_envs = getattr(config, "max_test_envs", None) or config.max_parallel_envs
        want = min(max_envs, total_test_episodes) if max_envs is not None else total_test_episodes
        want = max(1, want)

    best, went_up = select_test_num_envs(want, num_brains, total_test_episodes)
    if went_up:
        # Nothing at or below the request qualifies (e.g. want < num_brains). Going UP
        # beats breaking BrainTrainer or rendering distorted.
        config.logger.warning(
            "no square-grid num_envs <= %d is a multiple of num_brains=%d for %d test "
            "episodes; using %d (above the request) so the tile grid stays square and "
            "each brain keeps a whole env scope.",
            want, num_brains, total_test_episodes, best,
        )
    return best


def _training_boundaries(total: int, *, eval_freq: int) -> list[int]:
    """Points at which training is SPLIT INTO SEPARATE SUBPROCESSES.

    ⚠ ``checkpoint_freq`` deliberately does NOT appear here. Saving the model is a side
    effect of training, not a control-flow event: skrl writes ``agent_{timestep}.pt`` from
    INSIDE its training loop via ``cfg.experiment.checkpoint_interval`` (wired in
    ``brain/experiment.py``), which needs no chunking at all.

    It used to add boundaries, and the cost was severe (measured 2026-07-31):
      * a 500-episode run with ``checkpoint_freq=7812`` was split into **33 subprocesses**;
      * each boundary then copied ``final_agent.pt`` -> ``agent_{step}.pt`` via
        ``_copy_final_checkpoints_to_global_step``, redundantly with skrl's own in-loop
        write, and produced **33 byte-identical files** (one MD5 across all of them) --
        i.e. no usable snapshots, and any "learning curve" built from them is one policy
        measured N times;
      * and because every mode subprocess calls ``set_seeds(config.seed)`` with the SAME
        seed, a chunked run restarts its RNG stream at every boundary and is NOT
        trajectory-identical to the same seed run continuously.

    ``eval_freq`` still chunks, because a mid-training *evaluation* genuinely needs the
    trainer to stop. That remains the documented reason to prefer post-hoc checkpoint
    evaluation over ``eval_freq`` (see maintenance/checkpoint-eval-design.md).
    """
    boundaries = {int(total)}
    boundaries.update(range(eval_freq, total + 1, eval_freq))
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


def _finalize_env_artifacts(loaded, logger: logging.Logger) -> None:
    """Flush the env's Python-side outputs BEFORE the worker's hard exit.

    The worker ends via _exit_worker_cleanly -> ``atexit.register(os._exit, 0)``,
    and os._exit skips the interpreter finalization that would flush open files.
    NETTEnv.close() is never reached on that path, so everything it finalises was
    silently lost: the profiler JSON entirely, and the LAST grid-video round's
    manifest (earlier rounds self-finalise when the next round starts). It also
    left the mp4 to be completed by an orphaned ffmpeg after we exit (measured
    still writing ~9 MB six seconds later), which a process-group kill would
    truncate.

    We call ``finalize_artifacts()`` -- NOT ``close()`` -- because close() ends in
    Isaac/Kit teardown, which is the fragile step os._exit exists to bypass.
    finalize_artifacts is Kit-free and idempotent. Never fatal: a teardown problem
    must not fail a finished run.
    """
    obj, seen = loaded, set()
    for _ in range(8):  # unwrap skrl/body wrappers to reach NETTEnv
        if obj is None or id(obj) in seen:
            break
        seen.add(id(obj))
        finalize = getattr(obj, "finalize_artifacts", None)
        if callable(finalize):
            try:
                finalize()
            except Exception:
                logger.exception("env artifact finalize failed (non-fatal)")
            return
        obj = getattr(obj, "unwrapped", None) or getattr(obj, "_env", None)
    logger.debug("no finalize_artifacts() on the env chain; nothing to flush")


def _exit_worker_cleanly(logger: logging.Logger) -> None:
    """Flush Python-side outputs, then bypass Kit's fragile atexit teardown."""
    # ★ ARM THE TEARDOWN BACKSTOP FIRST -- BEFORE the disarms below, which is the
    # instant the post-stepping window opens. Everything from here to process death used
    # to be UNGUARDED by construction, and that is where the "multi-brain run hangs after
    # test" specimens live: stall_guard's watchdog is a daemon thread (CPython stops
    # scheduling it during finalization) and its SIGALRM backstop is armed only inside
    # _trigger, so neither could ever fire here. ⚠ Raising NETT_STALL_TIMEOUT_S cannot
    # substitute for this: a longer grace does not make a stopped thread run.
    stall_guard.arm_teardown_backstop()
    # This IS the clean path — retire both guards so teardown is exactly what it was
    # before they existed. stall_guard MUST be disarmed here: a healthy run stops
    # stepping and then spends real time in analysis/teardown, which a still-armed
    # watchdog would eventually read as a stall.
    crash_guard.disarm()
    stall_guard.disarm()
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
