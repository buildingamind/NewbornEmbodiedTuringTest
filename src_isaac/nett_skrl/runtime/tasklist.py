"""Task construction helpers — one Task per imprint condition."""

from __future__ import annotations

from pathlib import Path

from .device import visible_device_scope
from .lifecycle import register_reaper, unregister_reaper
from .reap import TaskReaper
from .task import Task, set_seeds


def build_tasks(
    brain,
    body,
    env,
    num_brains: int,
    num_envs: int,
    conditions: list[str],
    output_dir: Path,
    modes: list[str],
    episodes: dict[str, int] | None,
    memory: float,
    brain_id_offset: int = 0,
    eval_freq: int | None = None,
    max_parallel_envs: int | None = None,
    max_test_envs: int | None = None,
) -> list[Task]:
    return [
        Task(
            brain, body, env, condition, output_dir, modes,
            episodes=episodes,
            memory=memory,
            num_brains=num_brains,
            num_envs=num_envs,
            brain_id_offset=brain_id_offset,
            eval_freq=eval_freq,
            max_parallel_envs=max_parallel_envs,
            max_test_envs=max_test_envs,
        )
        for condition in conditions
    ]


def validate_tasklist(tasks: list[Task], close_env: bool = True) -> None:
    """Smoke-validate by loading each task's env once through its body.

    ``close_env=False`` skips the teardown, for callers that exit via ``os._exit``
    immediately afterwards. ⚠ NOT an optimisation: ``NETTEnv.close()`` ends in
    ``super().close()``, Kit's teardown, WHICH NEVER RETURNS in this setup -- the same
    reason ``task_runner`` bypasses it with ``os._exit`` and repoA's Kit test driver
    neutralises ``SimulationApp.close``. Called with the default here, validation hangs
    forever after a successful check (measured 2026-07-28: 10 min at ~0% CPU, no output).
    The OS reclaims the process's resources on exit regardless.
    """
    for task in tasks:
        config = task.config
        # Match the real run path (task_runner): seed before building the env so
        # the smoke-validation load is deterministic too (Critic W2).
        set_seeds(config.seed)
        run_config = config.for_mode("train")
        log_path = config.path / "logs"
        log_path.mkdir(exist_ok=True, parents=True)
        task.agent.body.adjust_to_agent(
            task.agent.env,
            num_brains=config.num_brains,
            num_envs=config.num_envs,
        )
        loaded = task.agent.env.load(run_config)
        try:
            loaded = task.agent.body.wrap(loaded)
            obs, _ = loaded.reset()
            if not hasattr(obs, "shape") and not isinstance(obs, dict):
                raise TypeError(f"Unexpected obs type from wrapped env: {type(obs)}")
        finally:
            if close_env and hasattr(loaded, "close"):
                loaded.close()


def _validation_child(tasks: list[Task]) -> None:
    """Child entrypoint: validate, then leave via ``os._exit`` like every Kit worker here.

    ``os._exit`` because Kit's teardown is not reliably reusable in-process -- the same
    reason ``task_runner`` ends every mode subprocess this way. Streams are flushed first:
    ``os._exit`` skips interpreter finalisation, so anything unflushed is simply lost (the
    repo has been bitten by exactly that, see nett_env.finalize_artifacts).
    """
    import os
    import sys
    import traceback

    # ⚠ THIS CHILD BOOTS KIT AND HOLDS A GPU CONTEXT, and unlike every mode subprocess it
    # is a DIRECT child of the driver, not of a pool worker -- so it was the one
    # Kit-booting process in the tree with no orphan guard at all. Killing the driver
    # during "Validating tasks…" stranded it at PPID=1 holding VRAM. Same net as
    # task_runner._run_single_mode: kernel-enforced, so it covers the -9 case too.
    from . import pdeathsig

    pdeathsig.arm()

    code = 0
    try:
        # close_env=False: os._exit below makes teardown unnecessary, and calling it would
        # HANG -- Kit's close never returns. See validate_tasklist's docstring.
        validate_tasklist(tasks, close_env=False)
    except BaseException:  # noqa: BLE001 - the traceback IS the report
        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


def validate_tasklist_subprocess(tasks: list[Task], logger=None, device: int = 0) -> int:
    """Run :func:`validate_tasklist` in a dedicated spawn process. Returns its exit code.

    ★ WHY NOT IN THE POOL (measured 2026-07-28). Validation used to be submitted to the
    shared ProcessPoolExecutor, where it booted Kit. **Kit cannot be booted in a pool
    worker on this host at all** -- it dies during boot and takes the pool with it
    (BrokenProcessPool), which fails every subsequent task rather than the one call.
    Isolated control, one variable changed: identical boot in a plain ``mp.Process``
    returned BOOTED in 9.2s with exitcode 0, while the same boot in a
    ``ProcessPoolExecutor`` worker (same spawn context, same host, same ambient kvdb lock)
    died. Under the old fork-based pool it segfaulted instead, on the parent's argv.

    So validation now uses the mechanism the rest of the runtime already uses for anything
    that touches Kit: one fresh spawn process, exiting via ``os._exit``.
    """
    import multiprocessing as mp

    # Give every task a provisional device BEFORE validating. Validation runs before
    # placement, so config.device is otherwise None and the env cannot construct -- which
    # is why this check could never have passed even once. _assign_task overwrites this
    # with the real placement immediately afterwards. Done here rather than in nett.py so
    # "validate the tasklist" stays ONE seam: a caller (or a test) that stubs this function
    # out skips the whole step, device assignment included.
    for task in tasks:
        task.set_device(device)

    ctx = mp.get_context("spawn")

    # ★ ONE TASK PER PROCESS, not one process for the list. Kit allows exactly one
    # SimulationContext per interpreter, and it cannot be torn down and rebuilt:
    #   * validating two tasks in one child -> "Simulation context already exists.
    #     Cannot create a new one." on the second (measured 2026-07-28 by the only e2e
    #     test that uses two conditions);
    #   * and closing the first to make room is not an option either -- NETTEnv.close()
    #     ends in Kit's teardown, which NEVER RETURNS (see validate_tasklist).
    # So the boot budget is one per task. That is the same rule task_runner follows for
    # every mode, and it is the reason it spawns rather than loops.
    # COST, stated plainly: validation adds ONE Kit boot (~10s) PER CONDITION to every run.
    for task in tasks:
        # A TaskReaper, registered for the duration, so a SIGINT/SIGTERM arriving at the
        # DRIVER while this Kit boot is in flight reaps it with ownership evidence rather
        # than leaving a VRAM-holding orphan. launch_scope() stamps the token that makes
        # the child attributable even after it reparents to PPID=1.
        reaper = TaskReaper(
            task_key=f"{getattr(task.config, 'name', '?')}/"
                     f"{getattr(task.config, 'condition', '?')}/validate",
            device=device,
            logger=logger,
        )
        register_reaper(reaper)
        try:
            proc = ctx.Process(target=_validation_child, args=([task],), daemon=False)
            # ⚠ THE PIN APPLIES HERE TOO. Without it the child gets the PHYSICAL index and
            # Kit hangs on "GPU N requested. GPUs other than cuda:0 are not currently
            # supported" -- exactly what happened when this subprocess was first added
            # (2026-07-28).
            with visible_device_scope(device), reaper.launch_scope():
                proc.start()
            reaper.adopt(proc.pid)
            proc.join()
        finally:
            unregister_reaper(reaper)
        code = proc.exitcode if proc.exitcode is not None else 1
        if code != 0:
            if logger is not None:
                logger.error(
                    "Validation FAILED for condition %s (exit code %s)",
                    getattr(task.config, "condition", "?"), code,
                )
            return code
    if logger is not None:
        logger.info("Validation passed for %d task(s)", len(tasks))
    return 0
