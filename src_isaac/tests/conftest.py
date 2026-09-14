"""Shared fixtures for the fast (Isaac-free) unit suite.

Only process-lifecycle plumbing lives here: the DEVICE_LOST crash-guard and the
reaper are the one part of the suite that creates *real* OS processes, so they
need a fixture that guarantees cleanup even when a test fails mid-assert.

Nothing in here imports Isaac Sim or touches a GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _pin_math_threads() -> None:
    """Pin BLAS/OpenMP to one thread per process, before torch builds its pool.

    ⚠ THIS IS WHAT MAKES ``-n`` USABLE ON THIS SUITE. Measured 2026-09-14 on
    chicken (96 cores), 1131 tests:

        serial, unpinned      168.2s wall / 300m32s CPU
        -n 8,   unpinned      228.8s wall   <-- SLOWER THAN SERIAL
        serial, pinned        166.4s wall /   1m20s CPU
        -n 8,   pinned         55.5s wall

    The suite does ~80 CPU-seconds of real work; the rest is waiting on
    watchdog timeouts and process death. Unpinned, every worker defaults to
    one OpenMP thread per core, so 8 workers oversubscribe 96 cores by 8x and
    spend their time in contention -- the same failure the e2e block of
    scripts/hooks/pre-push avoids with NETT_KIT_THREADS=cores/jobs.

    Pinning costs nothing serially (166.4s vs 168.2s, inside run-to-run noise),
    so it is unconditional rather than gated on whether -n was passed.
    ``setdefault`` leaves an explicit operator setting alone.
    """
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    # Only if torch is already up: the env vars govern a later import, and
    # importing torch here just to pin it would cost more than it saves.
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


_pin_math_threads()

from fault_injection import ProcessOrchard, crashed_run_artifacts  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_crash_guard_state():
    """Snapshot + restore crash_guard's module-level state around every test.

    crash_guard keeps process-wide mutable state (armed flag, trigger latch, and
    the crash-forensics accumulators populated from the render-thread callback).
    A leak between tests would let one test's injected crash lines bleed into the
    next. Restoring afterwards keeps the suite order-independent.
    """
    from nett_skrl.runtime import crash_guard as cg

    saved = (
        cg._armed,
        cg._logger_handle,
        list(cg._artifact_dirs),
        cg._trigger_reason,
        cg._device,
        list(cg._crash_artifact_paths),
        cg._pagefault_detail,
        cg._triggered.is_set(),
    )
    try:
        yield
    finally:
        cg._armed, cg._logger_handle = saved[0], saved[1]
        cg._artifact_dirs[:] = saved[2]
        cg._trigger_reason, cg._device = saved[3], saved[4]
        cg._crash_artifact_paths[:] = saved[5]
        cg._pagefault_detail = saved[6]
        (cg._triggered.set if saved[7] else cg._triggered.clear)()


@pytest.fixture
def orchard() -> ProcessOrchard:
    """Owns every process a test spawns; kills them all on teardown.

    The finalizer runs on pass, fail, and error alike, so a failed assertion can
    never leak a hung child onto a box that is also running real Isaac jobs.
    """
    farm = ProcessOrchard()
    try:
        yield farm
    finally:
        farm.kill_all()


@pytest.fixture
def crashed_run_dir(tmp_path: Path) -> Path:
    """A run output tree shaped like one that died mid-TEST to DEVICE_LOST."""
    return crashed_run_artifacts(tmp_path / "run")
