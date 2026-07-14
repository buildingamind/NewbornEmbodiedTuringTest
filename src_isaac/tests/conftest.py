"""Shared fixtures for the fast (Isaac-free) unit suite.

Only process-lifecycle plumbing lives here: the DEVICE_LOST crash-guard and the
reaper are the one part of the suite that creates *real* OS processes, so they
need a fixture that guarantees cleanup even when a test fails mid-assert.

Nothing in here imports Isaac Sim or touches a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fault_injection import ProcessOrchard, crashed_run_artifacts


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
