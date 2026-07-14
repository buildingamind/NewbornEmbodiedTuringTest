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
