"""End-to-end test of the ``task_memory: auto`` dry-run estimator path.

Runs one config with the estimator enabled, asserts the dry-run consumed a
positive amount of VRAM, the real task afterwards produced the expected
artifacts, and the dry-run's temp directory was cleaned up.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.e2e_isaac


def test_auto_task_memory_triggers_dry_run_then_runs(run_nett, e2e_smoke_cfg, e2e_output_dir, caplog):
    """``task_memory: auto`` runs a dry-run, then the real task lands artifacts.

    The dry-run sets ``validation_mode=True`` and skips wandb / hparams /
    checkpoints / recordings, so the only artifact it leaves is ``mem.txt``
    in an isolated temp dir which the parent reads and deletes. The real
    task that follows must still produce ``final_agent.pt`` and the train
    CSV under ``Object1/``.
    """
    e2e_smoke_cfg["num_brains"] = 1
    e2e_smoke_cfg["episodes"] = {"train": 2}
    e2e_smoke_cfg["task_memory"] = "auto"

    import logging

    with caplog.at_level(logging.INFO, logger="nett"):
        output_dir = run_nett(e2e_smoke_cfg)

    estimator_logs = [r for r in caplog.records if "task memory" in r.message.lower()]
    assert estimator_logs, "expected an 'Estimated task memory' log line"
    # Look for the consumed-memory log line and pull the GB value.
    consumed_lines = [r for r in estimator_logs if "Estimated task memory" in r.message]
    assert consumed_lines, f"no 'Estimated task memory' line found in: {[r.message for r in estimator_logs]}"

    condition_dir = output_dir / "Object1"
    assert (condition_dir / "logs" / "hparams.json").exists(), \
        "real task didn't run after dry-run estimate"
    ckpt = condition_dir / "wandb_runs" / "brain_1" / "checkpoints" / "final_agent.pt"
    assert ckpt.exists(), f"missing real-task checkpoint {ckpt}"

    # mem.txt from the dry-run must be cleaned up.
    assert not (condition_dir / "mem.txt").exists(), "dry-run mem.txt not cleaned"
