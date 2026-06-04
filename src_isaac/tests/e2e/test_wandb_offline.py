"""Wandb integration tests against a real Isaac Sim run, mode=offline.

Offline mode goes through the same ``wandb.init``/``wandb.finish`` path as
online mode but writes everything to disk instead of uploading — exactly
the right shape for CI verification of the wandb wiring without needing
``WANDB_API_KEY``.
"""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e_isaac


def _find_wandb_run_dirs(condition_dir: Path) -> list[Path]:
    """Locate ``wandb/{offline-,}run-*`` directories under one condition.

    ``apply_experiment_cfg`` sets ``wandb_kwargs["dir"]=str(output_dir)``
    where ``output_dir == condition_dir``, so all per-brain runs land in
    the same ``condition_dir/wandb/`` parent regardless of brain id.
    """
    wandb_root = condition_dir / "wandb"
    if not wandb_root.exists():
        return []
    prefixes = ("run-", "offline-run-")
    return sorted(
        p for p in wandb_root.iterdir()
        if p.is_dir() and p.name.startswith(prefixes)
    )


def test_wandb_offline_creates_one_run_per_brain(run_nett, e2e_smoke_cfg):
    """Each brain has a single offline wandb run that covers all phases.

    Train and test phases share the same run ID (phase is excluded from the
    hash), so the test subprocess resumes rather than creates a new run.
    For N brains, there are exactly N wandb run directories — one per brain.
    ``reinit='create_new'`` lets multiple brains initialise concurrently in
    the same subprocess without conflicting.
    """
    e2e_smoke_cfg["brain"]["wandb"] = {"mode": "offline", "project": "nett-e2e"}
    e2e_smoke_cfg["episodes"] = {"train": 2}

    output_dir = run_nett(e2e_smoke_cfg)
    condition_dir = output_dir / "Object1"

    run_dirs = _find_wandb_run_dirs(condition_dir)
    expected_count = e2e_smoke_cfg["num_brains"]
    assert len(run_dirs) == expected_count, \
        f"expected {expected_count} wandb run dirs under {condition_dir / 'wandb'}; got {[p.name for p in run_dirs]}"

    for run_dir in run_dirs:
        # The .wandb binary log is the authoritative per-run record — written
        # at init and appended throughout training. wandb-summary.json /
        # files/config.yaml are only flushed during a full wandb.finish,
        # which our ``_exit_worker_cleanly`` may short-circuit by design.
        wandb_logs = list(run_dir.glob("run-*.wandb"))
        assert wandb_logs and wandb_logs[0].stat().st_size > 0, \
            f"no non-empty run-*.wandb log in {run_dir.name}"
        assert (run_dir / "logs" / "debug.log").exists(), \
            f"no debug.log in {run_dir.name}"


def test_wandb_disabled_produces_no_wandb_dirs(run_nett, e2e_smoke_cfg):
    """Mode=disabled produces zero wandb run directories anywhere in output.

    The cfg fixture already defaults to ``mode: disabled``; we keep this
    test explicit so anyone copying the fixture for another suite cannot
    accidentally turn wandb back on without a real assertion failing.
    """
    e2e_smoke_cfg["brain"]["wandb"] = {"mode": "disabled"}
    e2e_smoke_cfg["episodes"] = {"train": 2}

    output_dir = run_nett(e2e_smoke_cfg)

    leaked = [
        str(p) for p in output_dir.rglob("wandb/*")
        if p.is_dir() and (p.name.startswith("run-") or p.name.startswith("offline-run-"))
    ]
    assert not leaked, f"wandb run dirs leaked despite mode=disabled: {leaked}"
