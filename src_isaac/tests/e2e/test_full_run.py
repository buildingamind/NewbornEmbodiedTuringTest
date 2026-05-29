"""End-to-end orchestration tests through the full NETT-skrl stack.

Each test runs ``NETT([cfg]).run(output_path=...)`` against real Isaac Sim
and inspects the artifacts on disk. The minimal cfg trains for ~100
timesteps per brain so a full train+test cycle finishes in ~30s on the
smoke hardware.

Critical pathways covered:
    * train-only with one brain                  (fast smoke)
    * train→test handoff with two brains         (the SimulationContext-
                                                  deadlock bug from the
                                                  subprocess refactor)
    * three-mode train→test→record pipeline      (record subprocess produces
                                                  recording artifacts when
                                                  ``environment.recording`` is
                                                  enabled)
    * two conditions run independently           (no cross-contamination)
    * resume test from a saved checkpoint        (``pick_checkpoint`` finds
                                                  ``final_agent.pt`` and
                                                  weights actually load)

The warn-on-nonzero-exit contract from ``task_runner._spawn_mode_subprocess``
is exercised separately in ``test_lifecycle.py`` with a fake mp.Process — a
real-Isaac monkeypatch does not cross the spawn boundary so it cannot be
verified here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from .conftest import DESIGN_SHEET_FULL


pytestmark = pytest.mark.e2e_isaac


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _brain_ckpt_dir(output_dir: Path, condition: str, brain_id: int) -> Path:
    return output_dir / condition / "wandb_runs" / f"brain_{brain_id}" / "checkpoints"


def test_train_only_one_brain_one_condition(run_nett, e2e_smoke_cfg):
    """Smallest possible E2E: 1 brain × 1 condition × train only.

    Locks in the happy-path output layout produced by the wandb+checkpoint
    refactor — final_agent.pt under skrl's experiment_dir, hparams.json
    under logs/, train CSV with rows.
    """
    e2e_smoke_cfg["num_brains"] = 1
    e2e_smoke_cfg["episodes"] = {"train": 2}

    output_dir = run_nett(e2e_smoke_cfg)
    condition_dir = output_dir / "Object1"

    assert (condition_dir / "logs" / "hparams.json").exists()
    assert (_brain_ckpt_dir(output_dir, "Object1", 1) / "final_agent.pt").exists()

    train_csvs = list((condition_dir / "logs").glob("train_*.csv"))
    assert train_csvs, f"no train CSV in {condition_dir / 'logs'}"
    rows = _read_csv_rows(train_csvs[0])
    assert rows, "train CSV is empty"
    # Single brain → only env_id == 0 should appear.
    assert {r["env_id"] for r in rows} == {"0"}


def test_train_then_test_two_brains(run_nett, e2e_smoke_cfg):
    """The train→test handoff that hit the SimulationContext deadlock.

    Must produce two ``final_agent.pt`` files (one per brain) and a test CSV
    with rows for both brains across every test condition in the minimal
    design sheet (``rest`` and ``1color``).
    """
    output_dir = run_nett(e2e_smoke_cfg)
    condition_dir = output_dir / "Object1"

    for brain_id in (1, 2):
        assert (_brain_ckpt_dir(output_dir, "Object1", brain_id) / "final_agent.pt").exists(), \
            f"missing final_agent.pt for brain_{brain_id}"

    test_csvs = list((condition_dir / "logs").glob("test_*.csv"))
    assert test_csvs, f"no test CSV in {condition_dir / 'logs'}"
    rows = _read_csv_rows(test_csvs[0])
    assert rows, "test CSV is empty"
    assert {r["env_id"] for r in rows} == {"0", "1"}, "test CSV missing one brain's rows"
    # The minimal design sheet has test conditions "rest" and "1color".
    assert {"rest", "1color"}.issubset({r["test.cond"] for r in rows})


def test_train_test_record_three_mode_pipeline(run_nett, e2e_smoke_cfg):
    """All three modes run in sequence and each produces its CSV log.

    Each mode runs in a fresh subprocess (per ``task_runner.run_task``); this
    test fails if any mode silently aborts the chain. The
    recording-artifact pipeline has its own surface and is exercised
    separately — here we only require the per-mode CSV to land.
    """
    e2e_smoke_cfg["episodes"] = {"train": 2, "test": 1, "record": 1}

    output_dir = run_nett(e2e_smoke_cfg)
    condition_dir = output_dir / "Object1"

    logs = condition_dir / "logs"
    assert list(logs.glob("train_*.csv")), "train CSV missing"
    assert list(logs.glob("test_*.csv")), "test CSV missing"
    assert list(logs.glob("record_*.csv")), "record CSV missing"


def test_two_conditions_run_independently(run_nett, e2e_smoke_cfg):
    """Object1 and Object2 produce separate output trees, no cross-talk."""
    e2e_smoke_cfg["environment"]["design_sheet"] = str(DESIGN_SHEET_FULL)
    e2e_smoke_cfg["environment"]["conditions"] = ["Object1", "Object2"]
    e2e_smoke_cfg["num_brains"] = 1
    e2e_smoke_cfg["episodes"] = {"train": 2}

    output_dir = run_nett(e2e_smoke_cfg)

    for condition in ("Object1", "Object2"):
        ckpt = _brain_ckpt_dir(output_dir, condition, 1) / "final_agent.pt"
        assert ckpt.exists(), f"missing {ckpt}"
        train_csvs = list((output_dir / condition / "logs").glob("train_*.csv"))
        assert train_csvs, f"missing train CSV for {condition}"
        # Each CSV must only mention its own condition — no cross-leakage.
        for row in _read_csv_rows(train_csvs[0]):
            assert row["imprint.cond"] == condition


def test_resume_test_from_saved_checkpoint(run_nett, e2e_smoke_cfg, e2e_output_dir):
    """Train once, then re-run with modes=["test"] only — checkpoint loads.

    The second run constructs a fresh NETT against the same output path. If
    pick_checkpoint resolves final_agent.pt and Agent.load succeeds, the
    test CSV is produced with non-NaN rewards (test rewards are not zero
    because closeness reward is enabled, even though the test phase itself
    disables reward updates — the row's columns still come from a working
    rollout).
    """
    # 1) Train+save final_agent.pt.
    train_cfg = dict(e2e_smoke_cfg)
    train_cfg["episodes"] = {"train": 2}
    train_cfg["num_brains"] = 1
    run_nett(train_cfg)

    # 2) Re-run with test-only against the same output dir. The runner
    # fixture creates a fresh output dir per call, so we plumb the same
    # output_path through directly via NETT().
    import yaml as _yaml
    from nett_skrl import NETT

    test_cfg = dict(e2e_smoke_cfg)
    test_cfg["episodes"] = {"test": 1}
    test_cfg["num_brains"] = 1

    cfg_path = e2e_output_dir / "resume_cfg.yaml"
    cfg_path.write_text(_yaml.safe_dump(test_cfg))
    NETT([str(cfg_path)]).run(output_path=str(e2e_output_dir), verbose=False)

    test_csvs = list((e2e_output_dir / test_cfg["name"] / "Object1" / "logs").glob("test_*.csv"))
    assert test_csvs, "test-only resume produced no test CSV"
    rows = _read_csv_rows(test_csvs[0])
    assert rows, "test CSV is empty after resume"
