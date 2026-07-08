"""Output-artifact contract tests.

A single session-scoped train+test run produces all the artifacts these
tests inspect — pure post-hoc filesystem assertions, no subprocess work
beyond the fixture itself. Anchors the file layout documented in
``CLAUDE.md`` so a directory-structure refactor in NETT-skrl will fail
loudly in CI rather than silently break downstream analysis tooling.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest
import yaml

from .conftest import MINIMAL_SMOKE_CFG


pytestmark = pytest.mark.e2e_isaac


# Canonical Unity LogChannel header order (mirrors
# isaac_lab/tests/test_unity_parity.py:test_csv_column_order_matches_unity).
# Duplicated rather than imported to keep this file's contract explicit —
# any drift between Unity and Isaac is a real bug and the diff should be
# visible right here.
UNITY_CSV_HEADER = [
    "env_id",
    "episode",
    "step",
    "agent.x",
    "agent.y",
    "agent.angle",
    "head.flexion",
    "head.lateral",
    "left.monitor",
    "right.monitor",
    "correct.monitor",
    "experiment.phase",
    "imprint.cond",
    "test.cond",
]


@pytest.fixture(scope="module")
def shared_run_output(tmp_path_factory) -> Path:
    """One train+test run shared by every test in this module."""
    out = tmp_path_factory.mktemp("outputs_module")
    cfg = copy.deepcopy(MINIMAL_SMOKE_CFG)
    cfg_path = out / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    from nett_skrl import NETT

    NETT([str(cfg_path)]).run(output_path=str(out), verbose=False)
    return out / cfg["name"]


def test_run_output_tree_matches_documented_structure(shared_run_output):
    """``{name}/config.yaml`` + per-condition logs + per-brain checkpoints."""
    assert (shared_run_output / "config.yaml").exists()

    condition_dir = shared_run_output / "Object1"
    assert (condition_dir / "logs" / "hparams.json").exists()

    for brain_id in (1, 2):
        ckpt_dir = condition_dir / "wandb_runs" / f"brain_{brain_id}" / "checkpoints"
        assert (ckpt_dir / "final_agent.pt").exists(), f"brain_{brain_id} final_agent.pt missing"
        # skrl writes a tfevents file even when wandb is disabled (it is the
        # transport that the wandb-online path piggybacks on); this is the
        # offline observability backbone.
        tfevents = list((condition_dir / "wandb_runs" / f"brain_{brain_id}").glob("events.out.tfevents.*"))
        assert tfevents, f"brain_{brain_id} produced no tfevents file"


def test_config_yaml_round_trips_back_into_NETT(shared_run_output):
    """Saved config.yaml passes schema validation when re-parsed."""
    from nett_skrl.validate import validate_config

    raw = yaml.safe_load((shared_run_output / "config.yaml").read_text())
    schema = json.loads((Path(__import__("nett_skrl").__file__).parent / "schema.json").read_text())
    validate_config(raw, schema)


def test_hparams_json_records_algorithm_and_lr(shared_run_output):
    """hparams.json captures the keys offline tooling expects."""
    hparams = json.loads((shared_run_output / "Object1" / "logs" / "hparams.json").read_text())
    expected_keys = {
        "algorithm",
        "encoder",
        "algorithm_cfg",
        "checkpoint_freq",
        "envs_per_brain",
        "total_timesteps",
    }
    assert expected_keys.issubset(hparams.keys()), \
        f"missing keys: {expected_keys - hparams.keys()}"
    # hparams stores the resolved class names rather than the registry aliases
    # ("PPO" → "PPO" because the alias matches; "small" → "SmallCNN").
    assert hparams["algorithm"]
    assert hparams["encoder"]
    assert isinstance(hparams["algorithm_cfg"]["learning_rate"], (float, int, str))


def test_csv_columns_match_unity_log_channel_header(shared_run_output):
    """Cross-stack contract: every CSV in this run uses the Unity column order."""
    logs_dir = shared_run_output / "Object1" / "logs"
    csv_paths = list(logs_dir.glob("*.csv"))
    assert csv_paths, "no LogChannel CSVs produced by shared run"
    for path in csv_paths:
        with path.open() as f:
            header = next(csv.reader(f))
        assert header == UNITY_CSV_HEADER, f"{path.name} header drift: {header}"
