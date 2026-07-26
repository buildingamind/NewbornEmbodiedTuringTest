"""Shared fixtures for the Isaac Lab + skrl end-to-end suite.

These tests spin up the full NETT stack against real Isaac Sim — each test
forks at least one subprocess that imports omni.kit. To keep the fast unit
suite runnable without Isaac/GPU we gate the whole tree behind two markers
(`e2e_isaac` and `e2e_perf`) and auto-skip when prerequisites are missing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import pytest
import yaml


from _repo_paths import repo_a_root

# $NETT_REPO_A overrides the sibling default; `_isaac_missing` below already
# auto-skips the tree when the assets it points at are absent.
PRIVATE_ROOT = repo_a_root()
DESIGN_SHEET_MINIMAL = PRIVATE_ROOT / "isaac_lab" / "assets" / "design_sheets" / "binding_minimal.csv"
DESIGN_SHEET_FULL = PRIVATE_ROOT / "isaac_lab" / "assets" / "design_sheets" / "binding.csv"
MEDIA_ROOT = PRIVATE_ROOT / "isaac_lab" / "assets" / "videos"
BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"
GOLDEN_PATH = BENCHMARKS_DIR / "golden.json"


# Minimal cfg cloned from examples/test_phase.yaml. Kept inline so a stray
# edit to the example file does not silently change the e2e baseline.
MINIMAL_SMOKE_CFG: dict = {
    "name": "e2e",
    "environment": {
        "design_sheet": str(DESIGN_SHEET_MINIMAL),
        "media_root": str(MEDIA_ROOT),
        "conditions": ["Object1"],
        "headless": True,
        "input_resolution": 64,
        "reward_types": ["closeness"],
    },
    "brain": {
        "algorithm": "PPO",
        "encoder": "small",
        "encoder_cfg": {"trainable": True},
        "algorithm_cfg": {
            "rollouts": 64,
            "mini_batches": 2,
            "learning_rate": 3e-4, #1.0e-5,
        },
        "wandb": {"mode": "disabled"},
    },
    "num_brains": 2,
    "episodes": {"train": 2, "test": 1},
    "steps_per_episode": 50,
    "task_memory": 4,  # explicit GB so e2e tests skip the dry-run estimator
}


# ---------------------------------------------------------------------------
# pytest plumbing
# ---------------------------------------------------------------------------


def pytest_addoption(parser) -> None:
    """``--update-golden`` rewrites benchmarks/golden.json from this run."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Overwrite e2e benchmarks/golden.json with values from this run.",
    )


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "e2e_isaac: end-to-end test requiring GPU + Isaac Sim"
    )
    config.addinivalue_line(
        "markers", "e2e_perf: performance/convergence test that reads/writes golden.json"
    )


# Auto-skip the whole e2e tree when its prerequisites are missing, so a
# developer without Isaac Sim / GPU can still run the rest of the suite.
def _isaac_missing() -> str | None:
    try:
        import torch  # noqa: F401
    except Exception as e:
        return f"torch import failed: {e}"
    import torch as _t
    if not _t.cuda.is_available():
        return "CUDA not available"
    try:
        import isaacsim  # noqa: F401
    except Exception as e:
        return f"isaacsim import failed: {e}"
    try:
        import nett_isaac  # noqa: F401
    except Exception as e:
        return f"nett_isaac import failed: {e}"
    if not DESIGN_SHEET_MINIMAL.exists():
        return f"missing design sheet: {DESIGN_SHEET_MINIMAL}"
    if not MEDIA_ROOT.exists():
        return f"missing media root: {MEDIA_ROOT}"
    return None


def pytest_collection_modifyitems(config, items) -> None:
    skip_reason = _isaac_missing()
    if not skip_reason:
        return
    skip = pytest.mark.skip(reason=f"e2e prerequisites missing: {skip_reason}")
    for item in items:
        if "tests/e2e" in item.nodeid and any(
            item.iter_markers(name=name) for name in ("e2e_isaac", "e2e_perf")
        ):
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_output_dir(tmp_path_factory) -> Path:
    """A clean tmp output root per test."""
    return tmp_path_factory.mktemp("e2e_out")


@pytest.fixture
def e2e_smoke_cfg(e2e_output_dir) -> dict:
    """Deep copy of the minimal smoke cfg so tests can mutate freely."""
    cfg = copy.deepcopy(MINIMAL_SMOKE_CFG)
    return cfg


@pytest.fixture
def run_nett(e2e_output_dir) -> Callable[[dict], Path]:
    """Write cfg to yaml + invoke NETT(cfg).run(); return the output dir.

    Returns the per-config output dir (``e2e_output_dir / cfg['name']``) so
    tests can assert on artifacts without rebuilding the path themselves.
    """

    def _runner(cfg: dict) -> Path:
        cfg_path = e2e_output_dir / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        # Import lazily — top-level conftest module load must not trigger
        # Isaac Sim imports for non-e2e test runs.
        from nett_skrl import NETT

        NETT([str(cfg_path)]).run(output_path=str(e2e_output_dir), verbose=False)
        return e2e_output_dir / cfg["name"]

    return _runner


# ---------------------------------------------------------------------------
# Golden numbers helpers (used by test_perf.py)
# ---------------------------------------------------------------------------


def load_golden() -> dict:
    if not GOLDEN_PATH.exists():
        return {}
    return json.loads(GOLDEN_PATH.read_text())


def save_golden(values: dict) -> None:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def golden(request) -> dict:
    """Read benchmarks/golden.json. Use ``--update-golden`` to rewrite it."""
    return load_golden()


@pytest.fixture
def update_golden(request) -> bool:
    return bool(request.config.getoption("--update-golden"))
