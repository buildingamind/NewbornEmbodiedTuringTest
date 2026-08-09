"""Shared fixtures for the Isaac Lab + skrl end-to-end suite.

These tests spin up the full NETT stack against real Isaac Sim — each test
forks at least one subprocess that imports omni.kit. To keep the fast unit
suite runnable without Isaac/GPU we gate the whole tree behind two markers
(`e2e_isaac` and `e2e_perf`) and auto-skip when prerequisites are missing.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Callable

import pytest
import yaml


from _repo_paths import (
    SRC_ISAAC,
    repo_a_root,
    stimulus_library,
    stimulus_library_candidates,
)

# $NETT_REPO_A overrides the sibling default; `_isaac_missing` below already
# auto-skips the tree when the assets it points at are absent.
PRIVATE_ROOT = repo_a_root()

# ⚠ UNTIL 2026-07-27 THESE DEFAULTS HAD NEVER RESOLVED ON ANY HOST, so the whole e2e tree
# auto-skipped with an accurate-but-terminal reason from the day it was written — the first
# time it ever executed was 2026-07-27, and it was rotted in three places. Two assets, two
# different homes, and the distinction is deliberate:
#
#   * binding_minimal.csv is a purpose-built TEST FIXTURE (Object1, two test rows). It is
#     vendored in repoA at assets/design_sheets/ so the smoke tests resolve with NO env var.
#   * The FULL sheet is EXPERIMENT DATA. Its real name is DesignSheet_Binding.csv and it
#     lives with the stimuli it references, at <workspace>/videos/binding/ — outside both
#     repositories, because the .mov clips it names are far too large to vendor.
#     `binding.csv` was only ever an alias for it in this file; no such file has ever
#     existed. Defaulting to the real path keeps ONE source of truth: a vendored copy of an
#     experiment design would drift from the sheet the runs actually use, silently.
#
# MEDIA_ROOT IS RESOLVED AGAINST THE SHEET, NOT PINNED TO ONE DIRECTORY (2026-07-29).
# A media root is only "the right one" relative to a design sheet: it has to contain EVERY
# clip that sheet names. Picking a fixed directory and hoping is what broke twice —
#
#   * pinned to a path that never existed → whole tree auto-skipped (fixed 2026-07-27);
#   * then pinned to repoA's isaac_lab/assets/videos, on the belief that its shipped pair
#     {O1_imprint.mov, White.mov} "is exactly the pair binding_minimal.csv names". It is
#     not: the sheet's `1color` test row also names O1_1Ca_1.mov, which repoA does not
#     ship. That was survivable while a missing clip was a logger.warning, i.e. while the
#     tests rendered BLANK MONITORS and passed; repoA 400683cd made unresolvable media a
#     hard error, at which point every test that reaches the test phase FAILS on a stock
#     checkout. A green run and a red run for the same reason — hence resolve, don't pin.
#
# `_resolve_media_root` walks the candidates in order and returns the first that satisfies
# the sheet, so the vendored fixture clips are still preferred when they are enough and the
# workspace stimulus library is used when they are not. Nothing is downloaded or built; the
# clips are experiment data far too large to vendor in full.
#
# Overrides, for a host that stores them elsewhere:
#   NETT_DESIGN_SHEET_MINIMAL / NETT_DESIGN_SHEET_FULL / NETT_MEDIA_ROOT
# NETT_MEDIA_ROOT is honoured verbatim and is NOT validated — an explicit path is a
# deliberate choice, and silently overruling it would be worse than the error it causes.
# ⚠ These are REAL training runs: minutes, not seconds. Do not set them inside a pre-push
# hook without knowing that.
def _asset(env_var: str, default: Path) -> Path:
    override = os.environ.get(env_var)
    return Path(override) if override else default


def _sheet_clips(sheet: Path) -> set[str]:
    """Every clip filename a design sheet references, or empty if unreadable.

    Mirrors ``nett_isaac.condition_manager.ConditionManager.video_set`` — columns 6 and 7
    (LeftMonitor / RightMonitor) of the 7-column row — but with the stdlib only, because
    this module is imported during the plain unit run, where ``nett_isaac`` need not be
    importable. Returning empty on any parse trouble makes the caller fall back to the
    first candidate, i.e. to the previous behaviour, rather than fail collection.
    """
    import csv

    try:
        with sheet.open(newline="") as f:
            rows = list(csv.reader(f))[1:]  # drop header
    except OSError:
        return set()
    return {
        cell.strip().strip('"').strip()
        for row in rows
        if len(row) >= 7
        for cell in row[5:7]
        if cell.strip()
    }


def _resolve_media_root(sheet: Path, candidates: tuple[Path, ...]) -> Path:
    """First candidate holding every clip ``sheet`` names; else the first candidate.

    The fallback is deliberate: when nothing satisfies the sheet, the tree should skip or
    fail naming a real directory, not silently pick the least-wrong one. ⚠ "Should skip"
    is now enforced -- see ``_media_incomplete``; for two weeks nothing checked, so the
    fallback was reached silently and the run died minutes in on the first clip repoA
    does not ship.
    """
    clips = _sheet_clips(sheet)
    if clips:
        for root in candidates:
            if all((root / clip).exists() for clip in clips):
                return root
    return candidates[0]


#: ⚠ THE SHARED RESOLVER, DELIBERATELY NOT A LOCAL COPY. ``tests/_repo_paths`` owns the
#: walk-up because ``test_design.py`` needs exactly the same one to find the binding
#: sheet. The first version of this fix lived only here, and that left test_design
#: silently SKIPPING two tests on every worktree run -- the same root cause, fixed at one
#: of its two sites. Re-exported under the private name because tests import it from this
#: conftest. See :func:`_repo_paths.stimulus_library_candidates` for the measurement.
_stimulus_library_candidates = stimulus_library_candidates


_DS = PRIVATE_ROOT / "isaac_lab" / "assets" / "design_sheets"
_LIBRARIES = _stimulus_library_candidates()
#: Nearest stimulus library, or the historical fixed guess when none is on disk (kept so
#: a skip message still names the path this tree has always documented).
_VIDEOS = stimulus_library()
DESIGN_SHEET_MINIMAL = _asset("NETT_DESIGN_SHEET_MINIMAL", _DS / "binding_minimal.csv")
DESIGN_SHEET_FULL = _asset("NETT_DESIGN_SHEET_FULL", _VIDEOS / "DesignSheet_Binding.csv")
#: Candidate media roots, most-vendored first: repoA's fixture clips, then the workspace
#: stimulus library the experiment sheets actually reference.
MEDIA_ROOTS = (PRIVATE_ROOT / "isaac_lab" / "assets" / "videos",) + tuple(
    lib / "videos" for lib in (_LIBRARIES or (_VIDEOS,))
)
MEDIA_ROOT = _asset("NETT_MEDIA_ROOT", _resolve_media_root(DESIGN_SHEET_MINIMAL, MEDIA_ROOTS))
#: The full sheet names both imprint objects, so it needs its own resolution — the minimal
#: fixture's root will not cover it. ``test_full_run`` swaps BOTH when it swaps the sheet.
MEDIA_ROOT_FULL = _asset("NETT_MEDIA_ROOT", _resolve_media_root(DESIGN_SHEET_FULL, MEDIA_ROOTS))


def _media_incomplete(sheet: Path, root: Path) -> str | None:
    """Names the clips ``sheet`` needs that ``root`` does not have, or None.

    ⚠ THIS IS A COLLECTION-TIME PREREQUISITE, NOT A NICETY. ``_resolve_media_root``
    falls back to the first candidate when nothing satisfies the sheet, and nothing used
    to notice: the tree collected, trained for minutes, and then blew up in the TEST
    phase inside ``NETTEnv.__init__`` on the first unresolvable clip. Until repoA
    ``nett_env`` learned to abandon a half-built env, that constructor failure did not
    even surface as a failure -- Kit's teardown ran against the partial env and HUNG the
    worker forever (measured 2026-08-09). Detecting it here costs a few ``stat`` calls
    and turns a multi-minute mystery into an accurate skip reason.

    Skipped entirely when ``NETT_MEDIA_ROOT`` is set: that path is documented as honoured
    verbatim, and second-guessing a deliberate choice is exactly what this file says not
    to do.
    """
    if os.environ.get("NETT_MEDIA_ROOT"):
        return None
    missing = sorted(clip for clip in _sheet_clips(sheet) if not (root / clip).exists())
    if not missing:
        return None
    return (
        f"media root {root} is missing {len(missing)} clip(s) that {sheet.name} names "
        f"({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}); tried "
        f"{[str(p) for p in MEDIA_ROOTS]} — set $NETT_MEDIA_ROOT to a directory holding "
        "every clip the sheet references"
    )


def _e2e_device() -> int:
    """Which PHYSICAL gpu this process's runs pin to.

    `$NETT_E2E_DEVICE` wins outright. Otherwise, under `pytest -n` each xdist worker takes
    its OWN card, derived from `PYTEST_XDIST_WORKER` ("gw0" -> 0, "gw3" -> 3) modulo the
    visible device count. That is what makes the tree parallel: every test here is a real
    training run, so serialised on one card the 23 of them cost ~17 min, while one run per
    GPU turns the wall clock into (tests / GPUs) rounds.

    ⚠ ONE RUN PER CARD, NOT MORE. Do not raise `-n` above the GPU count expecting more
    throughput: two Kit processes on one card contend for the same Vulkan/RTX queue, and
    VRAM does not scale linearly. Serial (no xdist) still means GPU 0, unchanged.
    """
    explicit = os.environ.get("NETT_E2E_DEVICE")
    if explicit:
        return int(explicit)
    worker = os.environ.get("PYTEST_XDIST_WORKER")  # set only under `-n`
    if not worker or not worker.startswith("gw"):
        return 0
    try:
        import torch
        n = torch.cuda.device_count() or 1
    except Exception:
        n = 1
    return int(worker[2:]) % n


#: Resolved once at import; every `NETT(...).run()` in this tree passes it as devices=[...].
E2E_DEVICE = _e2e_device()

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
    # An EXISTING media root that does not hold every clip the sheet names is the
    # failure mode that actually bites — see _media_incomplete.
    incomplete = _media_incomplete(DESIGN_SHEET_MINIMAL, MEDIA_ROOT)
    if incomplete:
        return incomplete
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

        # (The lazy `from nett_skrl import NETT` that used to sit here was dead — the run
        # goes through `run_nett_tolerating_stalls`, which does its own lazy import. The
        # invariant it guarded still holds and still matters: top-level conftest module
        # load must not trigger Isaac Sim imports for non-e2e runs.)

        # An EXPLICIT device pins each worker to a known card so placement is deterministic
        # run to run. ⚠ This passes E2E_DEVICE — a PHYSICAL index (`xdist worker index %
        # device_count`), with no CUDA_VISIBLE_DEVICES set — i.e. no-CVD + devices=[<phys>].
        # That is the convention that is correct both in a bare spawn and inside NETT.run();
        # the CVD=<phys> + devices=[0] pairing that `examples/_smoke_pin_launch.py` calls
        # "USD-safe" is NOT safe in a bare spawn (it lands on physical GPU 0 — measured by
        # maintenance/diagnostics/device_pin_isolation.py) even though it works inside
        # NETT.run(). Do not switch this tree to it. See blueprint.md §SESSION 2026-07-31b
        # item 7 — the primitive and the full path disagree and the mechanism is unresolved.
        # ⚠ HISTORY WORTH KEEPING: this used to be the ONLY thing standing between
        # this tree and an indefinite hang, because nett.py `set_device(most_free_gpu)`
        # handed Kit a PHYSICAL gpu index (pynvml IGNORES CUDA_VISIBLE_DEVICES) and Kit's
        # usdrt scenegraph supports ONLY cuda:0:
        #   "UsdStage::SelectPrims: GPU 3 requested. GPUs other than cuda:0 are not
        #    currently supported"
        # The run then HANGS at the Fabric XFormPrimView with no error, indefinitely.
        # ⚠ IT IS HOST-STATE DEPENDENT, which is why this tree could pass for months and
        # then wedge: the picker only strays off GPU0 when GPU0 is the busier card.
        # ⚠ Passing a non-zero PHYSICAL index here is safe only because `task_runner`
        # applies `visible_device_scope(device)` in the child, so the target card re-indexes
        # to cuda:0 and usdrt's constraint above is satisfied. That scope is load-bearing.
        # examples/campaign_train.py passes devices=[device] the same way. To aim the
        # tests at a specific physical GPU, set CUDA_VISIBLE_DEVICES in the SHELL: the
        # single visible card re-indexes to cuda:0 and this pin still holds.
        # MEASURED: the same workload wedged >26 min unpinned, and COMPLETES IN ~110s here.
        # ★ SINCE 2026-07-28 THE RUNTIME PINS EVERY TASK ITSELF (task_runner.
        # _visible_device_scope + runtime/device.py), so an unpinned run no longer hangs --
        # this is now determinism, not the sole guard. Keep it anyway: a test that picks a
        # different GPU per run is a test whose timings and VRAM behaviour drift.
        # Retries only a pure render-pump-wedge failure; a real DEVICE_LOST still fails.
        used = run_nett_tolerating_stalls(cfg_path, e2e_output_dir, [E2E_DEVICE])
        return used / cfg["name"]

    return _runner


# ---------------------------------------------------------------------------
# Stall tolerance: the gate must fail on the DIFF, not on a known infra wedge
# ---------------------------------------------------------------------------


def run_nett_tolerating_stalls(cfg_path, output_path, devices, attempts: int = 2):
    """Run NETT, retrying ONLY when every casualty was the Kit render-pump wedge.

    WHY THIS EXISTS
        Measured 2026-07-30 on an idle 8-GPU host: **46% of cells wedge** (15 of 32,
        across four 8-cell arms). Every e2e fixture that performs a real run therefore
        fails on a coin flip, and one did exactly that on a `git push` -- the wedge took
        out the module-scoped fixture and errored all four tests in test_outputs.py at
        setup, blocking a push whose diff was fine.

        The wedge is upstream and unfixed: it is Kit spinning inside ``_app.update()``
        with the GPU idle, emitting no DEVICE_LOST. Two attempts to move it with
        configuration both came back negative (carb.tasking 32 vs 8; test canvas 64 vs
        16 envs), and identical configs swing 2/8..6/8 run to run. So the gate has to
        tolerate it or it does not measure the diff.

    WHY IT IS NARROW
        Retries ONLY if *every* failure in the aggregate is a :class:`StallError`. A
        genuine DEVICE_LOST is a real GPU fault with forensics attached and MUST still
        fail -- otherwise this stops being tolerance and becomes "ignore failures".
        Anything that is not a stall re-raises immediately.

    Each attempt gets a FRESH output directory: a wedged run leaves partial artifacts,
    and tests assert on the tree, so reusing the directory would let attempt 1's debris
    satisfy (or corrupt) attempt 2's assertions.
    """
    from nett_skrl import NETT
    from nett_skrl.runtime.reap import DeviceLostRunError, is_pure_stall_failure

    base = Path(output_path)
    for attempt in range(1, attempts + 1):
        out = base if attempt == 1 else base.parent / f"{base.name}_retry{attempt - 1}"
        out.mkdir(parents=True, exist_ok=True)
        try:
            NETT([str(cfg_path)]).run(
                output_path=str(out), devices=devices, verbose=False
            )
            return out
        except DeviceLostRunError as exc:
            if not is_pure_stall_failure(exc) or attempt == attempts:
                raise
            print(
                f"\n[e2e] attempt {attempt}/{attempts} lost {len(exc.failures)} task(s) "
                f"to the Kit render-pump wedge (exit 77) -- retrying in a fresh output "
                f"dir. This is tolerated INFRA flake, not a result. Detail: {exc}",
                flush=True,
            )
    raise AssertionError("unreachable: loop either returns or raises")


# ---------------------------------------------------------------------------
# Golden numbers helpers (used by test_perf.py)
# ---------------------------------------------------------------------------


def load_golden() -> dict:
    if not GOLDEN_PATH.exists():
        return {}
    return json.loads(GOLDEN_PATH.read_text())


def save_golden(values: dict) -> None:
    """Write golden.json, stamped with WHEN and WHERE it was recorded.

    ★ The stamp is the point. The previous baseline sat unmodified for 112 commits --
    through the baked chamber, the current chick/rig, DLAA and the 24 Hz step clock --
    with nothing in the file to say how old it was, so healthy numbers kept reading as
    regressions. A baseline nobody re-records becomes an artifact; a dated one at least
    announces itself.
    """
    import datetime
    import platform

    gpu = "unknown"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    values = dict(values)
    values["recorded"] = {
        "date": datetime.date.today().isoformat(),
        "host": platform.node(),
        "gpu": gpu,
    }
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


@pytest.fixture
def golden(request) -> dict:
    """Read benchmarks/golden.json. Use ``--update-golden`` to rewrite it."""
    return load_golden()


@pytest.fixture
def update_golden(request) -> bool:
    return bool(request.config.getoption("--update-golden"))
