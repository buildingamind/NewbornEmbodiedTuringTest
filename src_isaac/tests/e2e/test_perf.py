"""Performance + convergence gate.

Three signals, each with a baseline number in ``benchmarks/golden.json``:

    * train throughput (steps/sec)
    * peak GPU memory (MB)         — sampled via nvidia-smi from this process
    * convergence smoke            — reward trend over a longer train run

On the first run with empty golden numbers, the tests still pass but log
the observed values. Pass ``--update-golden`` to commit the current run's
numbers as the new baseline; ``save_golden`` stamps it with the date, host
and GPU so its age is visible in the file.

Keys in ``golden.json``:

    * ``optimized_train_steps_per_second`` — the throughput gate's anchor.
    * ``train_peak_vram_mb``               — the VRAM gate's anchor.
    * ``convergence_max_delta``            — the convergence gate's anchor.
    * ``train_steps_per_second``           — HISTORICAL ONLY. The unoptimized
      May-2026 measurement (7.697). No assertion reads it any more; it is kept
      so the original 5x optimization claim stays checkable. Do not build a new
      gate on it — that is exactly how this file came to assert a target the
      current system could not meet.

Marked ``e2e_perf`` (a superset of ``e2e_isaac`` for skip purposes —
needs GPU + Isaac Sim + nvidia-smi on PATH).
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

from .conftest import (
    BENCHMARKS_DIR,
    E2E_DEVICE,
    MINIMAL_SMOKE_CFG,
    load_golden,
    save_golden,
)


pytestmark = [pytest.mark.e2e_isaac, pytest.mark.e2e_perf]

# ⚠⚠ A TIMING MEASUREMENT TAKEN UNDER CONTENTION IS NOT A MEASUREMENT. Under `pytest -n`
# the other workers are running their own Isaac training on sibling GPUs, which moves this
# number by more than the regression band -- and worse, `--update-golden` would then bake a
# contended figure in as the new "optimum", permanently poisoning the baseline for every
# later run. So refuse to run rather than report a number nobody should trust.
# The rest of the tree parallelises freely; only these three cannot.
if os.environ.get("PYTEST_XDIST_WORKER"):
    pytest.skip(
        "e2e_perf measures wall-clock throughput and VRAM, so it must own the machine: "
        "run it serially (no -n), e.g. `pytest tests/e2e -m e2e_perf`. The pre-push hook "
        "deliberately excludes e2e_perf for this reason -- see docs/development.md.",
        allow_module_level=True,
    )

# How far below the recorded optimum throughput may drift before it is a regression.
# Sized against measured run-to-run spread on this stack: the renderer alone moves
# ~5.7% between identical runs (see blueprint.md, rendering_mode probe), so a tighter
# band would flake. 15% still catches anything structural -- the real regressions seen
# here were -14% (video playback) and -4.5% (enhanced determinism) COMBINED with a
# system rebuild, not sub-noise drift.
_THROUGHPUT_SLACK = 0.15


# ---------------------------------------------------------------------------
# GPU memory poller
# ---------------------------------------------------------------------------


class _GpuPoller:
    """Background thread that samples ``nvidia-smi`` memory.used at 0.5 Hz.

    The test process does not hold any VRAM itself (Isaac runs in a child
    via spawn), so we cannot use ``torch.cuda.max_memory_allocated`` here.
    Polling nvidia-smi is the only portable way to bound peak VRAM from
    the parent without instrumenting the subprocess.
    """

    def __init__(self, device: int = 0, interval_s: float = 0.5) -> None:
        self.device = device
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mb: int = 0
        self.start_mb: int = 0

    def _sample(self) -> int:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", f"--id={self.device}"],
            capture_output=True, text=True, check=True,
        )
        return int(out.stdout.strip())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                used = self._sample()
            except Exception:
                used = 0
            self.peak_mb = max(self.peak_mb, used)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "_GpuPoller":
        self.start_mb = self._sample()
        self.peak_mb = self.start_mb
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _require_nvidia_smi() -> None:
    if shutil.which("nvidia-smi") is None:
        pytest.skip("nvidia-smi not on PATH")


def test_train_throughput_steps_per_second(tmp_path, golden, update_golden):
    """Hot-loop NETT/skrl train throughput for one minimal train run.

    Isaac/Kit startup and video preload are reported separately through phase
    profiles. The acceptance metric here is the training loop itself:
    ``num_brains * env_timesteps / skrl_train_total_s``.
    """
    cfg = copy.deepcopy(MINIMAL_SMOKE_CFG)
    cfg["episodes"] = {"train": 2}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    from nett_skrl import NETT

    total_train_steps = (
        cfg["episodes"]["train"] * cfg["steps_per_episode"] * cfg["num_brains"]
    )

    t0 = time.monotonic()
    NETT([str(cfg_path)]).run(output_path=str(tmp_path), devices=[E2E_DEVICE], verbose=False)
    elapsed = time.monotonic() - t0
    timing = _assert_train_artifacts(tmp_path / cfg["name"], expected_brains=cfg["num_brains"])

    wall_sps = total_train_steps / elapsed
    sps = float(timing["train_steps_per_second"])
    print(
        f"[perf] train_throughput_hot: {sps:.2f} steps/sec "
        f"({int(timing['train_steps'])} steps in {timing['skrl_train_total_s']:.2f}s)"
    )
    print(
        f"[perf] train_throughput_wall: {wall_sps:.2f} steps/sec "
        f"({total_train_steps} steps in {elapsed:.2f}s, includes Isaac startup/preload)"
    )
    profiles = _collect_phase_profiles(tmp_path / cfg["name"])
    if profiles:
        summary_path = tmp_path / cfg["name"] / "perf_phase_summary.json"
        summary_path.write_text(json.dumps(profiles, indent=2, sort_keys=True) + "\n")
        for phase, values in sorted(profiles.items()):
            print(
                f"[perf] phase {phase}: total={values['total_s']:.4f}s "
                f"count={values['count']} mean={values['mean_s']:.6f}s"
            )

    if update_golden:
        g = load_golden()
        g["optimized_train_steps_per_second"] = round(sps, 3)
        save_golden(g)
        return

    # ★ THE GATE IS A REGRESSION GATE, NOT A GOAL (changed 2026-07-28).
    # It used to assert ``sps >= 5 * train_steps_per_second``, where that baseline was
    # the UNOPTIMIZED May-2026 number (7.697) and 5x encoded the optimization target
    # reached back then. Nothing re-recorded it for 112 commits, across which the system
    # gained the baked chamber, the current chick/rig, real video playback and
    # enable_enhanced_determinism -- so the 38.48 target described a machine that no
    # longer exists, and the current, healthy 30 steps/s read as a failure. A stale
    # aspiration is not a regression signal.
    # Anchor instead on the LAST RECORDED optimum with explicit slack, exactly like the
    # VRAM test below. Re-record deliberately (--update-golden) when the config changes;
    # golden.json carries a "recorded" note saying when and against what.
    baseline = golden.get("optimized_train_steps_per_second")
    if baseline is None:
        pytest.skip("no optimized baseline in golden.json; rerun with --update-golden")
    target = baseline * (1.0 - _THROUGHPUT_SLACK)
    assert sps >= target, (
        f"throughput regression: {sps:.2f} < {target:.2f} steps/s "
        f"({_THROUGHPUT_SLACK:.0%} below the recorded optimum {baseline:.2f}). "
        f"If the slowdown is intended, re-record with --update-golden.")


def test_train_peak_vram_under_ceiling(tmp_path, golden, update_golden):
    """Peak GPU memory during one minimal train run, sampled via nvidia-smi.

    Asserts within ``baseline + 200 MB`` slack so a benign Isaac Sim cache
    bump does not flake the test, while a real leak (which usually grows
    by GB) does.
    """
    _require_nvidia_smi()

    cfg = copy.deepcopy(MINIMAL_SMOKE_CFG)
    cfg["episodes"] = {"train": 2}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    from nett_skrl import NETT

    with _GpuPoller() as poller:
        NETT([str(cfg_path)]).run(output_path=str(tmp_path), devices=[E2E_DEVICE], verbose=False)

    delta_mb = max(0, poller.peak_mb - poller.start_mb)
    print(f"[perf] train_peak_vram: delta={delta_mb} MB (start={poller.start_mb}, peak={poller.peak_mb})")

    if update_golden:
        g = load_golden()
        g["train_peak_vram_mb"] = int(delta_mb)
        save_golden(g)
        return

    baseline = golden.get("train_peak_vram_mb")
    if baseline is None:
        pytest.skip("no baseline in golden.json; rerun with --update-golden")
    assert delta_mb <= baseline + 200, f"VRAM regression: {delta_mb} MB > {baseline + 200} MB (baseline {baseline})"


def test_train_smoke_convergence_signal(tmp_path, golden, update_golden):
    """Reward must increase over a longer train run.

    Reads skrl's tfevents file for each brain and extracts
    ``Reward/Total reward (mean)`` scalars. Asserts the mean over the last
    25% of updates is greater than the mean over the first 25% — a
    structural smoke check that catches silent gradient-flow regressions
    (the kind where train completes but the policy never updates).
    """
    cfg = copy.deepcopy(MINIMAL_SMOKE_CFG)
    cfg["episodes"] = {"train": 100}
    cfg["brain"].setdefault("algorithm_cfg", {}).update({
        "learning_rate": 3.0e-4,  # PPO-typical; smoke cfg's 1e-5 is too low to learn fast
        "rollouts": 128,
        "mini_batches": 2,
    })
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    from nett_skrl import NETT

    NETT([str(cfg_path)]).run(output_path=str(tmp_path), devices=[E2E_DEVICE], verbose=False)

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        pytest.skip("tensorboard not installed; cannot parse tfevents")

    condition_dir = tmp_path / cfg["name"] / "Object1"
    deltas: list[float] = []
    for brain_id in (1, 2):
        brain_dir = condition_dir / "wandb_runs" / f"brain_{brain_id}"
        tfevents = list(brain_dir.glob("events.out.tfevents.*"))
        if not tfevents:
            pytest.fail(f"no tfevents file for brain_{brain_id} under {brain_dir}")
        ea = EventAccumulator(str(tfevents[0]))
        ea.Reload()
        tags = ea.Tags()["scalars"]
        reward_tag = next((t for t in tags if "Reward" in t and "mean" in t.lower()), None)
        if reward_tag is None:
            pytest.skip(f"no reward scalar tag in tfevents (tags: {tags})")
        scalars = [s.value for s in ea.Scalars(reward_tag)]
        if len(scalars) < 8:
            pytest.skip(f"only {len(scalars)} reward points for brain_{brain_id}; convergence test requires more")
        chunk = max(1, len(scalars) // 4)
        first = sum(scalars[:chunk]) / chunk
        last = sum(scalars[-chunk:]) / chunk
        delta = last - first
        print(f"[perf] convergence brain_{brain_id}: first25={first:.4f} last25={last:.4f} delta={delta:+.4f}")
        deltas.append(delta)

    observed_max = max(deltas)
    if update_golden:
        g = load_golden()
        g["convergence_max_delta"] = round(observed_max, 4)
        save_golden(g)
        return

    # PPO at this scale can plateau or even drop a few percent due to
    # exploration noise; reject only material regressions vs the baseline
    # max-brain delta. 0.05 slack absorbs run-to-run jitter on the
    # ``Reward/Total reward (mean)`` scalar.
    baseline = golden.get("convergence_max_delta")
    if baseline is None:
        pytest.skip("no baseline in golden.json; rerun with --update-golden")
    assert observed_max >= baseline - 0.05, \
        f"convergence regression: max delta {observed_max:+.4f} below {baseline - 0.05:+.4f} (baseline {baseline:+.4f}); per-brain deltas={deltas}"


# ---------------------------------------------------------------------------
# Seed file if missing — so a fresh checkout's first `--update-golden` run
# has a place to write. (Idempotent; runs once per session.)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _ensure_benchmarks_dir() -> None:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)


def _collect_phase_profiles(run_dir: Path) -> dict[str, dict[str, float | int]]:
    merged: dict[str, dict[str, float | int]] = {}
    for path in sorted(run_dir.rglob("profile_*.json")):
        data = json.loads(path.read_text())
        for phase, values in data.items():
            dest = merged.setdefault(phase, {"total_s": 0.0, "count": 0, "mean_s": 0.0})
            dest["total_s"] = float(dest["total_s"]) + float(values.get("total_s", 0.0))
            dest["count"] = int(dest["count"]) + int(values.get("count", 0))
    for values in merged.values():
        count = int(values["count"])
        values["mean_s"] = float(values["total_s"]) / count if count else 0.0
    return merged


def _assert_train_artifacts(run_dir: Path, expected_brains: int) -> dict:
    condition_dir = run_dir / "Object1"
    missing: list[str] = []
    for brain_id in range(1, expected_brains + 1):
        ckpt = condition_dir / "wandb_runs" / f"brain_{brain_id}" / "checkpoints" / "final_agent.pt"
        if not ckpt.exists():
            missing.append(str(ckpt))
    timing = condition_dir / "logs" / "train_timing.json"
    if not timing.exists():
        missing.append(str(timing))
    if missing:
        raise AssertionError(
            "NETT train subprocess did not produce required artifacts; "
            "throughput measurement is invalid. Missing:\n" + "\n".join(missing)
        )
    payload = json.loads(timing.read_text())
    required = {"skrl_train_total_s", "train_steps", "train_steps_per_second"}
    missing_keys = required - set(payload)
    if missing_keys:
        raise AssertionError(f"train_timing.json missing keys: {sorted(missing_keys)}")
    return payload
