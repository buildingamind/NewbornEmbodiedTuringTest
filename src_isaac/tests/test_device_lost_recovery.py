"""Regression tests for the DEVICE_LOST unclean-teardown hang.

THE BUG (observed 2026-07-14, TEST phase, 8-way wave): a transient renderer GPU
crash makes Kit print

    [carb.graphics-vulkan.plugin] VkResult: ERROR_DEVICE_LOST
    [gpu.foundation.plugin] A GPU crash occurred. Exiting the application...

and then *never exit*. The child hung ~13h holding ~5GB VRAM, its
multiprocessing.spawn workers reparented to PPID=1, and the parent's
``p.join()`` in ``_spawn_mode_subprocess`` blocked forever, stalling the wave.

THE FIX under test:
  * ``nett_skrl.runtime.crash_guard`` — in the child: confirm DEVICE_LOST, flush
    + fsync artifacts, ``os._exit(75)`` within a bounded window (SIGALRM
    backstop at -14 for a wedged GIL).
  * ``nett_skrl.runtime.reap`` — in the parent: bounded wait, then a task-owned
    reap (SIGTERM → grace → SIGKILL) of the child, its GPU compute-app PIDs, and
    its PPID=1 spawn orphans.

Test-process safety: every OS process created here is owned by the ``orchard``
fixture and killed on teardown even when the test fails. Nothing here signals a
process it did not create. No real GPU is touched: NVML is forced to miss and a
stub ``nvidia-smi`` supplies the compute-app list, so the box's live Isaac runs
are never enumerated, let alone signalled.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from fault_injection import (
    CARB_ERROR,
    DEVICE_LOST_BANNER,
    DEVICE_LOST_RECORDS,
    NEAR_MISS_RECORDS,
    PopenProc,
    StubProc,
    assert_alive,
    fake_gpu_compute_apps,
    fake_gpu_compute_apps_from_pidfile,
    fake_spawn_context,
    spawn_crashing_child_with_orphan,
    spawn_healthy_long_child,
    spawn_hung_child,
    spawn_orphan_worker,
    spawn_resource_tracker_lookalike,
    spawn_sentinel,
    stub_spawn_context,
)
from nett_skrl.nett import NETT
from nett_skrl.runtime import crash_guard, reap, task_runner
from nett_skrl.runtime.task_runner import _is_tolerated_isaac_teardown_exit


# The tolerated set in task_runner: a child exiting with one of these is waved
# through as "Isaac teardown noise, outputs on disk intact". If a forced
# DEVICE_LOST exit landed in here, a real GPU crash would be silently swallowed
# and the wave would score a corrupt run as a success. This set is the contract.
TOLERATED_TEARDOWN_EXITS = {-6, -9, -11, 134, 139}

# crash_guard's two escapes.
DEVICE_LOST_EXIT = 75          # NETT_DEVICE_LOST_EXIT_CODE default
SIGALRM_BACKSTOP_EXITS = (-14, 142)   # kernel backstop for a wedged GIL


# ---------------------------------------------------------------------------
# Exit-code contract
# ---------------------------------------------------------------------------


def test_tolerated_teardown_set_is_exactly_the_documented_signals():
    """Pin the tolerated set, so widening it has to be a deliberate act."""
    for code in TOLERATED_TEARDOWN_EXITS:
        assert _is_tolerated_isaac_teardown_exit(code), code
    for code in (1, 2, 3, 70, 137, -15, None):
        assert not _is_tolerated_isaac_teardown_exit(code), code


def test_device_lost_exit_code_is_not_swallowed_as_teardown_noise(monkeypatch):
    """crash_guard's forced exit code must NOT collide with the tolerated set.

    A collision is silent data loss: ``_spawn_mode_subprocess`` would log
    "outputs on disk should still be intact" and let the wave continue as if
    nothing had broken.
    """
    monkeypatch.delenv("NETT_DEVICE_LOST_EXIT_CODE", raising=False)
    code = crash_guard._env_int("NETT_DEVICE_LOST_EXIT_CODE", 75)

    assert code == DEVICE_LOST_EXIT
    assert code != 0, "a DEVICE_LOST crash must never exit 0"
    assert not _is_tolerated_isaac_teardown_exit(code), (
        f"crash_guard exit code {code} is in the tolerated teardown set "
        f"{sorted(TOLERATED_TEARDOWN_EXITS)}; the GPU crash would be swallowed"
    )


@pytest.mark.parametrize("code", SIGALRM_BACKSTOP_EXITS)
def test_sigalrm_backstop_exit_code_is_not_swallowed_either(code):
    """The wedged-GIL backstop kills via SIGALRM's default disposition.

    multiprocessing reports it as -14; a shell reports 142. Neither may be
    mistaken for tolerated teardown noise.
    """
    assert not _is_tolerated_isaac_teardown_exit(code)


# ---------------------------------------------------------------------------
# Detection — confirmed crash vs. false positive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(DEVICE_LOST_RECORDS))
def test_confirmed_device_lost_records_are_detected(name):
    source, level, message = DEVICE_LOST_RECORDS[name]
    assert crash_guard.is_device_lost_message(source, level, message) is True


@pytest.mark.parametrize("name", sorted(NEAR_MISS_RECORDS))
def test_near_miss_records_are_not_reported_as_device_lost(name):
    """False positives force-kill healthy multi-hour runs — the worse failure."""
    source, level, message = NEAR_MISS_RECORDS[name]
    assert crash_guard.is_device_lost_message(source, level, message) is False, name


def test_detection_requires_error_level_not_merely_the_words():
    source, _, message = DEVICE_LOST_RECORDS["vulkan_device_lost"]
    assert crash_guard.is_device_lost_message(source, CARB_ERROR, message) is True
    assert crash_guard.is_device_lost_message(source, CARB_ERROR - 1, message) is False


def test_detection_is_side_effect_free_and_leaves_the_guard_disarmed():
    """The predicate must be safe to call from tests and from the log hook."""
    for source, level, message in NEAR_MISS_RECORDS.values():
        crash_guard.is_device_lost_message(source, level, message)
    assert crash_guard._armed is False
    assert crash_guard._triggered.is_set() is False


def test_guard_kill_switch_keeps_it_disarmed(monkeypatch):
    monkeypatch.setenv("NETT_DEVICE_LOST_GUARD", "0")
    assert crash_guard.arm() is False
    assert crash_guard._armed is False


def test_arm_without_carb_is_a_no_op_so_the_fast_suite_can_import_it(monkeypatch):
    """No carb → arm() declines quietly instead of raising.

    carb is forced to be unimportable rather than assumed absent: ``isaacsim``
    IS installed in this venv, so whether ``import carb`` succeeds depends on
    which OTHER test imported it first. Asserting the real contract ("carb
    unavailable ⇒ decline") keeps this independent of collection order.
    """
    monkeypatch.delenv("NETT_DEVICE_LOST_GUARD", raising=False)
    monkeypatch.setitem(sys.modules, "carb.logging", None)  # -> ImportError

    try:
        assert crash_guard.arm() is False
        assert crash_guard._armed is False
    finally:
        crash_guard.disarm()


# ---------------------------------------------------------------------------
# The exit-code contract: how the child signals DEVICE_LOST to the parent
# ---------------------------------------------------------------------------


def test_exit_code_signature_covers_the_watchdog_and_the_backstop():
    assert reap.is_device_lost_exit(75) is True            # watchdog os._exit
    assert reap.is_device_lost_exit(-14) is True           # SIGALRM backstop
    assert reap.is_device_lost_exit(142) is True           # 128+signo form
    assert reap.is_device_lost_exit(0) is False
    assert reap.is_device_lost_exit(None) is False
    assert reap.is_device_lost_exit(1) is False


def test_exit_code_signature_never_overlaps_the_tolerated_teardown_set():
    """The two sets must stay disjoint or a GPU crash reads as teardown noise."""
    for code in TOLERATED_TEARDOWN_EXITS:
        assert reap.is_device_lost_exit(code) is False, code


def test_device_lost_exit_code_is_read_at_call_time_so_an_override_is_honoured(
    monkeypatch,
):
    """Parent and child agree via the env var, which the child inherits."""
    monkeypatch.setenv("NETT_DEVICE_LOST_EXIT_CODE", "81")
    assert reap.device_lost_exit_code() == 81
    assert reap.is_device_lost_exit(81) is True
    assert reap.is_device_lost_exit(75) is False
    # crash_guard must resolve the SAME override, or the sides disagree.
    assert crash_guard._env_int("NETT_DEVICE_LOST_EXIT_CODE", 75) == 81


def test_device_lost_exit_code_falls_back_when_the_override_is_garbage(monkeypatch):
    monkeypatch.setenv("NETT_DEVICE_LOST_EXIT_CODE", "not-a-number")
    assert reap.device_lost_exit_code() == 75
    assert crash_guard._env_int("NETT_DEVICE_LOST_EXIT_CODE", 75) == 75


# ---------------------------------------------------------------------------
# Artifact durability -> analyze() still yields a valid summary.json
# ---------------------------------------------------------------------------


def test_analyze_yields_valid_summary_from_a_crashed_runs_artifacts(crashed_run_dir):
    """The durability criterion, stated end-to-end.

    The child never writes summary.json — ``analysis.analyze()`` derives it
    post-hoc. So the requirement is: the child's fsynced artifacts survive the
    forced exit, and analyze() still turns them into a valid summary.
    """
    from nett_skrl.analysis import analyze

    out = analyze(crashed_run_dir, chick_experiment=None)
    summary = json.loads((out / "summary.json").read_text())

    assert set(summary) >= {"train", "test"}
    # Agent parked in the correct outer third for every logged step. NOTE:
    # ``correct_pct`` is a FRACTION in [0, 1] despite the name — matches
    # tests/test_analysis.py::test_analyze_produces_summary_with_test_metrics.
    assert summary["test"]["Object1"]["rest"]["correct_pct_mean"] == 1.0
    assert summary["test"]["Object1"]["rest"]["n_brains"] == 1


def test_child_artifacts_are_fsynced_before_a_hard_exit(orchard, tmp_path):
    """A child that fsyncs, then dies without unwinding, still leaves its data.

    Guards the mechanism crash_guard depends on: ``os._exit`` skips atexit and
    buffer flushing, so anything not fsynced FIRST is lost. SIGKILL here is
    strictly harsher than os._exit, so surviving it is the stronger claim.
    """
    run = tmp_path / "run"
    pid, _ = spawn_hung_child(
        orchard, tmp_path, banner=DEVICE_LOST_BANNER, artifacts_root=run
    )
    orchard.kill(pid)
    assert orchard.wait_gone(pid)

    csv_text = (run / "logs" / "eval_metrics.csv").read_text()
    assert csv_text.splitlines()[0].startswith("eval_step,")
    assert "50000,Object1,1,0.5,400" in csv_text


# ---------------------------------------------------------------------------
# Ownership — the safety half. These must hold for ANY reap implementation.
# ---------------------------------------------------------------------------


def _reaper(orchard, tmp_path, *, device=None):
    return reap.TaskReaper(task_key="run/Object1/test", device=device)


def test_reap_kills_the_owned_child_and_spares_an_unrelated_bystander(
    orchard, tmp_path
):
    """A bystander Isaac run must survive a reap aimed at someone else.

    The sentinel stands in for the other seven runs on the box: same user, same
    python, same-looking command — but no ownership token, so hands off.
    """
    reaper = _reaper(orchard, tmp_path)
    sentinel = spawn_sentinel(orchard, tmp_path)
    with reaper.launch_scope():
        victim, _ = spawn_hung_child(orchard, tmp_path, banner=DEVICE_LOST_BANNER)
    reaper.adopt(victim)

    report = reaper.reap("device-lost")

    assert orchard.wait_gone(victim), f"targeted child not reaped: {report}"
    assert victim in (report.terminated + report.killed)
    assert_alive(orchard, sentinel, "reap killed an unrelated bystander process")


def test_reap_declines_a_pid_whose_start_time_no_longer_matches(orchard, tmp_path):
    """PID reuse: recorded PID N died; N now belongs to somebody else.

    ``ProcessIdentity`` keys on ``(pid, start_ticks)``, so a stale record must
    not authorise a kill. The process now holding the PID is a sentinel, making
    a PID-only reaper directly observable as a dead sentinel.
    """
    reaper = _reaper(orchard, tmp_path)
    sentinel = spawn_sentinel(orchard, tmp_path)
    real = reap.ProcessIdentity.of(sentinel)
    assert real is not None

    # The recorded process started earlier and exited; the kernel recycled its
    # PID onto the sentinel.
    stale = reap.ProcessIdentity(pid=sentinel, start_ticks=real.start_ticks - 5000)
    assert stale.alive() is False, "stale identity must not read as alive"
    reaper.root = stale
    reaper._lineage.add(stale)

    report = reaper.reap("device-lost")

    assert_alive(
        orchard, sentinel,
        f"reap killed a process whose start_ticks did not match the record "
        f"(PID-reuse protection failed): {report}",
    )
    assert sentinel not in (report.terminated + report.killed)


def test_untokened_lookalike_child_is_not_owned(orchard, tmp_path):
    """Ownership is evidence-based: no token, not in lineage, not the root."""
    reaper = _reaper(orchard, tmp_path)
    stranger = spawn_sentinel(orchard, tmp_path)
    assert reaper.owns(stranger) is None


def test_reap_on_an_already_exited_task_is_a_bounded_no_op(orchard, tmp_path):
    reaper = _reaper(orchard, tmp_path)
    with reaper.launch_scope():
        pid, _ = spawn_hung_child(orchard, tmp_path)
    reaper.adopt(pid)
    orchard.kill(pid)
    assert orchard.wait_gone(pid)

    started = time.monotonic()
    first = reaper.reap("device-lost")
    second = reaper.reap("device-lost")   # idempotent

    assert not first.killed and not second.killed
    assert time.monotonic() - started < reap.TERM_GRACE * 2 + 5


def test_ppid1_spawn_orphans_are_reaped(orchard, tmp_path):
    """The hang leaves PPID=1 multiprocessing.spawn workers holding VRAM.

    Their parent link is gone, so the tree walk cannot find them; the inherited
    env token is what proves ownership.
    """
    reaper = _reaper(orchard, tmp_path)
    with reaper.launch_scope():
        orphan, _ = spawn_orphan_worker(orchard, tmp_path)

    assert reaper.owns(orphan) == "env-token"
    reaper.reap("device-lost")
    assert orchard.wait_gone(orphan), "PPID=1 spawn orphan survived the reap"


def test_shared_resource_tracker_is_never_reaped_even_though_it_holds_the_token(
    orchard, tmp_path
):
    """The multiprocessing resource_tracker is a per-worker singleton.

    It inherits the token if it is born inside ``launch_scope``, but killing it
    cannot unwedge this task — it only breaks every SUBSEQUENT task in the pool
    worker ("resources might leak").
    """
    reaper = _reaper(orchard, tmp_path)
    with reaper.launch_scope():
        tracker, _ = spawn_resource_tracker_lookalike(orchard, tmp_path)

    assert reaper.owns(tracker) is None, "shared infrastructure must never be owned"
    reaper.reap("device-lost")
    assert_alive(orchard, tracker, "reap killed the shared resource_tracker")


# ---------------------------------------------------------------------------
# GPU cleanup — faked end to end; the real devices are never enumerated
# ---------------------------------------------------------------------------


def test_owned_gpu_compute_app_is_reaped(orchard, tmp_path, monkeypatch):
    reaper = _reaper(orchard, tmp_path, device=0)
    with reaper.launch_scope():
        holder, _ = spawn_hung_child(orchard, tmp_path, banner=DEVICE_LOST_BANNER)
    reaper.adopt(holder)
    fake_gpu_compute_apps(monkeypatch, tmp_path, pids=[holder])

    report = reaper.reap("device-lost")

    assert holder in report.gpu_pids
    assert orchard.wait_gone(holder), "owned GPU compute-app PID was not reaped"


def test_unowned_gpu_compute_app_is_counted_but_left_alone(
    orchard, tmp_path, monkeypatch
):
    """THE dangerous case: another live 13h Isaac run shows up in nvidia-smi.

    Appearing on the GPU is not ownership evidence. Killing it would take down
    somebody else's job — the exact "aggressive pkill -9" failure the blueprint
    warns against.
    """
    reaper = _reaper(orchard, tmp_path, device=0)
    with reaper.launch_scope():
        mine, _ = spawn_hung_child(orchard, tmp_path)
    reaper.adopt(mine)
    stranger = spawn_sentinel(orchard, tmp_path)
    fake_gpu_compute_apps(monkeypatch, tmp_path, pids=[mine, stranger])

    report = reaper.reap("device-lost")

    assert report.skipped_unowned >= 1, report
    assert_alive(
        orchard, stranger,
        "reap killed an unowned GPU compute app — another user's Isaac run",
    )


def test_gpu_query_falls_back_to_nvidia_smi_and_parses_it(
    orchard, tmp_path, monkeypatch
):
    reaper = _reaper(orchard, tmp_path, device=0)
    fake_gpu_compute_apps(monkeypatch, tmp_path, pids=[4242, 4243])
    assert reaper.gpu_compute_pids() == [4242, 4243]


def test_gpu_query_survives_a_missing_nvidia_smi(orchard, tmp_path, monkeypatch):
    reaper = _reaper(orchard, tmp_path, device=0)
    monkeypatch.setattr(reap.TaskReaper, "_gpu_pids_nvml", lambda self: None)
    monkeypatch.setattr(reap.shutil, "which", lambda name: None)
    assert reaper.gpu_compute_pids() == []


# ---------------------------------------------------------------------------
# join_with_reap — the bounded wait
# ---------------------------------------------------------------------------


def test_healthy_long_running_child_is_not_killed_by_the_new_bound(orchard, tmp_path):
    """Training legitimately runs for HOURS.

    A naive join timeout would murder every healthy run. The bound may only fire
    on confirmed crash evidence — never on mere duration. Asserted by running
    the real join loop against a child that never exits and never crashes.
    """
    reaper = _reaper(orchard, tmp_path)
    with reaper.launch_scope():
        pid, _ = spawn_healthy_long_child(orchard, tmp_path)
    reaper.adopt(pid)
    proc = PopenProc(orchard.popen(pid))

    outcome: list[str] = []
    joiner = threading.Thread(
        target=lambda: outcome.append(
            reap.join_with_reap(proc, reaper, crash_evidence=lambda: False)
        ),
        daemon=True,
    )
    joiner.start()
    # Several poll cycles: long enough for any duration-based bound to fire.
    joiner.join(timeout=reap.POLL * 8)

    try:
        assert outcome == [], f"join_with_reap returned {outcome} on a healthy child"
        assert_alive(
            orchard, pid,
            "a healthy long-running child was killed by the new timeout bound",
        )
    finally:
        orchard.kill(pid)          # release the joiner thread
        joiner.join(timeout=10)


def test_absolute_timeout_is_disabled_by_default(monkeypatch):
    """Documents the deliberate default: no duration cap.

    ``NETT_REAP_TIMEOUT=0`` is what makes the hours-long training case safe. The
    cost is the flip side, pinned by the next test.
    """
    assert reap.ABSOLUTE_TIMEOUT == 0


def test_generic_hang_without_device_lost_is_unbounded_by_default(orchard, tmp_path):
    """A hang with NO DEVICE_LOST evidence is NOT reaped under the defaults.

    This is the honest statement of current behaviour, not an endorsement: with
    ``crash_evidence`` false and ``ABSOLUTE_TIMEOUT=0``, ``join_with_reap``
    polls forever — identical to today's ``p.join()``. Only an explicit
    ``NETT_REAP_TIMEOUT`` bounds it (next test). If the operator expects a
    generic wedge to self-clear, that expectation is wrong and this test says so.
    """
    reaper = _reaper(orchard, tmp_path)
    with reaper.launch_scope():
        pid, _ = spawn_hung_child(orchard, tmp_path, banner="")   # no banner
    reaper.adopt(pid)
    proc = PopenProc(orchard.popen(pid))

    outcome: list[str] = []
    joiner = threading.Thread(
        target=lambda: outcome.append(
            reap.join_with_reap(proc, reaper, crash_evidence=lambda: False)
        ),
        daemon=True,
    )
    joiner.start()
    joiner.join(timeout=reap.POLL * 6)

    try:
        assert outcome == [], "generic hang was reaped — expected unbounded default"
        assert_alive(orchard, pid, "generic hang was killed under default settings")
    finally:
        orchard.kill(pid)
        joiner.join(timeout=10)


def test_explicit_absolute_timeout_reaps_a_generic_hang(orchard, tmp_path):
    """With NETT_REAP_TIMEOUT set, a no-evidence wedge is bounded."""
    reaper = _reaper(orchard, tmp_path)
    with reaper.launch_scope():
        pid, _ = spawn_hung_child(orchard, tmp_path, banner="")
    reaper.adopt(pid)
    proc = PopenProc(orchard.popen(pid))

    outcome = reap.join_with_reap(
        proc, reaper, crash_evidence=lambda: False, absolute_timeout=1.0
    )

    assert outcome == "reaped-timeout"
    assert orchard.wait_gone(pid)


def test_crash_evidence_reaps_after_the_grace_window(orchard, tmp_path):
    """Confirmed DEVICE_LOST + a child that will not self-exit → reaped."""
    reaper = _reaper(orchard, tmp_path)
    with reaper.launch_scope():
        pid, _ = spawn_hung_child(orchard, tmp_path, banner=DEVICE_LOST_BANNER)
    reaper.adopt(pid)
    proc = PopenProc(orchard.popen(pid))

    started = time.monotonic()
    outcome = reap.join_with_reap(
        proc, reaper, crash_evidence=lambda: True, crash_grace=1.0
    )
    elapsed = time.monotonic() - started

    assert outcome == "reaped-crash"
    assert orchard.wait_gone(pid)
    # The grace window is honoured (the child gets its chance to self-exit)…
    assert elapsed >= 1.0
    # …but the whole thing is bounded — not 13 hours.
    assert elapsed < 1.0 + reap.TERM_GRACE * 2 + 10


def test_child_that_self_exits_on_crash_is_not_reaped(orchard, tmp_path):
    """The common path: crash_guard's own os._exit(75) wins the race.

    Evidence is present, but the child exits inside the grace window, so
    join_with_reap must return "exited" and leave the reaper idle.
    """
    reaper = _reaper(orchard, tmp_path)
    with reaper.launch_scope():
        pid, _ = spawn_hung_child(orchard, tmp_path, banner=DEVICE_LOST_BANNER)
    reaper.adopt(pid)
    proc = PopenProc(orchard.popen(pid))

    threading.Timer(0.5, lambda: orchard.kill(pid)).start()
    outcome = reap.join_with_reap(
        proc, reaper, crash_evidence=lambda: True, crash_grace=30.0
    )

    assert outcome == "exited"


def test_successful_shutdown_reports_exited_and_preserves_the_exit_code(
    orchard, tmp_path
):
    """A clean run must be indistinguishable from pre-patch behaviour."""
    reaper = _reaper(orchard, tmp_path)
    with reaper.launch_scope():
        pid, _ = spawn_hung_child(orchard, tmp_path)
    reaper.adopt(pid)
    proc = PopenProc(orchard.popen(pid))
    os.kill(pid, 15)   # our own child; a plain, graceful termination

    assert reap.join_with_reap(proc, reaper, crash_evidence=lambda: False) == "exited"
    assert proc.exitcode is not None


def test_reap_disable_switch_is_honoured(orchard, tmp_path, monkeypatch):
    """The escape hatch: NETT_REAP_DISABLE=1 must never kill anything."""
    monkeypatch.setattr(reap, "DISABLED", True)
    reaper = _reaper(orchard, tmp_path)
    with reaper.launch_scope():
        pid, _ = spawn_hung_child(orchard, tmp_path, banner=DEVICE_LOST_BANNER)
    reaper.adopt(pid)

    report = reaper.reap("device-lost")

    assert not report.killed and not report.terminated
    assert_alive(orchard, pid, "reap ran despite NETT_REAP_DISABLE=1")


# ---------------------------------------------------------------------------
# _spawn_mode_subprocess — the load-bearing integrated path
# ---------------------------------------------------------------------------


def _fake_task(tmp_path, *, device=0):
    """The bits ``_spawn_mode_subprocess`` touches, and nothing else."""
    return SimpleNamespace(
        config=SimpleNamespace(
            name="run",
            condition="Object1",
            device=device,
            path=tmp_path,
            logger=logging.getLogger("test.device_lost"),
        )
    )


def test_exit_75_reaps_leaked_workers_and_gpu_procs_then_raises_device_lost(
    orchard, tmp_path, monkeypatch
):
    """THE incident, end to end, through the real wiring.

    ``crash_guard`` calls ``os._exit(75)``: no atexit hooks run, so the child's
    spawn workers are never terminated. They reparent to PPID=1 and keep holding
    ~5GB of VRAM, while the parent's join returns CLEANLY — which is exactly why
    the exit-code-triggered reap is the only thing that frees the GPU.

    Everything but the child is production code here: the reaper, the
    launch_scope token, adopt, join_with_reap, and the exit-code branch all run
    for real. NVML is forced to miss and nvidia-smi is stubbed, so no real device
    is queried.
    """
    orphan_pidfile = tmp_path / "orphan.pid"
    fake_spawn_context(
        monkeypatch,
        lambda: spawn_crashing_child_with_orphan(
            orchard, tmp_path, orphan_pidfile, exit_code=75
        ),
    )
    fake_gpu_compute_apps_from_pidfile(monkeypatch, tmp_path, orphan_pidfile)

    with pytest.raises(reap.DeviceLostError) as excinfo:
        task_runner._spawn_mode_subprocess(_fake_task(tmp_path), "test")

    assert "75" in str(excinfo.value)
    leaked = int(orphan_pidfile.read_text().strip())
    assert orchard.wait_gone(leaked), (
        "the PPID=1 spawn worker leaked by os._exit(75) survived — it is still "
        "holding VRAM and will OOM the next wave item on this device"
    )


def test_exit_75_without_a_reap_would_leave_the_worker_alive(
    orchard, tmp_path, monkeypatch
):
    """The counter-case that proves the previous test has teeth.

    With the reap disabled, the identical child leaks the identical worker and
    it SURVIVES. If this ever starts passing-by-accident (worker dies on its
    own), the test above proves nothing.
    """
    monkeypatch.setattr(reap, "DISABLED", True)
    orphan_pidfile = tmp_path / "orphan.pid"
    fake_spawn_context(
        monkeypatch,
        lambda: spawn_crashing_child_with_orphan(
            orchard, tmp_path, orphan_pidfile, exit_code=75
        ),
    )

    with pytest.raises(reap.DeviceLostError):
        task_runner._spawn_mode_subprocess(_fake_task(tmp_path), "test")

    leaked = int(orphan_pidfile.read_text().strip())
    assert_alive(
        orchard, leaked,
        "the leaked worker died without a reap — the reap test proves nothing",
    )


def test_clean_exit_zero_raises_nothing(orchard, tmp_path, monkeypatch):
    """A healthy mode must be indistinguishable from pre-patch behaviour."""
    orphan_pidfile = tmp_path / "orphan.pid"
    fake_spawn_context(
        monkeypatch,
        lambda: spawn_crashing_child_with_orphan(
            orchard, tmp_path, orphan_pidfile, exit_code=0
        ),
    )
    task_runner._spawn_mode_subprocess(_fake_task(tmp_path), "test")


def test_ordinary_nonzero_exit_still_raises_plain_runtime_error(
    orchard, tmp_path, monkeypatch
):
    """An ordinary failure must NOT be typed as DEVICE_LOST, or the wave would
    wrongly continue past a real bug."""
    orphan_pidfile = tmp_path / "orphan.pid"
    fake_spawn_context(
        monkeypatch,
        lambda: spawn_crashing_child_with_orphan(
            orchard, tmp_path, orphan_pidfile, exit_code=1
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        task_runner._spawn_mode_subprocess(_fake_task(tmp_path), "test")

    assert not isinstance(excinfo.value, reap.ReapedTaskError)


# ---------------------------------------------------------------------------
# ORDERING INVARIANT — a reaped child exits -9, which IS tolerated
# ---------------------------------------------------------------------------


def test_a_reaped_child_exit_code_is_inside_the_tolerated_set():
    """The premise of the trap, stated outright.

    SIGKILL → exitcode -9 → ``_is_tolerated_isaac_teardown_exit(-9)`` is True.
    So the reap outcome MUST be consumed before any exitcode logic runs.
    """
    assert _is_tolerated_isaac_teardown_exit(-9) is True


@pytest.mark.parametrize(
    "outcome, expected",
    [("reaped-crash", reap.DeviceLostError), ("reaped-timeout", reap.TaskTimeoutError)],
)
def test_reap_outcome_is_checked_before_the_exitcode(
    tmp_path, monkeypatch, outcome, expected
):
    """Fails if anyone reorders the checks in ``_spawn_mode_subprocess``.

    A reaped child was SIGKILLed, so ``p.exitcode == -9`` — a member of the
    tolerated teardown set. If the exitcode branch is moved ahead of the
    outcome branch, this reaped task is logged "outputs on disk should still be
    intact", NO exception is raised, and the wave scores a killed run as a
    success. The stub proc reports exactly that -9 to make the trap live.
    """
    proc = StubProc(exitcode=-9)
    monkeypatch.setattr(task_runner, "join_with_reap", lambda *a, **k: outcome)
    monkeypatch.setattr(task_runner, "TaskReaper", lambda **kwargs: _NoopReaper())
    stub_spawn_context(monkeypatch, proc)

    with pytest.raises(expected):
        task_runner._spawn_mode_subprocess(_fake_task(tmp_path), "test")


class _NoopReaper:
    """A TaskReaper that records instead of signalling. No PIDs involved."""

    def __init__(self) -> None:
        self.reaped: list[str] = []

    @contextmanager
    def launch_scope(self):
        yield

    def adopt(self, pid):
        return None

    def reap(self, reason):
        self.reaped.append(reason)
        return reap.ReapReport(reason=reason)


def test_device_lost_exit_triggers_exactly_one_reap_with_the_right_reason(
    tmp_path, monkeypatch
):
    """The exit-75 branch must actually call reap() — that call IS the fix."""
    reaper = _NoopReaper()
    monkeypatch.setattr(task_runner, "TaskReaper", lambda **kwargs: reaper)
    monkeypatch.setattr(task_runner, "join_with_reap", lambda *a, **k: "exited")
    stub_spawn_context(monkeypatch, StubProc(exitcode=75))

    with pytest.raises(reap.DeviceLostError):
        task_runner._spawn_mode_subprocess(_fake_task(tmp_path), "test")

    assert reaper.reaped == ["device-lost-exit"]


def test_tolerated_teardown_exit_does_not_trigger_a_reap(tmp_path, monkeypatch):
    """A benign teardown SIGSEGV must not drag the GPU-reap machinery in."""
    reaper = _NoopReaper()
    monkeypatch.setattr(task_runner, "TaskReaper", lambda **kwargs: reaper)
    monkeypatch.setattr(task_runner, "join_with_reap", lambda *a, **k: "exited")
    stub_spawn_context(monkeypatch, StubProc(exitcode=-11))

    task_runner._spawn_mode_subprocess(_fake_task(tmp_path), "test")

    assert reaper.reaped == []


# ---------------------------------------------------------------------------
# Wave policy — _task_waiter
# ---------------------------------------------------------------------------


def test_device_lost_error_is_a_distinct_type_for_the_wave_continue_path():
    """The wave may continue for THIS family only; everything else fail-fast."""
    assert issubclass(reap.DeviceLostError, reap.ReapedTaskError)
    assert issubclass(reap.TaskTimeoutError, reap.ReapedTaskError)
    assert issubclass(reap.ReapedTaskError, RuntimeError)
    assert reap.DeviceLostError is not RuntimeError


def _nett_with_tasks(futures_and_cfgs):
    """A NETT wired up just enough to run ``_task_waiter``.

    Mirrors the object.__new__ + attribute-injection style already used by
    tests/test_orchestrator.py.
    """
    nett = object.__new__(NETT)
    nett.logger = logging.getLogger("test.wave")
    nett.failed_tasks = []
    nett.waitlist = []
    nett.devices = [0]
    nett.free_device_memory = {0: 1}
    nett.memory_manager = SimpleNamespace(get_free_memory=lambda d: 1)
    nett.task_sheet = {fut: cfg for fut, cfg in futures_and_cfgs}
    return nett


def _cfg(condition):
    return SimpleNamespace(name="run", condition=condition, device=None)


def _future(result=None, exc=None):
    fut = Future()
    if exc is not None:
        fut.set_exception(exc)
    else:
        fut.set_result(result)
    return fut


def test_wave_continues_after_a_device_lost_task_but_still_ends_nonzero():
    """USER DECISION, encoded: continue ONLY on this path, and still fail loudly.

    A transient renderer crash in one task must not cost the other N-1 tasks
    their hours of work — but the run must not report success either.
    """
    crashed = reap.DeviceLostError("reaped after DEVICE_LOST")
    nett = _nett_with_tasks([
        (_future(exc=crashed), _cfg("Object1")),
        (_future(result="ok"), _cfg("Object2")),
    ])

    with pytest.raises(reap.DeviceLostRunError) as excinfo:
        NETT._task_waiter(nett)

    # Every task was drained — the healthy one was NOT abandoned.
    assert nett.task_sheet == {}
    # …and the failure is reported, not swallowed.
    assert [key for key, _ in excinfo.value.failures] == ["run/Object1"]
    assert excinfo.value.failures[0][1] is crashed


def test_wave_continue_path_covers_reap_timeout_too():
    timed_out = reap.TaskTimeoutError("reaped on NETT_REAP_TIMEOUT")
    nett = _nett_with_tasks([
        (_future(exc=timed_out), _cfg("Object1")),
        (_future(result="ok"), _cfg("Object2")),
    ])

    with pytest.raises(reap.DeviceLostRunError):
        NETT._task_waiter(nett)


def test_ordinary_task_exception_still_fails_fast():
    """Everything outside the DEVICE_LOST/timeout family keeps today's re-raise."""
    boom = ValueError("an ordinary bug")
    nett = _nett_with_tasks([
        (_future(exc=boom), _cfg("Object1")),
        (_future(result="ok"), _cfg("Object2")),
    ])

    with pytest.raises(ValueError) as excinfo:
        NETT._task_waiter(nett)

    assert excinfo.value is boom
    assert nett.failed_tasks == [], "an ordinary bug must not join the continue path"


def test_a_fully_healthy_wave_raises_nothing():
    nett = _nett_with_tasks([
        (_future(result="ok"), _cfg("Object1")),
        (_future(result="ok"), _cfg("Object2")),
    ])

    NETT._task_waiter(nett)

    assert nett.task_sheet == {}
    assert nett.failed_tasks == []


def test_every_device_lost_task_in_a_wave_is_reported():
    """An 8-way wave losing several tasks must report all of them."""
    nett = _nett_with_tasks([
        (_future(exc=reap.DeviceLostError(f"crash {i}")), _cfg(f"Object{i}"))
        for i in range(3)
    ])

    with pytest.raises(reap.DeviceLostRunError) as excinfo:
        NETT._task_waiter(nett)

    assert len(excinfo.value.failures) == 3
    assert "3 task(s)" in str(excinfo.value)
