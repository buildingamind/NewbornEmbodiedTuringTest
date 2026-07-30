"""Task-owned process reaping for the unclean-teardown (DEVICE_LOST) hang.

A transient renderer GPU crash (``VkResult ERROR_DEVICE_LOST``) makes an Isaac
mode-subprocess hang instead of exiting: 0% CPU, ~5GB VRAM still held, its
``multiprocessing.spawn`` workers reparented to PPID=1, and the parent's
``p.join()`` in :mod:`nett_skrl.runtime.task_runner` blocked forever (~13h
observed). This module bounds that teardown window and reaps the wreckage.

DESIGN — the primary job is ORPHAN REAPING, not timing out the join
-------------------------------------------------------------------
Training legitimately runs for HOURS, so a blanket ``p.join(timeout)`` would
kill healthy runs. It is also not what actually fixes this bug.

``crash_guard`` (child side) already guarantees the join RETURNS: on a confirmed
DEVICE_LOST it hard-exits with 75 in ~0.1s, and if the GIL is too wedged for its
watchdog to run, an armed SIGALRM kernel backstop terminates it at
``NETT_DEVICE_LOST_EXIT_BUDGET_S`` (60s) -> -14. The kernel cannot be blocked by
a wedged interpreter, so the parent's ``p.join()`` unblocks either way.

What crash_guard CANNOT do is clean up after itself: ``os._exit`` runs no atexit
hooks, so the dying child's ``multiprocessing.spawn`` workers are NOT terminated
and its CUDA/RTX context is NOT released. Those workers reparent to PPID=1 and
keep holding ~5GB of VRAM. The parent then sees a returned join and a "finished"
task while the GPU stays wedged -- so the NEXT wave item OOMs on that device.
THAT is the residual bug this module fixes, and it is why the reap runs on the
device-lost EXIT path (75 / -14 / 142), not only on a timeout.

The timing bounds are therefore secondary safety nets, applying to the
TEARDOWN/CRASH window and never to total runtime:

* ``NETT_REAP_CRASH_GRACE`` starts ONLY once positive crash evidence appears --
  the net for a child so wedged it cannot run its own handler AND whose kernel
  backstop never armed (e.g. crash_guard disabled). It is INERT BY DEFAULT:
  ``join_with_reap`` is wired with ``crash_evidence=None`` because crash_guard
  signals DEVICE_LOST by EXIT CODE (75 / -14), not by a marker file, so the
  parent reaps off the exit code (see :func:`is_device_lost_exit`) rather than
  polling for evidence mid-run.
* ``NETT_REAP_TIMEOUT`` (absolute cap) DEFAULTS TO 0 = DISABLED. See BUDGETS.

OWNERSHIP — the hard safety requirement
---------------------------------------
This machine runs CONCURRENT NETT tasks and OTHER USERS. Killing an unrelated
process is the worst possible outcome, so a process is NEVER reaped merely for
appearing in ``nvidia-smi`` or for having PPID=1. Every kill requires positive
evidence the process belongs to THIS task:

1. ``NETT_REAP_TOKEN`` — a per-spawn uuid4 stamped into the child's environment
   before ``start()``. ``spawn`` children inherit it, as do all THEIR children,
   and it stays readable in ``/proc/<pid>/environ`` even after reparenting to
   PPID=1. This is the primary, reparent-proof evidence.
2. Root identity match — the adopted (pid, start_ticks) pair.
3. Lineage snapshots — descendants recorded BEFORE reparenting, as (pid,
   start_ticks) pairs, for any process whose environ is unreadable.

PID REUSE is handled explicitly: identity is (pid, start_ticks) where
start_ticks is field 22 of ``/proc/<pid>/stat``, which the kernel never reuses
for a recycled pid. Identity is re-verified immediately before EVERY signal.
UID is checked against our own uid so another user's process can never match.

BUDGETS — derived, and where the numbers come from
--------------------------------------------------
Repo evidence for run budgets is the shell wave drivers, which already wrap
every run in ``timeout``: ``.nett_perf/fast/wave.sh`` uses ``timeout 3000``,
``seedwave.sh`` 2600, ``drive_det.sh`` 700, ``drive_bench.sh`` 600,
``sync_ab.sh`` 400. These are WHOLE-RUN caps for specific fast recipes and vary
by 7.5x; the wheeled consistency runs take ~2h, which EXCEEDS wave.sh's 3000s.
There is therefore no single safe absolute default, and the shell layer already
owns absolute caps. Hence ``NETT_REAP_TIMEOUT`` defaults to 0 (disabled):
opt-in, for callers that know their own budget. The crash-gated grace is the
default mechanism because it keys off evidence, not off a guessed duration.

``NETT_REAP_CRASH_GRACE`` = 120s: once DEVICE_LOST is confirmed the child's own
path is ``os._exit`` after an artifact flush — near-instant. 120s is generous
for the flush while still being ~4% of the shortest observed run cap (400s).

``NETT_REAP_TERM_GRACE`` = 10s: SIGTERM -> SIGKILL. A wedged Isaac process is
already unresponsive; this only gives a still-live child time to flush.

Linux-only (``/proc``), stdlib-only. ``pynvml`` (declared dep ``nvidia-ml-py``,
already used by ``runtime.memory``) is used for GPU attribution when available,
with an ``nvidia-smi`` fallback; both are optional and failure is non-fatal.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

_LOG = logging.getLogger("nett.reap")

TOKEN_ENV = "NETT_REAP_TOKEN"

#: Grace for the child to self-exit AFTER crash evidence appears (seconds).
CRASH_GRACE = float(os.environ.get("NETT_REAP_CRASH_GRACE", "120"))
#: SIGTERM -> SIGKILL grace (seconds).
TERM_GRACE = float(os.environ.get("NETT_REAP_TERM_GRACE", "10"))
#: Absolute per-mode cap (seconds). 0 = disabled; the shell drivers own this.
ABSOLUTE_TIMEOUT = float(os.environ.get("NETT_REAP_TIMEOUT", "0"))
#: Join/kill poll interval (seconds).
POLL = float(os.environ.get("NETT_REAP_POLL", "0.5"))
#: Lineage snapshot interval (seconds).
LINEAGE_POLL = float(os.environ.get("NETT_REAP_LINEAGE_POLL", "5"))
#: Escape hatch: 1 restores the old unbounded-join behaviour.
DISABLED = os.environ.get("NETT_REAP_DISABLE", "0") == "1"


# --- /proc primitives ------------------------------------------------------


def _read_stat(pid: int) -> Optional[tuple[int, int, str]]:
    """Return ``(ppid, start_ticks, state)`` for *pid*, or None if it is gone.

    ``comm`` (field 2) can contain spaces AND parentheses, so the fields after
    it are parsed from the LAST ``)`` -- the standard-safe split.

    ``state`` matters: ``/proc/<pid>`` SURVIVES death. A killed-but-unreaped
    child stays as a ``Z`` (zombie) entry, with a readable start_ticks, until
    its parent waits on it. Treating a zombie as alive would make every reap
    burn both grace windows and then mis-report the corpse as a SURVIVOR.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            raw = fh.read().decode("utf-8", "replace")
        rest = raw[raw.rindex(")") + 2 :].split()
        # field 3 (state), field 4 (ppid), field 22 (starttime)
        return int(rest[1]), int(rest[19]), rest[0]
    except (FileNotFoundError, ProcessLookupError, ValueError, IndexError, PermissionError):
        return None


def _proc_uid(pid: int) -> Optional[int]:
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def _read_token(pid: int) -> Optional[str]:
    """Read ``NETT_REAP_TOKEN`` from *pid*'s environ, or None."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            blob = fh.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    prefix = (TOKEN_ENV + "=").encode()
    for entry in blob.split(b"\0"):
        if entry.startswith(prefix):
            return entry[len(prefix) :].decode("utf-8", "replace")
    return None


def _cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return ""


def _all_pids() -> list[int]:
    return [int(n) for n in os.listdir("/proc") if n.isdigit()]


def _prewarm_resource_tracker() -> None:
    """Start multiprocessing's resource_tracker before any token is stamped.

    See :meth:`TaskReaper.launch_scope`. Best-effort: a failure here only means
    the defensive :func:`_is_shared_infrastructure` check has to catch it.
    """
    try:
        from multiprocessing import resource_tracker

        resource_tracker.ensure_running()
    except Exception:  # pragma: no cover - never break a launch over this
        _LOG.debug("reap: resource_tracker pre-warm failed", exc_info=True)


def _is_shared_infrastructure(pid: int) -> bool:
    """True for processes that are SHARED by the whole worker, never task-owned.

    The multiprocessing resource_tracker is a per-interpreter singleton shared
    across every task this pool worker runs. It holds no GPU context and does
    none of the task's work, so killing it can never unwedge a task -- it only
    breaks the worker for SUBSEQUENT tasks. Defence in depth behind the
    pre-warm in ``launch_scope``.
    """
    cmd = _cmdline(pid)
    return "multiprocessing.resource_tracker" in cmd or "resource_tracker import main" in cmd


# --- identity --------------------------------------------------------------


@dataclass(frozen=True)
class ProcessIdentity:
    """A PID-reuse-safe process identity: ``(pid, start_ticks)``.

    ``start_ticks`` is ``/proc/<pid>/stat`` field 22. The kernel never reissues
    the same start time for a recycled pid, so this pair identifies one process
    for all time -- which is what makes a deferred kill safe.
    """

    pid: int
    start_ticks: int

    @classmethod
    def of(cls, pid: int) -> Optional["ProcessIdentity"]:
        stat = _read_stat(pid)
        if stat is None:
            return None
        return cls(pid=pid, start_ticks=stat[1])

    def alive(self) -> bool:
        """True only if THIS process still exists and is not a corpse.

        Excludes (a) pid-reuse impostors, via start_ticks, and (b) zombies: a
        ``Z`` entry is an exited process awaiting a parent's wait(), so there
        is nothing left to signal and nothing still holding the GPU.
        """
        stat = _read_stat(self.pid)
        return stat is not None and stat[1] == self.start_ticks and stat[2] != "Z"


@dataclass
class ReapReport:
    """Outcome of a :meth:`TaskReaper.reap`."""

    reason: str = ""
    terminated: list[int] = field(default_factory=list)
    killed: list[int] = field(default_factory=list)
    already_gone: list[int] = field(default_factory=list)
    survivors: list[int] = field(default_factory=list)
    gpu_pids: list[int] = field(default_factory=list)
    skipped_unowned: int = 0
    duration: float = 0.0

    def __str__(self) -> str:
        return (
            f"reap(reason={self.reason!r}) term={self.terminated} kill={self.killed} "
            f"gone={self.already_gone} gpu={self.gpu_pids} survivors={self.survivors} "
            f"skipped_unowned={self.skipped_unowned} in {self.duration:.1f}s"
        )


# --- the reaper ------------------------------------------------------------


class TaskReaper:
    """Tracks one task's process lineage and reaps it on crash/timeout.

    Usage (see module docstring for the wiring)::

        reaper = TaskReaper(task_key="run/cond/test", device=cfg.device)
        with reaper.launch_scope():
            p = ctx.Process(...); p.start()
        reaper.adopt(p.pid)
        ...
        reaper.reap("device-lost")

    Every public method is safe to call on an already-exited task and is
    idempotent; :meth:`reap` is bounded and never blocks indefinitely.
    """

    def __init__(
        self,
        task_key: str,
        device: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.task_key = task_key
        self.device = device
        self.logger = logger or _LOG
        self.token = f"{task_key}:{uuid.uuid4()}"
        self.root: Optional[ProcessIdentity] = None
        self._uid = os.getuid()
        self._lineage: set[ProcessIdentity] = set()
        self._lock = threading.Lock()
        self._sampler: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # -- launch-time identity ------------------------------------------

    @contextmanager
    def launch_scope(self) -> Iterator[None]:
        """Stamp the ownership token into the env for the duration of a spawn.

        Must wrap ``p.start()``. ``spawn`` snapshots ``os.environ`` at start(),
        so the child (and every descendant it forks) inherits the token; the
        parent's env is restored immediately after so a LATER healthy spawn is
        never mis-attributed to this task.

        The resource_tracker is pre-warmed FIRST, deliberately: spawn's
        ``Popen.__init__`` calls ``resource_tracker.ensure_running()``, so on
        the first spawn the tracker would otherwise be born INSIDE this scope
        and inherit the token -- making a reap kill the pool worker's shared
        infrastructure ("resource_tracker: process died unexpectedly ... Some
        resources might leak") rather than this task's processes. Starting it
        before the token exists keeps it correctly unowned.
        """
        _prewarm_resource_tracker()
        with self._lock:
            previous = os.environ.get(TOKEN_ENV)
            os.environ[TOKEN_ENV] = self.token
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop(TOKEN_ENV, None)
                else:
                    os.environ[TOKEN_ENV] = previous

    def adopt(self, pid: int) -> Optional[ProcessIdentity]:
        """Record the root child's identity. Call right after ``p.start()``."""
        self.root = ProcessIdentity.of(pid)
        if self.root is None:
            self.logger.warning(
                "reap[%s]: root pid %d exited before adopt; nothing to track",
                self.task_key, pid,
            )
            return None
        with self._lock:
            self._lineage.add(self.root)
        self.logger.debug(
            "reap[%s]: adopted root pid=%d start_ticks=%d token=%s device=%s",
            self.task_key, self.root.pid, self.root.start_ticks, self.token, self.device,
        )
        return self.root

    # -- lineage capture (BEFORE reparenting) --------------------------

    def sample_lineage(self) -> int:
        """Snapshot the root's descendant tree; returns the owned-set size.

        Called periodically so descendants are recorded while the parent links
        still exist. Once a spawn worker is reparented to PPID=1 the tree walk
        can no longer find it -- but its identity is already in ``_lineage``,
        and its inherited token independently proves ownership.
        """
        if self.root is None:
            return 0
        children: dict[int, list[int]] = {}
        for pid in _all_pids():
            stat = _read_stat(pid)
            if stat is not None:
                children.setdefault(stat[0], []).append(pid)

        found: set[ProcessIdentity] = set()
        stack = [self.root.pid]
        seen: set[int] = set()
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            for child in children.get(pid, ()):
                ident = ProcessIdentity.of(child)
                if ident is not None and self._same_user(child):
                    found.add(ident)
                stack.append(child)
        with self._lock:
            self._lineage |= found
            return len(self._lineage)

    def start_sampler(self) -> None:
        """Run :meth:`sample_lineage` on a daemon thread until :meth:`stop_sampler`."""
        if self._sampler is not None or self.root is None:
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.wait(LINEAGE_POLL):
                try:
                    self.sample_lineage()
                except Exception:  # never let sampling break a run
                    self.logger.debug("reap[%s]: lineage sample failed", self.task_key,
                                      exc_info=True)

        self._sampler = threading.Thread(
            target=loop, name=f"nett-reap-lineage[{self.task_key}]", daemon=True
        )
        self._sampler.start()

    def stop_sampler(self) -> None:
        self._stop.set()
        sampler, self._sampler = self._sampler, None
        if sampler is not None:
            sampler.join(timeout=LINEAGE_POLL + 1)

    # -- ownership evidence --------------------------------------------

    def _same_user(self, pid: int) -> bool:
        uid = _proc_uid(pid)
        return uid is not None and uid == self._uid

    def owns(self, pid: int) -> Optional[str]:
        """Return the ownership EVIDENCE for *pid*, or None if not ours.

        None means "do not touch". Appearing in nvidia-smi or having PPID=1 is
        NOT evidence and never reaches this function as a positive.
        """
        ident = ProcessIdentity.of(pid)
        if ident is None:
            return None
        if not self._same_user(pid):
            return None  # another user's process -- never ours
        if self.root is not None and ident == self.root:
            return "root-identity"
        if _is_shared_infrastructure(pid):
            return None  # worker-wide singleton; may carry our token, is not ours
        token = _read_token(pid)
        if token is not None and token == self.token:
            return "env-token"
        with self._lock:
            if ident in self._lineage:
                return "lineage-snapshot"
        return None

    def owned_targets(self) -> list[tuple[ProcessIdentity, str]]:
        """Every live process with positive ownership evidence.

        Union of: the root, its (possibly reparented) descendants found by
        token scan, and the pre-captured lineage snapshot. Scanning by token
        is what recovers PPID=1 orphans, whose parent link is gone.
        """
        targets: dict[ProcessIdentity, str] = {}
        candidates: set[int] = set(_all_pids())
        with self._lock:
            candidates |= {i.pid for i in self._lineage}
        if self.root is not None:
            candidates.add(self.root.pid)
        for pid in candidates:
            evidence = self.owns(pid)
            if evidence is None:
                continue
            ident = ProcessIdentity.of(pid)
            if ident is not None:
                targets[ident] = evidence
        return sorted(targets.items(), key=lambda kv: kv[0].pid)

    # -- GPU attribution -----------------------------------------------

    def gpu_compute_pids(self) -> list[int]:
        """PIDs with a compute context on the GPU (NVML first, nvidia-smi fallback).

        Returns CANDIDATES ONLY -- being here is not ownership evidence and
        confers no permission to kill. Scoped to ``self.device`` when known.
        """
        pids = self._gpu_pids_nvml()
        if pids is None:
            pids = self._gpu_pids_smi()
        return sorted(set(pids or []))

    def _gpu_pids_nvml(self) -> Optional[list[int]]:
        try:
            from pynvml import (  # declared dep: nvidia-ml-py (see runtime/memory.py)
                nvmlDeviceGetComputeRunningProcesses,
                nvmlDeviceGetCount,
                nvmlDeviceGetHandleByIndex,
                nvmlInit,
            )

            nvmlInit()  # idempotent; refcounted by NVML
            devices = [self.device] if self.device is not None else range(nvmlDeviceGetCount())
            out: list[int] = []
            for dev in devices:
                handle = nvmlDeviceGetHandleByIndex(int(dev))
                out += [int(p.pid) for p in nvmlDeviceGetComputeRunningProcesses(handle)]
            return out
        except Exception:
            self.logger.debug("reap[%s]: NVML compute-app query failed", self.task_key,
                              exc_info=True)
            return None

    def _gpu_pids_smi(self) -> list[int]:
        if shutil.which("nvidia-smi") is None:
            return []
        cmd = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
        if self.device is not None:
            cmd.append(f"--id={self.device}")
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except (subprocess.SubprocessError, OSError):
            return []
        return [int(l.strip()) for l in out.splitlines() if l.strip().isdigit()]

    # -- bounded escalation --------------------------------------------

    def _signal_if_owned(self, ident: ProcessIdentity, sig: int) -> bool:
        """Send *sig* to *ident*, re-verifying identity IMMEDIATELY beforehand.

        The re-check closes the PID-reuse race: between snapshot and signal the
        pid may have exited and been recycled onto an unrelated process.
        """
        if not ident.alive():
            return False
        if self.owns(ident.pid) is None:
            self.logger.warning(
                "reap[%s]: pid %d no longer owned at signal time; skipping",
                self.task_key, ident.pid,
            )
            return False
        try:
            os.kill(ident.pid, sig)
            return True
        except ProcessLookupError:
            return False  # exited between check and kill -- fine, idempotent
        except PermissionError:
            self.logger.warning(
                "reap[%s]: not permitted to signal pid %d; skipping", self.task_key, ident.pid,
            )
            return False

    def _wait_gone(self, idents: Iterable[ProcessIdentity], timeout: float) -> list[ProcessIdentity]:
        """Bounded wait; returns those still alive at the deadline."""
        deadline = time.monotonic() + max(0.0, timeout)
        remaining = list(idents)
        while remaining and time.monotonic() < deadline:
            remaining = [i for i in remaining if i.alive()]
            if not remaining:
                break
            time.sleep(POLL)
        return [i for i in remaining if i.alive()]

    def reap(self, reason: str) -> ReapReport:
        """SIGTERM -> grace -> SIGKILL every task-owned process. Idempotent, bounded.

        Total wall time is bounded by ~2 * TERM_GRACE, so a caller (the pool
        worker) is never blocked indefinitely by cleanup.
        """
        started = time.monotonic()
        report = ReapReport(reason=reason)
        self.stop_sampler()
        if DISABLED:
            self.logger.warning("reap[%s]: NETT_REAP_DISABLE=1; skipping reap", self.task_key)
            return report

        targets = self.owned_targets()

        # GPU compute apps: candidates are filtered through the SAME ownership
        # evidence. An unowned compute app (another user, another NETT task) is
        # counted and left strictly alone.
        for pid in self.gpu_compute_pids():
            evidence = self.owns(pid)
            if evidence is None:
                report.skipped_unowned += 1
                continue
            ident = ProcessIdentity.of(pid)
            if ident is not None and all(ident != t for t, _ in targets):
                targets.append((ident, f"gpu-compute+{evidence}"))
            if ident is not None:
                report.gpu_pids.append(pid)

        if not targets:
            self.logger.info("reap[%s]: nothing owned to reap (reason=%s)",
                             self.task_key, reason)
            report.duration = time.monotonic() - started
            return report

        for ident, evidence in targets:
            self.logger.warning(
                "reap[%s]: reason=%s target pid=%d evidence=%s cmd=%.120s",
                self.task_key, reason, ident.pid, evidence, _cmdline(ident.pid),
            )

        # SIGTERM everything owned, then one bounded grace for the whole set.
        for ident, _ in targets:
            if self._signal_if_owned(ident, signal.SIGTERM):
                report.terminated.append(ident.pid)
            else:
                report.already_gone.append(ident.pid)
        alive = self._wait_gone([i for i, _ in targets], TERM_GRACE)

        # SIGKILL the holdouts (a DEVICE_LOST-wedged Isaac never handles TERM).
        for ident in alive:
            if self._signal_if_owned(ident, signal.SIGKILL):
                report.killed.append(ident.pid)
        report.survivors = [i.pid for i in self._wait_gone(alive, TERM_GRACE)]

        report.duration = time.monotonic() - started
        if report.survivors:
            self.logger.error(
                "reap[%s]: SURVIVORS after SIGKILL: %s (uninterruptible sleep? "
                "GPU may stay held)", self.task_key, report.survivors,
            )
        self.logger.warning("reap[%s]: %s", self.task_key, report)
        return report


# --- crash evidence + bounded join ----------------------------------------


class ReapedTaskError(RuntimeError):
    """Base: a task was ended by the DEVICE_LOST/timeout path, not by its own logic.

    Exists so ``nett.py::_task_waiter`` can catch ONE type and continue the wave
    for exactly this path, while EVERY other exception keeps today's fail-fast
    re-raise. A bare ``RuntimeError`` (what ``_spawn_mode_subprocess`` raises
    today) is indistinguishable from an ordinary failure, which is why the
    typed signal is required. The run must still end NONZERO -- see the
    ``_task_waiter`` wiring proposal.

    Lives here, not in ``crash_guard``: this is the PARENT-side contract, and
    ``reap`` imports nothing from ``nett_skrl`` (stdlib + optional pynvml only),
    so ``task_runner`` and ``nett`` can both import it with no cycle. The parent
    never has to import the child-side guard.
    """


class DeviceLostError(ReapedTaskError):
    """Confirmed renderer DEVICE_LOST: child self-exited, or was reaped wedged."""


class TaskTimeoutError(ReapedTaskError):
    """Child exceeded an explicitly-configured ``NETT_REAP_TIMEOUT`` and was reaped."""


class StallError(ReapedTaskError):
    """Child made no env-step progress for the stall budget and self-exited.

    A ``ReapedTaskError`` on purpose: this IS a casualty to tolerate so the wave
    continues. The hang it covers is a Kit renderer wedge inside
    ``SimulationContext.render`` -> ``self._app.update()`` (measured 2026-07-30: GPU
    at 2% while the process burns 136% CPU), which produces NO ``DEVICE_LOST`` and no
    crash signature at all -- so ``crash_guard`` never fires and, with
    ``NETT_REAP_TIMEOUT`` disabled by default, the parent's ``join`` is unbounded. It
    is transient in the same sense DEVICE_LOST is: the same config re-run usually
    proceeds (2-4 cells of 8 hang, and a relaunched seed may not).
    """


class VramOomError(RuntimeError):
    """Child ran out of VRAM and self-exited via crash_guard's bounded OOM path.

    Deliberately NOT a ``ReapedTaskError``: that base means "a casualty to tolerate so
    the wave can continue", and an OOM is not transient -- the same config on the same
    GPU will do it again. Only a dry-run probe arms the OOM trigger, and the probe
    catches this and reads it as "too big"; if it ever escapes to ``_task_waiter`` it
    should fail fast like any other real error, which this base gives for free.
    """


class DeviceLostRunError(ReapedTaskError):
    """Aggregate: the run finished, but N tasks died on the DEVICE_LOST/timeout path.

    Raised by ``_task_waiter`` AFTER every remaining task has been scheduled and
    awaited. This is what keeps "continue the wave" from becoming "fail
    silently": the wave completes, and the run still ends NONZERO.
    """

    def __init__(self, failures: "list[tuple[str, BaseException]]") -> None:
        self.failures = list(failures)
        detail = "; ".join(f"{key}: {exc}" for key, exc in self.failures)
        super().__init__(
            f"{len(self.failures)} task(s) ended by DEVICE_LOST/timeout; "
            f"remaining tasks completed. Failures -> {detail}"
        )


#: SIGALRM kernel backstop (crash_guard arms it when the GIL is too wedged for
#: the watchdog to run). multiprocessing reports signals as negatives (-14); the
#: 128+signo form (142) is included for exec-wrapper/shell-reported paths.
SIGALRM_EXIT_CODES = frozenset({-14, 142})


def device_lost_exit_code() -> int:
    """The child's DEVICE_LOST exit code (``EX_TEMPFAIL`` 75 by default).

    Read from ``NETT_DEVICE_LOST_EXIT_CODE`` at CALL time with the same name and
    default that ``crash_guard`` uses, deliberately WITHOUT importing it: the
    value is a parent/child contract carried by the environment, and the child
    inherits the parent's env, so an operator override is automatically seen by
    both sides. Call-time (not import-time) so an override set after import, or
    in a test, still takes effect.
    """
    try:
        return int(os.environ.get("NETT_DEVICE_LOST_EXIT_CODE", "75"))
    except ValueError:
        return 75


def is_device_lost_exit(exitcode: Optional[int]) -> bool:
    """True if *exitcode* is crash_guard's DEVICE_LOST signature.

    Covers the watchdog's clean hard-exit (75 / env override) and the SIGALRM
    kernel backstop (-14 / 142). Both are deliberately OUTSIDE
    ``task_runner._is_tolerated_isaac_teardown_exit``'s tolerated set.
    """
    if exitcode is None:
        return False
    return exitcode == device_lost_exit_code() or exitcode in SIGALRM_EXIT_CODES


def vram_oom_exit_code() -> int:
    """The child's out-of-VRAM exit code (76 by default).

    Same parent/child env contract as :func:`device_lost_exit_code`, and read at call
    time for the same reason. Deliberately distinct from DEVICE_LOST's: an OOM is not
    a transient renderer casualty, it is "this many envs do not fit" -- which for a
    sizing probe is the answer it went looking for.
    """
    try:
        return int(os.environ.get("NETT_VRAM_OOM_EXIT_CODE", "76"))
    except ValueError:
        return 76


def is_vram_oom_exit(exitcode: Optional[int]) -> bool:
    """True if *exitcode* is crash_guard's out-of-VRAM signature.

    Note the SIGALRM backstop codes are NOT included: they are shared with
    DEVICE_LOST and cannot be attributed to either cause, so they stay with the
    conservative (casualty) reading.
    """
    if exitcode is None:
        return False
    return exitcode == vram_oom_exit_code()


def stall_exit_code() -> int:
    """The child's no-progress exit code (77 by default).

    Same parent/child env contract as :func:`device_lost_exit_code`, read at call time
    for the same reason. Distinct from 75 (DEVICE_LOST) and 76 (VRAM OOM) so the three
    causes stay separable in a wave's failure list -- they want different responses: a
    stall is worth retrying, an OOM is not.
    """
    try:
        return int(os.environ.get("NETT_STALL_EXIT_CODE", "77"))
    except ValueError:
        return 77


def is_stall_exit(exitcode: Optional[int]) -> bool:
    """True if *exitcode* is stall_guard's no-progress signature.

    The SIGALRM backstop codes are deliberately NOT included: they are shared with
    DEVICE_LOST (whose guard arms the same timer), so an ambiguous -14/142 keeps the
    conservative DEVICE_LOST reading rather than being claimed by this newer path.
    """
    if exitcode is None:
        return False
    return exitcode == stall_exit_code()


def join_with_reap(
    proc,
    reaper: TaskReaper,
    crash_evidence: Optional[Callable[[], bool]] = None,
    crash_grace: Optional[float] = None,
    absolute_timeout: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Join *proc*, bounding ONLY the teardown/crash window. Replaces ``p.join()``.

    Returns one of ``"exited"``, ``"reaped-crash"``, ``"reaped-timeout"``.

    A healthy run of ANY duration joins normally: the loop polls forever until
    either (a) the child exits -- the overwhelmingly common path, including the
    child's own DEVICE_LOST hard-exit -- or (b) crash evidence appears, which
    starts the ``crash_grace`` countdown, or (c) an explicitly-configured
    absolute cap elapses (default: disabled).
    """
    log = logger or reaper.logger
    grace = CRASH_GRACE if crash_grace is None else crash_grace
    cap = ABSOLUTE_TIMEOUT if absolute_timeout is None else absolute_timeout

    if DISABLED:
        proc.join()
        return "exited"

    reaper.start_sampler()
    started = time.monotonic()
    crash_deadline: Optional[float] = None
    try:
        while True:
            proc.join(timeout=POLL)
            if proc.exitcode is not None:
                return "exited"

            if crash_deadline is None and crash_evidence is not None:
                try:
                    if crash_evidence():
                        crash_deadline = time.monotonic() + grace
                        log.error(
                            "reap[%s]: DEVICE_LOST evidence for pid=%s; allowing %.0fs "
                            "to self-exit before reaping", reaper.task_key, proc.pid, grace,
                        )
                except Exception:
                    log.debug("reap[%s]: crash predicate raised", reaper.task_key,
                              exc_info=True)

            if crash_deadline is not None and time.monotonic() >= crash_deadline:
                log.error(
                    "reap[%s]: child pid=%s did not self-exit within %.0fs of DEVICE_LOST; "
                    "reaping", reaper.task_key, proc.pid, grace,
                )
                reaper.reap("device-lost")
                _drain(proc)
                return "reaped-crash"

            if cap > 0 and (time.monotonic() - started) > cap:
                log.error(
                    "reap[%s]: child pid=%s exceeded NETT_REAP_TIMEOUT=%.0fs; reaping",
                    reaper.task_key, proc.pid, cap,
                )
                reaper.reap("absolute-timeout")
                _drain(proc)
                return "reaped-timeout"
    finally:
        reaper.stop_sampler()


def _drain(proc, timeout: float = 30.0) -> None:
    """Reap the zombie after a kill so the child's exit status is collected."""
    proc.join(timeout=timeout)
    if proc.exitcode is None:
        _LOG.error("reap: pid=%s still not joinable after reap; leaking handle", proc.pid)
