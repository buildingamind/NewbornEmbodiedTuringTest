"""Fault-injection utilities for the DEVICE_LOST crash-guard / reap tests.

These build the *hazards* the crash-guard (``nett_skrl.runtime.crash_guard``)
and the reaper (``nett_skrl.runtime.reap``) are supposed to handle:

    * a child that prints the real Kit ``ERROR_DEVICE_LOST`` banner and hangs,
    * a child that hangs with no DEVICE_LOST banner at all,
    * a healthy long-running child (must NOT be killed),
    * a ``multiprocessing.spawn``-shaped orphan reparented to PPID=1,
    * a *sentinel*: an unrelated process that MUST survive every reap,
    * PID-reuse: an ownership record whose start-time no longer matches,
    * a fake ``nvidia-smi`` so GPU-process cleanup is testable without a GPU.

Safety contract (non-negotiable — these tests run on a box with live 13h
Isaac runs on its GPUs):

    1. Every process a test creates is registered with a :class:`ProcessOrchard`
       and killed by its finalizer, *including when the test fails mid-way*.
    2. The orchard refuses to signal any PID it did not create, and re-checks
       the PID's start-time before signalling so a recycled PID is never hit.
    3. Nothing here talks to a real GPU. GPU discovery is faked via a stub
       ``nvidia-smi`` on PATH (or by monkeypatching, see ``fake_gpu_processes``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil

# carb log levels, as crash_guard reads them: LEVEL_WARN=0, LEVEL_ERROR=1,
# LEVEL_FATAL=2. A carb log record is (source, level, message).
CARB_WARN, CARB_ERROR, CARB_FATAL = 0, 1, 2

# The two records Kit actually emits on the observed renderer GPU crash. These
# are the confirmed-crash contract: detection MUST fire on each.
DEVICE_LOST_RECORDS = {
    "vulkan_device_lost": (
        "carb.graphics-vulkan.plugin", CARB_ERROR,
        "VkResult: ERROR_DEVICE_LOST",
    ),
    "gpu_foundation_crash": (
        "gpu.foundation.plugin", CARB_ERROR,
        "A GPU crash occurred. Exiting the application...",
    ),
}

# Records that look GPU-ish but are NOT a confirmed device loss. Detection MUST
# reject every one: a false positive force-kills a healthy multi-hour run, which
# is a worse failure than the hang this all exists to fix.
NEAR_MISS_RECORDS = {
    # Right words, wrong (non-crash-emitting) plugin.
    "wrong_source": ("omni.usd", CARB_ERROR, "VkResult: ERROR_DEVICE_LOST"),
    # Right source + words, but only a warning — not confirmed fatal.
    "warn_level_only": (
        "carb.graphics-vulkan.plugin", CARB_WARN, "VkResult: ERROR_DEVICE_LOST",
    ),
    # Right source, different (non-fatal) Vulkan result.
    "other_vk_result": (
        "carb.graphics-vulkan.plugin", CARB_ERROR, "VkResult: ERROR_UNKNOWN",
    ),
    # A GPU-ish Python failure that must stay on the ordinary fail-fast path.
    "cuda_oom": ("py.warnings", CARB_ERROR, "CUDA out of memory."),
    # The phrase appearing in prose from an unrelated source.
    "prose_mention": (
        "omni.kit.app", CARB_ERROR, "note: ERROR_DEVICE_LOST is not happening here",
    ),
    # gpu.foundation.plugin talking about something else entirely.
    "gpu_foundation_benign": (
        "gpu.foundation.plugin", CARB_ERROR, "failed to create a texture",
    ),
}

# What the crash looks like in the child's stdout — used to shape injected
# children so their output matches the real incident.
DEVICE_LOST_BANNER = (
    "[Error] [carb.graphics-vulkan.plugin] VkResult: ERROR_DEVICE_LOST\n"
    "[Error] [gpu.foundation.plugin] A GPU crash occurred. Exiting the application...\n"
)

_GRACE_S = 5.0


class ProcessOrchard:
    """Owns every PID a test creates, and guarantees they all die.

    Ownership is (pid, create_time). ``kill_all`` is idempotent and is wired to
    a fixture finalizer, so processes are reaped even when assertions fail.
    """

    def __init__(self) -> None:
        # pid -> (create_time, popen_or_None, label)
        self._owned: dict[int, tuple[float, subprocess.Popen | None, str]] = {}

    # -- registration ----------------------------------------------------

    def adopt(self, pid: int, *, popen: subprocess.Popen | None = None,
              label: str = "") -> int:
        """Record a PID we created, pinning its start-time for ownership.

        Tolerates an already-exited PID (start-time ``-1``, reads as dead): an
        injected child may self-exit before the test gets to adopt it, and a
        raise here would skip the finalizer for everything else.
        """
        try:
            created = _create_time(pid)
        except psutil.NoSuchProcess:
            created = -1.0
        self._owned[pid] = (created, popen, label)
        return pid

    @property
    def pids(self) -> list[int]:
        return sorted(self._owned)

    def popen(self, pid: int) -> subprocess.Popen:
        """The Popen handle for an owned PID (orphans have none)."""
        self._assert_owned(pid)
        popen = self._owned[pid][1]
        assert popen is not None, f"PID {pid} has no Popen handle"
        return popen

    def create_time(self, pid: int) -> float:
        """The start-time pinned at adopt() — the ownership half of PID+time."""
        return self._owned[pid][0]

    # -- queries ---------------------------------------------------------

    def is_alive(self, pid: int) -> bool:
        """Alive *and still the process we adopted* (PID reuse reads as dead)."""
        self._assert_owned(pid)
        try:
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                return False
            return p.create_time() == self._owned[pid][0]
        except psutil.NoSuchProcess:
            return False

    def wait_gone(self, pid: int, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_alive(pid):
                return True
            time.sleep(0.02)
        return not self.is_alive(pid)

    def wait_for_output(self, path: Path, needle: str, timeout: float = 30.0) -> str:
        """Block until ``needle`` shows up in ``path``; return the text."""
        deadline = time.monotonic() + timeout
        text = ""
        while time.monotonic() < deadline:
            if path.exists():
                text = path.read_text(errors="replace")
                if needle in text:
                    return text
            time.sleep(0.02)
        raise AssertionError(
            f"timed out waiting for {needle!r} in {path}; got: {text!r}"
        )

    # -- teardown --------------------------------------------------------

    def _assert_owned(self, pid: int) -> None:
        if pid not in self._owned:
            raise AssertionError(
                f"refusing to touch PID {pid}: not created by this test"
            )

    def kill(self, pid: int) -> None:
        """SIGKILL one owned PID, but only if it is still the same process."""
        self._assert_owned(pid)
        pinned, popen, _ = self._owned[pid]
        try:
            proc = psutil.Process(pid)
            if proc.create_time() != pinned:
                return  # PID was recycled — not ours any more, hands off.
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        if popen is not None:
            try:
                popen.wait(timeout=_GRACE_S)
            except Exception:
                pass
        else:
            # Orphans (PPID=1) cannot be waited on; just confirm they are gone.
            self.wait_gone(pid, timeout=_GRACE_S)

    def kill_all(self) -> None:
        for pid in list(self._owned):
            try:
                self.kill(pid)
            except Exception:
                pass
        self._owned.clear()


def _create_time(pid: int) -> float:
    return psutil.Process(pid).create_time()


# ---------------------------------------------------------------------------
# Child-process shapes
# ---------------------------------------------------------------------------

# Each injected child carries a per-test marker in argv so we can find it again
# (and so we can never mistake somebody else's process for ours).
def new_marker(kind: str) -> str:
    return f"NETT_TEST_{kind}_{uuid.uuid4().hex}"


_HANG_SRC = r"""
import os, sys, time
banner, out_path, marker = sys.argv[1], sys.argv[2], sys.argv[3]
artifacts = sys.argv[4] if len(sys.argv) > 4 else ""
if artifacts:
    # Emulate the child's on-disk work *before* the crash: write, flush, fsync.
    d = os.path.join(artifacts, "logs")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "eval_metrics.csv")
    with open(p, "w") as f:
        f.write("eval_step,condition,brain_id,mean_reward,timesteps\n")
        f.write("50000,Object1,1,0.5,400\n")
        f.flush()
        os.fsync(f.fileno())
with open(out_path, "w", buffering=1) as f:
    if banner:
        f.write(banner)
    f.write("READY %s\n" % marker)
    os.fsync(f.fileno())
# The bug: after the GPU crash the process never exits. Emulate exactly that.
while True:
    time.sleep(0.05)
"""


def spawn_hung_child(
    orchard: ProcessOrchard,
    tmp_path: Path,
    *,
    banner: str = "",
    artifacts_root: Path | None = None,
    label: str = "hung",
) -> tuple[int, Path]:
    """A child that (optionally) emits ``banner`` and then hangs forever.

    ``banner=DEVICE_LOST_BANNER`` reproduces the observed failure; ``banner=""``
    reproduces a *generic* hang with no device loss (must be handled by the
    timeout path, not the DEVICE_LOST path).

    Returns ``(pid, output_path)``. Waits until the child is up.
    """
    marker = new_marker(label.upper())
    out = tmp_path / f"{marker}.out"
    args = [
        sys.executable, "-c", _HANG_SRC,
        banner, str(out), marker, str(artifacts_root or ""),
        marker,  # trailing argv marker: identifies this proc in `ps`/cmdline
    ]
    popen = subprocess.Popen(args, start_new_session=False)
    pid = orchard.adopt(popen.pid, popen=popen, label=label)
    orchard.wait_for_output(out, f"READY {marker}")
    return pid, out


_HEALTHY_SRC = r"""
import os, sys, time
out_path, marker = sys.argv[1], sys.argv[2]
with open(out_path, "w", buffering=1) as f:
    f.write("READY %s\n" % marker)
    os.fsync(f.fileno())
    # A healthy long training run: alive for hours, making progress, silent.
    while True:
        time.sleep(0.25)
        f.write("progress\n")
"""


def spawn_healthy_long_child(
    orchard: ProcessOrchard, tmp_path: Path
) -> tuple[int, Path]:
    """A well-behaved child that legitimately runs for a very long time.

    Regression guard: training runs for hours, so no new join bound may kill it.
    """
    marker = new_marker("HEALTHY")
    out = tmp_path / f"{marker}.out"
    popen = subprocess.Popen(
        [sys.executable, "-c", _HEALTHY_SRC, str(out), marker, marker]
    )
    pid = orchard.adopt(popen.pid, popen=popen, label="healthy")
    orchard.wait_for_output(out, f"READY {marker}")
    return pid, out


def spawn_resource_tracker_lookalike(
    orchard: ProcessOrchard, tmp_path: Path
) -> tuple[int, Path]:
    """A process whose cmdline matches multiprocessing's resource_tracker.

    Spawn this INSIDE ``launch_scope`` so it inherits the ownership token: it is
    the trap case for the reaper — token-owned by the letter of the rule, but a
    per-worker singleton that must never be killed.
    """
    marker = new_marker("TRACKER")
    out = tmp_path / f"{marker}.out"
    popen = subprocess.Popen([
        sys.executable, "-c", _HEALTHY_SRC, str(out), marker,
        "-c", "from multiprocessing.resource_tracker import main;main(4)",
        marker,
    ])
    pid = orchard.adopt(popen.pid, popen=popen, label="resource_tracker")
    orchard.wait_for_output(out, f"READY {marker}")
    return pid, out


def spawn_sentinel(orchard: ProcessOrchard, tmp_path: Path) -> int:
    """An unrelated bystander process. Any reap that kills this is a bug.

    Stands in for the *other* seven Isaac runs on the box: same user, same
    python, plausibly the same process group — but not this task's to kill.
    """
    marker = new_marker("SENTINEL")
    out = tmp_path / f"{marker}.out"
    popen = subprocess.Popen(
        [sys.executable, "-c", _HEALTHY_SRC, str(out), marker, marker]
    )
    pid = orchard.adopt(popen.pid, popen=popen, label="sentinel")
    orchard.wait_for_output(out, f"READY {marker}")
    return pid


# Double-fork: the grandchild is reparented to init (PPID=1), exactly like the
# multiprocessing.spawn workers left behind by the hang. `spawn_main` appears in
# its cmdline so cmdline-based matchers see the real shape.
_ORPHAN_SRC = r"""
import os, sys, time
pid_path, marker = sys.argv[1], sys.argv[2]
if os.fork() != 0:
    os._exit(0)                      # parent dies -> grandchild reparents to 1
os.setsid()
src = (
    "import os, sys, time\n"
    "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
    "while True: time.sleep(0.05)\n"
)
os.execv(sys.executable, [
    sys.executable, "-c", src, pid_path,
    "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=0)",
    marker,
])
"""


def spawn_orphan_worker(orchard: ProcessOrchard, tmp_path: Path) -> tuple[int, str]:
    """A PPID=1, spawn-shaped orphan worker. Returns ``(pid, marker)``.

    This is what the hang leaves behind: the task's process tree is gone but the
    spawn workers survive under init, holding VRAM.
    """
    marker = new_marker("ORPHAN")
    pid_file = tmp_path / f"{marker}.pid"
    subprocess.run(
        [sys.executable, "-c", _ORPHAN_SRC, str(pid_file), marker],
        check=True, timeout=30,
    )  # returns immediately: the intermediate parent exits at once
    deadline = time.monotonic() + 30
    pid = 0
    while time.monotonic() < deadline:
        if pid_file.exists() and pid_file.read_text().strip():
            pid = int(pid_file.read_text().strip())
            break
        time.sleep(0.02)
    assert pid, f"orphan {marker} never reported a PID"
    orchard.adopt(pid, popen=None, label="orphan")
    # Confirm the reparent actually happened before any test relies on it.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and psutil.Process(pid).ppid() != 1:
        time.sleep(0.02)
    assert psutil.Process(pid).ppid() == 1, "orphan was not reparented to PPID=1"
    return pid, marker


# ---------------------------------------------------------------------------
# PID-reuse simulation
# ---------------------------------------------------------------------------


# The load-bearing incident shape: crash_guard's watchdog calls os._exit(75),
# which runs NO atexit hooks -- so the child's spawn workers are never
# terminated and its CUDA/RTX context is never released. They reparent to PPID=1
# and keep holding ~5GB of VRAM. The parent's join returns CLEANLY, so only the
# exit-code-triggered reap frees the GPU. This child reproduces exactly that.
_CRASH_WITH_ORPHAN_SRC = r"""
import os, sys, time
orphan_pidfile, marker, exit_code = sys.argv[1], sys.argv[2], int(sys.argv[3])
if os.fork() == 0:
    os.setsid()
    src = (
        "import os, sys, time\n"
        "open(sys.argv[1], 'w').write(str(os.getpid()))\n"
        "while True: time.sleep(0.05)\n"
    )
    os.execv(sys.executable, [
        sys.executable, "-c", src, orphan_pidfile,
        "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=0)",
        marker,
    ])
for _ in range(3000):   # let the worker report itself before we die
    try:
        if open(orphan_pidfile).read().strip():
            break
    except OSError:
        pass
    time.sleep(0.01)
os._exit(exit_code)     # crash_guard's hard exit: no atexit, no child cleanup
"""


def spawn_crashing_child_with_orphan(
    orchard: ProcessOrchard,
    tmp_path: Path,
    orphan_pidfile: Path,
    *,
    exit_code: int = 75,
) -> subprocess.Popen:
    """A child that leaks a PPID=1 spawn worker and then hard-exits ``exit_code``.

    Spawn this inside ``TaskReaper.launch_scope()`` so both it and the leaked
    worker inherit the ownership token. The orphan is adopted by the orchard
    before this returns, so it is cleaned up even if the test fails mid-assert.
    """
    marker = new_marker("CRASH")
    popen = subprocess.Popen([
        sys.executable, "-c", _CRASH_WITH_ORPHAN_SRC,
        str(orphan_pidfile), marker, str(exit_code),
        marker,
    ])
    orchard.adopt(popen.pid, popen=popen, label="crashing-child")

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if orphan_pidfile.exists() and orphan_pidfile.read_text().strip():
            orchard.adopt(int(orphan_pidfile.read_text().strip()), label="leaked-worker")
            return popen
        time.sleep(0.02)
    raise AssertionError(f"crashing child {marker} never leaked a worker")


class StubProc:
    """A ``multiprocessing.Process`` stand-in with a fixed exit code.

    For the ordering tests, where the point is which CHECK runs first, not what
    a real process does.
    """

    def __init__(self, pid: int = 424242, exitcode: int | None = -9) -> None:
        self.pid = pid
        self.exitcode = exitcode
        self.join_calls = 0
        self.started = False

    def start(self) -> None:
        # No real OS process: these tests monkeypatch TaskReaper and
        # join_with_reap, so nothing ever inspects a live child. start() only
        # needs to satisfy _spawn_mode_subprocess's call inside launch_scope().
        self.started = True

    def join(self, timeout: float | None = None) -> None:
        self.join_calls += 1


def fake_spawn_context(monkeypatch, launcher) -> None:
    """Make ``mp.get_context("spawn").Process`` launch ``launcher()`` instead.

    ``_spawn_mode_subprocess`` does ``import multiprocessing as mp`` then
    ``mp.get_context("spawn")``, so patching ``mp.get_context`` swaps the child
    out while leaving every line of the production wiring — the reaper, the
    launch_scope token, adopt, join_with_reap, the exit-code branch — running for
    real against a real OS process.
    """
    import multiprocessing as mp

    class _Process:
        def __init__(self, target=None, args=(), daemon=None, **kwargs) -> None:
            self._popen: subprocess.Popen | None = None

        def start(self) -> None:
            self._popen = launcher()

        @property
        def pid(self) -> int:
            assert self._popen is not None
            return self._popen.pid

        def join(self, timeout: float | None = None) -> None:
            assert self._popen is not None
            try:
                self._popen.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

        @property
        def exitcode(self) -> int | None:
            assert self._popen is not None
            return self._popen.poll()

    class _Ctx:
        Process = _Process

    monkeypatch.setattr(mp, "get_context", lambda method=None: _Ctx())


def stub_spawn_context(monkeypatch, proc) -> None:
    """Make ``mp.get_context("spawn").Process(...)`` hand back *proc* itself.

    For tests about the parent's control flow (which check runs first, whether
    reap() is called), where a real child would only add noise and PIDs. Use
    :func:`fake_spawn_context` when a real process is the point.
    """
    import multiprocessing as mp

    class _Ctx:
        @staticmethod
        def Process(target=None, args=(), daemon=None, **kwargs):
            return proc

    monkeypatch.setattr(mp, "get_context", lambda method=None: _Ctx())


def fake_gpu_compute_apps_from_pidfile(
    monkeypatch, tmp_path: Path, pidfile: Path
) -> Path:
    """Like :func:`fake_gpu_compute_apps`, but reads its PID list at CALL time.

    The leaked worker's PID does not exist when the test is written, only once
    the child has forked it — so the stub ``nvidia-smi`` reads the pidfile the
    worker wrote, exactly as the real one would report a live compute context.
    """
    from nett_skrl.runtime import reap as _reap

    monkeypatch.setattr(_reap.TaskReaper, "_gpu_pids_nvml", lambda self: None)
    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "nvidia-smi"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *query-compute-apps* ]]; then\n'
        f'  cat "{pidfile}" 2>/dev/null || true\n'
        "  echo\n"
        "fi\n"
        "exit 0\n"
    )
    stub.chmod(0o755)
    monkeypatch.setattr(
        _reap.shutil, "which", lambda name: str(stub) if name == "nvidia-smi" else None
    )
    return stub


def pid_that_is_free() -> int:
    """A PID that is not currently in use (for 'already gone' code paths)."""
    for candidate in range(4_194_300, 4_000_000, -1):
        if not psutil.pid_exists(candidate):
            return candidate
    raise AssertionError("could not find a free PID")


class PopenProc:
    """Adapts a :class:`subprocess.Popen` to the bits ``join_with_reap`` uses.

    ``join_with_reap`` only touches ``proc.join(timeout=…)``, ``proc.exitcode``
    and ``proc.pid``. Adapting Popen rather than spawning a real
    ``multiprocessing.Process`` keeps the injected child's behaviour under the
    test's direct control (a spawn child would have to re-import the test
    module) while exercising the production join loop unmodified.
    """

    def __init__(self, popen: subprocess.Popen) -> None:
        self._popen = popen
        self.pid = popen.pid

    def join(self, timeout: float | None = None) -> None:
        try:
            self._popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    @property
    def exitcode(self) -> int | None:
        return self._popen.poll()


# ---------------------------------------------------------------------------
# GPU faking — never touches a real device
# ---------------------------------------------------------------------------


def fake_gpu_compute_apps(monkeypatch, tmp_path: Path, *, pids: list[int]) -> Path:
    """Make ``TaskReaper.gpu_compute_pids`` report exactly ``pids``.

    Two things happen, and both matter:

    1. NVML is forced to miss (it *works* on this box — leaving it live would
       enumerate the real 13h Isaac runs and feed their PIDs to a reaper).
    2. A stub ``nvidia-smi`` goes first on PATH, so the production
       ``_gpu_pids_smi`` fallback — the shell-out and its parsing — is what
       actually runs.

    Net effect: the real GPU is never queried, and the code path under test is
    still the real one.
    """
    from nett_skrl.runtime import reap as _reap

    monkeypatch.setattr(_reap.TaskReaper, "_gpu_pids_nvml", lambda self: None)

    bindir = tmp_path / "fakebin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "nvidia-smi"
    lines = "".join(f'printf "%s\\n" "{int(p)}"\n' for p in pids)
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$*" == *query-compute-apps* ]]; then\n'
        f"{lines}"
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    stub.chmod(0o755)
    monkeypatch.setattr(
        _reap.shutil, "which", lambda name: str(stub) if name == "nvidia-smi" else None
    )
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    return stub


# ---------------------------------------------------------------------------
# Artifact trees
# ---------------------------------------------------------------------------

_TEST_CSV_HEADER = [
    "env_id", "episode", "step", "agent.x", "agent.y", "agent.angle",
    "head.flexion", "head.lateral", "left.monitor", "right.monitor",
    "correct.monitor", "experiment.phase", "imprint.cond", "test.cond",
]


def write_test_phase_csv(path: Path, *, n_rows: int = 6, agent_x: float = -20.0,
                         correct: str = "left") -> None:
    """Write the per-condition ``logs/test_<cond>_<i>.csv`` that ``test_viz``
    reads. Shape mirrors ``tests/test_analysis.py``'s fixture rows."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TEST_CSV_HEADER)
        w.writeheader()
        for i in range(n_rows):
            row = {h: "" for h in _TEST_CSV_HEADER}
            row.update({
                "env_id": "0", "episode": "0", "step": str(i),
                "agent.x": str(agent_x), "agent.y": "0.0", "agent.angle": "0.0",
                "head.flexion": "0.0", "head.lateral": "0.0",
                "left.monitor": "L.mov", "right.monitor": "R.mov",
                "correct.monitor": correct,
                "experiment.phase": "test", "imprint.cond": "Object1",
                "test.cond": "rest",
            })
            w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def crashed_run_artifacts(root: Path, *, condition: str = "Object1") -> Path:
    """Build the on-disk tree a DEVICE_LOST-crashed run must leave behind.

    Durability criterion (per the run owner): the child does NOT write
    summary.json — ``analysis.analyze()`` derives it post-hoc. So what must
    survive the forced exit is the child's own artifacts, and ``analyze()`` must
    still turn them into a valid summary.
    """
    cond = root / condition
    (cond / "logs").mkdir(parents=True, exist_ok=True)
    (cond / "wandb_runs" / "brain_1" / "checkpoints").mkdir(parents=True, exist_ok=True)
    write_test_phase_csv(cond / "logs" / f"test_{condition}_0.csv")
    return root


def assert_all_dead(orchard: ProcessOrchard, pids: list[int], timeout: float = 15.0) -> None:
    still = [p for p in pids if not orchard.wait_gone(p, timeout=timeout)]
    assert not still, f"expected these PIDs to be reaped, still alive: {still}"


def assert_alive(orchard: ProcessOrchard, pid: int, msg: str) -> None:
    assert orchard.is_alive(pid), msg


def send_and_ignore(pid: int, sig: int) -> None:
    """Best-effort signal used only by tests on their own PIDs."""
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


__all__ = [
    "CARB_ERROR",
    "CARB_FATAL",
    "CARB_WARN",
    "DEVICE_LOST_BANNER",
    "DEVICE_LOST_RECORDS",
    "NEAR_MISS_RECORDS",
    "PopenProc",
    "ProcessOrchard",
    "StubProc",
    "assert_alive",
    "assert_all_dead",
    "crashed_run_artifacts",
    "fake_gpu_compute_apps",
    "fake_gpu_compute_apps_from_pidfile",
    "fake_spawn_context",
    "stub_spawn_context",
    "new_marker",
    "pid_that_is_free",
    "send_and_ignore",
    "spawn_crashing_child_with_orphan",
    "spawn_healthy_long_child",
    "spawn_hung_child",
    "spawn_orphan_worker",
    "spawn_resource_tracker_lookalike",
    "spawn_sentinel",
    "write_test_phase_csv",
]
