"""Bounded hard-exit guard for renderer GPU crashes (VkResult ERROR_DEVICE_LOST),
and -- opt-in, for dry-run probes only -- for out-of-VRAM (ERROR_OUT_OF_DEVICE_MEMORY).

WHY THIS EXISTS
---------------
A transient renderer GPU crash makes Kit log::

    [carb.graphics-vulkan.plugin] VkResult: ERROR_DEVICE_LOST
    [gpu.foundation.plugin] A GPU crash occurred. Exiting the application...

...and then the Python process HANGS instead of exiting -- observed 2026-07-14
for ~13h at 0% CPU while still holding ~5GB of VRAM, stalling every wait-on-PID
in the 8-way wave driver. Kit's own "Exiting the application" path deadlocks, so
neither ``atexit`` nor ``sim_app.close()`` can be trusted once the device is
lost: the thread that would run them never comes back.

THE SIGNAL
----------
``carb.logging.acquire_logging().add_logger(fn)`` installs a *synchronous* C++
log consumer.  Both crash emitters (``carb.graphics-vulkan.plugin`` and
``gpu.foundation.plugin``) are carb-registered plugins, so every one of their
messages passes through this callback -- on the emitting thread, at emit time,
BEFORE Kit's deadlocking shutdown runs.  That makes it the earliest reliable
in-process signal available; it is a real hook, not stderr scraping.

We deliberately do NOT trigger on the slightly-earlier "GPU crash is detected.
Trying to write crash dump into: ..." message: exiting there would truncate the
``.nv-gpudmp`` / ``.nvdbg`` post-mortem dumps that Kit is in the middle of
writing.  ``ERROR_DEVICE_LOST`` lands ~300ms later, after the dumps are on disk,
and is unrecoverable by Vulkan spec -- so it is both safe and final.

THE LIFECYCLE
-------------
::

    DISABLED ──(NETT_DEVICE_LOST_GUARD=0, or carb missing)──> stays off

    IDLE ──arm()──> ARMED ──disarm()──> IDLE          (clean-shutdown path)
                      │
                      │ log callback (render thread): predicate matches
                      ▼
                  TRIGGERED
                      │  t0 ── arm kernel SIGALRM backstop @ EXIT_BUDGET_S (60s)
                      │      ── latch _triggered Event; callback RETURNS at once
                      │         (no real work on the render thread)
                      ▼
     watchdog thread wakes:
           ── start FLUSHING thread (daemon): flush + fsync artifacts
           ── join(FLUSH_BUDGET_S, default 20s)   [bounded; never blocks forever]
        t1 ── os._exit(EXIT_CODE)                 [EXITED]   (measured t1-t0 ≈ 0.1s)

    backstop: if the GIL wedges and the watchdog can NEVER run, the kernel's
              default SIGALRM action terminates the process at EXIT_BUDGET_S
              with no Python involvement at all -> exit -14/142.
              This is why the timer is armed in the CALLBACK (which already
              holds the GIL) and not in the watchdog (which must acquire it):
              verified by test -- a real GIL wedge starves the watchdog, and
              only the callback-armed timer survives it.

No signal *handler* is installed -- we rely on SIGALRM's default disposition
(terminate) -- so there is no async-signal-safety question to answer.

EXIT CODE
---------
Default 75 (``EX_TEMPFAIL``): a device-lost renderer crash is a transient
hardware/driver fault, not a bug in the run.  Critically it is NOT in
``task_runner._is_tolerated_isaac_teardown_exit``'s ``{-6, -9, -11, 134, 139}``,
so the parent raises instead of silently logging "likely Isaac teardown SIGSEGV;
outputs should still be intact" -- which would be a lie here.  The SIGALRM
backstop yields -14 / 142, also outside that set.

ENV VARS
--------
``NETT_DEVICE_LOST_GUARD``        "1" (default) | "0" to disable entirely
``NETT_DEVICE_LOST_EXIT_CODE``    "75"
``NETT_DEVICE_LOST_FLUSH_S``      "20"  -- durability + forensics work budget
``NETT_DEVICE_LOST_EXIT_BUDGET_S`` "60" -- absolute detection->dead deadline
``NETT_DEVICE_LOST_FORENSICS``    "1" (default) | "0" to skip crash collection
``NETT_VRAM_OOM_EXIT_CODE``       "76"  -- distinct from 75: an OOM is "too many envs",
                                  not a transient device crash. Only fires when the run
                                  armed ``oom_fatal`` (dry-run probes; see
                                  ``_VRAM_OOM_SIGNATURES``).

CRASH FORENSICS
---------------
On trigger, within the bounded flush window, we copy the evidence Kit already
wrote -- the NVIDIA Aftermath ``.nv-gpudmp`` / ``.nvdbg`` dumps and the Kit
session log (found by scraping their paths out of the pre-crash log lines) --
plus the pagefault address and an ``nvidia-smi`` telemetry snapshot into
``<run>/logs/crash_forensics/``.  This does NOT explain the crash; it makes a
recurrence debuggable and attributable to a specific brain/condition/GPU instead
of an unlabeled dump in Kit's shared install directory.
"""

from __future__ import annotations

import logging
import os
import threading
import time

_log = logging.getLogger(__name__)

# Narrow, confirmed-fatal predicate. (source, required substring).
# Both verified present in the real 2026-07-14 crash log and both emitted by
# carb-registered plugins, i.e. they route through ILogging.
_DEVICE_LOST_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("carb.graphics-vulkan.plugin", "ERROR_DEVICE_LOST"),
    ("gpu.foundation.plugin", "A GPU crash occurred"),
)

# Out-of-VRAM. Same hook, same bounded exit, DIFFERENT exit code -- an OOM is not a
# transient device crash, it is "this many envs do not fit", which for a sizing probe
# is an ANSWER rather than a failure.
#
# OPT-IN PER RUN (arm(oom_fatal=...)), unlike DEVICE_LOST which is always fatal. The
# evidence is narrower: a Vulkan allocation OOM is confirmed fatal HERE, during scene /
# tiled-camera build, where it cascades ("vkAllocateMemory failed" -> "Texture creation
# failed" -> "Unable to create resource from camera projection texture desc") and then
# either exits nonzero or WEDGES -- observed both, on the same GPU, depending on where
# the allocation lands. Whether a steady-state OOM mid-training is equally unrecoverable
# is NOT established, and killing a 12-hour run on a transient allocation failure would
# be far worse than the hang this avoids. So only dry-run probes arm it: a probe is
# short, disposable, and deliberately reaches for counts that cannot fit.
_VRAM_OOM_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("carb.graphics-vulkan.plugin", "ERROR_OUT_OF_DEVICE_MEMORY"),
)

# carb.logging.LEVEL_ERROR == 1 (verified: LEVEL_WARN=0, LEVEL_ERROR=1,
# LEVEL_FATAL=2). Hard-coded so the predicate does not need carb imported.
_LEVEL_ERROR = 1

_state_lock = threading.Lock()
_triggered = threading.Event()
_armed = False
#: Whether an out-of-VRAM message is a fatal trigger for THIS run (see
#: _VRAM_OOM_SIGNATURES) and whether the trigger that fired was one.
_oom_fatal = False
_triggered_by_oom = False
_logger_handle = None  # carb.logging.LoggerHandle
_watchdog: threading.Thread | None = None
_artifact_dirs: list[str] = []
_trigger_reason: str = ""

# --- crash forensics -------------------------------------------------------
# We do NOT know what CAUSES the DEVICE_LOST; this state lets us COLLECT what
# Kit already wrote (Aftermath .nv-gpudmp + .nvdbg + the pagefault address) and
# a device telemetry snapshot into the run's own logs/ so a recurrence is
# debuggable and attributable to a specific brain/condition/GPU.
_device: int | None = None
# Paths Kit announces it is writing, scraped from the log lines that PRECEDE the
# ERROR_DEVICE_LOST trigger (bounded: at most a handful per crash).
_crash_artifact_paths: list[str] = []
_pagefault_detail: str = ""

# Substrings Kit logs while writing its crash artifacts. Cheap to test on the
# render thread; the path is whatever follows "into: ".
_ARTIFACT_ANNOUNCE = "Trying to write"
_ARTIFACT_PATH_SEP = "into:"
_PAGEFAULT_MARK = "pagefault"


def _forensics_enabled() -> bool:
    return os.environ.get("NETT_DEVICE_LOST_FORENSICS", "1") != "0"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _enabled() -> bool:
    return os.environ.get("NETT_DEVICE_LOST_GUARD", "1") != "0"


def _matches(signatures, source: str, level: int, message: str) -> bool:
    if level < _LEVEL_ERROR:
        return False
    for expected_source, needle in signatures:
        if source == expected_source and needle in message:
            return True
    return False


def is_device_lost_message(source: str, level: int, message: str) -> bool:
    """The trigger predicate. Pure, side-effect free, unit-testable.

    Requires ALL of: error-or-worse level, an exact known crash-emitting plugin
    source, and that plugin's confirmed-fatal substring. An ordinary Python
    exception, a normal shutdown, or any other plugin's error can never match.
    """
    return _matches(_DEVICE_LOST_SIGNATURES, source, level, message)


def is_vram_oom_message(source: str, level: int, message: str) -> bool:
    """Same shape as :func:`is_device_lost_message`, for out-of-VRAM.

    Only a trigger when the run armed ``oom_fatal`` -- see _VRAM_OOM_SIGNATURES for
    why that is opt-in rather than always-on.
    """
    return _matches(_VRAM_OOM_SIGNATURES, source, level, message)


def vram_oom_exit_code() -> int:
    """The child's out-of-VRAM exit code. Distinct from DEVICE_LOST's so the parent
    can tell "too many envs" from "the renderer crashed" -- they mean different things
    and only one of them is a casualty."""
    return _env_int("NETT_VRAM_OOM_EXIT_CODE", 76)


def register_artifact_dir(path) -> None:
    """Add a directory whose files should be fsync'd before a forced exit."""
    with _state_lock:
        p = str(path)
        if p not in _artifact_dirs:
            _artifact_dirs.append(p)


def arm(artifact_dir=None, device: int | None = None, oom_fatal: bool = False) -> bool:
    """Install the carb log consumer + start the watchdog. Idempotent.

    Returns True if the guard is active. Safe to call before/without Kit: if
    carb is unavailable it simply returns False and the caller is unaffected.
    Call AFTER SimulationApp construction -- Kit's startup resets carb logging.

    ``device`` is the physical GPU index this run was scheduled onto; it is only
    used to label the crash-forensics telemetry snapshot (nvidia-smi ignores
    CUDA_VISIBLE_DEVICES, so a physical index is what it wants).

    ``oom_fatal`` additionally treats an out-of-VRAM message as a bounded-exit
    trigger (exit ``vram_oom_exit_code()``). For DRY-RUN PROBES only: they exist to
    find the count that does not fit, so an OOM is their answer, and without this
    they can wedge holding the whole GPU. A real run leaves it off -- see
    _VRAM_OOM_SIGNATURES.
    """
    global _armed, _logger_handle, _watchdog, _device, _oom_fatal

    if artifact_dir is not None:
        register_artifact_dir(artifact_dir)
    if device is not None:
        _device = int(device)
    # Set before the early return and outside the idempotence check: the predicate
    # must be live the instant the consumer is installed.
    _oom_fatal = bool(oom_fatal)
    if not _enabled():
        return False

    with _state_lock:
        if _armed:
            return True
        try:
            import carb.logging as carb_logging

            handle = carb_logging.acquire_logging().add_logger(_on_carb_log)
        except Exception:
            _log.debug("device-lost guard: carb logging unavailable", exc_info=True)
            return False

        _logger_handle = handle
        _armed = True

    _watchdog = threading.Thread(
        target=_watchdog_main, name="nett-device-lost-watchdog", daemon=True
    )
    _watchdog.start()
    _log.info(
        "device-lost guard armed (exit_code=%d, flush_budget=%ds, exit_budget=%ds)",
        _env_int("NETT_DEVICE_LOST_EXIT_CODE", 75),
        _env_int("NETT_DEVICE_LOST_FLUSH_S", 20),
        _env_int("NETT_DEVICE_LOST_EXIT_BUDGET_S", 60),
    )
    return True


def disarm() -> None:
    """Remove the log consumer. Call on the CLEAN shutdown path so a healthy
    run's teardown is bit-for-bit what it was before this module existed."""
    global _armed, _logger_handle, _oom_fatal

    with _state_lock:
        if not _armed:
            _oom_fatal = False
            return
        handle, _logger_handle, _armed = _logger_handle, None, False
        _oom_fatal = False

    try:
        import carb.logging as carb_logging

        carb_logging.acquire_logging().remove_logger(handle)
    except Exception:
        _log.debug("device-lost guard: remove_logger failed", exc_info=True)
    # The watchdog is a daemon blocked on an Event that is now unreachable via
    # the (removed) callback; it dies with the process and costs nothing.


def _arm_kernel_backstop() -> None:
    """Arm SIGALRM's DEFAULT disposition (terminate) as an absolute deadline.

    This is the only escape that survives a wedged GIL: it needs no Python
    bytecode to run and no signal handler, so async-signal-safety is moot.
    Yields exit -14/142, which is outside the tolerated-teardown set.

    Armed from the log callback rather than the watchdog ON PURPOSE: the
    callback is already holding the GIL (carb is calling into Python), whereas
    the watchdog must first *acquire* it. If anything wedges the GIL in that
    window the watchdog would never run -- and would never have armed the
    backstop meant to cover exactly that case. Verified: setitimer works from a
    non-main thread.
    """
    try:
        import signal

        signal.setitimer(
            signal.ITIMER_REAL, float(_env_int("NETT_DEVICE_LOST_EXIT_BUDGET_S", 60))
        )
    except Exception:
        pass


def _on_carb_log(**kwargs) -> None:
    """carb log consumer. Runs on the EMITTING thread (often the render
    thread) inside carb's log call -- so it must do essentially nothing."""
    global _trigger_reason, _triggered_by_oom
    try:
        if _triggered.is_set():
            return
        source = kwargs.get("source") or ""
        message = kwargs.get("message") or ""
        level = kwargs.get("level", -1)
        # Opportunistically capture crash-artifact paths + the pagefault address
        # from the lines Kit emits just BEFORE ERROR_DEVICE_LOST. Cheap string
        # work only; the collector (bounded, off-thread) uses these later.
        _scrape_crash_artifacts(str(message))
        if is_device_lost_message(source, int(level), str(message)):
            pass
        elif _oom_fatal and is_vram_oom_message(source, int(level), str(message)):
            _triggered_by_oom = True
        else:
            return
        _trigger_reason = f"[{source}] {str(message).strip()[:200]}"
        _arm_kernel_backstop()
        _triggered.set()  # latch; the watchdog owns everything from here
    except Exception:
        # A raising log consumer must never destabilise Kit.
        pass


def _scrape_crash_artifacts(message: str) -> None:
    """Record Aftermath dump paths + the pagefault detail as they stream by.

    Runs on the render thread, so it stays to substring tests and a split.
    Kit logs e.g. ``GPU crash is detected. Trying to write crash dump into:
    <path>.nv-gpudmp`` and ``GPU pagefault occured on virtual address(0x...)``.
    """
    global _pagefault_detail
    if not _forensics_enabled():
        return
    if _ARTIFACT_ANNOUNCE in message and _ARTIFACT_PATH_SEP in message:
        path = message.split(_ARTIFACT_PATH_SEP, 1)[1].strip()
        if path and path not in _crash_artifact_paths:
            _crash_artifact_paths.append(path)
    elif _PAGEFAULT_MARK in message and "address" in message and not _pagefault_detail:
        _pagefault_detail = message.strip()[:300]


def _watchdog_main() -> None:
    if not _triggered.wait():
        return
    t0 = time.monotonic()
    exit_code = (
        vram_oom_exit_code()
        if _triggered_by_oom
        else _env_int("NETT_DEVICE_LOST_EXIT_CODE", 75)
    )
    flush_budget = _env_int("NETT_DEVICE_LOST_FLUSH_S", 20)
    # (the SIGALRM backstop was already armed by the log callback, which is the
    # earliest point at which the GIL is guaranteed to be held.)

    _emit(
        "%s confirmed -> forcing bounded exit. reason=%s"
        % ("VRAM_OOM" if _triggered_by_oom else "DEVICE_LOST", _trigger_reason)
    )

    flusher = threading.Thread(
        target=_flush_artifacts, name="nett-device-lost-flush", daemon=True
    )
    flusher.start()
    flusher.join(timeout=float(flush_budget))
    if flusher.is_alive():
        _emit("device-lost: artifact flush exceeded %ds budget; exiting anyway"
              % flush_budget)

    _emit(
        "device-lost: os._exit(%d) at +%.2fs after detection"
        % (exit_code, time.monotonic() - t0)
    )
    try:
        os._exit(exit_code)
    finally:  # pragma: no cover - os._exit does not return
        pass


def _emit(msg: str) -> None:
    """Log to BOTH the Python logger and raw stderr. The logging stack may be
    wedged behind the same deadlock we are escaping; stderr rarely is."""
    try:
        _log.error(msg)
    except Exception:
        pass
    try:
        import sys

        sys.stderr.write("[nett.crash_guard] " + msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Artifact durability
#
# NOTE ON SCOPE -- what is actually at risk, measured, not assumed:
#
#   * eval_metrics.csv / .jsonl: SAFE without us. Written under context
#     managers, so they are closed into the OS page cache, which survives
#     os._exit (only a HOST crash could lose them).
#   * logging handlers: SAFE without us. logging.StreamHandler.emit() flushes
#     on every record. Re-flushing is free insurance, not the load-bearing bit.
#   * tfevents: THE REAL EXPOSURE. SummaryWriter buffers through a background
#     thread. Measured on a 20-scalar writer: os._exit with no flush left only
#     9/20 points on disk; flushing first recovered 20/20.
#
# The fsync pass on top is what makes the bytes durable against a host/driver
# level hard reset, which a GPU crash can escalate into. Every step is
# individually try/except'd -- durability is best-effort and must NEVER
# prevent the exit.
# ---------------------------------------------------------------------------


def flush_artifacts_now(reason: str = "") -> None:
    """Public entry to the same artifact flush the DEVICE_LOST path uses.

    Exists so :mod:`nett_skrl.runtime.stall_guard` can flush IDENTICALLY on its own
    bounded-exit path rather than growing a second, drifting copy. Deliberately a thin
    wrapper: the ordering inside (durability before forensics) is load-bearing and
    should have exactly one implementation.

    ``reason`` LABELS the forensics for a caller that is not a DEVICE_LOST. Without it a
    stall wrote ``crash_summary.json`` with ``"reason": ""`` into a directory called
    ``crash_forensics/`` for a run that never crashed -- not false, but misleading
    exactly where someone would be looking for a cause. The GPU telemetry snapshot IS
    worth keeping for a stall (it captures device state at hang time), so the fix is to
    say what happened, not to skip it.
    """
    global _trigger_reason
    if reason and not _trigger_reason:
        _trigger_reason = reason
    _flush_artifacts()


def _flush_artifacts() -> None:
    # Order matters: the durability flushes (tfevents is the load-bearing one)
    # run FIRST, so if forensics collection is slow and the flush budget expires,
    # the critical training data is already saved. Forensics is written before
    # the final fsync pass so it, too, becomes durable.
    for step in (
        _flush_std,
        _flush_tensorboard,
        _flush_logging,
        _collect_forensics,
        _fsync_artifact_dirs,
    ):
        try:
            step()
        except Exception:
            _emit("device-lost: %s failed (continuing)" % step.__name__)


# ---------------------------------------------------------------------------
# Crash forensics
#
# We cannot say what CAUSED the DEVICE_LOST, but Kit + the driver already wrote
# rich evidence: NVIDIA Aftermath dumps (.nv-gpudmp / .nvdbg) and the pagefault
# virtual address. The problem is that evidence lands in Kit's SHARED install-dir
# log folder, wall-clock-timestamped and unattributed -- after an 8-way wave you
# cannot tell which run produced which dump. This step copies that evidence into
# THIS run's own logs/ (so it is attributable to a brain/condition/GPU) and adds
# a device telemetry snapshot (ECC/temperature/throttle) so a recurrence can be
# told apart as hardware-transient vs application bug. Best-effort, bounded,
# crash-path only; NEVER runs on a successful shutdown.
# ---------------------------------------------------------------------------


def _collect_forensics() -> None:
    if not _forensics_enabled():
        return
    with _state_lock:
        dirs = list(_artifact_dirs)
    if not dirs:
        return
    import json as _json
    import shutil
    import sys

    out = os.path.join(dirs[0], "logs", "crash_forensics")
    try:
        os.makedirs(out, exist_ok=True)
    except OSError:
        _emit("device-lost: could not create %s (skipping forensics)" % out)
        return

    # 1. Structured summary -- always written, even if the copies below fail.
    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "device": _device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "reason": _trigger_reason,
        "pagefault_detail": _pagefault_detail,
        "aftermath_dumps": list(_crash_artifact_paths),
    }
    try:
        with open(os.path.join(out, "crash_summary.json"), "w") as fh:
            _json.dump(summary, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError:
        _emit("device-lost: failed to write crash_summary.json")

    # 2. Copy the Aftermath dumps Kit announced, plus the Kit log they belong to,
    #    out of the shared install dir and into this run.
    for src in list(_crash_artifact_paths) + _derive_kit_logs(_crash_artifact_paths):
        try:
            if src and os.path.isfile(src):
                shutil.copy2(src, os.path.join(out, os.path.basename(src)))
        except OSError:
            _emit("device-lost: failed to copy %s" % src)

    # 3. Device telemetry snapshot (ECC errors, temperature, throttle reasons).
    #    nvidia-smi ignores CUDA_VISIBLE_DEVICES; the full -q always includes the
    #    crashing GPU and crash_summary.json records which index to read.
    _snapshot_nvidia_smi(out, sys)


def _derive_kit_logs(dump_paths: list[str]) -> list[str]:
    """Map ``.../kit_<ts>-<...>.nv-gpudmp`` back to ``.../kit_<ts>.log``.

    The dump filenames embed the Kit session's ``kit_<timestamp>`` prefix, so the
    session log that holds the full crash context sits right beside them.
    """
    logs: list[str] = []
    for p in dump_paths:
        base = os.path.basename(p)
        if not base.startswith("kit_"):
            continue
        # kit_20260714_080048-0.nv-gpudmp -> kit_20260714_080048
        prefix = base.split("-", 1)[0] if "-" in base else os.path.splitext(base)[0]
        candidate = os.path.join(os.path.dirname(p), prefix + ".log")
        if candidate not in logs:
            logs.append(candidate)
    return logs


def _snapshot_nvidia_smi(out: str, sys) -> None:
    import subprocess

    jobs = (
        ("nvidia-smi_full.txt", ["nvidia-smi", "-q"]),
        (
            "nvidia-smi_query.csv",
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,"
                "memory.used,memory.total,ecc.errors.uncorrected.volatile.total,"
                "clocks_throttle_reasons.active,power.draw",
                "--format=csv",
            ],
        ),
    )
    for name, argv in jobs:
        try:
            res = subprocess.run(
                argv, capture_output=True, text=True, timeout=8, check=False
            )
            with open(os.path.join(out, name), "w") as fh:
                fh.write(res.stdout or "")
                if res.stderr:
                    fh.write("\n--- stderr ---\n" + res.stderr)
        except (OSError, subprocess.SubprocessError):
            _emit("device-lost: nvidia-smi snapshot (%s) unavailable" % name)


def _flush_std() -> None:
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass


def _flush_tensorboard() -> None:
    """Flush every live tfevents writer without importing/coupling to skrl.

    SummaryWriter buffers through a background thread; .flush() is what pushes
    the queue to the fd. Found by object graph so this works for skrl's
    agent.writer, the recording writer, or any future one.
    """
    import gc
    import warnings

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return
    # Walking every live object touches lazy/deprecated module attributes and
    # emits unrelated warnings; they would be pure noise in a crash log.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                if isinstance(obj, SummaryWriter):
                    obj.flush()
            except Exception:
                continue


def _flush_logging() -> None:
    for handler in list(logging.root.handlers) + [
        h
        for lg in logging.Logger.manager.loggerDict.values()
        if isinstance(lg, logging.Logger)
        for h in list(lg.handlers)
    ]:
        try:
            handler.flush()
        except Exception:
            continue
        _fsync_stream(getattr(handler, "stream", None))


def _fsync_stream(stream) -> None:
    try:
        if stream is None or stream.closed:
            return
        fd = stream.fileno()
    except Exception:
        return
    try:
        os.fsync(fd)
    except (OSError, ValueError):
        pass  # not a real file (pipe/tty/socket) -- nothing to make durable


def _fsync_artifact_dirs() -> None:
    """fsync the on-disk artifacts summary.json is later derived from
    (logs/eval_metrics.csv, eval_metrics.jsonl, tfevents, checkpoints) plus the
    directory entries themselves, so the files' *existence* is durable too.

    The child never writes summary.json -- analysis/api.py:analyze() builds it
    post-hoc from exactly these files.
    """
    for root in list(_artifact_dirs):
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                _fsync_path(os.path.join(dirpath, name), directory=False)
            _fsync_path(dirpath, directory=True)


def _fsync_path(path: str, *, directory: bool) -> None:
    fd = None
    try:
        fd = os.open(path, os.O_RDONLY | (os.O_DIRECTORY if directory else 0))
        os.fsync(fd)
    except (OSError, ValueError):
        pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
