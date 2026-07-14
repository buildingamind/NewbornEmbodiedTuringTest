"""Bounded hard-exit guard for renderer GPU crashes (VkResult ERROR_DEVICE_LOST).

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
``NETT_DEVICE_LOST_FLUSH_S``      "20"  -- durability work budget
``NETT_DEVICE_LOST_EXIT_BUDGET_S`` "60" -- absolute detection->dead deadline
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

# carb.logging.LEVEL_ERROR == 1 (verified: LEVEL_WARN=0, LEVEL_ERROR=1,
# LEVEL_FATAL=2). Hard-coded so the predicate does not need carb imported.
_LEVEL_ERROR = 1

_state_lock = threading.Lock()
_triggered = threading.Event()
_armed = False
_logger_handle = None  # carb.logging.LoggerHandle
_watchdog: threading.Thread | None = None
_artifact_dirs: list[str] = []
_trigger_reason: str = ""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _enabled() -> bool:
    return os.environ.get("NETT_DEVICE_LOST_GUARD", "1") != "0"


def is_device_lost_message(source: str, level: int, message: str) -> bool:
    """The trigger predicate. Pure, side-effect free, unit-testable.

    Requires ALL of: error-or-worse level, an exact known crash-emitting plugin
    source, and that plugin's confirmed-fatal substring. An ordinary Python
    exception, a normal shutdown, or any other plugin's error can never match.
    """
    if level < _LEVEL_ERROR:
        return False
    for expected_source, needle in _DEVICE_LOST_SIGNATURES:
        if source == expected_source and needle in message:
            return True
    return False


def register_artifact_dir(path) -> None:
    """Add a directory whose files should be fsync'd before a forced exit."""
    with _state_lock:
        p = str(path)
        if p not in _artifact_dirs:
            _artifact_dirs.append(p)


def arm(artifact_dir=None) -> bool:
    """Install the carb log consumer + start the watchdog. Idempotent.

    Returns True if the guard is active. Safe to call before/without Kit: if
    carb is unavailable it simply returns False and the caller is unaffected.
    Call AFTER SimulationApp construction -- Kit's startup resets carb logging.
    """
    global _armed, _logger_handle, _watchdog

    if artifact_dir is not None:
        register_artifact_dir(artifact_dir)
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
    global _armed, _logger_handle

    with _state_lock:
        if not _armed:
            return
        handle, _logger_handle, _armed = _logger_handle, None, False

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
    global _trigger_reason
    try:
        if _triggered.is_set():
            return
        source = kwargs.get("source") or ""
        message = kwargs.get("message") or ""
        level = kwargs.get("level", -1)
        if not is_device_lost_message(source, int(level), str(message)):
            return
        _trigger_reason = f"[{source}] {str(message).strip()[:200]}"
        _arm_kernel_backstop()
        _triggered.set()  # latch; the watchdog owns everything from here
    except Exception:
        # A raising log consumer must never destabilise Kit.
        pass


def _watchdog_main() -> None:
    if not _triggered.wait():
        return
    t0 = time.monotonic()
    exit_code = _env_int("NETT_DEVICE_LOST_EXIT_CODE", 75)
    flush_budget = _env_int("NETT_DEVICE_LOST_FLUSH_S", 20)
    # (the SIGALRM backstop was already armed by the log callback, which is the
    # earliest point at which the GIL is guaranteed to be held.)

    _emit(
        "DEVICE_LOST confirmed -> forcing bounded exit. reason=%s" % _trigger_reason
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


def _flush_artifacts() -> None:
    for step in (_flush_std, _flush_tensorboard, _flush_logging, _fsync_artifact_dirs):
        try:
            step()
        except Exception:
            _emit("device-lost: %s failed (continuing)" % step.__name__)


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
