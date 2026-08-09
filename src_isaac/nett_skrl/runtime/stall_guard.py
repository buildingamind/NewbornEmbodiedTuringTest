"""Bounded hard-exit guard for a Kit renderer that stops making progress.

WHY THIS EXISTS
---------------
Measured 2026-07-30, reproducibly, on an idle 8-GPU host at one cell per GPU: **2-4 of
every 8 training cells wedge and never recover**. Two ``faulthandler`` samples 25s apart
from a stalled cell are byte-identical::

    skrl/trainers/torch/base.py:216 in train
      nett_skrl/body/skrl_adapter.py:66 in step
        nett_skrl/body/wrappers/channels_first.py:69 in step
          isaaclab/envs/direct_rl_env.py:382 in step
            isaaclab/sim/simulation_context.py:626 in render   <- self._app.update()

...with the **GPU at 2% utilization while the process burns 136% CPU** and 283 of ~300
threads in ``futex_wait``. Kit spins inside its own update pump and never returns.

WHY ``crash_guard`` DOES NOT COVER IT
-------------------------------------
``crash_guard`` triggers on a carb log signature (``VkResult: ERROR_DEVICE_LOST`` /
"A GPU crash occurred"). This hang emits **no DEVICE_LOST and no crash signature at
all** -- the renderer does not fail, it stops returning. So nothing fires, and because
``join_with_reap`` passes ``absolute_timeout=None`` for real runs and
``NETT_REAP_TIMEOUT`` defaults to 0, the parent's ``p.join()`` is unbounded. Before this
module the shell ``timeout`` was the only bound, at whole-run granularity: one hang cost
a 4.5h per-cell budget in the actor A/B wave.

Ruled out as causes, each by a control rather than by argument: the reward
``torch.compile`` (``NETT_REWARD_COMPILE=0``, no compile pool -- still stalls), the BC7
frame cache, and the shared OptiX shader cache (per-cell ``OPTIX_CACHE_PATH`` -- still
stalls). DLSS cannot be disabled at all (see ``nett_env_cfg`` on ``dlss_mode``). So the
hang is inside Kit and this guard BOUNDS it rather than fixing it.

THE SIGNAL
----------
Env-step progress, notified from :func:`note_progress` in the skrl env adapter -- the
one place every training and eval step passes through. The watchdog never inspects Kit;
it only asks "has a step completed recently". That makes it agnostic to the underlying
cause, which matters because the cause is upstream and unfixed here.

A Python watchdog thread CAN run during this hang: the stalled dump shows the tqdm
monitor, the tensorboard writer and crash_guard's own watchdog all alive and scheduled,
so ``self._app.update()`` releases the GIL. The SIGALRM backstop covers the case where
it does not.

BUDGETS -- and why they are generous
------------------------------------
The gap between consecutive env steps is NOT uniform. A PPO update runs between
rollouts (no env steps for its duration), and checkpointing and artifact writes also
pause stepping. A budget tight enough to catch a hang "quickly" would kill healthy runs
mid-update, which is far worse than a slow detect. So:

``NETT_STALL_TIMEOUT_S``        "600"  steady-state: no step completed for 10 minutes.
                                      Observed healthy cadence is ~19 steps/s, and the
                                      longest legitimate pause is a PPO update, orders
                                      of magnitude under this.
``NETT_STALL_STARTUP_GRACE_S``  "300"  before the FIRST step ever completes: Kit boot +
                                      scene build + first render. MEASURED worst case at
                                      24-way concurrency is kit_up 57.7s median / 59.6s
                                      max, and a declared-memory cell reaches its first
                                      step in ~1 min -- so 300s is ~5x the observed worst
                                      case. Was 900s, i.e. 10-15x, which meant an
                                      OOM-WEDGED cell (never progresses, ignores SIGTERM)
                                      held its GPU for a needless 15 minutes.
                                      ⚠ Raise it for a COLD texture cache or a very wide
                                      wave, where first render can legitimately take
                                      minutes longer. The VRAM dry-run probes do NOT rely
                                      on this -- they are bounded separately by
                                      TaskConfig.dry_run_timeout.
``NETT_STALL_EXIT_BUDGET_S``    "60"   SIGALRM backstop, for a wedged GIL.
``NETT_STALL_EXIT_CODE``        "77"   distinct from 75 (DEVICE_LOST) / 76 (VRAM OOM).
``NETT_STALL_GUARD``            "1"    DEFAULT ON. Set 0 to disable.

10 minutes still turns a 4.5h dead slot into ~10min, and the wave continues with the
cell recorded as a casualty (``StallError`` is a ``ReapedTaskError``).

⚠ ONE ALARM, THREE CONSUMERS -- REVISITED 2026-08-09. This module and ``crash_guard``
both arm ``ITIMER_REAL`` on their own trigger, and :func:`arm_teardown_backstop` now
arms it on ENTRY to teardown. A process has only one such timer. Every arming leads to
the same outcome -- bounded termination outside the tolerated-teardown set -- and the
windows barely overlap (the teardown one opens only once stepping has ended, which is
also when this guard disarms), so a race is still harmless. The cost of the overlap that
does exist is attribution, not safety: a ``-14``/``142`` remains ambiguous and keeps its
conservative DEVICE_LOST reading. Do not add a FOURTH consumer without redoing this
paragraph.

THE POST-STEPPING WINDOW, AND WHY THE WATCHDOG CANNOT COVER IT
---------------------------------------------------------------
:func:`disarm` is called the moment stepping ends, so everything after it -- artifact
flush, ``wandb.finish``, non-daemon thread joins, interpreter finalization -- was
UNGUARDED. That is where the "multi-brain run hangs after test" specimens live. Two
structural reasons the ordinary watchdog could never have covered it even if left armed:

  * it is a DAEMON thread, and CPython stops scheduling daemon threads once
    ``Py_FinalizeEx`` sets the finalizing flag; and
  * its SIGALRM backstop is armed INSIDE ``_trigger``, i.e. only after a detection that
    by then cannot happen.

⚠ SO DO NOT "FIX" THIS BY RAISING ``NETT_STALL_TIMEOUT_S``. A longer grace cannot make a
stopped thread run. :func:`arm_teardown_backstop` arms the kernel timer on ENTRY to the
window instead, which is the only thing that survives both problems.
"""

from __future__ import annotations

import logging
import os
import threading
import time

_log = logging.getLogger("nett.stall_guard")

_state_lock = threading.Lock()
_armed = False
_watchdog: threading.Thread | None = None
_stop = threading.Event()

# Progress state. A plain int + float under the lock: cheaper than a Queue on a path
# that runs every env step, and the watchdog only needs the latest value.
_steps = 0
_last_progress_monotonic = 0.0
_seen_progress = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _enabled() -> bool:
    return os.environ.get("NETT_STALL_GUARD", "1") != "0"


def note_progress(n: int = 1) -> None:
    """Record that *n* env steps completed. Called on the hot path -- keep it trivial.

    Safe and ~free when the guard is disarmed: it still takes the lock and bumps two
    numbers, which is nothing against a step that renders a scene. Deliberately does
    NOT check ``_enabled()`` -- reading an env var per step would cost more than the
    work it guards.
    """
    global _steps, _last_progress_monotonic, _seen_progress
    with _state_lock:
        _steps += n
        _last_progress_monotonic = time.monotonic()
        _seen_progress = True


def steps_seen() -> int:
    with _state_lock:
        return _steps


def arm() -> bool:
    """Start the watchdog. Idempotent. Returns True if the guard is active.

    Call once per mode subprocess, next to ``crash_guard.arm``. Needs no Kit and no
    carb: the signal is our own step counter, so this works before Kit is up and in
    tests.
    """
    global _armed, _watchdog, _last_progress_monotonic, _seen_progress, _steps

    if not _enabled():
        return False

    with _state_lock:
        if _armed:
            return True
        # Reset here, not at import: a spawn child inherits module state through the
        # pickled module only by accident, and a re-arm in the same process (tests)
        # must start from a clean clock rather than an ancient last-progress stamp.
        _steps = 0
        _seen_progress = False
        _last_progress_monotonic = time.monotonic()
        _armed = True
        _stop.clear()

    _watchdog = threading.Thread(
        target=_watchdog_main, name="nett-stall-watchdog", daemon=True
    )
    _watchdog.start()
    _log.info(
        "stall guard armed (timeout=%ds, startup_grace=%ds, exit_code=%d)",
        _env_int("NETT_STALL_TIMEOUT_S", 600),
        _env_int("NETT_STALL_STARTUP_GRACE_S", 300),
        _env_int("NETT_STALL_EXIT_CODE", 77),
    )
    return True


def disarm() -> None:
    """Stop the watchdog. Call on the CLEAN shutdown path.

    Without this, a healthy run that finishes stepping and then spends a long time in
    analysis/teardown would look stalled to a still-running watchdog.
    """
    global _armed
    with _state_lock:
        if not _armed:
            return
        _armed = False
    _stop.set()


def teardown_exit_code() -> int:
    """Exit code for "the mode finished, then teardown wedged" (78 by default).

    Distinct from 75 (DEVICE_LOST), 76 (VRAM OOM) and 77 (mid-run stall) so the four
    causes stay separable. It means something materially different from all three: the
    work COMPLETED and its artifacts were already flushed, and only Kit's shutdown hung.
    """
    return _env_int("NETT_TEARDOWN_EXIT_CODE", 78)


_teardown_backstop_armed = False


def arm_teardown_backstop(budget: float | None = None) -> bool:
    """Bound the post-stepping window. Call ON ENTRY to teardown, not on detection.

    Two nets, because neither alone is sufficient here:

    * A watchdog thread that hard-exits :func:`teardown_exit_code`. Preferred, because
      it names the cause. It works in the MEASURED signature of this hang -- the process
      is ``R`` at ~130% CPU, i.e. wedged in native Kit code that releases the GIL, where
      other Python threads still run. It does NOT work once CPython stops scheduling
      daemon threads during finalization.
    * ``ITIMER_REAL`` with its DEFAULT disposition. The kernel cannot be blocked by a
      wedged interpreter, a stopped daemon thread, or a main thread stuck inside a native
      call -- and note that a Python SIGALRM HANDLER would be useless here for that last
      reason, since CPython only runs handlers on the main thread between bytecodes.
      Costs attribution: it yields -14/142, which the parent reads conservatively as
      DEVICE_LOST. That is why it is the SECOND deadline, not the first.

    ⚠ Generous by default (``NETT_TEARDOWN_BUDGET_S`` = 300s). A legitimate teardown here
    is seconds -- ``_finalize_env_artifacts`` has already run by this point -- but
    ``wandb.finish`` can upload for a while, and killing a run whose work is DONE is the
    worse error. Idempotent; a second call is a no-op.
    """
    global _teardown_backstop_armed
    if not _enabled():
        return False
    with _state_lock:
        if _teardown_backstop_armed:
            return True
        _teardown_backstop_armed = True

    seconds = float(_env_int("NETT_TEARDOWN_BUDGET_S", 300)) if budget is None \
        else float(budget)
    kernel_grace = float(_env_int("NETT_TEARDOWN_KERNEL_GRACE_S", 60))

    def _fire() -> None:
        time.sleep(seconds)
        _log.error(
            "TEARDOWN GUARD: still shutting down %.0fs after the mode completed -- "
            "hard-exiting %d. The work itself FINISHED and its artifacts were flushed "
            "before this window opened; this is Kit's shutdown wedging (the specimen "
            "signature is R at ~130%% CPU with no output). The parent will reap so the "
            "GPU is released.",
            seconds, teardown_exit_code(),
        )
        os._exit(teardown_exit_code())

    threading.Thread(target=_fire, name="nett-teardown-backstop", daemon=True).start()

    try:
        import signal

        signal.setitimer(signal.ITIMER_REAL, seconds + kernel_grace)
    except Exception:  # noqa: BLE001 - a missing safety net must not break a run
        _log.debug("teardown backstop: setitimer unavailable", exc_info=True)
    _log.info(
        "teardown backstop armed (thread %.0fs -> exit %d, kernel %.0fs -> SIGALRM)",
        seconds, teardown_exit_code(), seconds + kernel_grace,
    )
    return True


def _deadline() -> float:
    if _seen_progress:
        return float(_env_int("NETT_STALL_TIMEOUT_S", 600))
    return float(_env_int("NETT_STALL_STARTUP_GRACE_S", 300))


def _arm_kernel_backstop() -> None:
    """Arm ITIMER_REAL's DEFAULT disposition (terminate) as an absolute deadline.

    The only escape that survives a wedged GIL: no Python bytecode, no handler, so
    async-signal-safety is moot. Yields -14/142, outside the tolerated-teardown set.
    Works from a non-main thread (``setitimer``, unlike ``signal.signal``).
    """
    try:
        import signal

        signal.setitimer(
            signal.ITIMER_REAL, float(_env_int("NETT_STALL_EXIT_BUDGET_S", 60))
        )
    except Exception:
        pass


def _watchdog_main() -> None:
    poll = max(1.0, float(_env_int("NETT_STALL_POLL_S", 15)))
    while not _stop.wait(poll):
        with _state_lock:
            if not _armed:
                return
            idle = time.monotonic() - _last_progress_monotonic
            steps = _steps
            first = not _seen_progress
        if idle < _deadline():
            continue
        _trigger(idle, steps, first)
        return


def _trigger(idle: float, steps: int, before_first_step: bool) -> None:
    """Flush what we can, then hard-exit. Bounded at every stage."""
    # Arm the kernel backstop FIRST: everything after this point wants the GIL, and the
    # whole point is that we cannot assume we will get it.
    _arm_kernel_backstop()
    _log.error(
        "STALL GUARD: no env-step progress for %.0fs (%s, %d steps seen) -- "
        "hard-exiting %d so the parent's join returns and the reap frees the GPU. "
        "This is the Kit render-pump wedge (SimulationContext.render -> "
        "_app.update); it emits no DEVICE_LOST, so crash_guard cannot see it.",
        idle,
        "before the first step ever completed" if before_first_step else "mid-run",
        steps,
        _env_int("NETT_STALL_EXIT_CODE", 77),
    )
    try:
        # Reuse crash_guard's flush verbatim so a stalled run's artifacts land exactly
        # like a crashed run's (tfevents are the real exposure). Bounded internally.
        from . import crash_guard

        crash_guard.flush_artifacts_now(
            reason=f"stall: no env-step progress for {idle:.0f}s "
                   f"({steps} steps seen) -- Kit render-pump wedge, NOT a crash"
        )
    except Exception:  # noqa: BLE001 - never let cleanup stop the exit
        _log.debug("stall guard: artifact flush failed", exc_info=True)
    os._exit(_env_int("NETT_STALL_EXIT_CODE", 77))
