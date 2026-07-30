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
``NETT_STALL_STARTUP_GRACE_S``  "900"  before the FIRST step ever completes: Kit boot +
                                      scene build + first render, which is minutes under
                                      8-way contention.
``NETT_STALL_EXIT_BUDGET_S``    "60"   SIGALRM backstop, for a wedged GIL.
``NETT_STALL_EXIT_CODE``        "77"   distinct from 75 (DEVICE_LOST) / 76 (VRAM OOM).
``NETT_STALL_GUARD``            "1"    DEFAULT ON. Set 0 to disable.

10 minutes still turns a 4.5h dead slot into ~10min, and the wave continues with the
cell recorded as a casualty (``StallError`` is a ``ReapedTaskError``).

⚠ ONE ALARM, TWO GUARDS. Both this module and ``crash_guard`` arm ``ITIMER_REAL`` as a
last-resort backstop, and a process has only one. They only ever arm it on their own
trigger, and either arming leads to the same outcome (bounded termination outside the
tolerated-teardown set), so a race between them is harmless -- but do not add a third
consumer of that timer without revisiting this.
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
        _env_int("NETT_STALL_STARTUP_GRACE_S", 900),
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


def _deadline() -> float:
    if _seen_progress:
        return float(_env_int("NETT_STALL_TIMEOUT_S", 600))
    return float(_env_int("NETT_STALL_STARTUP_GRACE_S", 900))


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
