"""Kernel-enforced "die with my parent" for task subprocesses.

WHY THIS EXISTS
---------------
``reap.py`` cleans up task-owned processes, but every ``reaper.reap()`` call site is on a
path the PARENT CHOOSES: the VRAM-OOM exit, the stall exit, the device-lost exit, and
``join_with_reap``'s absolute timeout. There is no SIGTERM handler in the parent. So when
the parent is killed from OUTSIDE -- the shell ``timeout`` that bounds every wave, a
``pkill``, an operator cleaning up -- it dies without reaping, and its
``multiprocessing.spawn`` child (which is the Isaac process, holding GPU memory)
reparents to PPID=1 and runs forever.

MEASURED 2026-07-30: **29 orphaned spawn processes at PPID=1, up to 8 hours old, holding
52 GB of RAM**, plus three separate Kit orphans holding ~31 GB of VRAM across two GPUs --
all produced by exactly this path. A wedged orphan is not merely idle: because admission
control reads *physical* free VRAM, it shrinks the headroom every later run is measured
against, and one caused a ``JobTooBigError`` on a GPU that was otherwise empty.

WHY NOT A SIGTERM HANDLER IN THE PARENT
---------------------------------------
A handler cannot run if the parent is SIGKILLed, and cannot run reliably if the parent is
itself wedged -- both of which happen here. ``PR_SET_PDEATHSIG`` is enforced by the
KERNEL at parent-death, so it covers the -9 case a handler structurally cannot.

TWO SUBTLETIES, both handled below:
  * PDEATHSIG fires when the parent **thread** that created us exits, not the process.
    CPython's multiprocessing starts children from the main thread, so this is the
    behaviour we want -- but do not move the ``Process.start()`` call onto a worker
    thread without revisiting this.
  * There is a RACE: if the parent dies between its fork and our prctl call, the signal
    has already been missed and we would linger forever. So re-check ``getppid()``
    immediately after arming and exit if we have already been reparented.

The setting is cleared across ``exec``, so it MUST be armed inside the child after the
spawn, not inherited from the parent.
"""

from __future__ import annotations

import ctypes
import logging
import os
import signal
import sys

_log = logging.getLogger("nett.pdeathsig")

#: linux/prctl.h
PR_SET_PDEATHSIG = 1


def arm(sig: int = signal.SIGTERM) -> bool:
    """Ask the kernel to send *sig* to this process when its parent dies.

    Returns True if armed (and the parent was still alive). Safe to call anywhere: on a
    non-Linux platform, or if libc/prctl is unavailable, it logs at debug and returns
    False rather than raising -- a missing safety net must never break a run.

    If we have ALREADY been reparented (the race above), this exits the process
    immediately with 0: there is no parent left to do work for, and lingering is the
    exact failure this module exists to prevent.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        rc = libc.prctl(PR_SET_PDEATHSIG, ctypes.c_ulong(sig), 0, 0, 0)
        if rc != 0:
            _log.debug("prctl(PR_SET_PDEATHSIG) failed: errno=%d", ctypes.get_errno())
            return False
    except Exception:  # noqa: BLE001 - a missing safety net must not break the run
        _log.debug("PDEATHSIG unavailable", exc_info=True)
        return False

    # Race guard: the parent may have died between fork and prctl, in which case the
    # signal was already missed and nothing else will ever tell us to stop.
    if os.getppid() == 1:
        _log.warning(
            "PDEATHSIG armed but parent is already gone (PPID=1) -- exiting rather "
            "than lingering as an orphan holding GPU memory."
        )
        os._exit(0)
    return True
