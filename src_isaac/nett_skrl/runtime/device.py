"""One source of truth for "which index does torch mean?".

★ THE PROBLEM THIS EXISTS TO PREVENT. Two GPU numbering schemes run through this codebase
and they are NOT interchangeable:

* **PHYSICAL (nvml)** — what pynvml and ``nvidia-smi`` report. ``nett.py``
  ``set_device(most_free_gpu)`` assigns tasks this way, and ``TaskReaper``,
  ``crash_guard.arm`` and ``MemoryManager.get_free_memory`` all consume it. ⚠ nvml
  **IGNORES** ``CUDA_VISIBLE_DEVICES`` (``crash_guard`` says so in as many words), so a
  physical index stays valid no matter how visibility is restricted.
* **TORCH / KIT** — what ``cuda:N`` means inside a process, which DOES honour
  ``CUDA_VISIBLE_DEVICES``. Under ``CUDA_VISIBLE_DEVICES=3`` the only valid device is
  ``cuda:0``.

Getting this wrong fails in two ways, and the quiet one is much worse:

* physical index with the pin        -> ``CUDA error: invalid device ordinal`` (loud)
* non-zero index WITHOUT the pin     -> Kit's usdrt scenegraph supports only ``cuda:0``,
  so the run **HANGS at the Fabric XFormPrimView with no error at all** (26-71 min
  observed, indefinite in principle).

⚠ HOST-STATE DEPENDENT: ``most_free_gpu`` only leaves GPU0 when GPU0 is the busier card,
so a codebase with this bug runs correctly for months and then wedges the day a colleague
starts a job on GPU0. That is exactly how it was found (2026-07-28).

WHY A MODULE AND NOT A LOCAL HELPER: the first fix reconciled ONE call site
(``AppLauncher``) and the run still failed, because ``cfg.sim.device`` (PhysX), the policy
device and the retina wrapper each built their own ``cuda:{index}`` string. Four call
sites, one rule -- so the rule lives in one place and every consumer imports it.
"""

from __future__ import annotations

import contextlib
import os


def visible_device_count() -> int | None:
    """How many GPUs this process can see, or None when unrestricted."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return None
    return len([e for e in visible.split(",") if e.strip() != ""])


def torch_device_index(configured: int | None) -> int:
    """Translate a PHYSICAL device index into the one ``torch``/Kit use here.

    Returns 0 whenever exactly one GPU is visible -- that card IS ``cuda:0`` to this
    process, whatever nvml calls it. Otherwise the configured index passes through
    unchanged, so unpinned callers keep their existing behaviour and this is additive.
    """
    if visible_device_count() == 1:
        return 0
    return int(configured or 0)


def torch_device_str(configured: int | None) -> str:
    """``"cuda:N"`` using the index this process actually understands."""
    return f"cuda:{torch_device_index(configured)}"


@contextlib.contextmanager
def visible_device_scope(device: int | None):
    """Expose ONLY ``device`` to a child process, so it indexes that GPU as ``cuda:0``.

    ⚠ EVERY process that boots Kit must be started inside this scope. Kit's usdrt
    scenegraph supports only ``cuda:0``; handed a physical index it errors
    ("GPU 3 requested. GPUs other than cuda:0 are not currently supported") and the run
    HANGS with no traceback. This has now been missed twice -- once for the mode
    subprocesses, once for the validation subprocess added later -- so the scope lives
    HERE, next to the index rule, rather than beside any single caller.

    ``spawn`` snapshots ``os.environ`` at ``start()``, so wrap the ``start()`` call and the
    parent's environment is restored immediately after. Safe because each task owns its
    pool worker; this is not the shared-parent race ``optuna_tune`` warns about.
    """
    if device is None:
        yield
        return
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(device))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
