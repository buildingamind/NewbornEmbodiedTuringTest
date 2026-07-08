"""Init-time PhysX placement strategy for parallelized NETT runs.

The wheeled chick is a replicated articulation (body + 4 driven wheels + head +
camera harness) stepped by the PhysX solver. To maximize throughput on highly
parallelized runs we want that solver on the **GPU** (``sim.device = cuda:N``)
so every env steps in one batched kernel instead of round-tripping per-env state
to the CPU. That is the default and the fast path.

But GPU PhysX physics buffers share VRAM with the tiled-camera renderer. At very
high ``num_envs`` the physics state can crowd out render memory and OOM the run
at the first reset. The goal explicitly permits offloading work to CPU memory
**only when doing so increases the parallelism that actually fits** — i.e. when
freeing the GPU-PhysX VRAM lets more rendered envs co-reside, *and* the host has
enough CPU threads to run the articulation in parallel without becoming the new
bottleneck. This module makes that decision once, at initialization, from the
live VRAM headroom and CPU thread count.

Hard constraints baked in (do not regress):
  * **Kinematic locomotion MUST use CPU PhysX.** GPU PhysX (GpuArticulationView /
    GpuRigidBodyView) core-dumps with an illegal memory access on the
    kinematic-only NETT scene at the first reset. Kinematic is never placed on
    GPU regardless of VRAM.
  * **``NETT_SIM_DEVICE`` is an absolute override.** If set, it wins over every
    heuristic (used by probes and apples-to-apples A/B compares).

The estimation constants are deliberately conservative heuristics (a small
articulation costs little VRAM) and are env-overridable so they can be tuned
against a real Isaac run without touching code. The *decision logic* — not the
exact byte counts — is the contract this module guarantees.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

_MIB = 1024 * 1024

# A valid PhysX device is "cpu", "cuda", or "cuda:<index>" — nothing else.
_DEVICE_RE = re.compile(r"^(cpu|cuda(:\d+)?)$")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- Heuristic constants (override via env for empirical tuning) --------------
# Fixed GPU-PhysX context/solver overhead, independent of env count.
GPU_PHYSX_BASE_MIB = _env_float("NETT_PHYSX_BASE_MIB", 512.0)
# Per-env articulation/rigid-body state on the GPU. The wheeled chick is a small
# articulation (~7 links), so this is modest.
GPU_PHYSX_PER_ENV_MIB = _env_float("NETT_PHYSX_PER_ENV_MIB", 8.0)
# Keep this much VRAM (×) headroom above the raw physics estimate before we call
# GPU PhysX "safe" — leaves room for the renderer and allocator fragmentation.
VRAM_SAFETY_FACTOR = _env_float("NETT_PHYSX_VRAM_SAFETY", 1.5)
# CPU PhysX only becomes a worthwhile offload target when the host can actually
# run the articulation across enough threads to keep up with the GPU renderer.
MIN_CPU_THREADS_FOR_OFFLOAD = _env_int("NETT_PHYSX_MIN_CPU_THREADS", 8)


@dataclass(frozen=True)
class PhysxStrategy:
    """The chosen PhysX device plus a human-readable rationale for logging."""

    device: str
    rationale: str
    #: True when the run is VRAM-tight but we could not offload (logged as a warning).
    vram_warning: bool = False

    @property
    def on_gpu(self) -> bool:
        return self.device.startswith("cuda")


def estimate_gpu_physx_mib(num_envs: int) -> float:
    """Conservative VRAM (MiB) the GPU-PhysX physics state needs for ``num_envs``."""
    return GPU_PHYSX_BASE_MIB + max(0, num_envs) * GPU_PHYSX_PER_ENV_MIB


def select_physx_strategy(
    *,
    locomotion: str,
    render_device_index: int,
    num_envs: int,
    free_vram_bytes: Optional[int],
    cpu_threads: int,
    override: Optional[str] = None,
) -> PhysxStrategy:
    """Decide ``sim.device`` (PhysX placement) at initialization.

    Args:
        locomotion: ``"wheeled"`` (articulation, GPU-friendly) or ``"kinematic"``
            (must stay on CPU PhysX).
        render_device_index: GPU index the renderer/AppLauncher use (``cuda:N``).
        num_envs: Parallel env count for this run.
        free_vram_bytes: Free VRAM on the render GPU, or ``None`` if unknown
            (e.g. probing failed) — then we cannot reason about VRAM and keep the
            locomotion default.
        cpu_threads: Usable CPU threads (``len(os.sched_getaffinity(0))``).
        override: ``NETT_SIM_DEVICE`` value; absolute when set.

    Returns:
        A :class:`PhysxStrategy` with the device string and rationale.
    """
    # A blank/whitespace override means "unset" — fall through to the heuristic.
    override = (override or "").strip()
    if override:
        if not _DEVICE_RE.match(override):
            raise ValueError(
                f"NETT_SIM_DEVICE={override!r} is not a valid PhysX device; "
                f"expected 'cpu', 'cuda', or 'cuda:<index>'"
            )
        return PhysxStrategy(override, f"NETT_SIM_DEVICE override -> {override}")

    gpu = f"cuda:{max(0, render_device_index)}"

    # Hard rule: GPU PhysX core-dumps on the kinematic-only scene.
    if locomotion != "wheeled":
        return PhysxStrategy(
            "cpu",
            f"kinematic locomotion pins CPU PhysX (GPU PhysX core-dumps on the "
            f"kinematic-only scene)",
        )

    # Wheeled: GPU PhysX is the max-parallel default. Only diverge when VRAM is
    # the binding constraint AND offloading to CPU would actually help.
    if free_vram_bytes is None:
        return PhysxStrategy(
            gpu,
            f"wheeled articulation on GPU PhysX (VRAM headroom unknown; keeping "
            f"the batched-solver fast path)",
        )

    need_mib = estimate_gpu_physx_mib(num_envs) * VRAM_SAFETY_FACTOR
    free_mib = free_vram_bytes / _MIB

    if need_mib <= free_mib:
        return PhysxStrategy(
            gpu,
            f"wheeled articulation on GPU PhysX: est {need_mib:.0f} MiB physics "
            f"(incl. {VRAM_SAFETY_FACTOR:g}x safety) fits in {free_mib:.0f} MiB "
            f"free VRAM at {num_envs} envs",
        )

    # VRAM-constrained. Offloading PhysX to CPU frees ~estimate MiB of VRAM for
    # the renderer (more envs fit) — but only worth it if the CPU can run the
    # articulation across enough threads to not become the new bottleneck.
    freed_mib = estimate_gpu_physx_mib(num_envs)
    if cpu_threads >= MIN_CPU_THREADS_FOR_OFFLOAD:
        return PhysxStrategy(
            "cpu",
            f"VRAM-constrained at {num_envs} envs (need ~{need_mib:.0f} MiB, only "
            f"{free_mib:.0f} MiB free): offloading wheeled PhysX to CPU frees "
            f"~{freed_mib:.0f} MiB VRAM for rendering ({cpu_threads} CPU threads "
            f"absorb the solver)",
        )

    # VRAM tight and too few CPU threads to absorb the solver — staying on GPU is
    # the lesser evil; surface a warning so the operator can lower num_envs.
    return PhysxStrategy(
        gpu,
        f"VRAM tight at {num_envs} envs (need ~{need_mib:.0f} MiB, only "
        f"{free_mib:.0f} MiB free) but only {cpu_threads} CPU threads "
        f"(< {MIN_CPU_THREADS_FOR_OFFLOAD}); keeping GPU PhysX — consider lowering "
        f"num_envs if this OOMs",
        vram_warning=True,
    )


def probe_free_vram_bytes(device_index: int) -> Optional[int]:
    """Free VRAM (bytes) on ``cuda:device_index``, or ``None`` if unavailable.

    Isolated here (and guarded) so the decision logic in
    :func:`select_physx_strategy` stays pure and unit-testable without CUDA.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(device_index)
        return int(free)
    except Exception:
        return None


def usable_cpu_threads() -> int:
    """Threads the process may actually schedule on (respects cpuset affinity)."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1
