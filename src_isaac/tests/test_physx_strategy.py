"""Tester: init-time PhysX placement strategy vs simple baselines.

The wheeled-physics goal is maximum throughput on highly-parallelized runs: keep
the articulation solver on the GPU (batched across envs) by default, and only
offload it to CPU PhysX when VRAM cannot hold the GPU-PhysX state for the
requested num_envs AND the host has enough threads to absorb the solver — which
frees VRAM for more rendered envs. These tests pin that decision table with
hand-computed baselines so the heuristic can be tuned without silently changing
behavior.

Pure Python (no CUDA / no Kit): select_physx_strategy takes plain numbers.
"""

from __future__ import annotations

import importlib

import pytest

strat = importlib.import_module("nett_skrl.environment.physx_strategy")
from nett_skrl.environment.physx_strategy import (  # noqa: E402
    PhysxStrategy,
    estimate_gpu_physx_mib,
    select_physx_strategy,
)

_MIB = 1024 * 1024
_GB = 1024 * _MIB


def _sel(**kw):
    base = dict(
        locomotion="wheeled",
        render_device_index=0,
        num_envs=64,
        free_vram_bytes=20 * _GB,
        cpu_threads=64,
        override=None,
    )
    base.update(kw)
    return select_physx_strategy(**base)


# --------------------------- hard-rule baselines ----------------------------

def test_kinematic_always_pins_cpu_even_with_infinite_vram():
    """GPU PhysX core-dumps on the kinematic-only scene, so kinematic is CPU no
    matter how much VRAM is free."""
    s = _sel(locomotion="kinematic", free_vram_bytes=80 * _GB)
    assert s.device == "cpu"
    assert not s.on_gpu
    assert "kinematic" in s.rationale


def test_kinematic_cpu_independent_of_thread_count():
    assert _sel(locomotion="kinematic", cpu_threads=1).device == "cpu"


def test_override_wins_over_every_heuristic():
    # Override beats even a GPU-friendly wheeled+ample-VRAM situation...
    assert _sel(override="cpu").device == "cpu"
    # ...and beats the kinematic CPU pin (probe / forced GPU compare).
    assert _sel(locomotion="kinematic", override="cuda:3").device == "cuda:3"


def test_blank_or_whitespace_override_is_treated_as_unset():
    """A blank/whitespace NETT_SIM_DEVICE must not become a bogus sim.device; it
    falls through to the heuristic (wheeled+ample VRAM -> GPU)."""
    for blank in ("", "   ", "\t", None):
        s = _sel(override=blank)
        assert s.device == "cuda:0", f"override={blank!r} should be ignored"


def test_malformed_override_raises_clear_error():
    """A typo'd device (e.g. 'gpu', 'cuda0') fails fast at init with a clear
    message instead of producing an opaque Isaac failure later."""
    for bad in ("gpu", "cuda0", "cpu:0", "cuda:", "0"):
        with pytest.raises(ValueError, match="NETT_SIM_DEVICE"):
            _sel(override=bad)
    # Valid forms are accepted.
    assert _sel(override="cuda").device == "cuda"
    assert _sel(override="cuda:7").device == "cuda:7"
    assert _sel(override="cpu").device == "cpu"


def test_negative_render_index_is_clamped_not_propagated():
    """A misconfigured negative device index must not yield 'cuda:-1'."""
    assert _sel(render_device_index=-1).device == "cuda:0"


# ---------------------- wheeled GPU-default baseline ------------------------

def test_wheeled_ample_vram_stays_on_gpu():
    s = _sel(num_envs=64, free_vram_bytes=20 * _GB)
    assert s.device == "cuda:0"
    assert s.on_gpu and not s.vram_warning


def test_wheeled_respects_render_device_index():
    assert _sel(render_device_index=5).device == "cuda:5"


def test_wheeled_unknown_vram_keeps_gpu_fast_path():
    """If we cannot probe VRAM we must not silently demote to CPU — GPU is the
    parallel default for the articulation."""
    s = _sel(free_vram_bytes=None)
    assert s.device == "cuda:0"
    assert "unknown" in s.rationale.lower()


# ------------------- VRAM-constrained offload decision ----------------------

def test_wheeled_vram_constrained_with_threads_offloads_to_cpu():
    """When GPU-PhysX state won't fit but the host has threads, offload to CPU to
    free VRAM for rendering."""
    # Pick num_envs whose estimate (×safety) clearly exceeds the free VRAM.
    need_mib = estimate_gpu_physx_mib(4096) * strat.VRAM_SAFETY_FACTOR
    free = int((need_mib - 256) * _MIB)  # 256 MiB short of the requirement
    s = _sel(num_envs=4096, free_vram_bytes=free, cpu_threads=64)
    assert s.device == "cpu"
    assert "offload" in s.rationale.lower()
    assert not s.vram_warning


def test_wheeled_vram_constrained_few_threads_stays_gpu_with_warning():
    """VRAM tight but too few CPU threads to absorb the solver: stay on GPU and
    warn (operator should lower num_envs) rather than create a CPU bottleneck."""
    need_mib = estimate_gpu_physx_mib(4096) * strat.VRAM_SAFETY_FACTOR
    free = int((need_mib - 256) * _MIB)
    s = _sel(
        num_envs=4096,
        free_vram_bytes=free,
        cpu_threads=strat.MIN_CPU_THREADS_FOR_OFFLOAD - 1,
    )
    assert s.device == "cuda:0"
    assert s.vram_warning is True


def test_offload_threshold_is_monotonic_in_num_envs():
    """At fixed (tight) VRAM, raising env count never flips CPU->GPU: more envs
    only ever makes GPU PhysX less likely to fit."""
    free = int(estimate_gpu_physx_mib(2048) * strat.VRAM_SAFETY_FACTOR * _MIB)
    decisions = [
        _sel(num_envs=n, free_vram_bytes=free, cpu_threads=64).on_gpu
        for n in (64, 512, 2048, 8192)
    ]
    # Once it leaves the GPU it must not come back as envs grow.
    assert decisions == sorted(decisions, reverse=True)


def test_exact_fit_boundary_stays_on_gpu():
    """At need == free (estimate×safety exactly equals free VRAM) we keep GPU —
    the safety factor already carries the margin."""
    need_bytes = int(estimate_gpu_physx_mib(100) * strat.VRAM_SAFETY_FACTOR * _MIB)
    s = _sel(num_envs=100, free_vram_bytes=need_bytes, cpu_threads=64)
    assert s.on_gpu


# ------------------------------ shape / typing ------------------------------

def test_strategy_is_frozen_dataclass_with_rationale():
    s = _sel()
    assert isinstance(s, PhysxStrategy)
    assert isinstance(s.rationale, str) and s.rationale
    with pytest.raises(Exception):
        s.device = "cuda:1"  # frozen


def test_estimate_grows_with_env_count():
    assert estimate_gpu_physx_mib(0) < estimate_gpu_physx_mib(1) < estimate_gpu_physx_mib(1000)
    assert estimate_gpu_physx_mib(-5) == estimate_gpu_physx_mib(0)  # clamped
