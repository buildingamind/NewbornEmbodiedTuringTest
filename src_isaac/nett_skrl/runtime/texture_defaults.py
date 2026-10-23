"""Default Kit texture-residency flags, applied to every Isaac boot.

ACTIVE: textureLoaderThreadCount=8 + enableTextureCaching=true. Texture STREAMING is
left at the engine default (ON): the monitor-texture flicker is fixed upstream by BC7
frame textures (repoA frame_format=bc7 default -- 4x fewer bytes, so async streaming
keeps up under load), so streaming no longer has to be disabled here. The streaming-off
flag is retained as a commented FALLBACK in TEXTURE_DEFAULT_ARGS below.

MEASURED 2026-07-21, 32-proc load, 32 jobs per arm (streaming-off era -- kept as the
throughput record for the fallback):

    THROUGHPUT (per-job result.json, n=32 -- the reliable measurement)
        control                                     3.617 steps/s
        streaming-off alone                         3.394        (-6.2%)
        streaming-off + these two defaults          3.718        (+2.8% vs control)

streaming-ON + these two defaults was not measured directly, but should be >= the 3.617
control: the two defaults are net-positive and dropping streaming-off removes its -6.2%
penalty. (streaming-off-era flicker 0.00, frame-0 blanks 0/64, stimulus unchanged.)

⚠ NOT a memory win. An earlier version of this file claimed -12.5% RSS/proc. That was
a MEASUREMENT BUG: the sweep's inline RSS average included samples taken while zero
processes were running, dragging the mean down. Recomputed from the raw samples the
effect is ~0.13 GB (7.16 -> 7.03) against a standard deviation of 1.2 GB, i.e.
indistinguishable from zero. Per-proc RSS here is far too noisy (sd 1.2-2.9 GB) to
resolve effects of this size -- do not quote memory deltas from it without many
repeats.

`enableTextureCaching=true` backs textures with the on-disk cache
(`~/.cache/ov/texturecache`, see `localTextureCachePath`) instead of holding
in-memory copies. That is a DISK-SPACE and first-run-latency trade, not free: the
first boot after a cache wipe pays to populate it.

NOTE these were predicted BACKWARDS before measurement -- "more loader threads and an
added cache should raise memory" -- so do not reason about them from first
principles; re-measure if the scene or media change.

Set NETT_TEXTURE_DEFAULTS=0 to disable (e.g. to reproduce a pre-2026-07-21 run).
"""
from __future__ import annotations

import os

_R = "/rtx-transient/resourcemanager"

TEXTURE_DEFAULT_ARGS: tuple[str, ...] = (
    f"--{_R}/textureLoaderThreadCount=8",
    f"--{_R}/enableTextureCaching=true",
    # STREAMING LEFT ON (engine default). The monitor-texture flicker is fixed UPSTREAM
    # by BC7 frame textures (repoA frame_format=bc7 default: 4x fewer bytes, so Kit's
    # async streaming keeps up under 32-proc load without forcing full residency). This
    # keeps full mips + fidelity and recovers streaming-off's ~6% throughput. See repoA
    # video/config.py (frame_format) and blueprint SESSION 2026-07-22.
    #
    # FALLBACK (commented, NOT active): enableTextureStreaming=false is a pure-TIMING
    # flicker fix — synchronous loads, every mip retained, texture bit-for-bit (0.00
    # flicker, 0/64 frame-0 blanks). It costs ~6% throughput (see the docstring table)
    # and keeps ALL textures resident. Its old test-phase host-MemoryError cost (5/16
    # jobs at 2/GPU) was resolved 2026-07-21 by the GPU rollout buffer (0/16). Re-enable
    # it — or maxMipCount=8 below — ONLY if flicker returns on a config WITHOUT BC7.
    #   f"--{_R}/enableTextureStreaming=false",
    # (reference, NOT active) maxMipCount=8 — flicker fix (32-proc load, 64 videos/arm):
    #
    #     original control (nothing set)   flicker 3.20   frame-0 blanks 41/64
    #     loader+cache defaults only       flicker 2.38   frame-0 blanks 31/64
    #     + maxMipCount=11                 flicker 2.05   frame-0 blanks 33/64
    #     + maxMipCount=8                  flicker 0.00   frame-0 blanks  0/64
    #
    # WHY 8 AND NOT 11: the monitor video is 1440x900 -> an 11-level mip chain, so
    # maxMipCount=11 (and 12..15, and 0) caps NOTHING and is the SAME configuration
    # as the uncapped default — measured, 2.05 vs 2.38 is inside the noise. The fix
    # requires actually REMOVING levels.
    #
    # FIDELITY VERIFIED, near and far, at two resolutions: at res128 and res256 the
    # stimulus area, in-stimulus edge sharpness and saturated-core fraction at 8 all
    # sit inside the same-config replicate noise floor (two runs of ONE setting differ
    # by max 88 LSB / 43% of pixels / 0.014 core — so 10..15 are indistinguishable and
    # any claim of decline in that range is noise). The real cliff is BELOW 8:
    # res256 mip6 core 0.853 vs 0.949 (7x the noise floor); mip<=4 badly blurred;
    # mip 1-2 render NO STIMULUS AT ALL. Do not lower this without re-measuring.
    #
    # ⚠ MECHANISM UNKNOWN. Neither "keeps the finest N" nor "keeps the coarsest N"
    # predicts the observed pattern (cap 4 blurs, cap 6 is pristine at res128), so
    # this is an empirical result, not a principled one. It is verified at res128 and
    # res256 only — RE-MEASURE before trusting it at higher resolutions.
    # f"--{_R}/maxMipCount=8",   # inactive: streaming-off chosen instead
)


def kit_texture_args(existing: str = "") -> str:
    """Compose ``existing`` kit_args with the measured texture defaults.

    Defaults go FIRST so anything the caller passes in ``existing`` (e.g. via
    NETT_EXTRA_KIT_ARGS) overrides them -- Kit takes the LAST occurrence of a flag
    when it re-parses argv. That ordering is what lets an experiment pin a different
    value without having to unset the defaults.
    """
    if os.environ.get("NETT_TEXTURE_DEFAULTS", "1").strip().lower() in ("0", "false", "no"):
        return existing
    parts = list(TEXTURE_DEFAULT_ARGS)
    if existing:
        parts.append(existing)
    return " ".join(parts)


# ---- texture-cache location guard (workspace DECISIONS §187 add. 15 + add. 16, owner fix (b)) ----
# ⛔ The owner's design is ONE FROZEN SHARED cache per node, never a per-row cache (DECISIONS §158: "It was
# supposed to be a shared cache and not a by-run cache"; §187 add. 16). Workspace tools/texcache_shared.py
# implements it: <root>/shared158/master (0444 files, sha256 master.manifest, verified at every launch) and,
# per run, <root>/shared158/views/<name> -- hard links to the master's own inodes, so every run reads the
# same frozen bytes; a texture the master lacks is written into the view and reaches the master only by a
# validated promote. A disk-cached Kit with any OTHER localTextureCachePath is refused here:
#   - none / Kit's node-wide default `${XDG_CACHE_HOME:-~/.cache}/ov/texturecache`: one writable directory,
#     no manifest, concurrent non-atomic writes -- where insect's corrupt entries lived (a 0-byte pair 10-13,
#     a TRUNCATED .0 Kit kept reading; FINDINGS §4dh.44h.211 add. 27);
#   - any other directory: a per-row cache, which the owner does not want.
# No opt-out: boot through tools/launch_arm.sh / tools/retest_arm.sh (they create the view), or pass
# enableTextureCaching=false.
_CACHE_FLAG = f"--{_R}/enableTextureCaching="
_PATH_FLAG = f"--{_R}/localTextureCachePath="


def _last_value(args: list[str], flag: str) -> str | None:
    """Kit takes the LAST occurrence of a flag, so the last one is the effective value."""
    value = None
    for a in args:
        if a.startswith(flag):
            value = a[len(flag):]
    return value


def default_texture_cache_dir() -> str:
    """Kit's node-wide default texture cache (where a boot without localTextureCachePath writes)."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "ov", "texturecache")


def is_shared_view(path: str) -> bool:
    """True for an existing <root>/views/<name> directory beside <root>/master.manifest (a §158 view)."""
    view = os.path.realpath(os.path.expanduser(path).rstrip("/") or "/")
    parent = os.path.dirname(view)
    return (os.path.isdir(view) and os.path.basename(parent) == "views"
            and os.path.isfile(os.path.join(os.path.dirname(parent), "master.manifest")))


def texture_cache_refusal(argv: list[str], kit_args: str) -> str | None:
    """Why this Kit boot must be refused, or None. ``argv`` + ``kit_args`` is the effective Kit
    command line (AppLauncher appends kit_args to sys.argv). Disk caching counts as ON only when
    the last enableTextureCaching is true -- TEXTURE_DEFAULT_ARGS sets it, NETT_TEXTURE_DEFAULTS=0
    drops it, and an explicit false turns it off."""
    args = list(argv) + (kit_args or "").split()
    if (_last_value(args, _CACHE_FLAG) or "").strip().lower() not in ("true", "1"):
        return None
    path = _last_value(args, _PATH_FLAG)
    if path and is_shared_view(path):
        return None
    default = default_texture_cache_dir()
    if not path:
        why = f"no localTextureCachePath was given (Kit would use its node-wide default {default})"
    else:
        expanded = os.path.realpath(os.path.expanduser(path).rstrip("/") or "/")
        legacy = {os.path.realpath(d) for d in (default, os.path.join(os.path.expanduser("~"), ".cache", "ov", "texturecache"))}
        why = (f"localTextureCachePath={path} is Kit's node-wide default" if expanded in legacy else
               f"localTextureCachePath={path} is not a view of a node's shared master (<root>/views/<name> "
               "beside <root>/master.manifest), i.e. a per-row cache")
    return (
        f"REFUSED to boot Kit: texture disk caching is on and {why}. Every run must read the node's ONE "
        "frozen shared cache through a §158 view (owner, workspace DECISIONS §158, §187 add. 16; FINDINGS "
        "§4dh.44h.211 add. 27). Launch through tools/launch_arm.sh or tools/retest_arm.sh (they create the "
        f"view), or pass {_CACHE_FLAG}false."
    )
