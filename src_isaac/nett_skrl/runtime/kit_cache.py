"""Per-cell Kit cache dirs, so packed cells stop fighting over shared caches.

Every Kit process points at the SAME two cache trees (visible in any kit log's
token block):

    omni_cache        -> <isaacsim>/kit/cache          (DerivedDataCache, shader cache)
    omni_global_cache -> ~/.cache/ov                   (texturecache)

That is fine for one cell. When N cells start at once it means N processes
contending for the same on-disk caches, and both are LOCKED by their first
owner:

  * omni.datastore: "Locked base cache layer '<...>/kit/cache/DerivedDataCache'"
  * omni.kvdb:      "Disabling key-value database because another kit process is
                     locking it"  (kvdb is leveldb; leveldb takes an exclusive
                     directory lock)

NOTE the kvdb line is NOT by itself a fault -- it is emitted by every secondary
Kit process and those go on to run fine (measured: it appeared in 23 of 24 cells
of a 24-way wave, 21 of which completed). Treat it as an indicator of shared-cache
contention, not as the cause of a hang.

Giving each cell its own cache root removes the contention. Set NETT_KIT_CACHE_ID
per cell in the wave launcher (0..N-1). Use a STABLE id (cell index), not a PID:
the cache is worth ~180 MB/cell after one run and reusing it across runs is what
keeps Kit startup fast. Unset -> the shared default, i.e. exactly today's
behavior, so a solo run is unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Cell identity within a wave. Unset -> shared caches (default behavior).
ENV_CACHE_ID = "NETT_KIT_CACHE_ID"
#: Where per-cell cache roots live. Local disk, not $HOME (which may be NFS).
ENV_CACHE_ROOT = "NETT_KIT_CACHE_ROOT"
DEFAULT_CACHE_ROOT = "/tmp/nett_kit_cache"


def cell_cache_dir() -> Path | None:
    """This cell's private Kit cache root, or None if it should share the default."""
    cache_id = os.environ.get(ENV_CACHE_ID)
    if not cache_id:
        return None
    root = Path(os.environ.get(ENV_CACHE_ROOT) or DEFAULT_CACHE_ROOT)
    return root / f"cell{cache_id}"


def kit_cache_args(existing: str = "") -> str:
    """Append the ``kit_args`` that repoint Kit's caches at this cell's own dirs.

    No-op (returns ``existing`` unchanged) when NETT_KIT_CACHE_ID is unset, so the
    shared-cache default is untouched.

    Both tokens are overridable on the Kit command line the same way Isaac already
    overrides ``--/app/tokens/exe-path``; verified by reading the token block back
    out of the launched process's kit log.
    """
    cache_dir = cell_cache_dir()
    if cache_dir is None:
        return existing
    omni = cache_dir / "omni"
    omni.mkdir(parents=True, exist_ok=True)
    parts = [existing] if existing else []
    parts.append(f"--/app/tokens/omni_cache={omni}")
    # omni_global_cache (~/.cache/ov -> texturecache) deliberately stays SHARED.
    # Splitting it too was measured and is a LOSS: it is read-mostly content that
    # every cell converts identically, so a private copy just makes each cell redo
    # the work. 24-way, warm, per-cell BOTH caches -> 8.61 it/s/cell (205.9 agg);
    # shared texturecache -> see NETT_KIT_CACHE_SPLIT_GLOBAL if you want to A/B it.
    if os.environ.get("NETT_KIT_CACHE_SPLIT_GLOBAL") == "1":
        glob = cache_dir / "global"
        glob.mkdir(parents=True, exist_ok=True)
        parts.append(f"--/app/tokens/omni_global_cache={glob}")
    return " ".join(parts)
