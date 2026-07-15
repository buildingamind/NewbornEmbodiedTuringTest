"""Per-cell Kit cache dirs (NETT_KIT_CACHE_ID)."""

from __future__ import annotations

import pytest

from nett_skrl.runtime.kit_cache import (
    DEFAULT_CACHE_ROOT,
    ENV_CACHE_ID,
    ENV_CACHE_ROOT,
    cell_cache_dir,
    kit_cache_args,
)


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    monkeypatch.delenv(ENV_CACHE_ID, raising=False)
    monkeypatch.delenv(ENV_CACHE_ROOT, raising=False)
    monkeypatch.delenv("NETT_KIT_CACHE_SPLIT_GLOBAL", raising=False)


def test_unset_is_a_no_op_so_solo_runs_keep_the_shared_cache():
    assert cell_cache_dir() is None
    assert kit_cache_args("--/x=1") == "--/x=1"
    assert kit_cache_args() == ""


def test_cache_id_gives_this_cell_a_private_omni_cache(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_CACHE_ID, "5")
    monkeypatch.setenv(ENV_CACHE_ROOT, str(tmp_path))
    args = kit_cache_args()
    assert f"--/app/tokens/omni_cache={tmp_path}/cell5/omni" in args
    assert (tmp_path / "cell5" / "omni").is_dir()


def test_texturecache_stays_shared_by_default(monkeypatch, tmp_path):
    """Splitting omni_global_cache too was measured as a loss, not a win."""
    monkeypatch.setenv(ENV_CACHE_ID, "0")
    monkeypatch.setenv(ENV_CACHE_ROOT, str(tmp_path))
    assert "omni_global_cache" not in kit_cache_args()


def test_split_global_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_CACHE_ID, "0")
    monkeypatch.setenv(ENV_CACHE_ROOT, str(tmp_path))
    monkeypatch.setenv("NETT_KIT_CACHE_SPLIT_GLOBAL", "1")
    args = kit_cache_args()
    assert f"--/app/tokens/omni_global_cache={tmp_path}/cell0/global" in args


def test_composes_after_existing_kit_args(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_CACHE_ID, "1")
    monkeypatch.setenv(ENV_CACHE_ROOT, str(tmp_path))
    args = kit_cache_args("--/threads=4").split()
    assert args[0] == "--/threads=4"


def test_default_root_is_local_disk_not_home():
    """$HOME may be NFS; a Kit cache on NFS is a bad idea."""
    assert DEFAULT_CACHE_ROOT.startswith("/tmp")
