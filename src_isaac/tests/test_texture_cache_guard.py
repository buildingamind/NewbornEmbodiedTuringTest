"""Kit texture-cache location guard (runtime/texture_defaults.texture_cache_refusal).

Workspace DECISIONS §187 add. 16, owner fix (b), and the owner's design (§158: one frozen shared
cache per node, never a per-row cache): a disk-cached Kit boot must read a §158 view of the node's
shared master. Kit's node-wide default (`${XDG_CACHE_HOME:-~/.cache}/ov/texturecache`, where
insect's corrupt entries lived, FINDINGS §4dh.44h.211 add. 27) and any other per-row directory
are refused. The refusal tests are the positive controls: the guard can fail.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from nett_skrl.runtime.texture_defaults import (
    TEXTURE_DEFAULT_ARGS,
    default_texture_cache_dir,
    kit_texture_args,
    texture_cache_refusal,
)

PATH = "--/rtx-transient/resourcemanager/localTextureCachePath="
CACHE_OFF = "--/rtx-transient/resourcemanager/enableTextureCaching=false"


def _view(tmp_path, name="cli_20261023T000000Z"):
    """A (fake) §158 view: <root>/views/<name> beside <root>/master.manifest."""
    root = tmp_path / "shared158"
    (root / "views" / name).mkdir(parents=True, exist_ok=True)
    (root / "master.manifest").write_text("# kind=master\n")
    return str(root / "views" / name)


@pytest.fixture(autouse=True)
def _defaults_on(monkeypatch):
    monkeypatch.delenv("NETT_TEXTURE_DEFAULTS", raising=False)


def test_refuses_disk_cache_without_a_path():
    assert any("enableTextureCaching=true" in a for a in TEXTURE_DEFAULT_ARGS)
    why = texture_cache_refusal([], kit_texture_args(""))
    assert why and why.startswith("REFUSED") and "no localTextureCachePath" in why


@pytest.mark.parametrize("suffix", ["", "/"])
def test_refuses_the_node_wide_default_path(monkeypatch, suffix):
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    legacy = os.path.join(os.path.expanduser("~"), ".cache", "ov", "texturecache")
    assert default_texture_cache_dir() == legacy
    why = texture_cache_refusal([], kit_texture_args(PATH + legacy + suffix))
    assert why and "node-wide default" in why
    # the `~` spelling resolves to the same directory
    assert texture_cache_refusal([], kit_texture_args(PATH + "~/.cache/ov/texturecache"))


def test_refuses_the_xdg_default_and_the_home_default(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_texture_cache_dir() == str(tmp_path / "xdg" / "ov" / "texturecache")
    assert texture_cache_refusal([], kit_texture_args(PATH + default_texture_cache_dir()))
    home_default = os.path.join(os.path.expanduser("~"), ".cache", "ov", "texturecache")
    assert texture_cache_refusal([], kit_texture_args(PATH + home_default))


def test_accepts_a_view_of_the_shared_master(tmp_path):
    v = _view(tmp_path)
    assert texture_cache_refusal([], kit_texture_args(PATH + v)) is None
    assert texture_cache_refusal([], kit_texture_args(PATH + v + "/")) is None


def test_refuses_a_per_row_directory(tmp_path):
    # the owner does not want a row-specific cache: a fresh directory, an existing one, a
    # "views" directory with no master beside it, and the master itself are all refused
    fresh = tmp_path / "fresh"
    assert "per-row cache" in texture_cache_refusal([], kit_texture_args(PATH + str(fresh)))
    fresh.mkdir()
    assert texture_cache_refusal([], kit_texture_args(PATH + str(fresh)))
    orphan = tmp_path / "x" / "views" / "v"
    orphan.mkdir(parents=True)
    assert texture_cache_refusal([], kit_texture_args(PATH + str(orphan)))
    v = _view(tmp_path)
    assert texture_cache_refusal([], kit_texture_args(PATH + os.path.dirname(os.path.dirname(v)) + "/master"))
    # a view that does not exist yet is not a view
    assert texture_cache_refusal([], kit_texture_args(PATH + v + "_missing"))


def test_accepts_disk_caching_off(monkeypatch):
    # NETT_TEXTURE_DEFAULTS=0 drops enableTextureCaching=true; an explicit false turns it off.
    monkeypatch.setenv("NETT_TEXTURE_DEFAULTS", "0")
    assert kit_texture_args("") == ""
    assert texture_cache_refusal([], kit_texture_args("")) is None
    monkeypatch.delenv("NETT_TEXTURE_DEFAULTS")
    assert texture_cache_refusal([], kit_texture_args(CACHE_OFF)) is None


def test_last_occurrence_wins_across_argv_and_kit_args(tmp_path):
    legacy = default_texture_cache_dir()
    v = _view(tmp_path)
    # a path on the host command line (tools/render_probe_sweep.py appends one to sys.argv)
    assert texture_cache_refusal([PATH + v], kit_texture_args("")) is None
    # kit_args are appended AFTER argv (AppLauncher._resolve_kit_args), so they win
    assert texture_cache_refusal([PATH + v], kit_texture_args(PATH + legacy))
    assert texture_cache_refusal([PATH + legacy], kit_texture_args(PATH + v)) is None
    # an empty value is no path
    assert texture_cache_refusal([], kit_texture_args(PATH))


def test_environment_load_refuses_before_booting_kit(tmp_path, monkeypatch):
    """The chokepoint: Environment.load raises before AppLauncher is constructed."""
    from nett_skrl.environment.environment import Environment

    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    monkeypatch.delenv("NETT_EXTRA_KIT_ARGS", raising=False)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    booted = []

    class FakeAppLauncher:
        def __init__(self, **kwargs):
            booted.append(kwargs)
            self.app = object()

    monkeypatch.setitem(sys.modules, "isaaclab.app", types.SimpleNamespace(AppLauncher=FakeAppLauncher))
    sheet = tmp_path / "design.csv"
    sheet.write_text(
        "ImprintCondition,Phase,TestCondition,TargetVideo,NonTargetVideo,LeftMonitor,RightMonitor\n"
        "Object1,Train,,a.mp4,b.mp4,a.mp4,b.mp4\n"
    )
    (tmp_path / "videos").mkdir()
    env = Environment(design_sheet=sheet, media_root=tmp_path / "videos")

    class _Task:
        current_mode = "train"
        condition = "Object1"
        episodes = {"train": 1, "test": 1, "record": 1}
        seed = 0
        dry_run = False
        eval_metrics_only = False
        path = tmp_path / "run"

    with pytest.raises(SystemExit) as exc:
        env.load(_Task())
    assert "REFUSED to boot Kit" in str(exc.value)
    assert booted == []
