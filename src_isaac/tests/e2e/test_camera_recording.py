"""Camera recording end-to-end tests.

Verifies both recording modes produce on-disk artifacts:

    * egocentric  — single ``frame.png`` per timestep per env (monocular eye)
    * chamber     — third-person camera under ``recordings/chamber/``

Each test runs ``modes=["record"]`` (smallest scope) with ``steps_per_episode``
trimmed so a full pass finishes in well under a minute.
"""

from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e_isaac


def _record_root(condition_dir: Path, kind: str) -> Path:
    """Directory the env writes raw PNGs into for ``record`` mode."""
    return condition_dir / "recordings" / kind / "record"


def _png_dirs_under(root: Path) -> list[Path]:
    """Per-frame PNG dirs (each one becomes one MP4 after export)."""
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("*.png")})


def _mp4_files_under(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.mp4"))


def test_egocentric_monocular_recording_produces_artifacts(run_nett, e2e_smoke_cfg):
    """One PNG sequence per env (no left/right split) + MP4 from export."""
    e2e_smoke_cfg["num_brains"] = 1
    e2e_smoke_cfg["episodes"] = {"record": 1}
    e2e_smoke_cfg["steps_per_episode"] = 20
    # ``tSNE`` holds one condition-sampling pose while advancing stimulus
    # frames, which is the data-collection path these tests should exercise.
    e2e_smoke_cfg["environment"]["record_mode"] = "tSNE"
    e2e_smoke_cfg["environment"]["recording"] = {
        "egocentric": {"record": None},   # null = every episode in the record phase
        "fps": 24,
    }

    output_dir = run_nett(e2e_smoke_cfg)
    ego_root = _record_root(output_dir / "Object1", "egocentric")

    png_dirs = _png_dirs_under(ego_root)
    assert png_dirs, f"no egocentric PNG sequence under {ego_root}"
    # Monocular eye: leaf dir must NOT be named 'left' or 'right'.
    leaf_names = {d.name for d in png_dirs}
    assert not (leaf_names & {"left", "right"}), \
        f"egocentric run produced unexpected left/right subdirs: {leaf_names}"

    mp4s = _mp4_files_under(ego_root)
    assert mp4s, f"export_recordings produced no MP4 under {ego_root}"


def test_chamber_recording_produces_artifacts(run_nett, e2e_smoke_cfg):
    """Chamber camera writes its own PNG tree under ``recordings/chamber/``."""
    e2e_smoke_cfg["num_brains"] = 1
    e2e_smoke_cfg["episodes"] = {"record": 1}
    e2e_smoke_cfg["steps_per_episode"] = 20
    # ``tSNE`` holds one condition-sampling pose while advancing stimulus
    # frames, which is the data-collection path these tests should exercise.
    e2e_smoke_cfg["environment"]["record_mode"] = "tSNE"
    e2e_smoke_cfg["environment"]["recording"] = {
        "chamber": {"record": None},   # null = every episode in the record phase
        "fps": 24,
    }

    output_dir = run_nett(e2e_smoke_cfg)
    chamber_root = _record_root(output_dir / "Object1", "chamber")

    png_dirs = _png_dirs_under(chamber_root)
    assert png_dirs, f"no chamber PNG sequence under {chamber_root}"

    mp4s = _mp4_files_under(chamber_root)
    assert mp4s, f"chamber export produced no MP4 under {chamber_root}"
