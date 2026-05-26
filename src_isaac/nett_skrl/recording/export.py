"""PNG-sequence to video export for NETT recordings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("nett.recording")


@dataclass
class RecordingCfg:
    """Post-run recording export settings."""

    root: Path
    fps: int = 24
    egocentric_enabled: bool = False
    chamber_enabled: bool = False


def export_recordings(cfg: RecordingCfg) -> None:
    """Convert recorded PNG sequences to MP4 where possible."""
    if cfg.egocentric_enabled:
        _png_dirs_to_mp4(cfg.root / "egocentric", cfg.fps)
    if cfg.chamber_enabled:
        _png_dirs_to_mp4(cfg.root / "chamber", cfg.fps)


def _png_dirs_to_mp4(root: Path, fps: int) -> None:
    if not root.exists():
        return
    for directory in sorted({p.parent for p in root.rglob("*.png")}):
        pngs = sorted(directory.glob("*.png"))
        if not pngs:
            continue
        mp4_path = directory / f"{directory.name}.mp4"
        try:
            import imageio.v3 as iio
            import imageio.v2 as iio2

            first = iio.imread(pngs[0])
            writer = iio2.get_writer(mp4_path, fps=fps)
            try:
                writer.append_data(first)
                for p in pngs[1:]:
                    writer.append_data(iio.imread(p))
            finally:
                writer.close()
        except Exception:
            logger.exception("failed to export recording directory %s", directory)
