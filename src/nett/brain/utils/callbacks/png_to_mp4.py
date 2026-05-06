import glob
import re
from pathlib import Path

import cv2

from stable_baselines3.common.callbacks import BaseCallback


def img2video(record_path: Path, expected_length: int, fps: int = 25):
    png_files = glob.glob(str(record_path / "*.png"))
    if not png_files:
        return

    # Group pngs by episode number
    episode_dict = {}
    pattern = re.compile(r"(\d+)_(\d+)\.png$")
    for png in png_files:
        match = pattern.search(png)
        if match:
            ep, frame = int(match.group(1)), int(match.group(2))
            episode_dict.setdefault(ep, []).append((frame, png))

    for ep, frames in episode_dict.items():
        # Sort frames by frame number
        frames_sorted = sorted(frames, key=lambda x: x[0])
        if len(frames_sorted) < expected_length:
            continue
        first_img = cv2.imread(frames_sorted[0][1])
        if first_img is None:
            continue
        height, width, layers = first_img.shape
        mp4_path = record_path / f"{ep}.mp4"
        out = cv2.VideoWriter(
            str(mp4_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        out.write(first_img)
        for _, png_path in frames_sorted[1:]:
            img = cv2.imread(png_path)
            if img is not None:
                out.write(img)
        out.release()
        # Optionally, remove PNGs after conversion
        for _, png_path in frames_sorted:
            try:
                Path(png_path).unlink()
            except Exception:
                pass


class PngToMp4Callback(BaseCallback):
    """
    Callback to convert episode PNG frames to MP4 videos after each rollout or episode.
    PNGs must be named as <episode>_<frame>.png.
    """

    def __init__(
        self, record_path: Path, expected_size: int, fps: int = 24, verbose: int = 0
    ):
        super().__init__(verbose)
        if not record_path.exists():
            record_path.mkdir(parents=True)

        self.record_path = record_path
        self.fps = fps
        self.expected_size = expected_size

    def _on_rollout_end(self) -> None:
        self._convert_pngs_to_mp4s()

    def _on_step(self) -> bool:
        # Optionally, you can also call conversion here if you want per-episode conversion
        return True

    def _convert_pngs_to_mp4s(self):
        return img2video(self.record_path, self.expected_size, self.fps)
