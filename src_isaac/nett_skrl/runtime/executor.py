"""Thin ``ProcessPoolExecutor`` wrapper that optionally mutes child stdout."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor


class Executor(ProcessPoolExecutor):
    def __init__(self, verbose: bool) -> None:
        def mute() -> None:
            sys.stdout = open(os.devnull, "w")

        super().__init__(
            max_workers=os.cpu_count(),
            initializer=None if verbose else mute,
        )

    def __enter__(self) -> "Executor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        super().__exit__(exc_type, exc_val, exc_tb)
        return False
