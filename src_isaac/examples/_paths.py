"""Env-overridable data paths for the NETT examples.

The defaults reproduce the original hardcoded locations, so the example
scripts behave identically out of the box. Override the env vars to run on
another machine, e.g.::

    NETT_VIDEOS_ROOT=/data/videos python examples/train_nature_cnn.py

Nothing here is machine-specific except the fallback default, which is only
used when the corresponding env var is unset.
"""

from __future__ import annotations

import os

# Root of the stimulus videos + design sheets.
VIDEOS_ROOT = os.environ.get("NETT_VIDEOS_ROOT", "/home/zlaborde/code/isaac/videos")
BINDING_DIR = f"{VIDEOS_ROOT}/binding"
BINDING_DESIGN_SHEET = f"{BINDING_DIR}/DesignSheet_Binding.csv"
BINDING_MEDIA_ROOT = f"{BINDING_DIR}/videos"
