"""Definitive pin test: load smoke_target.yaml but force devices=[0] so NETT
assigns config.device=0 -> cuda:0. Combined with CUDA_VISIBLE_DEVICES=<phys>
in the shell, the single visible physical GPU is used as cuda:0 (USD-safe).
"""
from __future__ import annotations
import sys
from pathlib import Path
from nett_skrl import NETT

cfg = str(Path(__file__).parent / "smoke_target.yaml")
NETT([cfg]).run(output_path="/tmp/smoke_pin_def", devices=[0], verbose=True)
print("SMOKE_PIN_DONE_OK")
