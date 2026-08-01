"""Pin smoke test: load smoke_target.yaml but force devices=[0] so NETT assigns
config.device=0 -> cuda:0. Run it with CUDA_VISIBLE_DEVICES=<phys> in the shell.

⚠ THIS SCRIPT DOES NOT PROVE WHICH PHYSICAL CARD IS USED, and it used to claim it did
("the single visible physical GPU is used as cuda:0 (USD-safe)"). Two measurements taken
2026-07-31 disagree, and the mechanism is NOT established:

  * In an isolated two-process spawn (maintenance/diagnostics/device_pin_isolation.py,
    which identifies the child's card by UUID -- the only identifier invariant under
    CUDA_VISIBLE_DEVICES remapping), CVD=<phys> + devices=[0] sends the child to PHYSICAL
    GPU 0 whatever the parent held, because visible_device_scope REPLACES CVD.
  * Inside a full NETT.run(), that same pairing demonstrably spreads across cards -- 8
    cells x 8 GPUs, 24 completed runs at ~11.7 GB each, which will not fit on one 24 GB
    card.

So this pairing works in the full path and is left alone deliberately; changing it would
change which card real runs land on. Do NOT copy it into a plain multiprocessing child,
and do not cite this docstring as evidence for either convention. See blueprint.md
SESSION 2026-07-31b item 7 before touching any production pin.
"""
from __future__ import annotations
from pathlib import Path
from nett_skrl import NETT

cfg = str(Path(__file__).parent / "smoke_target.yaml")
NETT([cfg]).run(output_path="/tmp/smoke_pin_def", devices=[0], verbose=True)
print("SMOKE_PIN_DONE_OK")
