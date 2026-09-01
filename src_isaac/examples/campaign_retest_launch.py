"""Packed/staggered/resumable launcher for test-only replays (campaign_retest.py).

Runs the deterministic TEST phase from saved checkpoints for a set of completed run
dirs, to obtain the FULL test=N episodes/condition a run may lack (e.g. an interrupted
test phase). Test is render-bound, so packing is lighter than training — default
2 jobs/GPU. Resumable via done-markers; staggered launch avoids concurrent Kit-init
DEVICE_LOST.

Env knobs:
  NETT_OUT_ROOT      campaign output root (default ~/nett_campaign)
  NETT_RETEST_GLOB   glob of run dirs (default <root>/*/*_off*)
  NETT_GPUS          device pool (default 0..7)
  NETT_JOBS_PER_GPU  packing (default 2)
  NETT_STAGGER_SECS  launch spacing (default 12)
  NETT_TEST_EPS      REPEATS PER TEST ROW (default from each run's config).
                     NOT episodes/condition -- that reading is short by the row
                     count of the design sheet (56x on the parsing sheet). See
                     campaign_train.py's env table for the derivation.
  NETT_RETEST_DIR    scheduler state dir (default <root>/_retest)
  NETT_PYTHON        interpreter (default: the one running this launcher)
  NETT_ISAAC_LAB     repo A isaac_lab/source path (REQUIRED)
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Portable: default the interpreter to the one running this launcher (the venv
# python), and require the repo A source path via NETT_ISAAC_LAB — no machine path baked in.
PY = os.environ.get("NETT_PYTHON") or sys.executable
RA = os.environ.get("NETT_ISAAC_LAB")
if not RA:
    raise SystemExit(
        "Set NETT_ISAAC_LAB to repo A's isaac_lab/source path "
        "(<repoA>/isaac_lab/source) so the worker can import nett_isaac."
    )
ROOT = Path(os.environ.get("NETT_OUT_ROOT", "~/nett_campaign")).expanduser()
RETEST_DIR = Path(os.environ.get("NETT_RETEST_DIR", str(ROOT / "_retest"))).expanduser()
LOGS = RETEST_DIR / "logs"
DONE = RETEST_DIR / "done"
STATUS = RETEST_DIR / "status.json"

GLOB = os.environ.get("NETT_RETEST_GLOB", str(ROOT / "*" / "*_off*"))
GPUS = [int(g) for g in os.environ.get("NETT_GPUS", "0,1,2,3,4,5,6,7").split(",")]
JOBS_PER_GPU = int(os.environ.get("NETT_JOBS_PER_GPU", "2"))
STAGGER = float(os.environ.get("NETT_STAGGER_SECS", "12"))
CAP = len(GPUS) * JOBS_PER_GPU


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_dirs() -> list[str]:
    return sorted(d for d in glob.glob(GLOB) if os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.yaml")))


def jid(d: str) -> str:
    return os.path.basename(d.rstrip("/"))


def write_status(state: dict) -> None:
    STATUS.write_text(json.dumps({"updated": _now(),
        "totals": {s: sum(1 for j in state.values() if j["status"] == s)
                   for s in ("done", "failed", "running", "pending")},
        "jobs": state}, indent=2))


def launch(d: str, gpu: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["NETT_DEVICE"] = "0"
    env["NETT_WANDB_MODE"] = os.environ.get("NETT_WANDB_MODE", "offline")
    env["PYTHONPATH"] = f"{HERE.parent}:{RA}"
    env["NETT_KIT_THREADS"] = str(max(1, 128 // max(1, CAP)))
    for k in ("NETT_RETEST_GLOB", "NETT_GPUS", "NETT_JOBS_PER_GPU", "NETT_STAGGER_SECS"):
        env.pop(k, None)
    log = (LOGS / f"{jid(d)}.log").open("w")
    log.write(f"# retest {_now()} gpu={gpu} {jid(d)}\n"); log.flush()
    p = subprocess.Popen([PY, "-u", str(HERE / "campaign_retest.py"), d],
                         stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
    p._log_fh = log
    return p


def main() -> int:
    for x in (LOGS, DONE):
        x.mkdir(parents=True, exist_ok=True)
    dirs = run_dirs()
    state = {jid(d): {"status": "done" if (DONE / f"{jid(d)}.done").exists() else "pending",
                      "gpu": None, "rc": None} for d in dirs}
    queue = [d for d in dirs if state[jid(d)]["status"] == "pending"]
    running: dict[int, tuple[str, subprocess.Popen, int]] = {}
    load = {g: 0 for g in GPUS}
    last = 0.0
    slot = 0
    print(f"[retest] {len(dirs)} run dirs ({len(queue)} pending), {len(GPUS)}gpu x{JOBS_PER_GPU} = {CAP} slots", flush=True)
    while queue or running:
        now = time.time()
        while queue and len(running) < CAP and (now - last) >= STAGGER:
            gpu = min(GPUS, key=lambda g: load[g])
            if load[gpu] >= JOBS_PER_GPU:
                break
            d = queue.pop(0)
            p = launch(d, gpu); slot += 1
            running[slot] = (jid(d), p, gpu); load[gpu] += 1; last = time.time()
            print(f"[retest] start {jid(d)} gpu{gpu} pid{p.pid} ({len(running)}/{CAP})", flush=True)
            now = time.time()
        # Mark launched jobs running so status.json reflects them mid-flight.
        for name, _, _ in running.values():
            if state[name]["status"] == "pending":
                state[name]["status"] = "running"
        write_status(state)
        time.sleep(15)
        for sk, (name, p, gpu) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            p._log_fh.flush(); p._log_fh.close()
            load[gpu] -= 1; state[name]["rc"] = rc
            if rc == 0:
                (DONE / f"{name}.done").write_text(_now()); state[name]["status"] = "done"
                print(f"[retest] DONE {name}", flush=True)
            else:
                state[name]["status"] = "failed"
                print(f"[retest] FAIL {name} rc={rc}", flush=True)
            del running[sk]
        write_status(state)   # persist final completions too
    d = sum(1 for j in state.values() if j["status"] == "done")
    f = sum(1 for j in state.values() if j["status"] == "failed")
    write_status(state)
    print(f"[retest] COMPLETE {d} done {f} failed", flush=True)
    return 0 if f == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
