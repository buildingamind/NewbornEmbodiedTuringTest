"""Scheduler for the 9-model x 3-condition campaign across the local GPUs.

Each JOB is one Isaac process pinned to a single GPU that runs a (model, condition)
via campaign_train.py (its `brains_per_process` brains as skrl agents share that one
env/GPU). The scheduler PACKS up to NETT_JOBS_PER_GPU such processes onto each GPU
(VRAM permitting; the OOM ladder below is the co-tenancy safety net) and STAGGERS
launches by NETT_STAGGER_SECS to avoid the concurrent Kit-init contention that
triggers DEVICE_LOST. Packing multiple procs/GPU is supported and safe — the old
"one job per GPU" rule was retired (see the speedup memory / the packing retraction).

Resumable: a job whose done-marker exists (<dir>/done/<job>.done, written only on
clean exit-0) is skipped. Status -> <dir>/status.json; per-job logs -> <dir>/logs/<job>.log.

OOM safety net: on OOM a job is retried one rung down a (mini_batches, max_envs)
ladder that shrinks the update batch then the env count. This is REACTIVE and does
NOT replace the measured VRAM estimator in runtime/parallel_envs.py (test sizing +
train verify/back-off) — the two are complementary and both kept.

Knobs: NETT_JOBS_PER_GPU (default 1) packing; NETT_STAGGER_SECS (default 12) launch
spacing; NETT_GPUS (default 0-7) device pool; NETT_ONLY_MODELS / NETT_ONLY_EXPERIMENTS
restrict jobs; NETT_BRAINS / NETT_TRAIN_EPS / NETT_RES forwarded to campaign_train.py;
NETT_PYTHON / NETT_CAMPAIGN_DIR override the interpreter / output dir.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Portable: default to the interpreter running this scheduler (the venv python).
PY = os.environ.get("NETT_PYTHON") or sys.executable
CAMPAIGN = Path(os.environ.get("NETT_CAMPAIGN_DIR",
                               str(Path.home() / "nett_campaign"))).expanduser()
LOGS = CAMPAIGN / "logs"
DONE = CAMPAIGN / "done"
STATUS = CAMPAIGN / "status.json"

MODELS = ["CNN", "3DCNN", "SimCLR-CLTT", "ViT", "ViT-CLTT", "ViT+VICReg",
          "ViViT", "ViViT+VICReg", "GuessWhatMoves"]
EXPERIMENTS = ["binding", "parsing", "viewinvariance"]

# OOM safety ladder (mini_batches, max_envs). rollouts=8192, steps=256, so
# scope=max_envs//8 must keep (8192//scope)%256==0 -> max_envs in {256,128,64,32}.
# Start at 256 envs (32/brain, max) to saturate the GPU; on OOM shed envs, then
# shrink the update batch (mini_batches 16->32 -> batch 512->256).
LADDER = [(16, 256), (16, 128), (16, 64), (16, 32), (32, 32)]

# Buffer placement is per-model (campaign_train.py): single-frame -> on-GPU uint8
# buffer (~6 GB) fits the renderer at 128 envs (~18.5 GB); framestack -> CPU
# buffer (its 2-frame GPU buffer OOMs even at 32 envs), so the GPU holds only the
# renderer and 128 envs fit comfortably. Both therefore start at 128 envs (rung 1).
FRAMESTACK_MODELS = {"3DCNN", "ViViT", "ViViT+VICReg", "GuessWhatMoves"}


def _slug(s: str) -> str:
    return s.replace("+", "").replace("-", "").replace(" ", "")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_jobs() -> list[dict]:
    only_m = {m.strip() for m in os.environ.get("NETT_ONLY_MODELS", "").split(",") if m.strip()}
    only_e = {e.strip() for e in os.environ.get("NETT_ONLY_EXPERIMENTS", "").split(",") if e.strip()}
    models = [m for m in MODELS if not only_m or m in only_m]
    exps = [e for e in EXPERIMENTS if not only_e or e in only_e]
    # Round-robin over experiments so the first wave exercises all design sheets.
    jobs = []
    for model in models:
        for exp in exps:
            rung = 1  # all start at 128 envs (single-frame: GPU buffer; framestack: CPU buffer)
            mb, envs = LADDER[rung]
            jobs.append({"id": f"{exp}_{_slug(model)}", "model": model,
                         "experiment": exp, "rung": rung, "mini_batches": mb, "max_envs": envs})
    return jobs


def job_oomed(job_id: str) -> bool:
    log = LOGS / f"{job_id}.log"
    if not log.exists():
        return False
    try:
        txt = log.read_text(errors="ignore")
    except OSError:
        return False
    return "out of memory" in txt or "OutOfMemoryError" in txt


# A just-freed GPU's VRAM is not released instantly (Isaac/CUDA teardown lag).
# Launching the next job onto it too soon makes NETT's pre-flight memory check
# (JobTooBigError) see <1 GB free and abort. So: (1) let a freed GPU settle before
# reusing it, and (2) treat JobTooBigError as a transient "GPU busy" condition and
# requeue the job rather than marking it permanently failed.
SETTLE_SECS = 90
MAX_TOOBIG_RETRIES = 8


def job_too_big(job_id: str) -> bool:
    log = LOGS / f"{job_id}.log"
    if not log.exists():
        return False
    try:
        txt = log.read_text(errors="ignore")
    except OSError:
        return False
    return "JobTooBigError" in txt or "Task size exceeds the free memory" in txt


def launch(job: dict, gpu: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["NETT_MODEL"] = job["model"]
    env["NETT_EXPERIMENT"] = job["experiment"]
    # Hard-pin the physical GPU: with CUDA_VISIBLE_DEVICES the process sees only
    # this GPU as cuda:0, so Isaac's RTX renderer AND torch both land on it. (The
    # AppLauncher device= arg alone did not pin the renderer — every job's
    # training piled onto one physical GPU.) NETT_DEVICE=0 -> NETT uses cuda:0.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["NETT_DEVICE"] = "0"
    env["NETT_MAX_ENVS"] = str(job["max_envs"])
    env["NETT_MINIBATCHES"] = str(job["mini_batches"])
    for k in ("NETT_ONLY_MODELS", "NETT_ONLY_EXPERIMENTS", "NETT_GPUS", "NETT_DEVICES",
              "NETT_JOBS_PER_GPU", "NETT_STAGGER_SECS"):
        env.pop(k, None)
    log = (LOGS / f"{job['id']}.log").open("w")
    log.write(f"# launch {_now()} gpu={gpu} model={job['model']} exp={job['experiment']} "
              f"mini_batches={job['mini_batches']} max_envs={job['max_envs']}\n")
    log.flush()
    p = subprocess.Popen(
        [PY, "-u", str(HERE / "campaign_train.py")],
        stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(HERE),
    )
    p._log_fh = log
    return p


def write_status(jobs, state, running):
    # running: job_id -> (gpu, proc). Multiple jobs may share a GPU (packing).
    on_gpu: dict[str, list[str]] = {}
    for jid, (g, _) in running.items():
        on_gpu.setdefault(str(g), []).append(jid)
    snap = {
        "updated": _now(),
        "totals": {
            "total": len(jobs),
            "done": sum(1 for j in jobs if state[j["id"]]["status"] == "done"),
            "failed": sum(1 for j in jobs if state[j["id"]]["status"] == "failed"),
            "running": sum(1 for j in jobs if state[j["id"]]["status"] == "running"),
            "pending": sum(1 for j in jobs if state[j["id"]]["status"] == "pending"),
        },
        "running_on_gpu": on_gpu,
        "jobs": {j["id"]: state[j["id"]] for j in jobs},
    }
    STATUS.write_text(json.dumps(snap, indent=2))


def _prebuild_frame_caches(jobs: list[dict]) -> None:
    """Serially materialize the shared frame cache for every experiment BEFORE the
    concurrent fan-out, so no training proc hits a cold cache under Kit contention.

    A cold BC7 cache built by the first of N concurrently-booting Kit procs
    deadlocks (contention on GPU/texture cache); the build itself is CPU-only, so we
    do it here, serially, up front. Idempotent (a warm cache is a fast no-op), Kit-
    free. Best-effort: on any failure we log and continue -- the per-proc build path
    still exists as the fallback, just without the serial-first guarantee.
    """
    res = int(os.environ.get("NETT_RES", "256"))
    fmt = os.environ.get("NETT_FRAME_FORMAT", "bc7").strip().lower()
    try:
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        from campaign_train import EXPERIMENTS as EXP_PATHS
        from nett_isaac.video import prebuild_frame_cache
    except Exception as exc:  # noqa: BLE001 - never let prebuild wiring break a run
        print(f"[campaign] prebuild-cache skipped (import failed: {exc})", flush=True)
        return
    seen: set[tuple[str, int]] = set()
    for exp in {j["experiment"] for j in jobs}:
        key = (exp, res)
        if key in seen or exp not in EXP_PATHS:
            continue
        seen.add(key)
        sheet, media, _imprint = EXP_PATHS[exp]
        sheet = os.environ.get("NETT_DESIGN_SHEET", sheet)
        media = os.environ.get("NETT_MEDIA_ROOT", media)
        try:
            print(f"[campaign] prebuild-cache exp={exp} res={res} fmt={fmt} ...", flush=True)
            prebuild_frame_cache(sheet, media, res, frame_format=fmt)
        except Exception as exc:  # noqa: BLE001
            print(f"[campaign] prebuild-cache {exp} FAILED ({exc}); per-proc build will handle it", flush=True)


def main() -> int:
    for d in (LOGS, DONE):
        d.mkdir(parents=True, exist_ok=True)
    gpus = [int(g) for g in os.environ.get("NETT_GPUS", "0,1,2,3,4,5,6,7").split(",")]
    jobs_per_gpu = max(1, int(os.environ.get("NETT_JOBS_PER_GPU", "1")))
    stagger = float(os.environ.get("NETT_STAGGER_SECS", "12"))
    jobs = build_jobs()
    _prebuild_frame_caches(jobs)
    state = {}
    for j in jobs:
        done_marker = DONE / f"{j['id']}.done"
        state[j["id"]] = {
            "model": j["model"], "experiment": j["experiment"],
            "status": "done" if done_marker.exists() else "pending",
            "mini_batches": j["mini_batches"], "max_envs": j["max_envs"],
            "gpu": None, "started": None, "ended": None, "returncode": None,
        }

    queue = [j for j in jobs if state[j["id"]]["status"] == "pending"]
    gpu_load: dict[int, int] = {g: 0 for g in gpus}       # running jobs per GPU
    gpu_ready_at: dict[int, float] = {g: 0.0 for g in gpus}  # earliest next-launch time
    toobig_retries: dict[str, int] = {}
    running: dict[str, tuple[int, subprocess.Popen]] = {}  # job_id -> (gpu, proc)
    last_launch = 0.0
    print(f"[campaign] {len(jobs)} jobs, {len(queue)} pending, gpus={gpus}, "
          f"{jobs_per_gpu} job(s)/GPU, stagger={stagger:g}s", flush=True)
    write_status(jobs, state, running)

    while queue or running:
        now = time.time()
        # Admit at most ONE job per iteration (stagger-gated) onto the least-loaded
        # eligible GPU — a GPU is eligible if it has a free slot AND has settled since
        # its last job ended (freed VRAM lags CUDA teardown).
        if queue and now >= last_launch + stagger:
            eligible = [g for g in gpus
                        if gpu_load[g] < jobs_per_gpu and now >= gpu_ready_at[g]]
            if eligible:
                gpu = min(eligible, key=lambda g: gpu_load[g])
                job = queue.pop(0)
                proc = launch(job, gpu)
                running[job["id"]] = (gpu, proc)
                gpu_load[gpu] += 1
                last_launch = now
                state[job["id"]].update(status="running", gpu=gpu, started=_now(),
                                        mini_batches=job["mini_batches"], max_envs=job["max_envs"])
                print(f"[campaign] start {job['id']} on gpu{gpu} (pid {proc.pid}) "
                      f"load={gpu_load[gpu]}/{jobs_per_gpu} "
                      f"mb={job['mini_batches']} envs={job['max_envs']}", flush=True)
                write_status(jobs, state, running)

        time.sleep(5)
        for jid, (gpu, proc) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            proc._log_fh.flush()
            proc._log_fh.close()
            state[jid].update(ended=_now(), returncode=rc)
            job = next(j for j in jobs if j["id"] == jid)
            if rc == 0:
                (DONE / f"{jid}.done").write_text(_now())
                state[jid]["status"] = "done"
                print(f"[campaign] DONE  {jid} (gpu{gpu})", flush=True)
            elif job["rung"] < len(LADDER) - 1 and job_oomed(jid):
                job["rung"] += 1
                job["mini_batches"], job["max_envs"] = LADDER[job["rung"]]
                state[jid].update(status="pending", gpu=None)
                queue.append(job)
                print(f"[campaign] OOM   {jid} -> retry rung {job['rung']} "
                      f"(mini_batches={job['mini_batches']}, envs={job['max_envs']})", flush=True)
            elif job_too_big(jid) and toobig_retries.get(jid, 0) < MAX_TOOBIG_RETRIES:
                toobig_retries[jid] = toobig_retries.get(jid, 0) + 1
                state[jid].update(status="pending", gpu=None)
                queue.append(job)
                print(f"[campaign] BUSY  {jid} (JobTooBig: GPU VRAM not yet free) -> "
                      f"requeue (attempt {toobig_retries[jid]}/{MAX_TOOBIG_RETRIES})", flush=True)
            else:
                state[jid]["status"] = "failed"
                print(f"[campaign] FAIL  {jid} rc={rc} (gpu{gpu})", flush=True)
            del running[jid]
            gpu_load[gpu] -= 1
            gpu_ready_at[gpu] = time.time() + SETTLE_SECS
        write_status(jobs, state, running)

    t = sum(1 for j in jobs if state[j["id"]]["status"] == "done")
    f = sum(1 for j in jobs if state[j["id"]]["status"] == "failed")
    print(f"[campaign] COMPLETE: {t} done, {f} failed of {len(jobs)}", flush=True)
    write_status(jobs, state, running)
    return 0 if f == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
