"""Test-only replay of a completed campaign run from its saved checkpoint.

Re-runs ONLY the test phase (episodes.train=0 -> modes=['test']) from a
run's saved config.yaml, so NETT loads the final checkpoint and produces the FULL
test=N episodes/condition needed for a comparable score — no retraining. Same env
count as the trained run, for score comparability (blueprint: env0 render depends on
num_envs). Launched in bulk by campaign_retest_launch.py.

Usage:
  NETT_DEVICE=0 NETT_TEST_EPS=20 python examples/campaign_retest.py <run_dir>
  <run_dir> = ~/nett_campaign/<exp>_<model>/<name>_offN_<ts>/
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: campaign_retest.py <run_dir>"); return 2
    run_dir = Path(sys.argv[1]).expanduser().resolve()
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        print(f"no config.yaml in {run_dir}"); return 2
    config = yaml.safe_load(cfg_path.read_text())

    # NETT writes resolved integers back into the output config.yaml (e.g.
    # resolved_test_num_envs); those are OUTPUT keys and fail the INPUT schema on
    # reload. Strip any top-level key the input schema doesn't accept.
    for k in [k for k in list(config) if k.startswith("resolved_")]:
        config.pop(k, None)

    # Test-only: train=0 -> _modes_from_episodes returns ['test'] -> test() loads
    # the latest checkpoint under <run_dir>/<cond>/wandb_runs/brain_i/checkpoints/.
    test_eps = int(os.environ.get("NETT_TEST_EPS", str(config.get("episodes", {}).get("test", 20))))
    config["episodes"] = {"train": 0, "test": test_eps}
    # offline wandb for unattended retest
    config.setdefault("brain", {}).setdefault("wandb", {})["mode"] = os.environ.get("NETT_WANDB_MODE", "offline")

    # Clear the PARTIAL test CSVs so the fresh full-test data is clean (an interrupted
    # run left incomplete episodes; the scorer filters those, but a clean slate avoids
    # any env_id/episode collision across the two test runs).
    cleared = 0
    for cond_dir in run_dir.iterdir():
        logs = cond_dir / "logs"
        if logs.is_dir():
            for f in logs.glob("test_*.csv"):
                f.unlink(); cleared += 1

    device = int(os.environ.get("NETT_DEVICE", "0"))
    from nett_skrl import NETT

    name = config["name"]
    out_root = str(run_dir.parent)   # NETT.run(output_path=X) -> X/<name>/
    print(f"[retest] {name} dev={device} test_eps={test_eps} cleared {cleared} old test csvs -> {out_root}")
    t0 = time.time()
    NETT(config).run(output_path=out_root, devices=[device], verbose=True)
    secs = time.time() - t0

    # ★ RE-RUN THE ANALYSIS. Without this the retest rewrote test_*.csv and left
    # analysis/summary.json describing the PREVIOUS run -- so anyone reading the summary
    # after a retest saw the OLD numbers and concluded the retest had changed nothing.
    # That happened on 2026-07-29: a stochastic-evaluation retest looked identical to the
    # deterministic baseline in all eight conditions, and only the implausibility of exact
    # 3-decimal agreement prompted a check of the file mtimes (CSV 10:30, summary 05:07).
    # Same failure class as a stale golden.json: an artifact that silently describes a
    # different run than the one just executed.
    from nett_skrl.analysis import analyze, log_analysis_to_wandb
    result = analyze(run_dir)
    log_analysis_to_wandb(run_dir, result)

    # Fail loudly if the summary is somehow still older than the data it summarises,
    # rather than leaving a stale file for the next reader to trust.
    summary = run_dir / "analysis" / "summary.json"
    if summary.is_file():
        newest_csv = max((f.stat().st_mtime for cond in run_dir.iterdir()
                          if (cond / "logs").is_dir()
                          for f in (cond / "logs").glob("test_*.csv")), default=0.0)
        if summary.stat().st_mtime < newest_csv:
            print(f"[retest] ERROR: {summary} is OLDER than the test CSVs it should "
                  f"summarise -- the analysis did not take effect."); return 1

    (run_dir / "campaign_retest_timing.json").write_text(json.dumps(
        {"name": name, "test_eps": test_eps, "retest_secs": round(secs, 1),
         "eval_stochastic": os.environ.get("NETT_EVAL_STOCHASTIC", "0"),
         "finished": datetime.now().isoformat()}, indent=2))
    print(f"[retest] DONE {name} in {secs:.1f}s -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
