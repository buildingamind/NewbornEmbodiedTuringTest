"""Test-only replay of a completed campaign run from its saved checkpoint.

Re-runs ONLY the test phase (episodes.train=0 -> modes=['test']) from a
run's saved config.yaml, so NETT loads the final checkpoint and produces the FULL
test=N episodes/condition needed for a comparable score — no retraining. Same env
count as the trained run, for score comparability (blueprint: env0 render depends on
num_envs). Launched in bulk by campaign_retest_launch.py.

★ THIS IS ALSO THE ONLY VALID WAY TO A/B A TEST-PHASE CHANGE. Training here is NOT
reproducible at a fixed seed -- the irreducible source is the RTX-rendered observation
under concurrent training GPU load (see the determinism notes in blueprint.md), and the
resulting run-to-run spread is LARGER than most test-phase effects. Measured 2026-08-01:
comparing two full train+test runs reported a 0.181 per-condition correct_pct delta for a
change that a same-checkpoint comparison put at 0.027, because each arm had trained its
own policy (same seed, same arm, replicate train tail reward 63.16 vs 61.91; three arms at
one seed 63.16 / 73.94 / 73.68). So: retest ONE checkpoint under each arm, never train per
arm. Copy the run directory per arm first -- this script rewrites the run's test CSVs and
summary.json in place.

Usage:
  NETT_DEVICE=0 NETT_TEST_EPS=20 python examples/campaign_retest.py <run_dir>
    (NETT_TEST_EPS = repeats per test ROW; see the note at the assignment below)
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

from nett_skrl.brain.trainer import eval_stochastic_enabled


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
    #
    # ⛔ NETT_TEST_EPS IS REPEATS PER TEST ROW -- not episodes, not episodes per
    # condition. See campaign_train.py's env table for the derivation. On the
    # parsing sheet a row IS a distinct stimulus configuration, so this knob adds
    # REPLICATES and NEVER a new stimulus. Measured on the MoTok corpus: the
    # intra-cluster correlation is 0.484 and the SE of a condition mean floors at
    # 0.0960 (vs 0.1014 at 8 replicates), so <=5.3% is available from ANY episode
    # count. ⛔ And the plain chi-square applied to the pooled episodes rejects a
    # TRUE null 50.9% of the time at 8 replicates, rising to 99.6% at 280 --
    # RAISING THIS KNOB MAKES THAT TEST MORE CONFIDENTLY WRONG. If you want
    # precision, add distinct stimuli (spread brains across the sheet's 7 imprint
    # conditions: 12 -> 72 configs, same GPU), not repeats.
    test_eps = int(os.environ.get("NETT_TEST_EPS", str(config.get("episodes", {}).get("test", 20))))
    config["episodes"] = {"train": 0, "test": test_eps}
    # offline wandb for unattended retest
    config.setdefault("brain", {}).setdefault("wandb", {})["mode"] = os.environ.get("NETT_WANDB_MODE", "offline")

    # Clear the PARTIAL test CSVs so the fresh full-test data is clean (an interrupted
    # run left incomplete episodes; the scorer filters those, but a clean slate avoids
    # any env_id/episode collision across the two test runs).
    # ⛔⛔⛔ THE COMMENT ABOVE STATED A PRECONDITION THE CODE NEVER CHECKED. It says
    # "PARTIAL" and the glob deleted EVERY test_*.csv, complete ones included.
    # Measured on seat:insect 2026-09-05: MoTokSeg_off28_ship1 ran its test loop to
    # 5000/5000, wedged AFTERWARDS in teardown, and had already written a COMPLETE,
    # VALID test CSV. A later retest logged "cleared 1 old test csvs" and replaced it
    # with a degenerate pose00-only set.
    # ⇒ THE WEDGE DESTROYED NOTHING; THE RECOVERY DESTROYED THE DATA.
    # Independently confirmed by seat:insect's roster header: with zero test CSVs left,
    # guards.assert_complete_arm returns None = SKIP rather than a failure, so a complete
    # arm that loses its certificates is INDISTINGUISHABLE from one that never reached
    # test -- no raise, no warning, only a lower count.
    #
    # ⛔⛔ v1 OF THIS FIX WAS A GUARD THAT DEFAULTED OFF, WHICH IS NOT A GUARD. It read
    #   _expect = int(os.environ.get("NETT_RETEST_EXPECT_ROWS", "0")) or None
    #   if not _force and _expect is not None:
    # With the variable UNSET -- the normal case, and the case in which the loss above
    # actually happened -- _expect is None, the check is skipped, and the unconditional
    # delete proceeds unchanged. It protected the corpus only for an operator who
    # already knew to protect it. It also compared against an operator-supplied
    # constant, when 560,000 is a *parsing* constant: a ViewInvariance arm is complete
    # at 500,000 and would have been deleted as "partial" by the guard meant to save it.
    #
    # ⇒ v2 REMOVES THE CLASS INSTEAD OF CHECKING FOR IT: NEVER DELETE, RENAME.
    # No threshold to get wrong, no environment variable to forget, no per-experiment
    # row count to derive. The stated goal is met exactly -- the renamed file no longer
    # matches test_*.csv, so the fresh run cannot collide on env_id/episode -- and the
    # bytes survive. A DESIGN THAT CANNOT EXPRESS THE BAD OUTCOME BEATS A CHECK THAT
    # DETECTS IT.
    _stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    cleared = 0
    preserved = []
    for cond_dir in run_dir.iterdir():
        logs = cond_dir / "logs"
        if logs.is_dir():
            for f in sorted(logs.glob("test_*.csv")):
                dest = f.with_name(f"{f.name}.superseded_{_stamp}")
                _n = 0
                while dest.exists():          # never overwrite a previous rescue either
                    _n += 1
                    dest = f.with_name(f"{f.name}.superseded_{_stamp}.{_n}")
                f.rename(dest)
                cleared += 1
                preserved.append(dest)
    if preserved:
        print(f"[retest] PRESERVED {len(preserved)} prior test CSV(s) by RENAME, not "
              f"deleted. A teardown-window wedge leaves VALID artifacts on disk, so "
              f"\"this arm wedged\" does not imply \"this arm's data is bad\":")
        for _d in preserved:
            print(f"[retest]   -> {_d}")

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
         # ⚠ THROUGH THE ACCESSOR, never a second os.environ.get with its own default.
         # This line used to hardcode "0" and would now record the OPPOSITE of the protocol
         # it actually ran under -- precisely the provenance failure that made the 2026-07-29
         # stochastic retest uncomparable to the deterministic summary it was read against.
         "eval_stochastic": eval_stochastic_enabled(),
         "finished": datetime.now().isoformat()}, indent=2))
    print(f"[retest] DONE {name} in {secs:.1f}s -> {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
