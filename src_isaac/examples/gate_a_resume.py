#!/usr/bin/env python3
"""Re-run Gate A from a TRAINED checkpoint, in the regime the gate was calibrated on.

    examples/gate_a_resume.py --from-run <archived arm dir> --model <MODELS key> \\
        [--updates 4] [--device 0] [--out-root ~/nett_gate_a] [--dry-run]

⛔ WHY A RESUME AND NOT A SMOKE. The transit mask corrects a pathology of a TRAINED agent:
the closeness reward makes it approach fast and then freeze, so 81.1% of a trained rollout's
steps are parked (`notes/researcher/the-parked-agent.md`, band 0.599-0.898 over 35 arms) and
63.3% of PPO updates draw a window with no transit motion at all. The first Gate A smoke ran
a FRESH policy, whose parked share was 0.0127 -- and a sampler that recovers degraded windows
cannot show a gain where the windows were never degraded. That smoke's near-null was not a
weak result; it was a result about nothing, and it read exactly like a real one. This tool
exists so the reference has headroom: it starts from weights that already park.

⛔ AND THE PARKED FRACTION IS MEASURED ON THE RESUMED POLICY'S OWN ROLLOUT, never inherited
from the source arm. A source arm's parked share is evidence about the source arm. Whether
the transplant landed -- right weights, right architecture, right env -- is a property of
THIS run, and the only way to know is to measure THIS run. So the seeded checkpoints go in
without their logs, and `tools/gate_a.py` reads the test CSV this run wrote.

## The four ways a resume fails silently, and what stops each

1. **No checkpoint at the source.** `load_latest_checkpoints` logs "no checkpoint found" and
   trains from random weights; nothing raises and the exit code is 0. -> Refused at preflight,
   before a card is spent.
2. **Architecture mismatch.** `agent.load` raises inside a `try/except Exception` that logs
   and continues -- again from random weights, again exit 0. -> Refused at preflight by
   comparing `MODELS[src]` and `MODELS[dst]` on `encoder`/`cfg`/`framestack`, which is what
   actually decides whether the state dict fits, rather than comparing the model NAMES.
3. **The seed lost a race with the run directory.** -> `NETT_RUN_NAME` pins the directory so
   the checkpoints are in place before the process starts, with no guessing and no race.
4. **It loaded, and the policy still is not in regime** (a partially-trained source, a
   different chamber). -> Not preventable here, and not this tool's call: `tools/gate_a.py`
   refuses a verdict below --min-parked and prints the measured share with its definition
   and its n either way.

## The optimizer is deliberately NOT restored

The checkpoint carries `policy`, `value`, `optimizer`, `value_preprocessor` -- and the
optimizer's second param group is the AUX HEAD's, whose size differs between aux kinds
(cltt's 4 tensors vs another loss's). Loading it across kinds raises mid-way through
`agent.load`, after `policy` has already been assigned, leaving a half-loaded agent that no
exit code distinguishes from a whole one. Gate A reads a diagnostic of WHICH WINDOWS ARE
SAMPLED, which depends on the policy's behaviour and not on Adam's moments, so this tool
writes a checkpoint containing only the modules it needs and drops the optimizer. skrl's
`load` iterates the FILE's keys (`agents/torch/base.py`), so an absent key is simply not
loaded -- no error, and here no loss.

⚠ That makes the first update after the resume take a fresh-moment step. It is a real
difference from an uninterrupted run and it is recorded in `resume_manifest.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from campaign_train import MODELS, _slug  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# Modules a resumed policy needs. `optimizer` is excluded on purpose -- see the docstring.
RESUME_MODULES = ("policy", "value", "value_preprocessor")
# The three fields that decide whether a state dict fits. Two models with different names
# but these three equal are checkpoint-compatible; two with the same name and any of these
# different are not. Comparing names would answer a question nobody asked.
SHAPE_FIELDS = ("encoder", "cfg", "framestack")


def find_gate_a() -> Path | None:
    """Locate tools/gate_a.py in the workspace clone, if it is on this host."""
    for cand in (
        Path(os.environ.get("NETT_WORKSPACE", "")) / "tools" / "gate_a.py",
        Path.home() / "code" / "isaac" / "NETT_Global_Workspace" / "tools" / "gate_a.py",
    ):
        if cand.is_file():
            return cand
    return None


def source_checkpoints(from_run: Path) -> dict[int, Path]:
    """{brain_index (1-based): newest checkpoint} under an archived arm."""
    out: dict[int, Path] = {}
    for ck_dir in sorted(from_run.glob("**/wandb_runs/brain_*/checkpoints")):
        try:
            idx = int(ck_dir.parent.name.removeprefix("brain_"))
        except ValueError:
            continue
        final = ck_dir / "final_agent.pt"
        numbered = sorted(
            (int(p.stem.removeprefix("agent_")), p)
            for p in ck_dir.glob("agent_*.pt")
            if p.stem.removeprefix("agent_").isdigit()
        )
        # final_agent.pt is the end of training; a numbered snapshot is a point inside it.
        # Prefer final, because "trained" is the property this whole tool needs.
        chosen = final if final.exists() else (numbered[-1][1] if numbered else None)
        if chosen is not None:
            out[idx] = chosen
    return out


def compatible(src_model: str, dst_model: str) -> tuple[bool, str]:
    """Do these two MODELS entries produce a policy the other's state dict fits?"""
    if src_model not in MODELS:
        return False, f"source model {src_model!r} is not in MODELS"
    if dst_model not in MODELS:
        return False, f"target model {dst_model!r} is not in MODELS"
    a, b = MODELS[src_model], MODELS[dst_model]
    diff = [f for f in SHAPE_FIELDS if a.get(f) != b.get(f)]
    if diff:
        return False, (
            f"{src_model} and {dst_model} differ on {diff} -- the policy state dict will not "
            f"fit. `agent.load` would raise inside load_latest_checkpoints' except-and-log, "
            f"and the run would train from RANDOM WEIGHTS and exit 0."
        )
    return True, f"{src_model} -> {dst_model}: {list(SHAPE_FIELDS)} all equal"


def seed_checkpoints(src: dict[int, Path], dest_cfg_path: Path, *, dry_run: bool) -> list[dict]:
    """Write policy/value-only checkpoints into the destination's skrl layout."""
    import torch

    manifest = []
    for idx, path in sorted(src.items()):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(blob, dict):
            raise SystemExit(f"⛔ {path} is not a skrl checkpoint dict ({type(blob)})")
        kept = {k: v for k, v in blob.items() if k in RESUME_MODULES}
        missing = [m for m in ("policy", "value") if m not in kept]
        if missing:
            raise SystemExit(
                f"⛔ {path} carries no {missing} -- there is nothing to resume. Keys: "
                f"{sorted(blob)}"
            )
        target = dest_cfg_path / "wandb_runs" / f"brain_{idx}" / "checkpoints"
        manifest.append({
            "brain": idx,
            "source": str(path),
            "source_keys": sorted(blob),
            "written_keys": sorted(kept),
            "dropped_keys": sorted(set(blob) - set(kept)),
            "target": str(target / "final_agent.pt"),
        })
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)
            torch.save(kept, target / "final_agent.pt")
    return manifest


def train_eps_for(updates: int, envs_per_brain: int) -> int:
    """Episodes needed for `updates` PPO updates.

    One update per `rollouts // envs_per_brain` = 500 iterations = one episode per env
    (campaign_train pins rollouts=8000 over 16 envs/brain), and train_timesteps divides
    the episode budget by envs_per_brain -- so updates = train_eps // envs_per_brain.
    """
    return updates * envs_per_brain


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--from-run", required=True, type=Path,
                    help="archived arm directory containing <imprint>/wandb_runs/brain_*/")
    ap.add_argument("--from-model", default=None,
                    help="MODELS key the source was trained with (default: inferred from "
                         "the arm directory name)")
    ap.add_argument("--model", required=True, help="MODELS key to resume INTO")
    ap.add_argument("--updates", type=int, default=4,
                    help="PPO updates to run. ⚠ Gate A's kill branch needs >=2 readings per "
                         "brain: on one point 'flat-or-declining' is undefined, not unmet.")
    ap.add_argument("--experiment", default="parsing")
    ap.add_argument("--imprint", default=None, help="default: inferred from the source arm")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--brains", type=int, default=7)
    ap.add_argument("--max-envs", type=int, default=112)
    ap.add_argument("--test-eps", type=int, default=4,
                    help="repeats per test row; the parked fraction is measured on this")
    ap.add_argument("--out-root", type=Path, default=Path("~/nett_gate_a"))
    ap.add_argument("--min-parked", type=float, default=0.50)
    ap.add_argument("--threshold", type=float, default=1.10)
    ap.add_argument("--allow-cross-model", action="store_true",
                    help="skip the architecture preflight. Only meaningful if you have "
                         "checked the state dicts by hand.")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight and print the plan; touch nothing, launch nothing")
    return ap


def main() -> int:
    a = build_parser().parse_args()

    from_run = a.from_run.expanduser().resolve()
    if not from_run.is_dir():
        print(f"⛔ --from-run is not a directory: {from_run}", file=sys.stderr)
        return 3

    # --- preflight 1: is there anything to resume FROM? ---------------------------------
    src = source_checkpoints(from_run)
    if not src:
        print(f"⛔ no checkpoints under {from_run}/**/wandb_runs/brain_*/checkpoints.\n"
              f"   Refusing here rather than letting the run start: load_latest_checkpoints "
              f"logs 'no checkpoint found' and trains from RANDOM WEIGHTS, and the run then "
              f"exits 0 having measured nothing this gate can use.", file=sys.stderr)
        return 3

    # --- preflight 2: will the state dict fit? -----------------------------------------
    src_model = a.from_model
    if src_model is None:
        # Arm dirs are named <slug(model)>_<exp>_<imprint>_off<N>_<stamp>; match the longest
        # MODELS key whose slug is the name's prefix, so ViTCLTT does not shadow ViTCLTT2F.
        stem = from_run.name
        cands = [m for m in MODELS if stem.startswith(_slug(m))]
        src_model = max(cands, key=len) if cands else None
    if src_model is None:
        print(f"⛔ could not infer the source model from {from_run.name!r}; pass --from-model.",
              file=sys.stderr)
        return 3

    # --- preflight 0: can Kit even boot in this process's environment? -----------------
    # ⛔ FAILS IN 0s HERE INSTEAD OF 9s AFTER ISAAC STARTS, AND SAYS WHY. Kit bootstraps at
    # import and, with the licence unaccepted, calls input() for "Do you accept the EULA?".
    # Under nohup/CI there is no stdin, so that raises EOFError, the bootstrap SystemExits,
    # nett.py reports "Task validation failed (exit code 1)", and the actual cause is 40
    # lines up a traceback about something else entirely. Measured 2026-09-10 -- it cost two
    # launches. `scripts/hooks/pre-push` exports this variable for the same reason.
    if not os.environ.get("OMNI_KIT_ACCEPT_EULA") and not sys.stdin.isatty():
        print("⛔ OMNI_KIT_ACCEPT_EULA is unset and there is no tty to answer the licence "
              "prompt on. Kit's import-time bootstrap will read stdin, get EOF, and exit -- "
              "which surfaces as 'Task validation failed (exit code 1)' with the real cause "
              "buried in a subprocess traceback. Re-invoke with OMNI_KIT_ACCEPT_EULA=YES.",
              file=sys.stderr)
        return 3

    ok, why = compatible(src_model, a.model)
    print(f"preflight: {why}")
    sys.stdout.flush()   # stderr is unbuffered: without this the refusal prints ABOVE the
                         # line it is refusing on, which reads as unsupported.
    if not ok and not a.allow_cross_model:
        print(f"⛔ REFUSING: {why}", file=sys.stderr)
        return 3
    if not ok:
        print("⚠ --allow-cross-model: proceeding over the preflight's objection. If the load "
              "fails it will be logged, not raised, and the run will train from random "
              "weights while exiting 0. Read the run log for 'failed to load'.")

    imprint = a.imprint or next(
        (p.parent.name for p in from_run.glob("*/wandb_runs")), "fork-1")
    envs_per_brain = a.max_envs // a.brains
    train_eps = train_eps_for(a.updates, envs_per_brain)
    stamp = datetime.now().strftime("%m%d_%H%M%S")
    run_name = f"gateA_{_slug(a.model)}_{imprint}_{stamp}"[:63]
    out_root = a.out_root.expanduser()
    run_dir = out_root / f"{a.experiment}_{_slug(a.model)}" / run_name
    cfg_path = run_dir / imprint

    print(f"  source arm   : {from_run}")
    print(f"  source model : {src_model}  ({len(src)} brain checkpoint(s))")
    print(f"  target model : {a.model}   imprint {imprint}")
    print(f"  updates      : {a.updates}  -> NETT_TRAIN_EPS={train_eps} "
          f"({a.updates} x {envs_per_brain} envs/brain)")
    print(f"  run dir      : {run_dir}")
    if a.updates < 2:
        print("⚠ --updates 1 yields ONE reading per brain. Gate A will report the ratio but "
              "will decline the kill branch: 'flat-or-declining' is undefined on one point.")

    manifest = seed_checkpoints(src, cfg_path, dry_run=a.dry_run)
    print(f"  seeded       : {len(manifest)} checkpoint(s), keys "
          f"{manifest[0]['written_keys']} (dropped {manifest[0]['dropped_keys']})")

    env = dict(os.environ)
    env.update({
        "NETT_MODEL": a.model,
        "NETT_EXPERIMENT": a.experiment,
        "NETT_IMPRINT": imprint,
        "NETT_DEVICE": str(a.device),
        "NETT_BRAINS": str(a.brains),
        "NETT_MAX_ENVS": str(a.max_envs),
        "NETT_TRAIN_EPS": str(train_eps),
        "NETT_TEST_EPS": str(a.test_eps),
        "NETT_OUT_ROOT": str(out_root),
        "NETT_RUN_NAME": run_name,
        # The two knobs Gate A is actually about. Without the diag there is no
        # `Loss / Aux inv ratio` at all and gate_a.py refuses for want of a series --
        # a refusal that would be about this line, not about the mask.
        "NETT_AUX_TRANSIT_MASK": "1",
        "NETT_AUX_VICREG_TT_DIAG": "1",
    })

    if not a.dry_run:
        (run_dir / "resume_manifest.json").write_text(json.dumps({
            "created": datetime.now().isoformat(),
            "from_run": str(from_run),
            "from_model": src_model,
            "to_model": a.model,
            "imprint": imprint,
            "updates_requested": a.updates,
            "train_eps": train_eps,
            "envs_per_brain": envs_per_brain,
            "optimizer_restored": False,
            "optimizer_note": "dropped on purpose: its aux-head param group differs by aux "
                              "kind and loading it across kinds raises mid-way through "
                              "agent.load, after policy has already been assigned",
            "checkpoints": manifest,
            "env": {k: v for k, v in env.items() if k.startswith("NETT_")},
        }, indent=2))

    cmd = [sys.executable, str(HERE / "campaign_train.py")]
    print(f"\n$ NETT_RUN_NAME={run_name} NETT_AUX_TRANSIT_MASK=1 ... {' '.join(cmd)}")
    if a.dry_run:
        print("--dry-run: stopping here. Nothing was written, nothing was launched.")
        return 0

    # Tee, because the load evidence is on stdout and nowhere else: campaign_train logs to
    # the console, so a run whose output is not captured leaves NO record of whether the
    # checkpoints loaded, and the question would be unanswerable after the fact.
    t0 = time.time()
    log_path = run_dir / "gate_a_resume.log"
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(cmd, env=env, cwd=str(REPO),
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log_fh.write(line)
        rc = proc.wait()
    print(f"training+test finished in {time.time() - t0:.0f}s (rc={rc})  log {log_path}")
    if rc != 0:
        print(f"⛔ campaign_train exited {rc}; not reading a gate from a run that failed.",
              file=sys.stderr)
        return rc

    # --- did the resume actually land? --------------------------------------------------
    # ⛔ COUNT THE SUCCESSES, DO NOT SCAN FOR THE FAILURE. "no 'failed to load' in the log"
    # is satisfied by an empty log, a renamed message, and a run that never reached the
    # loader -- all of which read as a clean load. `load_latest_checkpoints` emits one
    # "train: brain_N loaded ..." per brain it actually restored, so the number of those
    # lines is a positive count with a known expected value.
    text = log_path.read_text(errors="replace")
    loaded = len(re.findall(r"train: brain_\d+ loaded ", text))
    print(f"  checkpoints loaded: {loaded}/{len(manifest)} brains (counted in {log_path.name})")
    if loaded != len(manifest):
        print(f"⛔ {len(manifest) - loaded} brain(s) did NOT load a checkpoint. Whatever the "
              f"gate reads next is partly or wholly about RANDOM WEIGHTS -- and the parked "
              f"fraction cannot tell you that, because an untrained policy and an unparked "
              f"one give the same number.", file=sys.stderr)

    # --- hand the verdict to the gate ---------------------------------------------------
    gate = find_gate_a()
    if gate is None:
        print(f"\n⚠ tools/gate_a.py not found on this host; the run is complete at {run_dir}. "
              f"Set NETT_WORKSPACE or run the gate by hand.")
        return 0
    # ⛔ SUBPROCESS, NEVER `import`. `tools/queue.py` shadows the stdlib `queue` for anything
    # whose sys.path[0] is tools/, and the breakage surfaces far from the cause:
    # tensorboard -> boto3 -> botocore -> urllib3 -> queue.LifoQueue, as an AttributeError
    # that never mentions tools/. gate_a.py scrubs its own directory from sys.path at import;
    # running it as its own process keeps that guarantee whole.
    print(f"\n$ {gate} {run_dir}")
    return subprocess.run(
        [sys.executable, str(gate), str(run_dir),
         "--threshold", str(a.threshold), "--min-parked", str(a.min_parked)]
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
