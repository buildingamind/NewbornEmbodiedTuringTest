"""Capture ON-DISTRIBUTION observations from a trained run's test phase.

Phase 14/15 probed frozen encoders with the monitors' own video frames, because
that needs no Kit boot. It leaves one caveat standing: those are not what the agent
sees. The chamber view is smaller, off-centre, lit, DLSS-reconstructed, and often
shows one monitor rather than two. This driver removes that caveat by capturing the
policy's ACTUAL observation tensors during a normal test phase.

⚠⚠ STATUS: BUILT AND DEBUGGED, NEVER COMPLETED A FULL-SCHEDULE RUN. The largest capture that
finished was a smoke of 2352 frames, whose labels joined at 92.9% with exact accounting; no
full-schedule capture has ever run to completion. Two real bugs were fixed on 2026-08-12
AFTER that smoke and are therefore UNVERIFIED BY ANY RUN: it transferred the whole batch
GPU->CPU every step, and it never tore Kit down (three orphaned processes were found holding
GPU 0, one of them for two days -- check `nvidia-smi` after the first real run, and kill
leftovers before launching anything else). Treat the first full run as a debugging session,
not as data collection.

★ IT MODIFIES NO RUNTIME CODE. `Body.embed` returns the fully wrapped body env --
already CHW, already the encoder's input -- and `Brain.test` accepts any env. So a
pass-through `gym.Wrapper` slipped between the two sees exactly what the encoder sees,
while the standard eval path (checkpoint loading, action selection, CSV writing)
runs untouched. Nothing here is reachable from a training run.

Episode bookkeeping is reproduced the way `ChannelsFirst` does it (per-env episode
counter incremented on terminated|truncated), so every frame carries
``(env_id, episode, step)``. Those are exactly the keys repoA's `log_channel.py`
writes into `test_*.csv` alongside `left.monitor` / `right.monitor` /
`correct.monitor`, so LABELS ARE JOINED POST HOC and never guessed here.

⚠ Sub-sampling is mandatory, not an optimisation: a full test phase is
episodes x steps x envs frames, which at 112 envs is hundreds of GB. ``--every``
and ``--max-frames`` bound it; the cap is reported so a truncated capture can never
be mistaken for a complete one.

★★ ``--episodes`` MUST MATCH THE ORIGINAL RUN'S TEST COUNT. Subsample with
``--every`` / ``--max-frames``, never by shortening the schedule. The test schedule is
ORDERED -- conditions are grouped and the target-left design rows come first -- so a
short capture is a biased PREFIX, not a sample. Measured 2026-08-11: ``--episodes 2``
ran 104 of the run's 1040 episodes and produced 104 target-left episodes and 0
target-right, covering 4 of the 8 conditions with neither `binding` nor any `2*`
condition present. The side probe cannot even be defined on that set.
``campaign/logs/l5b/join_capture_labels.py`` refuses such a capture.

    cd NewbornEmbodiedTuringTest/src_isaac
    NETT_DEVICE=0 PYTHONPATH=. python examples/capture_observations.py <run_dir> \
        --out ~/nett_obs_capture --episodes 4 --every 20 --max-frames 6000

★★ FOR ANY TEMPORAL OR ACTION-CONDITIONED OBJECTIVE, PASS ``--window``. Added
2026-09-09. The default ``--window 1`` is the original strided capture and keeps
frames 0, 20, 40 ... -- **no two of them adjacent**, so ``(obs_t, a_t, obs_t+1)``
exists NOWHERE in the file and SPR / TACO / any temporal loss has nothing to train
on. It does not error; the file simply contains no transitions. ``--window 8``
keeps 8 CONSECUTIVE frames at each sampling point instead:

    ... --every 64 --window 8 --max-frames 6000     # 8-frame runs, 1 per 64 steps

The saved npz now carries ``actions`` and ``action_keys`` beside ``obs``/``keys``,
plus ``window``, ``every`` and ``n_transition_pairs``. The driver PRINTS the usable
pair count and warns loudly at zero, so a capture that cannot serve its purpose
says so at the end of the run rather than at the start of the next project.

★★ AND IT RECORDS WHETHER IT IS A PREFIX. ``episodes_requested``,
``episodes_seen``, ``episodes_source`` and ``is_prefix`` are saved too, because a
short capture is **well-formed, has a non-zero pair count, and is unusable** --
the ordered schedule means it covers some conditions and not others, and nothing
in the file used to say so. ⚠ A crash or kill is NOT this failure mode: the npz
is written only after ``brain.test()`` returns, so an interrupted capture leaves
no file at all. The cases that DO produce a misleading file are an ``--episodes``
that does not match the source run, a schedule that ends early, and
``--max-frames``. All three set ``is_prefix``.

⚠ ACTION ALIGNMENT. An action is attached to the frame it ACTS ON, never the frame
it produces. Getting this backwards shifts every action by one step, does not crash,
and trains a predictor on the action that FOLLOWED its target. The bookkeeping lives
in ``FrameAlignment`` at module level precisely so it can be unit-tested without
booting Kit -- see ``tests/test_capture_alignment.py``. A transition never spans a
reset.

⚠ POSE IS NOT CAPTURED HERE, AND DOES NOT NEED TO BE. ``agent.x`` / ``agent.angle``
join from the run's own ``test_*.csv`` on ``(env_id, episode, step)`` -- the same
keys this file writes. That is how the transit mask (81.1% of steps are parked;
``NETT_AUX_TRANSIT_MASK``) is reproduced offline against a replayed capture.

⚠ Do NOT set an outer CUDA_VISIBLE_DEVICES: `runtime/device.py::visible_device_scope`
overwrites it from NETT_DEVICE, and an outer pin puts every agent on card 0.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Pure per-env bookkeeping, module level and dependency-free ON PURPOSE.
#
# The Capture wrapper lives inside main() behind lazy torch/gym imports so that
# nothing here can be reached before Kit is configured. That makes the wrapper
# itself untestable without booting a simulator. The part that can be WRONG in a
# way no reader would notice -- which frame an action is attached to -- is
# therefore extracted here, where it can be tested with plain integers.
#
# ⚠ Why this matters more than usual: this driver has never completed a
# full-schedule run (see the module docstring), and a one-step action
# misalignment does not crash, does not look wrong in the file, and trains an
# action-conditioned predictor on the action that FOLLOWED its target.
# ---------------------------------------------------------------------------


def expected_episodes_per_env(rows: int, episodes_per_row: int, num_envs: int) -> int:
    """Episodes ONE env runs, from the two quantities the env is actually sized by.

    ⛔ THE UNIT CONVERSION THAT WAS MISSING. `episodes.test` in the config is episodes
    PER DESIGN ROW; `FrameAlignment.on_done` counts episodes PER ENV. environment.py:366
    multiplies rows x episodes_per_row into the GLOBAL `test_total_episodes`, and the env
    spreads that over `num_envs`. Comparing the per-env count against the per-row one
    stamped `is_prefix=True` on a capture that had reproduced its source run step for
    step -- 560,001 log rows in each -- and cost a peer a message asking why "only HALF
    the schedule ran". Nothing had gone wrong; the comparison had two units.

        56 rows x 20 per row = 1120 global / 112 envs = 10 per env
    """
    if rows <= 0 or episodes_per_row <= 0 or num_envs <= 0:
        return 0
    return (rows * episodes_per_row) // num_envs


def _design_coverage(logs_dir):
    """(distinct design rows reached, set of target sides) from a run's own test log.

    ⛔ THE ROW AXIS HAS NO OTHER WITNESS. A capture whose test phase visited 8 of 56
    design rows finishes cleanly, writes every frame it was asked for, and reports a
    healthy transition count -- the truncation is invisible in the npz because the npz
    has no column for it. The per-step log DOES, in `test.cond` + the two monitor clips,
    so the coverage is read back out of the artefact the run itself wrote.

    A design row is the (condition, left clip, right clip, correct side) tuple; `Rest`
    is included because it is a row of the sheet like any other. Returns (0, set()) when
    no log is found, so a caller can tell "not measured" from "measured as zero".
    """
    import csv as _csv
    from pathlib import Path as _Path
    logs = sorted(_Path(logs_dir).glob("test_*.csv"))
    if not logs:
        return 0, set()
    rows, sides = set(), set()
    with open(logs[0], newline="") as fh:
        rdr = _csv.DictReader(fh)
        need = {"test.cond", "left.monitor", "right.monitor", "correct.monitor"}
        if not need.issubset(rdr.fieldnames or ()):
            # Read by NAME or not at all -- a positional read of a second schema is how
            # an earlier analysis in this campaign got every implied n wrong silently.
            return 0, set()
        for r in rdr:
            rows.add((r["test.cond"], r["left.monitor"],
                      r["right.monitor"], r["correct.monitor"]))
            sides.add(r["correct.monitor"])
    return len(rows), sides


def resolve_num_envs(env, declared=None) -> int:
    """How many envs the capture will iterate -- resolved LOUDLY, never defaulted.

    ⛔ MEASURED 2026-09-11, AND IT COST A WHOLE CAPTURE. This was
    ``int(getattr(env, "num_envs", 1))``. ``gymnasium.Wrapper`` DROPPED attribute
    forwarding in 1.0 (``"__getattr__" in vars(gym.Wrapper)`` is False on the 1.2.0 in
    this venv), and the env handed to the wrapper is the terminal ``ChannelsFirst``
    wrapper, so the lookup missed and **the default 1 won silently**. The capture then
    recorded env 0 and nothing else: 10 episodes out of 1,120, **7 of the 56 design rows**,
    and 0 scorable episodes for the readout -- from a 49-minute run whose own log holds
    all 112 envs and all 56 rows.

    ⛔ THE DEFAULT IS THE DEFECT, NOT THE LOOKUP. A missing attribute and a genuine
    ``num_envs=1`` produced the same number and no message, so nothing downstream could
    tell a single-env capture from a 112-env capture that lost 111 of them.

    ⛔ AND THE RIGHT NUMBER WAS PRINTED ON THE NEXT LINE. The driver logged
    ``envs={run_config.num_envs}`` -- 112 -- immediately after constructing a wrapper that
    believed 1. A value printed BESIDE the computation is not the value used BY it, which
    is why ``declared`` is now an ARGUMENT and the printed line reads it back from here.

    Resolution order: the caller's declared count (from ``run_config``), cross-checked
    against the env itself when the env can be asked; otherwise whatever the env reports.
    A disagreement and a total absence are both refusals.
    """
    found = None
    for probe in (lambda: env.get_wrapper_attr("num_envs"),
                  lambda: env.unwrapped.num_envs,
                  lambda: env.num_envs):
        try:
            found = int(probe())
            break
        except Exception:
            continue
    if declared is None and found is None:
        raise SystemExit(
            "[capture] cannot determine num_envs: the env exposes none and the caller "
            "declared none. Refusing to default to 1 -- that default silently recorded "
            "1/112 of a capture on 2026-09-11.")
    if declared is None:
        return found
    declared = int(declared)
    if found is not None and found != declared:
        raise SystemExit(
            f"[capture] num_envs disagreement: caller declared {declared}, env reports "
            f"{found}. One of them describes a different run; refusing to guess.")
    return declared


class FrameAlignment:
    """Decides which frames to keep and which frame each action acts on.

    Contract, per env, within one episode:
      * frames are indexed 0, 1, 2 ... from the reset observation;
      * a frame is kept iff ``(index % every) < window``;
      * the action passed to ``step()`` acts on the LAST RECORDED frame, so it is
        attached to that frame's key -- never to the frame the step produces.
    """

    def __init__(self, num_envs: int, every: int, window: int = 1) -> None:
        self.num_envs = int(num_envs)
        self.every = int(every)
        self.window = max(1, int(window))
        if self.window > self.every:
            raise ValueError(
                f"window {self.window} > every {self.every}: windows would overlap "
                "and the capture would be a contiguous run, not a sample."
            )
        self._episode: dict[int, int] = {}
        self._step: dict[int, int] = {}
        self._last_key: dict[int, tuple[int, int, int] | None] = {}

    # -- queries ------------------------------------------------------------
    def steps(self) -> dict[int, int]:
        return {e: self._step.get(e, 0) for e in range(self.num_envs)}

    def wanted(self) -> list[int]:
        return [e for e, s in self.steps().items() if (s % self.every) < self.window]

    def key(self, env_id: int) -> tuple[int, int, int]:
        return (env_id, self._episode.get(env_id, 0), self._step.get(env_id, 0))

    def action_targets(self) -> list[tuple[int, tuple[int, int, int]]]:
        """(env_id, frame_key) for every env whose last frame was kept."""
        return [(e, k) for e, k in self._last_key.items() if k is not None]

    # -- transitions --------------------------------------------------------
    def begin_record(self) -> None:
        """Default every env to 'last frame not kept' before a record pass."""
        for e in range(self.num_envs):
            self._last_key[e] = None

    def mark_kept(self, env_id: int, key: tuple[int, int, int]) -> None:
        self._last_key[env_id] = key

    def advance(self) -> None:
        for e in range(self.num_envs):
            self._step[e] = self._step.get(e, 0) + 1

    def on_done(self, env_id: int) -> None:
        """A transition never spans a reset."""
        self._episode[env_id] = self._episode.get(env_id, 0) + 1
        self._step[env_id] = 0
        self._last_key[env_id] = None

    def on_reset(self) -> None:
        self._step.clear()
        self._last_key.clear()


import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml


def _load_config(run_dir: Path, episodes: int) -> dict:
    """A run's saved config, made test-only -- the campaign_retest contract."""
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        raise SystemExit(f"no config.yaml in {run_dir}")
    config = yaml.safe_load(cfg_path.read_text())
    # ★ READ `resolved_test_num_envs` BEFORE STRIPPING IT. NETT writes the resolved
    # integers back into the OUTPUT config (they fail the INPUT schema, hence the strip),
    # and `environment.num_envs` is None in a campaign config -- the count comes from
    # `max_parallel_envs` resolved per condition. Dropping it first leaves num_envs=1,
    # which fails the divisible-by-num_brains check. It also matters scientifically:
    # env0's render depends on num_envs, so a capture at a different count is not the
    # same observation distribution the run was scored on.
    config["_resolved_test_num_envs"] = dict(config.get("resolved_test_num_envs") or {})
    for k in [k for k in list(config) if k.startswith("resolved_")]:
        config.pop(k, None)
    # ★ RECORD THE SOURCE RUN'S TEST COUNT BEFORE OVERWRITING IT. Without this the
    # capture cannot say whether `--episodes` matched the run it came from, and a
    # SHORT capture is a biased PREFIX rather than a sample -- the test schedule is
    # ORDERED, so conditions are grouped and target-left design rows come first.
    # Measured 2026-08-11: `--episodes 2` ran 104 of 1040 episodes and produced 104
    # target-left and 0 target-right. That file is well-formed, has a non-zero pair
    # count, and is unusable; nothing in it said so until now.
    _src = (config.get("episodes") or {}).get("test")
    config["_source_test_episodes"] = int(_src) if _src is not None else None
    config["episodes"] = {"train": 0, "test": int(episodes)}
    config.setdefault("brain", {}).setdefault("wandb", {})["mode"] = "disabled"
    return config


def _warn_about_card_occupancy() -> None:
    """Say, before Kit boots, that this job will make every card look BUSY.

    ⛔ MEASURED 2026-09-10 on chicken. A capture pinned by NETT_DEVICE=0 put its work on
    card 0 (11,274 MiB) and still held a 92 MiB context on cards 1, 2 and 3. That is
    nothing in memory terms and total in scheduling terms: the workspace repo's
    `tools/launch_arm.sh:168` reads
    `if [ "$BUSY" -ne 0 ] || [ "$USED" -ge 1500 ]` -- an OR, so a card is refused on a
    non-zero process count REGARDLESS of MiB. Verified by reading that line, not by
    inference from the symptom. A single
    unpinned-parent capture made all four cards unlaunchable and chicken sat idle behind
    one diagnostic job.

    ⇒ The cost of not pinning is not this job's card. It is every card.

    ⚠ THIS ONLY WARNS. The obvious remedy -- an outer CUDA_VISIBLE_DEVICES -- is the thing
    the module docstring tells you not to do, because `runtime/device.py::
    visible_device_scope` overwrites it from NETT_DEVICE and an outer pin puts every agent
    on card 0. That warning was written for the multi-task launcher, and this driver runs
    exactly one task, so it may well be safe here -- but "may well be" is not a finding,
    and testing it costs a 3.5-hour capture. Until someone spends that, the honest thing
    is to tell the operator what the job is about to do to the fleet, not to change what
    it does on an untested guess. See `notes/researcher/the-unstated-n.md`.
    """
    dev = os.environ.get("NETT_DEVICE", "0 (default)")
    print(f"⚠ CARD OCCUPANCY: NETT_DEVICE={dev} pins the SIM, but this process still takes "
          f"a small CUDA context on every visible card.\n"
          f"  launch_arm.sh refuses any card with a non-zero process count regardless of "
          f"MiB, so while this runs, NO ARM CAN LAUNCH ON THIS HOST.\n"
          f"  If the fleet needs cards, run this when the host is otherwise idle, or "
          f"accept that it blocks the queue for its full duration.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--every", type=int, default=20,
                    help="start a keep-window every N steps")
    ap.add_argument("--window", type=int, default=1,
                    help="keep W CONSECUTIVE frames at each sampling point. "
                         "W=1 (default) reproduces the original strided capture. "
                         "W>=2 is REQUIRED for any temporal or action-conditioned "
                         "objective: at W=1 no two kept frames are adjacent, so "
                         "(obs_t, a_t, obs_t+1) exists nowhere in the file.")
    ap.add_argument("--max-frames", type=int, default=6000)
    ap.add_argument("--condition", default=None, help="defaults to the run's first")
    args = ap.parse_args()
    _warn_about_card_occupancy()

    run_dir = args.run_dir.expanduser().resolve()
    config = _load_config(run_dir, args.episodes)

    # Imports are deferred: `Body.embed` builds the SimulationApp, and several of these
    # modules must not be imported before Kit exists on this machine.
    import gymnasium as gym
    import torch

    from nett_skrl.body import Body  # noqa: F401  (import shape mirrors nett.single_run)
    from nett_skrl.brain import Brain
    from nett_skrl.environment import Environment
    from nett_skrl.nett import _make_body
    from nett_skrl.runtime.cpu_budget import apply_torch_thread_limits
    from nett_skrl.runtime.task import Task, set_seeds

    class Capture(gym.Wrapper):
        """Records the policy observation on every reset/step. Pass-through otherwise."""

        def __init__(self, env, every: int, max_frames: int, window: int = 1,
                     num_envs: int | None = None) -> None:
            super().__init__(env)
            self.max_frames = int(max_frames)
            self.num_envs = resolve_num_envs(env, num_envs)
            # ⛔ WINDOW EXISTS BECAUSE STRIDED SINGLE FRAMES CANNOT TRAIN A TEMPORAL
            # LOSS. With `--every 20 --window 1` (the original behaviour) the capture
            # keeps frames 0, 20, 40 ... and NEVER two consecutive frames, so
            # (obs_t, a_t, obs_t+1) does not exist anywhere in the file. Every
            # action-conditioned candidate -- SPR, TACO, the report's rank-1 design --
            # needs exactly that triple. window=1 reproduces the old capture exactly.
            self.align = FrameAlignment(self.num_envs, every, window)
            self.frames: list[np.ndarray] = []
            self.keys: list[tuple[int, int, int]] = []   # (env_id, episode, step)
            self.actions: list[np.ndarray] = []
            self.action_keys: list[tuple[int, int, int]] = []
            self.truncated_by_cap = False

        @property
        def every(self) -> int:
            return self.align.every

        @property
        def window(self) -> int:
            return self.align.window

        @staticmethod
        def _chw_uint8(obs) -> np.ndarray:
            policy = obs.get("policy", obs) if isinstance(obs, dict) else obs
            arr = (policy.detach().cpu().numpy()
                   if isinstance(policy, torch.Tensor) else np.asarray(policy))
            if arr.dtype != np.uint8:
                # The body chain hands the encoder uint8; anything else means a
                # normalising wrapper was inserted and the capture is NOT the
                # encoder's input. Fail rather than save silently-wrong data.
                raise TypeError(f"expected uint8 observations, got {arr.dtype}")
            return arr

        def _record(self, obs) -> None:
            # ⚠ TRANSFER ONLY WHEN A FRAME IS ACTUALLY KEPT. Calling _chw_uint8 every
            # step forces a GPU->CPU sync plus a full-batch copy (112x3x128x128 = 5.5 MB)
            # on EVERY step even when the stride discards it -- measured 2026-08-12 as
            # the reason a capture crawled. The step counters are per-env and cheap, so
            # decide first, copy second.
            wanted = self.align.wanted()
            self.align.begin_record()
            if wanted and len(self.frames) < self.max_frames:
                arr = self._chw_uint8(obs)
                # ⛔ THE THIRD SOURCE. `num_envs` now comes from the caller and is
                # cross-checked against the env; this checks it against the only thing
                # that cannot be wrong -- the tensor actually handed to the encoder. A
                # capture that iterates fewer envs than the batch holds drops the rest
                # with no error and no missing-data signature in the file.
                if arr.shape[0] != self.num_envs:
                    raise SystemExit(
                        f"[capture] observation batch is {arr.shape[0]} but the capture "
                        f"iterates {self.num_envs} envs. Recording would silently keep "
                        f"{min(arr.shape[0], self.num_envs)} of them.")
                for env_id in wanted:
                    if len(self.frames) >= self.max_frames:
                        self.truncated_by_cap = True
                        break
                    key = self.align.key(env_id)
                    self.frames.append(arr[env_id].copy())
                    self.keys.append(key)
                    self.align.mark_kept(env_id, key)
            elif wanted:
                self.truncated_by_cap = True
            self.align.advance()

        def _record_action(self, action) -> None:
            """Attach `action` to the frame it ACTS ON, not the frame it produces.

            ⚠ THIS IS THE ALIGNMENT THE WHOLE FILE TURNS ON. At entry to step() the
            action is being taken FROM the last recorded frame. Attaching it to the
            resulting frame instead shifts every action by one step and trains an
            action-conditioned predictor on the action that FOLLOWED its target --
            an error that does not crash and is invisible in the saved file.
            See FrameAlignment at module level, which is unit-tested.
            """
            targets = self.align.action_targets()
            if not targets:
                return
            arr = (action.detach().cpu().numpy()
                   if isinstance(action, torch.Tensor) else np.asarray(action))
            arr = np.atleast_2d(arr)
            for env_id, key in targets:
                if env_id < arr.shape[0]:
                    self.actions.append(arr[env_id].astype(np.float32, copy=True))
                    self.action_keys.append(key)

        def step(self, action):
            self._record_action(action)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._record(obs)
            done = np.asarray(
                (torch.as_tensor(terminated) | torch.as_tensor(truncated)).cpu())
            for env_id in np.flatnonzero(np.atleast_1d(done)):
                self.align.on_done(int(env_id))
            return obs, reward, terminated, truncated, info

    apply_torch_thread_limits()

    brain_cfg = config.get("brain") or {}
    env_cfg = dict(config["environment"])
    base_brain = Brain(**brain_cfg)
    if "reward_types" not in env_cfg:
        inferred = base_brain.env_reward_types()
        if inferred:
            env_cfg["reward_types"] = list(inferred)
    base_body = _make_body(config.get("body"))
    base_env = Environment(**env_cfg)

    # ⛔ Fail in 0s with the cause named, rather than minutes in with 'EOF when reading
    # a line'. Cost a launch on 2026-09-11 -- the third for this one variable.
    from nett_skrl.runtime.kit_preflight import require_eula_or_explain
    if not require_eula_or_explain():
        raise SystemExit(3)

    # ⛔⛔ THE ROW AXIS, SET HERE BECAUSE A LAUNCH LINE IS NOT A PLACE TO KEEP A
    # REQUIREMENT. `NETTEnvCfg.test_group_by_row` defaults TRUE, and
    # environment.py's `_set_if_present` overrides it only when this variable is SET --
    # so an unset var means grouped. `launch_arm.sh:269` exports 0 for every launcher
    # arm, which is why the declared default has never been the value that runs on this
    # fleet; a direct invocation like this one has no launcher and inherits True.
    #
    # ⛔ MEASURED 2026-09-11, TWICE, ON THE SAME DAY. Capture A exported it and reached
    # 56/56 design rows. Capture B -- same driver, same source run, launch line retyped
    # -- did not, and reached 8 rows: pose `_00` only, target-left only, 560,000 log rows
    # all saying `correct.monitor=left`. 40 minutes of a card.
    #
    # ⛔ AND NOTHING IN THE OUTPUT SAID SO. `is_prefix` compares the EPISODE axis; this
    # truncation is on the ROW axis. Capture B finished, wrote 4.2 GB, reported a healthy
    # n_transition_pairs and passed every self-check in this file.
    if "NETT_TEST_GROUP_BY_ROW" not in os.environ:
        os.environ["NETT_TEST_GROUP_BY_ROW"] = "0"
        print("[capture] NETT_TEST_GROUP_BY_ROW was unset -> forcing 0 (strided). Grouped "
              "ordering truncates the design to one pose and one side for a capture-length "
              "run, and nothing downstream can detect it.")
    elif os.environ["NETT_TEST_GROUP_BY_ROW"] not in ("0", "false", "False"):
        print(f"[capture] ⛔ NETT_TEST_GROUP_BY_ROW="
              f"{os.environ['NETT_TEST_GROUP_BY_ROW']!r} -- grouped ordering. This capture "
              f"will cover a PREFIX of the design sheet (one pose, one target side) and "
              f"will not say so anywhere in its output. Unset it or set it to 0.")

    condition = args.condition or base_env.conditions[0]
    out_root = args.out.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    num_brains = int(config.get("num_brains", 1) or 1)
    source_test_episodes = config.pop("_source_test_episodes", None)
    num_envs = int(config.pop("_resolved_test_num_envs", {}).get(condition)
                   or config.get("max_parallel_envs") or num_brains)
    if num_envs % num_brains:
        raise SystemExit(f"num_envs {num_envs} not divisible by num_brains {num_brains}")

    task = Task(
        base_brain, base_body, base_env, condition, out_root / config["name"],
        modes=["test"], episodes={"test": args.episodes},
        num_brains=num_brains,
        brain_id_offset=int(config.get("brain_id_offset", 0) or 0),
        num_envs=num_envs,
    )
    task.set_device(int(os.environ.get("NETT_DEVICE", "0")))

    cfg = task.config
    set_seeds(cfg.seed)
    (cfg.path / "logs").mkdir(exist_ok=True, parents=True)
    run_config = cfg.for_mode("test")
    agent = task.agent

    # ★ THE EVAL BUDGET COMES FROM THE BRAIN, NOT THE TASK. `eval_timesteps` reads
    # `brain.test_iterations[condition]` and `brain.steps_per_episode`
    # (brain/run_config.py:51); `Task(...)` sets NEITHER. NETT.single_run configures
    # them via calc_iterations + _apply_parallel_env_plan, and skipping that is silent:
    # the eval returns instantly, writes a 0-row CSV and captures nothing, which reads
    # as "the wrapper was bypassed" rather than "the budget was zero". Inlined here
    # because _apply_parallel_env_plan uses no instance state (nett.py:696).
    episodes = {"train": 0, "test": int(args.episodes)}
    steps_per_episode = int(config.get("steps_per_episode", 200) or 200)
    base_brain.calc_iterations(num_brains, base_env.iterations_per_test_episode,
                               episodes, steps_per_episode)
    base_brain.iterations_per_test_episode = base_env.iterations_per_test_episode
    # The number of design rows for this condition -- the same quantity environment.py:366
    # multiplies by episodes["test"] to get the GLOBAL episode budget. Kept in a named
    # variable because two different per-unit counts are derived from it below.
    num_test_rows = int(base_env.iterations_per_test_episode.get(condition, 0))
    base_brain.envs_per_brain = max(1, num_envs // max(1, num_brains))
    agent.body.adjust_to_agent(agent.env, num_brains=num_brains, num_envs=num_envs,
                               episode_steps=steps_per_episode)
    budget = base_brain.test_iterations.get(condition, 0) * base_brain.steps_per_episode
    if budget <= 0:
        raise SystemExit(
            f"eval budget is 0 (test_iterations={base_brain.test_iterations}, "
            f"steps_per_episode={base_brain.steps_per_episode}) -- refusing to run")

    # ★ Brain.test both LOADS checkpoints from and WRITES logs to `config.path`
    # (= output_dir / condition, derived in TaskConfig.__post_init__ and init=False).
    # Pointing output_dir at the original run would therefore OVERWRITE that run's
    # test_*.csv -- destroying scored experimental data to capture pixels. Stage
    # copies into the fresh output dir instead; the eval then reads and writes only
    # there, and the source run is untouched.
    src_ckpts = run_dir / condition / "wandb_runs"
    if not src_ckpts.is_dir():
        raise SystemExit(f"no wandb_runs/ under {run_dir / condition}; nothing to load")
    staged = 0
    for brain_dir in sorted(src_ckpts.glob("brain_*")):
        src = brain_dir / "checkpoints" / "final_agent.pt"
        if not src.is_file():
            continue
        dest = cfg.path / "wandb_runs" / brain_dir.name / "checkpoints"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / "final_agent.pt")
        staged += 1
    if not staged:
        raise SystemExit(f"no final_agent.pt found under {src_ckpts}")
    print(f"[capture] staged {staged} checkpoints into {cfg.path} (source untouched)")

    loaded = agent.body.embed(agent.env, run_config)      # boots Kit
    capture = Capture(loaded, args.every, args.max_frames, window=args.window,
                      num_envs=run_config.num_envs)
    # ⚠ Read the count BACK OUT of the wrapper. Printing `run_config.num_envs` here is
    # what made a 1-env capture look like a 112-env one for a whole run.
    print(f"[capture] condition={condition} envs={capture.num_envs} "
          f"episodes={args.episodes} every={args.every} cap={args.max_frames}")
    agent.brain.test(capture, run_config)

    if not capture.frames:
        raise SystemExit("[capture] no frames recorded -- the eval produced no steps")
    obs = np.stack(capture.frames)
    keys = np.array(capture.keys, dtype=np.int32)
    actions = (np.stack(capture.actions) if capture.actions
               else np.zeros((0, 0), dtype=np.float32))
    action_keys = np.array(capture.action_keys, dtype=np.int32).reshape(-1, 3)

    # Contiguity is the property the file is FOR, so report it rather than let a
    # consumer discover its absence. A pair is usable iff frames (e, ep, s) and
    # (e, ep, s+1) are both present AND an action is attached to the first.
    present = {tuple(int(v) for v in k) for k in keys}
    act_at = {tuple(int(v) for v in k) for k in action_keys}
    n_pairs = sum(1 for (e, ep, st) in present
                  if (e, ep, st + 1) in present and (e, ep, st) in act_at)

    # ⚠ COMPLETENESS, RECORDED IN THE FILE ITSELF. `truncated_by_cap` already covers
    # --max-frames. This covers the other way a capture ends up a biased prefix: an
    # --episodes that does not match the source run's test count, or a schedule that
    # ended before the requested count was reached. A consumer must be able to tell a
    # complete capture from a plausible short one WITHOUT the launch command.
    # ⚠ A crash or kill is NOT this failure mode: the npz is written only after
    # brain.test() returns, so an interrupted capture leaves no file at all.
    # ⛔ `episodes_seen` IS PER ENV AND `--episodes` IS PER DESIGN ROW. This comparison
    # used to be `episodes_seen < args.episodes`, which compared 10 against 20 on a
    # capture that had reproduced its source run STEP FOR STEP -- 560,001 log rows in
    # both files -- and stamped it `is_prefix=True`. A peer then spent a message on "only
    # HALF the schedule ran", correctly refusing to guess the cause.
    #
    # The conversion, from environment.py:366 where the env is told its own budget:
    #
    #     test_total_episodes = rows * episodes_test        (GLOBAL, all envs)
    #     episodes_per_env    = test_total_episodes / num_envs
    #
    # so 56 rows x 20 = 1120 global / 112 envs = 10 per env. `episodes.test` is episodes
    # PER DESIGN ROW; `FrameAlignment.on_done` counts episodes PER ENV. Two units, one
    # name, and the wrong one was the alarm.
    #
    # ⚠ A FALSE `is_prefix` IS NOT HARMLESS. It is the only completeness signal a consumer
    # has without the launch command, so crying wolf on a complete capture trains the next
    # reader to discount the flag that will one day be true.
    episodes_seen = int(keys[:, 1].max()) + 1 if len(keys) else 0
    expected_per_env = expected_episodes_per_env(num_test_rows, args.episodes, num_envs)
    is_prefix = bool(
        capture.truncated_by_cap
        or (source_test_episodes is not None and args.episodes != source_test_episodes)
        or (expected_per_env and episodes_seen < expected_per_env)
    )
    # ⛔⛔ THE OUTCOME CHECK ON THE ROW AXIS. Setting NETT_TEST_GROUP_BY_ROW is the INPUT;
    # this is whether the design was actually covered. They can disagree -- a design sheet
    # change, a scope bug in _episodes_per_row, an env var consumed by a different layer --
    # and only this one is evidence. Read from the capture's OWN log, which is the file the
    # readout joins against.
    rows_visited, sides = _design_coverage(cfg.path / "logs")
    if rows_visited:
        print(f"[capture] design rows reached: {rows_visited}"
              + (f" of {num_test_rows}" if num_test_rows else "")
              + f"   target sides: {sorted(sides)}")
        if num_test_rows and rows_visited < num_test_rows:
            print(f"[capture] ⛔⛔ THE DESIGN WAS TRUNCATED ON THE ROW AXIS: {rows_visited} of "
                  f"{num_test_rows} rows. is_prefix covers the EPISODE axis and will not "
                  f"catch this. Almost always NETT_TEST_GROUP_BY_ROW; see the preflight above.")
        if len(sides) < 2:
            print(f"[capture] ⛔⛔ EVERY EPISODE HAS THE TARGET ON THE SAME SIDE ({sides}). "
                  f"No preference readout can be scored on this corpus -- a side-locked "
                  f"answer key makes every statistic a measurement of the agent's side bias.")

    dest = out_root / f"obs_{run_dir.name}_{condition}.npz"
    np.savez_compressed(dest, obs=obs, keys=keys,
                        actions=actions, action_keys=action_keys,
                        window=int(capture.window), every=int(capture.every),
                        n_transition_pairs=int(n_pairs),
                        episodes_requested=int(args.episodes),
                        episodes_seen=episodes_seen,
                        episodes_source=(-1 if source_test_episodes is None
                                         else int(source_test_episodes)),
                        is_prefix=is_prefix,
                        design_rows_visited=int(rows_visited),
                        design_rows_total=int(num_test_rows or 0),
                        target_sides=int(len(sides)),
                        truncated_by_cap=capture.truncated_by_cap,
                        run_dir=str(run_dir), condition=condition)
    print(f"[capture] {obs.shape} uint8 -> {dest}"
          + ("  ⚠ TRUNCATED BY --max-frames" if capture.truncated_by_cap else ""))
    print(f"[capture] actions {actions.shape}  window={capture.window} every={capture.every}")
    print(f"[capture] usable (obs_t, a_t, obs_t+1) transitions: {n_pairs}")
    print(f"[capture] episodes requested={args.episodes} seen={episodes_seen} "
          f"source_run={source_test_episodes if source_test_episodes is not None else '?'}")
    if is_prefix:
        print("[capture] ⛔ THIS CAPTURE IS A BIASED PREFIX, NOT A SAMPLE. The test schedule is "
              "ORDERED (conditions grouped, target-left rows first), so a short capture covers "
              "some conditions and not others. is_prefix=True is recorded in the npz.")
    if n_pairs == 0:
        print("[capture] ⚠ ZERO TRANSITION PAIRS -- this file cannot train any temporal or "
              "action-conditioned objective. Re-run with --window 2 or more.")
    print("[capture] join labels from the run's test CSV on (env_id, episode)")
    print("[capture] agent.x / agent.angle join on (env_id, episode, step) from the same CSV; "
          "that is how the transit mask is reproduced offline.")

    # ★ TEAR KIT DOWN. Without this the SimulationApp never exits: three such
    # processes were found on 2026-08-12 still holding GPU 0 up to two days after
    # they had printed their results, starving every later run on that card. This is
    # the same exit path `runtime/task_runner.py` uses for its workers.
    from nett_skrl.runtime.task_runner import _exit_worker_cleanly
    _finalize = getattr(loaded, "close", None)
    if callable(_finalize):
        try:
            _finalize()
        except Exception:
            pass
    _exit_worker_cleanly(run_config.logger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
