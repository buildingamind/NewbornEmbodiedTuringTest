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

        def __init__(self, env, every: int, max_frames: int, window: int = 1) -> None:
            super().__init__(env)
            self.max_frames = int(max_frames)
            self.num_envs = int(getattr(env, "num_envs", 1))
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
    capture = Capture(loaded, args.every, args.max_frames, window=args.window)
    print(f"[capture] condition={condition} envs={run_config.num_envs} "
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
    episodes_seen = int(keys[:, 1].max()) + 1 if len(keys) else 0
    is_prefix = bool(
        capture.truncated_by_cap
        or (source_test_episodes is not None and args.episodes != source_test_episodes)
        or episodes_seen < args.episodes
    )
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
