"""Offline replay: train a candidate representation on FIXED recorded experience.

⛔ WHY THIS EXISTS. The fleet's same-specification within-key sd is ~0.08 on the
criterion conditions, and only ~0.002 of that is evaluation noise -- the rest is
introduced during training. A live arm therefore cannot resolve a contrast
smaller than its own reproducibility, and most candidate comparisons live there.

⚠ WHAT REPLAY DOES AND DOES NOT FIX -- corrected 2026-09-09, the first version of
this claim was too strong. Training-origin variance has TWO components:
  (a) initial weights and optimisation path -- replay removes NONE of it;
  (b) the experience each policy goes on to generate -- replay removes ALL of it.
Nothing measured so far separates them (`brain_id_offset` moves both at once:
agent_factory.py:195 seeds the weights, episode_seed.py:50 seeds the env draws),
so "replay removes the dominant term" is NOT established.
⇒ The defensible claim, true even if (b) is zero: **replay makes a sub-floor
effect affordable to resolve, not noise-free.** Every candidate is paired on
byte-identical data, and the residual is averaged down by offline runs at minutes
each instead of ~14-hour arms.

⛔ AND IT IS NOT THE NETT MEASUREMENT. This scores RETRIEVAL -- can the frozen
representation tell the objects apart -- while a NETT arm scores BEHAVIOUR, which
monitor the agent approaches. They are different quantities. A candidate that wins
here has earned an arm, not a result. If retrieval succeeds and preference fails,
the bottleneck is the policy or the readout, not the encoder.

THE READOUT IS THE IMPRINTING RULE, NOT A SUPERVISED PROBE. A linear probe fitted
on test labels would answer "is the object linearly decodable", which is a much
easier question than the one the chicks are asked and would flatter every
candidate. Instead: build ONE memory vector by averaging encoder features over the
training-exposure clip only (fork-1 sees `2A` and nothing else), then for each test
pair choose whichever member is closer to that memory in cosine similarity. That is
"approach the more familiar-looking thing", it uses no test labels for fitting, and
it reproduces the structure of `correct_pct`.
⇒ On `BU same-bg` (`2B` vs `1B`, both on a scene never trained on) a
background-bound representation scores ~0.5 because both members are equally
unfamiliar, and an object-bound one scores high. That is exactly the cell the
model search is targeting.

    # fixture (plumbing only -- see the warning below), no card needed:
    PYTHONPATH=. python examples/replay_harness.py --fixture --model ViViT --aux vicreg_tt

    # real capture, once one exists:
    PYTHONPATH=. python examples/replay_harness.py --capture obs_*.npz --model ViViT

⛔⛔ FIXTURE MODE IS NOT THE AGENT'S VIEW AND CAN NEVER CARRY A SCIENTIFIC CLAIM.
It decodes the monitor's own video files. The chamber view is smaller, off-centre,
lit, DLSS-reconstructed, equisolid-warped over a 300-degree field, and usually
shows one monitor rather than two -- `capture_observations.py` exists precisely to
remove that caveat. Fixture mode validates that the harness RUNS. Any number it
prints is labelled FIXTURE and is about the plumbing, never about a model.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))

#: Stimulus clips for the fork-1 imprint. digit = OBJECT (1 ship, 2 fork),
#: letter = BACKGROUND (A desert road, B forest, C beach), number = viewpoint.
#: Verified from DesignSheet_Parsing_mov.csv training rows, not from the names.
FIXTURE_ROOT = Path("~/code/isaac/videos/parsing/videos").expanduser()
TRAIN_CLIP = "2A_00.mov"                      # fork-1 trains on this and nothing else
READOUT_PAIRS = {                             # (target, distractor) -> cell
    ("2B_00.mov", "1B_00.mov"): "BU same-bg (forest)",
    ("2C_00.mov", "1C_00.mov"): "BU same-bg (beach)",
    ("2A_00.mov", "1A_00.mov"): "Both Familiar",
}


def decode(path: Path, width: int, height: int, limit: int = 240) -> np.ndarray:
    """Decode a clip to (N, H, W, 3) uint8 at the eye's pixel count."""
    cmd = ["ffmpeg", "-loglevel", "error", "-i", str(path),
           "-vf", f"scale={width}:{height}", "-frames:v", str(limit),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frame = height * width * 3
    n = len(raw) // frame
    if n == 0:
        raise SystemExit(f"[replay] decoded 0 frames from {path}")
    return np.frombuffer(raw[: n * frame], dtype=np.uint8).reshape(n, height, width, 3)


def framestack(frames: np.ndarray, depth: int) -> np.ndarray:
    """(N,H,W,3) -> (N-depth+1, H, W, 3*depth), matching the body's channel stack."""
    if depth == 1:
        return frames
    return np.concatenate([frames[i: len(frames) - depth + 1 + i] for i in range(depth)], axis=-1)


class ReplayMemory:
    """Duck-types the slice of skrl memory the aux losses read.

    ★ DELIBERATELY A SHIM, NOT A REIMPLEMENTATION. The aux losses draw their own
    windows through `attach_memory` / `memory.tensors[...]`; giving them that
    interface means this harness trains with the SAME loss object an arm runs,
    including NETT_AUX_TRANSIT_MASK's action weighting. Reimplementing the
    sampling here would test a replica and tell us nothing about the original.
    """

    def __init__(self, obs, actions=None) -> None:
        import torch
        self.tensors = {"observations": obs}
        if actions is not None:
            self.tensors["actions"] = actions
        else:
            # No action channel: the transit mask falls back to uniform and says so.
            self.tensors["actions"] = torch.zeros(obs.shape[0], obs.shape[1], 2)
        self.memory_size = int(obs.shape[0])
        self.memory_index = int(obs.shape[0])
        self.filled = True


def resolve_model(model_name: str) -> tuple[dict, str]:
    """Return the encoder spec for `model_name`, from one of TWO namespaces.

        "ViViT"                 -> MODELS, the launcher's arm registry
        "compact_cnn"           -> a registry ENCODER with no MODELS entry: a screening
                                   host, buildable here and not launchable by a queue row
        "compact_cnn:{...json}" -> the same, with an explicit cfg

    ⛔ SAME REASON AS `resolve_aux`, ON THE OTHER AXIS. A candidate loss needs a host to
    run on, and the host that suits it may not be one the fleet has an arm for --
    `compact_cnn` is in `encoder_mapping` but in no `MODELS` entry, and its pre-pool map is
    640 positions against `nature_cnn`'s 72, which is the difference between a slot method
    having something to compete over and not. Adding a MODELS entry to screen it would make
    it launchable before anything had screened it. Registration is the graduation event.
    """
    import json

    from campaign_train import MODELS
    from nett_skrl.brain.registry import encoder_mapping

    if model_name in MODELS:
        return MODELS[model_name], "registry"
    enc_name, _, cfg_json = model_name.partition(":")
    if enc_name not in encoder_mapping:
        raise SystemExit(
            f"[replay] {model_name!r} is neither a MODELS entry {sorted(MODELS)} nor a "
            f"registry encoder {sorted(encoder_mapping)}."
        )
    try:
        cfg = json.loads(cfg_json) if cfg_json else {}
    except json.JSONDecodeError as e:
        raise SystemExit(f"[replay] cfg after ':' is not JSON: {e}")
    print(f"⚠ SCREENING HOST: encoder {enc_name!r} cfg={cfg} -- NOT a MODELS entry, so no "
          f"queue row can launch it. Any number below is about a model the fleet does not "
          f"currently run.")
    # framestack is a MODELS-level decision the caller makes via --framestack here.
    return {"encoder": enc_name, "cfg": cfg, "framestack": None}, "screening"


def build_encoder(model_name: str, obs_shape, seed: int):
    import gymnasium as gym
    import torch
    from nett_skrl.brain.registry import encoder_mapping

    torch.manual_seed(seed)
    spec, _origin = resolve_model(model_name)
    space = gym.spaces.Box(low=0, high=255, shape=tuple(obs_shape), dtype=np.uint8)
    # ⛔ THE KEY IS `cfg`, NOT `encoder_kwargs`. Reading the wrong key does not raise --
    # it silently builds a DEFAULT-configured encoder, which is a different model from
    # the one the fleet runs. Caught by the param count disagreeing with the measured
    # 694,016 for ViViT at Box(80,128,6); it printed 509,312.
    cfg = spec.get("cfg")
    if cfg is None:
        raise SystemExit(
            f"[replay] MODELS[{model_name!r}] has no 'cfg'; refusing to build a "
            "default-configured encoder that would not be the model under test."
        )
    # ⚠ A SCREENING HOST'S EMPTY cfg IS A DELIBERATE {}, NOT A MISSING KEY -- the guard
    # above catches a MODELS entry that forgot its cfg, which is a different fault.
    enc = encoder_mapping[spec["encoder"]](space, **dict(cfg)).cpu()
    return enc


def resolve_aux(aux_kind: str):
    """Return the aux class named by `aux_kind`, from one of TWO namespaces.

        "vicreg_tt"                       -> AUX_LOSSES, the LAUNCHER's registry
        "path/to/mod.py:ClassName"        -> a SCREENING candidate, loaded by path

    ⛔ THE SECOND NAMESPACE EXISTS SO SCREENING DOES NOT REQUIRE REGISTERING. `AUX_LOSSES`
    is what `NETT_AUX_LOSS` resolves against, so putting a candidate there to let the
    harness reach it ALSO makes it launchable by a queue row -- and a queue row could then
    spend card-hours on a loss that nothing has screened, which is the exact ordering this
    harness was built to prevent. Registration is the graduation event, not the entry fee:
    a name in AUX_LOSSES means "this has been screened", and that only stays true if
    unscreened candidates have somewhere else to live.

    ⚠ A path-loaded candidate is deliberately NOT importable by the trainer. Promoting one
    means moving the module into `nett_skrl/brain/aux/` and adding the registry entry --
    a visible diff, in the repo the launcher reads, rather than a string in an invocation.
    """
    from nett_skrl.brain.aux.ppo_aux import AUX_LOSSES

    if ":" not in aux_kind:
        if aux_kind not in AUX_LOSSES:
            raise SystemExit(
                f"[replay] unknown aux {aux_kind!r}. Registered: {sorted(AUX_LOSSES)}. "
                f"For a candidate that is not registered yet, pass 'path/to/module.py:ClassName'."
            )
        return AUX_LOSSES[aux_kind], "registry"

    import importlib.util

    mod_path, _, cls_name = aux_kind.rpartition(":")
    path = Path(mod_path).expanduser()
    if not path.is_file():
        raise SystemExit(f"[replay] no such candidate module: {path}")
    spec = importlib.util.spec_from_file_location(f"_replay_candidate_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, cls_name, None)
    if cls is None:
        raise SystemExit(
            f"[replay] {path} defines no {cls_name!r}; found: "
            f"{[n for n in vars(module) if not n.startswith('_')]}"
        )
    # ⚠ Say it out loud on every run. A screening result quoted without this line reads
    # like a result about a model the fleet can launch, and it is not one yet.
    print(f"⚠ SCREENING CANDIDATE: {cls_name} loaded from {path} -- NOT in AUX_LOSSES, so no "
          f"queue row can launch it. Promoting means moving the module into "
          f"nett_skrl/brain/aux/ and registering it.")
    return cls, str(path)


def train_offline(encoder, aux_kind: str, obs, actions, steps: int, lr: float, seed: int):
    """Train encoder + aux on the fixed stream. No environment, no policy, no reward."""
    import torch

    torch.manual_seed(seed)
    aux_cls, _origin = resolve_aux(aux_kind)
    aux = aux_cls(encoder)
    mem = ReplayMemory(obs, actions)
    if hasattr(aux, "attach_memory"):
        aux.attach_memory(mem)
    params = list(encoder.parameters()) + list(aux.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = aux.compute(encoder, obs[0])
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return aux, losses



def familiarity_readout(encoder, train_frames, pairs: list) -> dict:
    """Imprinting rule: pick whichever pair member is closer to the training memory.

    ⚠ Uses NO test labels for fitting. The memory is built from the training-exposure
    clip alone, which is the only thing the agent was reared on.
    """
    import torch
    import torch.nn.functional as F

    encoder.eval()
    with torch.no_grad():
        def feats(x):
            return encoder.encode_prepared(encoder._prepare_image(torch.as_tensor(x)))

        memory = F.normalize(feats(train_frames).mean(0, keepdim=True), dim=-1)
        out = {}
        for tgt_frames, dst_frames, label in pairs:
            zt = F.normalize(feats(tgt_frames), dim=-1)
            zd = F.normalize(feats(dst_frames), dim=-1)
            n = min(len(zt), len(zd))
            # Per-frame contest, mirroring a per-step behavioural choice.
            correct = ((zt[:n] @ memory.T) > (zd[:n] @ memory.T)).float().mean()
            out[label] = float(correct)
    encoder.train()
    return out


def load_fixture(model_name: str, width: int, height: int, depth: int, envs: int = 4):
    """Decode the stimulus clips. ⛔ PLUMBING ONLY -- not the agent's view."""
    import torch
    clips = {TRAIN_CLIP}
    for tgt, dst in READOUT_PAIRS:
        clips |= {tgt, dst}
    decoded = {c: framestack(decode(FIXTURE_ROOT / c, width, height), depth) for c in sorted(clips)}
    train = decoded[TRAIN_CLIP]
    # The training stream the aux loss sees: the reared clip, shaped (T, n_env, H, W, C).
    t = len(train)
    stream = torch.as_tensor(np.stack([train] * envs, axis=1).copy())
    pairs = [(decoded[a], decoded[b], lab) for (a, b), lab in READOUT_PAIRS.items()]
    return stream, train, pairs, t


# --------------------------------------------------------------------------------------
# Capture-mode readout: the imprinting rule, run on the agent's OWN view.
#
# The rule is the fixture rule -- memory is the mean feature over exposure frames, and a
# test episode is scored by which stimulus is closer to it in cosine. What changes is
# where the exposure frames come from. A capture holds only `experiment.phase == test`,
# so there are no rearing frames in it; the rearing CLIP is available from the stimulus
# library, but its pixels are undistorted and the captured pixels carry the equisolid
# lens, the chamber, and DLSS. A cosine across those two domains measures the domain.
#
# ⭐ THE EXPOSURE SET IS ALREADY IN THE CAPTURE, UNDER A CONDITION NOBODY SCORES. `Rest`
# displays the imprinted object alone against a blank screen -- measured on
# 3DCNN_parsing_fork-1_off0_0831_160636, every Rest row is `2A_00.mov` opposite
# `White.mov`, 21,000 steps of it. Same lens, same chamber, same renderer as the scored
# frames, and `Rest` is excluded from the parked-fraction denominator and from every
# preference statistic, so using it fits nothing that is later reported.
#
# ⚠ WHAT THIS COSTS, STATED HERE BECAUSE IT DOES NOT SHOW UP IN THE OUTPUT NUMBER. The
# agent's view is egocentric, so "frames of the left monitor" is not a segmentation, it
# is a POSITION filter: frames where the agent stands on that side. That makes the
# available frames a consequence of where the agent CHOSE to stand, and the agent is
# parked ~80% of steps. An episode where it never crossed the chamber yields frames of
# one monitor and none of the other, and no contest can be run on it. Those episodes are
# EXCLUDED, and excluding them selects the less-parked episodes -- a biased subset, not a
# sample. Every number this returns is reported with the episodes it kept and the ones it
# could not score, because the exclusion is the largest thing about it.
# --------------------------------------------------------------------------------------

#: Outer third of the chamber (HALF_X 33.15 / 3). A frame counts as a view of a monitor
#: only from that side's outer third; nearer the middle both screens are in the 300deg
#: field and the frame is evidence about neither.
VIEW_X = 11.05
#: Below this many frames on a side, the side's mean is one or two frames of a parked
#: agent and the "contest" is noise. Reported, not silently applied.
MIN_SIDE_FRAMES = 3
#: Every captured frame's key exists in the log of the run that produced it, so a correct
#: join is ~100%. Anything materially short of it identifies the wrong file.
MIN_JOIN_RATE = 0.90
REST_COND = "Rest"
#: The blank screen Rest shows opposite the imprinted object.
BLANK = "White.mov"


def default_test_csv(run_dir: str, condition: str, capture: Path | None = None) -> Path:
    """The per-step log whose keys match the capture's.

    ⛔ NOT THE SOURCE RUN'S LOG, WHICH IS THE OBVIOUS AND WRONG CHOICE. A capture is a
    REPLAY: `capture_observations` re-runs the test phase into its own output tree with
    its own env/episode numbering, and writes its own `test_*.csv` there. The original
    run's log describes a different execution.

    ⛔ MEASURED 2026-09-10: joined against the SOURCE run's log, a 5,120-frame capture
    matched **64 frames -- 1.25%**, all of them `Rest`, which would have been adopted
    silently as the exposure set.

    ⚠ MY FIRST EXPLANATION OF THAT 64 WAS WRONG AND IS RECORDED HERE BECAUSE IT WAS
    NEARLY SHIPPED AS A MEASURED FACT. I wrote that the capture's `env_id 0` keys had
    COLLIDED with the source run's `env_id 0`. Then the capture's OWN log returned the
    same 64. A cause that predicts a difference between two files, tested on only one of
    them, explained nothing. The real cause is the episode numbering
    (`episode_index_map`), and it applies to BOTH files equally. Pointing at the capture's
    own log is still correct and still necessary; it is simply not sufficient, and it was
    never the reason for the 64.

    ⛔ NOT `analysis/test/test_preferences.csv` either. That file is a per-episode SUMMARY
    with no `agent.x`, and an awk survey in this campaign read it by position, found no
    such column, and reported a silent 1.0000.
    """
    if capture is not None:
        # The capture's own tree sits beside the npz, under the source run's NAME.
        for cand in sorted(Path(capture).parent.glob(f"*/{condition}/logs/test_*.csv")):
            return cand
    print(f"⚠ no per-step log inside the capture's own output tree; falling back to the "
          f"SOURCE run at {run_dir}. Expect a near-zero join: a capture is a replay with "
          f"its own env/episode numbering.")
    base = Path(run_dir) / condition / "logs"
    exact = base / f"test_{condition}_0.csv"
    if exact.exists():
        return exact
    found = sorted(base.glob("test_*.csv"))
    if not found:
        raise SystemExit(
            f"[replay] no test_*.csv under {base}. Pass --test-csv explicitly. (If the "
            f"only file you can find is analysis/test/test_preferences.csv, that is a "
            f"per-episode summary and carries no agent.x -- it cannot drive this readout.)")
    if len(found) > 1:
        raise SystemExit(f"[replay] {len(found)} test logs under {base}: "
                         f"{[p.name for p in found]}. Pass --test-csv to choose.")
    return found[0]


def load_test_labels(csv_path, keep=None):
    """``(env_id, episode, step) -> row`` from a run's ``test_*.csv``.

    ⚠ `keep`, when given, is the set of keys the caller actually needs, and rows outside
    it are discarded as they stream past. These logs are large -- the capture replay's own
    log is 4,480,001 rows / 580 MB against a 5,120-frame capture -- and materialising all
    of it costs gigabytes to answer a question about 0.1% of it.

    ⛔ The key is verified unique rather than assumed: on the reference run it is unique
    across all 560,000 rows, but a duplicated key would make `dict` keep the LAST row
    silently, and a positional read of a second schema is how an earlier analysis in this
    campaign got every implied n wrong without any number looking wrong. Read by NAME.
    """
    import csv as _csv

    need = ("env_id", "episode", "step", "agent.x", "test.cond",
            "left.monitor", "right.monitor", "correct.monitor")
    rows, dupes = {}, 0
    with open(csv_path, newline="") as fh:
        reader = _csv.DictReader(fh)
        missing = [c for c in need if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"[replay] {csv_path} is missing {missing}. Columns present: "
                f"{reader.fieldnames}. This is the wrong CSV -- analysis/test/"
                f"test_preferences.csv is a SUMMARY and carries no agent.x; the "
                f"per-step log is fork-1/logs/test_*.csv.")
        for r in reader:
            k = (int(r["env_id"]), int(r["episode"]), int(r["step"]))
            if keep is not None and k not in keep:
                continue
            if k in rows:
                dupes += 1
            rows[k] = r
    if dupes:
        raise SystemExit(f"[replay] {dupes} duplicate (env_id, episode, step) keys in "
                         f"{csv_path}; the join would silently keep one row per key.")
    return rows


def episode_index_map(csv_path) -> dict:
    """``(env_id, LOCAL episode index) -> global episode id``, derived from the log.

    ⛔ MEASURED 2026-09-10, AND THIS IS A REAL DEFECT IN THE DOCUMENTED JOIN.
    `capture_observations.py` numbers episodes with a PER-ENV counter --
    `FrameAlignment.on_done` does `self._episode[env_id] += 1` from 0 -- while the run log
    numbers them GLOBALLY across envs. On the reference capture, env 0's episodes in the
    log are `0, 112, 224, 336, ...` (stride = num_envs) and the capture recorded `0..79`.

    ⇒ Exactly ONE episode per env coincides: local 0 == global env_id. A 5,120-frame
    capture joined **64 frames, 1.25%** -- precisely the 64 frames of episode 0 -- and the
    module docstring's instruction to "join labels from the run's test CSV on (env_id,
    episode)" has never worked for any other episode. It went unnoticed because this
    driver had never completed a full-schedule run.

    ⚠ Derived from the log rather than computed as `local * num_envs + env_id`, so it does
    not depend on the striding staying regular.
    """
    import csv as _csv

    per_env: dict = {}
    seen = set()
    with open(csv_path, newline="") as fh:
        reader = _csv.reader(fh)
        header = next(reader)
        i_env, i_ep = header.index("env_id"), header.index("episode")
        for row in reader:
            pair = (int(row[i_env]), int(row[i_ep]))
            if pair in seen:
                continue
            seen.add(pair)
            per_env.setdefault(pair[0], []).append(pair[1])
    return {(env, i): ep
            for env, eps in per_env.items()
            for i, ep in enumerate(sorted(eps))}


def _side_of(x: float) -> str | None:
    """Which monitor this position is a view of, or None for the ambiguous middle."""
    if x >= VIEW_X:
        return "right"
    if x <= -VIEW_X:
        return "left"
    return None


def build_capture_pairs(blob, csv_path, verbose: bool = True):
    """Join a capture to its run's per-step log; return (memory_idx, episodes, report).

    ``memory_idx``  frame indices of the exposure set: Rest frames taken from the side
                    showing the imprinted object.
    ``episodes``    one entry per scorable test episode:
                    ``(cond, {"left": [idx...], "right": [idx...]}, correct)``.
    """
    keys = blob["keys"]
    # The capture's episode field is a per-env counter; the log's is global. Translate
    # before joining, or 99% of the capture silently fails to match.
    ep_map = episode_index_map(csv_path)
    translated, untranslatable = [], 0
    for k in keys:
        env, local, step = int(k[0]), int(k[1]), int(k[2])
        gep = ep_map.get((env, local))
        if gep is None:
            untranslatable += 1
            translated.append(None)
        else:
            translated.append((env, gep, step))
    if untranslatable:
        print(f"⚠ {untranslatable}/{len(keys)} captured frames name an episode index the "
              f"log has no episode for (the capture ran longer than the log records).")
    wanted = {t for t in translated if t is not None}
    labels = load_test_labels(csv_path, keep=wanted)

    joined, unjoined = [], 0
    for i, t in enumerate(translated):
        row = labels.get(t) if t is not None else None
        if row is None:
            unjoined += 1
            continue
        joined.append((i, row))
    # ⛔ A RATE, NOT A ZERO CHECK. Refusing only at zero passed a 1.25% join built entirely
    # from key COLLISIONS against another execution's log -- see default_test_csv. Every
    # captured frame's key exists in the log of the run that produced it, so anything
    # short of nearly total is the wrong file, not a partial capture.
    rate = len(joined) / max(len(keys), 1)
    if rate < MIN_JOIN_RATE:
        raise SystemExit(
            f"[replay] ONLY {len(joined)} of {len(keys)} captured frames ({rate:.2%}) "
            f"joined to {csv_path}.\n"
            f"  A capture's every frame is a step of the run that produced it, so a "
            f"partial join means the WRONG LOG, not a partial capture.\n"
            f"  ⛔ This is not a zero -- a partial join arrives looking like data. "
            f"Measured: a 5,120-frame capture joined 64 frames, all labelled Rest, which "
            f"would have become the exposure set.\n"
            f"  Two independent causes produce this, and BOTH must be right:\n"
            f"    1. the log must be the CAPTURE's own (a capture is a replay), and\n"
            f"    2. the capture's per-env episode counter must be translated to the "
            f"log's global episode ids -- see episode_index_map.")

    # --- the exposure set --------------------------------------------------------------
    memory_idx, rest_seen, rest_wrong_side = [], 0, 0
    for i, row in joined:
        if row["test.cond"] != REST_COND:
            continue
        rest_seen += 1
        side = _side_of(float(row["agent.x"]))
        if side is None:
            continue
        shown = row[f"{side}.monitor"]
        # The imprinted object is whichever monitor is NOT blank during Rest.
        if shown != BLANK:
            memory_idx.append(i)
        else:
            rest_wrong_side += 1

    # --- the scored episodes -----------------------------------------------------------
    by_ep: dict = {}
    for i, row in joined:
        cond = row["test.cond"]
        if cond == REST_COND:
            continue
        ep = (int(row["env_id"]), int(row["episode"]))
        side = _side_of(float(row["agent.x"]))
        slot = by_ep.setdefault(ep, {"cond": cond, "correct": row["correct.monitor"],
                                     "left": [], "right": [], "middle": 0})
        if side is None:
            slot["middle"] += 1
        else:
            slot[side].append(i)

    # --- the pooled fallback's groups, collected whether or not the contest is usable ---
    imprint_clip = next((row[f"{_side_of(float(row['agent.x']))}.monitor"]
                         for i, row in joined
                         if row["test.cond"] == REST_COND
                         and _side_of(float(row["agent.x"])) is not None
                         and row[f"{_side_of(float(row['agent.x']))}.monitor"] != BLANK),
                        None)
    by_object = {"imprinted object": [], "other object": []}
    if imprint_clip is not None:
        want = object_token(imprint_clip)
        for i, row in joined:
            if row["test.cond"] == REST_COND:
                continue
            side = _side_of(float(row["agent.x"]))
            if side is None:
                continue
            key = ("imprinted object" if object_token(row[f"{side}.monitor"]) == want
                   else "other object")
            by_object[key].append(i)

    # ⛔ A CONSTANT ANSWER KEY IS NOT AN ANSWER KEY. The imprinting rule scores its choice
    # against `correct.monitor`; if every scored episode has the SAME correct side, an
    # agent that always picks that side scores 1.000 and one that always picks the other
    # scores 0.000, and NEITHER number is about object familiarity. It is the side-lock
    # score wearing a preference label -- exactly the statistic a non-policy attains.
    #
    # ⛔ MEASURED 2026-09-10, AND THIS IS NOT HYPOTHETICAL: every capture this driver has
    # produced is 100% target-left. Two of them, at very different lengths:
    #     --episodes 160 -> 80 ep/env, 4,480,000 rows, correct.monitor: left 100%, right 0
    #     --episodes  20 -> 10 ep/env,   560,000 rows, correct.monitor: left 100%, right 0
    # while the SOURCE RUN over the identical 560,000 rows is balanced 50/50. ⇒ The bias
    # is NOT a prefix artefact and capturing more does not cure it: 8x the source
    # schedule yielded zero target-right rows. `capture_observations`'s own docstring
    # attributes this to an ordered schedule whose "target-left design rows come first",
    # which predicts that a longer capture reaches the right-target rows. It does not.
    sides = {row["correct.monitor"] for _, row in joined if row["test.cond"] != REST_COND}
    # ⚠ `== 1`, not `< 2`: an EMPTY set means no scored rows at all, which the
    # scorable-episode accounting below reports honestly. Only a corpus that scores
    # something against ONE answer is the degenerate case.
    if len(sides) == 1:
        raise SystemExit(
            f"[replay] REFUSING TO SCORE: every scored episode has correct.monitor="
            f"{sides.pop() if sides else 'NONE'}.\n"
            f"  With a constant answer key, a side-locked agent scores 1.000 and the "
            f"readout measures WHICH SIDE the agent prefers, not whether it recognises "
            f"the imprinted object.\n"
            f"  ⛔ This is a property of every capture this driver has produced (both "
            f"lengths measured 2026-09-10 are 100% target-left, against a balanced source "
            f"run), so it is not fixed by capturing more episodes.\n"
            f"  Fix the capture's schedule construction before any side-dependent readout "
            f"is run on it.")

    episodes, dropped = [], {}
    for ep, slot in sorted(by_ep.items()):
        if min(len(slot["left"]), len(slot["right"])) < MIN_SIDE_FRAMES:
            dropped[slot["cond"]] = dropped.get(slot["cond"], 0) + 1
            continue
        episodes.append((slot["cond"], {"left": slot["left"], "right": slot["right"]},
                         slot["correct"]))

    report = {"imprint_clip": imprint_clip, "by_object": by_object,
              "captured": len(keys), "joined": len(joined), "unjoined": unjoined,
              "rest_frames": rest_seen, "memory_frames": len(memory_idx),
              "rest_blank_side": rest_wrong_side, "episodes_total": len(by_ep),
              "episodes_scorable": len(episodes), "dropped_one_sided": dropped}

    if verbose:
        print(f"[replay] join: {len(joined)}/{len(keys)} frames matched the run log"
              + (f"  ⚠ {unjoined} UNJOINED" if unjoined else ""))
        print(f"[replay] exposure set: {len(memory_idx)} Rest frames viewing the "
              f"imprinted object (of {rest_seen} Rest frames captured; "
              f"{rest_wrong_side} viewed the blank)")
        n_drop = sum(dropped.values())
        print(f"[replay] scorable episodes: {len(episodes)}/{len(by_ep)}  "
              f"({n_drop} dropped: fewer than {MIN_SIDE_FRAMES} frames on one side)")
        if dropped:
            print("[replay] ⚠ DROPPED BY CONDITION " + ", ".join(
                f"{c}: {n}" for c, n in sorted(dropped.items())))
            print("[replay] ⚠ THE DROP IS THE SELECTION. An episode is unscorable "
                  "precisely when the agent never left one monitor, so what remains is "
                  "the less-parked tail. These numbers describe THAT subset.")

    if not memory_idx:
        raise SystemExit(
            f"[replay] the exposure set is EMPTY: {rest_seen} Rest frames captured, none "
            f"taken from the side showing the imprinted object. Without a memory there "
            f"is no imprinting rule to run -- this is not a score of 0.5, it is no score. "
            f"Recapture including Rest, or raise --episodes so Rest is reached.")
    return memory_idx, episodes, report


def object_token(clip: str) -> str:
    """The OBJECT half of a stimulus name. `2A_00.mov` -> `2`.

    ⚠ Verified from DesignSheet_Parsing_mov.csv, not from the names: digit = OBJECT
    (1 ship, 2 fork), letter = BACKGROUND (A desert, B forest, C beach). `Novel Familiar`
    pairs `1A` against `2B`/`2C` with `2*` correct -- so it contrasts OBJECT binding
    against BACKGROUND binding, and the object digit is the axis under test.
    """
    return clip[:1]


def object_contrast(encoder, obs, memory_idx, by_object, chunk: int = 256) -> dict:
    """Mean cosine to the exposure memory for frames viewing the imprinted object vs the
    other object, POOLED ACROSS EPISODES.

    ⛔ WHY THIS EXISTS ALONGSIDE THE PER-EPISODE CONTEST, WHICH IS THE PREFERRED
    STATISTIC. The contest needs one episode to contain frames of BOTH monitors, and a
    parked agent never supplies that: measured on the reference capture, all 70 test
    episodes had every one of their 56 frames on a single side, so 0 of 70 were scorable.
    Pooling asks a strictly weaker question -- does the representation place the imprinted
    OBJECT nearer the memory, wherever it was seen -- and it survives an agent that never
    crosses, because different episodes park at different monitors.

    ⚠ IT IS NOT A PREFERENCE, AND MUST NEVER BE REPORTED AS ONE. Which frames exist is
    decided by where the agent chose to stand, so the two groups are not balanced by
    design and their sizes are a behavioural result, not a sample size. Report n for each
    side and treat a large imbalance as a statement about the agent, not about the encoder.
    """
    import torch
    import torch.nn.functional as F

    encoder.eval()
    with torch.no_grad():
        def feats(idx):
            out = []
            for s in range(0, len(idx), chunk):
                out.append(encoder.encode_prepared(
                    encoder._prepare_image(torch.as_tensor(obs[list(idx[s:s + chunk])]))))
            return torch.cat(out)

        memory = F.normalize(feats(memory_idx).mean(0, keepdim=True), dim=-1)
        out = {}
        for label, idx in by_object.items():
            if not idx:
                continue
            z = F.normalize(feats(idx), dim=-1)
            out[label] = (float((z @ memory.T).mean()), len(idx))
    encoder.train()
    return out


def capture_readout(encoder, obs, memory_idx, episodes, chunk: int = 256) -> dict:
    """The imprinting rule on captured frames. Returns ``{cond: (acc, n_episodes)}``.

    ⚠ Uses NO label from the episodes being scored. The memory is built from Rest, whose
    monitor assignment is known the same way the rearing clip's identity is known in
    fixture mode; `correct.monitor` is read only to grade a choice already made.
    """
    import torch
    import torch.nn.functional as F

    encoder.eval()
    with torch.no_grad():
        def feats(idx):
            out = []
            for s in range(0, len(idx), chunk):
                batch = torch.as_tensor(obs[list(idx[s:s + chunk])])
                out.append(encoder.encode_prepared(encoder._prepare_image(batch)))
            return torch.cat(out)

        memory = F.normalize(feats(memory_idx).mean(0, keepdim=True), dim=-1)

        tally: dict = {}
        for cond, sides, correct in episodes:
            sim = {}
            for side in ("left", "right"):
                z = F.normalize(feats(sides[side]), dim=-1)
                sim[side] = float((z @ memory.T).mean())
            chosen = "left" if sim["left"] > sim["right"] else "right"
            hit, n = tally.get(cond, (0, 0))
            tally[cond] = (hit + int(chosen == correct), n + 1)
    encoder.train()
    return {c: (h / n, n) for c, (h, n) in tally.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="ViViT",
                    help="a MODELS key, or a registry encoder name (optionally "
                         "'name:{json cfg}') to screen a host the fleet has no arm for")
    ap.add_argument("--aux", default="vicreg_tt",
                    help="a registered aux kind, or 'path/to/module.py:ClassName' for a "
                         "candidate under screening that is deliberately not registered")
    ap.add_argument("--fixture", action="store_true",
                    help="decode stimulus clips instead of a capture. PLUMBING ONLY.")
    ap.add_argument("--capture", type=Path, default=None, help="an obs_*.npz from capture_observations")
    ap.add_argument("--test-csv", type=Path, default=None,
                    help="the run's per-step test log. Defaults to the capture's own "
                         "run_dir/<condition>/logs/test_<condition>_0.csv")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=80)
    ap.add_argument("--framestack", type=int, default=2)
    args = ap.parse_args()

    if not args.fixture and args.capture is None:
        raise SystemExit("[replay] give --capture <npz>, or --fixture for a plumbing check")

    import torch

    if args.fixture:
        print("⛔ FIXTURE MODE: stimulus clips, NOT the agent's view (no lens warp, no chamber, "
              "no DLSS, both monitors). Numbers below are about the PLUMBING, not the model.")
        stream, train_frames, pairs, t = load_fixture(
            args.model, args.width, args.height, args.framestack)
        actions = None
    else:
        blob = np.load(args.capture, allow_pickle=False)
        if int(blob.get("n_transition_pairs", 0)) == 0:
            raise SystemExit("[replay] capture has ZERO transition pairs -- recapture with --window >= 2")
        if bool(blob.get("is_prefix", False)):
            print("⚠ capture is flagged is_prefix: an ORDERED schedule cut short covers some "
                  "conditions and not others. Treat every number below as provisional.")
        obs = blob["obs"]
        stream = torch.as_tensor(obs[:, None])          # (T, 1, C, H, W)
        actions = torch.as_tensor(blob["actions"][:, None]) if len(blob["actions"]) else None
        csv_path = args.test_csv or default_test_csv(
            str(blob["run_dir"]), str(blob["condition"]), capture=args.capture)
        print(f"[replay] joining labels from {csv_path}")
        memory_idx, episodes, join_report = build_capture_pairs(blob, csv_path)
        train_frames, pairs = obs, []

    obs_shape = tuple(stream.shape[2:])
    results, baselines = {}, {}
    for seed in range(args.seeds):
        enc = build_encoder(args.model, obs_shape, seed)
        n_par = sum(p.numel() for p in enc.parameters())
        # ⛔ UNTRAINED CONTROL, ALWAYS. A frozen RANDOM encoder already separates images
        # by low-level statistics -- luminance, colour, contrast -- and the readout cannot
        # tell that apart from learned object identity. Without this baseline the trained
        # number is uninterpretable: it would credit the objective for whatever raw pixels
        # already gave away. The quantity of interest is the DELTA.
        def read_out(e):
            if args.fixture:
                return familiarity_readout(e, train_frames, pairs) if pairs else {}
            # n_episodes is identical before and after -- the same episodes are scored by
            # both encoders -- so it is carried in the label, not averaged as a number.
            return {f"{c} (n={n})": acc for c, (acc, n) in
                    capture_readout(e, obs, memory_idx, episodes).items()}

        before = read_out(enc)
        aux, losses = train_offline(enc, args.aux, stream, actions, args.steps, args.lr, seed)
        scored = read_out(enc)
        for k, v in scored.items():
            results.setdefault(k, []).append(v)
            baselines.setdefault(k, []).append(before[k])
        print(f"[replay] seed {seed}: encoder={args.model} params={n_par:,} aux={args.aux} "
              f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")
        for k in scored:
            print(f"           {k:24s} untrained {before[k]:.3f} -> trained {scored[k]:.3f} "
                  f"(delta {scored[k]-before[k]:+.3f})")

    if results:
        print(f"\n[replay] over {args.seeds} seeds ({'FIXTURE' if args.fixture else 'capture'}):")
        for k, vals in results.items():
            base = baselines[k]
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
            bsd = float(np.std(base, ddof=1)) if len(base) > 1 else float("nan")
            print(f"  {k:24s} untrained {np.mean(base):.4f} (sd {bsd:.4f})  "
                  f"trained {np.mean(vals):.4f} (sd {sd:.4f})  "
                  f"delta {np.mean(vals)-np.mean(base):+.4f}  n={len(vals)}")
        print("⚠ READ THE DELTA, NOT THE TRAINED COLUMN. A random encoder separates images by "
              "luminance and colour alone; only the delta is attributable to the objective.")
        print("⚠ sd here is the (a) seed component ALONE -- the data is identical across seeds. "
              "Compare it to the fleet's 0.058 same-specification mean gap to size what replay removes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
