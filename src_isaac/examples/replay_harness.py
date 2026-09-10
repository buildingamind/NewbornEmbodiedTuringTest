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


def build_encoder(model_name: str, obs_shape, seed: int):
    import gymnasium as gym
    import torch
    from campaign_train import MODELS
    from nett_skrl.brain.registry import encoder_mapping

    torch.manual_seed(seed)
    spec = MODELS[model_name]
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="ViViT")
    ap.add_argument("--aux", default="vicreg_tt",
                    help="a registered aux kind, or 'path/to/module.py:ClassName' for a "
                         "candidate under screening that is deliberately not registered")
    ap.add_argument("--fixture", action="store_true",
                    help="decode stimulus clips instead of a capture. PLUMBING ONLY.")
    ap.add_argument("--capture", type=Path, default=None, help="an obs_*.npz from capture_observations")
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
        train_frames, pairs = obs, []
        print("⚠ capture-mode readout needs the label join (env_id, episode) against the run's "
              "test CSV; not implemented. Training runs; readout is fixture-only for now.")

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
        before = familiarity_readout(enc, train_frames, pairs) if pairs else {}
        aux, losses = train_offline(enc, args.aux, stream, actions, args.steps, args.lr, seed)
        scored = familiarity_readout(enc, train_frames, pairs) if pairs else {}
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
