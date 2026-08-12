"""L5b: linear probes on FROZEN RL-TRAINED features. What survives in the encoder?

`SIDE_LOCK_INVESTIGATION.md` Phase 13 narrowed "why does the encoder matter" to two
accounts that fit every measured number IDENTICALLY:

  (a) INDUCTIVE BIAS  -- the ViT builds conjunctive features the CNN does not.
  (b) WEAK CONVERGER  -- the ViT simply commits less hard to the dominant single
                         feature, and binding is the one condition where committing
                         fast is a trap.

They differ in a testable place: what the RL-trained CNN's features still CARRY.
Phase 11 showed both encoders CAN carry shape when trained supervised, so the
question is what reinforcement learning left behind.

  shape decodable from CNN features but unused by the policy
      -> the POLICY is stuck on a colour shortcut; the representation is intact,
         and account (a) is not needed to explain the binding gap.
  shape NOT decodable from RL-trained CNN features
      -> the shortcut was carved into the REPRESENTATION itself.

★ THREE PROBES, BUILT FROM THE REAL EXPERIMENT STIMULI (`videos/binding/videos/`),
not synthetic shapes. Each is a balanced two-way discrimination whose nuisance
dimensions are matched by construction -- verified from the pixels, see the table in
`STIMULI` below:

  colour   imprint vs `1Ca`/`1Cb`   object AREA matched (~55.5k px), colour differs
  shape    imprint vs `1Sa`/`1Sb`   object COLOUR matched (~206/139/138), shape differs
  binding  imprint vs `B`           BOTH marginals matched -- only the CONJUNCTION
                                    differs. This is the actual task, off-policy.

⚠ DOMAIN GAP, AND THE CONTROL THAT ANSWERS IT. These are the monitor's video frames,
not the agent's rendered chamber view (which would need a Kit boot). A frozen encoder
is therefore probed slightly off its training distribution, so ABSOLUTE accuracies are
not comparable to on-policy performance. The fix is not to trust them absolutely: every
arm is reported against a RANDOM-INIT encoder of the SAME architecture, which faces the
identical domain gap. If RL-trained shape accuracy falls BELOW random-init, no
domain-gap story explains it -- training removed the information. Likewise the
within-encoder colour-minus-shape gap is immune to any constant offset.

⚠ `NETT_AMP` is forced OFF here: `NatureCNN.forward` autocasts to bf16 on CUDA, which
would quantise the very features being probed, and differently for the CNN arms than
for the ViT ones.

Usage (no training, no Kit, minutes on one GPU):

    cd NewbornEmbodiedTuringTest/src_isaac
    NETT_PROBE_DEVICE=0 PYTHONPATH=. python examples/probe_frozen_features.py
    ... --agents 16 --arms CNN,ViT
"""
from __future__ import annotations

import argparse
import os

# ⚠ MUST precede the encoder import: NatureCNN reads NETT_AMP at forward() time and
# would otherwise hand back bf16-rounded features on CUDA. See the module docstring.
os.environ["NETT_AMP"] = "off"

import glob  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# ⚠ SELF-LOCATING, not cwd-dependent. `examples/` is not an importable package, so a bare
# `import campaign_train` only resolves when Python put THIS file's directory on sys.path --
# i.e. when the driver is run as a script. Loading it any other way (the test suite execs it
# by path; so does any notebook or wrapper) raised ModuleNotFoundError. Fixed 2026-08-12 on
# the port into this tree.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign_train import MODELS  # noqa: E402
from nett_skrl.brain.registry import encoder_mapping  # noqa: E402

RES = 128
FEATURES_DIM = 512
VIDEO_DIR = Path(__file__).resolve().parents[3] / "videos" / "binding" / "videos"

# ⚠ RUN ROOT IS A KNOB, THE RUN-DIR NAMES ARE PROVENANCE. The directory names below identify
# the exact arms these results came from and must not be edited casually; only the root they
# sit under is machine-specific. Override with NETT_RUN_ROOT if the outputs move or the host
# changes (they lived in $HOME on the machine that produced them). Absolute paths used to be
# baked in here, which made the driver silently unrunnable anywhere else.
RUN_ROOT = Path(os.environ.get("NETT_RUN_ROOT", str(Path.home())))

# arm label -> (run dir relative to RUN_ROOT, model label in campaign_train.MODELS). Only arms
# scored at n>=49 under the post-2026-07-30 defaults; pre-07-30 runs do not pool with these.
#
# ⚠ ViT-Sp / ViT-Mixer-Sp are SQUARE-EYE arms (128x128 -> 8x8 tokens). They are rebuilt here at
# RES=128 square, which is why they still load even though the live eye is 128x80 and they can
# no longer be TRAINED. See campaign_train.py's spatial-arm note.
ARMS: dict[str, tuple[str, str]] = {
    "CNN":          ("nett_cnn112_20260731/binding_CNN", "CNN"),
    "CNN 10k eps":  ("nett_cnn_budget10k_20260802/binding_CNN", "CNN"),
    "ViT-Mixer-Sp": ("nett_vit_mixer_sp_20260803/binding_ViTMixerSp", "ViT-Mixer-Sp"),
    "ViT-Sp (qk)":  ("nett_vit_sp_20260803/binding_ViTSp", "ViT-Sp"),
    "ViT (cls)":    ("nett_pack112_20260730/binding_ViT", "ViT"),
}

# probe -> (class-0 clips, class-1 clips). Nuisance matching measured from the pixels
# (mean object RGB over non-white pixels; object area in px), 2026-08-08:
#   imprint  206.1/138.7/137.2  55561        1Sa_1  205.4/141.1/142.5  46520
#   1Ca_1    206.9/ 92.3/142.3  55469        1Sa_2  206.0/137.8/137.4  58477
#   1Cb_1    115.4/155.2/193.2  55583        1Sb_1  206.8/141.3/139.5  48446
#   B        202.2/135.3/141.9  55560   <- matched on colour AND area
STIMULI: dict[str, tuple[list[str], list[str]]] = {
    "colour":  (["O1_imprint"], ["O1_1Ca_1", "O1_1Ca_2", "O1_1Cb_1", "O1_1Cb_2"]),
    "shape":   (["O1_imprint"], ["O1_1Sa_1", "O1_1Sa_2", "O1_1Sb_1", "O1_1Sb_2"]),
    "binding": (["O1_imprint"], ["O1_B"]),
}
N_FRAMES = 180        # every clip is 180 frames
# ★ ONLY SOME FRAMES CARRY THE MANIPULATION, and this is not optional bookkeeping --
# getting it wrong silently produces an all-chance table. Measured 2026-08-08, mean
# |pixel difference| from `O1_imprint` per frame:
#
#   frames   0-89    object present; `1Ca`/`1Sa` differ by up to 3.3, `B` by up to 9.5
#   frames  90-179   `1Ca`/`1Sa` are IDENTICAL to the imprint (diff 0.01); `B` still differs
#
# So the single-feature clips diverge in 80-82 of 180 frames and `O1_B` in 165. A frame
# where the two stimuli are pixel-identical carries ZERO label information; including it
# just injects label noise. `DIFF_FLOOR` drops those. This selects on the IMAGES only --
# never on the encoder, the features or any behavioural score.
DIFF_FLOOR = 1.0      # mean |grey level| difference below which a frame is uninformative
BLOCK = 6             # alternating train/test blocks, in frames
# ⚠ POSE IS ~50x THE CLASS SIGNAL (imprint f0 vs f90 = 21.6; imprint vs 1Ca at the same
# pose = 0.50). Any split that separates poses between train and test measures pose, not
# class -- an earlier first-half/second-half split scored `binding` at 0.02, i.e. almost
# perfectly INVERTED. Both classes are therefore sampled at the SAME frame indices, which
# makes pose uninformative about the label by construction, and train/test alternate in
# blocks so the two splits still cover different poses.


def load_clip(name: str) -> np.ndarray:
    """All frames of one stimulus clip as uint8 HWC at RES, in capture order."""
    path = VIDEO_DIR / f"{name}.mov"
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(cv2.resize(frame[:, :, ::-1], (RES, RES), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(out) != N_FRAMES:
        raise ValueError(f"{name}: expected {N_FRAMES} frames, got {len(out)}")
    return np.stack(out)


def informative_frames(cache: dict[str, np.ndarray], neg: str, pos: str) -> np.ndarray:
    """Frame indices where two clips actually differ. See ``DIFF_FLOOR``."""
    diff = np.abs(cache[neg].astype(np.float32)
                  - cache[pos].astype(np.float32)).mean(axis=(1, 2, 3))
    return np.flatnonzero(diff > DIFF_FLOOR)


def build_probe_set(cache: dict[str, np.ndarray], probe: str):
    """(train images, train labels, test images, test labels) for one probe.

    Each class-1 clip is paired with the class-0 clip at the SAME frame indices, so the
    two classes are pose-matched and pose cannot predict the label. Train and test then
    alternate in blocks of ``BLOCK`` frames.
    """
    (neg,), pos_clips = STIMULI[probe]
    per_clip = {}
    for pos in pos_clips:
        idx = informative_frames(cache, neg, pos)
        if len(idx) < 2 * BLOCK:
            raise ValueError(f"{probe}: {pos} differs from {neg} in only {len(idx)} frames")
        per_clip[pos] = idx
    # The imprint contributes ONE copy of each informative index however many foil clips
    # it is contrasted with; each foil is then thinned to keep the classes balanced.
    neg_idx = np.unique(np.concatenate(list(per_clip.values())))
    take = max(BLOCK * 2, len(neg_idx) // len(pos_clips))
    samples = [(neg, neg_idx, 0)]
    for pos, idx in per_clip.items():
        keep = idx[np.linspace(0, len(idx) - 1, min(take, len(idx)), dtype=int)]
        samples.append((pos, keep, 1))

    tr_x, tr_y, te_x, te_y = [], [], [], []
    for clip, idx, label in samples:
        frames, train = cache[clip], (idx // BLOCK) % 2 == 0
        tr_x.append(frames[idx[train]]); tr_y.append(np.full(int(train.sum()), label))
        te_x.append(frames[idx[~train]]); te_y.append(np.full(int((~train).sum()), label))
    return (np.concatenate(tr_x), np.concatenate(tr_y),
            np.concatenate(te_x), np.concatenate(te_y))


def rescale(imgs: np.ndarray, scale: float, seed: int, jitter: int = 3) -> np.ndarray:
    """Shrink each frame to ``scale`` and paste it into a full-size white field.

    ★ THE DIFFICULTY AXIS, AND IT IS NOT OPTIONAL. At scale 1.0 every encoder --
    INCLUDING A RANDOM-INIT ONE -- scores 0.93-0.99 on all three probes, so the table
    measures nothing (`SIDE_LOCK_INVESTIGATION.md` Phase 11 method rule: sweep the
    difficulty until the arms separate). Scale stands in for VIEWING DISTANCE: the
    monitor fills the frame here, but subtends far less of the agent's real view, so
    shrinking is also the more faithful condition. The pad is white because that is
    already the clips' own background -- padding with anything else would add a
    background statistic the encoder never saw.

    A few pixels of jitter stop the probe reading the answer off a fixed location.
    """
    if scale >= 1.0:
        return imgs
    rng = np.random.default_rng(seed)
    side = max(4, int(round(RES * scale)))
    out = np.full_like(imgs, 255)
    for i, img in enumerate(imgs):
        small = cv2.resize(img, (side, side), interpolation=cv2.INTER_AREA)
        room = RES - side
        ox = int(np.clip(room // 2 + rng.integers(-jitter, jitter + 1), 0, room))
        oy = int(np.clip(room // 2 + rng.integers(-jitter, jitter + 1), 0, room))
        out[i, oy:oy + side, ox:ox + side] = small
    return out


def build_side_probe_set(cache: dict[str, np.ndarray], scale: float = 0.4,
                         gap: int = 4, seed: int = 0):
    """★ THE TASK-SHAPED PROBE: two monitors, and the label is WHICH SIDE the imprint is on.

    The other three probes ask "which object is this?" -- but the agent never has to name
    an object, it has to turn toward one of two monitors. So identity decodability can be
    high while the quantity the policy actually needs is absent.

    Each frame index yields BOTH orderings -- imprint-left/foil-right and the swap -- so
    the two classes contain PIXEL-IDENTICAL content in exchanged positions. Colour, shape,
    area, pose and luminance are matched by construction, and nothing but position
    distinguishes the classes. Train/test alternate by frame block exactly as elsewhere.
    """
    neg, (pos,) = STIMULI["binding"][0][0], STIMULI["binding"][1]
    idx = informative_frames(cache, neg, pos)
    side = max(4, int(round(RES * scale)))
    top = (RES - side) // 2
    rng = np.random.default_rng(seed)

    def place(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        canvas = np.full((RES, RES, 3), 255, dtype=np.uint8)
        jx = int(rng.integers(-2, 3))
        lx = max(0, RES // 2 - gap - side + jx)
        rx = min(RES - side, RES // 2 + gap + jx)
        canvas[top:top + side, lx:lx + side] = cv2.resize(
            left, (side, side), interpolation=cv2.INTER_AREA)
        canvas[top:top + side, rx:rx + side] = cv2.resize(
            right, (side, side), interpolation=cv2.INTER_AREA)
        return canvas

    tr_x, tr_y, te_x, te_y = [], [], [], []
    for i in idx:
        imprint, foil = cache[neg][i], cache[pos][i]
        train = (i // BLOCK) % 2 == 0
        for label, img in ((0, place(imprint, foil)), (1, place(foil, imprint))):
            (tr_x if train else te_x).append(img)
            (tr_y if train else te_y).append(label)
    return (np.stack(tr_x), np.array(tr_y), np.stack(te_x), np.array(te_y))


def load_encoder(model_label: str, ckpt: Path | None, device: torch.device,
                 seed: int = 0) -> nn.Module:
    """Encoder for ``model_label``; RL weights from ``ckpt``, or random init if None.

    The checkpoint's ``policy`` state dict prefixes encoder tensors with ``encoder.``
    (the value network holds a separate copy -- the POLICY's encoder is the one that
    chose the actions, so it is the one probed).
    """
    spec = MODELS[model_label]
    space = gym.spaces.Box(low=0, high=255, shape=(RES, RES, 3), dtype=np.uint8)
    torch.manual_seed(seed)
    enc = encoder_mapping[spec["encoder"]](space, **spec["cfg"])
    if ckpt is not None:
        policy = torch.load(ckpt, map_location="cpu", weights_only=False)["policy"]
        weights = {k[len("encoder."):]: v for k, v in policy.items()
                   if k.startswith("encoder.")}
        if not weights:
            raise KeyError(f"{ckpt}: no 'encoder.*' tensors in the policy state dict")
        missing, unexpected = enc.load_state_dict(weights, strict=False)
        if missing or unexpected:
            raise KeyError(f"{ckpt}: state dict mismatch {missing=} {unexpected=}")
    return enc.to(device).eval()


@torch.no_grad()
def features(enc: nn.Module, imgs: np.ndarray, device: torch.device,
             batch: int = 64) -> torch.Tensor:
    out = []
    for i in range(0, len(imgs), batch):
        x = torch.from_numpy(imgs[i:i + batch]).to(device)
        out.append(enc(x).float().cpu())
    return torch.cat(out)


def linear_probe(tr_x: torch.Tensor, tr_y: np.ndarray, te_x: torch.Tensor,
                 te_y: np.ndarray, seed: int = 0, steps: int = 400) -> float:
    """Held-out accuracy of a multinomial logistic regression on frozen features.

    Features are standardised with TRAIN statistics only. Weight decay is deliberate:
    with 512 dimensions and 180 training rows the problem is over-parameterised, and an
    unregularised fit separates almost anything.
    """
    torch.manual_seed(seed)
    mu, sd = tr_x.mean(0, keepdim=True), tr_x.std(0, keepdim=True).clamp_min(1e-6)
    xtr, xte = (tr_x - mu) / sd, (te_x - mu) / sd
    ytr = torch.from_numpy(tr_y).long()
    head = nn.Linear(xtr.shape[1], 2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-3)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        lossf(head(xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        pred = head(xte).argmax(1).numpy()
    return float((pred == te_y).mean())


def agent_checkpoints(root: str, limit: int) -> list[tuple[str, Path]]:
    """(agent label, final_agent.pt) pairs, ordered to match ``isaac_agents`` output.

    ``isaac_agents`` sorts a run's CSV by int(brain_id) -> index i, and the trainer
    writes that agent's checkpoint to ``wandb_runs/brain_{i+1}``. So sorting runs by
    name and brains numerically reproduces the scoring order, which is what lets probe
    accuracy be joined to a per-agent binding score.
    """
    out = []
    for run in sorted(glob.glob(f"{root}/*/Object1/wandb_runs")):
        brains = sorted(Path(run).glob("brain_*"),
                        key=lambda p: int(p.name.split("_")[1]))
        for b in brains:
            ckpt = b / "checkpoints" / "final_agent.pt"
            if ckpt.exists():
                out.append((f"{Path(run).parents[1].name.split('_off')[-1]}/{b.name}", ckpt))
    if not out:
        raise FileNotFoundError(f"no final_agent.pt under {root}")
    return out[:limit]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", type=int, default=16, help="agents probed per arm")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated arm labels")
    ap.add_argument("--seeds", type=int, default=8, help="random-init control seeds")
    ap.add_argument("--scales", default="1.0,0.5,0.25,0.125",
                    help="stimulus scales to sweep; see rescale()")
    ap.add_argument("--dump", type=Path, default=None,
                    help="write per-agent accuracies here, for joining to binding scores")
    args = ap.parse_args()
    scales = [float(s) for s in args.scales.split(",") if s.strip()]

    device = torch.device(f"cuda:{os.environ.get('NETT_PROBE_DEVICE', '0')}"
                          if torch.cuda.is_available() else "cpu")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; choose from {sorted(ARMS)}")

    clips = sorted({c for pair in STIMULI.values() for clips in pair for c in clips})
    cache = {c: load_clip(c) for c in clips}
    sets = {p: build_probe_set(cache, p) for p in STIMULI}
    # ⚠ The side probe is already a two-monitor COMPOSITE, so `rescale` must not shrink it
    # again -- its own `scale` argument sets how large each monitor is. It is therefore
    # built per-scale inside the sweep, not here.
    print(f"device={device}  stimuli={VIDEO_DIR}")
    for p, (tr_x, tr_y, te_x, te_y) in sets.items():
        print(f"  {p:<8} train {len(tr_y):>3} ({int(tr_y.sum())} pos)"
              f"   test {len(te_y):>3} ({int(te_y.sum())} pos)")

    probes = list(STIMULI) + ["side"]
    sets["side"] = build_side_probe_set(cache)

    # ★ PIPELINE VALIDATION, PRINTED EVERY RUN. A raw-pixel probe must solve all three
    # discriminations (the stimuli DO differ), and a shuffled-label probe must sit at
    # chance. Without this block an all-chance encoder table looks like a finding; it
    # cost two wrong tables on 2026-08-08 before the frame-selection bug was found.
    print(f"\n{'CONTROL':<15}{'':>4}" + "".join(f"{p:>18}" for p in probes))
    rng = np.random.default_rng(0)
    for name, relabel in (("raw pixels", False), ("shuffled labels", True)):
        cells = []
        for p in probes:
            tr_x, tr_y, te_x, te_y = sets[p]
            flat = lambda a: torch.from_numpy(  # noqa: E731 - 16x16 RGB mean-pool
                a.astype(np.float32).reshape(len(a), 16, 8, 16, 8, 3)
                .mean((2, 4)).reshape(len(a), -1))
            y = rng.permutation(tr_y) if relabel else tr_y
            cells.append(linear_probe(flat(tr_x), y, flat(te_x), te_y))
        print(f"{name:<15}{'':>4}" + "".join(f"{c:>18.3f}" for c in cells))
        if not relabel and min(cells) < 0.9:
            raise SystemExit(f"raw-pixel probe failed ({cells}) -- the stimulus set is "
                             "broken; fix it before reading anything below")

    dump_rows: list[list] = []

    def probe_encoder(enc, scaled) -> dict[str, float]:
        return {p: linear_probe(features(enc, scaled[p][0], device), sets[p][1],
                                features(enc, scaled[p][2], device), sets[p][3])
                for p in probes}

    for scale in scales:
        scaled = {p: (rescale(sets[p][0], scale, seed=1), sets[p][1],
                      rescale(sets[p][2], scale, seed=2), sets[p][3]) for p in STIMULI}
        # the two monitors are each `scale`-sized within the same canvas
        side_set = build_side_probe_set(cache, scale=min(scale, 0.45))
        sets["side"] = side_set
        scaled["side"] = side_set
        px = max(4, int(round(RES * scale))) if scale < 1 else RES
        print(f"\n{'='*78}\nSTIMULUS SCALE {scale}  ({px}x{px} px of a {RES}x{RES} frame)")
        print(f"{'arm':<15}{'n':>4}" + "".join(f"{p:>18}" for p in probes)
              + f"{'colour-shape':>14}")

        results: dict[str, dict[str, list[float]]] = {}
        for arm in arms:
            rel, label = ARMS[arm]
            ckpts = agent_checkpoints(str(RUN_ROOT / rel), args.agents)
            per = {p: [] for p in probes}
            for agent_label, ckpt in ckpts:
                enc = load_encoder(label, ckpt, device)
                accs = probe_encoder(enc, scaled)
                for p, acc in accs.items():
                    per[p].append(acc)
                if args.dump:
                    dump_rows.append([arm, agent_label, scale]
                                     + [f"{accs[p]:.6f}" for p in probes])
                del enc
            results[arm] = per
            print(f"{arm:<15}{len(ckpts):>4}"
                  + "".join(f"{mean(per[p]):>12.3f}±{sd(per[p]):<5.3f}" for p in probes)
                  + f"{mean(per['colour']) - mean(per['shape']):>+14.3f}")

        ctrl: dict[str, dict[str, list[float]]] = {}
        for arm in arms:
            label = ARMS[arm][1]
            if label in ctrl:
                continue
            per = {p: [] for p in probes}
            for s in range(args.seeds):
                enc = load_encoder(label, None, device, seed=s)
                for p, acc in probe_encoder(enc, scaled).items():
                    per[p].append(acc)
                del enc
            ctrl[label] = per
            print(f"{'  random ' + label:<15}{args.seeds:>4}"
                  + "".join(f"{mean(per[p]):>12.3f}±{sd(per[p]):<5.3f}" for p in probes))

        print(f"  -- RL minus random-init "
              f"(below zero = training REMOVED the information) --")
        for arm in arms:
            label = ARMS[arm][1]
            print(f"  {arm:<28}"
                  + "".join(f"{mean(results[arm][p]) - mean(ctrl[label][p]):>+18.3f}"
                            for p in probes))

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w") as f:
            f.write("arm,agent,scale," + ",".join(probes) + "\n")
            for row in dump_rows:
                f.write(",".join(str(c) for c in row) + "\n")
        print(f"\nper-agent accuracies -> {args.dump} ({len(dump_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
