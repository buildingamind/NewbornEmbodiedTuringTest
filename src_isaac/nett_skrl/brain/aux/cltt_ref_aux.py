"""CLTT reference rebuild as an AUXILIARY LOSS, under the distinct ``cltt_ref`` key.

WHAT IS FAITHFUL. buildingamind/ChicksAndDNNs_ViewInvariance draws 512 CONTIGUOUS
sliding windows from ONE image stream (dataloader shuffle defaults to False).
Its window_size=3 compares [t, t+1, t+2] with TWO positive offsets, SUMMED:
nt_xent(z1, z2) + nt_xent(z1, z3). We retain one-stream contiguous negatives,
backbone gradients, the Linear(512,512)->BatchNorm->ReLU->Linear(512,128,bias=False)
L2-normalised projector, and the effective CLI temperature 0.5.

NETT_AUX_BATCH now defaults to 512 and NETT_AUX_CLTT_REF_OFFSETS to "1,2",
matching training_jobs/train_simclr.sh and models/simclr.py shared_step(v0).
Old logs used 96 anchors and offsets (2,4) on whole stacks. Each view now uses
only the observation's CURRENT RGB frame. For a host encoder requiring T slots,
we repeat that frame across those slots, avoiding overlapping-stack shortcuts
without replacing the supplied encoder. This static-input adapter is a host
departure, especially for motion encoders, which see no within-view motion.

Other host departures: the encoder/preprocessing, PPO optimizer, learning rate,
schedule and update count remain host-controlled, rather than the reference's
offline ResNet training. The unchanged reset-aware sampler draws a random safe
contiguous slab, rather than walking the offline dataloader sequentially; short
episodes/buffers reduce B below 512 and last_scalars reports the effective B.
Episode boundaries and rollout ring seams are never crossed.

TWO REFERENCE DEFECTS. The shipped job's --aug False ENABLES augmentation: train_simclr.py
declares --aug with type=bool, so bool("False") is True and the
``if args.aug is True:`` branch applies SimCLRTrainDataTransform. We follow the
script's evident intent (NO augmentation), which differs from what the reference
binary actually ran. Also, its ``neg - math.e`` self-subtraction is wrong at
T=0.5: self-similarity contributes exp(1/T)=exp(2)=7.3891, but subtracting only
2.7183 leaves 4.67 in every denominator. Our unchanged nt_xent masks the diagonal
exactly instead of preserving this defect. These corrections prevent numerical
reproduction of the reference binary. Temperature 0.5 is already faithful to
add_model_specific_args (the constructor's unused default is 0.1).

Verified sources (main):
https://github.com/buildingamind/ChicksAndDNNs_ViewInvariance/blob/main/models/simclr.py
https://github.com/buildingamind/ChicksAndDNNs_ViewInvariance/blob/main/train_simclr.py
https://github.com/buildingamind/ChicksAndDNNs_ViewInvariance/blob/main/training_jobs/train_simclr.sh

THE NEGATIVES MAY BE THE PROBLEM, NOT THE FIX. Drawing a CONTIGUOUS block off one env stream makes
every negative a near-in-time frame of the SAME two-object world. The fleet's viewpoint plan v14
raises exactly this against NT-Xent here: "NT-Xent will push apart frames that may show the same
object at a different viewpoint -- actively penalising the invariance under test. CLTT tolerates
this because its sequences wander over 12-100 objects; a NETT rollout is a two-object world." So
this rebuild makes the reference's batch composition FAITHFUL and, by the same move, makes that
hazard MAXIMAL. That is deliberate: it is the controlled test of whether fidelity was what the
CLTT family lacked. If cltt_ref trails today's `cltt`, this paragraph is the first place to look,
and `vicreg` (no negatives) is the designed alternative.

The needs_memory / attach_memory hook was chosen precisely to AVOID the interface
change plan v14 parks in Tier 2 ("an objective that needs a signature change across
five compute() implementations is a code change with its own review"). compute()'s
signature is unchanged and the other four losses are untouched. The incumbent
``cltt`` also stays byte-identical: already-scored arms must keep their definition.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import logger

# Deliberately do NOT port the reference's `neg - math.e`: the self term is
# exp(1/T), equal to e only at T=1. At its T=0.5, exp(2)=7.3891 minus 2.7183
# leaves 4.67 of self-similarity per row. Our nt_xent masks the diagonal exactly.
from .simclr_aux import nt_xent
from .cltt_views import current_frame_stack, resolve_channels_per_frame


def _env_flag(name: str, default: bool = False) -> bool:
    """Every new boolean knob goes through the same spellings (fleet convention)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class CLTTReferenceProjectionHead(nn.Module):
    """Reference Linear -> BatchNorm -> ReLU -> bias-free Linear, L2-normalised."""

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)




def nt_xent_diagnostics(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> dict:
    """Is the contrastive task SOLVABLE, and is it solvable for the right reason?

    ⛔ THE LOSS VALUE CANNOT ANSWER EITHER, AND cltt_ref HAS TWO OPPOSITE FAILURE MODES that
    a falling loss is consistent with. Gate A killed `vicreg-tt+` because its temporal pair
    moved the objective ~5% of what a plain augmentation did -- i.e. the pairing barely
    reached the loss at all. `cltt_ref` has never been asked the analogous question, and the
    vicreg_tt control does not transfer: that one compares the temporal positive against
    ANOTHER AUGMENTATION of the anchor, and cltt_ref augments nothing (see the module
    docstring -- the reference's --aug flag is inert and we follow its evident intent).

    So this asks the two questions that are actually open here:

    1. **TOO EASY.** Adjacent stacks off one env stream are nearly identical images. If
       `pos_acc` is ~1.0 from the first update, the softmax is solved by low-level frame
       similarity and the gradient carries no pressure toward anything about the object. A
       loss near zero and an objective that teaches nothing look the same from outside.
    2. **TOO HARD / ACTIVELY WRONG.** The module docstring already raises this: negatives are
       near-in-time frames of the SAME two-object world, so NT-Xent pushes apart the same
       object at a different viewpoint -- penalising the invariance under test. If `pos_acc`
       sits at chance (1/(2B-1)) the pairing is unusable.

    `shuffled_acc` is the null: the same embeddings scored against a DERANGED positive
    assignment. `pos_acc` at or below it means the temporal offset carried no information --
    the reading the vicreg-tt+ kill was really about, expressed for a contrastive objective.

    ⚠ Every field is emitted on EVERY call. A diagnostic written only where it succeeds makes
    "engaged" and "fell through" both present as absent, and absent reads as benign.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim.fill_diagonal_(float("-inf"))
    labels = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    pred = sim.argmax(dim=1)
    pos_acc = float((pred == labels).float().mean())
    # Deranged null: shift the positive assignment by one within each half, so every row is
    # scored against a DIFFERENT real embedding rather than against noise.
    shifted = (labels + 1) % (2 * B)
    shuffled_acc = float((pred == shifted).float().mean())
    raw = sim * temperature
    pos_sim = float(raw.gather(1, labels[:, None]).mean())
    off = torch.ones_like(raw, dtype=torch.bool)
    off.fill_diagonal_(False)
    off.scatter_(1, labels[:, None], False)
    neg_sim = float(raw[off].mean())
    return {
        "pos_acc": pos_acc,
        "shuffled_acc": shuffled_acc,
        "chance": 1.0 / (2 * B - 1),
        "pos_sim": pos_sim,
        "neg_sim": neg_sim,
        "batch": B,
    }

def episode_window_starts(memory, offsets):
    """Return a (start, env) validity mask for the WHOLE inclusive window.

    Done flags describe the transition OUT of an observation, so a done at the
    endpoint is allowed; one before it is not. Replay supplies (env, episode,
    step) keys instead. Never infer continuity from the environment axis alone.
    """
    t_max = memory.memory_size if memory.filled else memory.memory_index
    tensors = memory.tensors
    keys = tensors.get("keys")
    if keys is not None:
        keys = torch.as_tensor(keys[:t_max]).cpu()
        if keys.ndim != 3 or keys.shape[0] != t_max or keys.shape[-1] not in (2, 3):
            raise ValueError("Episode windows require keys shaped (T, env, 2 or 3)")
        boundary = (keys[1:, :, :2] != keys[:-1, :, :2]).any(-1)
        if keys.shape[-1] == 3:
            boundary |= keys[1:, :, 2] != keys[:-1, :, 2] + 1
    else:
        signals = []
        for name in ("terminated", "truncated"):
            signal = tensors.get(name)
            if signal is None and hasattr(memory, "get_tensor_by_name"):
                try:
                    signal = memory.get_tensor_by_name(name)
                except (KeyError, ValueError, AttributeError):
                    pass
            if signal is None:
                raise ValueError(
                    f"Episode windows require replay 'keys' or both 'terminated' "
                    f"and 'truncated'; missing '{name}' boundary signal"
                )
            signal = torch.as_tensor(signal[:t_max]).bool().cpu()
            if signal.ndim == 3 and signal.shape[-1] == 1:
                signal = signal.squeeze(-1)
            if signal.ndim != 2 or signal.shape[0] != t_max:
                raise ValueError(f"Episode boundary signal '{name}' must have shape (T, env[, 1])")
            signals.append(signal)
        if signals[0].shape != signals[1].shape:
            raise ValueError("Episode boundary signals terminated/truncated must have matching shapes")
        boundary = (signals[0] | signals[1])[:-1]
    # A filled circular rollout can have a chronological seam at its write index.
    seam = int(getattr(memory, "memory_index", 0))
    if memory.filled and 0 < seam < t_max:
        boundary[seam - 1] = True
    width = max(offsets)
    if width < 0:
        raise ValueError("Episode window offsets must be nonnegative")
    prefix = torch.cat([torch.zeros(1, boundary.shape[1], dtype=torch.long),
                        boundary.long().cumsum(0)])
    avail = max(0, t_max - width)
    return (prefix[width:width + avail] - prefix[:avail]) == 0


def episode_window_batch(memory, offsets, max_samples):
    """Largest usable contiguous batch and its safe (start, env) mask."""
    valid = episode_window_starts(memory, offsets)
    if not valid.numel():
        raise ValueError("No episode-contiguous temporal window with B_eff >= 2")
    times = torch.arange(1, len(valid) + 1)[:, None].expand_as(valid)
    last_invalid = torch.where(valid, 0, times).cummax(0).values
    batch = min(max_samples, int((times - last_invalid).max()))
    if batch < 2:
        raise ValueError("No episode-contiguous temporal window with B_eff >= 2")
    # Every anchor in a slab must be valid, including intervening observations.
    starts = episode_window_starts(memory, (max(offsets) + batch - 1,))
    return batch, starts


def draw_episode_window(starts, *, draw_single_start=True):
    """Uniform environment, then uniform safe start within that environment."""
    envs = starts.any(0).nonzero().flatten()
    env = int(envs[int(torch.randint(len(envs), ()).item())])
    times = starts[:, env].nonzero().flatten()
    pick = int(torch.randint(len(times), ()).item()) if draw_single_start or len(times) > 1 else 0
    return env, int(times[pick])


class CLTTReferenceAuxLoss(nn.Module):
    """Sum temporal NT-Xent terms over contiguous windows from one rollout stream."""

    needs_memory = True

    # ⭐ THE TWO VARIANT POINTS, named so a subclass changes them and nothing else. The
    # incumbent's values are unchanged and MUST stay unchanged: already-scored ViT-CLTT-Ref and
    # SimCLR-CLTT-Ref arms keep their definition. See CLTTReferenceStackAuxLoss for the variant
    # that gives a MOTION encoder something to read. [[a-knob-nothing-reads-runs-the-control]]
    OFFSETS_ENV = "NETT_AUX_CLTT_REF_OFFSETS"
    DEFAULT_OFFSETS = "1,2"

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        # GATE A control, OFF unless asked for. One extra similarity matrix per offset, no
        # extra encoder forward -- it reuses embeddings the loss already computed.
        self.diag = _env_flag("NETT_AUX_CLTT_REF_DIAG")
        self.last_diag: dict | None = None
        self.last_scalars: dict = {}
        offsets = os.environ.get(self.OFFSETS_ENV, self.DEFAULT_OFFSETS)
        try:
            self.offsets = tuple(int(k.strip()) for k in offsets.split(","))
            if not self.offsets or any(k <= 0 for k in self.offsets):
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                f"{self.OFFSETS_ENV} must be a comma-separated list of "
                f"at least one positive integer; got {offsets!r}."
            ) from exc
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", "512"))
        self.temperature = float(os.environ.get("NETT_AUX_CLTT_REF_TEMP", "0.5"))
        self.head = CLTTReferenceProjectionHead(int(encoder.features_dim))
        self.head.to(next(encoder.parameters()).device)
        self._memory = None
        self.num_frames: int | None = None

    def attach_memory(self, memory) -> None:
        self._memory = memory

    def _make_views(self, views, encoder):
        """Reference-faithful: each view is the CURRENT RGB frame repeated across the T slots.

        ⛔ This is the line that decides what the objective is actually invariant to, and for a
        motion encoder it decides that the answer is "nothing temporal" -- the repeated slots make
        every view a still image. Overridden by CLTTReferenceStackAuxLoss.
        """
        return [current_frame_stack(view) for view in views]

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Ignore the PPO minibatch; draw temporal windows from attached memory."""
        if self._memory is None:
            raise RuntimeError(
                "CLTTReferenceAuxLoss draws its own temporal windows; "
                "call attach_memory(memory) before compute()."
            )
        memory = self._memory
        # Slice BEFORE transferring: HybridDeviceMemory.get_tensor_by_name would
        # copy the WHOLE ~2 GB observation buffer to the GPU, 160 times per update.
        raw = memory.tensors["observations"]
        t_max = memory.memory_size if memory.filled else memory.memory_index
        avail = t_max - max(self.offsets)
        # Upper bound only. The AUTHORITY on whether a window exists is
        # episode_window_batch below, which tests ADJACENCY. A count-based
        # refusal here would fire exactly on short captures and so would
        # shadow the contiguity check out of its own test domain -- and it
        # would blame max_samples/t_max for what is really a gap or a reset.
        batch = min(self.max_samples, avail)
        try:
            batch, starts = episode_window_batch(memory, self.offsets, batch)
        except ValueError as exc:
            # Re-raise with the loss-specific reason. The helper knows about
            # ADJACENCY; only this class knows why B>=2 matters to it.
            raise ValueError(
                f"{exc} (t_max={t_max}, offsets={self.offsets}, "
                f"NETT_AUX_BATCH={self.max_samples}). "
                "A contrastive softmax at B=1 is -log(1)=0 and teaches nothing."
            ) from exc
        env, t0 = draw_episode_window(starts, draw_single_start=avail >= self.max_samples)
        device = next(encoder.parameters()).device
        with torch.no_grad():
            views = [
                encoder._prepare_image(raw[t0 + k : t0 + batch + k, env].to(device))
                for k in (0, *self.offsets)
            ]
        if self.num_frames is None:
            # ⚠ NOT `// 3`. dvs_polarity makes one frame TWO channels, and a stack depth read
            # through the wrong divisor is reported in the log line below as fact.
            self.num_frames = views[0].shape[1] // resolve_channels_per_frame()
            # ⛔ THE BATCH BELONGS HERE. Every level claim about this objective depends on
            # B -- NT-Xent chance is 2*ln(2B-1) summed over two offsets -- and this line
            # used to emit the offsets and the stack depth and NOT the one parameter the
            # level turns on. A read protocol was published against an assumed B=96 while
            # the realised B at update 1 was ~45, because t_max is the memory FILL INDEX
            # until the buffer fills. The ValueError below already formats these exact
            # values, so the information existed in the file and was emitted only when the
            # run DIED. ⚠ B is not constant, so this one-time line calibrates update 1
            # only; the per-update series is published through `last_scalars`.
            logger.info(
                "CLTTReferenceAuxLoss: offsets=%s, stack depth T=%s, batch B=%s "
                "(t_max=%s, avail=%s, NETT_AUX_BATCH=%s, memory filled=%s) -> "
                "NT-Xent chance per offset ln(2B-1)=%.4f, summed over %s offsets=%.4f",
                self.offsets, self.num_frames, batch, t_max, avail, self.max_samples,
                bool(getattr(self._memory, "filled", False)),
                math.log(2 * batch - 1), len(self.offsets),
                len(self.offsets) * math.log(2 * batch - 1),
            )
        views = self._make_views(views, encoder)

        z_anchor = self.head(encoder.encode_prepared(views[0]))  # backbone grad ON
        total, diags = 0.0, []
        for view in views[1:]:
            z_pos = self.head(encoder.encode_prepared(view))
            total = total + nt_xent(z_anchor, z_pos, self.temperature)
            if self.diag:
                with torch.no_grad():
                    diags.append(nt_xent_diagnostics(z_anchor, z_pos, self.temperature))
        # Averaged over offsets, and emitted whether or not the diagnostic is on, so a reader
        # can tell "off" from "on and degenerate". See nt_xent_diagnostics.
        self.last_diag = ({k: sum(d[k] for d in diags) / len(diags) for k in diags[0]}
                          if diags else None)
        # ⛔ PUBLISH. `last_diag` alone reached no reader: nothing in ppo_aux read it, so
        # the diagnostic ran and vanished. `last_scalars` is the channel ppo_aux tracks.
        # B, t_max and avail go out ALWAYS -- not only when the diag is on -- because they
        # are what makes any level claim about this loss checkable, and because they vary
        # across updates while the startup log fires once.
        self.last_scalars = {"B": float(batch), "t_max": float(t_max),
                             "chance": float(len(self.offsets)
                                             * math.log(2 * batch - 1))}
        if self.last_diag:
            self.last_scalars.update({k: float(v) for k, v in self.last_diag.items()})
        return total
