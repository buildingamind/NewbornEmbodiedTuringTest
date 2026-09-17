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

⛔ OBJECTIVE CHANGE 2026-09-17 (owner, workspace DECISIONS): `cltt_ref` now excludes each anchor's OWN FRAME from its negatives, so every cltt_ref arm trained before this commit ran a different objective and is NOT comparable to one trained after it.
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
from .knobs import NOT_MEASURED
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




def same_frame_mask(batch: int, offset: int, device=None) -> tuple[torch.Tensor, int]:
    """(2B, 2B) boolean mask of entries whose ROW AND COLUMN ARE THE SAME FRAME, and its count.

    ⛔ ONE DERIVATION, TWO CONSUMERS -- the loss and the diagnostic. Deriving it twice is how the
    two drift, and the diagnostic's job is to describe the loss.

    Derived from the slicing in `CLTTReferenceAuxLoss.compute`, not from the symptom it caused::

        views[m] = raw[t0 + k_m : t0 + batch + k_m]        k_0 = 0, k_m = offsets[m-1]

    so for the pair (view_0, view_k) that forms one similarity matrix:

        row i  (i < B)   is frame t0 + i;      z2[j] is frame t0 + k + j, so the SAME frame sits
                         at j = i - k, i.e. column B + (i - k) -- which exists iff i >= k;
        row B + j        is frame t0 + k + j;  the same frame sits at column j + k in the first
                         half -- which exists iff j + k < B. That is the transpose of the above,
                         so the mask is symmetric by construction and is asserted to be.

    ⇒ 2(B - k) entries when 0 < k < B; the k edge rows of each half have no duplicate, and at
    k >= B nothing does. The count is RETURNED rather than assumed: a caller that logs "masked"
    while masking nothing is the failure this signature exists to prevent.

    ⚠ THE DIAGONAL IS NOT THIS. `fill_diagonal_` masks a row against its own embedding; this
    masks a row against ANOTHER INDEX holding the same frame. Both are needed and they are
    different facts -- the diagonal is exact self-similarity, this is an exact duplicate whose
    two encodings can differ (and if they do, the tie breaks and the loss can fall below
    n_offsets*ln2 without any temporal invariance being learned).
    """
    if batch < 1:
        raise ValueError(f"batch={batch} has no similarity matrix.")
    if offset <= 0:
        raise ValueError(
            f"offset={offset} is not a frame offset: at 0 the two views are the same slab and "
            f"the 'duplicate' IS the positive, which this mask must never remove.")
    n = 2 * batch
    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    if offset >= batch:
        return mask, 0                     # no i satisfies both i >= offset and i < batch
    rows = torch.arange(offset, batch, device=device)
    cols = batch + rows - offset
    flat = torch.cat([rows * n + cols, cols * n + rows])
    mask.view(-1).scatter_(0, flat, torch.ones_like(flat, dtype=torch.bool))
    return mask, int(flat.numel())


def positive_alias_mask(batch: int, offset: int, device=None) -> tuple[torch.Tensor, int]:
    """(2B, 2B) mask of columns holding the same frame as the ROW'S POSITIVE, and its count.

    ⛔ WITHOUT THIS THE DIAGNOSTIC STAYS PINNED, JUST ON A DIFFERENT ALIAS, and the requested
    control ("a perfect encoder must read pos_acc ~ 1") cannot pass however good the encoder is.
    Derivation, from the same slicing::

        row i < B    is frame t0+i, its positive is z2[i] = frame t0+k+i -- and THAT FRAME is
                     also in the FIRST half, at column i+k, whenever i+k < B;
        row B+j      is frame t0+k+j, its positive is z1[j] = frame t0+j -- also in the SECOND
                     half at column B+(j-k), whenever j >= k.

    Those columns are exact copies of the positive, so their similarity to the anchor EQUALS the
    positive's, to the last bit. `argmax` breaks that tie by index and the lower index wins, so
    the prediction lands on the copy and `pos_acc` reads 0 for an encoder that has in fact put
    the right frame on top.

    ⇒ The diagnostic counts a hit when the argmax lands on ANY column holding the positive's
    frame -- the question it exists to ask is "did the encoder rank the frame at t+k above every
    OTHER frame", and both columns are that frame. It is deliberately NOT removed from the loss:
    doing that would delete real negatives from the objective, which is a second owner decision
    and not the one taken on 2026-09-17.
    """
    if batch < 1:
        raise ValueError(f"batch={batch} has no similarity matrix.")
    if offset <= 0:
        raise ValueError(f"offset={offset} is not a frame offset.")
    n = 2 * batch
    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    if offset >= batch:
        return mask, 0
    rows = torch.arange(0, batch - offset, device=device)
    first = rows * n + (rows + offset)                       # row i -> column i+k
    second = (rows + batch + offset) * n + (batch + rows)    # row B+j (j>=k) -> column B+j-k
    flat = torch.cat([first, second])
    mask.view(-1).scatter_(0, flat, torch.ones_like(flat, dtype=torch.bool))
    return mask, int(flat.numel())


def nt_xent_same_frame_masked(z1: torch.Tensor, z2: torch.Tensor, temperature: float,
                              offset: int) -> tuple[torch.Tensor, int]:
    """NT-Xent with each anchor's OWN FRAME removed from its negatives. -> (loss, masked count).

    ⛔ OWNER DECISION, 2026-09-17 (workspace DECISIONS): "the CLTT objective should exclude an
    anchor's own frame from its negatives". Every cltt_ref arm trained BEFORE that decision ran
    the unmasked objective and is NOT comparable to one trained after it.

    ⛔ WHAT IT CHANGES, AND WHY IT IS NOT COSMETIC. Unmasked, each anchor's own frame sat among
    its negatives at the maximum similarity a temporally invariant encoder can produce, so the
    objective was partly "push this frame away from itself at another index" -- and it could be
    minimised by embedding ONE frame differently in the two views (breaking the tie) rather than
    by learning invariance. With the duplicate removed that route is gone, and the
    n_offsets*ln2 tie floor no longer applies: `chance` = n_offsets*ln(2B-1) is the only floor.

    ⛔ THIS IS NOT `simclr_aux.nt_xent` AND MUST NOT BE FOLDED INTO IT. The SimCLR arms pair two
    AUGMENTATIONS of one image; their 2B set contains no duplicate frame, so this mask would
    remove real negatives from a live objective.
    """
    batch = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim.fill_diagonal_(float("-inf"))
    mask, count = same_frame_mask(batch, offset, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))     # out-of-place: the graph keeps its gradient
    labels = (torch.arange(2 * batch, device=z.device) + batch) % (2 * batch)
    return F.cross_entropy(sim, labels), count


_DIAG_GENERATORS: dict = {}


def _diag_generator(device) -> torch.Generator:
    """One generator per device, private to the diagnostics.

    ⛔ THE DIAGNOSTIC MUST NOT MOVE THE TRAINING RNG STREAM. It runs under `no_grad` and cannot
    change a gradient in its own step, but drawing from the global generator would change every
    window drawn AFTER it -- so `NETT_AUX_CLTT_REF_DIAG=1` would silently be a different run
    from `=0`. Seeded from a constant, so a null is reproducible across processes.
    """
    g = _DIAG_GENERATORS.get(device)
    if g is None:
        g = torch.Generator(device=device)
        g.manual_seed(20260917)
        _DIAG_GENERATORS[device] = g
    return g


def nt_xent_diagnostics(z1: torch.Tensor, z2: torch.Tensor, temperature: float,
                        *, duplicate_offset: int | None = None) -> dict:
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

    `shuffled_acc` is the null: the same embeddings scored against a RANDOM VALID WRONG
    candidate. `pos_acc` at or below it means the temporal offset carried no information --
    the reading the vicreg-tt+ kill was really about, expressed for a contrastive objective.

    ⛔ CORRECTION, 2026-09-17 (workspace FINDINGS §4bw.1). Both readings above were unusable as
    shipped, and the numbers they produced in the field were reports about this function, not
    about any encoder. The views are contiguous slabs of ONE stream -- `view_k[i] IS view_0[i+k]`,
    the same frame, not a similar one -- so each anchor's own frame sat in the opposite half at
    the maximum similarity a temporally invariant encoder can produce, the argmax was pinned
    there, and `pos_acc` collapsed to ~2e-4, BELOW its own reported chance of 1.005e-3. The
    `labels + 1` null aliased that same duplicate at offset 1 and never at offset 2, so it read
    (B-1)/4B = 0.2495 on every arm -- a null 248x above chance beside a signal below it, which
    is an instrument signature and not a result. Both scalars were two readings of one bit.

    ⇒ The candidate set now EXCLUDES each row's duplicate, `pos_chance` is 1/|candidates| rather
    than 1/(2B-1), and the null is a random valid wrong candidate that cannot land on a
    duplicate. `duplicates` reports how many rows had one, so this version's numbers are
    distinguishable from the old ones at a glance.

    ⚠ READ THE LOSS AGAINST TWO FLOORS, NOT ONE. `chance` is n_offsets*ln(2B-1), the random
    floor. But a pair that contains the SAME frame twice can be driven below n_offsets*ln2 by
    embedding that one frame DIFFERENTLY in the two views -- breaking the tie rather than
    learning invariance. A loss under the tie floor is therefore evidence AGAINST temporal
    invariance, not for it; measured on a real arm (C72, all seven brains).

    ⚠ Every field is emitted on EVERY call. A diagnostic written only where it succeeds makes
    "engaged" and "fell through" both present as absent, and absent reads as benign.
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim.fill_diagonal_(float("-inf"))
    labels = (torch.arange(2 * B, device=z.device) + B) % (2 * B)

    # ⛔ EXCLUDE THE ANCHOR'S OWN FRAME FROM ITS OWN CANDIDATE SET. Derivation, from the slicing
    # in `compute` rather than from the symptom: view_m[i] = raw[t0 + k_m + i], so for the pair
    # (view_0, view_k) passed here,
    #     row i < B   is frame t0 + i      and the SAME frame sits in the second half wherever
    #                 t0 + k + j = t0 + i, i.e. at column B + (i - k), which exists iff i >= k;
    #     row B + j   is frame t0 + k + j  and the same frame sits in the first half at column
    #                 j + k, which exists iff j + k < B.
    # ⇒ 2(B - k) of the 2B rows carry an exact duplicate of their own anchor (max abs pixel
    # difference 0.0, not "similar"), and its similarity is the maximum a temporally invariant
    # encoder can produce -- so the argmax is pinned there and `pos_acc` collapses BELOW its own
    # reported chance. The k edge rows in each half have no duplicate.
    # ⇒ THE SAME MASK THE LOSS USES. `same_frame_mask` is the single derivation; a second one
    # here is how a diagnostic comes to describe an objective the code no longer runs.
    if duplicate_offset is None:
        dup_mask = torch.zeros_like(sim, dtype=torch.bool)
        n_dup = NOT_MEASURED           # the caller did not declare an overlap: unmeasured, not 0
    else:
        dup_mask, n_masked = same_frame_mask(B, int(duplicate_offset), device=z.device)
        n_dup = float(n_masked)
    sim = sim.masked_fill(dup_mask, float("-inf"))

    pred = sim.argmax(dim=1)
    # A hit is the positive OR any other column holding the positive's frame; see
    # `positive_alias_mask` for why the plain equality reads 0 for a perfect encoder.
    hit = torch.zeros_like(sim, dtype=torch.bool)
    hit.scatter_(1, labels[:, None], True)
    if duplicate_offset is not None:
        alias_mask, n_alias = positive_alias_mask(B, int(duplicate_offset), device=z.device)
        hit |= alias_mask
    else:
        alias_mask = torch.zeros_like(hit)
        n_alias = NOT_MEASURED
    pos_acc = float(hit.gather(1, pred[:, None]).float().mean())
    # The candidate set is what is LEFT: 2B minus the row itself, minus its duplicate where one
    # exists. Chance is the mean of 1/|candidates| over rows, not 1/(2B-1) -- that number
    # presumed 2B distinct samples, which this slab has never supplied.
    # The alias is a candidate the row can legitimately land on, so it stays in the count.
    candidates = (2 * B - 1) - dup_mask.sum(dim=1)
    chance = float((1.0 / candidates.float()).mean())

    # ⛔ THE NULL IS A RANDOM VALID WRONG CANDIDATE, NOT `labels + 1`. The shift aliased the
    # duplicate exactly when k = 1 (column B + i + 1 vs the duplicate at B + i - 1 -- they meet
    # under the +1 when the offsets are read across both halves) and never when k = 2, so the
    # "null" averaged (B-1)/4B = 0.2495 over two offsets and was reporting the same pinned
    # argmax the signal was. A null that can land on the duplicate measures the duplicate.
    # Drawn from a DEDICATED generator so the diagnostic consumes none of the training RNG
    # stream: with the global generator, turning the diagnostic on would change which windows
    # later updates draw, and this must not touch training at all.
    valid = torch.ones_like(sim, dtype=torch.bool)
    valid.fill_diagonal_(False)
    valid.scatter_(1, labels[:, None], False)
    valid &= ~dup_mask & ~alias_mask      # a null that can land on a copy of the positive

    noise = torch.rand(sim.shape, device=z.device, generator=_diag_generator(z.device))
    deranged = noise.masked_fill(~valid, -1.0).argmax(dim=1)
    shuffled_acc = float((pred == deranged).float().mean())

    raw_full = (torch.mm(z, z.t()))          # untempered, BEFORE any masking
    # ⛔ THE ONE STATISTIC THAT SEPARATES THE TWO WAYS A LOW LOSS CAN HAPPEN. `pos_acc` cannot:
    # an encoder that has learned temporal invariance and an encoder that has broken the tie by
    # embedding ONE frame differently in the two views both score high. But a pure function of
    # the image gives cos(z1[i], z2[i-k]) = 1 EXACTLY for a duplicate, because it is the same
    # image; anything below 1 means the embedding depends on which view the frame arrived in
    # (batch statistics are the live route -- BatchNorm in train() sees two different slabs).
    # Measured on a 600-step fixture where the tie broke: dup_sim 0.9910 BELOW pos_sim 0.9943.
    dup_sim = float(raw_full[dup_mask].mean()) if bool(dup_mask.any()) else NOT_MEASURED

    raw = sim * temperature
    pos_sim = float(raw.gather(1, labels[:, None]).mean())
    off = torch.ones_like(raw, dtype=torch.bool)
    off.fill_diagonal_(False)
    off.scatter_(1, labels[:, None], False)
    # ⚠ The duplicate is not a negative in any sense the number is read for -- it is the anchor
    # itself -- so it leaves `neg_sim` too, or the mean negative similarity is inflated by the
    # single largest entry in the row.
    off &= ~dup_mask
    neg_sim = float(raw[off].mean())
    return {
        "pos_acc": pos_acc,
        "shuffled_acc": shuffled_acc,
        # ⛔ NOT "chance". That key is the LOSS floor, n_offsets*ln(2B-1), and this dict is
        # merged into the same `last_scalars`: the diagnostic's probability silently overwrote
        # it, so the loss floor was lost on exactly the arms carrying the diagnostic.
        "pos_chance": chance,
        # ⛔ READ pos_acc AGAINST THIS, NOT AGAINST 1.0. On a contiguous slab the frame at
        # t - k is a candidate too, and for a TIME-HOMOGENEOUS encoder (similarity a function of
        # |Δt| alone -- which is what "learned temporal invariance at lag k" means) it is EXACTLY
        # as similar to the anchor as the positive at t + k. The argmax breaks that tie by index,
        # so half the rows miss whatever the encoder has learned. Only the k rows at each half's
        # start have no backward twin, which gives the closed form
        #     ceiling = 0.5 + k / 2B
        # verified to four decimals at (B,k) = (12,2), (20,3), (30,5) and (16,1) against a
        # lookup-table encoder whose similarity peaks exactly at lag k.
        # ⚠ IT IS A BOUND UNDER SYMMETRIC SIMILARITY, NOT A LAW. The fixture that verifies it is
        # symmetric by construction, so it establishes the arithmetic GIVEN the assumption, not
        # that the assumption holds of any real encoder. A reading materially above the ceiling
        # means sim(t, t+k) > sim(t, t-k) systematically -- which a direction-sensitive STIMULUS
        # can produce as readily as a view-dependent encoder (approach and retreat are not
        # mirror images). Above the ceiling is a question to ask, not a verdict.
        "pos_ceiling": (0.5 + duplicate_offset / (2.0 * B)) if duplicate_offset is not None
                       else NOT_MEASURED,
        "duplicates": n_dup,
        "positive_aliases": n_alias if n_alias == NOT_MEASURED else float(n_alias),
        "dup_sim": dup_sim,
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
        masked_pairs = 0
        for offset, view in zip(self.offsets, views[1:]):
            z_pos = self.head(encoder.encode_prepared(view))
            # ⛔ OWNER DECISION 2026-09-17: the anchor's own frame is not one of its negatives.
            # This CHANGES THE OBJECTIVE -- see `nt_xent_same_frame_masked`. Arms trained before
            # this commit are not comparable to arms trained after it.
            offset_loss, masked = nt_xent_same_frame_masked(
                z_anchor, z_pos, self.temperature, offset)
            total = total + offset_loss
            masked_pairs += masked
            if self.diag:
                with torch.no_grad():
                    # The offset IS the overlap: view_offset[i] is view_0[i + offset], the same
                    # frame. The diagnostic needs it to know which candidates are duplicates.
                    diags.append(nt_xent_diagnostics(z_anchor, z_pos, self.temperature,
                                                     duplicate_offset=offset))
        # Averaged over offsets, and emitted whether or not the diagnostic is on, so a reader
        # can tell "off" from "on and degenerate". See nt_xent_diagnostics.
        self.last_diag = ({k: sum(d[k] for d in diags) / len(diags) for k in diags[0]}
                          if diags else None)
        # ⛔ PUBLISH. `last_diag` alone reached no reader: nothing in ppo_aux read it, so
        # the diagnostic ran and vanished. `last_scalars` is the channel ppo_aux tracks.
        # B, t_max and avail go out ALWAYS -- not only when the diag is on -- because they
        # are what makes any level claim about this loss checkable, and because they vary
        # across updates while the startup log fires once.
        # ⛔ THE REGIME GOES OUT ON EVERY CALL, so an old arm and a new one are distinguishable
        # from tfevents alone -- the objective changed mid-wave and the loss VALUE alone cannot
        # say which one produced it. `masked_pairs` is the realised count, not the intent: at
        # offset >= B it is legitimately 0, and a regime flag without its count would report
        # masking that did not happen.
        self.last_scalars = {"B": float(batch), "t_max": float(t_max),
                             "same_frame_masked": 1.0,
                             "masked_pairs": float(masked_pairs),
                             "masked_frac": float(masked_pairs)
                             / float(len(self.offsets) * (2 * batch) ** 2),
                             "chance": float(len(self.offsets)
                                             * math.log(2 * batch - 1))}
        if self.last_diag:
            self.last_scalars.update({k: float(v) for k, v in self.last_diag.items()})
        return total
