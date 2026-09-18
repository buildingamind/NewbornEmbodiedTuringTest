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

⛔ AND ITS FOLLOW-UP, THE SAME DAY (owner, relayed by commander): the POSITIVE's own frame leaves
the negatives too. A contiguous slab puts the frame at t+k in BOTH halves, so the unmasked
objective told one minibatch that f_t and f_{t+k} should be close (the positive term) and far
(that column) -- a contradiction, not a negative, and one whose only solution is to encode ONE
frame differently in the two views. `NETT_AUX_CLTT_MASK_POSITIVE_TWIN` (default ON) is the knob;
`mask_regime` in last_scalars is how a reader tells the THREE objectives apart afterwards. So
cltt_ref has now run three different losses, and a loss VALUE means nothing without the regime.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import logger

# Deliberately do NOT port the reference's `neg - math.e`: the self term is
# exp(1/T), equal to e only at T=1. At its T=0.5, exp(2)=7.3891 minus 2.7183
# leaves 4.67 of self-similarity per row. Our nt_xent masks the diagonal exactly.
from .knobs import NOT_MEASURED, _env_flag_strict
from .cltt_views import current_frame_stack, resolve_channels_per_frame

#: ⛔ THE CANDIDATE SET IS THE OBJECTIVE, AND IT HAS CHANGED TWICE IN ONE DAY. These are the
#: three losses `cltt_ref` has run, in order, and the value published as `mask_regime`:
#:   0  NONE              -- every index is a negative, including two copies of the anchor's own
#:                           frame and of its positive's. Everything before 2e30ad2.
#:   1  ANCHOR            -- the anchor's own frame leaves (owner decision, 2026-09-17).
#:                           2e30ad2..2f4c96b, and this build with MASK_POSITIVE_TWIN off.
#:   2  ANCHOR_AND_TWIN   -- the positive's own frame leaves as well (owner follow-up, same day).
#:                           This build's default.
#: ⚠ A FOURTH STATE IS "ABSENT", AND IT IS NOT 0. An arm that predates the tag emits no
#: `mask_regime` at all; reading a missing tag as 0 would be right only by luck, because
#: `same_frame_masked` (emitted from 2e30ad2) tells regime 0 and regime 1 apart on its own.
#: The four-way decode is written out at the emission site in `compute`.
MASK_REGIME_NONE = 0.0
MASK_REGIME_ANCHOR = 1.0
MASK_REGIME_ANCHOR_AND_TWIN = 2.0


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
    two encodings can differ. ⚠ THE ln2 TIE THAT USED TO HANG ON THIS IS THE POSITIVE'S TWIN'S,
    not this one's: see `nt_xent_same_frame_masked`. This mask removes an entry that is the
    anchor itself; the p(positive) <= 1/2 cap came from the OTHER copy, and is removed by
    `mask_positive_twin`.
    """
    cols, exists = same_frame_columns(batch, offset, device=device)
    return _mask_from_columns(cols, exists), int(exists.sum())


def same_frame_columns(batch: int, offset: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per ROW: the column holding that row's own frame, and whether it exists. O(B), not O(B^2).

    ⛔ THE INDEX FORM IS THE DERIVATION; the (2B, 2B) mask above is built FROM it. A dense mask
    at cltt_ref's B=498 is 1 MB and nobody notices, but `nt_xent_diagnostics` is also called by
    cltt_patch with 2BM = 7,872 rows, where every (2B, 2B) helper is 62 MB as bool and 248 MB as
    float32 -- measured as +740 MiB of GPU peak on the row that is already the wave's ceiling.
    There is exactly ONE such column per row, so a row-length index vector says the same thing.

    Where no duplicate exists the column is the ROW ITSELF, which is safe to scatter into: the
    diagonal is masked anyway. `exists` is what callers must gate on.
    """
    if batch < 1:
        raise ValueError(f"batch={batch} has no similarity matrix.")
    if offset <= 0:
        raise ValueError(
            f"offset={offset} is not a frame offset: at 0 the two views are the same slab and "
            f"the 'duplicate' IS the positive, which this mask must never remove.")
    n = 2 * batch
    rows = torch.arange(n, device=device)
    first = rows < batch
    exists = torch.where(first, rows >= offset, (rows - batch) + offset < batch)
    if offset >= batch:
        exists = torch.zeros_like(exists)
    cols = torch.where(first, batch + rows - offset, rows - batch + offset)
    return torch.where(exists, cols, rows), exists


def _mask_from_columns(cols: torch.Tensor, exists: torch.Tensor) -> torch.Tensor:
    """(2B, 2B) bool from one column per row. Only for callers that really want a matrix."""
    n = cols.shape[0]
    mask = torch.zeros(n, n, dtype=torch.bool, device=cols.device)
    mask.scatter_(1, cols[:, None], exists[:, None])
    return mask


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
    OTHER frame", and both columns are that frame.

    ⛔ AND SINCE THE OWNER'S FOLLOW-UP OF 2026-09-17 IT LEAVES THE LOSS TOO, under
    `NETT_AUX_CLTT_MASK_POSITIVE_TWIN` (default ON). The earlier note here said removing it
    "would delete real negatives"; that was wrong about WHICH negatives. This column is not
    another frame that happens to look like the positive -- it IS the positive's frame, so the
    same minibatch was asking for f_t close to f_{t+k} (the positive term) and far from it (this
    column). ⚠ With the twin masked the diagnostic's hit rule and the loss's candidate set
    AGREE; with the knob off they still disagree, deliberately, because that is the objective
    the comparison arm runs. Both are derived from this one function -- see `excluded_columns`.
    """
    cols, exists = positive_alias_columns(batch, offset, device=device)
    return _mask_from_columns(cols, exists), int(exists.sum())


def positive_alias_columns(batch: int, offset: int,
                           device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per ROW: the other column holding that row's POSITIVE's frame, and whether it exists."""
    if batch < 1:
        raise ValueError(f"batch={batch} has no similarity matrix.")
    if offset <= 0:
        raise ValueError(f"offset={offset} is not a frame offset.")
    n = 2 * batch
    rows = torch.arange(n, device=device)
    first = rows < batch
    exists = torch.where(first, rows + offset < batch, (rows - batch) >= offset)
    if offset >= batch:
        exists = torch.zeros_like(exists)
    cols = torch.where(first, rows + offset, rows - offset)
    return torch.where(exists, cols, rows), exists


def transposed_columns(cols: torch.Tensor, exists: torch.Tensor) -> tuple[torch.Tensor,
                                                                            torch.Tensor]:
    """The TRANSPOSE of a one-column-per-row relation, INVERTED from it -- not re-derived.

    ⛔ "Mask it, and its transpose" (owner, 2026-09-17). The relation "this column holds my
    positive's frame" is DIRECTIONAL: row i excludes column i+k, but row i+k does not thereby
    exclude column i, so the mask built from it alone is not symmetric and the loss pushes that
    pair apart from one side. The contradiction the decision is about belongs to the PAIR --
    nothing in this matrix may push two frames apart that the positive term pulls together -- so
    both entries go.

    ⚠ INVERTED, NOT WRITTEN OUT AGAIN. `row i -> column i+k` transposed is `row i+k -> column i`,
    which is a second index formula one edit away from disagreeing with the first. Scattering
    the relation into its inverse cannot disagree with it. Safe because the relation is an
    injection (exactly one column per row, all distinct); where the inverse is undefined the
    column is the row itself, which is already masked as the diagonal.
    """
    n = cols.shape[0]
    rows = torch.arange(n, device=cols.device)
    t_cols, t_exists = rows.clone(), torch.zeros(n, dtype=torch.bool, device=cols.device)
    t_cols[cols[exists]] = rows[exists]
    t_exists[cols[exists]] = True
    return t_cols, t_exists


def excluded_columns(batch: int, offset: int, *, mask_positive_twin: bool,
                     device=None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Every (row -> column) pair that LEAVES the candidate set, in index form. ONE DERIVATION.

    ⛔ THE LOSS AND THE DIAGNOSTIC MUST NOT DERIVE THIS TWICE. They did not drift only because
    the second consumer was written the same afternoon as the first; the next edit is where a
    "which columns are excluded" written in two places stops agreeing, and the failure is
    silent -- the loss trains on one candidate set while the diagnostic reports another's
    chance level. Everything downstream (the mask, the per-row candidate count, the loss floor,
    the diagnostic's argmax exclusions and its hit rule) is built from this list.

    Each entry is (column per row, exists per row) exactly as its deriving function returns it:
      * `same_frame_columns`     -- the ANCHOR's own frame  (owner decision, 2026-09-17)
      * `positive_alias_columns` -- the POSITIVE's own frame (owner follow-up, same day), and
      * its transpose            -- the same pair from the other side, so the mask is symmetric
                                    and no row pushes apart a pair the objective pulls together.
    The last two only when `mask_positive_twin`.

    ⚠ WHAT IS DELIBERATELY STILL A NEGATIVE: the frame at t - k, which is a DIFFERENT frame and
    a real negative. Its first-half copy leaves row i as the transpose above (that entry is the
    pair {f_{i-k}, f_i}, which IS a positive pair), but its second-half copy at B + (i - 2k)
    stays. Masking THAT too -- every lag-k pair everywhere -- drives the attainable loss to
    exactly 0.0000 (measured, B+k free embeddings at (8,1) and (12,2)): the task becomes
    trivially solvable and the objective teaches nothing, which is failure mode 1 in
    `nt_xent_diagnostics`. So the line is drawn where the owner drew it.

    ⚠ THE FAMILIES ARE PAIRWISE DISJOINT, AND THAT IS CHECKED HERE rather than assumed, because
    the per-row candidate count SUBTRACTS each: for row i < B the anchor's twin is at B + (i - k)
    and the positive's at i + k and the transpose at i - k -- opposite halves, and k > 0 apart
    within the half. If a future offset convention made two coincide, the count would silently
    over-subtract and every chance level derived from it would be wrong.
    """
    out = [same_frame_columns(batch, offset, device=device)]
    if mask_positive_twin:
        alias = positive_alias_columns(batch, offset, device=device)
        out.append(alias)
        out.append(transposed_columns(*alias))
        for i, (a_col, a_ex) in enumerate(out):
            for b_col, b_ex in out[i + 1:]:
                if bool((a_ex & b_ex & (a_col == b_col)).any()):
                    raise AssertionError(
                        f"at batch={batch}, offset={offset} two excluded families resolve to the "
                        f"SAME column for some row; the candidate count subtracts both and would "
                        f"over-subtract. The index derivations have to be re-read, not this "
                        f"guard.")
    return out


def excluded_mask(batch: int, offset: int, *, mask_positive_twin: bool,
                  device=None) -> tuple[torch.Tensor, int]:
    """(2B, 2B) bool of every excluded entry, and how many. Only for callers wanting a matrix."""
    n = 2 * batch
    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    count = 0
    for cols, exists in excluded_columns(batch, offset, mask_positive_twin=mask_positive_twin,
                                         device=device):
        mask |= _mask_from_columns(cols, exists)
        count += int(exists.sum())
    return mask, count


def candidate_counts(batch: int, offset: int, *, mask_positive_twin: bool,
                     device=None) -> torch.Tensor:
    """Per row, how many columns the softmax actually runs over -- THE POSITIVE INCLUDED.

    ⛔ THIS IS THE NUMBER THAT MAKES A LOSS VALUE READABLE, and it is not 2B - 1. Start from
    2B - 1 (every column but the row itself), then drop the excluded ones::

        interior row      2B - 4   (its own frame, its positive's frame, and the transpose:
                                    the row at t - k whose positive IS this row's frame)
        edge rows         2B - 2 or 2B - 3   (one or two of the three are off the slab's end)
        offset >= B       2B - 1   (none exists; the objective is the unmasked one)
        twin knob OFF     2B - 2 interior, 2B - 1 at the k edge rows of each half

    Returned per row, not averaged, because the edge rows really do face a bigger candidate set
    and a reader who assumes one number for the whole batch is off by O(k/B).
    """
    n = 2 * batch
    counts = torch.full((n,), n - 1, dtype=torch.long, device=device)
    for _, exists in excluded_columns(batch, offset, mask_positive_twin=mask_positive_twin,
                                      device=device):
        counts = counts - exists.long()
    return counts


def random_floor(batch: int, offset: int, *, mask_positive_twin: bool, device=None) -> float:
    """The cross-entropy a UNIFORM GUESS over each row's own candidate set pays: mean_r ln|C_r|.

    ⚠ MEAN OF ln, NOT ln OF THE MEAN. The rows do not share a candidate set (the edge rows have
    more candidates), and cross_entropy averages the per-row -log p, so the floor is the mean of
    the per-row logs. The two agree to 5e-09 at B=498 and differ by 1.5e-03 (0.06%) at B=8, k=1
    with the twin masked -- measured, not estimated: the point is the derivation, and the gap is
    small enough that a number quoted for it would only ever be quoted wrong.
    """
    return float(torch.log(candidate_counts(batch, offset,
                                            mask_positive_twin=mask_positive_twin,
                                            device=device).double()).mean())


def nt_xent_same_frame_masked(z1: torch.Tensor, z2: torch.Tensor, temperature: float,
                              offset: int, *,
                              mask_positive_twin: bool) -> tuple[torch.Tensor, int]:
    """NT-Xent over a candidate set defined by FRAME IDENTITY, not by index. -> (loss, masked).

    ⛔ TWO OWNER DECISIONS, BOTH 2026-09-17, AND THEY ARE NOT THE SAME MASK.
      * the ANCHOR's own frame leaves its negatives (workspace DECISIONS, morning). Always on:
        no knob turns it off, because no arm is queued that wants it back.
      * the POSITIVE's own frame leaves as well (owner follow-up, relayed by commander).
        `NETT_AUX_CLTT_MASK_POSITIVE_TWIN`, default ON, one arm queued with it OFF to measure
        the size of what it removes -- hence `mask_positive_twin` is a REQUIRED keyword here:
        a call site that does not name its regime is a call site whose loss cannot be read.
    Every cltt_ref arm is therefore one of three objectives; see MASK_REGIME_* and `compute`.

    ⛔ WHY THE TWIN IS NOT A NEGATIVE. The views are contiguous slabs of ONE stream, so
    `z1[i + k] IS z2[i]` -- the frame the positive term is pulling the anchor TOWARD also sits
    in the first half as a column the same row is pushing AWAY. That is a contradiction, not a
    hard negative, and it has exactly one solution: encode that one frame differently in the two
    views. `dup_sim` is the statistic that catches it, and a wave-15 arm took the route -- 7/7 of
    its brains drove the loss below n_offsets*ln2, which is only reachable by breaking the tie.

    ⛔ THE FLOOR IS NOW n_offsets * ln(candidate count), AND THE CANDIDATE COUNT IS WHAT TO READ.
    Not ln(2B - 1): that presumed 2B distinct samples, which a contiguous slab has never
    supplied, and it overstated the floor under the anchor mask too. `random_floor` computes it
    per row and `compute` publishes it as `chance` beside the realised `candidates`, so a reader
    never has to know which regime ran in order to interpret a loss value.
    ⚠ THE ln2 CAP AND THE ATTAINABLE FLOOR ARE DIFFERENT NUMBERS, AND BOTH ARE MEASURED.
    While the twin is a candidate, p(positive) <= 1/2 for every row that has one -- an exact tie
    against an exact copy, for ANY pure-function encoder -- so the loss cannot go under ln2
    without encoding one frame two ways, which is the wave-15 route. Masking it removes the cap;
    whether that is worth anything is a question about what is then ATTAINABLE, so it was
    measured: minimise this loss directly over B+k free frame embeddings, the best any function
    of the image alone can do (d=96, Adam, two seeds, T=0.05; identical to 4 dp at T=0.02):

        (B, k)        (8,1)    (12,2)   (20,3)   (32,2)      ln2 = 0.6931
        twin  in      1.2130   1.1553   1.1784   1.2997      -- every one ABOVE ln2
        twin masked   0.4332   0.3466   0.3814   0.5634      -- every one BELOW it

    ⛔ THE TRANSPOSE IS WHAT MAKES THAT TRUE, AND IT IS NOT TIDINESS. Masking only the forward
    direction -- row i drops column i+k, row i+k keeps column i -- leaves the loss pushing that
    pair apart from one side, and the attainable floor stays at 0.8431 (B=12, k=2), still above
    ln2. Measured, after a symmetry assertion caught the half-mask. Both entries, or neither.

    ⇒ A loss below n_offsets*ln2 is a VIEW-DEPENDENCE reading while the twin is in the candidate
      set, and an ordinary reading once it is out; read it against `chance` and `candidates`
      instead, and use `dup_sim` for the question ln2 used to answer -- a pure function of the
      image gives exactly 1, and BatchNorm over two different slabs (z1 and z2 are separate
      forward passes) is the live route away from it, in either regime.

    ⛔ THIS IS NOT `simclr_aux.nt_xent` AND MUST NOT BE FOLDED INTO IT. The SimCLR arms pair two
    AUGMENTATIONS of one image; their 2B set contains no duplicate frame, so these masks would
    remove real negatives from a live objective.

    ⚠ The count is RETURNED, never assumed: at offset >= B nothing is masked and a caller that
    logs "masked" while masking nothing is the failure this signature exists to prevent.
    """
    batch = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    sim.fill_diagonal_(float("-inf"))
    mask, count = excluded_mask(batch, offset, mask_positive_twin=mask_positive_twin,
                                device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))     # out-of-place: the graph keeps its gradient
    labels = (torch.arange(2 * batch, device=z.device) + batch) % (2 * batch)
    return F.cross_entropy(sim, labels), count


#: Row block for the diagnostic's argmax, and the rejection rounds for its null. Not knobs:
#: they fix the diagnostic's COST, not its answer, and an arm that changed them would report
#: the same numbers. 1024 rows x 7,872 columns is 32 MB, against 248 MB for the full matrix.
DIAG_CHUNK_ROWS = 1024
DIAG_NULL_ROUNDS = 8

#: Every key `nt_xent_diagnostics` returns. ⛔ ONE list, asserted against the function's own
#: output by a test, because a call site that switches the diagnostic OFF must still emit the
#: same keys as sentinels -- and a second hand-written copy of this list would go quiet on
#: exactly the key that was added last.
DIAG_KEYS = ("pos_acc", "shuffled_acc", "pos_chance", "pos_ceiling", "duplicates",
             "positive_aliases", "dup_sim", "pos_sim", "neg_sim", "pos_candidates",
             "batch")

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


def nt_xent_diagnostics_absent(batch: int) -> dict:
    """The diagnostic's keys, all sentinel, for a call site where it is switched OFF.

    ⚠ SENTINELS, NOT SILENCE. A key that disappears when a flag is off makes "the diagnostic did
    not run" and "the diagnostic ran and found nothing" the same absence, and the sentinel-aware
    aggregation in ppo_aux publishes a fire rate of 0 against it, which is readable.
    """
    return {k: (float(batch) if k == "batch" else float(NOT_MEASURED)) for k in DIAG_KEYS}


def nt_xent_diagnostics(z1: torch.Tensor, z2: torch.Tensor, temperature: float,
                        *, duplicate_offset: int | None = None,
                        mask_positive_twin: bool | None = None) -> dict:
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
       sits at `pos_chance` (the mean of 1/|C_r| over rows -- NOT 1/(2B-1), which counts
       columns this objective does not offer) the pairing is unusable.

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

    ⚠ READ THE LOSS AGAINST THE FLOOR ITS OWN REGIME HAS. The random floor is
    n_offsets*mean_r ln|C_r| -- published as `chance`, with |C_r| published as `candidates`, so
    it needs no outside knowledge of which mask ran. The SECOND floor, n_offsets*ln2, applies
    only while the positive's twin is still a candidate: a pair holding the SAME frame twice can
    be driven under it by embedding that frame DIFFERENTLY in the two views -- breaking the tie
    rather than learning invariance, measured on a real arm (C72, all seven brains). With
    `mask_positive_twin` that cap is gone, and so is the reading: an encoder that is a pure
    function of the image reaches 0.35 (B=12, k=2) under the masked objective, so below ln2 is
    then an ordinary number and says nothing about view dependence. `dup_sim` is what answers
    that question in either regime, and is the one to look at first.

    ⛔ THE CANDIDATE SET HERE IS THE LOSS'S, FROM THE LOSS'S OWN DERIVATION. `mask_positive_twin`
    is required exactly when `duplicate_offset` is given, so no caller can half-specify it: with
    the twin masked, this diagnostic's hit rule and the loss agree that the twin column is
    neither a right answer nor a wrong one, because it is not an answer at all. With it unmasked
    they disagree -- the loss scores that column wrong, this counts it a hit -- and that
    disagreement is a property of the objective the OFF arm runs, not of this function.

    ⚠ Every field is emitted on EVERY call. A diagnostic written only where it succeeds makes
    "engaged" and "fell through" both present as absent, and absent reads as benign.
    """
    B = z1.shape[0]
    n = 2 * B
    z = torch.cat([z1, z2], dim=0)
    labels = (torch.arange(n, device=z.device) + B) % n
    rows = torch.arange(n, device=z.device)

    # ⇒ ONE DERIVATION, IN ITS INDEX FORM. `same_frame_columns` is what `same_frame_mask` is
    # built from, so the loss and the diagnostic still cannot drift -- but nothing here
    # materialises a (2B, 2B) helper. cltt_patch calls this with 2BM = 7,872 rows, where each
    # dense helper is 62 MB (bool) or 248 MB (float32): measured +740 MiB of GPU peak on the row
    # that is already the wave's memory ceiling, for a diagnostic costing r00 only +22 MiB.
    if (duplicate_offset is None) != (mask_positive_twin is None):
        raise ValueError(
            f"duplicate_offset={duplicate_offset!r} and mask_positive_twin="
            f"{mask_positive_twin!r} must be given together. The candidate set is the objective: "
            f"a caller that declares an overlap without saying whether the positive's twin is in "
            f"the loss gets a chance level for a loss nobody is training.")
    if duplicate_offset is None:
        dup_col, has_dup = rows, torch.zeros(n, dtype=torch.bool, device=z.device)
        alias_col, has_alias = rows, torch.zeros(n, dtype=torch.bool, device=z.device)
        n_dup = n_alias = NOT_MEASURED        # the caller did not declare an overlap
        twin_masked = False
        excluded: list = []
        candidates = torch.full((n,), n - 1, dtype=torch.long, device=z.device)
    else:
        k = int(duplicate_offset)
        twin_masked = bool(mask_positive_twin)
        dup_col, has_dup = same_frame_columns(B, k, device=z.device)
        alias_col, has_alias = positive_alias_columns(B, k, device=z.device)
        n_dup, n_alias = float(int(has_dup.sum())), float(int(has_alias.sum()))
        # ⛔ NOT `(n - 1) - has_dup - has_alias` WRITTEN OUT HERE. That is the loss's arithmetic,
        # and a second copy of it is how the diagnostic ends up reporting the chance level of an
        # objective nobody trained. `candidate_counts` is built from `excluded_columns`, which
        # is what `excluded_mask` -- the loss's mask -- is built from.
        excluded = excluded_columns(B, k, mask_positive_twin=twin_masked, device=z.device)
        candidates = candidate_counts(B, k, mask_positive_twin=twin_masked, device=z.device)
    twin_excluded = has_alias & twin_masked

    # ⛔ THE ARGMAX IS CHUNKED over rows. It is the one quantity that genuinely needs every
    # column, and a row block of DIAG_CHUNK_ROWS keeps the transient at 32 MB instead of 248 MB
    # at cltt_patch's shape while giving the identical answer (argmax is per row).
    pred = torch.empty(n, dtype=torch.long, device=z.device)
    for lo in range(0, n, DIAG_CHUNK_ROWS):
        hi = min(lo + DIAG_CHUNK_ROWS, n)
        block = torch.mm(z[lo:hi], z.t()) / temperature
        idx = rows[lo:hi]
        block.scatter_(1, idx[:, None], float("-inf"))                      # self
        # ⛔ THE SAME EXCLUSIONS THE LOSS APPLIES, FROM THE LOSS'S OWN LIST. Naming the two
        # families here instead cost exactly what this comment warns about: the transpose was
        # missing, so the diagnostic's argmax ran over a LARGER set than the loss and reported
        # 0.583 where the objective scored 0.750. Iterating the list cannot make that mistake.
        # ⚠ Where a family does not exist for a row its column IS the row, already -inf from the
        # line above -- the property `same_frame_columns` documents, and why this needs no gate.
        for col, _ in excluded:
            block.scatter_(1, col[lo:hi][:, None], float("-inf"))
        pred[lo:hi] = block.argmax(dim=1)
        del block

    # ⛔ THE HIT RULE IS THE COMPLEMENT OF THE CANDIDATE SET, and it must be, or the two
    # disagree about the same column. A hit is the positive OR another column holding the
    # positive's frame WHILE THAT COLUMN IS STILL A CANDIDATE. With the twin masked it is not a
    # candidate at all -- the argmax can never land there -- and the rule collapses to plain
    # equality; with the knob off it is a candidate the loss scores wrong and this scores right,
    # which is the OFF arm's objective and is stated as such in the docstring. See
    # `positive_alias_columns` for why the plain equality reads 0 for a perfect encoder there.
    hit = (pred == labels) | (has_alias & (not twin_masked) & (pred == alias_col))
    pos_acc = float(hit.float().mean())
    # Chance is the mean of 1/|candidates| over rows, not 1/(2B-1) -- that number presumed 2B
    # distinct samples, which this slab has never supplied.
    chance = float((1.0 / candidates.double()).mean())

    # ⛔ THE NULL IS A RANDOM VALID WRONG CANDIDATE, NOT `labels + 1`. The shift aliased the
    # duplicate exactly when k = 1 and never when k = 2, so the "null" averaged (B-1)/4B =
    # 0.2495 over two offsets and was reporting the same pinned argmax the signal was. A null
    # that can land on a duplicate -- or on a copy of the positive -- measures those, not chance.
    # Drawn by REJECTION on one column per row (at most four are excluded of 2B, so a handful of
    # rounds clears every row) from a DEDICATED generator, so the diagnostic consumes none of
    # the training RNG stream: with the global generator, turning the diagnostic on would change
    # which windows later updates draw, and this must not touch training at all.
    # ⚠ `torch.rand`, NOT `torch.randint`. The sampler's own draws go through `torch.randint`,
    # which tests and tools monkeypatch to pin a window; a diagnostic that reached for the same
    # symbol would eat those fixed draws and move the window it is supposed to describe. The
    # generator keeps it off the global stream; the symbol keeps it off the sampler's.
    gen = _diag_generator(z.device)

    def _draw():
        return (torch.rand(n, device=z.device, generator=gen) * n).long().clamp_(max=n - 1)

    def _invalid(c):
        bad = ((c == rows) | (c == labels)
               | (has_dup & (c == dup_col)) | (has_alias & (c == alias_col)))
        for col, ex in excluded:          # anything the loss removed is not a candidate either
            bad = bad | (ex & (c == col))
        return bad

    choice = _draw()
    bad = _invalid(choice)
    for _ in range(DIAG_NULL_ROUNDS):
        if not bool(bad.any()):
            break
        choice = torch.where(bad, _draw(), choice)
        bad = _invalid(choice)
    usable = ~bad
    # A row the rejection could not place is DROPPED rather than counted as a miss: counting it
    # would be a measurement the draw never made. At 2B >= 8 this is empty in practice.
    shuffled_acc = (float((pred == choice)[usable].float().mean()) if bool(usable.any())
                    else NOT_MEASURED)

    # ⛔ THE SIMILARITY SUMMARIES COME FROM VECTOR IDENTITIES, NOT FROM THE MATRIX.
    # Sum of every pairwise dot is ||Sum_i z_i||^2, so the whole (2B, 2B) product never exists;
    # the diagonal, the positives and the duplicates are one dot per row each. Accumulated in
    # float64 because the total is O(n^2) terms against per-term values of order 1.
    zd = z.double()
    pos_dot = (zd * zd[labels]).sum(dim=-1)
    dup_dot = (zd * zd[dup_col]).sum(dim=-1) * has_dup.double()
    total = float((zd.sum(dim=0) ** 2).sum())
    diag_sum = float((zd * zd).sum())
    pos_sim = float(pos_dot.mean())
    # ⛔ The one statistic that separates the two ways a low loss can happen. `pos_acc` cannot:
    # an encoder that has learned temporal invariance and an encoder that has broken the tie by
    # embedding ONE frame differently in the two views both score high. A pure function of the
    # image gives cos(z1[i], z2[i-k]) = 1 EXACTLY for a duplicate, because it is the same image;
    # anything below 1 means the embedding depends on which view the frame arrived in (batch
    # statistics are the live route -- BatchNorm in train() sees two different slabs).
    # Measured on a 600-step fixture where the tie broke: dup_sim 0.9910 BELOW pos_sim 0.9943.
    dup_count = int(has_dup.sum())
    dup_sim = float(dup_dot.sum() / dup_count) if dup_count else NOT_MEASURED
    # ⚠ THE EXCLUDED COLUMNS LEAVE `neg_sim` TOO, or the mean negative similarity is inflated by
    # the largest entries in the row. The duplicate is not a negative in any sense the number is
    # read for -- it is the anchor itself -- and where the twin is masked it is not a negative in
    # the loss either, so a `neg_sim` that still counted it would describe a different objective.
    excluded_sum, excluded_count = 0.0, 0
    for col, ex in excluded:
        excluded_sum += float(((zd * zd[col]).sum(dim=-1) * ex.double()).sum())
        excluded_count += int(ex.sum())
    neg_count = n * n - 2 * n - excluded_count
    neg_sim = ((total - diag_sum - float(pos_dot.sum()) - excluded_sum) / neg_count
               if neg_count > 0 else NOT_MEASURED)
    return {
        "pos_acc": pos_acc,
        "shuffled_acc": shuffled_acc,
        # ⛔ NOT "chance". That key is the LOSS floor, n_offsets*mean_r ln|C_r|, and this dict is
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
        # ⛔ AND IT IS NOT THE SAME NUMBER IN THE TWO REGIMES -- carrying the first formula into
        # the second would have published a ceiling a healthy encoder EXCEEDS, i.e. manufactured
        # the direction-sensitivity reading below. Both are derived from the same three tied
        # columns and both are verified exactly (orthonormal-basis stationary encoder, 8 shapes
        # including k >= B/2):
        #   twin IN the candidate set   ceiling = 0.5 + k / 2B
        #       the positive's copy at i + k ties and wins by index, but the hit rule counts it,
        #       so the misses are the second half's, except its k edge rows.
        #   twin MASKED                 ceiling = 0.5 + min(2k, B) / 2B
        #       that copy is gone and so is the first-half copy of f_{t-k} (the transpose), so
        #       the surviving tie is the SECOND-half copy of f_{t-k} at column B + (i - 2k):
        #       it exists for i >= 2k and always wins by index, leaving min(2k, B) hits in the
        #       first half. Every second-half row hits, because there its positive is the lowest
        #       of the tied indices.
        # verified to four decimals at (B,k) = (12,2), (20,3), (30,5) and (16,1) against a
        # lookup-table encoder whose similarity peaks exactly at lag k.
        # ⚠ IT IS A BOUND UNDER SYMMETRIC SIMILARITY, NOT A LAW. The fixture that verifies it is
        # symmetric by construction, so it establishes the arithmetic GIVEN the assumption, not
        # that the assumption holds of any real encoder. A reading materially above the ceiling
        # means sim(t, t+k) > sim(t, t-k) systematically -- which a direction-sensitive STIMULUS
        # can produce as readily as a view-dependent encoder (approach and retreat are not
        # mirror images). Above the ceiling is a question to ask, not a verdict.
        "pos_ceiling": NOT_MEASURED if duplicate_offset is None else
                       (0.5 + min(2 * int(duplicate_offset), B) / (2.0 * B)) if twin_masked
                       else (0.5 + int(duplicate_offset) / (2.0 * B)),
        "duplicates": n_dup,
        "positive_aliases": n_alias if n_alias == NOT_MEASURED else float(n_alias),
        "dup_sim": dup_sim,
        "pos_sim": pos_sim,
        "neg_sim": neg_sim,
        # ⛔ THE SIZE OF THE CANDIDATE SET, FROM THE LOSS'S DERIVATION, published so that
        # `pos_chance` and the loss floor can be CHECKED by a reader rather than believed, and
        # so that the two code paths (this and `compute`, which calls `random_floor`) can be
        # asserted equal instead of assumed equal. Mean over rows: the edge rows of each half
        # face more candidates than the interior.
        # ⛔ NOT "candidates" -- `compute` publishes THAT key on every call, diagnostic or not,
        # and this dict is merged into the same `last_scalars`. Under the old spelling the
        # sentinel from the diagnostic-off path overwrote the loss's own count with -9.0, which
        # is the identical collision `pos_chance` is named for. Same prefix, same reason.
        "pos_candidates": float(candidates.double().mean()),
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
    MASK_TWIN_ENV = "NETT_AUX_CLTT_MASK_POSITIVE_TWIN"

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        # GATE A control, OFF unless asked for. One extra similarity matrix per offset, no
        # extra encoder forward -- it reuses embeddings the loss already computed.
        self.diag = _env_flag("NETT_AUX_CLTT_REF_DIAG")
        # ⛔ OWNER FOLLOW-UP, 2026-09-17: the positive's own frame leaves the negatives. ON by
        # default; one arm is queued with it OFF to measure how large the shortcut it removes
        # was, so the OFF path is a supported objective and not a debug switch. Through the
        # STRICT helper: this knob decides which of three losses the arm trains, and a
        # misspelled ON that silently read as OFF would put that arm in the wrong comparison.
        self.mask_twin = _env_flag_strict(self.MASK_TWIN_ENV, True)
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
            # B -- the NT-Xent floor is mean_r ln|C_r| summed over the offsets -- and this line
            # used to emit the offsets and the stack depth and NOT the one parameter the
            # level turns on. A read protocol was published against an assumed B=96 while
            # the realised B at update 1 was ~45, because t_max is the memory FILL INDEX
            # until the buffer fills. The ValueError below already formats these exact
            # values, so the information existed in the file and was emitted only when the
            # run DIED. ⚠ B is not constant, so this one-time line calibrates update 1
            # only; the per-update series is published through `last_scalars`.
            # ⚠ ln(2B-1) IS NOT THIS OBJECTIVE'S FLOOR and has not been since 2e30ad2. The
            # candidate set is smaller than 2B-1 by up to three columns per row, so the line
            # reports the REALISED count and the floor derived from it -- the same pair
            # `last_scalars` publishes per update, so the startup line and the series agree.
            per_offset = [(k, float(candidate_counts(
                batch, k, mask_positive_twin=self.mask_twin).double().mean()),
                random_floor(batch, k, mask_positive_twin=self.mask_twin))
                for k in self.offsets]
            logger.info(
                "CLTTReferenceAuxLoss: offsets=%s, stack depth T=%s, batch B=%s "
                "(t_max=%s, avail=%s, NETT_AUX_BATCH=%s, memory filled=%s), "
                "positive twin masked=%s -> candidates per row (mean) %s of 2B-1=%s, "
"NT-Xent chance floor mean_r ln|C_r| per offset %s, summed over %s offsets=%.4f",
                self.offsets, self.num_frames, batch, t_max, avail, self.max_samples,
                bool(getattr(self._memory, "filled", False)), self.mask_twin,
                [round(c, 3) for _, c, _ in per_offset], 2 * batch - 1,
                [round(f, 4) for _, _, f in per_offset], len(self.offsets),
                sum(f for _, _, f in per_offset),
            )
        views = self._make_views(views, encoder)

        z_anchor = self.head(encoder.encode_prepared(views[0]))  # backbone grad ON
        total, diags = 0.0, []
        masked_pairs = 0
        candidates = 0.0
        floor = 0.0
        for offset, view in zip(self.offsets, views[1:]):
            z_pos = self.head(encoder.encode_prepared(view))
            # ⛔ TWO OWNER DECISIONS OF 2026-09-17: the anchor's own frame is not one of its
            # negatives, and (under MASK_TWIN_ENV, default on) neither is the positive's. This
            # CHANGES THE OBJECTIVE -- see `nt_xent_same_frame_masked`. Arms trained under a
            # different `mask_regime` are not comparable to these.
            offset_loss, masked = nt_xent_same_frame_masked(
                z_anchor, z_pos, self.temperature, offset,
                mask_positive_twin=self.mask_twin)
            total = total + offset_loss
            masked_pairs += masked
            candidates += float(candidate_counts(batch, offset, mask_positive_twin=self.mask_twin,
                                                 device=z_anchor.device).double().mean())
            floor += random_floor(batch, offset, mask_positive_twin=self.mask_twin,
                                  device=z_anchor.device)
            if self.diag:
                with torch.no_grad():
                    # The offset IS the overlap: view_offset[i] is view_0[i + offset], the same
                    # frame. The diagnostic needs it to know which candidates are duplicates --
                    # and the regime, because the candidate set IS the objective.
                    diags.append(nt_xent_diagnostics(z_anchor, z_pos, self.temperature,
                                                     duplicate_offset=offset,
                                                     mask_positive_twin=self.mask_twin))
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
        # from tfevents alone -- this objective has changed TWICE in one day and the loss VALUE
        # alone cannot say which of the three produced it. `masked_pairs` is the realised count,
        # not the intent: at offset >= B it is legitimately 0, and a regime flag without its
        # count would report masking that did not happen.
        #
        # ⛔ THE FOUR-WAY DECODE, WHICH IS WHAT FLEET PROVENANCE KEYS ON. `mask_regime` names
        # three objectives; a missing tag is a FOURTH state and must not be read as 0:
        #     same_frame_masked ABSENT, mask_regime ABSENT  -> pre-2e30ad2: regime 0, no mask
        #     same_frame_masked 1.0,    mask_regime ABSENT  -> 2e30ad2..2f4c96b: anchor only
        #     same_frame_masked 1.0,    mask_regime 1.0     -> this build, twin knob OFF
        #     same_frame_masked 1.0,    mask_regime 2.0     -> this build, twin knob ON (default)
        # `same_frame_masked` is kept for exactly this reason -- it is what separates the first
        # two rows, and dropping it would collapse "no mask" and "anchor mask" into one absence.
        #
        # ⛔ `chance` IS THE REALISED FLOOR, NOT ln(2B-1), and this is a CORRECTION as well as a
        # change: ln(2B-1) was already wrong at regime 1, where interior rows had 2B-2
        # candidates. A 2f4c96b arm and a knob-OFF arm here therefore publish DIFFERENT `chance`
        # for the SAME objective; `mask_regime` is what tells them apart, and `candidates` is
        # what makes either number checkable. Reported beside the loss so that reading a loss
        # value never requires knowing which regime ran -- that is the point of publishing it.
        self.last_scalars = {"B": float(batch), "t_max": float(t_max),
                             "same_frame_masked": 1.0,
                             "mask_regime": (MASK_REGIME_ANCHOR_AND_TWIN if self.mask_twin
                                             else MASK_REGIME_ANCHOR),
                             "masked_pairs": float(masked_pairs),
                             "masked_frac": float(masked_pairs)
                             / float(len(self.offsets) * (2 * batch) ** 2),
                             "candidates": candidates / float(len(self.offsets)),
                             "chance": floor}
        # ⛔ THE SENTINELS GO OUT ON THE OFF PATH TOO, and this class was left behind when the
        # sibling was fixed. With the diagnostic off these nine keys were simply ABSENT from the
        # tag list -- which is the failure `nt_xent_diagnostics_absent` exists to name: a key
        # that disappears when a flag is off makes "did not run" and "ran and found nothing" the
        # same absence, and the sentinel-aware aggregation in ppo_aux can only publish a fire
        # rate of 0 for a key it was told about.
        #
        # ⚠ THIS IS VISIBLE TO READERS OF OLD TFEVENTS, in the same way the regime flag is. An
        # arm run BEFORE this commit has these tags absent whenever the diagnostic was off; an
        # arm run after it has them present at -9.0. Absent therefore means "old arm", not "no
        # diagnostic", and the two must not be read as the same thing.
        self.last_scalars.update(
            {k: float(v) for k, v in self.last_diag.items()} if self.last_diag
            else nt_xent_diagnostics_absent(batch))
        return total
