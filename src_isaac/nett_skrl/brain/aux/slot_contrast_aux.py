"""CNN-SlotContrast, PROMOTED from `examples/candidates/` and registered as `slot_contrast`.

Promoted 2026-09-15 for the 14-condition wave. Until then it lived outside
`nett_skrl/brain/aux/` precisely so a queue row could not launch it unscreened; promotion
is the visible diff that makes `NETT_AUX_LOSS=slot_contrast` resolve. The screening path
still works and is still the cheap way to exercise it without a card:

    examples/replay_harness.py --fixture --model CNN2F --aux slot_contrast

⛔ TWO THINGS CHANGED ON PROMOTION, AND BOTH ARE LOAD-BEARING.

1. **The temporal sampler now refuses to cross an episode boundary.** The candidate's
   `_temporal_pair` sliced the buffer with a bare `randint`, so a (t, t+1) pair could
   straddle a reset or the circular buffer's write seam -- pairing the LAST frame of one
   episode with the FIRST frame of the next and calling it "the scene moved". That is the
   exact defect commit 345f291 fixed for every other temporal loss here; this module was
   written before it and did not inherit the fix. It now uses the same
   `episode_window_batch` / `draw_episode_window` helpers as `cltt_ref` and
   `cltt_schneider`, so there is one implementation of episode contiguity, not two.

2. **No usable window now RAISES instead of returning zero.** The candidate returned
   `new_zeros(())`, which enters the running mean as a real value and makes "there was no
   pair" indistinguishable from "the pair scored 0". The fleet convention is to refuse.

⚠ THE GRID IS THE FIRST QUESTION ABOUT THIS PORT, AHEAD OF ANYTHING ABOUT THE LOSS.
Slot attention partitions a set of spatial positions by competition. On `compact_cnn` the
pre-pool map is 20x32 = 640 positions; on `nature_cnn` -- which is what the whole fleet CNN
family actually runs -- it is 6x12 = **72**. At K slots that is ~72/K cells each, and
whether object-level competition is even expressible at that granularity is untested. The
wave runs `nature_cnn` anyway, because the point of holding the encoder constant across all
14 conditions is worth more than the extra positions, and because a win on a 72-cell grid
would be the stronger result. `NETT_SLOTC_SLOTS` defaults to 6 here but the wave sets 4:
during IMPRINTING the scene has ~3 regions (object, chamber, bezel), not the 4 the parsing
TEST has, and the aux loss only ever runs during training. See
notes/researcher/slotcontrast-cnn.md section 6.

Port of `martius-lab/slotcontrast` (Manasyan et al., CVPR 2025) with DINOv2 replaced by our
own CNN trunk, per the owner's instruction *"SlotContrast could just replace DINO with a CNN,
like how we have done with GWM."* Design rationale, the parameter table and the pre-registered
falsifiers are in `notes/researcher/slotcontrast-cnn.md`; only what the code needs is repeated
here.

## The one decision the port forces

The reference composes two losses (`configs/slotcontrast/movi_c.yaml`):

    loss_ss      Slot_Slot_Contrastive_Loss   weight 0.5
    loss_featrec MSELoss vs a FROZEN PRETRAINED DINOv2   weight 1.0

⛔ **`loss_ss` is degenerate on its own.** `slotcontrast/losses.py` detaches NEITHER side: if
every slot emitted a constant, slot-index-specific vector and ignored the image entirely, the
slot-slot similarity matrix would be exactly the identity, the cross-entropy would go to ~0,
and nothing would be segmented. It is safe in the reference only because `loss_featrec` pins
the slots to real image content through a target the slots cannot influence.

We have no pretrained weights, so the target must come from our own trunk — which is the
defect `nett_skrl/brain/aux/gwm_aux.py` records the vendored GWM as having: a target produced
by the thing being trained, giving a trivial joint optimum that trains, converges, and reports
a LOWER loss than the correct version. Hence the target here is a **stop-gradient EMA copy of
the trunk** (BYOL/SPR-style): the slots cannot move it within a step, and it cannot be driven
to a constant by the loss that consumes it because no gradient reaches it at all.

⚠ EMA adds no trainable parameters but doubles trunk activation memory, which under the
fleet's env budget means fewer parallel envs and therefore lower n.

## The slot map is read BEFORE the pool, and that is load-bearing

Slot attention is defined over a set of spatial features that compete to explain positions.
`CompactCNN.forward` is `linear(cnn(x))` where `cnn` ends in a 4x4 adaptive pool and a
flatten — after which the geometry is gone. So this module runs the conv stack up to (not
through) that pool, giving 64 channels over a 20x32 grid = 640 positions at the live 128x80
eye. Reaching a global feature vector instead and letting a head synthesise a field from it —
which is what `GWMHead` does — would be a different method wearing this one's name, so the
boundary is asserted rather than assumed.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cltt_ref_aux import episode_window_batch, draw_episode_window


def _env_flag(name: str, default: bool = False) -> bool:
    """Same spellings the fleet's other boolean knobs accept."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def spatial_features(encoder: nn.Module, prepared: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) from the conv trunk, BEFORE the pool that destroys the geometry.

    ⛔ REFUSES rather than falls back. A silent fallback to the pooled vector would still
    produce slots, still train, and still report a loss -- for a method that no longer has
    spatial positions to compete over. There would be nothing in the output to tell you.
    """
    cnn = getattr(encoder, "cnn", None)
    if not isinstance(cnn, nn.Sequential):
        raise TypeError(
            f"{type(encoder).__name__} exposes no `cnn` Sequential, so the pre-pool feature "
            f"map cannot be reached. Slot attention over a pooled global vector is a "
            f"different method; refusing rather than substituting one."
        )
    # Cut at the first module that destroys the grid. Located by TYPE, not by index: an
    # index would keep working after a reorder and silently read the wrong tensor.
    cut = next((i for i, m in enumerate(cnn)
                if isinstance(m, (nn.Flatten, nn.AdaptiveAvgPool2d))
                or "Pool" in type(m).__name__), None)
    if cut is None:
        raise TypeError(
            f"{type(encoder).__name__}.cnn has no pool/flatten boundary; cannot tell where "
            f"the spatial stage ends."
        )
    feats = prepared
    for module in list(cnn)[:cut]:
        feats = module(feats)
    if feats.dim() != 4:
        raise TypeError(f"expected a (B,C,h,w) map before the pool; got {tuple(feats.shape)}")
    return feats


class SlotAttention(nn.Module):
    """Slot attention (Locatello et al. 2020), the reference's iterative form.

    ⚠ Weights are SHARED across slots -- only `mu`/`log_sigma` are per-slot-distribution --
    so the slot count K does not change the parameter count. K is therefore chosen on the
    scene (2 objects + background + bezel -> K=6), not on the budget.
    """

    def __init__(self, in_dim: int, slot_dim: int, slots: int, iters: int = 3) -> None:
        super().__init__()
        self.slots, self.iters, self.scale = slots, iters, slot_dim ** -0.5
        self.mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
        self.log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.norm_in = nn.LayerNorm(in_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, slot_dim * 2), nn.ReLU(),
                                 nn.Linear(slot_dim * 2, slot_dim))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """inputs (B, N, in_dim) -> slots (B, K, slot_dim), attention (B, K, N)."""
        b, n, _ = inputs.shape
        x = self.norm_in(inputs)
        k, v = self.to_k(x), self.to_v(x)
        slots = self.mu + self.log_sigma.exp() * torch.randn(
            b, self.slots, self.mu.shape[-1], device=inputs.device, dtype=inputs.dtype)
        attn = None
        for _ in range(self.iters):
            q = self.to_q(self.norm_slots(slots))
            logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
            # ★ Softmax over SLOTS, not positions: that competition is what makes slots
            # partition the image instead of all describing the same thing.
            attn = logits.softmax(dim=1)
            weights = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            updates = torch.einsum("bkn,bnd->bkd", weights, v)
            slots = self.gru(updates.reshape(-1, updates.shape[-1]),
                             slots.reshape(-1, slots.shape[-1])).view_as(slots)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots, attn


def slot_contrast_diagnostics(s1: torch.Tensor, s2: torch.Tensor, temperature: float,
                              slots_per_image: int) -> dict:
    """Are the TEMPORAL POSITIVES informative, and informative for the right reason?

    ⛔ `loss_ss` FALLING CANNOT ANSWER THAT, AND THE MODULE DOCSTRING ALREADY SAYS WHY: the
    degenerate optimum is constant slots, which drive the contrastive term toward zero with
    nothing segmented. A falling `ss` is consistent with the objective working and with the
    collapse it exists to avoid. `mask_variance` does not close the gap either -- it asks
    whether slot attention is DEGENERATE, not whether the temporal pairing carries signal, and
    its bare form could not even do that until it was given a null.

    Shape is deliberately `nt_xent_diagnostics`'s (cltt_ref_aux.py:101): an accuracy, read
    against a null drawn from the SAME slots under a structure carrying no temporal signal, so
    the bare number is never the answer.

    ⭐⭐ BUT THE NULL FORKS HERE, AND COLLAPSING IT TO ONE NUMBER WOULD MEASURE NEITHER HALF.
    The flattened row index is `i = b*K + k`: batch element `b`, slot `k`. cltt_ref's null
    shifts the positive assignment by one, which there moves to a different SAMPLE. Here, at
    K=4, a shift of one lands on another slot of the SAME IMAGE three times out of four and
    crosses to a different image once -- an average over two different questions:

      * `shuffled_acc_within`  same image, a DIFFERENT SLOT. "Can a slot be told from its
        neighbours in the same scene?" This is the one the collapse breaks: identical slots
        make the within-image rows indistinguishable while leaving scenes distinguishable.
      * `shuffled_acc_across`  same slot index, a DIFFERENT IMAGE. A much easier task, solvable
        by scene appearance alone, and it stays solvable straight through a slot collapse.

    ⇒ Both are emitted. [[a-mean-over-a-mixture-estimates-nothing]]

    ⛔⛔⛔ AND DO NOT CARRY cltt_ref's RULE ACROSS. "pos_acc <= shuffled_acc means the pairing
    carried no information" is registered for rows 04/05 and is SATISFIED BY THE COLLAPSE HERE.
    Measured by the researcher seat against three constructed regimes, B=32 K=4:

        regime                                        pos_acc   shuffled   chance
        HEALTHY (input-dependent, stable over t)       1.0000    0.0000    0.0078
        DEGENERATE (constant per-slot vectors)         0.0312    0.0000    0.0078
        NO temporal correspondence (independent)       0.0156    0.0234    0.0078

    At the degeneracy pos_acc is STRICTLY GREATER than shuffled. The reason is structural and is
    the thing to internalise: in cltt_ref the positives are real images, so a collapse destroys
    the pairing; here CONSTANT SLOTS TRIVIALLY PRESERVE SLOT INDEX ACROSS TIME -- slot k at t and
    slot k at t+1 are the same constant -- so the temporal correspondence is perfect while
    nothing is segmented. Same instrument shape, different degenerate manifold.
    [[a-score-a-non-policy-attains]]

    ⇒ THIS STATISTIC CANNOT STAND ALONE. It separates "correspondence" from "no correspondence";
    it does NOT separate "slots track objects" from "slots are input-independent constants with
    stable indices". That axis is `mask_variance_excess`, and the two are read as a PAIR:

        pos_acc high    AND excess > 0   -> slot identity is real and input-driven
        pos_acc ~= 1/B  AND excess <= 0  -> constant-slot collapse
        pos_acc ~ chance                 -> no correspondence learned

    ⛔ AND THE LAST TWO ROWS ARE NOT TOLD APART BY ANY SINGLE READ OF THIS STATISTIC. Verified
    over 200 seeds: the orderings that look structural on one seed hold 132/200 and 59/200. The
    separator is that the COLLAPSE IS EXACT AND PERSISTENT -- `ss_pos_acc_x_batch` pinned at
    exactly 1.0 and `ss_input_dependence` at exactly 0.0, every read -- where no-correspondence
    only brushes those values and moves off them. Read three CONSECUTIVE updates, never one.

    ⚠ THE LAST ROW IS THE STARTING STATE, NOT A FAILURE, and that changes when this is read.
    `slots_t` and `slots_n` are separate forward calls each drawing their own `randn`, so there
    is no index correspondence at step 0 and chance-level pos_acc is CORRECT at init. Unlike
    cltt_ref -- whose "too easy" hazard shows AT the first update -- the failure here DEVELOPS,
    so an early read is uninformative rather than reassuring.

    ⚠ Sentinels are negative (an accuracy is >= 0) and are returned where a null is UNDEFINED
    rather than where it is uninteresting: `within` needs K >= 2 and `across` needs B >= 2. A
    null computed on a degenerate axis would equal the positive by construction and read as
    perfect agreement.
    """
    n = int(s1.shape[0])
    k = max(1, int(slots_per_image))
    b = n // k
    sim = (s1 @ s2.t()) / temperature
    labels = torch.arange(n, device=sim.device)
    pred = sim.argmax(dim=1)
    pos_acc = float((pred == labels).float().mean())

    b_idx, k_idx = labels // k, labels % k
    if k >= 2:
        within = b_idx * k + (k_idx + 1) % k
        within_acc = float((pred == within).float().mean())
    else:
        within_acc = -1.0
    if b >= 2:
        across = ((b_idx + 1) % b) * k + k_idx
        across_acc = float((pred == across).float().mean())
    else:
        across_acc = -1.0

    raw = sim * temperature
    pos_sim = float(raw.gather(1, labels[:, None]).mean())
    off = torch.ones_like(raw, dtype=torch.bool)
    off.scatter_(1, labels[:, None], False)
    neg_sim = float(raw[off].mean()) if bool(off.any()) else -1.0
    return {
        "ss_pos_acc": pos_acc,
        "ss_shuffled_acc_within": within_acc,
        "ss_shuffled_acc_across": across_acc,
        "ss_chance": 1.0 / n if n else -1.0,
        # ⭐ THE COLLAPSE SIGNATURE IS 1/B, NOT AN ABSOLUTE. With constant per-slot vectors the
        # B columns sharing a slot index tie at similarity 1 and argmax takes the first, so each
        # row is correct exactly once per batch: pos_acc -> 1/B with BOTH nulls at ~0. B is
        # emitted because the realised batch is a CEILING, not a constant -- `t_max` is the fill
        # index until the buffer fills -- so a rule written against a hardcoded 0.031 drifts off
        # the signature between updates. `ss_pos_acc_x_batch` is that rule's natural units:
        # ~1.0 AT the collapse, ~B when the correspondence is real. Emitted rather than left as
        # a division for the reader, for the same reason `mask_variance_excess` is.
        "ss_batch": float(b),
        "ss_pos_acc_x_batch": pos_acc * b if b else -1.0,
        # ⭐⭐ INPUT DEPENDENCE, AND IT FALLS OUT OF THE ACROSS-NULL RATHER THAN NEEDING A
        # SEPARATE INSTRUMENT. If slot k is the same vector whatever the image, the
        # across-image same-slot column is IN THE TIE SET with the positive, so `across`
        # RISES TO MEET `pos_acc` and the difference goes to zero.
        #     healthy     -> ~1.0   (input-driven slots)
        #     degenerate  -> EXACTLY 0.0, zero variance, 200/200 seeds x 4 (B,K)
        #
        # ⛔⛔⛔ AND IT DOES *NOT* SEPARATE THE COLLAPSE FROM NO-CORRESPONDENCE. I CLAIMED IT
        # DID, ON ONE SEED, AND A 200-SEED SWEEP BY THE RESEARCHER SEAT REFUTED IT:
        #     input_dep(degenerate) < input_dep(nocorr)   in 59/200   <- I claimed ~always
        #     `across == pos` true of degen but not nocorr   132/200  <- I claimed structural
        # The reason is that degenerate sits at EXACTLY 0 while no-correspondence is NOISE
        # CENTRED SLIGHTLY BELOW ZERO (mean -0.0011, range -0.0312..+0.0234), so the collapse
        # is ABOVE nocorr more often than below -- the opposite of the ordering I asserted. My
        # single seed happened to draw nocorr at +0.0078. ⇒ This field is a discriminator
        # against HEALTHY only. [[a-repeat-that-varies-nothing]] in reverse: I read one draw
        # of a noisy quantity as a structural constant because it sat where my story wanted it.
        #
        # ⇒ WHAT ACTUALLY SEPARATES COLLAPSE FROM NO-CORRESPONDENCE IS EXACTNESS AND
        # PERSISTENCE, NOT A THRESHOLD, and it is free because this runs every update: the
        # collapse is PINNED at exactly 1.0 / exactly 0.0, while no-correspondence merely
        # touches those values (~1.5% of reads) and scatters. See the registered kill rule --
        # three CONSECUTIVE reads, not one.
        # ⚠ COMPLEMENTARY TO `mask_variance_excess`, NOT A REPLACEMENT: that one measures input
        # dependence of the ATTENTION MASKS, this one of the SLOT VECTORS as the contrast
        # consumes them. Two different objects; agreement between them is evidence, and
        # disagreement is a finding rather than a fault in either.
        "ss_input_dependence": pos_acc - across_acc if b >= 2 else -1.0,
        "ss_pos_sim": pos_sim,
        "ss_neg_sim": neg_sim,
    }


class SlotContrastHead(nn.Module):
    """The whole trainable set of this loss, as ONE submodule named `head`.

    ⛔ `head` IS THE AUX INTERFACE, NOT A NAMING PREFERENCE. ppo_aux reaches for
    `aux.head` in FOUR places -- `checkpoint_modules` (:235), the optimiser param
    group (:252) and BOTH grad-norm-clip chains (:422, :425) -- and only the first
    is guarded by aux kind. Every other loss in `AUX_LOSSES` defines one. This file
    did not, so `build_agents` raised

        AttributeError: 'SlotContrastAuxLoss' object has no attribute 'head'

    ~60s into the arm, before a single training step. (Verified at runtime against
    all four sites, not by grep: tests/test_slot_contrast_aux.py.)

    ⛔ AND THE ONE-LINE FIX IS THE DANGEROUS ONE. `self.head = self.attn` satisfies
    every one of those four sites: the arm builds, trains, runs to completion and
    files results -- with the DECODER AND THE POSITIONAL GRID (78,081 + 4,608 =
    82,689 of the 137,025 params, measured) in NO optimiser param group and NO grad
    clip. That is a silent partial ablation wearing this
    method's name, and nothing downstream can tell it from the real thing. The head
    must be the whole trainable set, which is what `test_the_optimiser_group_covers_
    the_whole_trainable_set` pins.

    ⛔ IT ALSO OWNS THE DEVICE. `pos` was created by `nn.Parameter(torch.zeros(...))`
    with no `.to(device)`, and nothing calls `aux.to(device)` -- `_build_slot_contrast`
    returns the module as-is. On CPU that is invisible; on a GPU `_decode` adds a cuda
    `slots` to a cpu `pos` and raises. Holding all three here and moving the container
    once makes the device a property of the set rather than of each line.
    """

    def __init__(self, attn: nn.Module, pos: nn.Parameter, decoder: nn.Module) -> None:
        super().__init__()
        self.attn = attn
        self.pos = pos
        self.decoder = decoder


class SlotContrastAuxLoss(nn.Module):
    """Slot-slot temporal contrast + feature reconstruction against a stop-gradient EMA trunk.

    Draws its own (t, t+1) windows from the rollout buffer, like the other temporal losses.
    """

    needs_memory = True

    def __init__(self, encoder: nn.Module, *, slots: int = 6, slot_dim: int = 64,
                 iters: int = 3, temperature: float = 0.1, w_ss: float = 0.5,
                 w_rec: float = 1.0, ema: float = 0.996, max_samples: int = 32) -> None:
        super().__init__()
        self._memory = None
        self.slots = int(os.environ.get("NETT_SLOTC_SLOTS", slots))
        self.slot_dim = int(os.environ.get("NETT_SLOTC_DIM", slot_dim))
        self.temperature = float(os.environ.get("NETT_SLOTC_TEMP", temperature))
        self.w_ss = float(os.environ.get("NETT_SLOTC_W_SS", w_ss))
        self.w_rec = float(os.environ.get("NETT_SLOTC_W_REC", w_rec))
        self.ema = float(os.environ.get("NETT_SLOTC_EMA", ema))
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", max_samples))
        # ⛔ THE ABLATION IS A KNOB, NOT A CODE EDIT. The pre-registered EMA-target ablation
        # ("run the contrastive term with the target trunk NOT detached") has to be runnable
        # without touching this file, or the version that gets ablated is not the version
        # that gets shipped.
        self.no_detach = _env_flag("NETT_SLOTC_NO_DETACH")
        self.diag = _env_flag("NETT_SLOTC_DIAG")

        device = next(encoder.parameters()).device
        with torch.no_grad():
            probe = self._probe_map(encoder)
        self.in_dim, self.grid_h, self.grid_w = probe
        # Broadcast decoder: each slot decodes the whole grid plus an alpha, and the
        # alphas compose. This is the reference's spatial-broadcast form, minus its
        # positional embedding size, which the grid here does not need.
        # All three go in `self.head` -- see SlotContrastHead for why that name is the
        # interface and why splitting the set across attributes broke the arm twice.
        self.head = SlotContrastHead(
            SlotAttention(self.in_dim, self.slot_dim, self.slots, iters),
            nn.Parameter(torch.zeros(1, self.slot_dim, self.grid_h, self.grid_w)),
            nn.Sequential(
                nn.Conv2d(self.slot_dim, self.slot_dim, 3, padding=1), nn.ReLU(),
                nn.Conv2d(self.slot_dim, self.slot_dim, 3, padding=1), nn.ReLU(),
                nn.Conv2d(self.slot_dim, self.in_dim + 1, 1),
            ),
        ).to(device)
        # ⛔ THE EMA TARGET IS HELD IN A LIST, NOT AS AN ATTRIBUTE. Assigning an nn.Module
        # to an attribute of an nn.Module REGISTERS it, so `aux.parameters()` would then
        # carry the whole target trunk -- and the harness builds its optimiser as
        # `list(encoder.parameters()) + list(aux.parameters())`, which would hand Adam the
        # same tensors twice. A list attribute is not registered, so the copy stays out of
        # the parameter set while remaining reachable.
        # (The first version of this file also kept a `self._encoder_ref = encoder` that
        # nothing read; it silently added all 666,272 encoder parameters to
        # `aux.parameters()` and made the reported aux size 803,297 instead of 137,025.)
        self._target: list[nn.Module] = []
        # Diagnostics, read by the screening harness. Sentinels are negative so they cannot
        # collide with a variance, which is >= 0.
        self.last_mask_variance: float = -1.0
        #: Slot-init-noise floor for `last_mask_variance`. Same sentinel discipline: negative so
        #: "diagnostic off" cannot be read as "null was zero".
        self.last_mask_variance_null: float = -1.0
        # loss_ss discriminability. ⛔ A DICT OF SENTINELS, NOT None: `last_scalars` is built
        # unconditionally below, so a None here would raise inside the emitter on the diag-off
        # path instead of reporting "not measured". Negative because an accuracy is >= 0.
        self.last_ss_diag: dict = {"ss_pos_acc": -1.0, "ss_shuffled_acc_within": -1.0,
                                   "ss_shuffled_acc_across": -1.0, "ss_chance": -1.0,
                                   "ss_batch": -1.0, "ss_pos_acc_x_batch": -1.0,
                                   "ss_input_dependence": -1.0,
                                   "ss_pos_sim": -1.0, "ss_neg_sim": -1.0}
        self.last_terms: tuple[float, float, float] | None = None
        # ⛔ `last_scalars` IS THE CHANNEL ppo_aux READS (ppo_aux.py:442). `last_mask_variance`
        # alone reached no reader -- the identical defect `nt_xent_diagnostics` had, where the
        # diagnostic ran, cost its forward passes, and produced no log line and no scalar.
        # tests/test_aux_telemetry_reaches_a_reader.py catches the class; this is the fix.
        self.last_scalars: dict = {}

    # -- plumbing ---------------------------------------------------------------------

    def _probe_map(self, encoder: nn.Module) -> tuple[int, int, int]:
        from ...body.observation import image_channels_hw
        c, h, w = image_channels_hw(getattr(encoder, "observation_space", None))
        dummy = torch.zeros(1, c, h, w, device=next(encoder.parameters()).device)
        feats = spatial_features(encoder, dummy)
        return int(feats.shape[1]), int(feats.shape[2]), int(feats.shape[3])

    def attach_memory(self, memory) -> None:
        self._memory = memory

    @torch.no_grad()
    def _update_target(self, encoder: nn.Module) -> None:
        """EMA the target toward the live trunk. No gradient, by construction."""
        import copy
        if not self._target:
            tgt = copy.deepcopy(encoder).eval()
            for p in tgt.parameters():
                p.requires_grad_(False)
            self._target.append(tgt)
            return
        tgt = self._target[0]
        for tp, sp in zip(tgt.parameters(), encoder.parameters()):
            tp.mul_(self.ema).add_(sp.detach(), alpha=1.0 - self.ema)
        for tb, sb in zip(tgt.buffers(), encoder.buffers()):
            tb.copy_(sb)

    def _slots_for(self, encoder: nn.Module, prepared: torch.Tensor):
        feats = spatial_features(encoder, prepared)              # (B, C, h, w)
        b, c, h, w = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)                # (B, N, C)
        slots, attn = self.head.attn(tokens)
        return slots, attn, feats, (b, c, h, w)

    def _decode(self, slots: torch.Tensor, shape) -> torch.Tensor:
        b, c, h, w = shape
        grid = slots.reshape(b * self.slots, self.slot_dim, 1, 1) + self.head.pos
        out = self.head.decoder(grid)
        recon, alpha = out[:, : self.in_dim], out[:, self.in_dim:]
        recon = recon.view(b, self.slots, self.in_dim, h, w)
        alpha = alpha.view(b, self.slots, 1, h, w).softmax(dim=1)
        return (recon * alpha).sum(dim=1)                        # (B, C, h, w)

    # -- the loss ---------------------------------------------------------------------

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        from ...body.observation import prepare_image_tensor

        # Raises rather than returning a zero when no episode-contiguous pair exists --
        # see _temporal_pair. There is no longer a None branch to fall through.
        obs_t, obs_next = self._temporal_pair(encoder, observations)

        space = encoder.observation_space
        prep_t = prepare_image_tensor(obs_t, space)
        prep_n = prepare_image_tensor(obs_next, space)

        slots_t, attn_t, feats_t, shape_t = self._slots_for(encoder, prep_t)
        slots_n, _, _, _ = self._slots_for(encoder, prep_n)

        # --- slot-slot temporal contrast (the reference's Slot_Slot_Contrastive_Loss) ---
        s1 = F.normalize(slots_t, p=2.0, dim=-1)
        s2 = F.normalize(slots_n, p=2.0, dim=-1)
        # Merge batch and slot dims so negatives span the batch, as the reference does.
        sim = (s1.reshape(-1, self.slot_dim) @ s2.reshape(-1, self.slot_dim).T) / self.temperature
        target_idx = torch.arange(sim.shape[0], device=sim.device)
        ss = F.cross_entropy(sim, target_idx)

        # --- feature reconstruction against the stop-gradient EMA trunk ---
        self._update_target(encoder)
        if self.no_detach:
            # ⚠ ABLATION ONLY. The target is then the LIVE trunk, which the same loss
            # trains -- the degenerate joint optimum this module exists to avoid. Expect a
            # LOWER loss and no better readout; that is the point of running it.
            tgt = spatial_features(encoder, prep_t)
        else:
            with torch.no_grad():
                tgt = spatial_features(self._target[0], prep_t)
        rec = F.mse_loss(self._decode(slots_t, shape_t), tgt)

        if self.diag:
            # ⛔ THE DEGENERACY CHECK: are the masks INPUT-DEPENDENT? If slots collapse to
            # constants, `ss` goes to ~0 with nothing segmented -- and a falling loss reads
            # as progress. Variance of each slot's mask ACROSS the batch is near zero
            # exactly in that case, and the loss cannot tell you.
            #
            # ⛔⛔⛔ AND THE BARE VARIANCE CANNOT ANSWER THAT. IT HAS A NULL NOW, AND HERE IS WHY.
            # `SlotAttention.forward` draws `mu + log_sigma.exp() * randn(b, K, D)` -- a random
            # slot initialisation PER BATCH ELEMENT. So across-batch variance has TWO sources,
            # and only one of them is the signal:
            #     input dependence  (the thing being measured)   <- grows with training
            #     slot-init noise   (a per-sample random draw)   <- SHRINKS as log_sigma trains
            # Measured at fresh init, B=32, K=4: variance on a batch of IDENTICAL inputs is
            # 3.86e-03 against 3.78e-03 on DISTINCT inputs -- i.e. the bare measure has ZERO
            # discriminating power there, because identical inputs carry no input-dependent
            # variance by construction and it reports the same number anyway.
            # ⇒ A FALLING mask_variance early in training is exactly what the noise term
            # shrinking looks like, and the bare measure cannot tell that from slots collapsing.
            # The insect seat measured precisely that (7/7 brains falling, ratio 0.266-0.493, over
            # the first 4 updates) and correctly declined to call it degeneracy; this is the
            # instrument-level reason their caution was right, and it is a defect in MY diagnostic
            # rather than a limitation of their reading.
            #
            # ⇒ THE NULL: the same statistic on a batch whose inputs are IDENTICAL, so its only
            # source is the slot-init noise. `mask_variance` above `mask_variance_null` is the
            # input-dependent component; at or below it, the masks are not input-dependent and a
            # falling `ss` means nothing. Same shape as `shuffled_acc` in cltt_ref's
            # `nt_xent_diagnostics` -- a null drawn from the same embeddings under a structure
            # that carries no signal. I wrote that pattern for the other module and did not apply
            # it here. [[a-score-a-non-policy-attains]] [[unanimity-is-an-instrument-signal]]
            self.last_mask_variance = float(attn_t.var(dim=0).mean().detach())
            with torch.no_grad():
                flat = prep_t[:1].expand_as(prep_t).contiguous()
                _, attn_null, _, _ = self._slots_for(encoder, flat)
                self.last_mask_variance_null = float(attn_null.var(dim=0).mean())
                # Discriminability of the TEMPORAL POSITIVES -- the question `mask_variance`
                # does not ask and `loss_ss` cannot answer. Uses the SAME slots the loss just
                # consumed, so it measures the objective as run, not a re-derivation of it.
                self.last_ss_diag = slot_contrast_diagnostics(
                    s1.reshape(-1, self.slot_dim).detach(),
                    s2.reshape(-1, self.slot_dim).detach(),
                    self.temperature, self.slots)

        total = self.w_ss * ss + self.w_rec * rec
        self.last_terms = (float(ss.detach()), float(rec.detach()), float(total.detach()))
        # Emitted on EVERY call, whether or not the diagnostic is on -- a value written only
        # where it succeeds makes "off" and "on and degenerate" both present as absent, and
        # absent reads as benign. The sentinel is negative so it cannot be mistaken for a
        # variance, which is >= 0.
        self.last_scalars = {"slots": float(self.slots),
                             "positions": float(self.grid_h * self.grid_w),
                             "mask_variance": float(self.last_mask_variance),
                             "mask_variance_null": float(self.last_mask_variance_null),
                             # The signal, reported directly so nobody has to subtract two
                             # series by hand and nobody reads the bare variance as the answer.
                             "mask_variance_excess": float(self.last_mask_variance
                                                           - self.last_mask_variance_null),
                             **{k: float(v) for k, v in self.last_ss_diag.items()}}
        return total

    def _temporal_pair(self, encoder, observations):
        """(obs_t, obs_{t+1}) drawn from ONE episode-contiguous slab of one env stream.

        ⛔ Uses the shared helpers rather than a bare randint. `episode_window_batch`
        tests ADJACENCY -- it will not return a slab that crosses a reset or the circular
        buffer's write seam -- and `draw_episode_window` picks uniformly among the starts
        that survive. A pair spanning a reset shows a teleport, not motion, and slot
        attention asked to keep a slot on "the same thing" across it is being trained on
        a correspondence that does not exist.
        """
        mem = self._memory
        if mem is None:
            raise RuntimeError(
                "SlotContrastAuxLoss draws its own temporal windows; "
                "call attach_memory(memory) before compute().")
        tensors = getattr(mem, "tensors", None)
        if not tensors or "observations" not in tensors:
            raise ValueError("SlotContrastAuxLoss needs an 'observations' tensor in memory.")
        buf = tensors["observations"]
        t_max = mem.memory_size if getattr(mem, "filled", False) else mem.memory_index
        offsets = (1,)
        try:
            batch, starts = episode_window_batch(mem, offsets, min(self.max_samples, t_max - 1))
        except ValueError as exc:
            # ⛔ RAISE, never return a zero. A zero enters the running mean as a real value
            # and makes "no usable window" look exactly like "the objective scored 0".
            raise ValueError(
                f"{exc} (t_max={t_max}, offsets={offsets}, "
                f"NETT_AUX_BATCH={self.max_samples}). SlotContrast needs at least one "
                "episode-contiguous (t, t+1) pair; without it the slot-slot contrast has "
                "no temporal positive and would be scoring a frame against itself."
            ) from exc
        env, t0 = draw_episode_window(starts)
        # ⛔ .to(device) IS NOT OPTIONAL. campaign_train forces NETT_MEMORY_DEVICE=cpu for
        # EVERY framestacked arm (rollout buffer -> host ram), and this loss is only ever
        # declared on framestacked arms -- so the slice is on the CPU while the encoder is on
        # the GPU, and the first conv raises "Input type (torch.FloatTensor) and weight type
        # (torch.cuda.FloatTensor) should be the same". The candidate never hit this because
        # the screening harness keeps everything on one device. cltt_ref already does the same
        # transfer for the same reason; slicing BEFORE the transfer matters too, or the whole
        # ~2 GB observation buffer is copied to the GPU on every aux call.
        device = next(encoder.parameters()).device
        return (buf[t0: t0 + batch, env].to(device),
                buf[t0 + 1: t0 + 1 + batch, env].to(device))
