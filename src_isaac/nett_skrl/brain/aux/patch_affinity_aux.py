"""P2 `patch_affinity`: a VideoSAUR-style temporal token-affinity target, on our own EMA teacher.

Row 02 of wave 17 asks whether a patch-to-patch temporal affinity target adds anything to CLTT.
The reference (Zadaianchuk, Seitzer & Martius, NeurIPS 2023; code `7a9f85d`) builds a soft target
from a FROZEN PRETRAINED DINO ViT's last-block keys and has slots predict it; both halves must be
substituted here (no pretrained weights on this fleet, and this row has no slots).

REFERENCE -> HERE, with what is faithful and what is not:

| piece | reference | here |
|---|---|---|
| target features | frozen DINO keys, L2-normed | EMA-teacher post-norm tokens, L2-normed |
| similarity | cosine, source patch t vs dest patch t+1 | same, t vs t+k |
| threshold | `s < 0 -> -inf` BEFORE the temperature; an all `-inf` row becomes uniform | faithful |
| temperature | 0.075 (MOVi) .. 0.25 (YT-VIS) | 0.1 (knob), read `target_entropy` |
| softmax axis | destination patch | destination token j |
| offset | 1 frame | k = 8 steps (knob): at k=1 the target IS the identity (spec §0.1) |
| predictor | SlotMixer decoder over slots | token MLP D -> 2D -> N |
| loss | soft-target cross-entropy | faithful |

⚠ WHAT IS HONESTLY LOST. The reference's slots are a bottleneck: the prediction must pass through
K slot vectors, so meeting the target requires grouping. A per-token head has no bottleneck and
can meet the target with a motion code in each token. This row is therefore "VideoSAUR-like
target", not VideoSAUR.

⛔ PREDICT FROM FRAME t ALONE. A bilinear student `<z^t_i, z^{t+k}_j>` would reproduce its own
similarity structure and learn nothing about motion; the reference likewise decodes t's slots and
must anticipate where content goes.

⛔ MEASURED 2026-09-17, AND IT IS WHY `NETT_AUX_AFF_CENTER` EXISTS. At fresh init the teacher's
tokens are dominated by a component every token and every image shares. Decomposing
`u[b,n,:] = m + p[n] + c[b,n]` on 81 real Isaac frames through the production encoder:

    ||m|| (global DC) = 11.73      ||p|| (position) = 2.44      ||c|| (content) = 0.26

Every pairwise cosine is then in [0.881, 1.000] and the τ = 0.1 softmax of a range that narrow is
FLAT: target entropy 0.9939 of ln N, and the whole objective's headroom -- `ce - ce_floor` -- was
0.030 nats on the GPU verification. There is next to nothing to learn, and the falsifier
(`target_var` vs its permuted null) reads as a wash because both are ~1e-7.

Subtracting a running mean of the teacher's features before the L2-norm removes m, the cosine
un-saturates to [-0.74, 1.00], and the target sharpens to 0.213 of ln N with the off-diagonal
mass concentrating on the columns that actually move (object/background 1.21 vs 1.00 uncentred).

⚠ THIS IS DINO'S IDEA, NOT DINO'S OPERATION, AND THE DIFFERENCE MATTERS. DINO centres the
teacher's LOGITS just before its softmax. The literal analogue here would centre the (B,N,N)
cosine matrix, and that CANNOT fix this: when every entry is ≈ 1, subtracting a per-column mean
leaves ≈ 0 and the softmax is just as flat. The saturation is in the FEATURES, so the centring
has to be too -- per feature dimension, before the normalisation that destroys the scale. Same
motivation (a teacher output collapsing onto one direction), different tensor.

⚠ AND IT IS NOT A CURE FOR POSITIONAL DOMINANCE. p still outweighs c by 34x in energy after
centring, so on a PARKED camera the target stays near-identity (0.997) -- which is the correct
answer there, not a defect. Centring by the PER-POSITION mean instead would remove p, and that is
wrong: measured, it destroys the identity structure (identity 0.997 -> 0.165) and the residual
signal with it. `pair_dep` per column band is what separates "identity because nothing moved"
from "identity because the target is blind".

WHY THE TARGET IS NOT DEGENERATE UNDER AN EMA TEACHER. T is detached, so the student's optimum is
`softmax(logits) = T` with loss floor H(T). A CONSTANT teacher gives cosine 1 everywhere, hence a
UNIFORM T and the MAXIMUM floor ln N -- collapse to a constant is not rewarded. The live
degenerate modes are the identity target (k too small or parked), an image-blind positional
target, a uniform target (τ too high), and the threshold wipe; each has a diagnostic below, each
with a null, and all are emitted on every call.

⛔ OBJECTIVE CHANGE 2026-09-17 (owner, workspace DECISIONS): `cltt_ref` now excludes each anchor's OWN FRAME from its negatives, so every cltt_ref arm trained before this commit ran a different objective and is NOT comparable to one trained after it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .knobs import _env_flag_strict, _env_positive_float, _env_unit_interval
from .token_term import NOT_MEASURED, TokenWindow, TokenWindowTerm, parked_transit, stratum_mean


def affinity_target(u_t: torch.Tensor, u_tk: torch.Tensor,
                    temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """VideoSAUR's temporal feature-similarity target -> (T (B,N,N), rows_all_negative (B,N)).

    ⚠ ORDER MATTERS AND IS THE REFERENCE'S: threshold at 0 BEFORE dividing by τ
    (`videosaur/modules/utils.py:492-495`), and a row whose every entry was thresholded away is
    set to zeros and so becomes UNIFORM after the softmax (`:503-508`) rather than NaN.
    """
    a = F.normalize(u_t, p=2.0, dim=-1)
    b = F.normalize(u_tk, p=2.0, dim=-1)
    s = torch.bmm(a, b.transpose(1, 2))                      # (B, N, N) cosine
    negative = s < 0.0
    all_negative = negative.all(dim=-1)                      # (B, N)
    s = s.masked_fill(negative, float("-inf")) / temperature
    s = s.masked_fill(all_negative.unsqueeze(-1), 0.0)
    return torch.softmax(s, dim=-1), all_negative


class PatchAffinityTerm(TokenWindowTerm):
    """Soft-target CE between a token MLP's logits at t and the EMA teacher's (t, t+k) affinity."""

    BATCH_ENV = "NETT_AUX_AFF_BATCH"
    DEFAULT_BATCH = 32
    OFFSET_ENV = "NETT_AUX_AFF_OFFSET"
    DEFAULT_OFFSET = 8
    TEMP_ENV = "NETT_AUX_AFF_TEMP"
    CENTER_ENV = "NETT_AUX_AFF_CENTER"
    CENTER_MOMENTUM_ENV = "NETT_AUX_AFF_CENTER_MOMENTUM"
    TRANSIT_WEIGHTED = True

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(encoder)
        self.temperature = _env_positive_float(self.TEMP_ENV, 0.1)
        self.centred = _env_flag_strict(self.CENTER_ENV, True)
        self.center_momentum = _env_unit_interval(self.CENTER_MOMENTUM_ENV, 0.9)
        if self.centred and self.center_momentum >= 1.0:
            raise ValueError(
                f"{self.CENTER_MOMENTUM_ENV}=1.0 freezes the centre at its zero init, so "
                f"{self.CENTER_ENV}=1 would subtract nothing while every log line says the "
                f"target is centred. Use a momentum < 1, or turn the centring off explicitly.")
        d, n = self.token_dim, self.n_tokens
        self.head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, n))
        self.head.to(next(encoder.parameters()).device)
        # ⚠ A BUFFER, NOT A PARAMETER, AND NOT PART OF `head`. It must never receive gradient
        # (it is a statistic of the teacher, which is itself stop-grad) and must never reach the
        # optimiser param groups the composite asserts over. persistent=False for the same
        # reason the teacher is not checkpointed: nothing in this wave restores aux state.
        self.register_buffer("center", torch.zeros(1, 1, d, device=next(encoder.parameters()).device),
                             persistent=False)

    def _centre(self, u_t: torch.Tensor, u_tk: torch.Tensor):
        """Subtract the running teacher-feature mean, then update it. DINO's ORDER.

        The batch contributes to the centre used by the NEXT call, not to its own -- otherwise
        the subtraction is partly of the batch itself and a single-sample batch would centre to
        exactly zero. Both frames share one centre: two centres would introduce a t-vs-t+k
        offset that the cosine would read as motion.
        """
        if not self.centred:
            return u_t, u_tk
        c = self.center
        out = (u_t - c, u_tk - c)
        batch_mean = torch.cat([u_t, u_tk]).mean(dim=(0, 1), keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_mean, alpha=1.0 - self.center_momentum)
        return out

    def _core(self, encoder: nn.Module, window: TokenWindow):
        from .token_features import spatial_tokens

        u_t, _ = self.teacher.tokens(window.prepared_t)
        u_tk, _ = self.teacher.tokens(window.prepared_tk)
        u_t, u_tk = self._centre(u_t, u_tk)
        target, all_negative = affinity_target(u_t, u_tk, self.temperature)   # (B,N,N)

        z = spatial_tokens(encoder, window.prepared_t)[0]                     # grad ON
        logits = self.head(z)                                                 # (B,N,N)
        loss = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        with torch.no_grad():
            scalars = self._diagnostics(window, target, all_negative, u_t, u_tk, logits)
        return loss, scalars

    @torch.no_grad()
    def _diagnostics(self, window, target, all_negative, u_t, u_tk, logits) -> dict:
        n = target.shape[-1]
        match = target.argmax(dim=-1)                                         # (B, N)
        entropy = -(target.clamp_min(1e-12).log() * target).sum(-1).mean()
        # ⛔ PAIRED INPUT-DEPENDENCE NULL. An image-blind target (teacher tokens ~ the positional
        # embedding) is SHARP and looks healthy on entropy and identity alike, but is the same
        # matrix for every sample. The null re-computes T with the t+k tokens PERMUTED across the
        # batch, i.e. the same marginal structure with the pairing destroyed.
        #
        # ⛔ READ `input_dep` AS AN EXACTNESS TEST, NEVER AS A SIGNED MAGNITUDE -- the correction
        # ego_residual's D4 already carries, and the reason it is repeated here is that this one
        # was read the other way in review. `var - var_null` is NEGATIVE for a healthy target and
        # that is expected arithmetic, not a failure: consecutive frames give SIMILAR matrices
        # across the batch (low variance), while random pairings give dissimilar ones (high
        # variance). Measured on real frames with centring: var 8.97e-05, null 2.02e-04. The
        # informative statement is |excess| > 0, i.e. the pairing reaches the target at all.
        #
        # ⇒ `pair_dep` is the statistic that answers it without the sign trap: the total-variation
        # distance between each row of T and the same row under the permuted pairing. It is
        # EXACTLY 0 for an image-blind target, in [0, 1], and monotone in how much the
        # destination frame matters. Verified against a batch of identical frames, where it is 0.
        perm = torch.randperm(u_tk.shape[0], device=u_tk.device)
        var = float(target.var(dim=0).mean()) if target.shape[0] > 1 else NOT_MEASURED
        if target.shape[0] > 1:
            null, _ = affinity_target(u_t, u_tk[perm], self.temperature)
            var_null = float(null.var(dim=0).mean())
            pair_dep = float(0.5 * (target - null).abs().sum(dim=-1).mean())
        else:
            var_null = pair_dep = NOT_MEASURED
        # Is the student anywhere near the target it is being asked for? The CE floor is H(T);
        # the gap is what is left to learn, and a gap pinned at 0 with entropy at ln N means the
        # task was uniform (nothing to learn), not solved.
        ce = float(-(target * F.log_softmax(logits, dim=-1)).sum(-1).mean())
        parked, transit = parked_transit(window.turn)
        rows_neg = all_negative.float().mean(dim=1)
        return {
            **self.identity_fraction(match, window.turn),
            **self.shift_correlation(match, window.a_bar[:, 0]),
            "target_entropy": float(entropy),
            "target_entropy_frac": float(entropy) / math.log(n),
            "rows_all_neg": float(rows_neg.mean()),
            "rows_all_neg_parked": stratum_mean(rows_neg, parked),
            "rows_all_neg_transit": stratum_mean(rows_neg, transit),
            "input_dep": (var - var_null) if NOT_MEASURED not in (var, var_null)
                         else NOT_MEASURED,
            "target_var": var,
            "target_var_null": var_null,
            "pair_dep": pair_dep,
            "centred": 1.0 if self.centred else 0.0,
            # The centre's size against the tokens it is subtracted from: a centre that stays at
            # 0 is the knob not working, and one that grows without bound is a drifting teacher.
            "center_norm": float(self.center.norm()),
            "token_norm": float(u_t.norm(dim=-1).mean()),
            "ce": ce,
            "ce_floor": float(entropy),
            "temperature": float(self.temperature),
        }
