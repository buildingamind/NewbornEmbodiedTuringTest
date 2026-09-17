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

WHY THE TARGET IS NOT DEGENERATE UNDER AN EMA TEACHER. T is detached, so the student's optimum is
`softmax(logits) = T` with loss floor H(T). A CONSTANT teacher gives cosine 1 everywhere, hence a
UNIFORM T and the MAXIMUM floor ln N -- collapse to a constant is not rewarded. The live
degenerate modes are the identity target (k too small or parked), an image-blind positional
target, a uniform target (τ too high), and the threshold wipe; each has a diagnostic below, each
with a null, and all are emitted on every call.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .knobs import _env_positive_float
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
    TRANSIT_WEIGHTED = True

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(encoder)
        self.temperature = _env_positive_float(self.TEMP_ENV, 0.1)
        d, n = self.token_dim, self.n_tokens
        self.head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, n))
        self.head.to(next(encoder.parameters()).device)

    def _core(self, encoder: nn.Module, window: TokenWindow):
        from .token_features import spatial_tokens

        u_t, _ = self.teacher.tokens(window.prepared_t)
        u_tk, _ = self.teacher.tokens(window.prepared_tk)
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
        # batch, i.e. the same marginal structure with the pairing destroyed; the excess is the
        # part that depends on which frames were actually paired. Same shape as
        # `mask_variance_null` in slot_contrast_aux.
        perm = torch.randperm(u_tk.shape[0], device=u_tk.device)
        var = float(target.var(dim=0).mean()) if target.shape[0] > 1 else NOT_MEASURED
        if target.shape[0] > 1:
            null, _ = affinity_target(u_t, u_tk[perm], self.temperature)
            var_null = float(null.var(dim=0).mean())
        else:
            var_null = NOT_MEASURED
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
            "ce": ce,
            "ce_floor": float(entropy),
            "temperature": float(self.temperature),
        }
