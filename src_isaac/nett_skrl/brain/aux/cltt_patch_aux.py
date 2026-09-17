"""P1 `cltt_patch`: CLTT on the PATCH TOKENS, with DenseCL's correspondence and VICRegL's filter.

Owner item 1 for wave 17: CLTT today contrasts the CLS readout (`cltt_ref`), which is one vector
per frame; this row asks whether the same temporal contrast over TOKENS teaches the trunk
something the readout cannot. The row is `cltt_ref` (UNCHANGED, so row 00 keeps its definition)
plus this term.

WHY argmax CORRESPONDENCE AND NOT SAME-LOCATION. Same-location grid contrast assumes the camera
is still between the two views; here the agent turns, so at 8 steps of transit the image has
moved ~0.75 of a patch. DenseCL (Wang et al. CVPR 2021, `5d1c456`) takes the positive as
`c_i = argmax_j sim(f_i, f'_j)` on BACKBONE features of a MOMENTUM encoder -- no geometry needed,
and it degrades gracefully to same-location exactly when the agent is parked.

```
views      cltt_ref's construction: each view is the CURRENT frame repeated across the T slots
teacher    u^t, u^{t+k} = encode_tokens_EMA(view)          (B,N,D) no_grad
match      S = norm(u^t)·norm(u^{t+k})^T;  c_i = argmax_j S_ij;  conf_i = max_j S_ij
filter     keep the top-γ tokens by conf per sample (γ = 20 of 40), then sample M = 8 of them
student    p = g(z^t_i), p' = g(z^{t+k}_{c_i}),  g = cltt_ref's 512-BN-128 projector, L2-normed
loss       NT-Xent over the 2·B·M tokens at τ = 0.5 -- the SAME function and temperature as
           cltt_ref, so the row stays inside one contrastive family
```

⚠ TOP-γ IS VICRegL's, AND IT IS NOT COSMETIC. The background here is near-uniform texture, on
which the argmax is a near-tie and the "correspondence" is noise; dropping the least confident
half removes most of those. VICRegL keeps its top 20 pairs for the same reason.

⚠ WHY M = 8 AND NOT ALL 40. Per-sample negatives (the other 39 tokens of the same pair) are
~17.6 tokens of the SAME OBJECT at the campaign geometry, so nearly half the negatives would
penalise within-object coherence. All-pairs at M = 40 and B = 512 OOMs a 24 GB card (measured,
research spec §6.4). M = 8 over the batch gives 2BM = 8,192 rows.

⚠ NOT RUNNABLE UNDER `NETT_AUX_STRICT=1`. The anchors are selected with `gather`, whose BACKWARD
is `index_add`-shaped and has no deterministic CUDA kernel. That is fine as shipped -- ppo_aux
runs the auxiliary backward inside `relaxed_determinism()` -- but the fused fully-strict control
path would raise. The same is true of any token-selection objective; it is stated here rather
than discovered on the control run.

⛔ COMPARATOR HAZARD, HANDLED BY DRAWING SEPARATELY. If this term shared cltt_ref's slab by
adding offset 8 to its offsets, the set of valid slab starts would change (8 more contiguous
steps required), so row 00's own sampler would no longer see the same valid-start set as the
control it is compared to. This term therefore draws its OWN window through the P0 sampler and
`cltt_ref` is called exactly as the control calls it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cltt_ref_aux import CLTTReferenceProjectionHead, nt_xent_diagnostics
from .cltt_views import current_frame_stack
from .knobs import _env_positive_float, _env_positive_int
from .simclr_aux import nt_xent
from .token_term import TokenWindow, TokenWindowTerm


class CLTTPatchTerm(TokenWindowTerm):
    """Dense NT-Xent over EMA-matched token pairs from one episode-contiguous window."""

    BATCH_ENV = "NETT_AUX_PATCH_BATCH"
    DEFAULT_BATCH = 512
    OFFSET_ENV = "NETT_AUX_CLTT_PATCH_OFFSET"
    DEFAULT_OFFSET = 8
    TOPG_ENV = "NETT_AUX_PATCH_TOPG"
    SAMPLES_ENV = "NETT_AUX_PATCH_M"
    TEMP_ENV = "NETT_AUX_PATCH_TEMP"
    TRANSIT_WEIGHTED = True

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(encoder)
        self.top_gamma = _env_positive_int(self.TOPG_ENV, max(1, self.n_tokens // 2))
        self.samples = _env_positive_int(self.SAMPLES_ENV, 8)
        self.temperature = _env_positive_float(self.TEMP_ENV, 0.5)
        if self.top_gamma > self.n_tokens:
            raise ValueError(
                f"{self.TOPG_ENV}={self.top_gamma} exceeds the token count {self.n_tokens}: the "
                f"filter would keep every token and the knob would be inert.")
        if self.samples > self.top_gamma:
            raise ValueError(
                f"{self.SAMPLES_ENV}={self.samples} exceeds {self.TOPG_ENV}={self.top_gamma}: "
                f"sampling more anchors than the filter keeps would silently re-admit the "
                f"low-confidence matches the filter exists to drop.")
        # The projector cltt_ref uses, on the TOKEN dim rather than features_dim, so the two
        # terms of this row stay one contrastive family.
        self.head = CLTTReferenceProjectionHead(self.token_dim)
        self.head.to(next(encoder.parameters()).device)

    def _views(self, window: TokenWindow) -> tuple[torch.Tensor, torch.Tensor]:
        """cltt_ref's view construction: the CURRENT frame repeated across the stack slots.

        ⚠ This is what keeps t and t+k from sharing a literal frame when the framestack overlaps,
        and it is the construction the control row runs, so the two terms differ in WHAT is
        contrasted (tokens vs the CLS readout) rather than in what they look at.
        """
        return current_frame_stack(window.prepared_t), current_frame_stack(window.prepared_tk)

    def _core(self, encoder: nn.Module, window: TokenWindow):
        from .token_features import spatial_tokens

        view_t, view_tk = self._views(window)
        u_t, _ = self.teacher.tokens(view_t)
        u_tk, _ = self.teacher.tokens(view_tk)
        with torch.no_grad():
            sim = torch.bmm(F.normalize(u_t, dim=-1), F.normalize(u_tk, dim=-1).transpose(1, 2))
            conf, match = sim.max(dim=-1)                       # (B, N) each
            keep = conf.topk(self.top_gamma, dim=1).indices     # (B, γ) most confident tokens
            # Sample M of the kept per sample, WITHOUT replacement: argsort of uniform noise is
            # a permutation, and gather-only (no scatter) keeps the forward deterministic under
            # the strict-determinism guard the PPO update runs inside.
            pick = torch.rand(keep.shape, device=keep.device).argsort(dim=1)[:, :self.samples]
            anchors = keep.gather(1, pick)                      # (B, M) source token indices
            partners = match.gather(1, anchors)                 # (B, M) their matched tokens

        z_t = spatial_tokens(encoder, view_t)[0]
        z_tk = spatial_tokens(encoder, view_tk)[0]
        d = z_t.shape[-1]
        anchor_tok = z_t.gather(1, anchors.unsqueeze(-1).expand(-1, -1, d)).reshape(-1, d)
        partner_tok = z_tk.gather(1, partners.unsqueeze(-1).expand(-1, -1, d)).reshape(-1, d)
        p, p_pos = self.head(anchor_tok), self.head(partner_tok)
        loss = nt_xent(p, p_pos, self.temperature)

        with torch.no_grad():
            scalars = {
                **self.identity_fraction(match, window.turn),
                **self.shift_correlation(match, window.a_bar[:, 0]),
                # ⛔ pos_acc against the DERANGED-positive null, the same instrument cltt_ref
                # uses: at 2BM rows a low loss can come from an easy task rather than a learned
                # correspondence, and the null is what tells those apart.
                **{f"patch_{k}": v for k, v in
                   nt_xent_diagnostics(p.detach(), p_pos.detach(), self.temperature).items()},
                "match_conf": float(conf.mean()),
                "match_conf_kept": float(conf.gather(1, keep).mean()),
                "top_gamma": float(self.top_gamma),
                "anchors_per_sample": float(self.samples),
                "rows": float(p.shape[0] * 2),
            }
        return loss, scalars
