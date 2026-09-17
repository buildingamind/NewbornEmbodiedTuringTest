"""P3 `ego_residual`: predict the token transport the ACTION causes; what is left over is the object.

This is wave 17's C3 test (plan of record, item 3): ego-motion is larger than the object's own
motion on this apparatus, so "remove what the action explains" is the one manipulation that can
expose the object without depth, parallax or a background prior.

```
ā   = Σ_{t'∈[t,t+k)} actions[t', :2]                          (B, 2)   cumulative command
z   = encode_tokens(obs_t)                       (B, N, D)  student, grad ON
u0  = encode_tokens_EMA(obs_t), u1 = ..._EMA(obs_{t+k})      (B, N, D) teacher, no_grad
A   = softmax_j( MLP_A(ā) + c·I )                            (B, N, N) content-INDEPENDENT routing
ẑ   = A · q(z)                                               (B, N, D)
L   = mean_{b,i} ( 2 − 2·cos(ẑ_bi, sg u1_bi) )                        BYOL/SPR normalised MSE
r   = normalize(u1) − normalize(A·u0)   (teacher on BOTH sides, no_grad)
e   = ‖r‖²/2 = 1 − cos(...)                                  (B, N)    per-token residual
w   = softmax_i( z-scored e )                                (B, N)    per-token objectness, detached
```

⛔ THE ROUTING IS CONTENT-INDEPENDENT ON PURPOSE, AND THAT IS THE WHOLE DESIGN. The stimulus is a
video that LOOPS DETERMINISTICALLY (research spec §0.6). A content-aware predictor -- a
transformer over tokens with an action token, say -- can learn the loop, predict the object's own
motion, and drive the object's residual to ZERO: the exact opposite of what the residual is for,
reached by a model with a better forward loss. Parameterising the transport by the action ALONE
(Oh et al. 2015's multiplicative action gating; Iso-Dream's action-conditioned / action-free
split) makes the residual "what a global, action-driven transport cannot explain".

⛔ AND THE TARGET IS THE EMA TEACHER, NOT THE LIVE TRUNK. With a live target, z ≡ constant gives
ẑ = A·q(c) = q(c), cosine 1 and L = 0 with nothing encoded, and the routing gets no gradient
because all tokens are equal. The stop-grad EMA target plus the predictor q (the BYOL asymmetry)
removes that minimiser from gradient reach. It is not a proof: `token_std` is emitted every call
so a drift toward zero is visible.

⚠ c (NETT_AUX_EGO_IDENTITY_BIAS) IS CALIBRATED, NOT ARBITRARY. At ā = 0 the routing must start
near the identity, but c = 10 gives p_diag = e¹⁰/(e¹⁰+N−1) = 0.998 and a softmax gradient factor
p(1−p) ≈ 0.002, so A would SIT at the identity whatever ā is -- manufacturing D5's "not engaged"
reading. c = 4 gives p_diag ≈ 0.58 at N = 40. `route_p_diag_at_zero` is emitted so the realised
value is read rather than assumed.

⚠ SAMPLING IS A 50/50 MIXTURE OF TRANSIT-WEIGHTED AND UNIFORM WINDOWS, drawn as TWO half-slabs
through the same P0 sampler. Parked windows are where objectness is cleanest (A(0) ≈ I, so the
residual is the object's own motion); transit windows are where the routing learns ego
compensation. One Bernoulli choice per update would leave 63% of updates with no transit step at
all, and then the parked/transit median split every diagnostic here uses would be degenerate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .knobs import _env_nonneg_float, _env_unit_interval
from .token_term import (
    NOT_MEASURED, TokenWindow, TokenWindowTerm, parked_transit, rank_corr, stratum_mean,
)


def objectness_from_residual(e: torch.Tensor) -> torch.Tensor:
    """Per-token residual -> per-token objectness in [0, 1], detached and SCALE-FREE.

    z-score across tokens, softmax over tokens, then divide by the row max. The z-score is what
    makes it scale-free: the residual's absolute size changes with the token norm and with how
    well the routing currently works, and a fixed threshold on it would mean a different thing at
    every update.
    """
    z = (e - e.mean(dim=1, keepdim=True)) / (e.std(dim=1, keepdim=True) + 1e-6)
    w = torch.softmax(z, dim=1)
    return (w / w.amax(dim=1, keepdim=True).clamp_min(1e-12)).detach()


class EgoResidualTerm(TokenWindowTerm):
    """Action-conditioned token routing + residual objectness. See the module docstring."""

    BATCH_ENV = "NETT_AUX_EGO_BATCH"
    DEFAULT_BATCH = 32
    OFFSET_ENV = "NETT_AUX_EGO_OFFSET"
    DEFAULT_OFFSET = 8
    IDENTITY_BIAS_ENV = "NETT_AUX_EGO_IDENTITY_BIAS"
    TRANSIT_FRAC_ENV = "NETT_AUX_EGO_TRANSIT_FRAC"
    #: Sampling is the explicit 50/50 mixture below, not the base class's single draw.
    TRANSIT_WEIGHTED = False

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(encoder)
        self.identity_bias = _env_nonneg_float(self.IDENTITY_BIAS_ENV, 4.0)
        self.transit_frac = _env_unit_interval(self.TRANSIT_FRAC_ENV, 0.5)
        n, d = self.n_tokens, self.token_dim
        route = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, n * n))
        # ⛔ ZERO-INIT THE LAST LAYER. The routing must START at the identity-biased softmax --
        # "no action, no transport" -- so that whatever it learns is a departure the action
        # bought. A random init would put a fixed random permutation between the two frames at
        # step 0 and the loss would spend its early budget undoing it.
        nn.init.zeros_(route[-1].weight)
        nn.init.zeros_(route[-1].bias)
        # ⚠ CONSEQUENCE, PINNED BY A TEST RATHER THAN LEFT TO SURPRISE: with the last layer at
        # zero, the FIRST layer receives exactly zero gradient on the first backward (dL/dh =
        # W_last^T dL/dlogits = 0). The last layer itself does get gradient, so the pathway
        # unblocks after one optimizer step. A reader who checks "does the action head train?"
        # at step 0 and sees zeros is looking at the init, not at a dead knob.
        self.head = nn.ModuleDict({
            "route": route,
            "predict": nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)),
        })
        self.head.to(next(encoder.parameters()).device)
        eye = torch.eye(n, device=next(encoder.parameters()).device)
        self.register_buffer("_eye", eye, persistent=False)

    # -- the model ------------------------------------------------------------------

    def routing(self, a_bar: torch.Tensor) -> torch.Tensor:
        """(B, 2) cumulative action -> (B, N, N) row-stochastic transport.

        ``A[b, i, j]`` is the weight of SOURCE token j into DESTINATION token i; the softmax is
        over j, so every destination is a convex combination of sources.
        """
        n = self.n_tokens
        logits = self.head["route"](a_bar[:, :2]).view(-1, n, n)
        return torch.softmax(logits + self.identity_bias * self._eye, dim=-1)

    def loss_and_objectness(self, encoder: nn.Module, window: TokenWindow):
        """-> (loss, objectness w (B,N) detached, scalars). The entry point P5 shares.

        ⚠ P5 (slot_fg + ego) calls this on ITS OWN window, so the objectness map is aligned with
        the slot masks it supervises. Returning w rather than storing it on `self` is deliberate:
        a `self.last_objectness` read by the other term would pair whatever window ran last.
        """
        from .token_features import spatial_tokens

        u0, _ = self.teacher.tokens(window.prepared_t)
        u1, _ = self.teacher.tokens(window.prepared_tk)
        a_bar = window.a_bar
        A = self.routing(a_bar)
        z = spatial_tokens(encoder, window.prepared_t)[0]
        pred = torch.bmm(A, self.head["predict"](z))
        loss = (2.0 - 2.0 * F.cosine_similarity(pred, u1, dim=-1)).mean()

        with torch.no_grad():
            e = self._residual(A, u0, u1)
            w = objectness_from_residual(e)
            scalars = self._diagnostics(window, A, u0, u1, e, w, z)
        return loss, w, scalars

    @staticmethod
    def _residual(A: torch.Tensor, u0: torch.Tensor, u1: torch.Tensor) -> torch.Tensor:
        """1 − cos(u1_i, (A·u0)_i) per token. TEACHER ON BOTH SIDES.

        ⛔ Not `ẑ` against u1: ẑ runs through the trainable predictor q, so a residual read off
        it would fall as q fits and would report "the object disappeared" when only the predictor
        improved. The residual must be a property of the TRANSPORT, so both sides are teacher
        tokens and only A is shared with the loss.
        """
        moved = F.normalize(torch.bmm(A, u0), p=2.0, dim=-1)
        return 1.0 - (F.normalize(u1, p=2.0, dim=-1) * moved).sum(dim=-1)

    # -- sampling -------------------------------------------------------------------

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        window = self.draw_mixed(encoder, transit_frac=self.transit_frac)
        loss, _w, scalars = self.loss_and_objectness(encoder, window)
        self.last_window_turn = float(window.mean_turn)
        self.last_scalars = {"B": float(window.prepared_t.shape[0]), "k": float(self.offset),
                             "window_turn": float(window.mean_turn),
                             **{k: float(v) for k, v in scalars.items()}}
        return loss

    def _core(self, encoder: nn.Module, window: TokenWindow):
        loss, _w, scalars = self.loss_and_objectness(encoder, window)
        return loss, scalars

    # -- diagnostics (research spec §5.5; every one has a null) ----------------------

    def _columns(self):
        """(centre mask, periphery mask, no-edge periphery mask) over the N tokens.

        Centre = the middle half of the COLUMNS, all rows. ⚠ Rows are not cropped: at the parked
        distance the object spans ~4.4 of the 5 rows, so a row-cropped box would push object
        tokens into "periphery" and dilute the ratio. The no-edge variant drops the outermost
        columns, where new content enters during transit and no routing can predict it.
        """
        n_h, n_w = self.grid
        col = torch.arange(self.n_tokens, device=self._eye.device) % n_w
        lo, hi = n_w // 4, n_w - n_w // 4
        centre = (col >= lo) & (col < hi)
        periph = ~centre
        no_edge = periph & (col > 0) & (col < n_w - 1)
        return centre, periph, no_edge

    @torch.no_grad()
    def _diagnostics(self, window, A, u0, u1, e, w, z) -> dict:
        centre, periph, no_edge = self._columns()
        parked, transit = parked_transit(window.turn)
        turn = window.turn

        def ratio(values, periph_mask):
            num = values[:, centre].mean(dim=1)
            den = values[:, periph_mask].mean(dim=1).clamp_min(1e-12)
            return num / den

        # D1 centre ratio, with BOTH comparators: the raw-pixel ratio (the in-loop analogue of
        # C3's 12-48x) and a within-sample token permutation (which must give ~1).
        r_res = ratio(e, periph)
        r_res_noedge = ratio(e, no_edge)
        perm_tokens = torch.stack([e[b][torch.randperm(e.shape[1], device=e.device)]
                                   for b in range(e.shape[0])])
        r_res_null = ratio(perm_tokens, periph)
        r_pix = ratio(self._pixel_change(window), periph)

        # D2 action gain: the same residual with ā PERMUTED across the batch. If the routing
        # ignores the action, the permutation changes nothing and G = 1.
        perm = torch.randperm(A.shape[0], device=A.device)
        e_perm = self._residual(self.routing(window.a_bar[perm]), u0, u1)
        gain = e_perm.mean(dim=1) / e.mean(dim=1).clamp_min(1e-12)

        # D3 compensation gain: how much of the peripheral change the routing removed.
        raw_change = (F.normalize(u1, dim=-1) - F.normalize(u0, dim=-1)).norm(dim=-1)
        comp = (raw_change[:, periph].mean(dim=1)
                / (2.0 * e[:, periph]).clamp_min(1e-12).sqrt().mean(dim=1).clamp_min(1e-12))

        # D4 fixed-position artefact: is w input-dependent, or the same map every time? Paired
        # null -- the SAME u0 and routing, with the t+k teacher tokens permuted across the batch.
        # ⛔ READ THE EXCESS AS AN EXACTNESS TEST, NOT AS A MAGNITUDE, and this is the lesson
        # slot_contrast_aux paid for with a 200-seed sweep. If w does not depend on WHICH frames
        # were paired -- the fixed-position artefact, e.g. a bezel edge or a disocclusion band --
        # permuting the destination frames changes nothing and the excess is EXACTLY 0.0. When w
        # does depend on the pairing the excess is nonzero, but its SIGN is not informative: a
        # destroyed pairing raises the residual everywhere, which can raise the across-batch
        # variance too. So: |excess| ~ 0 means "the map is the same whatever we paired"; a
        # nonzero value means the pairing reaches w, and how much is not this statistic's answer.
        if w.shape[0] > 1:
            w_null = objectness_from_residual(self._residual(A, u0, u1[perm]))
            var, var_null = float(w.var(dim=0).mean()), float(w_null.var(dim=0).mean())
            w_input_dep = var - var_null
        else:
            var = var_null = w_input_dep = NOT_MEASURED

        # D5 routing identity, top vs bottom quartile of |turn|.
        diag = A.diagonal(dim1=1, dim2=2).mean(dim=1)
        if turn.numel() >= 4:
            # On the CPU for the same reason the median split is (see token_term.parked_transit):
            # order statistics under the update's strict-determinism guard.
            q = torch.quantile(turn.detach().cpu(), torch.tensor([0.25, 0.75]))
            q_lo, q_hi = q[0].to(turn.device), q[1].to(turn.device)
            diag_top = stratum_mean(diag, turn >= q_hi)
            diag_bottom = stratum_mean(diag, turn <= q_lo)
        else:
            diag_top = diag_bottom = NOT_MEASURED

        zero_action = self.routing(torch.zeros(1, 2, device=A.device))
        norm_z = F.normalize(z, dim=-1)
        return {
            "res_centre_ratio": float(r_res.mean()),
            "res_centre_ratio_parked": stratum_mean(r_res, parked),
            "res_centre_ratio_transit": stratum_mean(r_res, transit),
            "res_centre_ratio_noedge": float(r_res_noedge.mean()),
            "res_centre_ratio_null": float(r_res_null.mean()),
            "pix_centre_ratio": float(r_pix.mean()),
            "pix_centre_ratio_parked": stratum_mean(r_pix, parked),
            "pix_centre_ratio_transit": stratum_mean(r_pix, transit),
            "action_gain": float(gain.mean()),
            "action_gain_parked": stratum_mean(gain, parked),
            "action_gain_transit": stratum_mean(gain, transit),
            "compensation_gain": float(comp.mean()),
            "compensation_gain_parked": stratum_mean(comp, parked),
            "compensation_gain_transit": stratum_mean(comp, transit),
            "objectness_var": var,
            "objectness_var_null": var_null,
            "objectness_input_dep": w_input_dep,
            "route_diag_top_quartile": diag_top,
            "route_diag_bottom_quartile": diag_bottom,
            "route_p_diag_at_zero": float(zero_action.diagonal(dim1=1, dim2=2).mean()),
            # Collapse watch: the per-dimension spread of the normalised student tokens. Falling
            # toward 0 is the constant-z mode the EMA target exists to keep out of reach.
            "token_std": float(norm_z.reshape(-1, norm_z.shape[-1]).std(dim=0).mean()),
            "residual_mean": float(e.mean()),
            "transit_frac": float(self.transit_frac),
            # Free ground truth: does the residual's centre concentration move with the action?
            "res_ratio_rho_turn": rank_corr(r_res, window.a_bar[:, 0]),
        }

    @torch.no_grad()
    def _pixel_change(self, window) -> torch.Tensor:
        """Per-token mean |Δpixel| of the CURRENT frame, as D1's raw comparator.

        ⚠ The current frame only: a framestack's older channels are shared between t and t+k at
        small k and would dilute the change. Pooled on the token grid with an exact reshape --
        the patch size divides the eye by construction (CompactViT asserts it).
        """
        from .cltt_views import resolve_channels_per_frame
        cpf = resolve_channels_per_frame()
        n_h, n_w = self.grid
        diff = (window.prepared_tk[:, -cpf:] - window.prepared_t[:, -cpf:]).abs().mean(dim=1)
        b, h, w = diff.shape
        ph, pw = h // n_h, w // n_w
        return diff.view(b, n_h, ph, n_w, pw).mean(dim=(2, 4)).reshape(b, n_h * n_w)
