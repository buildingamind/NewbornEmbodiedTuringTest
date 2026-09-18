"""P4 `slot_fg`: slots over the ViT TOKENS, with the SAVi correspondence our CNN port is missing,
plus Tian et al.'s foreground/background indicator -- and, under one knob, P5: the ego residual
as the foreground target.

THREE CHANGES FROM `slot_contrast_aux.py` (which is a SHIPPED arm and is not touched here):

1. ⛔ SLOT CORRESPONDENCE BY CONSTRUCTION. The port draws a FRESH random slot init at t and again
   at t+k, so "slot k at t matches slot k at t+k" -- the identity target its contrastive loss
   asks for -- is a correspondence that does not exist; the two share only a distribution. That
   is the leading candidate for I77 ("the fleet's only slot arm never engaged, no-correspondence
   at 72 positions"). Here the init at t is a FIXED LEARNED vector (SlotContrast's
   `FixedLearnedInit`, SAVi's `ParamStateInit`) and the init at t+k is `predictor(slots_t)`
   (SAVi's corrector/predictor loop, `savi/modules/video.py:40-80`), so slot k at t+k STARTS from
   slot k at t.
2. K = 4, not 6: 40 tokens over 6 slots is 6.7 tokens per slot against an object that occupies
   ~17.6 (C1). Object, background, bezel/room, spare.
3. A decoder and a reconstruction target that are now KNOBS, because a held-out screen moved
   both (see THE SCREENED CONFIGURATION below).

FOREGROUND INDICATOR (Tian et al., CVPR 2025). ⛔ The paper's fg/bg symmetry is broken ONLY by
its pretrained DINO features (its code hard-codes slot index 0 as foreground), and under C4 --
one background for the whole campaign -- its L_stuff (bg of image i vs bg of image j) is
uninformative on its own. So:
- `NETT_AUX_SLOTFG_EGO=1` (row 05) composes P3 INSIDE this term: the ego residual's per-token
  objectness `w` supervises slot 0's mask, which is the symmetry breaker the method is missing,
  and the ego forward loss is added so the routing that produces `w` is actually trained. The
  ego window is THIS term's window, so `w` and the slot masks are aligned token for token.
- Without it (row 04) there is no symmetry breaker and this row must be read as "2..4-slot SA
  over tokens + temporal slot contrast + reconstruction". `fg_mass` on slot 0 is then arbitrary
  by construction, and a centre correlation at chance is the EXPECTED null, not a bug.

⚠ L_sep's AXIS IS UNVERIFIED IN THE PAPER and the choice here is deliberate: the entropy is taken
over the slot-MASS MARGINAL (mean over tokens of the per-token slot assignment), because the
ablation it is justified by is "one slot binds to the entire image and the other remains empty" --
a statement about the marginal. Maximising the PER-TOKEN entropy instead would push every token
to a uniform assignment, i.e. to no segmentation at all, which is the opposite of the stated
purpose. `fg_mass` and `sep_entropy` are emitted so the realised behaviour is read.

⚠ λ = 1.0 and γ = 0.1 are PLACEHOLDERS: the paper gives no values (§2.6 of the research spec
records this as UNVERIFIED). Both are knobs.

THE SCREENED CONFIGURATION (FINDINGS §4by.1, held-out synthetic foreground-ARI, n=3 seeds
against a measured seed floor). The shipped pair -- per-token MLP decoder onto EMA-TEACHER TOKEN
targets -- reads 0.320 +/- 0.028. A conv spatial-broadcast decoder onto PIXEL targets reads
0.531 +/- 0.033, and that pair is the default here.

⛔ THE PAIR IS THE EFFECT; NEITHER HALF IS. Decoder alone is +0.121 (2.75 se) and target alone
+0.089 (2.02 se) -- neither separates -- while together they are +0.211 (8.4 se), with 3-4x
lower seed spread. So `NETT_AUX_SLOTFG_DECODER` and `NETT_AUX_SLOTFG_TARGET` exist to keep all
FOUR cells reproducible, not because any half-move is a fallback. ⚠ And K stays 4: K=3 x
pixel-SBD reads 0.439 +/- 0.030, 3.6 se BELOW pixel-SBD at K=4, so the two do not compose.

⛔ MEMORY IS THE BINDING CONSTRAINT, AND `NETT_AUX_SLOTFG_DECODE_SCALE` IS THE HONEST FALLBACK.
One slot-term step at patch 16, aux only: the shipped pair peaks at 161 MiB at B=32, the winner
at 1,659 MiB -- 10.3x, linear in B (844 MiB at B=16). Seven brains is ~11.6 GB of aux
activations before PPO's own rollouts, on a 24 GiB card where row 04 already peaks at 15,314
MiB. If it does not fit, the fallback is DECODING AT LOWER RESOLUTION AND UPSAMPLING, never one
of the half-moves: each half loses most of the effect, so a half-move buys memory by deleting
the finding.

⛔ OBJECTIVE CHANGE 2026-09-17 (owner, workspace DECISIONS): `cltt_ref` now excludes each anchor's OWN FRAME from its negatives, so every cltt_ref arm trained before this commit ran a different objective and is NOT comparable to one trained after it.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .knobs import (
    _env_choice, _env_flag_strict, _env_nonneg_float, _env_positive_float, _env_positive_int,
)
from .cltt_views import resolve_channels_per_frame
from .slot_contrast_aux import SlotAttention, slot_contrast_diagnostics
from .token_term import NOT_MEASURED, TokenWindow, TokenWindowTerm, rank_corr, window_turn_scalar


class SlotFGTerm(TokenWindowTerm):
    """Slots over tokens + temporal slot contrast + reconstruction + a foreground indicator."""

    BATCH_ENV = "NETT_AUX_SLOTFG_BATCH"
    DEFAULT_BATCH = 32
    OFFSET_ENV = "NETT_AUX_SLOTFG_OFFSET"
    DEFAULT_OFFSET = 8
    SLOTS_ENV = "NETT_AUX_SLOTFG_SLOTS"
    DIM_ENV = "NETT_AUX_SLOTFG_DIM"
    DEC_HIDDEN_ENV = "NETT_AUX_SLOTFG_DEC_HIDDEN"
    TEMP_ENV = "NETT_AUX_SLOTFG_TEMP"
    W_SS_ENV = "NETT_AUX_SLOTFG_W_SS"
    W_REC_ENV = "NETT_AUX_SLOTFG_W_REC"
    LAMBDA_ENV = "NETT_AUX_SLOTFG_LAMBDA"
    GAMMA_ENV = "NETT_AUX_SLOTFG_GAMMA"
    EGO_ENV = "NETT_AUX_SLOTFG_EGO"
    RAMP_ENV = "NETT_AUX_SLOTFG_RAMP_CALLS"
    DECODER_ENV = "NETT_AUX_SLOTFG_DECODER"
    TARGET_ENV = "NETT_AUX_SLOTFG_TARGET"
    DECODE_SCALE_ENV = "NETT_AUX_SLOTFG_DECODE_SCALE"
    #: The screen's 2x2. `tokmlp`/`ema_tokens` is the shipped pair; the defaults are the winner.
    DECODERS = ("tokmlp", "convsbd")
    TARGETS = ("ema_tokens", "pixels")
    #: 1 = full resolution, 2 = half, 4 = quarter -- pixel targets only. Expressed as the
    #: DIVISOR because that is what the memory scales with, and because "scale=2" reading as
    #: "twice as big" is the kind of ambiguity a screen result gets filed under.
    DECODE_SCALES = ("1", "2", "4")
    #: The SBD broadcast grid is the image divided by this, doubled once per ConvTranspose.
    SBD_START_DIV = 8
    SBD_HIDDEN = 64
    TRANSIT_WEIGHTED = False

    #: Slot-attention iterations: 3 on the first frame, 2 on the second (SlotContrast
    #: `first_step_corrector_args`, SAVi's 1-after-init; the newer reference is followed).
    ITERS_FIRST = 3
    ITERS_NEXT = 2

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(encoder)
        self.slots = _env_positive_int(self.SLOTS_ENV, 4)
        self.slot_dim = _env_positive_int(self.DIM_ENV, 64)
        self.temperature = _env_positive_float(self.TEMP_ENV, 0.1)
        self.w_ss = _env_positive_float(self.W_SS_ENV, 0.5)
        self.w_rec = _env_positive_float(self.W_REC_ENV, 1.0)
        self.lambda_stuff = _env_nonneg_float(self.LAMBDA_ENV, 1.0)
        self.gamma_sep = _env_nonneg_float(self.GAMMA_ENV, 0.1)
        self.use_ego = _env_flag_strict(self.EGO_ENV)
        # ⚠ THE RAMP IS COUNTED IN COMPUTE CALLS, NOT PPO UPDATES, and the difference is 160x:
        # the aux runs once per MINIBATCH (learning_epochs x mini_batches = 10 x 16 at fleet
        # defaults), so a 125-update arm makes ~20,000 calls. 2,000 calls is the "first 10% of
        # training" the paper's warm-up asks for, expressed in the unit this code can count.
        self.ramp_calls = _env_positive_int(self.RAMP_ENV, 2000)
        self._calls = 0
        # ⛔ THE SCREENED FACTORS. Both default to the winning pair; all four cells stay
        # constructible because the screen has to remain reproducible from the shipped code --
        # a result whose configuration can no longer be built is a result nobody can check.
        self.decoder_kind = _env_choice(self.DECODER_ENV, "convsbd", self.DECODERS)
        self.target_kind = _env_choice(self.TARGET_ENV, "pixels", self.TARGETS)
        self.decode_scale = int(_env_choice(self.DECODE_SCALE_ENV, "1", self.DECODE_SCALES))
        # ⛔ AND THE COMBINATION IS VALIDATED, NOT JUST THE VALUES. A token target decodes onto
        # the 5x8 TOKEN grid, where a resolution divisor means nothing -- so a run that set it
        # would train the full-resolution model while its config, its log line and whoever reads
        # them all believed otherwise. An unread knob is a control that is not there.
        if self.decode_scale != 1 and self.target_kind != "pixels":
            raise ValueError(
                f"{self.DECODE_SCALE_ENV}={self.decode_scale} needs {self.TARGET_ENV}=pixels: "
                f"it divides the resolution of a PIXEL reconstruction, and a token target "
                f"decodes onto the {self.grid[0]}x{self.grid[1]} token grid, which this knob "
                f"cannot change. Refusing to accept a knob this configuration does not read.")
        # ⛔ WHICH TERMS USE THE SHARED EMA TEACHER, DECLARED PER INSTANCE. `cltt_ref` itself
        # uses NO teacher (verified: zero references in cltt_ref_aux.py); the composite builds
        # one only because a TERM asks for it, and the askers are cltt_patch, patch_affinity,
        # ego_residual and this term's TOKEN target. With a pixel target and the ego off,
        # nothing in the row reads it -- so this row no longer asks for one, and the composite
        # then holds no deepcopy of the trunk and steps no EMA. ⚠ Row 05 (EGO=1) still needs it
        # for the ego residual's own token target, so the flag is the OR of the two.
        self.needs_teacher = (self.target_kind == "ema_tokens") or self.use_ego
        d, hidden = self.token_dim, _env_positive_int(self.DEC_HIDDEN_ENV, 256)
        device = next(encoder.parameters()).device
        from ...body.observation import image_channels_hw
        _c, self.img_h, self.img_w = image_channels_hw(
            getattr(encoder, "observation_space", None))
        # ⚠ THE PER-FRAME CHANNEL COUNT IS A KNOB ELSEWHERE AND IS READ, NOT ASSUMED: dvs_polarity
        # makes one frame TWO channels, so a decoder hardcoded to 3 would reconstruct two of a
        # frame's channels and call it the image.
        self.cpf = resolve_channels_per_frame()
        #: The resolution the decoder EMITS. The loss is always taken at full resolution (the
        #: prediction is upsampled), so this is a memory knob and not a change of objective.
        self.decode_hw = ((self.img_h // self.decode_scale, self.img_w // self.decode_scale)
                          if self.target_kind == "pixels" else None)

        modules = {
            # Reference `two_layer_mlp` FEAT -> SLOT_DIM with LayerNorm (ytvis2021.yaml:64-69):
            # the ViT's post-norm tokens are D-dimensional and the slots are 64.
            "input_mlp": nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.ReLU(),
                                       nn.Linear(2 * d, self.slot_dim)),
            # ⛔ IMPORTED, NOT COPIED. slot_contrast_aux.SlotAttention is the shipped arm's module
            # and is used unchanged; only how it is INITIALISED and CHAINED changes here.
            "attn": SlotAttention(self.slot_dim, self.slot_dim, self.slots, self.ITERS_NEXT),
            # FixedLearnedInit: ONE learned (1, K, slot_dim), identical for every sample.
            "init": _Holder(nn.Parameter(torch.randn(1, self.slots, self.slot_dim) * 0.02)),
            # Between-frame predictor: 1 pre-norm transformer block over the K slots (SAVi's
            # predictor is slot self-attention only; SlotContrast's is a TransformerEncoder).
            "predictor": nn.TransformerEncoderLayer(
                d_model=self.slot_dim, nhead=4, dim_feedforward=4 * self.slot_dim,
                dropout=0.0, batch_first=True, norm_first=True),
        }
        modules.update(self._decoder_modules(d, hidden))
        self.ego = None
        if self.use_ego:
            from .ego_residual_aux import EgoResidualTerm
            # ⛔ ROW 05 RUNS THE EGO TERM ON THIS TERM'S WINDOW, so its own window knobs would be
            # IGNORED -- and an ignored knob that stays settable is how a config says one thing
            # while the run does another. They are refused here instead, naming the knobs that do
            # work. (The objectness map must be aligned token-for-token with the slot masks it
            # supervises, which is only true on one shared window.)
            ignored = [k for k in (EgoResidualTerm.BATCH_ENV, EgoResidualTerm.OFFSET_ENV)
                       if os.environ.get(k) is not None]
            if ignored:
                raise ValueError(
                    f"{self.EGO_ENV} is set, so the ego residual runs on slot_fg's window and "
                    f"{', '.join(ignored)} would be ignored. Set {self.BATCH_ENV} / "
                    f"{self.OFFSET_ENV} instead; the ego term's other knobs "
                    f"({EgoResidualTerm.IDENTITY_BIAS_ENV}, "
                    f"{EgoResidualTerm.TRANSIT_FRAC_ENV}) are read as usual.")
            self.ego = EgoResidualTerm(encoder)
            # ⛔ The ego head goes INSIDE this term's head, or its routing and predictor reach no
            # optimizer param group: AuxLossPPO optimizes `aux.head.parameters()` and nothing
            # else, and a silently untrained routing would make w pure ego-motion forever.
            modules["ego"] = self.ego.head
        self.head = nn.ModuleDict(modules).to(device)
        # ⚠ SlotAttention's per-sample random-init parameters are DEAD HERE BY DESIGN: this term
        # always passes an explicit init (the learned `init` at t, predictor(slots_t) at t+k), so
        # `mu`/`log_sigma` are never read. Freezing them says so -- otherwise they sit in the
        # optimizer's param group receiving no gradient, which is indistinguishable from a
        # pathway that is meant to train and does not.
        self.head["attn"].mu.requires_grad_(False)
        self.head["attn"].log_sigma.requires_grad_(False)

    # -- plumbing -------------------------------------------------------------------

    def attach_memory(self, memory) -> None:
        super().attach_memory(memory)
        if self.ego is not None:
            self.ego.attach_memory(memory)

    def attach_teacher(self, teacher) -> None:
        super().attach_teacher(teacher)
        if self.ego is not None:
            self.ego.attach_teacher(teacher)

    def _decoder_modules(self, d: int, hidden: int) -> dict:
        """The head modules for THIS cell of the screen's 2x2, and only those.

        ⛔ ONLY THOSE. The screen's fork kept the shipped per-token MLP in the head alongside the
        SBD it was actually using, which is why its reported parameter count (496,725) is 88,721
        higher than this port's for the same cell: those were dead parameters in an optimizer
        param group, receiving no gradient -- indistinguishable from a pathway meant to train
        and silently not training, which this file already refuses for `attn.mu`.

        ⛔ ALPHA RESOLUTION IS HELD WITH THE TARGET, NOT WITH THE DECODER, and that is what makes
        the 2x2 a clean design. A pixel target gets a per-PIXEL alpha in BOTH decoders (the
        token MLP emits ph*pw alphas per token, not one) and a token target gets a per-TOKEN
        alpha in both. Otherwise alpha resolution would be an aligned confound of the target and
        neither column would be an architecture contrast at all.
        """
        s_dim = self.slot_dim
        if self.decoder_kind == "tokmlp":
            # `pos` is the per-token query grid; it belongs to this decoder and is built only
            # with it, for the same reason as above.
            mods = {"pos": _Holder(torch.nn.Parameter(
                torch.zeros(1, 1, self.n_tokens, s_dim)))}
            if self.target_kind == "ema_tokens":
                width = d + 1                                      # the SHIPPED head, unchanged
            else:
                if self.img_h % self.grid[0] or self.img_w % self.grid[1]:
                    raise ValueError(
                        f"a pixel target needs whole patches: {self.img_h}x{self.img_w} does not "
                        f"divide into a {self.grid[0]}x{self.grid[1]} token grid.")
                self.ph, self.pw = self.img_h // self.grid[0], self.img_w // self.grid[1]
                if self.ph % self.decode_scale or self.pw % self.decode_scale:
                    raise ValueError(
                        f"{self.DECODE_SCALE_ENV}={self.decode_scale} does not divide this "
                        f"decoder's {self.ph}x{self.pw} patch.")
                width = (self.ph // self.decode_scale) * (self.pw // self.decode_scale) \
                    * (self.cpf + 1)
            mods["decoder"] = nn.Sequential(nn.Linear(s_dim, hidden), nn.ReLU(),
                                            nn.Linear(hidden, hidden), nn.ReLU(),
                                            nn.Linear(hidden, width))
            return mods
        hid = self.SBD_HIDDEN
        if self.target_kind == "ema_tokens":
            # Spatial broadcast at the TOKEN grid: stride-1 3x3, so the resolution never moves.
            return {"sbd_pos": _Holder(torch.nn.Parameter(
                        torch.zeros(1, s_dim, self.grid[0], self.grid[1]))),
                    "sbd": nn.Sequential(
                        nn.Conv2d(s_dim, hid, 3, 1, 1), nn.ReLU(),
                        nn.Conv2d(hid, hid, 3, 1, 1), nn.ReLU(),
                        nn.Conv2d(hid, d + 1, 3, 1, 1))}
        # Spatial broadcast at the IMAGE: start at H/8 x W/8 and double per ConvTranspose.
        # ⛔ THE SCALE KNOB DROPS STAGES; IT DOES NOT SHRINK THE START GRID. Shrinking the grid
        # would need H/32 to be an integer at scale 4 (it is 2.5 here) and would keep a stage
        # whose whole cost is the resolution we are trying not to pay. Dropping the last stage
        # halves the output and removes the layer that produced it: 3 stages -> full, 2 -> half,
        # 1 -> quarter, and the upsample back to full happens AFTER the alpha mix, on cpf
        # channels instead of K x (cpf+1).
        div = self.SBD_START_DIV
        if self.img_h % div or self.img_w % div:
            raise ValueError(
                f"the pixel spatial-broadcast decoder starts at {self.img_h}/{div} x "
                f"{self.img_w}/{div}, which is not whole. Use {self.TARGET_ENV}=ema_tokens or an "
                f"image size divisible by {div}.")
        stages = 3 - int(math.log2(self.decode_scale))
        layers, ch = [], s_dim
        for _ in range(stages):
            layers += [nn.ConvTranspose2d(ch, hid, 4, 2, 1), nn.ReLU()]
            ch = hid
        layers += [nn.Conv2d(ch, self.cpf + 1, 3, 1, 1)]
        return {"sbd_pos": _Holder(torch.nn.Parameter(
                    torch.zeros(1, s_dim, self.img_h // div, self.img_w // div))),
                "sbd": nn.Sequential(*layers)}

    def _slots(self, tokens: torch.Tensor, slots_init: torch.Tensor, iters: int):
        """Slot attention over projected tokens with a GIVEN init. -> slots (B,K,S), attn (B,K,N).

        ⚠ `iters` is set on the imported module for the duration of the call and restored. The
        class is the shipped arm's and is not modified; the iteration count is a plain int with
        no parameters attached to it, and the reference genuinely uses a different count on the
        first frame (3) than after it (2).
        """
        attn_module = self.head["attn"]
        prev = attn_module.iters
        attn_module.iters = int(iters)
        try:
            return attn_module(self.head["input_mlp"](tokens), slots_init)
        finally:
            attn_module.iters = prev

    def _decode(self, slots: torch.Tensor) -> torch.Tensor:
        """This cell's reconstruction: (B, N, D) for a token target, (B, C, h, w) for pixels.

        ⚠ Alpha is softmaxed over the SLOT axis in every cell -- that is the compositing rule,
        and it is what makes the reconstruction a competition between slots rather than a sum.
        """
        b, k, s = slots.shape
        if self.decoder_kind == "tokmlp":
            grid = slots[:, :, None, :] + self.head["pos"].value      # (B, K, N, S)
            out = self.head["decoder"](grid)
            if self.target_kind == "ema_tokens":                      # (B, K, N, D+1)
                recon, alpha = out[..., :-1], out[..., -1:]
                return (recon * alpha.softmax(dim=1)).sum(dim=1)
            # Per-token MLP whose output is ITS PATCH'S PIXELS plus a per-pixel alpha, so that
            # only the TARGET differs from the shipped cell.
            n_h, n_w = self.grid
            ph, pw = self.ph // self.decode_scale, self.pw // self.decode_scale
            out = out.view(b, k, n_h, n_w, ph, pw, self.cpf + 1)
            out = out.permute(0, 1, 6, 2, 4, 3, 5).reshape(b, k, self.cpf + 1, n_h * ph, n_w * pw)
        else:
            g = slots.reshape(b * k, s, 1, 1) + self.head["sbd_pos"].value
            out = self.head["sbd"](g)
            if self.target_kind == "ema_tokens":
                out = out.view(b, k, self.token_dim + 1, self.n_tokens)
                recon, alpha = out[:, :, :-1], out[:, :, -1:]
                return (recon * alpha.softmax(dim=1)).sum(1).transpose(1, 2)   # (B, N, D)
            out = out.view(b, k, self.cpf + 1, *self.decode_hw)
        recon, alpha = out[:, :, :-1], out[:, :, -1:]
        return (recon * alpha.softmax(dim=1)).sum(1)                  # (B, C, h, w)

    def _reconstruction(self, window: TokenWindow, slots_t: torch.Tensor) -> torch.Tensor:
        """The reconstruction loss for this cell. -> scalar.

        ⛔ THE PIXEL TARGET IS THE TENSOR THE ENCODER SAW, NOT A RE-PREPARED COPY.
        `window.prepared_t` is the exact tensor `spatial_tokens` was called on in `_core`; the
        target is a SLICE of it, so no second `_prepare_image` can drift in normalisation,
        resize or frame selection. `[:, -cpf:]` is the CURRENT frame: preparation orders
        channels [oldest, ..., current] (cltt_views.current_frame_stack takes the same slice and
        documents the order), which at the campaign eye's 6 RGB channels is `[:, 3:6]` -- what
        the screen measured. Reading it from the end rather than the middle is what keeps it
        right under dvs_polarity, where one frame is two channels.

        ⚠ AND IT IS IN PREPARED UNITS. The target is normalised image data, so `rec` here is not
        comparable to a raw-pixel MSE, nor to the token-target cell's `rec` -- different targets,
        different scales. Compare cells by the screen's held-out ARI, never by this number.

        ⛔ THE LOSS IS ALWAYS TAKEN AT FULL RESOLUTION. At DECODE_SCALE > 1 the PREDICTION is
        upsampled to the target rather than the target downsampled to the prediction, so the
        number means the same thing at every scale and the scales are comparable. Downsampling
        the target instead would change the objective and quietly rescale the loss, and a screen
        comparing scales would then be comparing two things at once.
        """
        if self.target_kind == "ema_tokens":
            u_t, _ = self.teacher.tokens(window.prepared_t)
            return F.mse_loss(self._decode(slots_t), u_t)
        target = window.prepared_t[:, -self.cpf:]
        pred = self._decode(slots_t)
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode="bilinear",
                                 align_corners=False)
        return F.mse_loss(pred, target)

    # -- the loss -------------------------------------------------------------------

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Row 04 draws uniformly; row 05 draws the ego term's 50/50 transit mixture.

        ⚠ The mixture belongs to the EGO half of row 05: its routing only learns ego compensation
        on transit windows, while the objectness it feeds the slots is cleanest on parked ones.
        Row 04 has no action-conditioned part, so it keeps the plain uniform draw.

        ⚠ The `|Δt| < k` false-negative mask in the slot contrast indexes the CONCATENATED batch,
        so it treats the seam between the two half-slabs as near-in-time (a few valid negatives
        are over-masked, which is harmless) and it cannot see an overlap ACROSS the halves when
        they happen to land close together in the same env (a few false negatives survive). Both
        are bounded by the slab count, which is 2.
        """
        if self.ego is not None:
            window = self.draw_mixed(encoder, transit_frac=self.ego.transit_frac)
        else:
            window = self.draw(encoder)
        loss, scalars = self._core(encoder, window)
        self.last_window_turn = float(window.mean_turn)
        self.last_scalars = {"B": float(window.prepared_t.shape[0]), "k": float(self.offset),
                             "window_turn": window_turn_scalar(window.mean_turn),
                             **{k: float(v) for k, v in scalars.items()}}
        return loss

    def _core(self, encoder: nn.Module, window: TokenWindow):
        from .token_features import spatial_tokens

        self._calls += 1
        z_t = spatial_tokens(encoder, window.prepared_t)[0]
        z_tk = spatial_tokens(encoder, window.prepared_tk)[0]
        b = z_t.shape[0]
        init = self.head["init"].value.expand(b, -1, -1)
        slots_t, attn_t = self._slots(z_t, init, self.ITERS_FIRST)
        # ⭐ THE CORRESPONDENCE: the t+k init is the PREDICTION of the t slots, not a new draw.
        slots_tk, attn_tk = self._slots(z_tk, self.head["predictor"](slots_t), self.ITERS_NEXT)

        ss = self._slot_contrast(slots_t, slots_tk)
        rec = self._reconstruction(window, slots_t)
        marginal = attn_t.mean(dim=-1)                                # (B, K) slot mass
        sep = -(marginal.clamp_min(1e-12).log() * marginal).sum(-1).mean()   # MAXIMISE
        total = self.w_ss * ss + self.w_rec * rec - self.gamma_sep * sep

        # ⛔ THE CELL GOES OUT ON EVERY CALL. This row has FOUR objectives now, and `rec` is not
        # comparable across them -- a token target and a pixel target are different tensors in
        # different units -- so a loss series without its cell is unreadable. ⚠ An arm from
        # before this commit emits NONE of these tags, and absent must NOT be read as
        # decoder=0: 0 is `tokmlp`, which is also the shipped cell, so the old arm and the
        # deliberately-shipped-cell arm are distinguishable only by the tag's absence.
        # `head_params` rides along because memory is the reason the cell is a knob at all.
        scalars = {"ss": float(ss.detach()), "rec": float(rec.detach()),
                   "sep_entropy": float(sep.detach()),
                   "sep_entropy_frac": float(sep.detach()) / math.log(self.slots),
                   "slots": float(self.slots), "ramp": 0.0,
                   "decoder": float(self.DECODERS.index(self.decoder_kind)),
                   "target": float(self.TARGETS.index(self.target_kind)),
                   "decode_scale": float(self.decode_scale),
                   "head_params": float(sum(p.numel() for p in self.head.parameters())),
                   "fg_bce": NOT_MEASURED, "stuff": NOT_MEASURED,
                   "ego_loss": NOT_MEASURED, "fg_target_engaged": NOT_MEASURED}
        m_fg = attn_t[:, 0, :]                                        # slot 0's per-token mass
        if self.ego is not None:
            ego_loss, w, ego_scalars = self.ego.loss_and_objectness(encoder, window)
            ramp = min(1.0, self._calls / float(self.ramp_calls))
            # ⚠ WRITTEN OUT RATHER THAN F.binary_cross_entropy, WHICH IS AUTOCAST-UNSAFE. The aux
            # runs inside `torch.autocast(enabled=cfg.mixed_precision)` (ppo_aux.py:384), and
            # under CUDA autocast BCE raises unconditionally ("unsafe to autocast, use
            # binary_cross_entropy_with_logits") whatever the input dtype. `mixed_precision`
            # defaults False and nothing in this repo sets it, so the crash is latent -- and a
            # latent crash on a knob someone may flip is worth two lines. The value is identical
            # (tested); the mask is already clamped, so the logs are finite.
            m = m_fg.clamp(1e-6, 1 - 1e-6)
            fg_bce = -(w * m.log() + (1 - w) * (1 - m).log()).mean()
            stuff = self._stuff(z_t, w)
            total = total + ego_loss + ramp * (fg_bce - self.lambda_stuff * stuff)
            scalars.update({"ego_loss": float(ego_loss.detach()), "ramp": float(ramp),
                            "fg_bce": float(fg_bce.detach()), "stuff": float(stuff.detach()),
                            **{f"ego_{k}": float(v) for k, v in ego_scalars.items()}})
            # ⛔ THE FOREGROUND TARGET CARRIES ITS OWN VALIDITY, AND A CORRELATION AGAINST AN
            # INVALID TARGET IS THE PLAUSIBLE-LOOKING ZERO THE SENTINEL RULE FORBIDS. When the
            # ego term's paired null cannot distinguish `w` from noise, "slot 0 does not match
            # the objectness" and "there was no objectness to match" produce the SAME small
            # number, and only one of them is a finding. Uninterpretable is reported as
            # uninterpretable; the training is NOT gated, so the row still runs and the flag
            # says how to read it.
            engaged = ego_scalars.get("objectness_engaged", NOT_MEASURED)
            scalars["fg_target_engaged"] = float(engaged)
            scalars["fg_objectness_corr"] = (rank_corr(m_fg.mean(dim=0), w.mean(dim=0))
                                             if engaged == 1.0 else NOT_MEASURED)
        with torch.no_grad():
            scalars.update(self._diagnostics(encoder, window, z_t, slots_t, slots_tk,
                                             attn_t, m_fg, init))
        return total, scalars

    def _slot_contrast(self, slots_t: torch.Tensor, slots_tk: torch.Tensor) -> torch.Tensor:
        """SlotContrast's slot-slot temporal contrast, with the near-in-time negatives MASKED.

        ⛔ THE MASK IS NOT IN THE REFERENCE AND IS REQUIRED HERE. The reference's negatives are
        B independent VIDEOS; ours are B CONSECUTIVE STEPS of one env stream, so a negative
        sampled |Δt| < k away is a near-duplicate of the positive -- the loss would be asking
        "is slot j at t closer to slot j at t+8 than to slot j at t+9", which is noise with a
        confident gradient. Samples closer than the offset are removed from the denominator.
        """
        b, k, s = slots_t.shape
        s1 = F.normalize(slots_t, dim=-1).reshape(b * k, s)
        s2 = F.normalize(slots_tk, dim=-1).reshape(b * k, s)
        sim = (s1 @ s2.T) / self.temperature
        idx = torch.arange(b, device=sim.device)
        near = (idx[:, None] - idx[None, :]).abs() < self.offset
        near.fill_diagonal_(False)                        # the positive's own sample stays
        block = near.repeat_interleave(k, 0).repeat_interleave(k, 1)
        sim = sim.masked_fill(block, float("-inf"))
        return F.cross_entropy(sim, torch.arange(b * k, device=sim.device))

    def _stuff(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Tian Eq 2 adapted: bg prototypes agree ACROSS samples and differ from fg prototypes.

        Prototypes are objectness-weighted token means. The comparison side is DETACHED, which is
        this port's stand-in for the paper's EMA teacher on the prototype path, and the diagonal
        (a sample against itself) is excluded as Eq 2 does (j != i).
        """
        wt = w.unsqueeze(-1)
        p_f = (wt * z).sum(1) / wt.sum(1).clamp_min(1e-6)
        p_b = ((1 - wt) * z).sum(1) / (1 - wt).sum(1).clamp_min(1e-6)
        b = z.shape[0]
        if b < 2:
            return torch.zeros((), device=z.device)
        f = F.normalize(p_f, dim=-1)
        bg = F.normalize(p_b, dim=-1)
        off = ~torch.eye(b, dtype=torch.bool, device=z.device)
        bb = (bg @ bg.detach().T)[off].mean()
        bf = (bg @ f.detach().T)[off].mean()
        return bb - bf

    # -- diagnostics ----------------------------------------------------------------

    @torch.no_grad()
    def _diagnostics(self, encoder, window, z_t, slots_t, slots_tk, attn_t, m_fg, init) -> dict:
        from .token_features import spatial_tokens
        n_h, n_w = self.grid
        col = torch.arange(self.n_tokens, device=m_fg.device) % n_w
        centre = ((col >= n_w // 4) & (col < n_w - n_w // 4)).float()
        # ⛔ The mask-variance null is PAIRED and, with a FIXED learned init, structurally exact:
        # identical inputs share the init, so if attention does not depend on the input the two
        # variances are equal and the excess is EXACTLY 0. (slot_contrast's random per-sample init
        # made its null a noise floor; this one has no init noise to cancel, which is a property
        # of the FixedLearnedInit change, not of the statistic.)
        flat = window.prepared_t[:1].expand_as(window.prepared_t).contiguous()
        z_flat = spatial_tokens(encoder, flat)[0]
        _, attn_null = self._slots(z_flat, init, self.ITERS_FIRST)
        diag = slot_contrast_diagnostics(slots_t.reshape(-1, self.slot_dim),
                                         slots_tk.reshape(-1, self.slot_dim),
                                         self.temperature, self.slots)
        fg_mass = float(m_fg.mean())
        return {
            **{k: float(v) for k, v in diag.items()},
            "mask_variance": float(attn_t.var(dim=0).mean()),
            "mask_variance_null": float(attn_null.var(dim=0).mean()),
            "mask_variance_excess": float(attn_t.var(dim=0).mean() - attn_null.var(dim=0).mean()),
            # ⚠ `fg_mass` must sit strictly inside (0, 1) -- about 17.6/40 = 0.44 if the object is
            # bound (C1). At 0 or 1 one slot has taken everything, which is the collapse L_sep
            # exists to prevent. WITHOUT the ego supervision, WHICH slot is "fg" is arbitrary, so
            # read this as "slot 0's share", not as "the object's share".
            "fg_mass": fg_mass,
            "fg_mass_max_slot": float(attn_t.mean(dim=-1).max(dim=-1).values.mean()),
            # Is slot 0 on the CENTRE, where the object is? With a null that destroys only the
            # spatial arrangement, so a slot covering everything scores at the null.
            "fg_centre_corr": rank_corr(m_fg.mean(dim=0), centre),
            "fg_centre_corr_null": rank_corr(
                m_fg.mean(dim=0)[torch.randperm(self.n_tokens, device=m_fg.device)], centre),
            "used_ego": 1.0 if self.ego is not None else 0.0,
        }


class _Holder(nn.Module):
    """A bare Parameter inside `head`. ⛔ `head` is what AuxLossPPO optimizes, and an nn.ModuleDict
    holds MODULES, not Parameters -- a learned init or positional grid assigned beside it would be
    trained by nothing and would look exactly like a working one."""

    def __init__(self, value: nn.Parameter) -> None:
        super().__init__()
        self.value = value
