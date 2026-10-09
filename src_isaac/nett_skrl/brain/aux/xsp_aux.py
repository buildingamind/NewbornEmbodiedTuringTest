"""XSP predictive head: x^_{t+1} = g(x_t, v_t (.) d_t), trained by ||x^_{t+1} - x_{t+1}||^2.

The `xsp` aux (label "XSP", encoder brain/encoders/xsp.py). The paper's Methods (owner, verbatim):
"Their outputs are combined multiplicatively and passed to the predictor alongside the current
frame, x^_{t+1} = g(x_t, v_t (.) d_t). All components are trained jointly to minimize the
next-frame prediction error, e = ||x^_{t+1} - x_{t+1}||^2." -- with v_t = softmax(phi_v(z_t)) a
TWO-WAY softmax per location and d_t = phi_d(z_{t-1}, z_t).

```
(obs_t, a_t, obs_{t+1})  <- action_windows.draw_action_window(memory, k=1)     (as nextframe_aux)
z_{t-T+1..t}, m_t, d_t   <- encoder.encode_streams(obs_t)    phi_e siamese, phi_v, phi_d; grads ON
                            m_t = sigmoid(phi_v(z_t)) (B,1,h,w); d_t from ALL T frames
c      = [m_t * d_t, (1-m_t) * d_t]   NETT_XSP_COMBINE=outer (DEFAULT) = v_t (.) d_t   (B, 2K, h, w)
       | m_t * d_t                    NETT_XSP_COMBINE=fg (knob)                       (B,  K, h, w)
h      = ReLU([FiLM_{a_t}] Conv1x1(c))          NETT_XSP_G_MAP=0 (DEFAULT, paper: g sees no map)
       | ReLU([FiLM_{a_t}] Conv1x1(cat(m_t, c)))  NETT_XSP_G_MAP=1 (knob: g also reads m_t)
h      = (nearest 2x -> Conv3x3 -> ReLU) x log2(8/ds)     map grid (H/8, W/8) -> target (H/ds, W/ds)
x^     = cur + Conv3x3(ReLU(Conv3x3(cat(h, cur))))        cur = avgpool_ds(x_t)   RESIDUAL
target = avgpool_ds(NEWEST frame of obs_{t+1})
L      = mean (x^ - target)^2
```

⛔ THE TARGET IS THE NEWEST FRAME OF obs[t+1], NOT obs[t+1] WHOLE -- the leak documented in
nextframe_aux.py: obs[t+1] = [x_{t-T+2}, .., x_t, x_{t+1}] (T-major, FrameStack's deque), and
every frame but its newest is ALREADY in obs[t], which g's streams receive. The same `_frames`
slicer (last cpf channels) builds the target and the copy baseline, for any T.

DEFAULT COMBINE = "outer" (coordinator ruling 2026-10-09, from the Methods text) IS THE PAPER'S
v_t (.) d_t. With a two-way softmax v_t = [v_fg, v_bg] per location and d_t of K channels, the
per-location product of the two streams is [v_fg * d_t, v_bg * d_t] (2K channels). The encoder
parameterises v_t by ONE logit l per location, m_t = sigmoid(l), and sigmoid(l) =
softmax([l, 0])_0: [m_t, 1 - m_t] IS a two-way softmax with its redundant second logit fixed at 0
(the softmax is invariant to adding a constant to both logits, so nothing is lost). Hence "outer"
= [m_t * d_t, (1 - m_t) * d_t] = v_t (.) d_t exactly (tests/test_xsp.py proves it numerically).
Consequence: the loss is symmetric under m <-> 1 - m (swap the two halves of g's first conv), so
NOTHING decides which side of the sigmoid is "figure"; the measured split is in tests/test_xsp.py
section 4, and policy_input="both" is the readout that does not depend on it. "fg" (m_t * d_t
only: the ground half dropped) is a knob value that breaks that symmetry.

NETT_XSP_G_MAP ("0" default, paper-faithful): g's input is ONLY (x_t, v_t (.) d_t). "1" also
concatenates m_t itself into g's first conv -- the pre-Methods behaviour, kept as a knob.

DESIGN CHOICES (stated, not knobbed):
* RESIDUAL OUTPUT. g predicts the CHANGE on top of the pooled current frame. That is still
  g(x_t, v_t (.) d_t) -- it is one parameterisation of it -- and it makes "next = current" the
  zero point, so `skill` starts near 0 instead of strongly negative and the streams are trained
  on the part of the image that changes, which is the part motion can explain.
* By default g gets NEITHER m_t NOR d_t directly, only their product c (Methods). Under "outer"
  d_t is still recoverable as the sum of the two groups; under "fg" it is visible only where m_t
  is on. NETT_XSP_G_MAP=1 adds m_t as a direct input.
* x_t enters g at the TARGET grid (avgpool_ds), concatenated after the up-sampling stages, so
  the frame's appearance is available at full target resolution and the streams' maps carry
  only what the frame does not: which locations are figure, and how things move.
* Action: by default g takes NO action (paper-faithful). NETT_XSP_ACTION=film adds nextframe's
  zero-initialised FiLM on the first two action components (turn, move).

WHAT NOTHING HERE FORCES. The loss does not force m to be a figure-ground split. Under "outer"
(the default) m and 1 - m are interchangeable; under "fg" only m gates motion, which breaks that symmetry but
does not stop m from saturating at 0 everywhere (g sees no motion) or at 1 everywhere (c = d,
the gate passes everything and selects nothing), nor from staying UNDECIDED near .5 while g
compensates: on the synthetic square one seed predicted as well as the rest (skill .96) with m
in .41-.59 everywhere and no separation (tests/test_xsp.py section 4), so skill alone does not
certify the map. `fg_frac`, `fg_entropy` and `fg_frac_std` are published every call so a
collapsed map (fg_frac ~0 or ~1, entropy ~0, std ~0) and an undecided one (fg_frac ~.5,
entropy ~1) are visible in the run's own logs.

CHECKPOINTING: as nextframe -- the decoder `head` is in PPO's optimizer but NOT in
`checkpoint_modules`; phi_e/phi_v/phi_d are inside the encoder and so ARE in the policy checkpoint.
A resume therefore restarts g from scratch.

TELEMETRY (last_scalars, per call): B, mse, copy_mse, skill (= 1 - mse/copy_mse, NOT_MEASURED when
copy_mse == 0), mse_parked, mse_transit, copy_parked, copy_transit (median split on |a_t turn|),
window_turn, action_dim, fg_frac (mean m_t), fg_frac_std (std over the batch of each sample's
mean m_t), fg_entropy (mean per-location binary entropy of m_t, in bits: 1 = undecided at .5,
0 = hard), d_absmean (mean |d_t|), film_gain (mean |gamma - 1|) under NETT_XSP_ACTION=film only,
and the ORIENTATION group fg_motion_sep, fg_motion_sep_parked, fg_motion_n_parked (below).

ORIENTATION (label-free, per brain). Under combine=outer the loss is symmetric under m <-> 1 - m,
so which side of m lands on the object is a per-seed coin; the parsing read needs it per brain,
and the env has no masks. Pixel motion stands in: e = mean over channels of |x_{t+1} - x_t| on
the NEWEST frame of each stack (the frames the target and copy baseline use), average-pooled to
m's grid. Per sample, "moving" = the top 10% of cells by e, "static" = the bottom 50%; a sample
QUALIFIES only if mean e(moving) > 2 * mean e(static) + 1e-6 (localized motion). Then
  fg_motion_sep        = mean over qualifying samples of [mean m(moving) - mean m(static)]
  fg_motion_sep_parked = the same over qualifying PARKED samples (the parked_transit split)
  fg_motion_n_parked   = the number of qualifying parked samples
NOT_MEASURED when no sample qualifies. > 0: the moving thing is on the HIGH side of m.
⚠ AN APPROXIMATION, NOT GROUND TRUTH. It assumes a static background: during parked steps the
only pixel change is the screen video, so "moving" is the stimulus; during transit the whole
view moves and the top-10% cells are wherever the parallax is largest. It is a DIAGNOSTIC,
computed under no_grad from a detached m -- never a training signal (the loss and every gradient
are bit-identical with and without it; tests/test_xsp.py proves it).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_windows import draw_action_window
from .cltt_views import resolve_channels_per_frame
from .knobs import NOT_MEASURED, _env_choice, _env_positive_int, _env_unit_interval
from .token_term import MASK_UNSET, parked_transit, stratum_mean, window_turn_scalar

#: Motor components FiLM conditions on under NETT_XSP_ACTION=film: (turn, move), as nextframe.
ACTION_COMPONENTS = 2
#: Index 0 is the default. "outer" = the Methods' v_t (.) d_t (see the module docstring).
COMBINES = ("outer", "fg")
G_MAPS = ("0", "1")
ACTIONS = ("none", "film")
#: The XSP trunk's total stride; the decoder's input grid is (H/8, W/8).
TRUNK_STRIDE = 8
#: Orientation telemetry: top fraction of cells = "moving", bottom fraction = "static", and the
#: localisation test mean e(moving) > MOTION_RATIO * mean e(static) + MOTION_EPS.
MOTION_TOP, MOTION_BOTTOM, MOTION_RATIO, MOTION_EPS = 0.10, 0.50, 2.0, 1e-6


@torch.no_grad()
def motion_orientation(m: torch.Tensor, x_t: torch.Tensor, x_next: torch.Tensor,
                       parked: torch.Tensor) -> dict[str, float]:
    """The ORIENTATION scalar group (module docstring). m: (B, 1, h, w) figure map; x_t, x_next:
    (B, cpf, H, W) newest frames of obs_t / obs_{t+1} at full resolution; parked: (B,) bool."""
    m = m.detach().float()[:, 0]                                         # (B, h, w)
    b, h, w = m.shape
    e = (x_next.float() - x_t.float()).abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
    kh, kw = e.shape[-2] // h, e.shape[-1] // w
    if (kh * h, kw * w) != tuple(e.shape[-2:]):
        raise RuntimeError(f"motion grid {tuple(e.shape[-2:])} is not a multiple of m's {(h, w)}")
    e = F.avg_pool2d(e, (kh, kw))[:, 0].flatten(1)                     # (B, h*w)
    n = h * w
    k_move, k_static = max(1, math.ceil(MOTION_TOP * n)), max(1, int(MOTION_BOTTOM * n))
    order = torch.sort(e, dim=1, descending=True, stable=True).indices
    move, static = order[:, :k_move], order[:, n - k_static:]
    mf = m.flatten(1)
    e_move, e_static = e.gather(1, move).mean(1), e.gather(1, static).mean(1)
    sep = mf.gather(1, move).mean(1) - mf.gather(1, static).mean(1)    # (B,)
    ok = e_move > MOTION_RATIO * e_static + MOTION_EPS
    ok_parked = ok & parked.to(ok.device)
    return {
        "fg_motion_sep": float(sep[ok].mean()) if bool(ok.any()) else NOT_MEASURED,
        "fg_motion_sep_parked": float(sep[ok_parked].mean()) if bool(ok_parked.any()) else NOT_MEASURED,
        "fg_motion_n_parked": float(ok_parked.sum()),
    }


def combine_streams(m: torch.Tensor, d: torch.Tensor, mode: str) -> torch.Tensor:
    """c from the figure map m (B,1,h,w) and the motion field d (B,K,h,w): "fg" -> m*d (B,K,h,w);
    "outer" -> cat(m*d, (1-m)*d) (B,2K,h,w), the figure group then the ground group."""
    if mode == "outer":
        return torch.cat([m * d, (1.0 - m) * d], dim=1)
    if mode == "fg":
        return m * d
    raise ValueError(f"unknown XSP combine {mode!r}")


class XSPDecoder(nn.Module):
    """g: (cur, c[, m][, a]) -> predicted next frame on the target grid (residual on cur).

    ``g_map`` False (the default, Methods): m is NOT read -- the first conv sees c alone.
    """

    def __init__(self, c_ch: int, cpf: int, n_up: int, hidden: int, film: bool,
                 g_map: bool = False) -> None:
        super().__init__()
        self.g_map = bool(g_map)
        self.inp = nn.Conv2d(c_ch + (1 if self.g_map else 0), hidden, kernel_size=1)
        self.film = nn.Linear(ACTION_COMPONENTS, 2 * hidden) if film else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        # ⚠ NEAREST, not bilinear: bilinear's CUDA backward has no deterministic kernel (nextframe_aux).
        self.up = nn.ModuleList(nn.Conv2d(hidden, hidden, 3, padding=1) for _ in range(n_up))
        self.fuse = nn.Conv2d(hidden + cpf, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, cpf, 3, padding=1)
        self.last_film_gain = 0.0

    def forward(self, cur: torch.Tensor, m: torch.Tensor, c: torch.Tensor,
                action: torch.Tensor | None = None) -> torch.Tensor:
        h = self.inp(torch.cat([m, c], dim=1) if self.g_map else c)
        if self.film is not None:
            gamma, beta = self.film(action).chunk(2, dim=-1)
            self.last_film_gain = float(gamma.detach().abs().mean())
            h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = F.relu(h)
        for conv in self.up:
            h = F.relu(conv(F.interpolate(h, scale_factor=2.0, mode="nearest")))
        if tuple(h.shape[-2:]) != tuple(cur.shape[-2:]):
            raise RuntimeError(f"XSP decoder grid {tuple(h.shape[-2:])} != target {tuple(cur.shape[-2:])}")
        h = F.relu(self.fuse(torch.cat([h, cur], dim=1)))
        return cur + self.out(h)


class XSPTerm(nn.Module):
    """The `xsp` objective. Owns g in `head`; draws its own windows from memory (as NextFrameTerm)."""

    needs_memory = True
    needs_teacher = False

    BATCH_ENV = "NETT_XSP_BATCH"
    TRANSIT_FRAC_ENV = "NETT_XSP_TRANSIT_FRAC"
    COMBINE_ENV = "NETT_XSP_COMBINE"
    ACTION_ENV = "NETT_XSP_ACTION"
    DOWNSAMPLE_ENV = "NETT_XSP_DOWNSAMPLE"
    HIDDEN_ENV = "NETT_XSP_HIDDEN"
    G_MAP_ENV = "NETT_XSP_G_MAP"

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        if not hasattr(encoder, "encode_streams"):
            raise TypeError(
                f"the xsp aux needs the XSP encoder's streams (encode_streams); "
                f"{type(encoder).__name__} has none. Refusing to train a decoder on a stand-in.")
        self.batch = _env_positive_int(self.BATCH_ENV, 64)
        if self.batch < 2:
            raise ValueError(f"{self.BATCH_ENV}={self.batch}: the parked/transit split needs >= 2.")
        self.transit_frac = _env_unit_interval(self.TRANSIT_FRAC_ENV, 0.5)
        slabs = (int(round(self.transit_frac * self.batch)), self.batch - int(round(self.transit_frac * self.batch)))
        if any(0 < n < 2 for n in slabs):
            raise ValueError(
                f"{self.BATCH_ENV}={self.batch} with {self.TRANSIT_FRAC_ENV}={self.transit_frac} splits into "
                f"slabs {slabs}; draw_action_window needs >= 2 per non-empty slab. Raise the batch.")
        self.combine = _env_choice(self.COMBINE_ENV, "outer", COMBINES)
        self.g_map = _env_choice(self.G_MAP_ENV, "0", G_MAPS) == "1"
        self.action = _env_choice(self.ACTION_ENV, "none", ACTIONS)
        self.downsample = _env_positive_int(self.DOWNSAMPLE_ENV, 4)
        self.hidden = _env_positive_int(self.HIDDEN_ENV, 64)

        from ...body.observation import image_channels_hw
        c, h, w = image_channels_hw(getattr(encoder, "observation_space", None))
        if TRUNK_STRIDE % self.downsample:
            raise ValueError(
                f"{self.DOWNSAMPLE_ENV}={self.downsample} must divide the trunk stride {TRUNK_STRIDE} "
                f"(1, 2, 4 or 8): the decoder reaches the target grid by integer 2x stages from "
                f"the (H/8, W/8) map.")
        if h % self.downsample or w % self.downsample:
            raise ValueError(f"{self.DOWNSAMPLE_ENV}={self.downsample} does not divide the {h}x{w} eye.")
        self.cpf = resolve_channels_per_frame()
        if self.cpf != encoder.cpf:
            raise ValueError(f"aux channels-per-frame {self.cpf} != encoder's {encoder.cpf}.")
        self.target_hw = (h // self.downsample, w // self.downsample)
        n_up = int(round(math.log2(TRUNK_STRIDE // self.downsample)))
        k = int(encoder.dorsal_dim)
        c_ch = 2 * k if self.combine == "outer" else k
        device = next(encoder.parameters()).device
        self.head = XSPDecoder(c_ch, self.cpf, n_up, self.hidden, self.action == "film",
                               g_map=self.g_map).to(device)

        self._memory = None
        self.last_scalars: dict = {}
        self.last_window_turn: float = MASK_UNSET
        try:
            from skrl import logger
            logger.warning(
                "[NETT xsp] combine=%s g_map=%d action=%s downsample=%d (target %dx%d) hidden=%d batch=%d "
                "transit_frac=%s policy_input=%s frames=%s | encoder params %d, decoder (aux head) params %d",
                self.combine, int(self.g_map), self.action, self.downsample, self.target_hw[0], self.target_hw[1],
                self.hidden, self.batch, self.transit_frac, getattr(encoder, "policy_input", "?"),
                getattr(encoder, "n_frames", "?"),
                sum(p.numel() for p in encoder.parameters()),
                sum(p.numel() for p in self.head.parameters()))
        except ImportError:                                  # pragma: no cover - skrl is a dependency
            pass

    def attach_memory(self, memory) -> None:
        self._memory = memory

    def _frames(self, prepared: torch.Tensor) -> torch.Tensor:
        """NEWEST frame of a prepared T-major stack, pooled to the target grid (no grad)."""
        newest = prepared[:, -self.cpf:]
        return F.avg_pool2d(newest, self.downsample) if self.downsample > 1 else newest

    def _target(self, prepared_tk: torch.Tensor) -> torch.Tensor:
        """x_{t+1}: the NEWEST frame of obs[t+1] -- never an older frame, each of which is in obs[t]."""
        return self._frames(prepared_tk)

    def _current(self, prepared_t: torch.Tensor) -> torch.Tensor:
        """x_t: the newest frame of obs[t]; g's frame input and the copy baseline."""
        return self._frames(prepared_t)

    def _draw(self, encoder: nn.Module):
        if self._memory is None:
            raise RuntimeError("XSPTerm draws its own windows; call attach_memory(memory) first.")
        device = next(encoder.parameters()).device
        n_transit = int(round(self.transit_frac * self.batch))
        parts, turn_state = [], None
        for size, weighted in ((n_transit, True), (self.batch - n_transit, False)):
            if size <= 0:
                continue
            win = draw_action_window(self._memory, 1, size, transit_weighted=weighted, device=device)
            parts.append(win)
            if weighted:
                turn_state = win.mean_turn
        if turn_state is None:
            turn_state = parts[0].mean_turn
        obs_t = torch.cat([p.obs_t for p in parts])
        obs_tk = torch.cat([p.obs_tk for p in parts])
        actions = torch.cat([p.actions[:, 0] for p in parts])          # a_t: (B, A)
        return obs_t, obs_tk, actions, turn_state

    def predict(self, encoder: nn.Module, prepared_t: torch.Tensor,
                action: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """x^_{t+1} on the target grid from a PREPARED obs_t stack, and the streams it used."""
        s = encoder.encode_streams(prepared_t)
        with torch.no_grad():
            cur = self._current(prepared_t)
        c = combine_streams(s["m"], s["d"], self.combine)
        a = action if self.action == "film" else None
        return self.head(cur, s["m"], c, a), s

    def score(self, encoder: nn.Module, prepared_t: torch.Tensor, prepared_tk: torch.Tensor,
              actions: torch.Tensor, turn_state: float = MASK_UNSET) -> torch.Tensor:
        """The loss on one batch of prepared (obs_t, obs_{t+1}) stacks; publishes last_scalars."""
        if self.action == "film" and actions.shape[-1] < ACTION_COMPONENTS:
            raise ValueError(
                f"NETT_XSP_ACTION=film conditions on {ACTION_COMPONENTS} motor components; the stored "
                f"actions have {actions.shape[-1]}.")
        a = actions[:, :ACTION_COMPONENTS].float()
        with torch.no_grad():
            target = self._target(prepared_tk)
            current = self._current(prepared_t)
        pred, s = self.predict(encoder, prepared_t, a)
        per_sample = (pred - target).pow(2).mean(dim=(1, 2, 3))
        loss = per_sample.mean()
        with torch.no_grad():
            copy = (current - target).pow(2).mean(dim=(1, 2, 3))
            parked, transit = parked_transit(a[:, 0].abs())
            copy_mse = float(copy.mean())
            mse = float(loss.detach())
            fg = s["m"].detach().float()[:, 0]                       # (B, h, w)
            p, q = fg.clamp_min(1e-12), (1.0 - fg).clamp_min(1e-12)
            ent = -(fg * p.log2() + (1.0 - fg) * q.log2())            # binary entropy, bits in [0, 1]
            scalars = {
                "B": float(a.shape[0]),
                "mse": mse,
                "copy_mse": copy_mse,
                "skill": 1.0 - mse / copy_mse if copy_mse > 0 else NOT_MEASURED,
                "mse_parked": stratum_mean(per_sample.detach(), parked),
                "mse_transit": stratum_mean(per_sample.detach(), transit),
                "copy_parked": stratum_mean(copy, parked),
                "copy_transit": stratum_mean(copy, transit),
                "window_turn": window_turn_scalar(turn_state),
                "action_dim": float(actions.shape[-1]),
                "fg_frac": float(fg.mean()),
                "fg_frac_std": float(fg.mean(dim=(1, 2)).std()) if fg.shape[0] > 1 else NOT_MEASURED,
                "fg_entropy": float(ent.mean()),
                "d_absmean": float(s["d"].detach().abs().mean()),
            }
            if self.action == "film":
                scalars["film_gain"] = float(self.head.last_film_gain)
            scalars.update(motion_orientation(s["m"], prepared_t[:, -self.cpf:],
                                              prepared_tk[:, -self.cpf:], parked))
            self.last_scalars = scalars
        self.last_window_turn = float(turn_state)
        return loss

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Ignore the PPO minibatch; draw (obs_t, a_t, obs_{t+1}) windows and score the prediction."""
        obs_t, obs_tk, actions, turn_state = self._draw(encoder)
        with torch.no_grad():
            prepared_t = encoder._prepare_image(obs_t)
            prepared_tk = encoder._prepare_image(obs_tk)
        return self.score(encoder, prepared_t, prepared_tk, actions, turn_state)
