"""XSP predictive head: x^_{t+1} = g(x_t, v_t, d_t), trained by ||x^_{t+1} - x_{t+1}||^2.

The `xsp` aux (label "XSP", encoder brain/encoders/xsp.py). Owner design 2026-10-09: "Multiplies
the outputs of the two streams and uses the current frame, the ventral map and the dorsal map to
predict the next frame", trained end-to-end with that single self-supervised loss.

```
(obs_t, a_t, obs_{t+1})  <- action_windows.draw_action_window(memory, k=1)     (as nextframe_aux)
z_{t-1}, z_t, v_t, d_t   <- encoder.encode_streams(obs_t)    phi_e siamese, phi_v, phi_d; grads ON
m      = v_t (x) d_t          NETT_XSP_COMBINE=outer (default): (B, 2K, h, w) = [v_fg*d, v_bg*d]
       | v_fg * d_t           NETT_XSP_COMBINE=fg:              (B,  K, h, w)
h      = ReLU([FiLM_{a_t}] Conv1x1(cat(v_t, m)))          FiLM only under NETT_XSP_ACTION=film
h      = (nearest 2x -> Conv3x3 -> ReLU) x log2(8/ds)     map grid (H/8, W/8) -> target (H/ds, W/ds)
x^     = cur + Conv3x3(ReLU(Conv3x3(cat(h, cur))))        cur = avgpool_ds(x_t)   RESIDUAL
target = avgpool_ds(NEWEST frame of obs_{t+1})
L      = mean (x^ - target)^2
```

⛔ THE TARGET IS THE NEWEST FRAME OF obs[t+1], NOT obs[t+1] WHOLE -- the leak documented in
nextframe_aux.py: obs[t+1] = [x_t, x_{t+1}] (T-major), and its older half IS x_t, which g already
receives. The same `_frames` slicer (last cpf channels) builds the target and the copy baseline.

DESIGN CHOICES (stated, not knobbed):
* RESIDUAL OUTPUT. g predicts the CHANGE on top of the pooled current frame. That is still
  g(x_t, v_t, d_t) -- it is one parameterisation of it -- and it makes "next = current" the
  zero point, so `skill` starts near 0 instead of strongly negative and the streams are trained
  on the part of the image that changes, which is the part motion can explain.
* g gets v_t ITSELF besides the product m: the owner's sentence names "the ventral map and the
  dorsal map" as inputs and the multiplication as how they meet. d_t alone is NOT given to g
  (under "outer" it is recoverable as the sum of the two groups; under "fg" it is deliberately
  visible only where the figure map is on).
* x_t enters g at the TARGET grid (avgpool_ds), concatenated after the up-sampling stages, so
  the frame's appearance is available at full target resolution and the streams' maps carry
  only what the frame does not: which locations are figure, and how things move.
* Action: by default g takes NO action (paper-faithful). NETT_XSP_ACTION=film adds nextframe's
  zero-initialised FiLM on the first two action components (turn, move).

WHAT NOTHING HERE FORCES. The loss does not force v to be a figure-ground split. Under "outer"
the two channels are symmetric (which one lands on the object is not determined by the loss);
under "fg" only channel 0 gates motion, which breaks the symmetry but does not stop v_fg from
saturating at 1 everywhere. `fg_frac`, `fg_entropy` and `fg_frac_std` are published every call
so a collapsed map (fg_frac ~0 or ~1, entropy ~0, std ~0) is visible in the run's own logs.

CHECKPOINTING: as nextframe -- the decoder `head` is in PPO's optimizer but NOT in
`checkpoint_modules`; phi_e/phi_v/phi_d are inside the encoder and so ARE in the policy checkpoint.
A resume therefore restarts g from scratch.

TELEMETRY (last_scalars, per call): B, mse, copy_mse, skill (= 1 - mse/copy_mse, NOT_MEASURED when
copy_mse == 0), mse_parked, mse_transit, copy_parked, copy_transit (median split on |a_t turn|),
window_turn, action_dim, fg_frac (mean v_fg), fg_frac_std (std over the batch of each sample's
mean v_fg), fg_entropy (mean per-location binary entropy of v, in bits: 1 = undecided, 0 = hard),
d_absmean (mean |d_t|), and film_gain (mean |gamma - 1|) under NETT_XSP_ACTION=film only.
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
COMBINES = ("outer", "fg")
ACTIONS = ("none", "film")
#: The XSP trunk's total stride; the decoder's input grid is (H/8, W/8).
TRUNK_STRIDE = 8


def combine_streams(v: torch.Tensor, d: torch.Tensor, mode: str) -> torch.Tensor:
    """m: "outer" -> (B, 2K, h, w), the fg group [v_fg*d] then the bg group [v_bg*d]; "fg" -> v_fg*d."""
    if mode == "outer":
        return (v[:, :, None] * d[:, None]).flatten(1, 2)
    if mode == "fg":
        return v[:, :1] * d
    raise ValueError(f"unknown XSP combine {mode!r}")


class XSPDecoder(nn.Module):
    """g: (cur, v, m[, a]) -> predicted next frame on the target grid (residual on cur)."""

    def __init__(self, m_ch: int, cpf: int, n_up: int, hidden: int, film: bool) -> None:
        super().__init__()
        self.inp = nn.Conv2d(2 + m_ch, hidden, kernel_size=1)
        self.film = nn.Linear(ACTION_COMPONENTS, 2 * hidden) if film else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        # ⚠ NEAREST, not bilinear: bilinear's CUDA backward has no deterministic kernel (nextframe_aux).
        self.up = nn.ModuleList(nn.Conv2d(hidden, hidden, 3, padding=1) for _ in range(n_up))
        self.fuse = nn.Conv2d(hidden + cpf, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, cpf, 3, padding=1)
        self.last_film_gain = 0.0

    def forward(self, cur: torch.Tensor, v: torch.Tensor, m: torch.Tensor,
                action: torch.Tensor | None = None) -> torch.Tensor:
        h = self.inp(torch.cat([v, m], dim=1))
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
        m_ch = 2 * k if self.combine == "outer" else k
        device = next(encoder.parameters()).device
        self.head = XSPDecoder(m_ch, self.cpf, n_up, self.hidden, self.action == "film").to(device)

        self._memory = None
        self.last_scalars: dict = {}
        self.last_window_turn: float = MASK_UNSET
        try:
            from skrl import logger
            logger.warning(
                "[NETT xsp] combine=%s action=%s downsample=%d (target %dx%d) hidden=%d batch=%d "
                "transit_frac=%s policy_input=%s | encoder params %d, decoder (aux head) params %d",
                self.combine, self.action, self.downsample, self.target_hw[0], self.target_hw[1],
                self.hidden, self.batch, self.transit_frac, getattr(encoder, "policy_input", "?"),
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
        """x_{t+1}: the NEWEST frame of obs[t+1] -- never its older half, which IS x_t (the leak)."""
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
        m = combine_streams(s["v"], s["d"], self.combine)
        a = action if self.action == "film" else None
        return self.head(cur, s["v"], m, a), s

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
            v = s["v"].detach().float()
            fg = v[:, 0]
            p = v.clamp_min(1e-12)
            ent = -(p * p.log2()).sum(dim=1)                          # bits, in [0, 1] for 2 classes
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
