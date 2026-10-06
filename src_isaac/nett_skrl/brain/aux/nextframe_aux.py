"""Next-frame predictive coding: decode frame t+1 from the trunk's spatial map at t and the action.

U35 (owner, direct, 2026-10-05: "Could we train the encoder on predictive coding of the next
frame? In other words, the encoder should be able to calculate the future image." -- then
"Implement your suggested predictive coding model in Isaac."). Registered as `nextframe`
(standalone, e.g. "3DCNN-NextFrame") and `nextframe_with_cltt_ref` (WithCLTTRef + this term,
"ViT-CLTT-Ref-NextFrame").

```
(obs_t, a_t, obs_{t+1})  <- action_windows.draw_action_window(memory, k=1)     alignment proven there
m      = spatial map of obs_t          conv map before the pool (3DCNN) | patch tokens (CompactViT)
h      = ReLU(FiLM_{a_t}(Conv1x1(m)))  gamma = 1 + W_g a, beta = W_b a   (W zero-initialised)
x_hat  = decoder(h)                    nearest-upsample + 3x3 conv stages -> (cpf, H/ds, W/ds)
target = avgpool_ds(newest frame of obs_{t+1})                      (T-major: the LAST cpf channels)
L      = mean (x_hat - target)^2       optionally per-sample normalised target (NETT_AUX_NF_NORM_PIX)
```

WHAT IS NOT EoO. EoO (eoo_aux.py) also scores frame t+1, but only through `warp(frame_t, flow)`:
it can move pixels it already has and cannot generate any it does not. Here the decoder must
GENERATE the next image, so appearance change (the stimulus video's own frames, disocclusion,
the view swinging onto new parts of the room) is part of the target.

⛔ THE TARGET IS THE NEWEST FRAME OF obs[t+1], NOT obs[t+1] WHOLE. With framestack, obs[t+1] is
the stack [t, t+1] (T-major, observation.py:96) and its OLDER half is obs[t]'s newest frame --
a target that included it would hand the decoder half the answer from its own input.

⚠ ACTION: the first TWO components of a_t (the wheeled motor command: turn, move -- the same two
`ego_residual_aux` uses). An action vector with fewer than two components refuses.

⚠ CONDITIONING: FiLM (Perez et al. 2018) on the projected map, with the modulation weights
zero-initialised so the term starts as an unconditioned predictor and the action's influence is
learned, not assumed. Concat-tiling the action as extra channels was the alternative; FiLM is the
multiplicative action gating of Oh et al. 2015, which ego_residual_aux also cites.

THE LOOP HAZARD (ego_residual_aux.py: the stimulus video loops deterministically, so a predictor
can learn the loop). For THIS target that is not a defect to guard against but part of what the
objective asks: predicting the stimulus's next frame from the current one REQUIRES the trunk to
encode the stimulus (its identity and its phase), which is the representation the arm is for --
the ego-residual's problem was different, because its RESIDUAL was the readout and a predictor
that explained the object's motion erased it. Two things keep the loop from being the whole
answer here: (1) the stimulus is ~4% of the frame while the ego-motion-driven change covers most
of it, and that change is explained only by a_t, never by the loop; (2) `copy_mse` -- the MSE of
"next = current" on the same target -- and its parked/transit split are published every call, so
a predictor that only learned the screen loop shows as skill > 0 when parked and ~0 in transit.

TELEMETRY (last_scalars, all per call): B, mse, copy_mse, skill (= 1 - mse/copy_mse), mse_parked,
mse_transit, copy_parked, copy_transit (median split on |a_t turn|, token_term.parked_transit),
window_turn, action_dim, film_gain (mean |gamma - 1|, how much the action is used).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_windows import draw_action_window
from .cltt_views import resolve_channels_per_frame
from .knobs import NOT_MEASURED, _env_flag_strict, _env_positive_int, _env_unit_interval
from .token_term import MASK_UNSET, parked_transit, stratum_mean, window_turn_scalar

#: Motor components conditioned on: (turn, move).
ACTION_COMPONENTS = 2


def spatial_map(encoder: nn.Module, prepared: torch.Tensor) -> torch.Tensor:
    """(B, C, h, w) spatial features of a PREPARED image, gradients ON. Refuses rather than pools.

    CompactViT/UnityViT: the post-norm patch tokens folded onto their grid. Otherwise the base
    `encode_spatial` hook (Compact3DCNN: stem + conv map before its 4x4 pool). Anything else
    raises: decoding an image from the pooled vector is a different (bottlenecked) method and
    must not wear this name.
    """
    from .token_features import _has_token_path, spatial_tokens, tokens_as_map
    if _has_token_path(encoder):
        tokens, grid = spatial_tokens(encoder, prepared)
        m = tokens_as_map(tokens, grid)
    else:
        try:
            m = encoder.encode_spatial_prepared(prepared)
        except (AttributeError, NotImplementedError) as exc:
            raise TypeError(
                f"{type(encoder).__name__} exposes neither a token path nor `encode_spatial`, so "
                f"there is no spatial map to decode the next frame from ({exc}). Refusing to "
                f"decode from the pooled vector instead.") from None
    if m.dim() != 4:
        raise TypeError(f"spatial map must be (B, C, h, w); got {tuple(m.shape)}")
    return m


class NextFrameDecoder(nn.Module):
    """(B, C, h, w) map + (B, 2) action -> (B, cpf, Ht, Wt) predicted next frame."""

    def __init__(self, in_ch: int, in_hw: tuple[int, int], out_ch: int,
                 out_hw: tuple[int, int], hidden: int) -> None:
        super().__init__()
        self.out_hw = (int(out_hw[0]), int(out_hw[1]))
        self.inp = nn.Conv2d(in_ch, hidden, kernel_size=1)
        self.film = nn.Linear(ACTION_COMPONENTS, 2 * hidden)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        # Nearest-neighbour 2x stages until the map reaches the target grid. ⚠ NEAREST, not
        # bilinear: bilinear's CUDA backward has no deterministic kernel, and although the aux
        # backward runs under relaxed_determinism, nothing here needs to spend that exemption.
        ratio = max(self.out_hw[0] / max(in_hw[0], 1), self.out_hw[1] / max(in_hw[1], 1))
        self.n_up = max(0, math.ceil(math.log2(ratio))) if ratio > 1 else 0
        self.up = nn.ModuleList(nn.Conv2d(hidden, hidden, 3, padding=1) for _ in range(self.n_up))
        self.mid = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, out_ch, 3, padding=1)
        self.last_film_gain = 0.0

    def forward(self, m: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self.inp(m)
        gamma, beta = self.film(action).chunk(2, dim=-1)
        self.last_film_gain = float(gamma.detach().abs().mean())
        h = F.relu(h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None])
        for conv in self.up:
            h = F.relu(conv(F.interpolate(h, scale_factor=2.0, mode="nearest")))
        if tuple(h.shape[-2:]) != self.out_hw:
            h = F.interpolate(h, size=self.out_hw, mode="nearest")
        h = F.relu(self.mid(h))
        return self.out(h)


class NextFrameTerm(nn.Module):
    """The `nextframe` objective. Owns its decoder in `head`; draws its own windows from memory."""

    needs_memory = True
    needs_teacher = False

    NF_BATCH_ENV = "NETT_AUX_NF_BATCH"
    NF_DOWNSAMPLE_ENV = "NETT_AUX_NF_DOWNSAMPLE"
    NF_HIDDEN_ENV = "NETT_AUX_NF_HIDDEN"
    NF_TRANSIT_FRAC_ENV = "NETT_AUX_NF_TRANSIT_FRAC"
    NF_NORM_PIX_ENV = "NETT_AUX_NF_NORM_PIX"

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.batch = _env_positive_int(self.NF_BATCH_ENV, 64)
        if self.batch < 2:
            raise ValueError(f"{self.NF_BATCH_ENV}={self.batch}: the parked/transit split needs >= 2.")
        self.downsample = _env_positive_int(self.NF_DOWNSAMPLE_ENV, 4)
        hidden = _env_positive_int(self.NF_HIDDEN_ENV, 64)
        self.transit_frac = _env_unit_interval(self.NF_TRANSIT_FRAC_ENV, 0.5)
        self.norm_pix = _env_flag_strict(self.NF_NORM_PIX_ENV, False)

        from ...body.observation import image_channels_hw
        c, h, w = image_channels_hw(getattr(encoder, "observation_space", None))
        if h % self.downsample or w % self.downsample:
            raise ValueError(
                f"{self.NF_DOWNSAMPLE_ENV}={self.downsample} does not divide the {h}x{w} eye; the "
                f"target would be an uneven pool. Pick a common divisor.")
        self.cpf = resolve_channels_per_frame()
        if c % self.cpf:
            raise ValueError(f"{c} input channels are not a whole number of {self.cpf}-channel frames.")
        self.target_hw = (h // self.downsample, w // self.downsample)
        device = next(encoder.parameters()).device
        with torch.no_grad():
            probe = spatial_map(encoder, torch.zeros(1, c, h, w, device=device))
        self.map_ch, self.map_hw = int(probe.shape[1]), (int(probe.shape[2]), int(probe.shape[3]))
        self.head = NextFrameDecoder(self.map_ch, self.map_hw, self.cpf, self.target_hw, hidden).to(device)

        self._memory = None
        self.last_scalars: dict = {}
        self.last_window_turn: float = MASK_UNSET

    def attach_memory(self, memory) -> None:
        self._memory = memory

    def _frames(self, prepared: torch.Tensor) -> torch.Tensor:
        """Newest frame of a prepared T-major stack, pooled to the target grid (no grad)."""
        newest = prepared[:, -self.cpf:]
        return F.avg_pool2d(newest, self.downsample) if self.downsample > 1 else newest

    def _draw(self, encoder: nn.Module):
        if self._memory is None:
            raise RuntimeError("NextFrameTerm draws its own windows; call attach_memory(memory) first.")
        device = next(encoder.parameters()).device
        n_transit = int(round(self.transit_frac * self.batch))
        parts, turn_state = [], None
        # Two half-slabs (transit-weighted, uniform), as token_term.draw_mixed does and for the
        # same reason: a uniform slab is often all-parked, and the parked/transit split needs both.
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

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Ignore the PPO minibatch; draw (obs_t, a_t, obs_{t+1}) windows and score the prediction."""
        obs_t, obs_tk, actions, turn_state = self._draw(encoder)
        if actions.shape[-1] < ACTION_COMPONENTS:
            raise ValueError(
                f"NextFrameTerm conditions on {ACTION_COMPONENTS} motor components; the stored "
                f"actions have {actions.shape[-1]}.")
        a = actions[:, :ACTION_COMPONENTS].float()
        with torch.no_grad():
            prepared_t = encoder._prepare_image(obs_t)
            prepared_tk = encoder._prepare_image(obs_tk)
            target = self._frames(prepared_tk)
            current = self._frames(prepared_t)
            if self.norm_pix:
                mu = target.mean(dim=(1, 2, 3), keepdim=True)
                sd = target.std(dim=(1, 2, 3), keepdim=True) + 1e-6
                target = (target - mu) / sd
                current = (current - mu) / sd
        pred = self.head(spatial_map(encoder, prepared_t), a)
        per_sample = (pred - target).pow(2).mean(dim=(1, 2, 3))
        loss = per_sample.mean()
        with torch.no_grad():
            copy = (current - target).pow(2).mean(dim=(1, 2, 3))
            parked, transit = parked_transit(a[:, 0].abs())
            copy_mse = float(copy.mean())
            mse = float(loss.detach())
            self.last_scalars = {
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
                "film_gain": float(self.head.last_film_gain),
            }
        self.last_window_turn = float(turn_state)
        return loss
