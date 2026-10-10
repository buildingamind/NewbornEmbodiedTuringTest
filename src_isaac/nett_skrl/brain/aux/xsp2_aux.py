"""XSP2 predictive head and its OWN optimizer: x^_{t+1} = g(x_t, c_t), trained by sum (x^ - x_{t+1})^2.

The `xsp2` aux (label "XSP2", encoder brain/encoders/xsp2.py), owner spec workspace DECISIONS §87.

```
(obs_t, a_t, obs_{t+1})  <- action_windows.draw_action_window(memory, k=1)       (as xsp_aux)
x_t, m_t, d_t, c_t       <- encoder.encode_streams(obs_t)   c_t = up8(m_t * d_t)  (B, 2, 80, 128)
NETT_XSP2_PREDICTOR=warp (DEFAULT):
    x_w  = warp(x_t, c_t)        x_w(p) = bilinear sample of x_t at p - c_t(p), border clamped
    x^   = x_w + refine(cat(x_w, x_t, c_t))       refine: 3 Conv2d 3x3, 8 -> R -> R -> 3, ReLU between
                                                  R = NETT_XSP2_REFINE_HIDDEN (32)
NETT_XSP2_PREDICTOR=conv:
    x^   = conv(cat(c_t, x_t))                    4 Conv2d 3x3, 5 -> H -> H -> H -> 3, ReLU between,
                                                  NOT residual; H = NETT_XSP2_HIDDEN (64)
target = the NEWEST frame of obs_{t+1}            (B, 3, 80, 128), pixels in [0, 1]
e      = sum over batch, channels and pixels of (x^ - target)^2
```

Pixels are in [0, 1]: ``encoder._prepare_image`` -> body/observation.py ``prepare_image_tensor``
divides a uint8 observation by 255 exactly once.

⛔ THE TARGET IS THE NEWEST FRAME OF obs[t+1], NOT obs[t+1] WHOLE (the leak documented in
nextframe_aux.py / xsp_aux.py): obs[t+1] = [x_t, x_{t+1}], and its older frame IS obs[t]'s newest,
which the predictor receives. ``prepared[:, -cpf:]`` builds the target and the copy baseline.

THE WARP. ``warp_bilinear`` gathers the four neighbours of p - c(p) from x_t and blends them with
bilinear weights. c is in PIXEL units of the eye: channel 0 = x (width) displacement, channel 1 =
y (height) displacement, so with c = (+3, -2) on a region the output there is x_t shifted right 3
and up 2: out[y, x] = x_t[y + 2, x - 3] (BACKWARD warp: each output pixel says where it CAME
FROM). Where m = 0, c = 0 and the pixel is copied bitwise. Coordinates outside the image are
clamped to the border (= ``F.grid_sample(align_corners=True, padding_mode="border")``; tests
compare the two numerically on CPU).
⛔ NOT F.grid_sample: its CUDA backward has no deterministic kernel, and this aux steps inside
PPO's ``strict_determinism`` scope (skrl_patches.strict_update wraps AuxLossPPO.update, and
``compute`` runs inside it) -- grid_sample would RAISE there on CUDA. Here x_t is detached, the
gather indices carry no gradient, and the gradient reaches c only through the bilinear weights:
elementwise arithmetic, deterministic on any device.

ENCODER DETACHED FROM THE REWARD; A SEPARATE OPTIMIZER (owner). phi_e, phi_v, phi_d and the
predictor are optimised ONLY by e, with THIS term's own ``torch.optim.Adam``:
* lr = PPO's CURRENT learning rate x NETT_XSP2_LR_SCALE (default 1.0), read from PPO's first
  param group at EVERY aux step, so NETT_DIAG_LR_ANNEAL=linear or a skrl scheduler moves both
  together. eps and betas are torch's defaults (NOT NETT_ADAM_EPS: with a SUMMED loss the
  gradients are orders of magnitude above eps, so it is immaterial; stated, not knobbed).
* NO gradient clipping on this optimizer (PPO's global grad_norm_clip shared with a summed loss
  would starve the policy gradient; that is why the optimizers are separate).
* ``bind_agent`` (called by AuxLossPPO.__init__) REMOVES every encoder parameter from PPO's
  optimizer, mutating ``param_groups`` in place (the NETTSharedEncoderMixin idiom: the scheduler
  and checkpoint_modules["optimizer"] keep pointing at the same object). The encoder has no other
  parameters: its policy output is the detached map itself (encoders/xsp2.py), so nothing of it
  belongs to PPO.
* THE STEP. ``compute`` is called by AuxLossPPO.update ONCE PER PPO MINIBATCH (the existing aux
  cadence: learning_epochs x mini_batches aux steps per update), after the minibatch's policy and
  value forward passes and before PPO's zero_grad/backward/step. It draws its own windows, does
  zero_grad -> backward -> step on its own Adam, sets every grad it made back to None, and
  RETURNS A DETACHED ZERO. PPO then backpropagates ``ppo_loss + 0``: the same value and, through
  autograd's add, the same gradients as with the aux disabled, and its clip norm sees no
  leftover gradient on any encoder parameter. ⚠ CONSEQUENCE FOR READERS: the tfevents scalar
  ``Loss / Aux (xsp2) loss`` is 0.0 BY CONSTRUCTION; the prediction loss is
  ``Loss / Aux xsp2 loss`` (last_scalars["loss"], summed) and ``... mse`` (per pixel).
* The aux step runs with autocast DISABLED (AuxLossPPO calls compute inside the minibatch's
  autocast context): fp32 throughout.
* ``bind_agent`` REFUSES an unshared encoder (the value net would read a second, never-trained
  figure map) and an aux weight other than 1.0 (the weight never reaches this optimizer, so
  NETT_AUX_WEIGHT_OVERRIDE would be inert on this label; use NETT_XSP2_LR_SCALE).

CHECKPOINTING. The encoder is in the policy checkpoint (it is the policy's ``encoder``). The aux
Adam and the predictor are ALSO checkpointed: ``bind_agent`` registers them as
``checkpoint_modules["xsp2_optimizer"]`` and ``["xsp2_predictor"]`` (the eoo_dual/gwm_dual
``aux_head`` precedent), so a resume continues the predictor and the Adam moments exactly.
skrl's Agent.load skips a key the agent does not have (with a warning), so a test-phase agent
built without the aux still loads the policy.

DESIGN CHOICES the spec left open (each stated, none knobbed):
* Windows: the 50/50 transit-weighted/uniform slab split xsp_aux uses (NETT_XSP_TRANSIT_FRAC's
  default), fixed here; the batch must be >= 4 (>= 2 per slab).
* NETT_XSP2_HIDDEN is read only under "conv" and NETT_XSP2_REFINE_HIDDEN only under "warp"; the
  other one SET refuses (the owner's rule for NETT_XSP2_POOL: refuse, never ignore).
* Predictor and refiner keep PyTorch's default init (owner, spec change 2; no zero-init).
* ORIENTATION telemetry (xsp_aux.motion_orientation) runs on m at its native 10x16 grid, the
  grid XSP's own orientation numbers use.

TELEMETRY (last_scalars, per call; names as xsp_aux where they exist): B, loss (summed e), mse
(e per pixel-channel), copy_mse (x_t as the prediction), skill (= 1 - mse/copy_mse, NOT_MEASURED
when copy_mse == 0), warp_mse (x_w alone, "warp" only), fg_frac, fg_frac_std, fg_entropy,
fg_motion_sep, fg_motion_sep_parked, fg_motion_n_parked, flow_absmean (mean |c_t| at 80x128),
d_absmean (mean |d_t|), enc_grad_norm (L2 norm of the aux gradient over phi_e's parameters,
before the step), lr (the aux Adam's lr this step), window_turn.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from .action_windows import draw_action_window
from .knobs import NOT_MEASURED, _env_choice, _env_positive_float, _env_positive_int
from .token_term import MASK_UNSET, parked_transit, window_turn_scalar
from .xsp_aux import motion_orientation

#: Index 0 is the default (owner: "warp").
PREDICTORS = ("warp", "conv")
#: The transit-weighted share of each batch, as xsp_aux's default (fixed here, not a knob).
TRANSIT_FRAC = 0.5


def warp_bilinear(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """out(p) = bilinear sample of x at p - c(p), coordinates clamped to the border.

    x: (B, C, H, W) source, treated as a CONSTANT (detached); c: (B, 2, H, W) displacement in
    pixels, channel 0 = x (width), channel 1 = y (height). Gradients reach c only through the
    bilinear weights (the gather indices are integers from a detached floor).
    """
    if c.shape[1] != 2 or c.shape[0] != x.shape[0] or c.shape[-2:] != x.shape[-2:]:
        raise ValueError(f"warp_bilinear: c {tuple(c.shape)} does not match x {tuple(x.shape)} as (B, 2, H, W)")
    x = x.detach()
    b, ch, h, w = x.shape
    ys = torch.arange(h, device=c.device, dtype=c.dtype).view(1, h, 1)
    xs = torch.arange(w, device=c.device, dtype=c.dtype).view(1, 1, w)
    sx = (xs - c[:, 0]).clamp(0.0, float(w - 1))                       # (B, H, W)
    sy = (ys - c[:, 1]).clamp(0.0, float(h - 1))
    x0 = sx.detach().floor()
    y0 = sy.detach().floor()
    wx = (sx - x0).unsqueeze(1)                                        # (B, 1, H, W)
    wy = (sy - y0).unsqueeze(1)
    x0i, y0i = x0.long(), y0.long()
    x1i, y1i = (x0i + 1).clamp(max=w - 1), (y0i + 1).clamp(max=h - 1)
    flat = x.reshape(b, ch, h * w)

    def tap(yi: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        idx = (yi * w + xi).view(b, 1, h * w).expand(b, ch, h * w)
        return flat.gather(2, idx).view(b, ch, h, w)

    top = (1.0 - wx) * tap(y0i, x0i) + wx * tap(y0i, x1i)
    bottom = (1.0 - wx) * tap(y1i, x0i) + wx * tap(y1i, x1i)
    return (1.0 - wy) * top + wy * bottom


class WarpRefiner(nn.Module):
    """"warp": x^ = x_w + refine(cat(x_w, x_t, c)), x_w = warp_bilinear(x_t, c). Default init."""

    def __init__(self, cpf: int, hidden: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(2 * cpf + 2, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, cpf, 3, padding=1),
        )

    def forward(self, x_t: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_w = warp_bilinear(x_t, c)
        return x_w + self.refine(torch.cat([x_w, x_t.detach(), c], dim=1)), x_w


class ConvPredictor(nn.Module):
    """"conv": x^ = net(cat(c, x_t)), 4 Conv2d 3x3, output 3 channels directly. Default init."""

    def __init__(self, cpf: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 + cpf, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, cpf, 3, padding=1),
        )

    def forward(self, x_t: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, None]:
        return self.net(torch.cat([c, x_t.detach()], dim=1)), None


def exclude_from_optimizer(optimizer, params) -> int:
    """Remove ``params`` from every param group of ``optimizer`` IN PLACE; return how many went.

    In place (not a rebuilt optimizer) so a scheduler and checkpoint_modules["optimizer"] keep
    referring to the same object. Any optimizer state already held for them is dropped too.
    """
    ids = {id(p) for p in params}
    removed = 0
    for group in optimizer.param_groups:
        kept = [p for p in group["params"] if id(p) not in ids]
        removed += len(group["params"]) - len(kept)
        group["params"] = kept
    for p in params:
        optimizer.state.pop(p, None)
    return removed


class XSP2Term(nn.Module):
    """The `xsp2` objective: owns the predictor and the Adam over the encoder's streams."""

    needs_memory = True
    needs_teacher = False
    #: AuxLossPPO calls ``bind_agent(agent)`` for a term that owns its optimizer.
    owns_optimizer = True

    BATCH_ENV = "NETT_XSP2_BATCH"
    PREDICTOR_ENV = "NETT_XSP2_PREDICTOR"
    HIDDEN_ENV = "NETT_XSP2_HIDDEN"
    REFINE_HIDDEN_ENV = "NETT_XSP2_REFINE_HIDDEN"
    LR_SCALE_ENV = "NETT_XSP2_LR_SCALE"

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        from ..encoders.xsp2 import XSP2Encoder
        if not isinstance(encoder, XSP2Encoder):
            raise TypeError(
                f"the xsp2 aux trains the XSP2 encoder's streams; got {type(encoder).__name__}. "
                f"Refusing to train a predictor on a stand-in.")
        self.batch = _env_positive_int(self.BATCH_ENV, 64)
        if self.batch < 4:
            raise ValueError(f"{self.BATCH_ENV}={self.batch}: the transit/uniform slabs need >= 2 each (>= 4).")
        self.predictor_kind = _env_choice(self.PREDICTOR_ENV, "warp", PREDICTORS)
        if self.predictor_kind == "warp":
            if self.HIDDEN_ENV in os.environ:
                raise ValueError(f"{self.HIDDEN_ENV} sizes the 'conv' predictor; {self.PREDICTOR_ENV}=warp "
                                 f"would ignore it. Unset it (the warp refiner's width is {self.REFINE_HIDDEN_ENV}).")
            self.hidden = _env_positive_int(self.REFINE_HIDDEN_ENV, 32)
        else:
            if self.REFINE_HIDDEN_ENV in os.environ:
                raise ValueError(f"{self.REFINE_HIDDEN_ENV} sizes the 'warp' refiner; {self.PREDICTOR_ENV}=conv "
                                 f"would ignore it. Unset it (the conv predictor's width is {self.HIDDEN_ENV}).")
            self.hidden = _env_positive_int(self.HIDDEN_ENV, 64)
        self.lr_scale = _env_positive_float(self.LR_SCALE_ENV, 1.0)
        self.cpf = int(encoder.cpf)
        device = next(encoder.parameters()).device
        cls = WarpRefiner if self.predictor_kind == "warp" else ConvPredictor
        self.predictor = cls(self.cpf, self.hidden).to(device)
        #: AuxLossPPO adds ``head`` to PPO's optimizer and its grad clip: EMPTY on purpose, so
        #: nothing of this term reaches either. The predictor lives in ``predictor``.
        self.head = nn.Module()
        # A plain list, not registered: the encoder's parameters belong to the encoder module.
        self._stream_params = encoder.stream_parameters()
        self._trunk_params = list(encoder.trunk.parameters())
        self.optimizer: torch.optim.Optimizer | None = None
        self._lr_source = None            # PPO's optimizer once bound
        self._base_lr: float | None = None
        self.n_steps = 0
        self._memory = None
        self.last_scalars: dict = {}
        self.last_window_turn: float = MASK_UNSET

    # ------------------------------------------------------------------ optimizer
    def parameters_for_optimizer(self) -> list[nn.Parameter]:
        return [*self._stream_params, *self.predictor.parameters()]

    def build_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Standalone use (tests, probes): the aux Adam at a fixed PPO-equivalent lr."""
        self._base_lr = float(lr)
        self.optimizer = torch.optim.Adam(self.parameters_for_optimizer(), lr=self.current_lr())
        return self.optimizer

    def current_lr(self) -> float:
        if self._lr_source is not None:
            return float(self._lr_source.param_groups[0]["lr"]) * self.lr_scale
        if self._base_lr is None:
            raise RuntimeError("XSP2Term has no learning rate: bind_agent(agent) or build_optimizer(lr) first.")
        return self._base_lr * self.lr_scale

    def bind_agent(self, agent) -> None:
        """Called by AuxLossPPO.__init__: exclude the streams from PPO, build the aux Adam, checkpoint."""
        policy = getattr(agent, "policy", None)
        encoder = getattr(policy, "encoder", None)
        if encoder is None or {id(p) for p in encoder.stream_parameters()} != {id(p) for p in self._stream_params}:
            raise ValueError("XSP2Term.bind_agent: the agent's policy encoder is not the encoder this term was built on.")
        value = getattr(agent, "value", None)
        if value is not None and value is not policy and getattr(value, "encoder", None) is not encoder:
            raise ValueError(
                "xsp2 needs shared_encoder=True: the value network has its OWN encoder, whose figure map "
                "this aux would never train. Refusing rather than giving the critic a random map.")
        weight = float(getattr(agent, "_aux_weight", 1.0))
        if weight != 1.0:
            raise ValueError(
                f"xsp2 aux_weight={weight}: the weight never reaches this term's own Adam (it scales a "
                f"detached zero), so NETT_AUX_WEIGHT / NETT_AUX_WEIGHT_OVERRIDE would be inert. Use "
                f"{self.LR_SCALE_ENV} to change the encoder's step size.")
        ppo_opt = getattr(agent, "optimizer", None)
        if ppo_opt is None:
            raise ValueError("xsp2 needs PPO's optimizer: its learning rate is the aux Adam's base.")
        removed = exclude_from_optimizer(ppo_opt, self._stream_params)
        left = {id(p) for g in ppo_opt.param_groups for p in g["params"]} & {id(p) for p in self._stream_params}
        if left:
            raise RuntimeError(f"xsp2: {len(left)} encoder parameters are still in PPO's optimizer.")
        self._lr_source = ppo_opt
        self.optimizer = torch.optim.Adam(self.parameters_for_optimizer(), lr=self.current_lr())
        checkpoints = getattr(agent, "checkpoint_modules", None)
        if checkpoints is not None:
            checkpoints["xsp2_optimizer"] = self.optimizer
            checkpoints["xsp2_predictor"] = self.predictor
        try:
            from skrl import logger
            logger.warning(
                "[NETT xsp2] predictor=%s hidden=%d batch=%d lr_scale=%s | encoder params %d (all aux-only; "
                "%d removed from PPO's optimizer), predictor params %d; policy features = flattened %dx%d "
                "figure map (%d), detached",
                self.predictor_kind, self.hidden, self.batch, self.lr_scale,
                sum(p.numel() for p in self._stream_params), removed,
                sum(p.numel() for p in self.predictor.parameters()), encoder.map_hw[0], encoder.map_hw[1],
                encoder.features_dim)
        except ImportError:                                  # pragma: no cover - skrl is a dependency
            pass

    # ------------------------------------------------------------------ data
    def attach_memory(self, memory) -> None:
        self._memory = memory

    def _draw(self, encoder: nn.Module):
        if self._memory is None:
            raise RuntimeError("XSP2Term draws its own windows; call attach_memory(memory) first.")
        device = next(encoder.parameters()).device
        n_transit = int(round(TRANSIT_FRAC * self.batch))
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

    # ------------------------------------------------------------------ objective
    def predict(self, encoder: nn.Module, prepared_t: torch.Tensor):
        """(x^_{t+1}, x_w or None, streams) from a PREPARED obs_t stack; gradients on."""
        s = encoder.encode_streams(prepared_t)
        pred, x_w = self.predictor(s["x_t"], s["c"])
        return pred, x_w, s

    def score(self, encoder: nn.Module, prepared_t: torch.Tensor, prepared_tk: torch.Tensor,
              actions: torch.Tensor, turn_state: float = MASK_UNSET) -> torch.Tensor:
        """e = sum (x^ - x_{t+1})^2 on one batch of prepared stacks; publishes last_scalars."""
        with torch.no_grad():
            target = prepared_tk[:, -self.cpf:]
            current = prepared_t[:, -self.cpf:]
        pred, x_w, s = self.predict(encoder, prepared_t)
        if pred.shape != target.shape:
            raise RuntimeError(f"XSP2 prediction {tuple(pred.shape)} != target {tuple(target.shape)}")
        loss = (pred - target).pow(2).sum()
        with torch.no_grad():
            n = float(target.numel())
            mse = float(loss.detach()) / n
            copy_mse = float((current - target).pow(2).mean())
            parked, _ = parked_transit(actions[:, 0].abs().float())
            fg = s["m"].detach().float()[:, 0]                       # (B, h, w)
            p, q = fg.clamp_min(1e-12), (1.0 - fg).clamp_min(1e-12)
            ent = -(fg * p.log2() + (1.0 - fg) * q.log2())
            scalars = {
                "B": float(target.shape[0]),
                "loss": float(loss.detach()),
                "mse": mse,
                "copy_mse": copy_mse,
                "skill": 1.0 - mse / copy_mse if copy_mse > 0 else NOT_MEASURED,
                "fg_frac": float(fg.mean()),
                "fg_frac_std": float(fg.mean(dim=(1, 2)).std()) if fg.shape[0] > 1 else NOT_MEASURED,
                "fg_entropy": float(ent.mean()),
                "flow_absmean": float(s["c"].detach().abs().mean()),
                "d_absmean": float(s["d"].detach().abs().mean()),
                "window_turn": window_turn_scalar(turn_state),
            }
            if x_w is not None:
                scalars["warp_mse"] = float((x_w - target).pow(2).mean())
            scalars.update(motion_orientation(s["m"], current, target, parked))
            self.last_scalars = scalars
        self.last_window_turn = float(turn_state)
        return loss

    def step(self, encoder: nn.Module, prepared_t: torch.Tensor, prepared_tk: torch.Tensor,
             actions: torch.Tensor, turn_state: float = MASK_UNSET) -> torch.Tensor:
        """ONE aux Adam step on one batch: zero_grad -> e.backward -> step -> grads back to None."""
        if self.optimizer is None:
            raise RuntimeError("XSP2Term has no optimizer: bind_agent(agent) or build_optimizer(lr) first.")
        lr = self.current_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = self.score(encoder, prepared_t, prepared_tk, actions, turn_state)
            loss.backward()
        grads = [p.grad for p in self._trunk_params if p.grad is not None]
        enc_grad_norm = float(torch.sqrt(sum(g.pow(2).sum() for g in grads))) if grads else 0.0
        self.optimizer.step()
        # ⛔ BACK TO None, not zero: PPO's grad clip runs over policy.parameters(), which include
        # these, and a leftover gradient would change PPO's clip coefficient.
        self.optimizer.zero_grad(set_to_none=True)
        self.n_steps += 1
        self.last_scalars["enc_grad_norm"] = enc_grad_norm
        self.last_scalars["lr"] = lr
        return loss.detach()

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Ignore the PPO minibatch; draw windows, take ONE aux step, return a DETACHED ZERO."""
        obs_t, obs_tk, actions, turn_state = self._draw(encoder)
        device = next(encoder.parameters()).device
        with torch.autocast(device_type=device.type, enabled=False):
            with torch.no_grad():
                prepared_t = encoder._prepare_image(obs_t)
                prepared_tk = encoder._prepare_image(obs_tk)
            self.step(encoder, prepared_t, prepared_tk, actions, turn_state)
        return torch.zeros((), device=device)
