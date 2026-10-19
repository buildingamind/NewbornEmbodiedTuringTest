"""`slot_recon`: frame t+1 decoded from the slots of frame t, trained by sum (x^ - x_{t+1})^2, with its OWN Adam.

Owner, workspace DECISIONS §162 item 4 as changed by the owner: "the encoder [should] receive frame t and the loss
calculated against frame t+1. Make sure that the loss is the squared L2 norm of the difference. This should be
similar to what we did with the XSP2 loss". Label "ViT-SlotRecon" (encoder brain/encoders/vit_slot_recon.py).
Built on the XSP2 machinery (brain/aux/xsp2_aux.py): same windows, same detachment, same optimizer handling.

```
(obs_t, a_t, obs_{t+1})  <- action_windows.draw_action_window(memory, k=1)      (as xsp2's default "slab" draw)
x_t     = NEWEST frame of obs_t                   (B, 3, 80, 128), pixels in [0, 1]
slots   = encoder.encode_slots(x_t, sample=True)  (B, 3, 64), fresh slot noise (the paper's training)
rgb_k, a_k = decoder(slots_k)                     spatial broadcast to the 5x8 token grid + soft position
                                                  embedding, 4 x ConvT 5x5 s2 -> 80x128, ConvT 5x5, ConvT 3x3
                                                  -> RGB + 1 alpha per slot (64 channels, Locatello et al. E.3)
x^      = sum_k softmax_k(a)_k * rgb_k            alphas softmaxed ACROSS SLOTS (a partition of unity per pixel)
target  = the NEWEST frame of obs_{t+1}           (B, 3, 80, 128)
e       = sum over batch, channels and pixels of (x^ - target)^2      -- XSP2's form, exactly
```

⛔ THE TARGET IS THE NEWEST FRAME OF obs[t+1], NOT obs[t+1] WHOLE (xsp2_aux.py, nextframe_aux.py): obs[t+1] =
[x_t, x_{t+1}] and its older frame IS the encoder's input. ``prepared[:, -cpf:]`` builds input, target and copy.

WINDOWS: draw_action_window's slab draw, 50/50 transit-weighted / uniform (xsp2's default, ``TRANSIT_FRAC``):
every window lies in one episode and off the ring-buffer seam (episode_window_batch's validity), so no pair
crosses a done or a reset. The actions feed only ``window_turn``; the decoder is not action-conditioned.

ENCODER DETACHED FROM THE REWARD; A SEPARATE OPTIMIZER -- xsp2_aux.py's rules, point for point:
* ``bind_agent`` (AuxLossPPO.__init__) REMOVES every encoder parameter from PPO's optimizer in place
  (xsp2_aux.exclude_from_optimizer). The encoder's policy output is its detached slots, so nothing of it is PPO's.
* This term's ``torch.optim.Adam`` over the encoder + decoder: lr = PPO's CURRENT lr x NETT_SLOTRECON_LR_SCALE
  (default 1.0), re-read every step; torch's default betas/eps; NO gradient clipping.
* ``compute`` (once per PPO minibatch) draws its windows, takes ONE zero_grad -> backward -> step, sets its grads
  back to None and RETURNS A DETACHED ZERO, so PPO's loss, gradients and clip norm are plain PPO's.
  ⚠ The tfevents ``Loss / Aux (slot_recon) loss`` is 0.0 by construction; read ``loss`` / ``mse`` below.
* fp32 with autocast DISABLED (AuxLossPPO calls compute inside the minibatch's autocast context).
* REFUSES an unshared encoder (the critic would read a second, never-trained encoder) and an aux weight other
  than 1.0 (it would scale a detached zero; use NETT_SLOTRECON_LR_SCALE).
* The decoder and the Adam are checkpointed: ``checkpoint_modules["slot_recon_decoder"]`` and
  ``["slot_recon_optimizer"]`` (xsp2_predictor / xsp2_optimizer).

NO CLTT TERM, no other objective: e is the whole loss. NETT_SLOTRECON_BATCH (default 64, the paper's batch, >= 4
for the two slabs) sizes the draw; the full-resolution broadcast decoder holds B x 3 slots x 64 x 80 x 128
activations per layer, which is why it has its own knob.

DECLARED: the decoder keeps PyTorch's default init (it lives in this term, not in the encoder the builder
initialises; xsp2's predictor is the precedent); pixels are the prepared [0, 1] frame on both sides (xsp2), not
the paper's [-1, 1]; the decoder is the paper's at 16x upsampling from the 5x8 token grid, where the paper starts
at 8x8 and reaches 128x128.

TELEMETRY (last_scalars, per call): B, loss (summed e), mse (e per pixel-channel), copy_mse (x_t as the
prediction), skill (= 1 - mse/copy_mse, NOT_MEASURED when copy_mse == 0), alpha_max (mean over pixels of the
largest slot alpha: 1/3 = no partition, 1 = hard), slot_share_min / slot_share_max (the smallest / largest slot's
mean alpha), slot_norm (mean |slot|), enc_grad_norm (L2 norm of the aux gradient over the ViT trunk, before the
step), lr, window_turn.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .action_windows import draw_action_window
from .knobs import NOT_MEASURED, _env_positive_float, _env_positive_int
from .slot_attention_ae import SoftPositionEmbed
from .token_term import MASK_UNSET, window_turn_scalar
from .xsp2_aux import TRANSIT_FRAC, exclude_from_optimizer


class SlotBroadcastDecoder(nn.Module):
    """(B, K, D) slots -> (x^ (B, C, H, W), alphas (B, K, H, W)): Locatello et al.'s spatial-broadcast decoder."""

    def __init__(self, slot_dim: int, grid: tuple[int, int], out_ch: int, n_up: int, ch: int = 64) -> None:
        super().__init__()
        self.grid, self.out_ch = (int(grid[0]), int(grid[1])), int(out_ch)
        self.pos = SoftPositionEmbed(slot_dim, self.grid)
        layers, c = [], slot_dim
        for _ in range(n_up):
            layers += [nn.ConvTranspose2d(c, ch, 5, stride=2, padding=2, output_padding=1), nn.ReLU()]
            c = ch
        layers += [nn.ConvTranspose2d(c, ch, 5, stride=1, padding=2), nn.ReLU(),
                   nn.ConvTranspose2d(ch, out_ch + 1, 3, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, k, d = slots.shape
        z = slots.reshape(b * k, d, 1, 1).expand(-1, -1, *self.grid)
        out = self.net(self.pos(z))                                   # (B*K, C+1, H, W)
        out = out.reshape(b, k, self.out_ch + 1, *out.shape[-2:])
        alphas = torch.softmax(out[:, :, self.out_ch], dim=1)        # over SLOTS
        return (out[:, :, : self.out_ch] * alphas.unsqueeze(2)).sum(dim=1), alphas


class SlotReconTerm(nn.Module):
    """The `slot_recon` objective: owns the decoder and the Adam over the ViT-SlotRecon encoder."""

    needs_memory = True
    needs_teacher = False
    #: AuxLossPPO calls ``bind_agent(agent)`` for a term that owns its optimizer.
    owns_optimizer = True

    BATCH_ENV = "NETT_SLOTRECON_BATCH"
    LR_SCALE_ENV = "NETT_SLOTRECON_LR_SCALE"

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        from ..encoders.vit_slot_recon import ViTSlotRecon
        if not isinstance(encoder, ViTSlotRecon):
            raise TypeError(f"the slot_recon aux trains the ViT-SlotRecon encoder; got {type(encoder).__name__}. "
                            f"Refusing to train a decoder on a stand-in.")
        self.batch = _env_positive_int(self.BATCH_ENV, 64)
        if self.batch < 4:
            raise ValueError(f"{self.BATCH_ENV}={self.batch}: the transit/uniform slabs need >= 2 each (>= 4).")
        self.lr_scale = _env_positive_float(self.LR_SCALE_ENV, 1.0)
        self.cpf = int(encoder.cpf)
        n_up = int(round(math.log2(encoder.patch_size)))
        if 2 ** n_up != encoder.patch_size:
            raise ValueError(f"slot_recon upsamples the token grid by stride-2 layers; patch {encoder.patch_size} "
                             f"is not a power of 2.")
        device = next(encoder.parameters()).device
        self.decoder = SlotBroadcastDecoder(encoder.slot_dim, encoder.token_grid, self.cpf, n_up).to(device)
        with torch.no_grad():
            probe, _ = self.decoder(torch.zeros(1, encoder.num_slots, encoder.slot_dim, device=device))
        if tuple(probe.shape[-2:]) != tuple(encoder.image_hw):
            raise RuntimeError(f"slot_recon decoder output {tuple(probe.shape[-2:])} != eye {tuple(encoder.image_hw)}")
        #: AuxLossPPO adds ``head`` to PPO's optimizer and its grad clip: EMPTY on purpose (xsp2).
        self.head = nn.Module()
        # A plain list, not registered: the encoder's parameters belong to the encoder module.
        self._stream_params = encoder.stream_parameters()
        self._trunk_params = list(encoder.vit.parameters())
        self.optimizer: torch.optim.Optimizer | None = None
        self._lr_source = None            # PPO's optimizer once bound
        self._base_lr: float | None = None
        self.n_steps = 0
        self._memory = None
        self.last_scalars: dict = {}
        self.last_window_turn: float = MASK_UNSET

    # ------------------------------------------------------------------ optimizer
    def parameters_for_optimizer(self) -> list[nn.Parameter]:
        return [*self._stream_params, *self.decoder.parameters()]

    def build_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Standalone use (tests, probes): the aux Adam at a fixed PPO-equivalent lr."""
        self._base_lr = float(lr)
        self.optimizer = torch.optim.Adam(self.parameters_for_optimizer(), lr=self.current_lr())
        return self.optimizer

    def current_lr(self) -> float:
        if self._lr_source is not None:
            return float(self._lr_source.param_groups[0]["lr"]) * self.lr_scale
        if self._base_lr is None:
            raise RuntimeError("SlotReconTerm has no learning rate: bind_agent(agent) or build_optimizer(lr) first.")
        return self._base_lr * self.lr_scale

    def bind_agent(self, agent) -> None:
        """Called by AuxLossPPO.__init__: exclude the encoder from PPO, build the aux Adam, checkpoint."""
        policy = getattr(agent, "policy", None)
        encoder = getattr(policy, "encoder", None)
        if encoder is None or {id(p) for p in encoder.stream_parameters()} != {id(p) for p in self._stream_params}:
            raise ValueError("SlotReconTerm.bind_agent: the agent's policy encoder is not the encoder this term was built on.")
        value = getattr(agent, "value", None)
        if value is not None and value is not policy and getattr(value, "encoder", None) is not encoder:
            raise ValueError(
                "slot_recon needs shared_encoder=True: the value network has its OWN encoder, whose slots this "
                "aux would never train. Refusing rather than giving the critic a random encoder.")
        weight = float(getattr(agent, "_aux_weight", 1.0))
        if weight != 1.0:
            raise ValueError(
                f"slot_recon aux_weight={weight}: the weight never reaches this term's own Adam (it scales a "
                f"detached zero), so NETT_AUX_WEIGHT / NETT_AUX_WEIGHT_OVERRIDE would be inert. Use "
                f"{self.LR_SCALE_ENV} to change the encoder's step size.")
        ppo_opt = getattr(agent, "optimizer", None)
        if ppo_opt is None:
            raise ValueError("slot_recon needs PPO's optimizer: its learning rate is the aux Adam's base.")
        removed = exclude_from_optimizer(ppo_opt, self._stream_params)
        left = {id(p) for g in ppo_opt.param_groups for p in g["params"]} & {id(p) for p in self._stream_params}
        if left:
            raise RuntimeError(f"slot_recon: {len(left)} encoder parameters are still in PPO's optimizer.")
        self._lr_source = ppo_opt
        self.optimizer = torch.optim.Adam(self.parameters_for_optimizer(), lr=self.current_lr())
        checkpoints = getattr(agent, "checkpoint_modules", None)
        if checkpoints is not None:
            checkpoints["slot_recon_optimizer"] = self.optimizer
            checkpoints["slot_recon_decoder"] = self.decoder
        try:
            from skrl import logger
            logger.warning(
                "[NETT slot_recon] batch=%d lr_scale=%s | encoder params %d (all aux-only; %d removed from PPO's "
                "optimizer), decoder params %d; policy features = %d slots x %d, detached",
                self.batch, self.lr_scale, sum(p.numel() for p in self._stream_params), removed,
                sum(p.numel() for p in self.decoder.parameters()), encoder.num_slots, encoder.slot_dim)
        except ImportError:                                  # pragma: no cover - skrl is a dependency
            pass

    # ------------------------------------------------------------------ data
    def attach_memory(self, memory) -> None:
        self._memory = memory

    def _draw(self, encoder: nn.Module):
        """xsp2's "slab" draw, unchanged: two slabs of B/2 windows, one transit-weighted, one uniform."""
        if self._memory is None:
            raise RuntimeError("SlotReconTerm draws its own windows; call attach_memory(memory) first.")
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
    def score(self, encoder: nn.Module, prepared_t: torch.Tensor, prepared_tk: torch.Tensor,
              turn_state: float = MASK_UNSET) -> torch.Tensor:
        """e = sum (x^ - x_{t+1})^2 on one batch of prepared stacks; publishes last_scalars."""
        with torch.no_grad():
            target = encoder.current_frame(prepared_tk)
        current = encoder.current_frame(prepared_t)
        slots = encoder.encode_slots(current, sample=True)
        pred, alphas = self.decoder(slots)
        if pred.shape != target.shape:
            raise RuntimeError(f"slot_recon prediction {tuple(pred.shape)} != target {tuple(target.shape)}")
        loss = (pred - target).pow(2).sum()
        with torch.no_grad():
            n = float(target.numel())
            mse = float(loss.detach()) / n
            copy_mse = float((current - target).pow(2).mean())
            share = alphas.mean(dim=(0, 2, 3))                          # (K,) mean alpha per slot
            self.last_scalars = {
                "B": float(target.shape[0]),
                "loss": float(loss.detach()),
                "mse": mse,
                "copy_mse": copy_mse,
                "skill": 1.0 - mse / copy_mse if copy_mse > 0 else NOT_MEASURED,
                "alpha_max": float(alphas.max(dim=1).values.mean()),
                "slot_share_min": float(share.min()),
                "slot_share_max": float(share.max()),
                "slot_norm": float(slots.detach().norm(dim=-1).mean()),
                "window_turn": window_turn_scalar(turn_state),
            }
        self.last_window_turn = float(turn_state)
        return loss

    def step(self, encoder: nn.Module, prepared_t: torch.Tensor, prepared_tk: torch.Tensor,
             turn_state: float = MASK_UNSET) -> torch.Tensor:
        """ONE aux Adam step on one batch: zero_grad -> e.backward -> step -> grads back to None."""
        if self.optimizer is None:
            raise RuntimeError("SlotReconTerm has no optimizer: bind_agent(agent) or build_optimizer(lr) first.")
        lr = self.current_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad():
            loss = self.score(encoder, prepared_t, prepared_tk, turn_state)
            loss.backward()
        grads = [p.grad for p in self._trunk_params if p.grad is not None]
        enc_grad_norm = float(torch.sqrt(sum(g.pow(2).sum() for g in grads))) if grads else 0.0
        self.optimizer.step()
        # ⛔ BACK TO None, not zero: PPO's grad clip runs over policy.parameters(), which include these.
        self.optimizer.zero_grad(set_to_none=True)
        self.n_steps += 1
        self.last_scalars["enc_grad_norm"] = enc_grad_norm
        self.last_scalars["lr"] = lr
        return loss.detach()

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Ignore the PPO minibatch; draw windows, take ONE aux step, return a DETACHED ZERO."""
        obs_t, obs_tk, _actions, turn_state = self._draw(encoder)
        device = next(encoder.parameters()).device
        with torch.autocast(device_type=device.type, enabled=False):
            with torch.no_grad():
                prepared_t = encoder._prepare_image(obs_t)
                prepared_tk = encoder._prepare_image(obs_tk)
            self.step(encoder, prepared_t, prepared_tk, turn_state)
        return torch.zeros((), device=device)
