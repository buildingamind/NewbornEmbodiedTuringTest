"""`stp`: the STP-E predictor's OWN Adam, trained on the rollout by a Charbonnier next-frame loss (DECISIONS §170).

Labels "STP-E", "STP-E-NoWarp", "STP-E-Cmd" (encoder brain/encoders/stp_e.py). Built on the XSP2 machinery
(xsp2_aux.py): the term owns an optimizer, removes the predictor from PPO's, steps once per PPO minibatch and
returns a detached zero.

```
row     <- B stored observations, iid uniform over the rollout's rows whose pose row says valid = 1
           (each row IS a transition: its 2-frame stack holds x_{t-1}, x_t and the t-1 -> t pose change)
g_s     = encoder.azel(x_s)                            s = t-1, t   (fixed remap, no gradient)
a       = encoder.action(pose)                         dead zone, scaled
xhat, wv = encoder.predictor(g_{t-1}, a, vm)
w       = vm * (wv > 0.999)                            pixels the warps can fill; rim and border excluded
e       = sum_{b,c,p} sqrt((xhat - g_t)^2 + 1e-6) w  /  max(3 sum w, 1)      Charbonnier, mean over valid pixels
```

ACROSS EPISODE RESETS. A stored row whose pose row says valid = 0 is the first frame of an episode (the wrapper
marks it at reset() and for every env that reset in a step), and its stack is [x_new, x_new]. Such rows are
never drawn, so the loss never spans a reset. Because every sample is ONE stored row, the ring-buffer seam and
the done flags of neighbouring rows play no part.

SEPARATE OPTIMIZER (xsp2_aux.py's rules):
* ``bind_agent`` REMOVES the predictor's parameters from PPO's optimizer in place (exclude_from_optimizer) and
  builds this term's ``torch.optim.Adam`` over them alone: lr = NETT_STP_LR (default 2e-3, the offline recipe),
  FIXED -- not tied to PPO's lr or its schedule; torch's default betas / eps; no gradient clipping. The policy CNN
  stays in PPO's optimizer and is never touched here.
* ``compute`` (once per PPO minibatch, inside AuxLossPPO.update) draws its rows, does zero_grad -> backward ->
  step under autocast DISABLED, sets the grads it made back to None, and RETURNS A DETACHED ZERO. ⚠ The tfevents
  ``Loss / Aux (stp) loss`` is 0.0 by construction; the prediction loss is ``Loss / Aux stp loss``.
* No RL gradient reaches the predictor: the encoder's forward builds the residual under no_grad.
* REFUSES an unshared encoder (the critic would read a second, untrained predictor) and an aux weight other than
  1.0 (it would scale a detached zero).
* Checkpointed: ``checkpoint_modules["stp_optimizer"]``. The predictor itself is in the policy checkpoint (it is
  part of the policy's encoder).

NoWarp (``warp == "identity"``): the predictor is never used by the policy, so it is NOT trained -- no Adam is
built and ``compute`` only publishes the copy baseline (the arm's own residual). Its parameters still leave PPO's
optimizer (they would get no gradient there anyway), so its PPO sees exactly the policy CNN.

KNOBS (each refuses junk; a stray NETT_STP_* on another label refuses in campaign_train):
* NETT_STP_LR     finite float > 0, default 2e-3.
* NETT_STP_BATCH  integer >= 2, default 32 (the offline recipe).

TELEMETRY (last_scalars): B, loss (the Charbonnier mean), copy_loss (g_{t-1} as the prediction, same w), skill
(1 - loss / copy_loss), valid_frac (w's share of the grid), moving_frac (rows with a != 0), mask_bg (mean weight
of the unchanged copy), g_turn (mean |turn generator x-shift|), valid_rows (share of rollout rows with valid = 1),
lr.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from .knobs import NOT_MEASURED, _env_positive_float, _env_positive_int
from .xsp2_aux import exclude_from_optimizer

#: Charbonnier epsilon^2 (offline: sqrt(d^2 + 1e-6)).
CHARBONNIER_EPS2 = 1e-6


def charbonnier(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Mean Charbonnier over the pixels with weight ``w`` (B, 1, H, W) and the channels; 0 if none is valid."""
    d = torch.sqrt((pred - target).pow(2) + CHARBONNIER_EPS2)
    return (d * w).sum() / (w.sum() * pred.shape[1]).clamp_min(1.0)


class STPTerm(nn.Module):
    """The `stp` objective: owns the Adam over the STP-E predictor."""

    needs_memory = True
    needs_teacher = False
    #: AuxLossPPO calls ``bind_agent(agent)`` for a term that owns its optimizer.
    owns_optimizer = True

    LR_ENV = "NETT_STP_LR"
    BATCH_ENV = "NETT_STP_BATCH"

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        from ..encoders.stp_e import STPEEncoder
        if not isinstance(encoder, STPEEncoder):
            raise TypeError(f"the stp aux trains the STP-E predictor; got {type(encoder).__name__}. "
                            f"Refusing to train a predictor on a stand-in.")
        self.lr = _env_positive_float(self.LR_ENV, 2e-3)
        self.batch = _env_positive_int(self.BATCH_ENV, 32)
        if self.batch < 2:
            raise ValueError(f"{self.BATCH_ENV}={self.batch}: a batch of 1 makes every per-batch scalar a single sample.")
        self.trains = encoder.warp != "identity"
        #: AuxLossPPO adds ``head`` to PPO's optimizer: EMPTY on purpose (xsp2).
        self.head = nn.Module()
        # A plain list, not registered: the predictor belongs to the encoder module.
        self._pred_params = encoder.predictor_parameters()
        self.optimizer: torch.optim.Optimizer | None = None
        self.n_steps = 0
        self._memory = None
        self.last_scalars: dict = {}

    # ------------------------------------------------------------------ optimizer
    def build_optimizer(self) -> torch.optim.Optimizer | None:
        """The predictor's Adam (None under NoWarp, which never trains it)."""
        self.optimizer = torch.optim.Adam(self._pred_params, lr=self.lr) if self.trains else None
        return self.optimizer

    def bind_agent(self, agent) -> None:
        """Called by AuxLossPPO.__init__: exclude the predictor from PPO, build the aux Adam, checkpoint."""
        policy = getattr(agent, "policy", None)
        encoder = getattr(policy, "encoder", None)
        if encoder is None or {id(p) for p in encoder.predictor_parameters()} != {id(p) for p in self._pred_params}:
            raise ValueError("STPTerm.bind_agent: the agent's policy encoder is not the encoder this term was built on.")
        value = getattr(agent, "value", None)
        if value is not None and value is not policy and getattr(value, "encoder", None) is not encoder:
            raise ValueError("stp needs shared_encoder=True: the value network has its OWN encoder, whose predictor "
                             "this aux would never train.")
        weight = float(getattr(agent, "_aux_weight", 1.0))
        if weight != 1.0:
            raise ValueError(f"stp aux_weight={weight}: the weight never reaches this term's own Adam (it scales a "
                             f"detached zero). Use {self.LR_ENV} to change the predictor's step size.")
        ppo_opt = getattr(agent, "optimizer", None)
        if ppo_opt is None:
            raise ValueError("stp needs PPO's optimizer to remove the predictor from it.")
        removed = exclude_from_optimizer(ppo_opt, self._pred_params)
        left = {id(p) for g in ppo_opt.param_groups for p in g["params"]} & {id(p) for p in self._pred_params}
        if left:
            raise RuntimeError(f"stp: {len(left)} predictor parameters are still in PPO's optimizer.")
        self.build_optimizer()
        checkpoints = getattr(agent, "checkpoint_modules", None)
        if checkpoints is not None and self.optimizer is not None:
            checkpoints["stp_optimizer"] = self.optimizer
        try:
            from skrl import logger
            counts = encoder.parameter_counts()
            logger.warning(
                "[NETT stp] warp=%s trains=%s lr=%g batch=%d | predictor params %d (aux-only; %d tensors removed "
                "from PPO's optimizer), policy CNN params %d, encoder total %d; grid %dx%d (H x W), fov %g",
                encoder.warp, self.trains, self.lr, self.batch, counts["predictor"], removed, counts["policy_cnn"],
                counts["total"], encoder.grid_hw[0], encoder.grid_hw[1], encoder.fov_deg)
        except ImportError:                                  # pragma: no cover - skrl is a dependency
            pass

    # ------------------------------------------------------------------ data
    def attach_memory(self, memory) -> None:
        self._memory = memory

    def _draw(self, encoder: nn.Module):
        """B stored observations, iid uniform over rows whose newest frame's pose row says valid = 1."""
        if self._memory is None:
            raise RuntimeError("STPTerm draws its own rows; call attach_memory(memory) first.")
        raw = self._memory.tensors["observations"]
        # Rows written so far: all of them once the ring has filled (PPO updates on a full rollout).
        t_max = raw.shape[0] if bool(getattr(self._memory, "filled", True)) else int(self._memory.memory_index)
        from ...body.wrappers.stp_pose import VALID
        pose = encoder.stored_pose(raw[:t_max])                                    # (T, env, 4)
        valid = pose[..., VALID] == 1.0
        idx = valid.reshape(-1).nonzero().flatten()
        frac = float(valid.float().mean()) if valid.numel() else 0.0
        if idx.numel() == 0:
            return None, frac
        pick = idx[torch.randint(idx.numel(), (self.batch,))]
        t, env = pick // valid.shape[1], pick % valid.shape[1]
        device = encoder.vm.device
        return raw[t.to(raw.device), env.to(raw.device)].to(device), frac

    # ------------------------------------------------------------------ objective
    def score(self, encoder: nn.Module, observations: torch.Tensor):
        """-> (loss with grad when training, scalars) on one batch of raw stored observations."""
        with torch.no_grad():
            pose, x_prev, x_cur = encoder.split(observations)
            g_prev, g_cur = encoder.azel(x_prev), encoder.azel(x_cur)
            a = encoder.action(pose)
        if self.trains:
            xhat, wv, m = encoder.predictor(g_prev, a, encoder.vm)
        else:
            xhat, wv, m = g_prev, encoder.vm.expand(g_prev.shape[0], 1, *encoder.grid_hw), None
        w = (encoder.vm * (wv > 0.999).to(g_cur.dtype)).detach()
        loss = charbonnier(xhat, g_cur, w)
        with torch.no_grad():
            copy = float(charbonnier(g_prev, g_cur, w))
            G = encoder.predictor.G
            scalars = {
                "B": float(g_cur.shape[0]),
                "loss": float(loss.detach()),
                "copy_loss": copy,
                "skill": 1.0 - float(loss.detach()) / copy if copy > 0 else NOT_MEASURED,
                "valid_frac": float(w.mean()),
                "moving_frac": float((a != 0).any(dim=1).float().mean()),
                "mask_bg": float(m[:, -1].mean()) if m is not None else NOT_MEASURED,
                "g_turn": float(G[2, :, 0, 2].abs().mean()),
            }
        return loss, scalars

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Ignore the PPO minibatch; draw rows, take ONE aux step (none under NoWarp), return a DETACHED ZERO."""
        device = encoder.vm.device
        obs, frac = self._draw(encoder)
        if obs is None:
            self.last_scalars = {"valid_rows": frac}
            return torch.zeros((), device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            if self.trains:
                if self.optimizer is None:
                    raise RuntimeError("STPTerm has no optimizer: bind_agent(agent) or build_optimizer() first.")
                self.optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    loss, scalars = self.score(encoder, obs)
                    loss.backward()
                self.optimizer.step()
                # ⛔ BACK TO None, not zero: PPO's grad clip runs over policy.parameters(), which include these.
                self.optimizer.zero_grad(set_to_none=True)
                self.n_steps += 1
            else:
                with torch.no_grad():
                    _, scalars = self.score(encoder, obs)
        scalars["valid_rows"] = frac
        scalars["lr"] = self.lr if self.trains else 0.0
        self.last_scalars = scalars
        return torch.zeros((), device=device)
