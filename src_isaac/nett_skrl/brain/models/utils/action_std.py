"""NETT_FIXED_ACTION_STD: the policy's action std held at exactly 1 (workspace DECISIONS §187 add. 9).

Owner, 2026-10-23: "Do not forget to make sure that action noise is fixed at 1."

⚠ WHY NETT_MAX_LOG_STD=0 IS NOT ENOUGH. That knob only CLAMPS a trainable ``log_std`` nn.Parameter at
max 0 inside skrl's ``act`` (min stays at skrl's -20). At init log_std is exactly 0 = the clamp bound, where
torch.clamp passes the gradient, so the FIRST update decides: pushed up, the parameter sits above 0, the
clamp zeroes its gradient and the effective std stays exactly 1; pushed down, it keeps training below 0
and the std drifts under 1. Measured on insect 2026-10-23: wave B1/F1 checkpoints froze at +.003..+.005
(std 1 by luck), the h1 Kit smoke's final_agent.pt holds [0.0, -0.0004] (std .9996).

With the knob on, ``log_std`` is a persistent BUFFER of zeros, not a parameter:
  * it is not in ``policy.parameters()``, so skrl's optimizer (built from ``policy.parameters()``) never
    sees it, it has no gradient, and nothing can step it;
  * it IS in ``state_dict()`` under the same key ``log_std``, so checkpoints keep ``policy/log_std``
    (sigma_check.py, probe_policy_readout.py and the retest harness read it unchanged), and a knob-on
    checkpoint loads into knob-off code (the zeros fill the parameter);
  * loading a checkpoint whose ``log_std`` is not exactly 0 under the knob REFUSES (load_state_dict
    pre-hook). Chosen over silently overriding to 0: that checkpoint was trained (or tested) at a
    different std, and continuing or retesting it "at std 1" would be a different policy than the one
    that produced its numbers. The escape is to unset the knob for that run. The error is
    :class:`FixedActionStdCheckpointError`, which ``recording.checkpoints.load_latest_checkpoints`` re-raises in
    every context (its train path otherwise logs a failed load and continues from fresh weights).
Unset / "" / "0" = the original nn.Parameter, statement for statement. "1" = fixed. Anything else refuses.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

FIXED_STD_ENV = "NETT_FIXED_ACTION_STD"
BANNER = "[NETT action std] fixed at 1 (log_std constant 0, not trained) (DECISIONS §187 add. 9)"
_banner_printed = False


class FixedActionStdCheckpointError(RuntimeError):
    """A checkpoint's log_std / tril_offdiag is not exactly 0 under NETT_FIXED_ACTION_STD=1."""


def fixed_action_std_enabled() -> bool:
    """True when NETT_FIXED_ACTION_STD=1. Read per call. "" / unset / "0" = off; anything else refuses."""
    raw = os.environ.get(FIXED_STD_ENV, "").strip()
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise ValueError(f"{FIXED_STD_ENV}={raw!r}: use 1 (std fixed at 1) or unset/0 (trainable log_std); refusing to guess.")


def _refuse_nonzero(names):
    def hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        for name in names:
            key = prefix + name
            if key in state_dict:
                v = state_dict[key]
                if not torch.is_tensor(v) or bool(torch.any(v != 0)):
                    raise FixedActionStdCheckpointError(
                        f"{FIXED_STD_ENV}=1 but the checkpoint's {key} = "
                        f"{v.tolist() if torch.is_tensor(v) else v!r} is not exactly 0: it was trained or tested at a "
                        f"different action noise. Refusing to load it as a std-1 policy; unset {FIXED_STD_ENV} for this run."
                    )
    return hook


def install_fixed_zero_buffer(module: nn.Module, name: str, numel: int) -> None:
    """Knob on only: ``name`` becomes a persistent zero buffer, refused on load unless the checkpoint holds zeros.
    Used for MultivariateGaussianActor.tril_offdiag, whose off-diagonal Cholesky terms would otherwise still move the
    action covariance away from the identity (dim 2's std is sqrt(exp(log_std_2)^2 + offdiag^2))."""
    module.register_buffer(name, torch.zeros(numel), persistent=True)
    module._register_load_state_dict_pre_hook(_refuse_nonzero((name,)))


def install_log_std(module: nn.Module, num_actions: int, initial_log_std: float) -> None:
    """Give ``module`` its ``log_std``: the original trainable parameter, or (knob on) a fixed zero buffer."""
    global _banner_printed
    if not fixed_action_std_enabled():
        module.log_std = nn.Parameter(torch.full((num_actions,), float(initial_log_std)))
        return
    if float(initial_log_std) != 0.0:
        raise ValueError(
            f"{FIXED_STD_ENV}=1 fixes log_std at 0 (std 1) but the model cfg asks for initial_log_std="
            f"{initial_log_std!r}; refusing the contradiction."
        )
    module.register_buffer("log_std", torch.zeros(num_actions), persistent=True)
    module._register_load_state_dict_pre_hook(_refuse_nonzero(("log_std",)))
    if not _banner_printed:
        print(BANNER, flush=True)
        _banner_printed = True
