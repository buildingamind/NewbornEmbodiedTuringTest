"""Action-aligned temporal windows from rollout memory: (obs_t, obs_{t+k}, a_t .. a_{t+k-1}).

Wave 17's ego-motion residual (P3) predicts the token change an ACTION causes, so it needs the
actions that lie BETWEEN two observations -- and an off-by-one here would train a forward model
on the action that was taken AFTER the frame it is meant to explain, which still trains, still
reports a falling loss, and learns a correlate instead of a cause. The alignment is therefore
established from the source below, not assumed.

★ ALIGNMENT EVIDENCE (skrl 2.1.0 as installed in the fleet venv; NETT repo A @ 2141520)

1. skrl/trainers/torch/sequential.py `train()` (the trainer NETT runs:
   brain/trainer.py `_run_skrl_train` builds a `SequentialTrainer`), per timestep:
       actions, _ = agent.act(observations[scope])                     # decided FROM obs
       next_observations, rewards, terminated, truncated, infos = env.step(actions)
       agent.record_transition(observations=observations[scope], actions=actions[scope],
                               next_observations=..., terminated=..., truncated=..., ...)
       ...
       observations = next_observations
   The single-agent path (`base.py` `single_agent_train`) has the same order.
2. skrl/agents/torch/ppo/ppo.py `record_transition` -> `self.memory.add_samples(
   observations=observations, actions=actions, terminated=terminated, truncated=truncated, ...)`,
   and skrl/memories/torch/base.py `add_samples` writes EVERY named tensor at the SAME
   `memory_index`, then increments it once.
3. NETT's overrides do not reorder this: skrl_patches.NETTBootstrapMixin.record_transition only
   moves `rewards`/`truncated` to the agent device before delegating; HybridDeviceMemory and
   Uint8StatesMemory change storage device/dtype, not indexing; the env wrappers
   (SegmentationStatsEnvWrapper, IntrinsicRewardEnvWrapper, framestack) pass `actions` straight
   to the inner `step`.

⇒ At memory row t: `observations[t]` is the observation the policy saw, `actions[t]` is the
action sampled FROM it and applied by the `env.step` that produced `observations[t+1]`, and
`terminated[t] | truncated[t]` flags THAT SAME transition. So
       obs[t] --a[t]--> obs[t+1] --a[t+1]--> ... --a[t+k-1]--> obs[t+k]
and a window is clean exactly when done[t .. t+k-1] are all False -- which is precisely the
convention `cltt_ref_aux.episode_window_starts` already enforces ("a done at the endpoint is
allowed; one before it is not"), ring-buffer seam included. This module reuses it unchanged.

⚠ CAVEATS THAT THE MEMORY CANNOT SEE
- The stored action is the policy's SAMPLE after its own clip (`model_cfg.clip_actions=True`),
  which Repo B's `MotorSystem.apply` receives and clamps to [-1, 1] again. That is the applied
  command only while `screens.decision_period == 1` (the default, nett_isaac video/config.py and
  nett_skrl environment.py). At decision_period > 1 `_apply_motor_actions` zeroes non-decision
  sub-steps, so the stored action is COMMANDED, not EFFECTIVE, on those rows.
- With framestack, "obs[t]" is the stack whose NEWEST frame is t.
- NOT VERIFIED HERE: whether the Isaac camera render for obs[t+1] reflects the pose after a[t]
  or lags one physics step. That is a property of the sim's render/sensor update order and can
  only be measured in Isaac (e.g. a replay with a pure-turn action and a known yaw gain).

TRANSIT WEIGHTING. `window_mean_abs_turn` / `transit_weights` are the arithmetic of
`VICRegTemporalAuxLoss._select_window`'s transit mask, moved here so both callers share one
copy; vicreg_tt's branching, sentinels, warnings and RNG draw order are untouched (pinned against
a frozen copy in tests/test_action_windows.py). Why |turn| and not |move|: see that docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cltt_ref_aux import draw_episode_window, episode_window_batch

#: `mean_turn` states, numerically identical to VICRegTemporalAuxLoss's sentinels so
#: `ppo_aux.track_transit_mask` reads them without translation. >= 0.0 means engaged.
MASK_OFF = -1.0
MASK_NO_MOTION = -3.0


def window_mean_abs_turn(actions: torch.Tensor, rows: int, n_env: int, span: int) -> torch.Tensor:
    """Mean |turn| over every contiguous ``span``-row window of ``actions[:rows, :n_env, 0]``.

    Returns (rows - span + 1, n_env) on CPU; row i is the mean over actions[i : i + span].
    """
    n_windows = rows - span + 1
    turn = actions[:rows, :n_env, 0].abs().float().cpu()      # (rows, n_env)
    # Mean |turn| over every contiguous window, via cumulative sum.
    csum = torch.cat([torch.zeros(1, turn.shape[1]), turn.cumsum(0)], dim=0)
    return (csum[span:] - csum[:n_windows]) / float(span)    # (n_windows, n_env)


def transit_weights(win: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    """Flattened (t0, env) sampling weights: the window's mean |turn|, zero where unsafe."""
    return win.masked_fill(~starts, 0.0).flatten().clamp_min(0.0)


@dataclass
class ActionWindow:
    """One contiguous in-episode slab of ``batch`` anchors from one env stream.

    For anchor i (time t = t0 + i): ``obs_t[i]`` = observations[t], ``obs_tk[i]`` =
    observations[t + k], ``actions[i, j]`` = actions[t + j] for j in 0..k-1 -- the actions
    taken between them, in order. Tensors are sliced from STORAGE (moved to ``device`` only if
    one was given), never the whole buffer.
    """

    env: int
    t0: int
    batch: int
    k: int
    obs_t: torch.Tensor           # (B, *obs)
    obs_tk: torch.Tensor          # (B, *obs)
    actions: torch.Tensor         # (B, k, action_dim)
    mean_turn: float              # >= 0 engaged transit weighting; MASK_OFF / MASK_NO_MOTION


def draw_action_window(memory, k: int, max_samples: int, *, transit_weighted: bool = False,
                       device=None) -> ActionWindow:
    """Draw (obs_t, obs_{t+k}, a_t..a_{t+k-1}) for a contiguous slab that crosses no boundary.

    Safety is `episode_window_batch(memory, (k,), ...)`: every anchor's window [t, t+k] has no
    done flag before its endpoint and does not straddle the ring-buffer seam, including the
    intervening anchors of the slab. The actions a_t..a_{t+k-1} therefore all belong to the
    same episode as both observations.

    ``transit_weighted`` samples the slab with probability proportional to its mean |turn| over
    EVERY action the slab returns (rows t0 .. t0+batch+k-2), instead of uniformly. Agents park
    ~81% of steps (C3 in the wave-17 plan), so uniform slabs mostly carry a_t ~ 0. If the
    rollout has no commanded rotation at all it samples uniformly and says so through
    ``mean_turn == MASK_NO_MOTION`` -- a state the caller must publish, never a silent fallback.

    ⛔ No `actions` tensor RAISES. This helper's output is the alignment; a window without its
    actions is not a degraded version of it.
    """
    if int(k) < 1:
        raise ValueError(f"draw_action_window: k must be a positive integer; got {k!r}.")
    k = int(k)
    actions = memory.tensors.get("actions")
    if actions is None or actions.ndim != 3 or actions.shape[-1] < 1:
        raise ValueError(
            "draw_action_window: rollout memory has no usable 'actions' tensor shaped "
            f"(T, env, action_dim); got {None if actions is None else tuple(actions.shape)}. "
            "Refusing to return observation pairs without the actions between them."
        )
    raw = memory.tensors["observations"]
    t_max = memory.memory_size if memory.filled else memory.memory_index
    avail = t_max - k
    try:
        batch, starts = episode_window_batch(memory, (k,), min(int(max_samples), avail))
    except ValueError as exc:
        raise ValueError(f"{exc} (t_max={t_max}, k={k}, max_samples={max_samples}).") from exc
    n_env = raw.shape[1]
    starts = starts[:avail - batch + 1, :n_env]
    if not starts.any():
        raise ValueError("No episode-contiguous action window for the selected batch")
    # ⚠ The uniform draw happens FIRST even when weighting, mirroring vicreg_tt, so the RNG
    # stream consumed per call does not depend on whether the rollout moved.
    env, t0 = draw_episode_window(starts, draw_single_start=avail >= max_samples)
    mean_turn = MASK_OFF
    if transit_weighted:
        span = batch + k - 1
        win = window_mean_abs_turn(actions, avail - batch + 1 + span - 1, n_env, span)
        weights = transit_weights(win, starts)
        total = float(weights.sum())
        if total > 0.0 and bool(torch.isfinite(weights).all()):
            flat = int(torch.multinomial(weights, 1).item())
            t0, env = divmod(flat, win.shape[1])
            mean_turn = float(win[t0, env])
        else:
            mean_turn = MASK_NO_MOTION

    def _take(tensor, lo, hi):
        out = tensor[lo:hi, env]
        return out if device is None else out.to(device)

    a = _take(actions, t0, t0 + batch + k - 1)                  # (B+k-1, A)
    return ActionWindow(
        env=int(env), t0=int(t0), batch=int(batch), k=k,
        obs_t=_take(raw, t0, t0 + batch),
        obs_tk=_take(raw, t0 + k, t0 + k + batch),
        actions=a.unfold(0, k, 1).permute(0, 2, 1),             # (B, k, A)
        mean_turn=mean_turn,
    )
