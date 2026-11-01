"""`embdiv_unif` / `embdiv_cos2`: push the ENCODER OUTPUT apart in embedding space (owner 2026-11-01).

Owner request (presenter chat, 2026-11-01): two "push apart in embedding space" regularizers on the ViT's
encoder output, at weights lambda = 0.01 and 0.05. On where: "(b) encoder output" -- the features_dim vector
the policy and value heads read (``CompactViT.forward``: LayerNorm'd CLS -> ``head`` Linear), NOT a projector.
There is no head here: the term has NO parameters of its own, so its whole gradient lands on the encoder.

```
z_i  = encoder.encode_prepared(x_i)                 (B, features_dim), grad ON, the policy's own forward
zh_i = z_i / max(||z_i||_2, eps)                    eps = 1e-8 (F.normalize)
G    = zh zh^T                                      cosines; ||zh_i - zh_j||^2 = 2 - 2 G_ij

uniformity (Wang & Isola 2020, t = NETT_AUX_EMBDIV_T, default 2):
  L_unif = log( mean_{i != j} exp(-t ||zh_i - zh_j||^2) )
         = logsumexp_{i != j}( -t (2 - 2 G_ij) ) - log(B (B - 1))          in [-2tB/(B-1), 0]
cos2 (mean squared pairwise cosine):
  L_cos2 = mean_{i != j} G_ij^2                                            in [0, 1]
```

L enters PPO as ``aux_weight x L`` through AuxLossPPO (the label's aux_weight, 0.01; NETT_AUX_WEIGHT_OVERRIDE
for 0.05). Both losses are minimised by spreading the batch: L_unif at its uniform-on-the-sphere value, L_cos2 at
0 (an orthogonal batch, reachable only while B <= features_dim; at B = 64 << 512 it is).

ONE TERM, THE FORM FIXED BY THE AUX KEY, so a run's label (and its ``Loss / Aux (<key>) loss`` tag) names the form:
  ``embdiv_unif``  -> form "uniformity"     labels "ViT+UNIF"
  ``embdiv_cos2``  -> form "cos2"           labels "ViT+COS2"
``NETT_AUX_EMBDIV_FORM`` (uniformity | cos2) may restate the form; it REFUSES any other value and any value that
disagrees with the key. There is no env-only generic key: a required knob with no default would make the term
unbuildable under the registry sweep (tests/test_slot_contrast_aux.py), and a label whose form lived only in a
row's env would not tell the form.

WHERE THE IMAGES COME FROM: the PPO minibatch (``observations``), like ``mae`` / ``simclr`` -- a single-view term
needs no temporal pair -- subsampled WITHOUT replacement to ``NETT_AUX_EMBDIV_BATCH`` (default 64) by randperm.
Its own knob, NOT NETT_AUX_BATCH (that one sets cltt_ref's / simclr's B; this term must not move another term's
batch). The images are prepared once (``_prepare_image``, no grad) and re-encoded through ``encode_prepared``,
i.e. the encoder's normal ``forward`` (and so its NETT_AMP autocast; the output is float32). No augmentation:
the pairs are distinct frames of the rollout, not views.

ENCODERS: CompactViT only (``compact_vit``, the S3 "ViT"). Anything else REFUSES: the request is for the ViT,
and a different trunk under the same aux key would be a different arm.

NUMERICS. Computed in float32 whatever the autocast. Uniformity takes logsumexp over the off-diagonal entries
(the diagonal is masked to -inf, never indexed out), so it cannot underflow at large t and is exactly 0 for a
batch of identical points; squared distances are clamped at 0 against the rounding of 2 - 2G. cos2 masks the
diagonal to 0. Mean over i != j is the same as Wang & Isola's mean over i < j (torch.pdist): both are symmetric.

DETERMINISM. The only random draw is the randperm subsample (the global torch RNG, after PPO's own draws in the
minibatch, as mae / simclr). The ops are matmul, masked_fill, clamp, logsumexp, pow and sum, none of which is on
PyTorch's nondeterministic list, so the forward and backward run under use_deterministic_algorithms(True), and two
same-seed calls agree exactly (tests/test_embdiv_aux.py). masked_fill is used, not boolean indexing, so the
backward has no index_put.

DEGENERATE BATCH. Both losses average over the i != j pairs; at B = 1 there are none. ``NETT_AUX_EMBDIV_BATCH``
refuses < 2 at construction, and a minibatch of fewer than 2 rows refuses at compute.

KNOBS (each refuses junk; a stray NETT_AUX_EMBDIV_* on another label refuses in campaign_train):
* NETT_AUX_EMBDIV_FORM        uniformity | cos2; optional, must equal the key's form.
* NETT_AUX_EMBDIV_T           finite float > 0, default 2 (uniformity only; REFUSED on the cos2 key).
* NETT_AUX_EMBDIV_BATCH       integer >= 2, default 64.
* NETT_AUX_EMBDIV_DIAG_EVERY  integer >= 1, default 1: the collapse diagnostics below run on PPO updates
                              0, K, 2K, ...

TELEMETRY.
* Per minibatch (``last_scalars`` -> ``Loss / Aux <key> <name>`` in tfevents, averaged per PPO update by
  AuxLossPPO): ``loss`` (the UNWEIGHTED L) and ``B``.
* COLLAPSE DIAGNOSTICS, DURING TRAINING (owner 2026-11-01, so lambda can be adjusted before a run ends). Once per
  diag update, from AuxLossPPO's ``after_update`` hook, on the LAST minibatch's encoder-output batch of that update
  (a detached copy; no extra forward, no RNG draw, float64 on CPU, so nothing here touches the training graph):
    std_raw      mean over dimensions of the per-dimension batch std of z
    std_norm     the same for zh
    cos_mean     mean_{i != j} G_ij
    cos2_mean    mean_{i != j} G_ij^2
    unif_t2      L_unif at t = 2 (whatever NETT_AUX_EMBDIV_T is)
    erank        effective rank of z (RankMe, Garrido et al. 2023): exp(entropy of the singular values of the
                 raw, uncentred z normalised to sum 1); at most min(B, features_dim) = 64 at the default batch
    loss_raw     the update's mean UNWEIGHTED L over its minibatches; loss_weighted = aux_weight x loss_raw
  THE LEARNING SIGNAL, beside them in the same row (owner 2026-11-01: lambda is judged live from the CSV), from
  the values AuxLossPPO itself publishes for this update (passed to the hook as ``ppo``):
    policy_loss / value_loss / entropy_loss   ``Loss / Policy loss``, ``Loss / Value loss``, ``Loss / Entropy loss``
                                              (sum over the update's minibatches / (epochs x mini_batches), as
                                              tfevents; entropy_loss = -entropy_loss_scale x entropy, 0 if unscaled)
    ppo_total    policy_loss + value_loss + entropy_loss (the PPO loss WITHOUT the aux term)
    aux_share    |loss_weighted| / |ppo_total|   (NOT_MEASURED if ppo_total is exactly 0)
    reward_per_step        mean of the rollout's stored ``rewards`` (every env x timestep of this update's buffer)
    episode_return_recent  mean return of the episodes skrl has seen FINISH since its last tracking write (its
                           ``_track_rewards``, the source of tfevents ``Reward / Total reward (mean)``);
                           NOT_MEASURED when none finished in that window -- join on that tfevents key instead.
  emitted three ways: ``Loss / Aux <key> diag_<name>`` via track_data (tfevents; the collapse metrics and
  aux_share), one ``[NETT embdiv]`` stdout line (flushed, so a tail of the run log shows it live), and one row
  appended to ``<agent.experiment_dir>/embdiv_diag.csv`` (columns DIAG_COLUMNS; header written when the file is
  new). The update counter is this process's: a resumed run restarts it at 0, the ``timestep`` column does not.
  ⚠ Diagnostic only: nothing here changes the loss or stops a run.

SAMPLING ACROSS THE ROLLOUT. The minibatch is skrl RandomMemory's (all three NETT buffer classes inherit its
``sample``): ``randperm`` over the FLATTENED (timesteps x envs) buffer without replacement, split into
mini_batches chunks -- a uniformly random, duplicate-free draw across every env and timestep, from the global torch
RNG seeded per brain (agent_factory). This term's randperm subsample of that chunk is again uniform and
duplicate-free, so the B frames are a random subsample of the whole rollout, never contiguous frames of one env
(tests/test_embdiv_aux.py checks it through the production agent).
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.compact_vit import CompactViT
from .knobs import NOT_MEASURED, _env_choice, _env_positive_float, _env_positive_int

FORMS = ("uniformity", "cos2")
FORM_ENV = "NETT_AUX_EMBDIV_FORM"
T_ENV = "NETT_AUX_EMBDIV_T"
BATCH_ENV = "NETT_AUX_EMBDIV_BATCH"
DIAG_EVERY_ENV = "NETT_AUX_EMBDIV_DIAG_EVERY"
#: aux key -> form. The ONLY two keys; ppo_aux.AUX_LOSSES registers exactly these.
KEY_FORM = {"embdiv_unif": "uniformity", "embdiv_cos2": "cos2"}
NORM_EPS = 1e-8
DIAG_CSV = "embdiv_diag.csv"
DIAG_METRICS = ("std_raw", "std_norm", "cos_mean", "cos2_mean", "unif_t2", "erank", "loss_raw", "loss_weighted")
#: The learning signal written beside the collapse metrics (module docstring).
SIGNAL_METRICS = ("policy_loss", "value_loss", "entropy_loss", "ppo_total", "aux_share", "reward_per_step",
                  "episode_return_recent")
DIAG_COLUMNS = ("timestep", "update", "form", "t", "aux_weight", "B", *DIAG_METRICS, *SIGNAL_METRICS)


def _require_pairs(b: int) -> None:
    if b < 2:
        raise ValueError(f"embdiv: a batch of {b} has no i != j pairs; the loss would be a mean over nothing.")


def _gram(z: torch.Tensor) -> torch.Tensor:
    """(B, D) -> (B, B) cosines of the L2-normalised rows (F.normalize, eps NORM_EPS)."""
    zh = F.normalize(z, dim=-1, eps=NORM_EPS)
    return zh @ zh.transpose(0, 1)


def _diagonal(b: int, device) -> torch.Tensor:
    """(B, B) bool, True on the diagonal: the i == j entries every loss here masks out."""
    return torch.eye(b, dtype=torch.bool, device=device)


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """L_unif = log mean_{i != j} exp(-t ||zh_i - zh_j||^2), by logsumexp over the masked Gram matrix."""
    b = z.shape[0]
    _require_pairs(b)
    g = _gram(z)
    sq = (2.0 - 2.0 * g).clamp_min(0.0)
    logits = (-float(t) * sq).masked_fill(_diagonal(b, z.device), float("-inf"))
    return torch.logsumexp(logits.flatten(), dim=0) - math.log(b * (b - 1))


def cos2_loss(z: torch.Tensor) -> torch.Tensor:
    """L_cos2 = mean_{i != j} cos(z_i, z_j)^2."""
    b = z.shape[0]
    _require_pairs(b)
    g2 = _gram(z).pow(2).masked_fill(_diagonal(b, z.device), 0.0)
    return g2.sum() / (b * (b - 1))


def effective_rank(z: torch.Tensor) -> float:
    """RankMe: exp(entropy of the singular values of z normalised to sum 1). 1.0 for a rank-1 batch."""
    s = torch.linalg.svdvals(z)
    total = float(s.sum())
    if not total > 0.0:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def embedding_diagnostics(z: torch.Tensor) -> dict:
    """The collapse diagnostics of one encoder-output batch, float64 on CPU (keys: DIAG_METRICS[:6])."""
    z = z.detach().to("cpu", torch.float64)
    b = z.shape[0]
    _require_pairs(b)
    zh = F.normalize(z, dim=-1, eps=NORM_EPS)
    g = zh @ zh.T
    off = ~_diagonal(b, z.device)
    pairs = float(b * (b - 1))
    return {
        "std_raw": float(z.std(dim=0).mean()),
        "std_norm": float(zh.std(dim=0).mean()),
        "cos_mean": float((g * off).sum() / pairs),
        "cos2_mean": float((g.pow(2) * off).sum() / pairs),
        "unif_t2": float(uniformity_loss(z, 2.0)),
        "erank": effective_rank(z),
    }


def _key_form(key: str) -> str:
    if key not in KEY_FORM:
        raise ValueError(f"embdiv: unknown aux key {key!r}; registered: {sorted(KEY_FORM)}.")
    form = KEY_FORM[key]
    stated = _env_choice(FORM_ENV, form, FORMS)
    if stated != form:
        raise ValueError(
            f"{FORM_ENV}={stated!r} but the aux key {key!r} is the {form!r} form. The label names the form; "
            f"refusing to train one form under the other's label. Unset {FORM_ENV} or use the matching label.")
    return form


class EmbDivTerm(nn.Module):
    """Uniformity or mean-squared-cosine on the encoder output of a PPO-minibatch subsample (module docstring)."""

    needs_memory = False

    def __init__(self, encoder: nn.Module, key: str) -> None:
        super().__init__()
        if type(encoder) is not CompactViT:
            raise TypeError(
                f"EmbDivTerm supports CompactViT (the S3 'ViT') only; got {type(encoder).__name__}. Refusing to "
                f"run the embedding-spread term on another trunk under the same aux key.")
        self.key = key
        self.form = _key_form(key)
        if self.form == "uniformity":
            self.t = _env_positive_float(T_ENV, 2.0)
        else:
            if os.environ.get(T_ENV) is not None:
                raise ValueError(f"{T_ENV} is set but the aux key {key!r} is the cos2 form, which has no "
                                 f"temperature; nothing would read it. Unset it.")
            self.t = None
        self.max_samples = _env_positive_int(BATCH_ENV, 64)
        if self.max_samples < 2:
            raise ValueError(f"{BATCH_ENV}={self.max_samples}: the loss averages over i != j pairs; use >= 2.")
        self.diag_every = _env_positive_int(DIAG_EVERY_ENV, 1)
        #: AuxLossPPO adds ``head`` to PPO's optimizer and its grad clip: EMPTY, the term has no parameters.
        self.head = nn.Module()
        self.last_scalars: dict = {}
        self._updates = 0
        self._diag_z: torch.Tensor | None = None
        self._loss_sum = 0.0
        self._loss_n = 0

    def loss(self, z: torch.Tensor) -> torch.Tensor:
        with torch.autocast(z.device.type, enabled=False):   # fp32 Gram even inside AuxLossPPO's autocast
            z = z.float()
            return uniformity_loss(z, self.t) if self.form == "uniformity" else cos2_loss(z)

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            prepared = encoder._prepare_image(observations)          # (B, C*T, H, W) float [0, 1]
            if prepared.shape[0] > self.max_samples:
                idx = torch.randperm(prepared.shape[0], device=prepared.device)[:self.max_samples]
                prepared = prepared[idx]
        _require_pairs(prepared.shape[0])
        z = encoder.encode_prepared(prepared)                         # (B, features_dim), grad ON
        loss = self.loss(z)
        value = float(loss.detach())
        self._diag_z = z.detach()
        self._loss_sum += value
        self._loss_n += 1
        self.last_scalars = {"loss": value, "B": float(z.shape[0])}
        return loss

    # ---------------------------------------------------------------- collapse diagnostics
    def after_update(self, agent, *, timestep: int, timesteps: int, ppo: dict | None = None) -> dict | None:
        """AuxLossPPO calls this once at the end of every PPO update. On diag updates: emit + append a CSV row.

        ``ppo``: this update's {"policy_loss", "value_loss", "entropy_loss"} as AuxLossPPO tracks them."""
        u = self._updates
        self._updates += 1
        z, loss_sum, loss_n = self._diag_z, self._loss_sum, self._loss_n
        self._diag_z, self._loss_sum, self._loss_n = None, 0.0, 0
        if u % self.diag_every:
            return None
        weight = float(getattr(agent, "_aux_weight", float("nan")))
        if z is None or z.shape[0] < 2:
            # Every minibatch of this update hit the KL early stop before the aux ran: nothing was measured.
            row = {k: float(NOT_MEASURED) for k in DIAG_METRICS}
            b = 0
        else:
            row = embedding_diagnostics(z)
            row["loss_raw"] = loss_sum / loss_n
            row["loss_weighted"] = weight * row["loss_raw"]
            b = int(z.shape[0])
        row.update(_learning_signal(agent, ppo, row["loss_weighted"]))
        kind = getattr(agent, "_aux_kind", self.key)
        for name in (*DIAG_METRICS, "aux_share"):
            agent.track_data(f"Loss / Aux {kind} diag_{name}", row[name])
        exp_dir = getattr(agent, "experiment_dir", None)
        if not exp_dir:
            raise ValueError("embdiv: the agent has no experiment_dir, so embdiv_diag.csv has nowhere to go.")
        print(f"[NETT embdiv] {Path(exp_dir).name} update {u} timestep {timestep} {kind} B {b} "
              + " ".join(f"{k} {row[k]:.6g}" for k in (*DIAG_METRICS, *SIGNAL_METRICS)), flush=True)
        path = Path(exp_dir) / DIAG_CSV
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(DIAG_COLUMNS)
            w.writerow([int(timestep), u, self.form, "" if self.t is None else self.t, weight, b,
                        *(row[k] for k in (*DIAG_METRICS, *SIGNAL_METRICS))])
        return row


def _learning_signal(agent, ppo: dict | None, loss_weighted: float) -> dict:
    """SIGNAL_METRICS for one update (module docstring); NOT_MEASURED where a value cannot be read."""
    out = {k: float(NOT_MEASURED) for k in SIGNAL_METRICS}
    if ppo:
        for k in ("policy_loss", "value_loss", "entropy_loss"):
            out[k] = float(ppo[k])
        out["ppo_total"] = out["policy_loss"] + out["value_loss"] + out["entropy_loss"]
        if out["ppo_total"] != 0.0 and loss_weighted != float(NOT_MEASURED):
            out["aux_share"] = abs(loss_weighted) / abs(out["ppo_total"])
    memory = getattr(agent, "memory", None)
    if memory is not None:
        rewards = memory.get_tensor_by_name("rewards")
        out["reward_per_step"] = float(rewards.float().mean())
    finished = getattr(agent, "_track_rewards", None)
    if finished:
        out["episode_return_recent"] = float(sum(finished) / len(finished))
    return out


def build(key: str):
    """The AUX_LOSSES builder for one key."""
    def _build(encoder):
        return EmbDivTerm(encoder, key)
    return _build
