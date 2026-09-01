"""MoTokSeg — the Isaac equivalent of Unity's ``GwmSegWrapper``.

PORTED FROM, read at the file:
    scripts/wrappers/seg_wrappers.py:122-215   ``GwmSegWrapper``
    scripts/gwmEventBased1/trainParsing.py:328-400  ``GwmPPO_MoTokNet._train_ventral``

⛔★★★★★ WHY THIS IS A WRAPPER AND NOT AN AUXILIARY LOSS
------------------------------------------------------
The owner's high-conviction MoTok result (``_gwm_train_rA10_parsing_motok``, 8
brains at chick-level parsing) ran MoTok as a **separately-optimised network
whose foreground mask MULTIPLIES THE OBSERVATION** — ``seg_wrappers.py:213``,
``masked = obs * M_np``. The policy never shares a gradient with it.

That is a DIFFERENT HYPOTHESIS from an auxiliary loss. An aux loss asks *"does an
object-centric objective SHAPE THE REPRESENTATION usefully?"*; this asks *"does
handing the policy a SEGMENTED IMAGE help?"*. **The Unity evidence is for the
second**, so replicating the second is what replicates the result. An arm running
the aux-loss form is a different arm and must be named for what it tests.

⛔ THE GRADIENT ISOLATION IS THE PROPERTY THAT MAKES THE ARM FAITHFUL, so the
reference's ``RuntimeError`` at ``seg_wrappers.py:190`` is kept verbatim in
spirit: if the mask ever carries grad, this RAISES. A mask that quietly carried
grad would let PPO's return backpropagate into the segmenter and silently convert
this arm into a (differently-parameterised) aux-loss arm — the two hypotheses
would merge and neither would have been tested.

DIVERGENCES FROM THE REFERENCE, DECLARED RATHER THAN DISCOVERED
---------------------------------------------------------------
1. ⚠ **WHEN THE SEGMENTER TRAINS.** Unity trained it in ``_train_ventral`` at each
   PPO update, off the rollout buffer. Isaac exposes no such hook to a body
   wrapper, so this keeps a small internal ring buffer and trains every
   ``train_every`` observations. ``train_step()`` is also PUBLIC so a runner can
   drive it at true update boundaries later; the internal cadence is a
   stand-in, not a claim of equivalence. **The optimiser and its schedule are the
   part a reviewer should check first.**
2. ⚠ **LAYOUT.** Body wrappers run BEFORE ``ChannelsFirst``, so observations here
   are HWC ``(H,W,C)`` or batched ``(N,H,W,C)`` — the reference took CHW. Converted
   internally; the wrapper returns the layout it was given.
3. ⚠ **FRAMESTACK ORDER.** If this runs AFTER framestack the input carries
   ``C = 3*T`` channels. Each frame is then masked INDEPENDENTLY with its own
   mask, which is exactly equivalent to running this BEFORE framestack — not a
   guess, but it costs T forward passes. Declaring ``motok_seg`` before
   ``framestack`` is cheaper and is the recommended order.
4. ⚠ **NO ``num_queries`` SWEEP.** Fixed by env var, default 2 (fg/bg), matching
   ``gwm_num_queries: 2`` in the winning config.

⛔ SLOT 1 IS FOREGROUND BY CONVENTION ONLY
------------------------------------------
``MoTokNet.get_mask`` returns ``get_masks(frame)[:, 1:2]`` unconditionally
(``model.py:645``) and the reference does the same. **Nothing in a
reconstruction+VQ objective binds slot 1 to the object**, and with 2 slots and a
permutation-symmetric init which slot lands on the object is a COIN FLIP PER
SEED. This wrapper therefore offers ``NETT_SEG_FG_SLOT=auto`` (default), which
picks the slot by a stated, seed-independent rule and RECORDS which one it chose
in ``last_stats`` — so the choice is auditable instead of assumed. Any parsing
score computed downstream must still be permutation-invariant; see
``permutation_invariant_iou`` below.

CONFIG (env vars — the launcher constructs body wrappers with the env ALONE)
---------------------------------------------------------------------------
    NETT_SEG_MODEL        motok                  (only value implemented)
    NETT_SEG_FG_SLOT      auto | 0 | 1           (default auto; 'auto' = smaller-area slot)
    NETT_SEG_QUERIES      2                      gwm_num_queries
    NETT_SEG_UPSAMPLE     0                      gwm_upsample -- KEEP 0 on 128x80
    NETT_SEG_VQ_COEF      0.1                    gwm_vq_coef
    NETT_SEG_LR           1e-4                   gwm_lr
    NETT_SEG_WD           1e-4                   gwm_wd
    NETT_SEG_BATCH        8                      gwm_batch_size
    NETT_SEG_TRAIN_EVERY  64                     observations between train steps
    NETT_SEG_BUFFER       256                    ring-buffer capacity (frames)
    NETT_SEG_DEVICE       cuda | cpu             (default: cuda if available)
"""

from __future__ import annotations

import logging
import os

import gymnasium as gym
import numpy as np
import torch

logger = logging.getLogger("nett.body.motok_seg")


def permutation_invariant_iou(masks: torch.Tensor, gt: torch.Tensor) -> tuple[float, int]:
    """Best IoU over ALL slot->foreground assignments. Returns ``(iou, n)``.

    ⛔ THIS IS NOT A REFINEMENT, IT IS THE ONLY VALID FORM. Nothing binds slot 1
    to the object, so a metric that reads slot 1 measures which slot the seed
    happened to put the object in. Across seeds that returns a plausible mean
    with pure noise underneath — the same shape as every degenerate pass this
    campaign has found.

    Args:
        masks: (B, K, H, W) slot masks, K slots.
        gt:    (B, 1, H, W) or (B, H, W) binary ground-truth foreground.
    Returns:
        (mean best-IoU over the batch, B) — **report N with the number.**
    """
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)
    b, k = masks.shape[0], masks.shape[1]
    g = (gt > 0.5).float()
    best = torch.full((b,), -1.0, device=masks.device)
    for s in range(k):
        m = (masks[:, s : s + 1] > 0.5).float()
        inter = (m * g).flatten(1).sum(1)
        union = ((m + g) > 0).float().flatten(1).sum(1).clamp(min=1e-8)
        best = torch.maximum(best, inter / union)
    return float(best.mean()), int(b)


class MoTokSeg(gym.ObservationWrapper):
    """Multiply the observation by a learned foreground mask (see module docstring).

    The observation space is UNCHANGED — masking is elementwise and preserves
    shape and dtype, exactly as ``GwmSegWrapper`` did.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = env.observation_space   # masking preserves shape

        kind = os.environ.get("NETT_SEG_MODEL", "motok").strip().lower()
        if kind != "motok":
            raise ValueError(
                f"NETT_SEG_MODEL={kind!r} is not implemented. Only 'motok' exists. "
                "Refusing rather than silently running an unsegmented arm that "
                "would look like a segmentation result."
            )
        self.kind = kind
        self.num_queries = int(os.environ.get("NETT_SEG_QUERIES", "2"))
        self.upsample = int(os.environ.get("NETT_SEG_UPSAMPLE", "0"))
        self.vq_coef = float(os.environ.get("NETT_SEG_VQ_COEF", "0.1"))
        self.lr = float(os.environ.get("NETT_SEG_LR", "1e-4"))
        self.wd = float(os.environ.get("NETT_SEG_WD", "1e-4"))
        self.batch = int(os.environ.get("NETT_SEG_BATCH", "8"))
        self.train_every = int(os.environ.get("NETT_SEG_TRAIN_EVERY", "64"))
        self.buffer_cap = int(os.environ.get("NETT_SEG_BUFFER", "256"))

        fg = os.environ.get("NETT_SEG_FG_SLOT", "auto").strip().lower()
        if fg not in ("auto",) and not fg.isdigit():
            raise ValueError(f"NETT_SEG_FG_SLOT={fg!r} must be 'auto' or a slot index.")
        self.fg_slot = fg

        dev = os.environ.get("NETT_SEG_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)

        self._model = None
        self._optim = None
        self._buf: list[torch.Tensor] = []
        self._seen = 0
        # ⛔ ALWAYS EMITTED, including when nothing has trained yet. A statistic
        # that vanishes exactly when its condition occurs cannot be gated on.
        self.last_stats: dict[str, float] = {
            "seg/train_steps": 0.0,
            "seg/loss": float("nan"),
            "seg/fg_slot": -1.0,
            "seg/fg_area": float("nan"),
        }

    # ------------------------------------------------------------------
    def _ensure(self, in_ch: int) -> None:
        if self._model is not None:
            return
        from ...brain.aux.motok_aux import MoTokNet

        self._model = MoTokNet(
            num_queries=self.num_queries, upsample_size=self.upsample, in_ch=in_ch
        ).to(self.device).eval()
        self._optim = torch.optim.AdamW(
            self._model.parameters(), lr=self.lr, weight_decay=self.wd
        )
        # ⚠ NOTE THE DIVERGENCE, DELIBERATE: the reference put its never-trained
        # dorsal stream in this optimiser at a separate LR (trainParsing.py:145),
        # where AdamW's decoupled weight decay shrank 75,906 unused parameters
        # (55.5% of MoTokNet) toward zero. This port has no dorsal, so there is a
        # single param group and gwm_backbone_lr has no meaning here.
        logger.info(
            "motok_seg: MoTok on %s, %d queries, lr=%g wd=%g, train every %d obs",
            self.device, self.num_queries, self.lr, self.wd, self.train_every,
        )

    @staticmethod
    def _to_bchw(a: np.ndarray) -> tuple[torch.Tensor, bool]:
        """HWC or NHWC uint8 -> (B,C,H,W) float[0,1]. Returns (tensor, was_batched)."""
        if a.ndim == 3:
            t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0)
            return t.float().div_(255.0), False
        if a.ndim == 4:
            t = torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)
            return t.float().div_(255.0), True
        raise ValueError(f"MoTokSeg expects HWC or NHWC, got shape {a.shape}")

    def _pick_fg(self, masks: torch.Tensor) -> int:
        """Which slot is foreground.

        ``auto`` = the slot with the SMALLER mean area, on the stated assumption
        that the object occupies less of the frame than the background. ⚠ That
        assumption is a rule, not a fact — it is recorded in ``last_stats`` every
        step so a reader can check it rather than inherit it.
        """
        if self.fg_slot != "auto":
            return int(self.fg_slot)
        areas = masks.mean(dim=(0, 2, 3))          # (K,)
        return int(torch.argmin(areas).item())

    # ------------------------------------------------------------------
    def train_step(self) -> float | None:
        """One reconstruction+VQ step. PUBLIC so a runner can drive the cadence.

        Loss matches ``trainParsing.py:371`` exactly:
        ``F.mse_loss(recon, frame) + vq_coef * commit``.
        """
        import torch.nn.functional as F

        if self._model is None or len(self._buf) < max(2, self.batch):
            return None
        idx = torch.randperm(len(self._buf))[: self.batch]
        batch = torch.cat([self._buf[i] for i in idx.tolist()], dim=0).to(self.device)

        # ⛔⛔⛔ `enable_grad` IS LOAD-BEARING AND ITS ABSENCE WAS THE ARM'S BLOCKER.
        # This runs on the ROLLOUT path: skrl's sequential trainer wraps the whole
        # interaction block -- `agent.act` AND `self.env.step(actions)` -- in
        # `torch.no_grad()` (skrl/trainers/torch/sequential.py:90, stepping at :108).
        # `train_step` is reached from `_mask_one` inside that block, so without this
        # `loss.backward()` raises
        #     RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # at iteration 62 -- the step where `_seen` first reaches `train_every`, which is
        # why the arm always died at exactly 62/62500 on every node that tried it.
        # ⚠ THIS DOES NOT WEAKEN THE GRADIENT ISOLATION, and that is the property to check:
        #   1. `get_masks` keeps its OWN `torch.no_grad()` above, and the `masks.requires_grad`
        #      guard is untouched and still raises.
        #   2. The observation this wrapper returns is cast to uint8 numpy before it leaves,
        #      which severs any graph regardless of the ambient mode.
        #   3. The scope here is the SEGMENTER's own optimiser only -- the policy shares no
        #      parameter with it, so PPO's return still cannot reach the segmenter.
        # skrl uses `no_grad`, NOT `inference_mode`, so the buffered frames are ordinary
        # tensors and are legal autograd inputs here. Under `inference_mode` they would not
        # be, and the fix would have to clone at capture instead.
        with torch.enable_grad():
            self._model.train()
            self._optim.zero_grad(set_to_none=True)
            recon, commit = self._model.reconstruct(batch)
            loss = F.mse_loss(recon, batch) + self.vq_coef * commit
            if not torch.isfinite(loss):
                self._model.eval()
                logger.warning("motok_seg: non-finite loss, step skipped")
                return None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optim.step()
        self._model.eval()
        # ⛔ The reference asserts this rather than trusting it (trainParsing.py:384):
        # GroupNorm at batch=1 is unstable in train() mode and masking runs at B=1.
        if self._model.training:
            raise RuntimeError("motok_seg: model.eval() did not take effect after training.")
        self.last_stats["seg/train_steps"] += 1.0
        self.last_stats["seg/loss"] = float(loss.item())
        return float(loss.item())

    # ------------------------------------------------------------------
    def observation(self, obs):
        if isinstance(obs, dict):
            return {k: self._mask_one(v) if k == "policy" else v for k, v in obs.items()}
        return self._mask_one(obs)

    def _mask_one(self, obs):
        arr = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.asarray(obs)
        x, batched = self._to_bchw(arr)
        c = x.shape[1]

        # Framestack-after-segmentation: mask each frame independently. Exactly
        # equivalent to running this BEFORE framestack (see module docstring).
        if c > 3 and c % 3 == 0:
            frames = [x[:, i : i + 3] for i in range(0, c, 3)]
        else:
            frames = [x]

        self._ensure(frames[0].shape[1])
        out = []
        for f in frames:
            fd = f.to(self.device)
            with torch.no_grad():
                masks = self._model.get_masks(fd)                 # (B,K,H,W)
            # ⛔ THE ISOLATION GUARD, kept from seg_wrappers.py:190. If this ever
            # fires, PPO's return could backpropagate into the segmenter and this
            # arm would silently become an aux-loss arm.
            if masks.requires_grad:
                raise RuntimeError(
                    "MoTokSeg: mask carries requires_grad=True. The segmenter "
                    "must stay isolated from the policy gradient — that isolation "
                    "is what makes this the 'segmented image' hypothesis rather "
                    "than an auxiliary loss."
                )
            slot = self._pick_fg(masks)
            m = masks[:, slot : slot + 1]
            self.last_stats["seg/fg_slot"] = float(slot)
            self.last_stats["seg/fg_area"] = float(m.mean().item())
            out.append((fd * m).clamp(0.0, 1.0).cpu())

            if len(self._buf) < self.buffer_cap:
                self._buf.append(f.cpu())
            else:
                self._buf[self._seen % self.buffer_cap] = f.cpu()

        self._seen += 1
        if self.train_every > 0 and self._seen % self.train_every == 0:
            self.train_step()

        y = torch.cat(out, dim=1)
        y = (y * 255.0).round().clamp(0, 255).to(torch.uint8)
        y = y.permute(0, 2, 3, 1)                                  # back to NHWC
        if not batched:
            y = y[0]
        res = y.numpy()
        return torch.from_numpy(res) if isinstance(obs, torch.Tensor) else res
