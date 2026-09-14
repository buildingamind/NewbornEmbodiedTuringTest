"""Shared scaffolding for separately optimized observation segmenters.

Only the perception optimizer sees reconstruction gradients. Masks are inferred
under no_grad and checked before multiplying raw observations. Subclasses define
the model, objective, and whether buffered samples are frames or frame pairs.
"""

from __future__ import annotations

import logging
import os

import gymnasium as gym
import numpy as np
import torch


class SegmentationObservationWrapper(gym.ObservationWrapper):
    """Mask observations without coupling the perception model to the policy."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = env.observation_space   # masking preserves shape

        self._configure()
        self.lr = float(os.environ.get("NETT_SEG_LR", "1e-4"))
        self.wd = float(os.environ.get("NETT_SEG_WD", "1e-4"))
        self.batch = int(os.environ.get("NETT_SEG_BATCH", "8"))
        self.train_every = int(os.environ.get("NETT_SEG_TRAIN_EVERY", "64"))
        self.buffer_cap = int(os.environ.get("NETT_SEG_BUFFER", "256"))

        fg = os.environ.get("NETT_SEG_FG_SLOT", "auto").strip().lower()
        if fg not in ("auto",) and not fg.isdigit():
            raise ValueError(f"NETT_SEG_FG_SLOT={fg!r} must be 'auto' or a slot index.")
        self.fg_slot = fg
        rule = os.environ.get("NETT_SEG_MASK_RULE", "auto").strip().lower()
        if rule not in {"auto", "foreground", "not_background"}:
            raise ValueError(
                f"NETT_SEG_MASK_RULE={rule!r} must be 'auto', 'foreground' or "
                "'not_background'. 'auto' keeps the foreground slot at K<=2 and "
                "suppresses only the background slot at K>2 -- see _keep_mask for why "
                "keeping one slot at K>2 can delete a test alternative."
            )
        self.mask_rule = rule

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

    def _configure(self) -> None:
        raise NotImplementedError

    def _ensure(self, in_ch: int) -> None:
        raise NotImplementedError

    def _loss(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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

    def _keep_mask(self, masks: torch.Tensor) -> torch.Tensor:
        """Which pixels SURVIVE into the observation. Returns (B,1,H,W) in [0,1].

        ⛔ THE RULE MUST CHANGE WITH THE SLOT COUNT, AND THE REASON IS THE TEST, NOT
        THE TRAINING. With K=2 we keep the foreground slot and suppress the chamber,
        which is the reference behaviour. With K>2 keeping ONE slot would suppress
        every other slot -- and the parsing test shows TWO objects on two monitors.
        If they land in different slots, keeping one DELETES ONE OF THE TWO
        ALTERNATIVES and the agent is shown a single option in a two-alternative
        forced choice. The arm would score at chance for a reason that has nothing
        to do with the hypothesis.

        So for K>2 we instead suppress the BACKGROUND slot and keep everything else:
        because the slots are a softmax partition, ``1 - M_bg`` is exactly the sum of
        all non-background slots. Background is taken as the LARGEST-area slot -- the
        dual of the foreground rule above, seed-independent, and RECORDED every step
        so a reader can check the assumption rather than inherit it.
        """
        areas = masks.mean(dim=(0, 2, 3))                      # (K,)
        rule = self.mask_rule
        if rule == "auto":
            rule = "foreground" if masks.shape[1] <= 2 else "not_background"
        if rule == "foreground":
            slot = self._pick_fg(masks)
            m = masks[:, slot : slot + 1]
        elif rule == "not_background":
            slot = int(torch.argmax(areas).item())
            m = 1.0 - masks[:, slot : slot + 1]
        else:
            raise ValueError(
                f"NETT_SEG_MASK_RULE={rule!r} unknown; expected "
                "'auto', 'foreground' or 'not_background'."
            )
        # ⛔ EMIT THE LEGACY KEYS UNCONDITIONALLY. Renaming them out from under the
        # existing tests broke 14 cases including MoTok's acceptance test, whose
        # logged stats ARE part of the behaviour the spec said must not change. A new
        # rule adds fields; it does not repurpose the ones already being read.
        fg = self._pick_fg(masks)
        self.last_stats["seg/fg_slot"] = float(fg)
        self.last_stats["seg/fg_area"] = float(masks[:, fg : fg + 1].mean().item())
        self.last_stats["seg/mask_rule_not_background"] = float(rule == "not_background")
        self.last_stats["seg/selected_slot"] = float(slot)
        self.last_stats["seg/kept_area"] = float(m.mean().item())
        return m

    def slot_diagnostics(self, masks: torch.Tensor) -> dict:
        """Is the partition DECIDING anything, or just diluting across slots?

        MEASURED 2026-09-14 on real frames at 150 steps: no slot ever dies, but the
        fraction of pixels confidently assigned to ANY slot falls 0.79 (K=2) -> 0.24
        (K=3) -> 0.18 (K=5), while the closest slot PAIR rises to 0.906 cosine at
        K=5 -- near-duplicates. Neither is visible in the loss value. An arm that
        dilutes must be readable as diluted from its own logs.
        """
        with torch.no_grad():
            occ = masks.mean(dim=(0, 2, 3))
            conf = float((masks.max(dim=1).values > 0.6).float().mean().item())
            k = masks.shape[1]
            f = masks.permute(1, 0, 2, 3).reshape(k, -1)
            f = f / f.norm(dim=1, keepdim=True).clamp(min=1e-8)
            sim = (f @ f.t()).fill_diagonal_(-1.0)
            return {
                "seg/confident_pixels": conf,
                "seg/slot_pair_cosine_max": float(sim.max().item()),
                "seg/slot_occ_min": float(occ.min().item()),
                "seg/slot_occ_max": float(occ.max().item()),
            }

    def _frames_and_samples(self, x):
        c = x.shape[1]
        # Framestack-after-segmentation: mask each frame independently. Exactly
        # equivalent to running this BEFORE framestack (see module docstring).
        if c > 3 and c % 3 == 0:
            frames = [x[:, i : i + 3] for i in range(0, c, 3)]
        else:
            frames = [x]

        return frames, frames

    def _store_sample(self, sample):
        if len(self._buf) < self.buffer_cap:
            self._buf.append(sample.cpu())
        else:
            self._buf[self._seen % self.buffer_cap] = sample.cpu()

    def train_step(self) -> float | None:
        """One isolated perception update; also callable at runner update boundaries."""

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
            loss = self._loss(batch)
            if not torch.isfinite(loss):
                self._model.eval()
                logging.getLogger(f"nett.body.{self.kind}_seg").warning(
                    "%s_seg: non-finite loss, step skipped", self.kind
                )
                return None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
            self._optim.step()
        self._model.eval()
        # ⛔ The reference asserts this rather than trusting it (trainParsing.py:384):
        # GroupNorm at batch=1 is unstable in train() mode and masking runs at B=1.
        if self._model.training:
            raise RuntimeError(f"{self.kind}_seg: model.eval() did not take effect after training.")
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

        frames, samples = self._frames_and_samples(x)

        self._ensure(frames[0].shape[1])
        out = []
        for f, sample in zip(frames, samples):
            fd = f.to(self.device)
            with torch.no_grad():
                masks = self._model.get_masks(fd)                 # (B,K,H,W)
            # ⛔ THE ISOLATION GUARD, kept from seg_wrappers.py:190. If this ever
            # fires, PPO's return could backpropagate into the segmenter and this
            # arm would silently become an aux-loss arm.
            if masks.requires_grad:
                raise RuntimeError(
                    f"{type(self).__name__}: mask carries requires_grad=True. The segmenter "
                    "must stay isolated from the policy gradient — that isolation "
                    "is what makes this the 'segmented image' hypothesis rather "
                    "than an auxiliary loss."
                )
            m = self._keep_mask(masks)
            out.append((fd * m).clamp(0.0, 1.0).cpu())

            if sample is not None:
                self._store_sample(sample)

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
