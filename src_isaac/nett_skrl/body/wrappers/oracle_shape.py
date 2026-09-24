"""OracleShapeGate: the red-object oracle mask, with each object GATED by global-shape familiarity.

⛔ DIAGNOSTIC CONTROL, NOT A MODEL, NEVER A CANDIDATE. Both stages use knowledge of the stimulus:
the mask is the fixed red-object colour rule of ``oracle_seg.py``, and the shape feature
(solidity = area / convex-hull area of a connected component) was picked offline because it
separates THIS object pair across views (FINDINGS §4dh.43j.1, E16: fork .56-.61, blob .89-.90 at
pose 00/30/60; appearance codes learned from the imprint clip do not, §4dh.43i.4).

What it asks. With a perfect mask the policy is a geometry chooser with no familiarity preference
(§4dh.43j): single-object imprinting with the content-blind approach reward teaches "approach red
mass". This wrapper changes WHAT MASS the policy sees, not the reward: each component is scaled by

    g = exp(-0.5 * ((solidity - template) / sigma) ** 2)

where ``template`` is the running mean solidity of the largest measurable component seen during the
TRAIN phase (the imprint object -- the White monitor is masked out) and is frozen for test. During
imprinting g ~= 1 on the only target IN THE CLIPS; at test a component
unlike the imprint is attenuated, so "approach red mass" becomes "approach familiar mass" by
construction. The gate is SOFT on purpose: a hard gate would make every test trial a Rest trial and
say nothing about how much familiarity signal the policy needs.

⚠ MEASURED 2026-09-24 (chicken probes, 256 train eps, FINDINGS §4dh.43j.5): in Isaac the train-phase
gate is NOT neutral -- seg/gate_min .02-.37 with one object on screen -- so training does not match
CNN2F+ORACLE-RedSeg. The pre-gate stats below (gate_raw_area, gate_kept_frac, gate_largest) exist
to measure by how much.

Components too small to measure (bounding box under ``NETT_SHAPE_GATE_MIN_PX`` on its long side;
solidity collapses toward 1 below ~10 px, §4dh.43j.2 E17a) get the NEUTRAL gate ``g = 1``, so a
distant object is neither favoured nor penalised by an unmeasurable shape.

Knobs (defaults are the registered arm's): NETT_SHAPE_GATE_SIGMA (0.15), NETT_SHAPE_GATE_MIN_PX (10).
"""

from __future__ import annotations

import math
import os

import cv2
import numpy as np
import torch
from torch import nn

from .oracle_seg import OracleColorSeg, red_object_mask


def component_solidity(mask: np.ndarray, min_px: int):
    """uint8 {0,1} HxW -> (labels HxW int32, {label: solidity or None if unmeasurable})."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = {}
    for k in range(1, n):
        side = max(int(st[k, cv2.CC_STAT_WIDTH]), int(st[k, cv2.CC_STAT_HEIGHT]))
        if side < min_px:
            out[k] = None
            continue
        comp = (lab == k).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cs, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(cnt))
        # contour area / hull area: the EXACT definition E16/E17a validated (notes/researcher/
        # offline-seg/e16.py, e17a.py). A different area convention would not carry that evidence.
        out[k] = float(cv2.contourArea(cnt) / hull) if hull > 0 else None
    return lab, out


class _ShapeGateModel(nn.Module):
    def __init__(self, sigma: float, min_px: int):
        super().__init__()
        self.sigma = float(sigma)
        self.min_px = int(min_px)
        self.learning = True
        # Saved with the segmenter state, so the test phase gates against the TRAIN-phase template.
        self.register_buffer("rule_version", torch.tensor(1))
        self.register_buffer("template_sum", torch.tensor(0.0, dtype=torch.float64))
        # counts FRAMES x ENVS measured (after framestack each stacked frame counts), not env steps
        self.register_buffer("template_n", torch.tensor(0, dtype=torch.int64))
        self.last = {"n_components": 0.0, "n_measured": 0.0, "gate_min": float("nan"), "gate_mean": float("nan"),
                     "gate_raw_area": float("nan"), "gate_kept_frac": float("nan"), "gate_largest": float("nan")}

    @property
    def template(self) -> float:
        n = int(self.template_n.item())
        return float(self.template_sum.item()) / n if n else float("nan")

    def get_masks(self, frame):
        fg = red_object_mask(frame)                                   # (B,1,H,W) {0,1}
        gate = torch.zeros_like(fg)
        t = self.template
        ncomp = nmeas = 0
        gates = []
        largest = []
        for b in range(fg.shape[0]):
            m = fg[b, 0].cpu().numpy().astype(np.uint8)
            lab, sol = component_solidity(m, self.min_px)
            g = np.zeros(m.shape, np.float32)
            measured = [(k, s) for k, s in sol.items() if s is not None]
            if self.learning and measured:
                # the imprint object = the largest measurable component (the White monitor is masked out)
                k_big = max(measured, key=lambda ks: int((lab == ks[0]).sum()))[0]
                self.template_sum += sol[k_big]
                self.template_n += 1
            k_big_any = max(measured, key=lambda ks: int((lab == ks[0]).sum()))[0] if measured else None
            for k, s in sol.items():
                if s is None or math.isnan(t):
                    gk = 1.0                                          # unmeasurable, or no template yet
                else:
                    gk = math.exp(-0.5 * ((s - t) / self.sigma) ** 2)
                    nmeas += 1
                g[lab == k] = gk
                if k == k_big_any:
                    largest.append(gk)
                gates.append(gk)
                ncomp += 1
            gate[b, 0] = torch.from_numpy(g).to(gate.device)
        self.last = {
            "n_components": float(ncomp), "n_measured": float(nmeas),
            "gate_min": float(min(gates)) if gates else float("nan"),
            "gate_mean": float(np.mean(gates)) if gates else float("nan"),
        }
        kept = fg * gate
        raw = float(fg.sum().item())
        # PRE-gate: fg_area/kept_area downstream are both measured AFTER the gate
        self.last["gate_raw_area"] = float(fg.mean().item())
        self.last["gate_kept_frac"] = float(kept.sum().item()) / raw if raw > 0 else float("nan")
        self.last["gate_largest"] = float(np.mean(largest)) if largest else float("nan")
        return torch.cat((kept, 1.0 - kept), dim=1)


class OracleShapeGate(OracleColorSeg):
    """Oracle red mask with a soft per-component global-shape familiarity gate. DIAGNOSTIC."""

    def _configure(self):
        self.kind = "oracle_shape"
        self.num_queries = 2
        self.gate_sigma = float(os.environ.get("NETT_SHAPE_GATE_SIGMA", "0.15"))
        self.gate_min_px = int(os.environ.get("NETT_SHAPE_GATE_MIN_PX", "10"))
        if not (self.gate_sigma > 0 and self.gate_min_px >= 3):
            raise ValueError(
                f"NETT_SHAPE_GATE_SIGMA={self.gate_sigma} must be > 0 and "
                f"NETT_SHAPE_GATE_MIN_PX={self.gate_min_px} must be >= 3 (solidity is undefined below).")
        self._learning = True

    def bind_phase(self, phase, state_path, *, resume=False, allow_missing=False):
        self._learning = phase == "train"
        super().bind_phase(phase, state_path, resume=resume, allow_missing=allow_missing)
        if self._model is not None:
            self._model.learning = self._learning

    def _apply_pending_state(self):
        super()._apply_pending_state()
        # ⛔ A test phase against an EMPTY template is silently the oracle (every gate = 1): the arm
        # would read as "the gate did not help" when the gate never fired. Refuse it, with the
        # base class's escape. A probe that ends here learned no template in train -- that is a
        # finding (the object never reached NETT_SHAPE_GATE_MIN_PX), not noise.
        if (not self._learning and int(self._model.template_n.item()) == 0
                and os.environ.get("NETT_SEG_ALLOW_UNTRAINED", "0") != "1"):
            raise RuntimeError(
                "OracleShapeGate: the loaded state holds no shape template (template_n == 0): no "
                f"component reached NETT_SHAPE_GATE_MIN_PX={self.gate_min_px} in train, so the "
                "gate would pass everything and this arm would be ORACLE-RedSeg. "
                "NETT_SEG_ALLOW_UNTRAINED=1 runs it anyway on purpose.")

    def _ensure(self, in_ch):
        if self._model is None:
            self._model = _ShapeGateModel(self.gate_sigma, self.gate_min_px).to(self.device).eval()
            self._model.learning = self._learning

    def _keep_mask(self, masks):
        m = super()._keep_mask(masks)
        # ⛔ ALWAYS EMITTED (see SegmentationObservationWrapper.last_stats): a template that never
        # formed must read as NaN, not vanish.
        self.last_stats["seg/gate_template"] = self._model.template
        self.last_stats["seg/gate_template_n"] = float(self._model.template_n.item())
        # the RESOLVED knobs, so a train/test mismatch is readable from the run's own stats
        self.last_stats["seg/gate_sigma"] = self._model.sigma
        self.last_stats["seg/gate_min_px"] = float(self._model.min_px)
        for k, v in self._model.last.items():
            self.last_stats[f"seg/gate_{k}" if not k.startswith("gate_") else f"seg/{k}"] = v
        return m
