"""Dual-stream GWM with the vendored trainable flow and quadratic reconstruction.

Separate parameters prevent shared-trunk coupling. They do NOT freeze flow:
the joint reconstruction objective differentiates through both streams, exactly
as in the vendored reference. Its globally constant-flow degeneracy is retained;
this variant makes no claim to remove it or to reproduce frozen-RAFT GWM.
"""

from contextlib import nullcontext

import torch

from .dual_stream import DualStreamAuxLoss
from .gwm_dual_loss import flow_reconstruction_loss


class GWMDualAuxLoss(DualStreamAuxLoss):
    def compute(self, encoder, observations):
        prev, curr = self.frames(encoder, observations)
        # Host AMP is disabled here: the reference QR solve requires float32.
        # No detached flow target or additional photometric objective is imposed.
        with torch.autocast(device_type=prev.device.type, enabled=False):
            masks = self.masks(encoder, curr)
            flow = self.head.dorsal.forward_single(prev, curr)
            # QR's batched matrix products also need the auxiliary exemption
            # during FORWARD on CUDA. PPO's split backward alone cannot cover it.
            from ...nett import _env_flag
            from ..skrl_patches import relaxed_determinism
            context = nullcontext() if _env_flag("NETT_AUX_STRICT") else relaxed_determinism()
            with context:
                loss = flow_reconstruction_loss(masks, flow)
        self.last_scalars = {
            "reconstruction": float(loss.detach()),
            "min_occupancy": float(masks.detach().mean((2, 3)).min()),
            "flow_absmax": float(flow.detach().abs().max()),
        }
        return loss
