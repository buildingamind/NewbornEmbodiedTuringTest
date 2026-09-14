"""Dual-stream EoO using the host ventral encoder and vendored unFlow loss.

Legacy `eoo` retains its historical equations. Here normalized mask-weighted
photometric reconstruction restores spatial objectness supervision, with joint
optimization of independent ventral and dorsal parameters. This does not guarantee
semantic objects: the reference can prefer easy-to-reconstruct regions.
"""

import torch

from .dual_stream import DualStreamAuxLoss
from .eoo_dual_loss import unflow_loss as objectness_loss


class EoODualAuxLoss(DualStreamAuxLoss):
    def compute(self, encoder, observations):
        prev, curr = self.frames(encoder, observations)
        # The host uses AMP; reference training uses float32. Keep both streams
        # and mask normalization in float32, including near-empty mask support.
        # Backward MUST use PPO's existing relaxed_determinism split: grid_sample
        # has no deterministic CUDA backward. A forward-only context cannot fix it.
        with torch.autocast(device_type=prev.device.type, enabled=False):
            mask = self.masks(encoder, curr)[:, 1:2]
            fwd, bwd = self.head.dorsal(prev, curr)
            loss = objectness_loss(prev, curr, mask, fwd, bwd)
        self.last_scalars = {
            "objectness": float(loss.detach()),
            "mask_mean": float(mask.detach().mean()),
            "mask_std": float(mask.detach().std()),
            "flow_absmax": float(torch.maximum(fwd.detach().abs().max(), bwd.detach().abs().max())),
        }
        return loss
