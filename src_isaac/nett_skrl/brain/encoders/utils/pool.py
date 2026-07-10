"""Deterministic replacement for ``nn.AdaptiveAvgPool2d``.

``adaptive_avg_pool2d``'s CUDA *backward* (``adaptive_avg_pool2d_backward_cuda``)
has no deterministic implementation — it accumulates with ``atomicAdd``, whose
float reduction order varies run-to-run. Under
``torch.use_deterministic_algorithms(True, warn_only=True)`` (the render path's
tolerance policy) it therefore runs SILENTLY and injects nondeterministic
gradients into every encoder, every step. This module gives the same forward
result via ops that ARE deterministic on CUDA:

  * output ``(1, 1)``      -> a global spatial mean (``x.mean(dim=(2,3))``),
  * fixed input divisible  -> an exact ``F.avg_pool2d`` (deterministic backward).

NETT runs fixed square resolutions, so the divisible branch always applies. A
non-divisible input RAISES rather than silently falling back to the
nondeterministic adaptive kernel (that would defeat the purpose).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class DeterministicAvgPool2d(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = (
            (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
        )

    def forward(self, x):
        oh, ow = self.output_size
        if oh == 1 and ow == 1:
            return x.mean(dim=(2, 3), keepdim=True)
        h, w = x.shape[-2], x.shape[-1]
        if h % oh == 0 and w % ow == 0:
            return F.avg_pool2d(x, kernel_size=(h // oh, w // ow))
        raise ValueError(
            f"DeterministicAvgPool2d: input {(h, w)} not divisible by output "
            f"{self.output_size}; pin a divisible input_resolution so pooling "
            "stays deterministic (adaptive_avg_pool2d backward is not)."
        )

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"
