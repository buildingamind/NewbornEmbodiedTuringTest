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

NETT's eye is NON-SQUARE as of 2026-08-02 (128x80), so do not assume the
divisible branch always applies -- it did when the sensor was square. A
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


def pool_grid_for(feat_h: int, feat_w: int, cells: int = 16) -> tuple[int, int]:
    """A grid that DIVIDES ``(feat_h, feat_w)`` and holds about ``cells`` cells.

    ``DeterministicAvgPool2d`` refuses a ragged grid (see above) rather than
    falling back to the nondeterministic adaptive kernel. Every encoder here used
    to hardcode ``(4, 4)``, which divides a square feature map and nothing else --
    so at the 128x80 eye ``nature_cnn``, ``compact_cnn``, ``small_cnn`` and
    ``simclr_cltt`` all RAISED AT CONSTRUCTION, before a single step. An arm that
    dies at construction loses its slot, not its run: no partial result, no log,
    and nothing that looks like a scientific negative.

    ⚠ THE QUANTITY TO CONSERVE IS THE CELL COUNT, NOT THE PER-AXIS DIVISOR.
    ``cells`` sets the flatten width, which sets the dominant Linear's parameters,
    which is what ``campaign_train.py`` holds near ~700K across every arm -- and
    that parity is the whole licence for reading a between-arm difference as an
    ARCHITECTURE effect rather than a capacity effect. Choosing each axis
    independently against a constant 4 does not conserve the product: on the eye
    it sends ``simclr_cltt`` to 20 cells (+24% params) and ``nature_cnn`` to 12
    (-22%) -- two comparators drifting APART, which is exactly what the budget
    rule exists to prevent.

    So: minimise ``|h*w - cells|`` first, break ties toward the grid whose aspect
    best matches the feature map's. On a square map this returns the historical
    ``(4, 4)`` exactly, so square-resolution arms are bit-identical to before.

    ★ It also happens to favour horizontal resolution on a wide sensor -- (2, 8)
    rather than (4, 4) at 128x80 -- which is the axis the parsing endpoint reads,
    the scorer being ``sign(agent.x)``.
    """
    def _divisors(n: int) -> list[int]:
        return [d for d in range(1, n + 1) if n % d == 0]

    aspect = feat_h / feat_w
    return min(((a, b) for a in _divisors(feat_h) for b in _divisors(feat_w)),
               key=lambda g: (abs(g[0] * g[1] - cells), abs(g[0] / g[1] - aspect)))
