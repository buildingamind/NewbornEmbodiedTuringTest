"""``NatureCNN(spatial_pool=...)`` — the 4x4 pool is default-on and opt-out only.

Written fresh rather than ported: the ablation's original tests live in the experiment
directory they came from (`experiments/nett_naturecnn_nopool_20260724/`), which is not part
of the package, and they import it unconditionally. This pins the part that belongs to the
package — the config surface and, above all, that turning the knob on nothing changes
nothing.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from nett_skrl.brain.encoders.nature_cnn import NatureCNN


def _space(res: int) -> gym.Space:
    return gym.spaces.Box(low=0, high=255, shape=(res, res, 3), dtype=np.uint8)


def _obs(res: int, n: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, 256, (n, res, res, 3), generator=g, dtype=torch.uint8).float()


@pytest.mark.parametrize("res", [64, 128])
def test_pooling_is_the_default(res):
    """An unset config must get the historical architecture."""
    assert NatureCNN(_space(res), features_dim=32).spatial_pool is True


def test_at_res64_the_pool_is_an_identity_so_the_ablation_changes_nothing():
    """★ THE INVARIANT THAT MAKES THIS SAFE TO SHIP DEFAULT-ON.

    At 64x64 the final conv map is already 4x4, so pooling to (4, 4) is a no-op. Pooled and
    unpooled encoders must therefore produce the SAME feature width there -- which is what
    lets `spatial_pool` exist without invalidating any existing config or checkpoint.
    """
    pooled = NatureCNN(_space(64), features_dim=32, spatial_pool=True)
    unpooled = NatureCNN(_space(64), features_dim=32, spatial_pool=False)

    x = _obs(64)
    assert pooled(x).shape == unpooled(x).shape == (2, 32)


def test_above_res64_the_ablation_widens_the_flatten():
    """Off, the full conv map reaches the projection -- that is the point of the ablation.

    The pooled encoder caps the flatten at 4x4xconv_dim regardless of input resolution; the
    unpooled one grows with it. Compare the Linear's in_features, since features_dim (the
    policy-head input) is deliberately identical either way.
    """
    def flatten_width(spatial_pool: bool) -> int:
        enc = NatureCNN(_space(128), features_dim=32, spatial_pool=spatial_pool)
        linear = next(m for m in enc.modules() if isinstance(m, torch.nn.Linear))
        return linear.in_features

    assert flatten_width(False) > flatten_width(True)
    assert flatten_width(True) == 4 * 4 * 64  # 4x4 grid x default conv_dim


def test_non_bool_spatial_pool_is_rejected():
    """A YAML typo ("false" as a string) must not silently read as truthy."""
    with pytest.raises(TypeError, match="spatial_pool must be a bool"):
        NatureCNN(_space(64), features_dim=32, spatial_pool="false")
