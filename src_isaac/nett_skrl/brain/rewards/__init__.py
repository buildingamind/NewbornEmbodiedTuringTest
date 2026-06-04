"""skrl-compatible intrinsic rewards for the Isaac NETT backend.

These classes provide the legacy NETT reward names without depending on SB3
callbacks or the optional ``rllte`` package. They operate on the flattened
policy observations emitted by :class:`nett_skrl.body.skrl_adapter.IsaacEnvWrapper`.
"""

from __future__ import annotations

from .cltt import CLTTReward
from .e3b import E3B
from .icm import ICM
from .pseudo_counts import PseudoCounts
from .ride import RIDE
from .unsupported import UnsupportedIntrinsicReward

# --- Public aliases for legacy NETT reward names ---------------------------

NGU = PseudoCounts

Disagreement = UnsupportedIntrinsicReward.named("Disagreement")
Fabric = UnsupportedIntrinsicReward.named("Fabric")
RE3 = UnsupportedIntrinsicReward.named("RE3")
RND = UnsupportedIntrinsicReward.named("RND")


__all__ = [
    "CLTTReward",
    "Disagreement",
    "E3B",
    "Fabric",
    "ICM",
    "NGU",
    "PseudoCounts",
    "RE3",
    "RIDE",
    "RND",
    "UnsupportedIntrinsicReward",
]
