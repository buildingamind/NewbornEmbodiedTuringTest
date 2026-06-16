"""Optional auxiliary self-supervised losses for shaping the visual encoder."""

from __future__ import annotations

from .ppo_aux import AuxLossPPO
from .simclr_aux import SimCLRAuxLoss, SimCLRProjectionHead, nt_xent
from .vicreg_aux import VICRegAuxLoss, vicreg_loss

__all__ = [
    "AuxLossPPO", "SimCLRAuxLoss", "SimCLRProjectionHead", "nt_xent",
    "VICRegAuxLoss", "vicreg_loss",
]
