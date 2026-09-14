"""Optional auxiliary self-supervised losses for shaping the visual encoder."""

from __future__ import annotations

from .ppo_aux import AuxLossPPO
from .simclr_aux import SimCLRAuxLoss, SimCLRProjectionHead, nt_xent
from .vicreg_aux import VICRegAuxLoss, vicreg_loss
from .cltt_schneider_aux import CLTTSchneiderAuxLoss, CLTTSchneiderProjectionHead
from .eoo_dual_aux import EoODualAuxLoss
from .gwm_dual_aux import GWMDualAuxLoss

__all__ = [
    "AuxLossPPO", "SimCLRAuxLoss", "SimCLRProjectionHead", "nt_xent",
    "VICRegAuxLoss", "vicreg_loss",
    "CLTTSchneiderAuxLoss", "CLTTSchneiderProjectionHead",
    "EoODualAuxLoss", "GWMDualAuxLoss",
]
