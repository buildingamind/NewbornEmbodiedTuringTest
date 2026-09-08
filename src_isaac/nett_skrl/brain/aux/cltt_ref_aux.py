"""CLTT reference rebuild as an AUXILIARY LOSS, under the distinct ``cltt_ref`` key.

WHAT IS FAITHFUL. buildingamind/ChicksAndDNNs_ViewInvariance draws 512 CONTIGUOUS
sliding windows from ONE image stream (dataloader shuffle defaults to False).
Its window_size=3 compares [t, t+1, t+2] with TWO positive offsets, SUMMED:
nt_xent(z1, z2) + nt_xent(z1, z3). We retain one-stream contiguous negatives,
backbone gradients, the Linear(512,512)->BatchNorm->ReLU->Linear(512,128,bias=False)
L2-normalised projector, and the effective CLI temperature 0.5.

DELIBERATE DEPARTURES -- DO NOT "FIX" THESE BACK TO THE REFERENCE.
NETT_AUX_BATCH defaults to 96, not 512: the reference trains offline without PPO;
512 anchors x 3 views x 160 minibatch steps per update is prohibitive here.
NETT_AUX_CLTT_REF_OFFSETS defaults to "2,4": with a 2-frame framestack,
stack(t)=[f(t-1),f(t)] and stack(t+1)=[f(t),f(t+1)] SHARE a frame, so offset 1
lets the encoder match on a literally identical frame. Offsets that are multiples
of the stack depth give disjoint frame sets. We use whole, temporally offset
stacks and log the realised offsets and discovered stack depth T once, then
refuse offsets that are not multiples of T to prevent shared-frame shortcuts.

TWO REFERENCE DEFECTS. The shipped job's --aug False is INERT: train_simclr.py
declares --aug with type=bool, so bool("False") is True and the
``if args.aug is True:`` branch applies SimCLRTrainDataTransform. We follow the
script's evident intent (NO augmentation), which differs from what the reference
binary actually ran. Also, its ``neg - math.e`` self-subtraction is wrong at
T=0.5: self-similarity contributes exp(1/T)=exp(2)=7.3891, but subtracting only
2.7183 leaves 4.67 in every denominator. Our unchanged nt_xent masks the diagonal
exactly instead of preserving this defect.

THE NEGATIVES MAY BE THE PROBLEM, NOT THE FIX. Drawing a CONTIGUOUS block off one env stream makes
every negative a near-in-time frame of the SAME two-object world. The fleet's viewpoint plan v14
raises exactly this against NT-Xent here: "NT-Xent will push apart frames that may show the same
object at a different viewpoint -- actively penalising the invariance under test. CLTT tolerates
this because its sequences wander over 12-100 objects; a NETT rollout is a two-object world." So
this rebuild makes the reference's batch composition FAITHFUL and, by the same move, makes that
hazard MAXIMAL. That is deliberate: it is the controlled test of whether fidelity was what the
CLTT family lacked. If cltt_ref trails today's `cltt`, this paragraph is the first place to look,
and `vicreg` (no negatives) is the designed alternative.

The needs_memory / attach_memory hook was chosen precisely to AVOID the interface
change plan v14 parks in Tier 2 ("an objective that needs a signature change across
five compute() implementations is a code change with its own review"). compute()'s
signature is unchanged and the other four losses are untouched. The incumbent
``cltt`` also stays byte-identical: already-scored arms must keep their definition.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import logger

# Deliberately do NOT port the reference's `neg - math.e`: the self term is
# exp(1/T), equal to e only at T=1. At its T=0.5, exp(2)=7.3891 minus 2.7183
# leaves 4.67 of self-similarity per row. Our nt_xent masks the diagonal exactly.
from .simclr_aux import nt_xent


class CLTTReferenceProjectionHead(nn.Module):
    """Reference Linear -> BatchNorm -> ReLU -> bias-free Linear, L2-normalised."""

    def __init__(self, in_dim: int, hidden: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class CLTTReferenceAuxLoss(nn.Module):
    """Sum temporal NT-Xent terms over contiguous windows from one rollout stream."""

    needs_memory = True

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        offsets = os.environ.get("NETT_AUX_CLTT_REF_OFFSETS", "2,4")
        try:
            self.offsets = tuple(int(k.strip()) for k in offsets.split(","))
            if not self.offsets or any(k <= 0 for k in self.offsets):
                raise ValueError
        except ValueError as exc:
            raise ValueError(
                "NETT_AUX_CLTT_REF_OFFSETS must be a comma-separated list of "
                f"at least one positive integer; got {offsets!r}."
            ) from exc
        self.max_samples = int(os.environ.get("NETT_AUX_BATCH", "96"))
        self.temperature = float(os.environ.get("NETT_AUX_CLTT_REF_TEMP", "0.5"))
        self.head = CLTTReferenceProjectionHead(int(encoder.features_dim))
        self.head.to(next(encoder.parameters()).device)
        self._memory = None
        self.num_frames: int | None = None

    def attach_memory(self, memory) -> None:
        self._memory = memory

    def compute(self, encoder: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        """Ignore the PPO minibatch; draw temporal windows from attached memory."""
        if self._memory is None:
            raise RuntimeError(
                "CLTTReferenceAuxLoss draws its own temporal windows; "
                "call attach_memory(memory) before compute()."
            )
        memory = self._memory
        # Slice BEFORE transferring: HybridDeviceMemory.get_tensor_by_name would
        # copy the WHOLE ~2 GB observation buffer to the GPU, 160 times per update.
        raw = memory.tensors["observations"]
        t_max = memory.memory_size if memory.filled else memory.memory_index
        avail = t_max - max(self.offsets)
        batch = min(self.max_samples, avail)
        if batch < 2:
            raise ValueError(
                f"CLTTReferenceAuxLoss needs B_eff >= 2, got {batch} "
                f"(t_max={t_max}, offsets={self.offsets}, NETT_AUX_BATCH={self.max_samples}). "
                "A contrastive softmax at B=1 is -log(1)=0 and teaches nothing."
            )
        env = torch.randint(raw.shape[1], ()).item()
        t0 = torch.randint(avail - batch + 1, ()).item() if avail >= self.max_samples else 0
        device = next(encoder.parameters()).device
        with torch.no_grad():
            views = [
                encoder._prepare_image(raw[t0 + k : t0 + batch + k, env].to(device))
                for k in (0, *self.offsets)
            ]
        if self.num_frames is None:
            self.num_frames = views[0].shape[1] // 3
            logger.info(
                "CLTTReferenceAuxLoss: offsets=%s, stack depth T=%s",
                self.offsets, self.num_frames,
            )
        if any(k % self.num_frames for k in self.offsets):
            raise ValueError(
                f"CLTTReferenceAuxLoss: realised stack depth T={self.num_frames}, "
                f"offsets={self.offsets}. Offsets not aligned to the stack depth "
                "can make positive views share a literally identical frame, "
                "allowing a shared-frame matching shortcut. Choose "
                f"NETT_AUX_CLTT_REF_OFFSETS that are multiples of T={self.num_frames}."
            )

        z_anchor = self.head(encoder.encode_prepared(views[0]))  # backbone grad ON
        return sum(
            nt_xent(z_anchor, self.head(encoder.encode_prepared(view)), self.temperature)
            for view in views[1:]
        )
