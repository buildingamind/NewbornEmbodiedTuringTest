"""Initialization helpers shared by NETT skrl models."""

from __future__ import annotations

import os

import torch.nn as nn

from ..model_cfg import ModelCfg


def conv3d_ortho_init_enabled() -> bool:
    """NETT_CONV3D_ORTHO_INIT: '0' (default) = PyTorch's default init; '1' = orthogonal Conv3d.

    orthogonal_init matched only (Conv2d, Linear) from the first Isaac commit until fb555b2, so every
    Conv3d stem (compact_3dcnn ``conv3d``, guess_what_moves ``moves_3d``, dual_stream) kept
    kaiming_uniform(a=sqrt(5)) and a uniform bias: for the 3DCNN stem (fan_in 54) that is weight std
    .079 against ~.192 under orthogonal at gain sqrt(2), and a nonzero bias.

    ⚠ DEFAULT-OFF since workspace DECISIONS §176 (owner, 2026-10-21: "Switch off the orthogonal init
    for Conv3d, that should be the default"), after the §199 add. 16 bisect found the init carried
    most of the VI loss. It was default-on from §63 (2026-10-03) until then; opt-in at fb555b2 (§61).
    Unset or '0' builds every model byte-identically to the old (Conv2d, Linear)-only rule. An arm
    that must keep the §63 init sets '1' explicitly. Any other value -- including an empty string --
    refuses.
    """
    v = os.environ.get("NETT_CONV3D_ORTHO_INIT", "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"NETT_CONV3D_ORTHO_INIT={v!r}: expected '0' (default) or '1'")
    return v == "1"


INIT_SCHEMES = ("kaiming", "orthogonal")

#: Encoders whose hidden nonlinearity is ReLU, by class name along the MRO (so the *Pretrained subclasses
#: inherit their base's entry). Workspace DECISIONS §179 add. 1 (owner, 2026-10-21: "For any encoder that uses
#: ReLU, change their default to Kaiming init"). tests/test_init_scheme.py audits it against what each registered
#: encoder's forward actually calls, and refuses an encoder_mapping entry that is in neither this set, the
#: test's non-ReLU set, nor OWNS_INIT. XSPEncoder's one sigmoid is its figure gate; its hidden layers are ReLU.
#: NOT here: CompactViT / UnityViT / CompactViViT (GELU), ViTSlotRecon (GELU ViT + ReLU slot MLP, mixed), and the
#: encoders that own their init (XSP2, MatchedEncoder, JEPAEncoder, STPEEncoder).
RELU_ENCODERS = frozenset({
    "NatureCNN", "Compact3DCNN", "CompactCNN", "SmallCNN", "Resnet10CNN", "Resnet18CNN",
    "SimCLRCLTT", "GuessWhatMoves", "XSPEncoder",
})


def init_scheme() -> str:
    """NETT_INIT_SCHEME: 'kaiming' (default) | 'orthogonal'. Decides the fleet init of a RELU encoder only.

    ⚠ DEFAULT 'kaiming' since workspace DECISIONS §179 (owner, 2026-10-21: "Kaiming init should be the
    default moving forward. Don't let that affect the compatibility of the current runs." / "For any encoder
    that uses ReLU, change their default to Kaiming init"). Under 'kaiming' a ReLU encoder (RELU_ENCODERS)
    gets kaiming_init; every other encoder, and the MLP trunk, keep orthogonal_init as before. 'orthogonal'
    rebuilds every model byte-identically to the pre-§179 rule (tests/test_init_scheme.py), so a row that
    extends a pre-§179 series of a ReLU encoder declares it. Any other value -- including '' -- refuses.
    """
    v = os.environ.get("NETT_INIT_SCHEME", "kaiming").strip()
    if v not in INIT_SCHEMES:
        raise ValueError(f"NETT_INIT_SCHEME={v!r}: expected one of {INIT_SCHEMES} (default 'kaiming')")
    return v


def encoder_uses_relu(encoder) -> bool:
    return any(cls.__name__ in RELU_ENCODERS for cls in type(encoder).__mro__)


def kaiming_init(module: nn.Module) -> None:
    """Kaiming He normal (fan_in, ReLU), zero bias, on the module set orthogonal_init's default rule touches:
    Conv2d and Linear. Conv3d keeps PyTorch's default (DECISIONS §176)."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_encoder(encoder: nn.Module, gain: float) -> None:
    """The fleet's encoder init (builder.py shared encoder, backbone.py per-model encoder).

    Kaiming for a ReLU encoder under the default scheme; otherwise exactly the pre-§179 call
    ``encoder.apply(lambda m: orthogonal_init(m, gain))``. NETT_CONV3D_ORTHO_INIT=1 has a meaning only
    under the orthogonal rule, so it refuses on a ReLU encoder under 'kaiming'.
    """
    scheme = init_scheme()
    relu = encoder_uses_relu(encoder)
    if scheme == "kaiming" and relu:
        if conv3d_ortho_init_enabled():
            raise ValueError(f"NETT_CONV3D_ORTHO_INIT=1 needs NETT_INIT_SCHEME=orthogonal for the ReLU encoder "
                             f"{type(encoder).__name__}; under the default 'kaiming' its Conv3d keeps PyTorch's "
                             f"default (DECISIONS §176, §179).")
        encoder.apply(kaiming_init)
        used = "kaiming"
    else:
        encoder.apply(lambda module: orthogonal_init(module, gain))
        used = "orthogonal"
    # stdout, not a logger: the spawn child's logger is not bridged to the driver log, and this line is
    # how a close-time check proves which init reached the arm.
    print(f"[NETT init] encoder {type(encoder).__name__} {used} (NETT_INIT_SCHEME={scheme}, "
          f"relu={'yes' if relu else 'no'})", flush=True)


def encoder_owns_init(encoder) -> bool:
    """True for an encoder that declares ``OWNS_INIT = True``: the builder and the backbone then
    skip the fleet's orthogonal_init for it and keep the init its constructor chose.

    Only XSP2 declares it (owner, workspace DECISIONS §87 spec change 2: Kaiming trunk, PyTorch-
    default streams). Every other encoder reads False, so its init is unchanged.
    """
    return bool(getattr(encoder, "OWNS_INIT", False))


def orthogonal_init(module: nn.Module, gain: float) -> None:
    types = (nn.Conv2d, nn.Linear)
    if conv3d_ortho_init_enabled():
        types = types + (nn.Conv3d,)
        if isinstance(module, nn.Conv3d):
            # stdout, not a logger: the spawn child's logger is not bridged to the driver log,
            # and this line is how a close-time check proves the knob reached the arm.
            print(f"[NETT init] Conv3d orthogonal x{gain:.4f} {tuple(module.weight.shape)}", flush=True)
    if isinstance(module, types):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def init_output(layer: nn.Linear, cfg: ModelCfg, gain: float | None = None) -> None:
    # SB3 uses gain=0.01 for the POLICY head but gain=1.0 for the VALUE head.
    # SKRL applied cfg.output_gain (0.01) to BOTH, which makes the critic output
    # near-zero at init; with correctly-scaled [0,1] CNN input (features ~255x
    # larger than the old crushed input) the mis-scaled critic produces garbage
    # advantages and training collapses. Pass gain=1.0 for the value head.
    if cfg.orthogonal_init:
        nn.init.orthogonal_(layer.weight, gain=cfg.output_gain if gain is None else gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
