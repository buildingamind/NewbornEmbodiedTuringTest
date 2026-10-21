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


INIT_SCHEMES = ("family", "orthogonal")
FAMILIES = ("relu", "transformer")

#: Encoders whose hidden nonlinearity is ReLU, by class name along the MRO (so the *Pretrained subclasses
#: inherit their base's entry): every Conv2d / Conv3d / Linear gets Kaiming. Workspace DECISIONS §179 add. 1
#: (owner, 2026-10-21: "For any encoder that uses ReLU, change their default to Kaiming init") and add. 3
#: (Conv3d: "Yes use Kaiming init by default"). tests/test_init_scheme.py audits it against what each registered
#: encoder's forward actually calls. XSPEncoder's one sigmoid is its figure gate; its hidden layers are ReLU.
RELU_ENCODERS = frozenset({
    "NatureCNN", "Compact3DCNN", "CompactCNN", "SmallCNN", "Resnet10CNN", "Resnet18CNN",
    "SimCLRCLTT", "GuessWhatMoves", "XSPEncoder",
})

#: Transformer encoders, by class name along the MRO: every Conv2d / Linear and every MultiheadAttention
#: in_proj gets trunc_normal. Workspace DECISIONS §179 add. 2 (owner, 2026-10-21: "all transformer-based encoders
#: should use truncated normal init by default").
TRANSFORMER_ENCODERS = frozenset({"CompactViT", "UnityViT", "CompactViViT"})

#: The ViT-paper / timm / I-JEPA init, and what CompactViT's and CompactViViT's constructors put on their Linears.
TRUNC_NORMAL_STD = 0.02

# An encoder that mixes families declares them per submodule (DECISIONS §179 add. 3, owner: "Truncated normal for
# the transformer-based layers. Kaiming for ReLU layers. Orthogonal for tanh layers."):
#     INIT_FAMILIES = {"<dotted submodule prefix>": "relu" | "transformer", ...}
# Longest prefix wins; every Conv / Linear / MultiheadAttention in such an encoder must fall under one prefix, or the
# build refuses (no silent fallback). The NERVE of the rule is the layer, not the class: ViTSlotRecon's CompactViT
# trunk is transformer, its ReLU slot MLPs are relu.
#
# Not touched by either family, as under the pre-§179 fleet pass: LayerNorm, GRUCell, bare Parameters (CLS tokens,
# position embeddings, slot mu / log_sigma) and every module outside the encoder. Tanh layers -- the MLP trunk at
# activation 'tanh' and the tanh-squashed heads -- already get orthogonal (orthogonal_init / init_output), so the
# owner's "Orthogonal for tanh layers" needs no change there.


def init_scheme() -> str:
    """NETT_INIT_SCHEME: 'family' (default) | 'orthogonal'. Decides the fleet init of an encoder.

    ⚠ DEFAULT 'family' since workspace DECISIONS §179 (owner, 2026-10-21: "Kaiming init should be the
    default moving forward. Don't let that affect the compatibility of the current runs." / add. 1 "For any
    encoder that uses ReLU, change their default to Kaiming init" / add. 2 "all transformer-based encoders
    should use truncated normal init by default" / add. 3 "Truncated normal for the transformer-based layers.
    Kaiming for ReLU layers. Orthogonal for tanh layers."). Under 'family' ReLU layers get kaiming_init and
    transformer layers trunc_normal_init; an encoder in no family, and the MLP trunk, keep orthogonal_init as
    before. 'orthogonal' rebuilds every model byte-identically to the pre-§179 rule (tests/test_init_scheme.py),
    so a row that extends a pre-§179 series declares it. 'kaiming' (the unpushed 796a12b's name for the default)
    and any other value -- including '' -- refuse.
    """
    v = os.environ.get("NETT_INIT_SCHEME", "family").strip()
    if v == "kaiming":
        raise ValueError("NETT_INIT_SCHEME='kaiming' was renamed 'family' (DECISIONS §179 add. 2): Kaiming for "
                         "ReLU layers, trunc_normal for transformer layers")
    if v not in INIT_SCHEMES:
        raise ValueError(f"NETT_INIT_SCHEME={v!r}: expected one of {INIT_SCHEMES} (default 'family')")
    return v


def _in_set(encoder, names: frozenset) -> bool:
    return any(cls.__name__ in names for cls in type(encoder).__mro__)


def encoder_uses_relu(encoder) -> bool:
    return _in_set(encoder, RELU_ENCODERS)


def encoder_is_transformer(encoder) -> bool:
    return _in_set(encoder, TRANSFORMER_ENCODERS)


def encoder_family(encoder) -> str:
    """'relu' | 'transformer' | 'mixed' (declares INIT_FAMILIES) | 'other'."""
    if getattr(type(encoder), "INIT_FAMILIES", None):
        return "mixed"
    if encoder_is_transformer(encoder):
        return "transformer"
    return "relu" if encoder_uses_relu(encoder) else "other"


def _initialisable(module: nn.Module) -> bool:
    return isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear, nn.MultiheadAttention))


def kaiming_init(module: nn.Module) -> None:
    """Kaiming He normal (fan_in, ReLU), zero bias, on Conv2d, Conv3d and Linear (DECISIONS §179 add. 1, add. 3)."""
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def trunc_normal_init(module: nn.Module) -> None:
    """trunc_normal_(std=0.02), zero bias, on Conv2d, Conv3d, Linear and nn.MultiheadAttention's packed QKV
    ``in_proj_weight`` (its ``out_proj`` is a Linear and is reached on its own) -- DECISIONS §179 add. 2, add. 3.

    ⚠ nn.init.trunc_normal_'s bounds are absolute (a=-2, b=2), i.e. +-100 sigma at std .02 -- effectively a plain
    normal, exactly as in the constructors and the reference repos (timm, I-JEPA).
    """
    if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=TRUNC_NORMAL_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention) and module.in_proj_weight is not None:
        nn.init.trunc_normal_(module.in_proj_weight, std=TRUNC_NORMAL_STD)
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)


_RULES = {"relu": (kaiming_init, "kaiming"), "transformer": (trunc_normal_init, "trunc_normal")}


def layer_families(encoder: nn.Module) -> dict[str, str]:
    """{dotted module name: 'relu' | 'transformer'} for every Conv / Linear / MultiheadAttention of a family
    encoder, in named_modules order. Refuses a mixed encoder with an undeclared layer or an unknown family."""
    fam = encoder_family(encoder)
    decl = dict(getattr(type(encoder), "INIT_FAMILIES", None) or {})
    bad = {k: v for k, v in decl.items() if v not in FAMILIES}
    if bad:
        raise ValueError(f"{type(encoder).__name__}.INIT_FAMILIES: unknown families {bad}; expected {FAMILIES}")
    out = {}
    for name, module in encoder.named_modules():
        if not _initialisable(module):
            continue
        if fam in FAMILIES:
            out[name] = fam
            continue
        hits = [k for k in decl if name == k or name.startswith(k + ".")]
        if not hits:
            raise ValueError(f"{type(encoder).__name__}: layer {name!r} ({type(module).__name__}) is under no "
                             f"INIT_FAMILIES prefix {sorted(decl)} (DECISIONS §179 add. 3)")
        out[name] = decl[max(hits, key=len)]
    return out


def init_encoder(encoder: nn.Module, gain: float) -> None:
    """The fleet's encoder init (builder.py shared encoder, backbone.py per-model encoder).

    Under the default 'family' scheme: Kaiming on ReLU layers, trunc_normal on transformer layers (layer_families);
    otherwise exactly the pre-§179 call ``encoder.apply(lambda m: orthogonal_init(m, gain))``.
    NETT_CONV3D_ORTHO_INIT=1 has a meaning only under the orthogonal rule, so it refuses on a family encoder under
    'family'.
    """
    scheme = init_scheme()
    family = encoder_family(encoder)
    if scheme == "family" and family != "other":
        if conv3d_ortho_init_enabled():
            raise ValueError(f"NETT_CONV3D_ORTHO_INIT=1 needs NETT_INIT_SCHEME=orthogonal for the {family} encoder "
                             f"{type(encoder).__name__}; under the default 'family' its layers take their family's "
                             f"init (DECISIONS §176, §179).")
        fams = layer_families(encoder)
        modules = dict(encoder.named_modules())
        for name, f in fams.items():
            _RULES[f][0](modules[name])
        used = "+".join(_RULES[f][1] for f in FAMILIES if f in fams.values()) or "none"
    else:
        encoder.apply(lambda module: orthogonal_init(module, gain))
        used = "orthogonal"
    # stdout, not a logger: the spawn child's logger is not bridged to the driver log, and this line is
    # how a close-time check proves which init reached the arm.
    print(f"[NETT init] encoder {type(encoder).__name__} {used} (NETT_INIT_SCHEME={scheme}, "
          f"family={family})", flush=True)


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
