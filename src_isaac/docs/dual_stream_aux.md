# Dual-stream auxiliary objectness losses

`CNN+EoO-Dual` and `CNN+GWM-Dual` use the agent's NatureCNN as their ventral
encoder. The observation retains two RGB frames (`framestack=True`), while
`input_frames=1` builds the host for three channels and selects the last frame.
The single-frame `CNN` is the policy control. Legacy `eoo` / `gwm` arms and loss
equations remain unchanged.

## Architecture and parameter budget

At the campaign's real HWC 80 × 128 × 6 observation, with `features_dim=512`,
`conv_dim=75`, default spatial pooling, and `input_frames=1`, both dual arms have:

| Module | Parameters |
| --- | ---: |
| Host NatureCNN (including policy feature projection) | 773,995 |
| Separate video dorsal | 75,906 |
| Ventral decoder | 66,498 |
| Auxiliary head (dorsal + decoder) | 142,404 |
| Host + dorsal + decoder | **916,399** |

**The combined total exceeds the ~700K budget by 216,399 parameters (30.9%).**
Even the unchanged CNN control width exceeds 700K at this rectangular eye. No
size series is declared and no widths have been silently reduced. These totals
exclude policy/value output layers; PPO logs the full breakdown at construction.

The host exposes its 75 × 6 × 12 conv map before deterministic pooling and
flattening. The auxiliary decoder retains the reference `dec3/dec2/dec1/head`
stages, with `dec3` adapted to 75 input channels. Three bilinear ×2 stages and
final logit resizing produce a full-resolution, two-channel softmax mask of the
current frame. There is no standalone ventral encoder or flat-vector fallback.
The policy and objectness loss train the same host conv parameters; the host's
linear feature projection remains on the policy path.

`Small3DCNNDorsal` is unchanged from the read-only small EoO/GWM reference. Its
`(2, 3, 3)` Conv3d consumes both frames and collapses time, followed by the same
2D flow decoder. EoO uses both flow directions; GWM uses forward flow. The dorsal
has no parameters shared with the host or mask decoder and never feeds policy
inference. The joint auxiliary objective trains both streams.

## Objectives and integration

EoO retains the vendored `unflow_loss`: normalized mask-weighted, occlusion-gated
bidirectional reconstruction plus flow smoothness. GWM retains the mask-weighted
quadratic basis, regularized QR fit, and reconstruction MSE. Flow is trainable
and not detached. The reference limitations (including empty visible support in
EoO and constant-flow degeneracy in GWM) are retained. Mask inputs now come from
the host's current-frame spatial features, as required by the host integration.

These are auxiliary losses, never rewards. All weights start from random
initialization; no segmentation labels or pretrained weights are used. PPO
supplies the optimizer, clipping, schedule, and checkpointing. The host remains
in the policy checkpoint; only its decoder and dorsal belong to `aux_head`.
`NETT_AUX_BATCH` caps samples, defaulting to 32. Prepared current frames bypass
normalization on re-encoding so uint8 observations are divided by 255 only once.

PPO's split backward keeps policy/value backward strict and auxiliary backward
inside `relaxed_determinism`. GWM's QR/CuBLAS forward solve also needs that
exemption, so its reconstruction call has a narrow relaxed context that restores
strict mode on exit. `NETT_AUX_STRICT` preserves strict diagnostic execution.
No test-specific `CUBLAS_WORKSPACE_CONFIG` is introduced.

## Verification

Tests invert the former no-host-gradient contract: real host features work,
auxiliary gradients are finite and nonzero after multiple optimizer steps, and
host parameters and policy representations change. Tests cover the real eye,
frame ordering, uint8 normalization, HWC/CHW/flattened layouts, invalid frame
counts, and bit-identical default NatureCNN forward/readout behavior.

Dorsal architecture/output parity and both loss/gradient comparisons against
the vendored reference remain exact (`rtol=0`, `atol=0`). CPU sentinels check
production determinism scopes; the optimizer test uses CUDA when available.
CUDA is unavailable in this verification environment, so GPU kernels could not
be exercised here.

Full unit gate on 2026-09-14:

```bash
PYTHONPATH=.:/home/zlaborde/code/isaac/priv-wt-epr/isaac_lab/source /home/zlaborde/code/.venv/nett-isaac/bin/python -m pytest tests -q
```

Result: **1,102 passed, 0 failed, 12 skipped, 23 deselected**, 26 warnings, exit 0.
The 23 deselections follow the repository's default Isaac/performance marker
exclusions. skrl/wandb emitted closed-stream logging errors during interpreter
shutdown after the successful test result. No training arm was launched.

Measured host conv gradient L1 sums at steps 0/1/2 were
370.8211 / 292.9542 / 215.1129 for EoO and
0.13306 / 0.12607 / 0.12051 for GWM. Host/dorsal parameter overlap was empty.
A direct comparison against the pre-change NatureCNN confirmed identical state
tensors and forward outputs for 3- and 6-channel inputs, pooled and unpooled.

All 43 new or changed test cases fail against an isolated copy of the source
as it stood before this rewire. The reference files remain read-only:
`NETT_Global_Workspace/archive/2026-08_campaign/scripts/{eoo,gwm}/{model,losses}.py`.
