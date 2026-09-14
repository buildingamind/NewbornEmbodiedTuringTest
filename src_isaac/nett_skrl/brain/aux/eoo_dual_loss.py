"""
unFlow objective ported from vendored scripts/eoo/losses.py.

Equations, including its warp direction convention, sign-based census term
(zero flow gradient almost everywhere), and shared mask for both directions,
are retained for reference fidelity. These are reference limitations, not fixes.
AMP is disabled by EoODualAuxLoss at the call site for numerical stability.

Implements photometric reconstruction loss (L1 + SSIM + Ternary),
edge-aware flow smoothness, and bidirectional occlusion masking.
All inputs are float tensors in [0, 1] range with shape (B, C, H, W).
"""

import torch as th
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Frame warping
# ─────────────────────────────────────────────────────────────────────────────

def warp_frame(frame: th.Tensor, flow: th.Tensor) -> th.Tensor:
    """Bilinear warp of frame by a pixel-displacement flow field.

    Args:
        frame: (B, C, H, W) float in [0, 1]
        flow:  (B, 2, H, W) pixel displacements (dx, dy)

    Returns:
        warped: (B, C, H, W)
    """
    if flow.shape[1] != 2:
        raise ValueError(
            f"warp_frame: flow must have 2 channels (dx, dy), got {flow.shape[1]}"
        )
    if frame.shape[0] != flow.shape[0]:
        raise ValueError(
            f"warp_frame: batch size mismatch — frame {frame.shape[0]} vs flow {flow.shape[0]}"
        )
    if frame.shape[2:] != flow.shape[2:]:
        raise ValueError(
            f"warp_frame: spatial size mismatch — frame {frame.shape[2:]} vs flow {flow.shape[2:]}"
        )
    B, C, H, W = frame.shape

    # Build base grid in pixel coordinates
    grid_y, grid_x = th.meshgrid(
        th.arange(H, device=frame.device, dtype=th.float32),
        th.arange(W, device=frame.device, dtype=th.float32),
        indexing="ij",
    )
    # (1, 2, H, W): x first then y to match flow channel order
    base = th.stack([grid_x, grid_y], dim=0).unsqueeze(0)

    # Add flow displacement
    sampling = base + flow  # (B, 2, H, W)

    # Normalize to [-1, 1] for grid_sample
    sampling[:, 0] = 2.0 * sampling[:, 0] / max(W - 1, 1) - 1.0
    sampling[:, 1] = 2.0 * sampling[:, 1] / max(H - 1, 1) - 1.0

    grid = sampling.permute(0, 2, 3, 1)  # (B, H, W, 2)
    warped = F.grid_sample(
        frame, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return warped


# ─────────────────────────────────────────────────────────────────────────────
# Photometric sub-losses
# ─────────────────────────────────────────────────────────────────────────────

def charbonnier_loss(diff: th.Tensor, eps: float = 1e-3) -> th.Tensor:
    """Charbonnier (robust L1) loss: sqrt(x^2 + eps^2)."""
    return th.sqrt(diff.pow(2) + eps * eps)


def ssim_loss(x: th.Tensor, y: th.Tensor, window_size: int = 11) -> th.Tensor:
    """Returns per-pixel (1 - SSIM) averaged over channels, shape (B, 1, H, W).

    Uses average-pooling as an approximation to the Gaussian-weighted SSIM.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    pad = window_size // 2

    mu_x  = F.avg_pool2d(x, kernel_size=window_size, stride=1, padding=pad)
    mu_y  = F.avg_pool2d(y, kernel_size=window_size, stride=1, padding=pad)
    mu_xx = mu_x.pow(2)
    mu_yy = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sig_x  = F.avg_pool2d(x.pow(2), kernel_size=window_size, stride=1, padding=pad) - mu_xx
    sig_y  = F.avg_pool2d(y.pow(2), kernel_size=window_size, stride=1, padding=pad) - mu_yy
    sig_xy = F.avg_pool2d(x * y,    kernel_size=window_size, stride=1, padding=pad) - mu_xy

    num = (2.0 * mu_xy + C1) * (2.0 * sig_xy + C2)
    den = (mu_xx + mu_yy + C1) * (sig_x + sig_y + C2)

    ssim_map = num / den.clamp(min=1e-7)  # (B, C, H, W)
    # Average over channels → (B, 1, H, W), clamp to [0, 1]
    return (1.0 - ssim_map).mean(dim=1, keepdim=True).clamp(0.0, 1.0)


def ternary_loss(x: th.Tensor, y: th.Tensor, patch_size: int = 7) -> th.Tensor:
    """Soft census-transform (ternary) loss, shape (B, 1, H, W)."""
    pad = patch_size // 2
    B, C, H, W = x.shape
    P = patch_size * patch_size

    x_unf = F.unfold(x, kernel_size=patch_size, padding=pad)  # (B, C*P, H*W)
    y_unf = F.unfold(y, kernel_size=patch_size, padding=pad)

    x_unf = x_unf.view(B, C, P, H * W)
    y_unf = y_unf.view(B, C, P, H * W)

    # Compare each pixel to the central pixel
    cx = x_unf[:, :, P // 2 : P // 2 + 1, :]  # (B, C, 1, H*W)
    cy = y_unf[:, :, P // 2 : P // 2 + 1, :]

    tx = th.sign(x_unf - cx)
    ty = th.sign(y_unf - cy)

    # Soft Hamming distance averaged over patch and channels
    loss = (tx - ty).pow(2).mean(dim=2).mean(dim=1)  # (B, H*W)
    return loss.view(B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Flow smoothness
# ─────────────────────────────────────────────────────────────────────────────

def smooth_grad_1st(flow: th.Tensor, img: th.Tensor, alpha: float = 10.0) -> th.Tensor:
    """Edge-aware first-order flow smoothness (scalar).

    Args:
        flow: (B, 2, H, W) optical flow
        img:  (B, C, H, W) reference image for edge weights
    """
    # Flow spatial gradients
    flow_dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs()   # (B, 2, H, W-1)
    flow_dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs()   # (B, 2, H-1, W)

    # Image luminance gradient as edge proxy
    lum = img.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    img_dx = (lum[:, :, :, 1:] - lum[:, :, :, :-1]).abs()      # (B, 1, H, W-1)
    img_dy = (lum[:, :, 1:, :] - lum[:, :, :-1, :]).abs()      # (B, 1, H-1, W)

    w_dx = th.exp(-alpha * img_dx)  # suppress at edges
    w_dy = th.exp(-alpha * img_dy)

    return (w_dx * flow_dx).mean() + (w_dy * flow_dy).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Bidirectional occlusion masking
# ─────────────────────────────────────────────────────────────────────────────

def occlusion_masks(
    F_fwd: th.Tensor, F_bwd: th.Tensor
) -> tuple[th.Tensor, th.Tensor]:
    """Bidirectional consistency occlusion masks.

    Returns (occ_fwd, occ_bwd), each (B, 1, H, W) float in {0, 1}.
    1 = non-occluded (consistent motion), 0 = occluded.
    """
    F_bwd_w = warp_frame(F_bwd, F_fwd)  # backward flow warped by forward flow
    F_fwd_w = warp_frame(F_fwd, F_bwd)  # forward flow warped by backward flow

    err_fwd = (F_fwd + F_bwd_w).pow(2).sum(dim=1, keepdim=True)
    err_bwd = (F_bwd + F_fwd_w).pow(2).sum(dim=1, keepdim=True)

    mag_fwd = F_fwd.pow(2).sum(dim=1, keepdim=True) + F_bwd_w.pow(2).sum(dim=1, keepdim=True)
    mag_bwd = F_bwd.pow(2).sum(dim=1, keepdim=True) + F_fwd_w.pow(2).sum(dim=1, keepdim=True)

    occ_fwd = (err_fwd < 0.01 * mag_fwd + 0.5).float()
    occ_bwd = (err_bwd < 0.01 * mag_bwd + 0.5).float()

    return occ_fwd, occ_bwd


# ─────────────────────────────────────────────────────────────────────────────
# Combined unFlow loss
# ─────────────────────────────────────────────────────────────────────────────

def unflow_loss(
    frame_t:  th.Tensor,
    frame_t1: th.Tensor,
    M:        th.Tensor,
    F_fwd:    th.Tensor,
    F_bwd:    th.Tensor,
    w_ssim:   float = 0.85,
    w_smooth: float = 75.0,
) -> th.Tensor:
    """Combined photometric + smoothness + occlusion loss.

    Args:
        frame_t:  (B, C, H, W) float [0, 1]  — frame at time t
        frame_t1: (B, C, H, W) float [0, 1]  — frame at time t+1
        M:        (B, 1, H, W) float [0, 1]  — foreground probability mask
        F_fwd:    (B, 2, H, W) — forward flow (t → t+1)
        F_bwd:    (B, 2, H, W) — backward flow (t+1 → t)
        w_ssim:   SSIM weight in photometric mix (Charb weight = 1 - w_ssim)
        w_smooth: smoothness loss coefficient

    Returns:
        scalar loss
    """
    B, C, H, W = frame_t.shape

    # --- Shape consistency guards ---
    if frame_t1.shape != frame_t.shape:
        raise ValueError(
            f"unflow_loss: frame_t and frame_t1 must have the same shape, "
            f"got {frame_t.shape} vs {frame_t1.shape}"
        )
    if M.shape != (B, 1, H, W):
        raise ValueError(
            f"unflow_loss: M must be (B, 1, H, W)={(B, 1, H, W)}, got {M.shape}"
        )
    if F_fwd.shape != (B, 2, H, W):
        raise ValueError(
            f"unflow_loss: F_fwd must be (B, 2, H, W)={(B, 2, H, W)}, got {F_fwd.shape}"
        )
    if F_bwd.shape != (B, 2, H, W):
        raise ValueError(
            f"unflow_loss: F_bwd must be (B, 2, H, W)={(B, 2, H, W)}, got {F_bwd.shape}"
        )

    occ_fwd, occ_bwd = occlusion_masks(F_fwd, F_bwd)

    # Synthesize frames via flow warp
    synth_t1 = warp_frame(frame_t,  F_fwd)  # t → t+1
    synth_t  = warp_frame(frame_t1, F_bwd)  # t+1 → t

    # --- Photometric loss (Charbonnier + SSIM + Ternary) ---
    def _photo(synth: th.Tensor, target: th.Tensor) -> th.Tensor:
        charb = charbonnier_loss(synth - target).mean(dim=1, keepdim=True)
        ssim  = ssim_loss(synth, target)
        tern  = ternary_loss(synth, target)
        return (1.0 - w_ssim) * charb + w_ssim * ssim + tern

    photo_fwd = _photo(synth_t1, frame_t1)
    photo_bwd = _photo(synth_t,  frame_t)

    # Mask-weighted, occlusion-gated photometric loss
    w_fwd = occ_fwd * M
    w_bwd = occ_bwd * M
    loss_photo = (
        (w_fwd * photo_fwd).sum() / (w_fwd.sum() + 1e-7)
        + (w_bwd * photo_bwd).sum() / (w_bwd.sum() + 1e-7)
    )

    # --- Smoothness loss ---
    loss_smooth = smooth_grad_1st(F_fwd, frame_t) + smooth_grad_1st(F_bwd, frame_t1)
    H, W = frame_t.shape[2:]
    loss_smooth = loss_smooth / (H * W)

    return loss_photo + w_smooth * loss_smooth
