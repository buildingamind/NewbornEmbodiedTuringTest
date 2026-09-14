"""
Quadratic flow reconstruction ported from vendored scripts/gwm/losses.py.

The flow remains trainable: there is deliberately no detach. The reference's
mask-weighted design matrix, unweighted fit RHS, and outer mask weighting are
retained exactly. GWMDualAuxLoss runs this QR solve in float32 outside AMP.

Each predicted mask M_k explains a region of the optical flow via a 6-parameter
rigid-affine (quadratic) motion model.  The loss measures how well the set of
masks collectively reconstructs the observed optical flow.

Reference:
  Karazija et al., "Guess What Moves", BMVC 2022.
"""

import warnings

import torch as th


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_coord_grid(H: int, W: int, device: th.device) -> th.Tensor:
    """Return normalized (x, y) coordinate grids, each in [-1, 1].

    Returns:
        coords: (2, H, W) — coords[0] = x (column axis), coords[1] = y (row axis)
    """
    ys = th.linspace(-1.0, 1.0, H, device=device)
    xs = th.linspace(-1.0, 1.0, W, device=device)
    grid_y, grid_x = th.meshgrid(ys, xs, indexing="ij")
    return th.stack([grid_x, grid_y], dim=0)  # (2, H, W)


def make_quadratic_basis(M_k: th.Tensor, coords: th.Tensor) -> th.Tensor:
    """Build the 6-column weighted basis matrix for mask M_k.

    Basis functions: [x·M, y·M, x²·M, y²·M, xy·M, M]

    Args:
        M_k:    (B, H, W) — soft mask values in [0, 1]
        coords: (2, H, W) — (x, y) coordinate grids in [-1, 1]

    Returns:
        B_k: (B, H*W, 6) — weighted design matrix
    """
    x = coords[0]   # (H, W)
    y = coords[1]   # (H, W)

    # Each basis function: (H, W) → broadcast with M_k (B, H, W)
    bases = th.stack([
        x * M_k,       # x·M
        y * M_k,       # y·M
        (x * x) * M_k, # x²·M
        (y * y) * M_k, # y²·M
        (x * y) * M_k, # xy·M
        M_k,           # constant·M
    ], dim=-1)  # (B, H, W, 6)

    B, H, W, _ = bases.shape
    return bases.reshape(B, H * W, 6)  # (B, H*W, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Per-mask quadratic flow fitting
# ─────────────────────────────────────────────────────────────────────────────

def fit_quadratic_flow(
    M_k: th.Tensor,
    flow: th.Tensor,
    coords: th.Tensor,
    reg: float = 0.01,
) -> th.Tensor:
    """Fit a 6-parameter quadratic motion model to the flow within mask M_k.

    Matches the GWM paper implementation (flow_reconstruction.py):
      A = weighted design matrix (B, N, 6)
      Solve  A @ c = flow_component  via QR decomposition
      (lstq with lambda=0.01 in original code)

    The system is solved independently for each batch element and each
    flow component (u and v).

    Args:
        M_k:    (B, H, W) float [0, 1]
        flow:   (B, 2, H, W) optical flow
        coords: (2, H, W) coordinate grids
        reg:    regularization  (0.01 as in original GWM code)

    Returns:
        flow_hat_k: (B, 2, H, W) — per-mask reconstructed flow
    """
    B, _, H, W = flow.shape
    N = H * W

    A = make_quadratic_basis(M_k, coords)  # (B, N, 6)

    flow_flat = flow.reshape(B, 2, N).transpose(1, 2)  # (B, N, 2)

    # QR decomposition as in the original GWM code
    # Add Tikhonov regularization by appending reg*I rows to A
    # (equivalent to lstq with lambda=reg)
    reg_rows = (reg ** 0.5) * th.eye(6, device=flow.device).unsqueeze(0).expand(B, -1, -1)
    reg_rhs  = th.zeros(B, 6, 2, device=flow.device)

    A_aug   = th.cat([A,        reg_rows], dim=1)  # (B, N+6, 6)
    rhs_aug = th.cat([flow_flat, reg_rhs], dim=1)  # (B, N+6, 2)

    # QR decomposition: A_aug = Q @ R  then c = R^{-1} @ (Q^T @ rhs_aug)
    Q, R = th.linalg.qr(A_aug)            # Q: (B, N+6, 6),  R: (B, 6, 6)
    Qt_b = th.bmm(Q.transpose(1, 2), rhs_aug)  # (B, 6, 2)
    c    = th.linalg.solve_triangular(R, Qt_b, upper=True)  # (B, 6, 2)

    # --- Numerical stability guard ---
    # NaN in c means R was (near-)singular — mask had near-zero support so the
    # basis columns are linearly dependent.  Replace NaN coefficients with 0 so
    # the corresponding flow_hat is zero rather than NaN-poisoning the loss.
    if th.isnan(c).any():
        n_nan = int(th.isnan(c).sum())
        warnings.warn(
            f"fit_quadratic_flow: {n_nan} NaN coefficient(s) after QR solve "
            f"(mask may have near-zero support — reg={reg} may need increasing). "
            "NaN entries replaced with 0 for this batch.",
            RuntimeWarning,
            stacklevel=2,
        )
        c = th.nan_to_num(c, nan=0.0)

    # Reconstruct flow from coefficients using the original (unaugmented) basis
    flow_hat_flat = th.bmm(A, c)                              # (B, N, 2)
    flow_hat_k    = flow_hat_flat.transpose(1, 2).reshape(B, 2, H, W)
    return flow_hat_k


# ─────────────────────────────────────────────────────────────────────────────
# Combined GWM reconstruction loss
# ─────────────────────────────────────────────────────────────────────────────

def flow_reconstruction_loss(
    masks: th.Tensor,
    flow:  th.Tensor,
    reg:   float = 0.01,
) -> th.Tensor:
    """Quadratic flow reconstruction loss over all N_q mask slots.

    Each mask M_k is a softmax probability map (per-query, partition of unity).
    The total reconstruction is:
        F_hat = Σ_k  M_k ⊙ F_hat_k
    masks must sum to 1 per pixel across the query dimension (softmax output).

    Args:
        masks: (B, N_q, H, W) float [0, 1] — per-query softmax masks (sum to 1 per pixel)
        flow:  (B, 2, H, W)  float          — optical flow from the trainable dorsal stream
        reg:   QR regularization (0.01 as in original GWM code)

    Returns:
        scalar MSE reconstruction loss
    """
    B, N_q, H, W = masks.shape

    # --- Shape guard ---
    if flow.shape != (B, 2, H, W):
        raise ValueError(
            f"flow_reconstruction_loss: flow must be (B, 2, H, W)={(B, 2, H, W)}, "
            f"got {flow.shape}"
        )

    # --- Softmax partition-of-unity guard ---
    # masks must sum to 1 per pixel across the query dimension.
    # This is guaranteed by the ventral softmax but checked here so
    # any accidental sigmoid/sigmoid renormalisation is caught immediately.
    with th.no_grad():
        max_dev = (masks.sum(dim=1) - 1.0).abs().max().item()
    if max_dev > 1e-3:
        warnings.warn(
            f"flow_reconstruction_loss: masks do not sum to 1 per pixel "
            f"(max deviation from 1 = {max_dev:.6f}). "
            "Ensure the ventral output uses softmax(dim=1), not sigmoid.",
            RuntimeWarning,
            stacklevel=2,
        )

    device = flow.device

    coords = _make_coord_grid(H, W, device)  # (2, H, W)

    # masks are softmax-normalised (sum to 1 across queries per pixel),
    # so they directly serve as partition weights for the reconstruction.
    F_hat = th.zeros_like(flow)  # (B, 2, H, W)

    for k in range(N_q):
        M_k     = masks[:, k, :, :]  # (B, H, W)
        F_hat_k = fit_quadratic_flow(M_k, flow, coords, reg=reg)  # (B, 2, H, W)
        F_hat   = F_hat + M_k.unsqueeze(1) * F_hat_k

    loss = (F_hat - flow).pow(2).mean()
    return loss

