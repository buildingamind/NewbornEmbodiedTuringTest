"""STP-E ("ego-STP"): the policy reads what self-motion does NOT explain (workspace DECISIONS §170).

Labels "STP-E", "STP-E-NoWarp", "STP-E-Cmd" in examples/campaign_train.py; aux brain/aux/stp_aux.py; pose
transport body/wrappers/stp_pose.py. Design source: the presenter's Variant E (regwt notes
chicken/presenter_stp/README.md §4, prototype stp_lite.py, offline recipe tools/presenter_stp_offline.py).

```
obs     = 2-frame T-major stack, each frame (3, H + 1, W): row 0 = the pose row (stp_pose.py), rows 1.. = the eye
pose    = the NEWEST frame's row: [fwd, lat, dyaw, valid], the t-1 -> t change (executed, or commanded)
g_s     = R(x_s)                fixed az/el remap of the equisolid eye, 1.5 deg per cell, s = t-1, t
                                448x280 / FOV 150 -> 100 x 60 cells; 128x80 / FOV 300 -> 200 x 99
a       = deadzone(pose) / scale   per component: sign(p) max(|p| - dz, 0), dz = (0.03, 0.03 units, 0.5 deg),
                                   then / (1, 1 units, 10 deg); a = 0 where valid = 0
m       = softmax(U-Net(g_{t-1}))  (B, M + 1, h, w), M = 4, from the CURRENT (older) frame ONLY
A_i(a)  = expm(sum_k a_k G_ik)     G (3, M, 2, 3) affine generators: linear in a, no bias, so a = 0 -> A = I
xhat    = sum_i m_i warp(g_{t-1}, A_i) + m_M g_{t-1}
wv      = sum_i m_i warp(vm, A_i) + m_M vm      composited validity; w = vm * (wv > 0.999)
r       = (g_t - xhat) * w                       the residual: object motion + what the warps miss
policy  = CNN(cat(r, g_t))                       (B, 6, h, w) -> features_dim
```

(Here "x_t / x_{t+1}" of the README are the stack's older / newer frame: the predictor maps the older frame and the
pose change between the two onto the newer one, which is what the agent sees now.)

THREE ARMS, ONE CLASS. ``warp="affine"`` (STP-E, STP-E-Cmd) is the model above; the two differ only in which
wrapper writes the pose row. ``warp="identity"`` (STP-E-NoWarp) keeps the predictor -- same modules, same
parameter count -- but never calls it: xhat = g_{t-1} and wv = vm exactly, so r = g_t - g_{t-1} on the valid
cells, the plain frame difference (tests/test_stp_e.py checks it bit for bit).

EXACT IDENTITY AT a = 0. matrix_exp(0) is I, but a bilinear resample at an identity grid is not bitwise the
input (the normalised-coordinate round trip) and softmax weights do not sum to exactly 1 in float. So every row
whose whole action is 0 takes xhat = g_{t-1} and wv = vm through ``torch.where``: the dead zone's zeros are
exact zeros of the residual's predictor part, not near-zeros.

REMAP. A fixed bilinear gather from the eye into the az/el grid (the offline ``lens_grid`` geometry, equisolid,
r = (W/2) sin(theta/2) / sin(theta_max/2), centre (W/2, H/2), pixel centres at i + 0.5). Indices and weights are
non-persistent buffers built once on the CPU in float64. A cell is VALID (vm = 1) only if it lies inside the lens
field and inside the image AND every tap it uses lies inside the image and OFF THE RIM: a tap at a pixel whose
centre radius is >= cx - 2 px is the repo-B rim mask's zone (eye_rim_mask.py, RIM_MASK_MARGIN_PX = 2), zeroed
there by default and flickering when that mask is off. Invalid cells read 0. The FOV is not guessed: the
campaign driver writes the resolved aperture into the cfg (``fov_deg``), and only the equisolid lens is
accepted (NETT_FISHEYE_METHOD refuses anything else).

STRICT DETERMINISM. The predictor's backward runs inside PPO's strict_determinism scope (xsp2_aux.py has the
argument). So: no F.grid_sample (no deterministic CUDA backward) -- ``warp_affine`` is a gather whose indices
come from a detached floor, the gradient reaching G only through the bilinear weights; and no bilinear
F.interpolate in the U-Net -- ``_up2`` is nearest x2 by expand + reshape, cropped to the skip's size.

POLICY PATH. ``forward`` builds r under ``torch.no_grad()`` and autocast disabled (fp32, as the aux trains it),
so no RL gradient can reach the predictor; only the policy CNN is in the RL graph. The predictor's parameters
are removed from PPO's optimizer by the aux (``STPTerm.bind_agent``) and trained by its own Adam.

INIT (``OWNS_INIT = True``, so the builder skips the fleet pass): the policy CNN gets the fleet's rule itself,
orthogonal at gain sqrt(2), bias 0, on every Conv2d and Linear; the predictor keeps PyTorch's default init (the
offline recipe), G is warm-started as offline -- the turn generator shifts content left/right by the measured
yaw gain, spread over the M warps as linspace(1, 0.25); the fwd and lat generators start at N(0, 0.01).

DESIGN CHOICES the README left open, each stated in the build report: three action components (fwd, lat, dyaw;
lat is 0 under the commanded arm), nearest upsampling in the U-Net, rim taps excluded from validity, policy
features_dim sized at the label, GroupNorm(4 | 1) + SiLU after every conv (the prototype's ``conv``).
"""

from __future__ import annotations

import math
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ...body.observation import image_channels_hw, image_layout
from ...body.wrappers.stp_pose import POSE_BYTES, POSE_ROWS, VALID, bytes_to_pose
from .hwc_feature_extractor import HWCFeatureExtractor
from .utils.temporal import validate_framestack_depth

#: Degrees of visual angle per az/el cell (the offline recipe's 1.5).
DPP = 1.5
#: The repo-B rim mask's margin (eye_rim_mask.RIM_MASK_MARGIN_PX), copied: repo A does not import repo B here.
RIM_MARGIN_PX = 2.0
#: Dead zone per pose component, physical units: fwd, lat (chamber units), dyaw (degrees). Below it -> exactly 0.
DEADZONE = (0.03, 0.03, 0.5)
#: Physical -> predictor units (offline: a_turn = dyaw / 10 deg, a_move = forward step in units).
ACTION_SCALE = (1.0, 1.0, 10.0)
ACT_DIM = 3
TURN = 2
#: Composited-validity threshold (offline: wv > 0.999).
VALID_WV = 0.999
WARPS = ("affine", "identity")
N_FRAMES = 2
CPF = 3


def lens_azel_taps(width: int, height: int, fov_deg: float, dpp: float = DPP,
                   rim_margin: float = RIM_MARGIN_PX) -> dict:
    """The fixed eye -> az/el bilinear gather: 4 taps per cell, validity, and the grid size.

    Returns ``idx`` (4, Ho*Wo) int64 flat pixel indices, ``wt`` (4, Ho*Wo) float32 weights (0 on invalid cells),
    ``vm`` (Ho, Wo) float32 validity, and ``Ho``, ``Wo``, ``tv_deg`` (the vertical half-field).
    """
    if not (0.0 < fov_deg <= 360.0):
        raise ValueError(f"fov_deg must be in (0, 360]; got {fov_deg!r}")
    tm = math.radians(fov_deg / 2)
    cx, cy = width / 2.0, height / 2.0
    k = cx / math.sin(tm / 2)
    tv = 2 * math.asin(min(1.0, cy / k))
    wo = int(round(fov_deg / dpp))
    ho = int(round(2 * math.degrees(tv) / dpp))
    az = np.radians((np.arange(wo) + 0.5) * dpp - fov_deg / 2)
    el = np.radians(math.degrees(tv) - (np.arange(ho) + 0.5) * dpp)        # top row = up
    AZ, EL = np.meshgrid(az, el)
    dx, dy, dz = np.cos(EL) * np.sin(AZ), np.sin(EL), np.cos(EL) * np.cos(AZ)
    th = np.arccos(np.clip(dz, -1, 1))
    ph = np.arctan2(dy, dx)
    r = k * np.sin(th / 2)
    u = cx + r * np.cos(ph)                                                # px, x right
    v = cy - r * np.sin(ph)                                                # px, y down
    lens_ok = (th <= tm + 1e-9) & (u >= 0) & (u <= width) & (v >= 0) & (v <= height)
    p, q = u - 0.5, v - 0.5                                                # pixel-centre coordinates
    x0, y0 = np.floor(p), np.floor(q)
    wx, wy = p - x0, q - y0
    taps = [(y0, x0, (1 - wy) * (1 - wx)), (y0, x0 + 1, (1 - wy) * wx),
            (y0 + 1, x0, wy * (1 - wx)), (y0 + 1, x0 + 1, wy * wx)]
    ok = lens_ok.copy()
    idx = np.zeros((4, ho * wo), np.int64)
    wt = np.zeros((4, ho * wo), np.float64)
    for n, (yi, xi, w) in enumerate(taps):
        inside = (xi >= 0) & (xi <= width - 1) & (yi >= 0) & (yi <= height - 1)
        rad = np.hypot(xi + 0.5 - cx, yi + 0.5 - cy)
        good = inside & (rad < cx - rim_margin)
        ok &= good | (w == 0)
        idx[n] = np.where(inside, yi * width + xi, 0).reshape(-1).astype(np.int64)
        wt[n] = np.where(good, w, 0.0).reshape(-1)
    wt[:, ~ok.reshape(-1)] = 0.0
    return {"idx": idx, "wt": wt.astype(np.float32), "vm": ok.astype(np.float32), "Ho": ho, "Wo": wo,
            "tv_deg": math.degrees(tv)}


def _conv(ci: int, co: int, k: int = 3, s: int = 1) -> nn.Sequential:
    """The prototype's block: Conv2d(k, s, pad k//2) + GroupNorm(4 if co % 4 == 0 else 1) + SiLU."""
    return nn.Sequential(nn.Conv2d(ci, co, k, s, k // 2), nn.GroupNorm(4 if co % 4 == 0 else 1, co), nn.SiLU())


def _up2(z: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Nearest x2 by expand + reshape (deterministic backward), cropped to ``hw``. Stride-2 convs with padding
    k // 2 give ceil(n / 2), so 2 * ceil(n / 2) is n or n + 1 and the crop always suffices."""
    b, c, h, w = z.shape
    if 2 * h < hw[0] or 2 * w < hw[1]:
        raise ValueError(f"_up2: {h}x{w} cannot reach {hw[0]}x{hw[1]} by a factor of 2")
    up = z[:, :, :, None, :, None].expand(b, c, h, 2, w, 2).reshape(b, c, 2 * h, 2 * w)
    return up[:, :, : hw[0], : hw[1]]


def warp_affine(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W), (B, M, 2, 3) -> (B, M, C, H, W): bilinear resample of x under each affine A.

    Same convention and values as ``F.grid_sample(x, F.affine_grid(A, align_corners=False), padding_mode="zeros",
    align_corners=False)`` (tests compare the two and their gradients w.r.t. A on CPU). x is a constant (detached);
    the gather indices are integers from a detached floor, so the gradient reaches A only through the bilinear
    weights -- elementwise arithmetic, deterministic on any device.
    """
    x = x.detach()
    b, c, h, w = x.shape
    m = A.shape[1]
    if A.shape != (b, m, 2, 3):
        raise ValueError(f"warp_affine: A {tuple(A.shape)} is not (B={b}, M, 2, 3)")
    xs = (2 * torch.arange(w, device=x.device, dtype=A.dtype) + 1) / w - 1
    ys = (2 * torch.arange(h, device=x.device, dtype=A.dtype) + 1) / h - 1
    gx = A[..., 0, 0, None, None] * xs.view(1, 1, 1, w) + A[..., 0, 1, None, None] * ys.view(1, 1, h, 1) \
        + A[..., 0, 2, None, None]                                       # (B, M, H, W) normalised source x
    gy = A[..., 1, 0, None, None] * xs.view(1, 1, 1, w) + A[..., 1, 1, None, None] * ys.view(1, 1, h, 1) \
        + A[..., 1, 2, None, None]
    px = ((gx + 1) * w - 1) / 2
    py = ((gy + 1) * h - 1) / 2
    x0 = px.detach().floor()
    y0 = py.detach().floor()
    wx, wy = px - x0, py - y0
    x0i, y0i = x0.long(), y0.long()
    flat = x.reshape(b, c, h * w)
    out = None
    for yi, xi, wgt in ((y0i, x0i, (1 - wy) * (1 - wx)), (y0i, x0i + 1, (1 - wy) * wx),
                        (y0i + 1, x0i, wy * (1 - wx)), (y0i + 1, x0i + 1, wy * wx)):
        inb = ((xi >= 0) & (xi <= w - 1) & (yi >= 0) & (yi <= h - 1)).to(x.dtype)
        idx = (yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1)).reshape(b, 1, m * h * w).expand(b, c, m * h * w)
        tap = flat.gather(2, idx).view(b, c, m, h, w)
        term = tap * (wgt * inb).unsqueeze(1)
        out = term if out is None else out + term
    return out.permute(0, 2, 1, 3, 4)


class STPPredictor(nn.Module):
    """Variant E: masks from the older frame only, M affine warps exp(sum_k a_k G_k), one unchanged copy."""

    def __init__(self, grid_w: int, num_warps: int = 4, act_dim: int = ACT_DIM, dpp: float = DPP,
                 unit_turn: float = ACTION_SCALE[TURN]) -> None:
        super().__init__()
        self.M = int(num_warps)
        self.e1, self.e2, self.e3, self.e4 = _conv(3, 24, 5, 2), _conv(24, 48, 3, 2), _conv(48, 64, 3, 2), _conv(64, 96, 3, 2)
        self.d3, self.d2, self.d1 = _conv(96 + 64, 48), _conv(48 + 48, 32), _conv(32 + 24, 16)
        self.mask = nn.Conv2d(16, self.M + 1, 1)
        G = torch.zeros(act_dim, self.M, 2, 3)
        # Warm start (offline EgoSTP.build, sgn = +1): a turn of +unit_turn deg (a_turn = 1) shifts content by
        # unit_turn / dpp cells, i.e. 2 * that / Wo in normalised x, at strengths linspace(1, 0.25) over the warps.
        gains = torch.linspace(1.0, 0.25, self.M) if self.M > 1 else torch.ones(1)
        G[TURN, :, 0, 2] = -(unit_turn / dpp) * (2.0 / grid_w) * gains
        for k in range(act_dim):
            if k != TURN:
                G[k] = 0.01 * torch.randn(self.M, 2, 3)
        self.G = nn.Parameter(G)

    def affines(self, a: torch.Tensor) -> torch.Tensor:
        """(B, K) -> (B, M, 2, 3): expm of the generator combination, no bias (a = 0 -> exactly I)."""
        g = torch.einsum("bk,kmij->bmij", a, self.G)
        g3 = torch.cat([g, torch.zeros(a.shape[0], self.M, 1, 3, device=a.device, dtype=g.dtype)], dim=2)
        return torch.linalg.matrix_exp(g3)[..., :2, :]

    def masks(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.e1(x); f2 = self.e2(f1); f3 = self.e3(f2); f4 = self.e4(f3)
        u = self.d3(torch.cat([_up2(f4, f3.shape[-2:]), f3], 1))
        u = self.d2(torch.cat([_up2(u, f2.shape[-2:]), f2], 1))
        u = self.d1(torch.cat([_up2(u, f1.shape[-2:]), f1], 1))
        return _up2(self.mask(u), x.shape[-2:]).softmax(1)              # (B, M + 1, H, W)

    def forward(self, x: torch.Tensor, a: torch.Tensor, vm: torch.Tensor):
        """x (B, 3, H, W) older frame on the grid, a (B, K), vm (H, W) -> (xhat, wv (B, 1, H, W), masks)."""
        m = self.masks(x)
        A = self.affines(a)
        b = x.shape[0]
        vmb = vm.to(x.dtype).expand(b, 1, *vm.shape[-2:])
        J = warp_affine(torch.cat([x, vmb], 1), A)                       # (B, M, 4, H, W): RGB + validity
        J = torch.cat([J, torch.cat([x, vmb], 1)[:, None]], 1)           # + the unchanged copy
        comp = (J * m[:, :, None]).sum(1)                                # (B, 4, H, W)
        xhat, wv = comp[:, :3], comp[:, 3:]
        still = (a == 0).all(dim=1).view(b, 1, 1, 1)
        return torch.where(still, x, xhat), torch.where(still, vmb, wv), m


class STPEEncoder(HWCFeatureExtractor):
    """Ego-STP residual encoder: az/el remap, action-warp predictor (aux-trained), residual policy CNN."""

    #: Read by models/utils/init.py ``encoder_owns_init``: the builder skips the fleet's orthogonal_init; this
    #: class applies it to the policy CNN only (module docstring, INIT).
    OWNS_INIT = True

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 170,
        fov_deg: float | None = None,
        warp: str = "affine",
        num_warps: int = 4,
        num_frames: int | None = None,
        channels_per_frame: int | None = None,
    ):
        super().__init__(observation_space, features_dim)
        if warp not in WARPS:
            raise ValueError(f"STPEEncoder warp must be one of {WARPS}; got {warp!r}.")
        if fov_deg is None or type(fov_deg) not in (int, float) or not (0 < float(fov_deg) <= 360):
            raise ValueError(
                f"STPEEncoder needs the RESOLVED camera aperture as fov_deg (campaign_train writes it into the "
                f"cfg); got {fov_deg!r}. Refusing to guess the field of view.")
        method = os.environ.get("NETT_FISHEYE_METHOD", "equisolid").strip().lower() or "equisolid"
        if method != "equisolid":
            raise ValueError(f"STPEEncoder's az/el remap is the equisolid lens; NETT_FISHEYE_METHOD={method!r}.")
        if type(num_warps) is not int or num_warps < 1:
            raise ValueError(f"STPEEncoder num_warps must be a positive int; got {num_warps!r}.")
        if type(features_dim) is not int or features_dim < 1:
            raise ValueError(f"STPEEncoder features_dim must be a positive int; got {features_dim!r}.")
        if channels_per_frame is not None and int(channels_per_frame) != CPF:
            raise ValueError(f"STPEEncoder needs RGB frames (channels_per_frame {CPF}); got {channels_per_frame}.")
        channels, obs_h, width = image_channels_hw(observation_space)
        if num_frames is not None:
            validate_framestack_depth(channels, num_frames, type(self).__name__, channels_per_frame=CPF)
            if int(num_frames) != N_FRAMES:
                raise ValueError(f"STPEEncoder needs a stack of exactly {N_FRAMES} frames; num_frames={num_frames}.")
        if channels != CPF * N_FRAMES:
            raise ValueError(
                f"STPEEncoder needs {N_FRAMES} stacked RGB frames ({CPF * N_FRAMES} channels, framestack=True, "
                f"NETT_FRAMESTACK_N={N_FRAMES}); the observation has {channels}.")
        height = obs_h - POSE_ROWS
        if height < 1 or width < POSE_BYTES:
            raise ValueError(f"STPEEncoder: observation {obs_h}x{width} has no room for the pose row.")
        self.warp = warp
        self.fov_deg = float(fov_deg)
        self.image_hw = (height, width)
        taps = lens_azel_taps(width, height, self.fov_deg)
        self.grid_hw = (taps["Ho"], taps["Wo"])
        ho, wo = self.grid_hw
        self.register_buffer("tap_idx", torch.from_numpy(taps["idx"]), persistent=False)
        self.register_buffer("tap_wt", torch.from_numpy(taps["wt"]), persistent=False)
        self.register_buffer("vm", torch.from_numpy(taps["vm"]), persistent=False)
        self.register_buffer("deadzone", torch.tensor(DEADZONE, dtype=torch.float32), persistent=False)
        self.register_buffer("action_scale", torch.tensor(ACTION_SCALE, dtype=torch.float32), persistent=False)

        self.predictor = STPPredictor(wo, num_warps)
        self.policy_cnn = nn.Sequential(_conv(2 * CPF, 32, 5, 2), _conv(32, 64, 3, 2), _conv(64, 64, 3, 2),
                                        _conv(64, 64, 3, 2), nn.Flatten())
        with torch.no_grad():
            flat = int(self.policy_cnn(torch.zeros(1, 2 * CPF, ho, wo)).shape[1])
        self.policy_cnn.append(nn.Linear(flat, features_dim))
        self.policy_cnn.append(nn.SiLU())
        self.policy_flatten = flat
        for module in self.policy_cnn.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ parameter groups
    def predictor_parameters(self) -> list[nn.Parameter]:
        """The U-Net, the mask head and G: trained ONLY by the stp aux's Adam, never by PPO."""
        return list(self.predictor.parameters())

    def policy_parameters(self) -> list[nn.Parameter]:
        """The residual policy CNN: PPO's."""
        return list(self.policy_cnn.parameters())

    def parameter_counts(self) -> dict:
        n = lambda ps: sum(p.numel() for p in ps)                           # noqa: E731
        return {"predictor": n(self.predictor_parameters()), "policy_cnn": n(self.policy_parameters()),
                "total": n(self.parameters())}

    # ------------------------------------------------------------------ pieces
    def split(self, observations: torch.Tensor):
        """Raw observation -> (pose (B, 4) of the NEWEST frame, x_{t-1}, x_t (B, 3, H, W) in [0, 1])."""
        prepared = self._prepare_image(observations).float()
        if prepared.shape[1:] != (CPF * N_FRAMES, self.image_hw[0] + POSE_ROWS, self.image_hw[1]):
            raise ValueError(f"STPEEncoder: prepared observation {tuple(prepared.shape)} does not match "
                             f"({CPF * N_FRAMES}, {self.image_hw[0] + POSE_ROWS}, {self.image_hw[1]}).")
        pose = bytes_to_pose(prepared[:, CPF, 0, :POSE_BYTES] * 255.0)
        valid = pose[:, VALID]
        if not bool(((valid == 0) | (valid == 1)).all()):
            raise ValueError(
                "STPEEncoder: the newest frame's row 0 does not carry a pose row (valid flag not 0/1). The arm "
                "must declare pre=('stp_pose_exec',) or ('stp_pose_cmd',) before framestack.")
        frames = prepared[:, :, POSE_ROWS:, :]
        return pose, frames[:, :CPF], frames[:, CPF:]

    def stored_pose(self, rows: torch.Tensor) -> torch.Tensor:
        """Stored observations, flat (..., D) or shaped (..., *observation_space.shape), as uint8 or 0..255 floats
        -> (..., 4) the newest frame's pose, read straight from the bytes (no image is prepared)."""
        shape = tuple(int(v) for v in self.observation_space.shape)
        x = rows if tuple(rows.shape[-len(shape):]) == shape else rows.reshape(*rows.shape[:-1], *shape)
        if image_layout(shape) == "chw":
            b = x[..., CPF, 0, :POSE_BYTES]
        else:
            b = x[..., 0, :POSE_BYTES, CPF]
        return bytes_to_pose(b)

    def azel(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) eye -> (B, C, Ho, Wo) az/el grid; invalid cells are exactly 0."""
        b, c = x.shape[:2]
        flat = x.reshape(b, c, -1)
        out = None
        for n in range(4):
            term = flat.index_select(2, self.tap_idx[n]) * self.tap_wt[n]
            out = term if out is None else out + term
        return out.view(b, c, *self.grid_hw)

    def action(self, pose: torch.Tensor) -> torch.Tensor:
        """(B, 4) pose -> (B, 3) predictor action: dead zone in physical units, then scaled; 0 where invalid."""
        p = pose[:, :ACT_DIM].float()
        p = torch.sign(p) * torch.clamp(p.abs() - self.deadzone, min=0.0)
        p = p * pose[:, VALID:VALID + 1].float()
        return p / self.action_scale

    def predict(self, g_prev: torch.Tensor, a: torch.Tensor):
        """-> (xhat, wv (B, 1, Ho, Wo)). ``identity``: xhat = g_prev and wv = vm, the predictor never runs."""
        if self.warp == "identity":
            return g_prev, self.vm.expand(g_prev.shape[0], 1, *self.grid_hw)
        xhat, wv, _ = self.predictor(g_prev, a, self.vm)
        return xhat, wv

    def residual_input(self, observations: torch.Tensor) -> torch.Tensor:
        """The policy CNN's input cat(r, g_t), (B, 6, Ho, Wo), under no_grad and fp32."""
        device_type = self.vm.device.type
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            pose, x_prev, x_cur = self.split(observations)
            g_prev, g_cur = self.azel(x_prev), self.azel(x_cur)
            xhat, wv = self.predict(g_prev, self.action(pose))
            w = self.vm * (wv > VALID_WV).to(g_cur.dtype)
            r = (g_cur - xhat) * w
            return torch.cat([r, g_cur], dim=1)

    # ------------------------------------------------------------------ policy path
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Policy features. ⛔ The predictor runs under no_grad (``residual_input``): the RL graph starts at the
        policy CNN, so no reward gradient can reach the predictor -- and the stp aux may step its weights in place
        between PPO's forward and backward."""
        return self.policy_cnn(self.residual_input(observations))
