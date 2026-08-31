"""Metric SE(2) alignment for native TransFuser++ BEV features.

The frozen ``backbone_bev`` tensor has a deliberately narrow geometry
contract: a native 64 x 64 feature grid covering ``[-32, 32]`` metres in
both ego axes.  Tensor height is ego ``y`` (right) and tensor width is ego
``x`` (forward).  Pixel locations represent cell centres.

``torch.nn.functional.grid_sample`` performs inverse sampling.  Therefore,
for every output point in the current ego frame, this module computes its
coordinate in the source ego frame (``T_source^-1 T_current``) and samples
the source tensor there.  Every source frame is warped directly to the
current frame; callers must not chain adjacent-frame warps.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BEVGeometryContract:
    """Frozen metric interpretation of the native TF++ BEV feature grid."""

    height: int = 64
    width: int = 64
    min_x: float = -32.0
    max_x: float = 32.0
    min_y: float = -32.0
    max_y: float = 32.0
    align_corners: bool = False

    def __post_init__(self) -> None:
        if (self.height, self.width) != (64, 64):
            raise ValueError("native TF++ BEV geometry must be exactly 64x64")
        if (self.min_x, self.max_x, self.min_y, self.max_y) != (
            -32.0,
            32.0,
            -32.0,
            32.0,
        ):
            raise ValueError("native TF++ BEV bounds must be x/y [-32, 32] metres")
        if self.align_corners:
            raise ValueError("native TF++ BEV sampling requires align_corners=False")

    @property
    def metres_per_cell(self) -> tuple[float, float]:
        """Return ``(x, y)`` metric cell sizes."""

        return (
            (self.max_x - self.min_x) / self.width,
            (self.max_y - self.min_y) / self.height,
        )

    def metric_cell_centres(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a ``[H,W,2]`` grid of current-frame ``(x,y)`` centres."""

        if not dtype.is_floating_point:
            raise TypeError("metric grid dtype must be floating point")
        x_step, y_step = self.metres_per_cell
        x = self.min_x + (torch.arange(self.width, device=device, dtype=dtype) + 0.5) * x_step
        y = self.min_y + (torch.arange(self.height, device=device, dtype=dtype) + 0.5) * y_step
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((grid_x, grid_y), dim=-1)


NATIVE_BEV_GEOMETRY = BEVGeometryContract()


def native_bev_geometry_metadata() -> dict[str, Any]:
    """Return the exact JSON contract required from a v2 cache index."""

    return {
        "metric_bounds_m": {"x": [-32.0, 32.0], "y": [-32.0, 32.0]},
        "raster_shape": [256, 256],
        "native_feature_shape": [64, 64],
        "meters_per_native_cell": 1.0,
        "tensor_width_axis": "ego_x_forward",
        "tensor_height_axis": "ego_y_right",
        "pose": "global_x_y_yaw_radians",
        "sampling": "cell_center",
        "align_corners": False,
        "pooling": "none",
        "feature_capture": "model.backbone.output[0]",
    }


def validate_native_bev_geometry_metadata(value: Any) -> None:
    """Fail closed if cache provenance does not certify the native geometry."""

    if not isinstance(value, Mapping):
        raise ValueError("control cache source is missing BEV geometry provenance")
    actual = dict(value)
    expected = native_bev_geometry_metadata()
    if actual != expected:
        raise ValueError(
            "control cache BEV geometry contract mismatch; "
            f"expected {expected}, got {actual}"
        )


def _working_dtype(tensor: torch.Tensor) -> torch.dtype:
    if not tensor.is_floating_point():
        raise TypeError("source_bev must be floating point")
    # CPU grid_sample does not support half precision.  Frozen cached features
    # also benefit from constructing the metric sampling grid in float32.
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _validate_pose(name: str, pose: torch.Tensor) -> None:
    if not pose.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if pose.shape[-1:] != (3,):
        raise ValueError(f"{name} must end in global (x,y,yaw), got {tuple(pose.shape)}")
    if not torch.isfinite(pose).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _flatten_bev_and_poses(
    source_bev: torch.Tensor,
    source_pose: torch.Tensor,
    current_pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Normalize either ``[B,C,H,W]`` or ``[B,T,C,H,W]`` inputs."""

    if source_bev.ndim not in (4, 5):
        raise ValueError("source_bev must be [B,C,H,W] or [B,T,C,H,W]")
    if tuple(source_bev.shape[-2:]) != (
        NATIVE_BEV_GEOMETRY.height,
        NATIVE_BEV_GEOMETRY.width,
    ):
        raise ValueError(
            "source_bev violates native 64x64 TF++ geometry: "
            f"got {tuple(source_bev.shape[-2:])}"
        )
    if source_bev.shape[-3] < 1:
        raise ValueError("source_bev must have at least one channel")

    source_pose = torch.as_tensor(source_pose, device=source_bev.device)
    current_pose = torch.as_tensor(current_pose, device=source_bev.device)
    _validate_pose("source_pose", source_pose)
    _validate_pose("current_pose", current_pose)

    if source_bev.ndim == 4:
        batch = source_bev.shape[0]
        if source_pose.shape != (batch, 3) or current_pose.shape != (batch, 3):
            raise ValueError("4D BEV poses must both be [B,3]")
        prefix = (batch,)
        return source_bev, source_pose, current_pose, prefix

    batch, history = source_bev.shape[:2]
    if source_pose.shape != (batch, history, 3):
        raise ValueError("5D source_pose must be [B,T,3]")
    if current_pose.shape == (batch, 3):
        current_pose = current_pose[:, None, :].expand(-1, history, -1)
    elif current_pose.shape != (batch, history, 3):
        raise ValueError("5D current_pose must be [B,3] or [B,T,3]")
    prefix = (batch, history)
    return (
        source_bev.reshape(batch * history, *source_bev.shape[2:]),
        source_pose.reshape(batch * history, 3),
        current_pose.reshape(batch * history, 3),
        prefix,
    )


def source_sampling_grid(
    source_pose: torch.Tensor,
    current_pose: torch.Tensor,
    *,
    geometry: BEVGeometryContract = NATIVE_BEV_GEOMETRY,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build normalized source coordinates for a current-frame output grid.

    Poses are ``[N,3]`` global ``(x,y,yaw)`` in radians.  The result is
    ``[N,64,64,2]`` in the normalized ``(width,height)`` order expected by
    ``grid_sample``.
    """

    source_pose = torch.as_tensor(source_pose)
    current_pose = torch.as_tensor(current_pose, device=source_pose.device)
    _validate_pose("source_pose", source_pose)
    _validate_pose("current_pose", current_pose)
    if source_pose.ndim != 2 or source_pose.shape != current_pose.shape:
        raise ValueError("source_pose and current_pose must have matching [N,3] shapes")
    if not dtype.is_floating_point:
        raise TypeError("sampling-grid dtype must be floating point")

    device = source_pose.device
    source_pose = source_pose.to(dtype=dtype)
    current_pose = current_pose.to(dtype=dtype)
    target_xy = geometry.metric_cell_centres(device=device, dtype=dtype)
    target_x = target_xy[..., 0].unsqueeze(0)
    target_y = target_xy[..., 1].unsqueeze(0)

    current_yaw = current_pose[:, 2, None, None]
    current_cos, current_sin = torch.cos(current_yaw), torch.sin(current_yaw)
    global_x = current_cos * target_x - current_sin * target_y
    global_y = current_sin * target_x + current_cos * target_y
    global_x = global_x + current_pose[:, 0, None, None]
    global_y = global_y + current_pose[:, 1, None, None]

    dx = global_x - source_pose[:, 0, None, None]
    dy = global_y - source_pose[:, 1, None, None]
    source_yaw = source_pose[:, 2, None, None]
    source_cos, source_sin = torch.cos(source_yaw), torch.sin(source_yaw)
    source_x = source_cos * dx + source_sin * dy
    source_y = -source_sin * dx + source_cos * dy

    normalized_x = 2.0 * (source_x - geometry.min_x) / (geometry.max_x - geometry.min_x) - 1.0
    normalized_y = 2.0 * (source_y - geometry.min_y) / (geometry.max_y - geometry.min_y) - 1.0
    return torch.stack((normalized_x, normalized_y), dim=-1)


def warp_bev_to_current(
    source_bev: torch.Tensor,
    source_pose: torch.Tensor,
    current_pose: torch.Tensor,
    *,
    geometry: BEVGeometryContract = NATIVE_BEV_GEOMETRY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp source-frame BEV features directly into the current ego frame.

    Args:
        source_bev: ``[B,C,64,64]`` or ``[B,T,C,64,64]``.
        source_pose: ``[B,3]`` or ``[B,T,3]`` matching ``source_bev``.
        current_pose: ``[B,3]`` (broadcast over time) or ``[B,T,3]``.

    Returns:
        A pair ``(aligned_bev, validity)``.  Validity has one channel, lies in
        ``[0,1]``, and records the bilinearly sampled support of each warp.
        Float16/bfloat16 inputs are promoted to float32.
    """

    source_bev = torch.as_tensor(source_bev)
    working_dtype = _working_dtype(source_bev)
    flat_bev, flat_source_pose, flat_current_pose, prefix = _flatten_bev_and_poses(
        source_bev, source_pose, current_pose
    )
    flat_bev = flat_bev.to(dtype=working_dtype)
    grid = source_sampling_grid(
        flat_source_pose,
        flat_current_pose,
        geometry=geometry,
        dtype=working_dtype,
    )
    aligned = F.grid_sample(
        flat_bev,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=geometry.align_corners,
    )
    support = torch.ones(
        (flat_bev.shape[0], 1, geometry.height, geometry.width),
        dtype=working_dtype,
        device=flat_bev.device,
    )
    validity = F.grid_sample(
        support,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=geometry.align_corners,
    ).clamp_(0.0, 1.0)

    channels = aligned.shape[1]
    aligned = aligned.reshape(*prefix, channels, geometry.height, geometry.width)
    validity = validity.reshape(*prefix, 1, geometry.height, geometry.width)
    return aligned, validity


class SE2BEVWarper(nn.Module):
    """Stateless ``nn.Module`` wrapper for :func:`warp_bev_to_current`."""

    def __init__(self, geometry: BEVGeometryContract = NATIVE_BEV_GEOMETRY) -> None:
        super().__init__()
        self.geometry = geometry

    def forward(
        self,
        source_bev: torch.Tensor,
        source_pose: torch.Tensor,
        current_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return warp_bev_to_current(
            source_bev,
            source_pose,
            current_pose,
            geometry=self.geometry,
        )
