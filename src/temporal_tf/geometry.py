"""Coordinate transforms used by the temporal cache dataset."""

from __future__ import annotations

import torch


def transform_trajectory_between_egos(
    trajectory: torch.Tensor,
    source_pose: torch.Tensor,
    target_pose: torch.Tensor,
) -> torch.Tensor:
    """Express source-ego waypoints in the target ego coordinate frame.

    ``trajectory`` is ``[..., N, 2]`` and poses are ``[..., 3]`` containing
    global ``x, y, yaw`` with yaw in radians.  The convention matches the
    released TransFuser++ ``inverse_conversion_2d`` helper: local coordinates
    are obtained with ``R(yaw).T @ (global - translation)``.
    """

    trajectory = torch.as_tensor(trajectory)
    source_pose = torch.as_tensor(source_pose, dtype=trajectory.dtype, device=trajectory.device)
    target_pose = torch.as_tensor(target_pose, dtype=trajectory.dtype, device=trajectory.device)

    if trajectory.shape[-1] != 2:
        raise ValueError(f"trajectory must end in 2 coordinates, got {tuple(trajectory.shape)}")
    if source_pose.shape[-1] != 3 or target_pose.shape[-1] != 3:
        raise ValueError("source_pose and target_pose must end in (x, y, yaw)")

    source_yaw = source_pose[..., 2]
    target_yaw = target_pose[..., 2]
    source_cos, source_sin = torch.cos(source_yaw), torch.sin(source_yaw)
    target_cos, target_sin = torch.cos(target_yaw), torch.sin(target_yaw)

    x_local, y_local = trajectory.unbind(dim=-1)
    x_global = source_cos[..., None] * x_local - source_sin[..., None] * y_local
    y_global = source_sin[..., None] * x_local + source_cos[..., None] * y_local
    x_global = x_global + source_pose[..., 0, None]
    y_global = y_global + source_pose[..., 1, None]

    dx = x_global - target_pose[..., 0, None]
    dy = y_global - target_pose[..., 1, None]
    x_target = target_cos[..., None] * dx + target_sin[..., None] * dy
    y_target = -target_sin[..., None] * dx + target_cos[..., None] * dy
    return torch.stack((x_target, y_target), dim=-1)
