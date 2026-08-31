"""Deterministic synthetic cache used only to exercise the complete pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cache import CacheWriter


def _world_path(s: np.ndarray, amplitude: float, phase: float) -> np.ndarray:
    x = s
    y = amplitude * np.sin(0.035 * s + phase) + 0.25 * np.sin(0.11 * s + 0.5 * phase)
    return np.stack((x, y), axis=-1)


def _yaw_at(s: float, amplitude: float, phase: float) -> float:
    derivative = amplitude * 0.035 * math.cos(0.035 * s + phase)
    derivative += 0.25 * 0.11 * math.cos(0.11 * s + 0.5 * phase)
    return math.atan2(derivative, 1.0)


def _to_local(points: np.ndarray, position: np.ndarray, yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation_t = np.array([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return (rotation_t @ (points - position).T).T


def _bev_feature(
    *,
    channels: int,
    size: int,
    amplitude: float,
    yaw: float,
    bias: np.ndarray,
    phase: float,
) -> torch.Tensor:
    grid = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    feature = np.zeros((channels, size, size), dtype=np.float32)
    feature[0] = amplitude / 3.0
    feature[1] = yaw
    feature[2] = bias[0]
    feature[3] = bias[1]
    feature[4] = np.sin(3.0 * xx + phase)
    feature[5] = np.cos(3.0 * yy - phase)
    feature[6] = np.exp(-((xx - 0.25 * bias[0]) ** 2 + (yy - 0.25 * bias[1]) ** 2) / 0.2)
    feature[7] = xx * yy
    for channel in range(8, channels):
        feature[channel] = np.sin((channel + 1) * 0.15 * xx + phase) * np.cos(
            (channel + 1) * 0.13 * yy - phase
        )
    return torch.from_numpy(feature)


def generate_synthetic_cache(
    destination: str | Path,
    *,
    num_routes: int = 15,
    frames_per_route: int = 32,
    num_waypoints: int = 10,
    bev_channels: int = 16,
    bev_size: int = 12,
    seed: int = 17,
) -> Path:
    """Create temporal signals with a deliberately learnable residual.

    The cache exists to verify data flow, gradients, ablations, metrics, and
    reporting.  Its performance must never be interpreted as CARLA evidence.
    """

    if num_routes < 3:
        raise ValueError("at least three routes are needed for train/val/test")
    if frames_per_route < 6:
        raise ValueError("frames_per_route must leave room for temporal windows")
    if num_waypoints < 1:
        raise ValueError("num_waypoints must be positive")
    if bev_channels < 8:
        raise ValueError("bev_channels must be at least 8 for the smoke generator")
    if bev_size < 2:
        raise ValueError("bev_size must be at least 2")
    rng = np.random.default_rng(seed)
    writer = CacheWriter(Path(destination))
    waypoint_distances = 2.5 + np.arange(num_waypoints, dtype=np.float64)

    for route_index in range(num_routes):
        route_id = f"synthetic_route_{route_index:03d}"
        amplitude = float(rng.uniform(-2.5, 2.5))
        phase = float(rng.uniform(-math.pi, math.pi))
        bias = rng.normal(0.0, 0.12, size=2)
        for frame_id in range(frames_per_route):
            s = frame_id * 0.8 + route_index * 0.1
            position = _world_path(np.array([s]), amplitude, phase)[0]
            yaw = _yaw_at(s, amplitude, phase)
            future_world = _world_path(s + waypoint_distances, amplitude, phase)
            gt = _to_local(future_world, position, yaw).astype(np.float32)

            # Slowly varying frozen-model bias plus flicker.  Past predictions
            # and past scene features provide complementary smoke-test signals.
            innovation = rng.normal(0.0, 0.035, size=2)
            bias = 0.88 * bias + innovation
            horizon_scale = np.linspace(0.35, 1.0, num_waypoints, dtype=np.float32)[:, None]
            structured_error = horizon_scale * bias.astype(np.float32)[None, :]
            lateral_calibration = np.array([0.0, 0.06 * amplitude], dtype=np.float32)
            prediction = gt + structured_error + horizon_scale * lateral_calibration
            prediction += rng.normal(0.0, 0.018, size=prediction.shape).astype(np.float32)

            bev = _bev_feature(
                channels=bev_channels,
                size=bev_size,
                amplitude=amplitude,
                yaw=yaw,
                bias=bias,
                phase=phase + 0.025 * frame_id,
            )
            writer.add(
                {
                    "bev_feature": bev,
                    "pred_trajectory": torch.from_numpy(prediction),
                    "gt_trajectory": torch.from_numpy(gt),
                    "ego_pose": torch.tensor([position[0], position[1], yaw], dtype=torch.float64),
                    "route_id": route_id,
                    "frame_id": frame_id,
                    "timestamp": frame_id * 0.1,
                    "trajectory_source": "frozen_model_prediction",
                    "metadata": {"synthetic": True},
                }
            )
    return writer.finalize(
        source={
            "kind": "synthetic",
            "target_semantics": "geometric_path",
            "warning": "pipeline smoke only; not research evidence",
            "generator_seed": seed,
        },
        split_seed=seed,
    )
