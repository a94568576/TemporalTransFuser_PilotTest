"""Offline path metrics with shared baseline-failure subsets.

The helpers in this module intentionally remain sample-oriented.  Route-level
aggregation and paired cluster inference live in :mod:`temporal_tf.route_stats`
so callers cannot accidentally mistake a window count for an independent
sample count.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch


def per_sample_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 2:
        raise ValueError("prediction and target must both be [B,N,2]")
    displacement = torch.linalg.vector_norm(prediction - target, dim=-1)
    waypoint_l1 = (prediction - target).abs().mean(dim=(-1, -2))
    if prediction.shape[1] >= 3:
        acceleration = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
        smoothness = torch.linalg.vector_norm(acceleration, dim=-1).mean(dim=-1)
        target_acceleration = target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2]
        second_difference_error = torch.linalg.vector_norm(
            acceleration - target_acceleration, dim=-1
        ).mean(dim=-1)
    else:
        smoothness = torch.zeros(prediction.shape[0], dtype=prediction.dtype, device=prediction.device)
        second_difference_error = torch.zeros_like(smoothness)
    return {
        "ade": displacement.mean(dim=-1),
        "fde": displacement[:, -1],
        "waypoint_l1": waypoint_l1,
        "smoothness": smoothness,
        "second_difference_error": second_difference_error,
    }


def mean_metrics(
    per_sample: Mapping[str, torch.Tensor], indices: torch.Tensor | None = None
) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, values in per_sample.items():
        selected = values if indices is None else values[indices]
        if selected.numel() == 0:
            raise ValueError(f"metric '{name}' has no selected samples")
        result[name] = float(selected.mean().item())
    return result


def _quantile_label(quantile: float) -> str:
    percentage = quantile * 100.0
    if math.isclose(percentage, round(percentage), abs_tol=1e-9):
        return f"p{int(round(percentage))}"
    return f"p{percentage:g}".replace(".", "_")


def quantile_metrics(
    per_sample: Mapping[str, torch.Tensor],
    *,
    quantiles: Iterable[float] = (0.9, 0.95),
    indices: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    """Return deterministic empirical quantiles for one-dimensional metrics.

    Quantiles are descriptive only.  Route-cluster confidence intervals are
    computed separately by :func:`temporal_tf.route_stats.paired_route_comparison`.
    """

    requested = tuple(float(quantile) for quantile in quantiles)
    if not requested or any(not 0.0 <= quantile <= 1.0 for quantile in requested):
        raise ValueError("quantiles must be a non-empty iterable within [0, 1]")
    labels = tuple(_quantile_label(quantile) for quantile in requested)
    if len(set(labels)) != len(labels):
        raise ValueError("quantiles produce duplicate output labels")

    result: dict[str, dict[str, float]] = {}
    for name, values in per_sample.items():
        selected = values if indices is None else values[indices]
        if selected.ndim != 1 or selected.numel() == 0:
            raise ValueError(f"metric '{name}' must be a non-empty one-dimensional tensor")
        selected = selected.detach().to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(selected).all():
            raise ValueError(f"metric '{name}' contains NaN or Inf")
        computed = torch.quantile(
            selected,
            torch.tensor(requested, dtype=selected.dtype),
            interpolation="linear",
        )
        result[name] = {
            label: float(value.item()) for label, value in zip(labels, computed, strict=True)
        }
    return result


def worst_fraction_indices(baseline_ade: torch.Tensor, fraction: float = 0.2) -> torch.Tensor:
    if baseline_ade.ndim != 1 or baseline_ade.numel() == 0:
        raise ValueError("baseline_ade must be a non-empty vector")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    count = max(1, int(math.ceil(baseline_ade.numel() * fraction)))
    return torch.topk(baseline_ade, k=count, largest=True, sorted=True).indices
