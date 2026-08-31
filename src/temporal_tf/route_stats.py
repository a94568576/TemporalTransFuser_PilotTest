"""Route-macro summaries and paired route-cluster inference.

Sliding windows from a route are correlated.  This module therefore treats a
route, rather than a cached window, as the independent unit for primary
summaries and bootstrap uncertainty.  All public return values are JSON-safe.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .metrics import quantile_metrics


DEFAULT_QUANTILES = (0.9, 0.95)
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000


def _validated_metric_arrays(
    per_sample: Mapping[str, torch.Tensor], route_ids: Sequence[str]
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    normalized_routes = tuple(str(route_id) for route_id in route_ids)
    if not normalized_routes:
        raise ValueError("route_ids must be non-empty")
    if any(not route_id.strip() for route_id in normalized_routes):
        raise ValueError("route_ids must not contain empty identifiers")
    if not per_sample:
        raise ValueError("per_sample metrics must be non-empty")

    arrays: dict[str, np.ndarray] = {}
    expected = len(normalized_routes)
    for name, values in per_sample.items():
        tensor = torch.as_tensor(values).detach().to(device="cpu", dtype=torch.float64)
        if tensor.ndim != 1 or tensor.numel() != expected:
            raise ValueError(
                f"metric '{name}' must be one-dimensional with {expected} samples, "
                f"got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"metric '{name}' contains NaN or Inf")
        arrays[str(name)] = tensor.numpy().copy()
    return arrays, normalized_routes


def _route_metric_matrix(
    arrays: Mapping[str, np.ndarray], route_ids: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    metric_names = tuple(arrays)
    routes = tuple(sorted(set(route_ids)))
    matrix = np.empty((len(routes), len(metric_names)), dtype=np.float64)
    route_array = np.asarray(route_ids, dtype=object)
    for route_index, route_id in enumerate(routes):
        mask = route_array == route_id
        for metric_index, metric_name in enumerate(metric_names):
            matrix[route_index, metric_index] = float(arrays[metric_name][mask].mean())
    return routes, metric_names, matrix


def _matrix_as_per_route(
    routes: Sequence[str], metric_names: Sequence[str], matrix: np.ndarray
) -> dict[str, dict[str, float]]:
    return {
        route_id: {
            metric_name: float(matrix[route_index, metric_index])
            for metric_index, metric_name in enumerate(metric_names)
        }
        for route_index, route_id in enumerate(routes)
    }


def _matrix_quantiles(
    metric_names: Sequence[str], matrix: np.ndarray, quantiles: Sequence[float]
) -> dict[str, dict[str, float]]:
    tensors = {
        metric_name: torch.from_numpy(matrix[:, metric_index].copy())
        for metric_index, metric_name in enumerate(metric_names)
    }
    return quantile_metrics(tensors, quantiles=quantiles)


def summarize_route_metrics(
    per_sample: Mapping[str, torch.Tensor],
    route_ids: Sequence[str],
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    include_per_route: bool = True,
) -> dict[str, Any]:
    """Summarize metrics with equally weighted routes as the primary unit.

    ``sample_quantiles`` are retained as useful tail diagnostics, while
    ``route_macro`` and ``route_quantiles`` prevent long routes from receiving
    more weight merely because they generate more overlapping windows.
    """

    arrays, normalized_routes = _validated_metric_arrays(per_sample, route_ids)
    routes, metric_names, route_matrix = _route_metric_matrix(arrays, normalized_routes)
    result: dict[str, Any] = {
        "route_count": len(routes),
        "route_macro": {
            metric_name: float(route_matrix[:, metric_index].mean())
            for metric_index, metric_name in enumerate(metric_names)
        },
        "route_quantiles": _matrix_quantiles(metric_names, route_matrix, quantiles),
        "sample_quantiles": quantile_metrics(per_sample, quantiles=quantiles),
    }
    if include_per_route:
        result["per_route_metrics"] = _matrix_as_per_route(routes, metric_names, route_matrix)
    return result


def _fraction_summary(values: np.ndarray, tolerance: float) -> dict[str, float]:
    improved = values < -tolerance
    harmed = values > tolerance
    unchanged = ~(improved | harmed)
    return {
        "improved": float(improved.mean()),
        "harmed": float(harmed.mean()),
        "unchanged": float(unchanged.mean()),
    }


def _cluster_bootstrap_cis(
    route_delta_matrix: np.ndarray,
    metric_names: Sequence[str],
    *,
    seed: int,
    num_resamples: int,
    confidence_level: float,
) -> dict[str, dict[str, Any]]:
    if num_resamples < 1:
        raise ValueError("num_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be within (0, 1)")

    route_count, metric_count = route_delta_matrix.shape
    if route_count == 1:
        estimates = np.repeat(route_delta_matrix, num_resamples, axis=0)
    else:
        rng = np.random.default_rng(int(seed))
        estimates = np.empty((num_resamples, metric_count), dtype=np.float64)
        # Bound the temporary index matrix for large route collections while
        # retaining a single deterministic RNG stream.
        chunk_size = max(1, min(num_resamples, 2048))
        for start in range(0, num_resamples, chunk_size):
            stop = min(start + chunk_size, num_resamples)
            sampled_indices = rng.integers(
                0, route_count, size=(stop - start, route_count), endpoint=False
            )
            estimates[start:stop] = route_delta_matrix[sampled_indices].mean(axis=1)

    alpha = (1.0 - confidence_level) / 2.0
    bounds = np.quantile(estimates, (alpha, 1.0 - alpha), axis=0, method="linear")
    observed = route_delta_matrix.mean(axis=0)
    return {
        metric_name: {
            "estimate": float(observed[metric_index]),
            "lower": float(bounds[0, metric_index]),
            "upper": float(bounds[1, metric_index]),
            "confidence_level": float(confidence_level),
            "num_resamples": int(num_resamples),
            "seed": int(seed),
            "cluster_unit": "route",
            "method": "percentile",
        }
        for metric_index, metric_name in enumerate(metric_names)
    }


def paired_route_comparison(
    adapter_per_sample: Mapping[str, torch.Tensor],
    baseline_per_sample: Mapping[str, torch.Tensor],
    route_ids: Sequence[str],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
    primary_metric: str = "ade",
    improvement_tolerance: float = 0.0,
    include_per_route: bool = True,
) -> dict[str, Any]:
    """Compare paired adapter/baseline windows using route-level deltas.

    Deltas are always ``adapter - baseline``; negative values therefore mean
    improvement for the error metrics used by this project.  Pair identity and
    ordering should be checked by the caller using the preserved ``sample_ids``.
    """

    if improvement_tolerance < 0.0:
        raise ValueError("improvement_tolerance must be non-negative")
    adapter_arrays, normalized_routes = _validated_metric_arrays(adapter_per_sample, route_ids)
    baseline_arrays, baseline_routes = _validated_metric_arrays(baseline_per_sample, route_ids)
    if normalized_routes != baseline_routes:
        raise ValueError("adapter and baseline route ordering differs")
    if set(adapter_arrays) != set(baseline_arrays):
        raise ValueError("adapter and baseline metric names differ")
    if primary_metric not in adapter_arrays:
        raise ValueError(f"primary metric '{primary_metric}' is unavailable")

    delta_arrays = {
        metric_name: adapter_arrays[metric_name] - baseline_arrays[metric_name]
        for metric_name in adapter_arrays
    }
    routes, metric_names, route_delta_matrix = _route_metric_matrix(
        delta_arrays, normalized_routes
    )
    primary_index = metric_names.index(primary_metric)
    primary_sample_delta = delta_arrays[primary_metric]
    primary_route_delta = route_delta_matrix[:, primary_index]
    result: dict[str, Any] = {
        "delta_definition": "adapter_minus_baseline",
        "primary_metric": primary_metric,
        "route_count": len(routes),
        "route_macro_delta": {
            metric_name: float(route_delta_matrix[:, metric_index].mean())
            for metric_index, metric_name in enumerate(metric_names)
        },
        "route_bootstrap_ci": _cluster_bootstrap_cis(
            route_delta_matrix,
            metric_names,
            seed=bootstrap_seed,
            num_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        ),
        "sample_fractions": _fraction_summary(primary_sample_delta, improvement_tolerance),
        "route_fractions": _fraction_summary(primary_route_delta, improvement_tolerance),
        "improvement_tolerance": float(improvement_tolerance),
    }
    if include_per_route:
        result["per_route_delta"] = _matrix_as_per_route(
            routes, metric_names, route_delta_matrix
        )
    return result
