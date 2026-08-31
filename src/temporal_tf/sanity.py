"""Read-only sanity analysis for frozen-planner cache artifacts.

This module deliberately performs no fitting, model selection, or tuning.  It
checks raw cached frames and summarizes the frozen prediction against its
cached target with both sample-weighted and equal-route-weighted statistics.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .cache import load_index, sha256_file, validate_record


SANITY_SCHEMA_VERSION = 1
METRIC_NAMES = ("ade", "fde", "point_l1", "smoothness")
SUMMARY_NAMES = ("mean", "p90", "p95")
ALLOWED_CACHE_SPLITS = frozenset({"train", "val", "test"})
TEST_MARKER_FILENAMES = frozenset(
    {"test_opened.marker.json", "study_test_opened.marker.json"}
)


def _validated_splits(splits: set[str] | None) -> set[str] | None:
    if splits is None:
        return None
    normalized = {str(split) for split in splits}
    if not normalized:
        raise ValueError("splits must not be empty")
    unknown = normalized.difference(ALLOWED_CACHE_SPLITS)
    if unknown:
        raise ValueError(f"splits contains unsupported values: {sorted(unknown)}")
    return normalized


def raw_frame_metrics(prediction: Any, target: Any) -> dict[str, float]:
    """Compute model-free path metrics for one raw cache frame.

    ``smoothness`` is the mean Euclidean norm of the prediction's second
    spatial difference.  The checkpoints can be geometric rather than
    time-sampled, so this is intentionally not called acceleration or jerk.
    """

    prediction_tensor = torch.as_tensor(prediction, dtype=torch.float64)
    target_tensor = torch.as_tensor(target, dtype=torch.float64)
    if (
        prediction_tensor.shape != target_tensor.shape
        or prediction_tensor.ndim != 2
        or prediction_tensor.shape[-1] != 2
        or prediction_tensor.shape[0] < 1
    ):
        raise ValueError("prediction and target must have matching non-empty [N,2] shapes")
    if not torch.isfinite(prediction_tensor).all() or not torch.isfinite(target_tensor).all():
        raise ValueError("prediction and target must be finite")

    error = prediction_tensor - target_tensor
    displacement = torch.linalg.vector_norm(error, dim=-1)
    if prediction_tensor.shape[0] >= 3:
        second_difference = (
            prediction_tensor[2:]
            - 2.0 * prediction_tensor[1:-1]
            + prediction_tensor[:-2]
        )
        smoothness = torch.linalg.vector_norm(second_difference, dim=-1).mean()
    else:
        smoothness = torch.zeros((), dtype=torch.float64)
    return {
        "ade": float(displacement.mean().item()),
        "fde": float(displacement[-1].item()),
        "point_l1": float(error.abs().mean().item()),
        "smoothness": float(smoothness.item()),
    }


def _distribution_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty metric collection")
    tensor = torch.tensor(values, dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError("metric collection contains NaN or Inf")
    quantiles = torch.quantile(
        tensor,
        torch.tensor([0.9, 0.95], dtype=torch.float64),
        interpolation="linear",
    )
    return {
        "count": len(values),
        "mean": float(tensor.mean().item()),
        "p90": float(quantiles[0].item()),
        "p95": float(quantiles[1].item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def aggregate_raw_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame metrics without allowing long routes to dominate macro results."""

    if not samples:
        raise ValueError("cannot aggregate zero valid samples")
    by_route: dict[str, list[dict[str, float]]] = defaultdict(list)
    for sample in samples:
        route_id = str(sample["route_id"])
        metrics = sample["metrics"]
        if set(metrics) != set(METRIC_NAMES):
            raise ValueError(f"sample has unexpected metric keys: {sorted(metrics)}")
        by_route[route_id].append(metrics)

    sample_micro = {
        metric: _distribution_summary(
            [float(sample["metrics"][metric]) for sample in samples]
        )
        for metric in METRIC_NAMES
    }
    per_route: dict[str, Any] = {}
    route_means: dict[str, list[float]] = {metric: [] for metric in METRIC_NAMES}
    for route_id, route_samples in sorted(by_route.items()):
        route_metrics = {
            metric: _distribution_summary(
                [float(sample_metrics[metric]) for sample_metrics in route_samples]
            )
            for metric in METRIC_NAMES
        }
        per_route[route_id] = {
            "samples": len(route_samples),
            "metrics": route_metrics,
        }
        for metric in METRIC_NAMES:
            route_means[metric].append(float(route_metrics[metric]["mean"]))

    equal_route_macro = {
        metric: _distribution_summary(route_means[metric]) for metric in METRIC_NAMES
    }
    return {
        "sample_micro": sample_micro,
        "equal_route_macro": equal_route_macro,
        "per_route": per_route,
    }


def _sequence_checks(
    entries: list[Mapping[str, Any]],
    *,
    expected_frame_step: int,
    cadence_tolerance: float,
) -> dict[str, Any]:
    frames = [int(entry["frame_id"]) for entry in entries]
    timestamps = [float(entry["timestamp"]) for entry in entries]
    frame_duplicates: list[list[int]] = []
    frame_nonmonotonic: list[list[int]] = []
    frame_gaps: list[list[int]] = []
    frame_step_mismatches: list[list[int]] = []
    timestamp_duplicates: list[list[float]] = []
    timestamp_nonmonotonic: list[list[float]] = []
    rates: list[float] = []

    for left_frame, right_frame, left_time, right_time in zip(
        frames[:-1], frames[1:], timestamps[:-1], timestamps[1:], strict=True
    ):
        frame_delta = right_frame - left_frame
        time_delta = right_time - left_time
        if frame_delta == 0:
            frame_duplicates.append([left_frame, right_frame])
        elif frame_delta < 0:
            frame_nonmonotonic.append([left_frame, right_frame])
        elif frame_delta > expected_frame_step:
            frame_gaps.append([left_frame, right_frame])
        elif frame_delta != expected_frame_step:
            frame_step_mismatches.append([left_frame, right_frame])

        if time_delta == 0.0:
            timestamp_duplicates.append([left_time, right_time])
        elif time_delta < 0.0:
            timestamp_nonmonotonic.append([left_time, right_time])
        if frame_delta > 0 and time_delta > 0.0:
            rates.append(time_delta / frame_delta)

    cadence = statistics.median(rates) if rates else None
    cadence_mismatches: list[float] = []
    if cadence is not None:
        cadence_mismatches = [
            rate for rate in rates if abs(rate - cadence) > cadence_tolerance
        ]
    return {
        "records": len(entries),
        "frame": {
            "first": frames[0] if frames else None,
            "last": frames[-1] if frames else None,
            "expected_step": expected_frame_step,
            "duplicate_count": len(frame_duplicates),
            "nonmonotonic_count": len(frame_nonmonotonic),
            "gap_count": len(frame_gaps),
            "step_mismatch_count": len(frame_step_mismatches),
            "duplicate_examples": frame_duplicates[:20],
            "nonmonotonic_examples": frame_nonmonotonic[:20],
            "gap_examples": frame_gaps[:20],
            "step_mismatch_examples": frame_step_mismatches[:20],
        },
        "timestamp": {
            "first": timestamps[0] if timestamps else None,
            "last": timestamps[-1] if timestamps else None,
            "seconds_per_frame_median": cadence,
            "cadence_tolerance": cadence_tolerance,
            "duplicate_count": len(timestamp_duplicates),
            "nonmonotonic_count": len(timestamp_nonmonotonic),
            "cadence_mismatch_count": len(cadence_mismatches),
            "duplicate_examples": timestamp_duplicates[:20],
            "nonmonotonic_examples": timestamp_nonmonotonic[:20],
            "cadence_mismatch_examples": cadence_mismatches[:20],
        },
    }


def _reference_metrics(reference: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = reference.get("metrics", reference)
    if not isinstance(metrics, Mapping):
        raise ValueError("reference JSON must contain a metrics mapping")
    return metrics


def compare_metric_reference(
    observed: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Compare requested aggregate statistics against a JSON reference."""

    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("reference tolerance must be a finite non-negative number")
    expected_metrics = _reference_metrics(reference)
    comparisons: list[dict[str, Any]] = []
    errors: list[str] = []
    supported_aggregations = {"sample_micro", "equal_route_macro", "per_route"}
    unknown_aggregations = set(expected_metrics).difference(supported_aggregations)
    if unknown_aggregations:
        errors.append(
            f"reference contains unknown metric aggregations: {sorted(unknown_aggregations)}"
        )
    for aggregation in ("sample_micro", "equal_route_macro"):
        expected_aggregation = expected_metrics.get(aggregation)
        if expected_aggregation is None:
            continue
        if not isinstance(expected_aggregation, Mapping):
            errors.append(f"reference {aggregation} must be a mapping")
            continue
        observed_aggregation = observed.get(aggregation, {})
        for metric, expected_summary in expected_aggregation.items():
            if metric not in METRIC_NAMES:
                errors.append(f"reference {aggregation} contains unknown metric: {metric}")
                continue
            if not isinstance(expected_summary, Mapping):
                errors.append(f"reference {aggregation}.{metric} must be a mapping")
                continue
            supported_summaries = {"count", "mean", "p90", "p95", "min", "max"}
            unknown_summaries = set(expected_summary).difference(supported_summaries)
            if unknown_summaries:
                errors.append(
                    f"reference {aggregation}.{metric} contains unknown summaries: "
                    f"{sorted(unknown_summaries)}"
                )
            if not set(SUMMARY_NAMES).intersection(expected_summary):
                errors.append(
                    f"reference {aggregation}.{metric} has no comparable mean/p90/p95 values"
                )
                continue
            observed_summary = observed_aggregation.get(metric, {})
            for summary_name in SUMMARY_NAMES:
                if summary_name not in expected_summary:
                    continue
                path = f"metrics.{aggregation}.{metric}.{summary_name}"
                if summary_name not in observed_summary:
                    errors.append(f"reference path missing from observed report: {path}")
                    continue
                expected_value = float(expected_summary[summary_name])
                observed_value = float(observed_summary[summary_name])
                absolute_difference = abs(observed_value - expected_value)
                passed = (
                    math.isfinite(expected_value)
                    and math.isfinite(observed_value)
                    and absolute_difference <= tolerance
                )
                comparisons.append(
                    {
                        "path": path,
                        "observed": observed_value,
                        "reference": expected_value,
                        "absolute_difference": absolute_difference,
                        "tolerance": tolerance,
                        "pass": passed,
                    }
                )
                if not passed:
                    errors.append(
                        f"reference mismatch at {path}: observed={observed_value}, "
                        f"reference={expected_value}, tolerance={tolerance}"
                    )
    if not comparisons and not errors:
        errors.append("reference contains no comparable mean/p90/p95 metric values")
    return {
        "status": "pass" if not errors else "fail",
        "tolerance": tolerance,
        "comparisons": comparisons,
        "errors": errors,
    }


def analyze_cache(
    cache_root: str | Path,
    *,
    reference: Mapping[str, Any] | str | Path | None = None,
    reference_tolerance: float = 1e-6,
    expected_frame_step: int = 1,
    cadence_tolerance: float = 1e-6,
    splits: set[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable integrity and raw-metric sanity report."""

    selected_splits = _validated_splits(splits)
    selected_scope: list[str] | str = (
        sorted(selected_splits) if selected_splits is not None else "all"
    )
    root = Path(cache_root)
    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "schema_version": SANITY_SCHEMA_VERSION,
        "kind": "temporal_tf_cache_sanity",
        "cache_root": str(root.resolve()),
        "parameters": {
            "expected_frame_step": expected_frame_step,
            "cadence_tolerance": cadence_tolerance,
            "reference_tolerance": reference_tolerance,
            "selected_splits": selected_scope,
        },
        "status": "fail",
        "warnings": warnings,
        "errors": errors,
    }
    if expected_frame_step < 1:
        errors.append("expected_frame_step must be positive")
        return report
    if not math.isfinite(cadence_tolerance) or cadence_tolerance < 0.0:
        errors.append("cadence_tolerance must be finite and non-negative")
        return report

    try:
        index = load_index(root)
    except Exception as exc:
        errors.append(f"index: {exc}")
        return report

    all_entries = index["records"]
    entries = [
        entry
        for entry in all_entries
        if selected_splits is None or str(entry["split"]) in selected_splits
    ]
    by_route: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    split_record_counts: Counter[str] = Counter()
    split_routes: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        route_id = str(entry["route_id"])
        split = str(entry["split"])
        by_route[route_id].append(entry)
        split_record_counts[split] += 1
        split_routes[split].add(route_id)

    split_policy = index.get("split_policy", {})
    ratios = split_policy.get("ratios", {}) if isinstance(split_policy, Mapping) else {}
    positive_split_ratios: set[str] = set()
    if isinstance(ratios, Mapping):
        try:
            positive_split_ratios = {
                str(split) for split, ratio in ratios.items() if float(ratio) > 0.0
            }
        except (TypeError, ValueError) as exc:
            errors.append(f"invalid split ratio metadata: {exc}")
    else:
        errors.append("invalid split ratio metadata: ratios must be a mapping")
    expected_splits = (
        selected_splits
        if selected_splits is not None
        else positive_split_ratios or {"train", "val", "test"}
    )
    missing_splits = sorted(expected_splits.difference(split_record_counts))
    if missing_splits:
        if selected_splits is not None:
            errors.append(f"empty selected splits: {missing_splits}")
        elif len(by_route) < len(expected_splits):
            if len(by_route) == 1:
                warning_prefix = "single-route cache"
            else:
                warning_prefix = f"cache with only {len(by_route)} routes"
            warnings.append(
                f"{warning_prefix} cannot populate {len(expected_splits)} route-level splits; "
                f"empty splits are warning-only: {missing_splits}"
            )
        else:
            errors.append(f"empty required splits: {missing_splits}")

    sequence_by_route: dict[str, Any] = {}
    aggregate_sequence_counts: Counter[str] = Counter()
    for route_id, route_entries in sorted(by_route.items()):
        sequence = _sequence_checks(
            route_entries,
            expected_frame_step=expected_frame_step,
            cadence_tolerance=cadence_tolerance,
        )
        sequence["split"] = str(route_entries[0]["split"])
        sequence_by_route[route_id] = sequence
        for name in ("duplicate_count", "nonmonotonic_count", "gap_count", "step_mismatch_count"):
            aggregate_sequence_counts[f"frame_{name}"] += int(sequence["frame"][name])
        for name in ("duplicate_count", "nonmonotonic_count", "cadence_mismatch_count"):
            aggregate_sequence_counts[f"timestamp_{name}"] += int(sequence["timestamp"][name])

    for check_name, count in aggregate_sequence_counts.items():
        if count:
            errors.append(f"{check_name}: {count}")

    shape_counts: dict[str, Counter[str]] = {
        "bev_feature": Counter(),
        "pred_trajectory": Counter(),
        "gt_trajectory": Counter(),
        "ego_pose": Counter(),
    }
    samples: list[dict[str, Any]] = []
    hash_failures = 0
    invalid_records = 0
    identity_mismatches = 0
    timestamp_mismatches = 0
    for entry in entries:
        relative = Path(str(entry["path"]))
        path = root / relative
        if not path.is_file():
            errors.append(f"missing record: {path}")
            invalid_records += 1
            continue
        try:
            actual_hash = sha256_file(path)
        except OSError as exc:
            errors.append(f"cannot hash record {path}: {exc}")
            invalid_records += 1
            continue
        if actual_hash != str(entry["sha256"]):
            errors.append(f"record SHA256 mismatch: {path}")
            hash_failures += 1
            invalid_records += 1
            continue
        try:
            record = torch.load(path, map_location="cpu", weights_only=True)
            validate_record(record)
        except Exception as exc:
            errors.append(f"invalid record {path}: {exc}")
            invalid_records += 1
            continue

        record_integrity_invalid = False
        if record.get("schema_version") != index["schema_version"]:
            errors.append(
                f"index/record schema mismatch: {path} "
                f"({record.get('schema_version')} != {index['schema_version']})"
            )
            record_integrity_invalid = True

        if (
            str(record["route_id"]) != str(entry["route_id"])
            or int(record["frame_id"]) != int(entry["frame_id"])
        ):
            errors.append(f"index/record identity mismatch: {path}")
            identity_mismatches += 1
            record_integrity_invalid = True
        if "timestamp" not in record:
            errors.append(f"cached record is missing canonical timestamp: {path}")
            timestamp_mismatches += 1
            record_integrity_invalid = True
        elif not math.isclose(
            float(record["timestamp"]),
            float(entry["timestamp"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append(f"index/record timestamp mismatch: {path}")
            timestamp_mismatches += 1
            record_integrity_invalid = True
        if record_integrity_invalid:
            invalid_records += 1
            continue

        for field in shape_counts:
            shape_counts[field][str(tuple(torch.as_tensor(record[field]).shape))] += 1
        try:
            metrics = raw_frame_metrics(record["pred_trajectory"], record["gt_trajectory"])
        except Exception as exc:
            errors.append(f"metric computation failed for {path}: {exc}")
            invalid_records += 1
            continue
        samples.append(
            {
                "route_id": str(entry["route_id"]),
                "split": str(entry["split"]),
                "frame_id": int(entry["frame_id"]),
                "timestamp": float(entry["timestamp"]),
                "metrics": metrics,
            }
        )

    heterogeneous_shapes: dict[str, dict[str, int]] = {}
    for field, counts in shape_counts.items():
        if len(counts) > 1:
            heterogeneous_shapes[field] = dict(counts)
            errors.append(f"heterogeneous {field} shapes: {dict(counts)}")

    report["index"] = {
        "schema_version": index["schema_version"],
        "records": len(entries),
        "total_index_records": len(all_entries),
        "routes": len(by_route),
        "selected_splits": selected_scope,
        "split_record_counts": {
            split: split_record_counts[split] for split in ("train", "val", "test")
        },
        "split_route_counts": {
            split: len(split_routes[split]) for split in ("train", "val", "test")
        },
        "source": index.get("source", {}),
    }
    report["cadence"] = {
        "aggregate_counts": dict(aggregate_sequence_counts),
        "routes": sequence_by_route,
    }
    report["checks"] = {
        "records_expected": len(entries),
        "records_valid": len(samples),
        "invalid_records": invalid_records,
        "hash_failures": hash_failures,
        "identity_mismatches": identity_mismatches,
        "timestamp_mismatches": timestamp_mismatches,
        "shape_counts": {field: dict(counts) for field, counts in shape_counts.items()},
        "heterogeneous_shapes": heterogeneous_shapes,
        "all_valid_tensors_finite": invalid_records == 0,
    }
    report["metric_definitions"] = {
        "ade": "mean Euclidean point displacement between frozen prediction and GT",
        "fde": "Euclidean displacement at the final cached point",
        "point_l1": "mean absolute coordinate error over all cached points",
        "smoothness": "mean Euclidean second spatial difference of frozen prediction",
        "aggregation": {
            "sample_micro": "all raw frames weighted equally",
            "equal_route_macro": "per-route means weighted equally; p90/p95 over route means",
        },
    }
    if samples:
        report["metrics"] = aggregate_raw_metrics(samples)
    else:
        report["metrics"] = {}
        errors.append("no valid records available for raw metric aggregation")

    if reference is not None and report["metrics"]:
        try:
            if isinstance(reference, Mapping):
                reference_data = reference
                reference_path = None
            else:
                reference_path = str(Path(reference).resolve())
                reference_data = json.loads(Path(reference).read_text(encoding="utf-8"))
            comparison = compare_metric_reference(
                report["metrics"], reference_data, tolerance=reference_tolerance
            )
            comparison["reference_path"] = reference_path
            report["reference_comparison"] = comparison
            errors.extend(comparison["errors"])
        except Exception as exc:
            errors.append(f"reference comparison: {exc}")

    report["status"] = "pass" if not errors else "fail"
    return report


def deterministic_sample_entries(
    index: Mapping[str, Any],
    *,
    max_samples: int = 20,
    splits: set[str] | None = None,
) -> list[Mapping[str, Any]]:
    """Choose stable, evenly spaced records from route/frame sorted entries."""

    if not 1 <= max_samples <= 20:
        raise ValueError("max_samples must be within [1, 20]")
    selected_splits = _validated_splits(splits)
    records = index.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("index must contain records")
    scoped_records = [
        entry
        for entry in records
        if selected_splits is None or str(entry["split"]) in selected_splits
    ]
    if not scoped_records:
        scope = sorted(selected_splits) if selected_splits is not None else "all"
        raise ValueError(f"index contains no records for selected splits: {scope}")
    ordered = sorted(
        scoped_records,
        key=lambda entry: (
            str(entry["route_id"]),
            int(entry["frame_id"]),
            float(entry["timestamp"]),
            str(entry["path"]),
        ),
    )
    if len(ordered) <= max_samples:
        return ordered
    if max_samples == 1:
        return [ordered[len(ordered) // 2]]
    positions = [
        round(offset * (len(ordered) - 1) / (max_samples - 1))
        for offset in range(max_samples)
    ]
    return [ordered[position] for position in positions]


def route_safe_history_entries(
    index: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    history_length: int,
) -> list[Mapping[str, Any]]:
    """Return only earlier records from the current route and split."""

    if history_length < 0:
        raise ValueError("history_length must be non-negative")
    if history_length == 0:
        return []
    route_id = str(current["route_id"])
    split = str(current["split"])
    current_key = (int(current["frame_id"]), float(current["timestamp"]))
    candidates = [
        entry
        for entry in index["records"]
        if str(entry["route_id"]) == route_id
        and str(entry["split"]) == split
        and (int(entry["frame_id"]), float(entry["timestamp"])) < current_key
    ]
    candidates.sort(key=lambda entry: (int(entry["frame_id"]), float(entry["timestamp"])))
    return candidates[-history_length:]


def write_json_artifact(payload: Mapping[str, Any], output: str | Path) -> Path:
    """Atomically write a deterministic JSON artifact."""

    path = Path(output)
    if path.name in TEST_MARKER_FILENAMES:
        raise ValueError(f"refusing to overwrite a permanent test marker: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
