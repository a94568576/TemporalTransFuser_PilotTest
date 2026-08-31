"""Cache integrity and anti-leakage audit used by both CLI and engine."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .cache import load_index, sha256_file, validate_record


FORBIDDEN_INPUT_NAMES = {
    "past_gt_trajectory",
    "future_waypoints",
    "future_pose",
    "oracle_grid",
    "actor_gt",
    "failure_label",
}
ALLOWED_CACHE_SPLITS = frozenset({"train", "val", "test"})


def _validated_splits(splits: set[str] | None, *, name: str) -> set[str] | None:
    if splits is None:
        return None
    normalized = {str(split) for split in splits}
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    unknown = normalized.difference(ALLOWED_CACHE_SPLITS)
    if unknown:
        raise ValueError(f"{name} contains unsupported splits: {sorted(unknown)}")
    return normalized


def audit_cache(
    cache_root: str | Path,
    *,
    deep: bool,
    deep_splits: set[str] | None = None,
    splits: set[str] | None = None,
) -> dict[str, Any]:
    selected_splits = _validated_splits(splits, name="splits")
    deep_splits = _validated_splits(deep_splits, name="deep_splits")
    selected_scope: list[str] | str = (
        sorted(selected_splits) if selected_splits is not None else "all"
    )
    cache_root = Path(cache_root)
    try:
        index = load_index(cache_root)
    except Exception as exc:
        return {
            "status": "fail",
            "deep": deep,
            "selected_splits": selected_scope,
            "deep_splits": sorted(deep_splits) if deep_splits is not None else "all",
            "records": 0,
            "total_index_records": 0,
            "routes": 0,
            "split_record_counts": {},
            "bev_shapes": {},
            "trajectory_shapes": {},
            "input_analysis": {
                "checked_records": 0,
                "presence": {
                    "both": 0,
                    "speed_t_only": 0,
                    "command_t_only": 0,
                    "neither": 0,
                },
                "speed_t": {
                    "present": 0,
                    "missing": 0,
                    "unit": "m/s",
                    "min": None,
                    "max": None,
                },
                "command_t": {
                    "present": 0,
                    "missing": 0,
                    "encoding": "TF++ six-way one-hot",
                    "class_counts_zero_based": {},
                },
            },
            "errors": [f"index: {exc}"],
        }

    all_entries = index["records"]
    entries = [
        entry
        for entry in all_entries
        if selected_splits is None or str(entry["split"]) in selected_splits
    ]
    identities: set[tuple[str, int]] = set()
    route_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    bev_shapes: Counter[str] = Counter()
    trajectory_shapes: Counter[str] = Counter()
    checked_records = 0
    input_presence: Counter[str] = Counter()
    speed_values: list[float] = []
    command_classes: Counter[str] = Counter()
    errors: list[str] = []

    for entry in entries:
        identity = (str(entry["route_id"]), int(entry["frame_id"]))
        if identity in identities:
            errors.append(f"duplicate identity: {identity}")
        identities.add(identity)
        route_splits[identity[0]].add(str(entry["split"]))
        split_counts[str(entry["split"])] += 1
        relative_path = Path(entry["path"])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"unsafe record path: {entry['path']}")
            continue
        path = cache_root / relative_path
        if not path.is_file():
            errors.append(f"missing record: {path}")
            continue
        audit_record = deep and (deep_splits is None or str(entry["split"]) in deep_splits)
        if audit_record:
            actual_hash = sha256_file(path)
            if actual_hash != entry["sha256"]:
                errors.append(f"{path}: record SHA256 mismatch")
                continue
            try:
                record = torch.load(path, map_location="cpu", weights_only=True)
                validate_record(record)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
                continue
            checked_records += 1
            forbidden = FORBIDDEN_INPUT_NAMES.intersection(record)
            if forbidden:
                errors.append(f"{path}: forbidden cached inputs {sorted(forbidden)}")
            bev_shapes[str(tuple(record["bev_feature"].shape))] += 1
            trajectory_shapes[str(tuple(record["pred_trajectory"].shape))] += 1
            has_speed = "speed_t" in record
            has_command = "command_t" in record
            if has_speed and has_command:
                input_presence["both"] += 1
            elif has_speed:
                input_presence["speed_t_only"] += 1
            elif has_command:
                input_presence["command_t_only"] += 1
            else:
                input_presence["neither"] += 1
            if has_speed:
                speed_values.append(float(torch.as_tensor(record["speed_t"]).item()))
            if has_command:
                command_class = int(torch.as_tensor(record["command_t"]).argmax().item())
                command_classes[str(command_class)] += 1
            if record["route_id"] != identity[0] or int(record["frame_id"]) != identity[1]:
                errors.append(f"index identity mismatch: {path}")

    leaking_routes = {route: sorted(splits) for route, splits in route_splits.items() if len(splits) != 1}
    if leaking_routes:
        errors.append(f"routes assigned to multiple splits: {leaking_routes}")
    required_splits = selected_splits or set(ALLOWED_CACHE_SPLITS)
    missing_splits = required_splits.difference(split_counts)
    if missing_splits:
        errors.append(f"empty splits: {sorted(missing_splits)}")
    if deep and len(bev_shapes) > 1:
        errors.append(f"heterogeneous BEV shapes: {dict(bev_shapes)}")
    if deep and len(trajectory_shapes) > 1:
        errors.append(f"heterogeneous trajectory shapes: {dict(trajectory_shapes)}")

    return {
        "status": "pass" if not errors else "fail",
        "deep": deep,
        "selected_splits": selected_scope,
        "deep_splits": sorted(deep_splits) if deep_splits is not None else "all",
        "records": len(entries),
        "total_index_records": len(all_entries),
        "routes": len(route_splits),
        "split_record_counts": dict(split_counts),
        "bev_shapes": dict(bev_shapes),
        "trajectory_shapes": dict(trajectory_shapes),
        "input_analysis": {
            "checked_records": checked_records,
            "presence": {
                name: input_presence[name]
                for name in ("both", "speed_t_only", "command_t_only", "neither")
            },
            "speed_t": {
                "present": len(speed_values),
                "missing": checked_records - len(speed_values),
                "unit": "m/s",
                "min": min(speed_values) if speed_values else None,
                "max": max(speed_values) if speed_values else None,
            },
            "command_t": {
                "present": sum(command_classes.values()),
                "missing": checked_records - sum(command_classes.values()),
                "encoding": "TF++ six-way one-hot",
                "class_counts_zero_based": dict(command_classes),
            },
        },
        "trajectory_history_source": index["trajectory_history_source"],
        "errors": errors,
    }
