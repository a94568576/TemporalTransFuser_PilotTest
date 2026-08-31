#!/usr/bin/env python3
"""Collect the fresh, support-stratified Town05 MPC-grounding V3 dataset.

V3 deliberately has its own entry point so the frozen V1/V2 CLI, state
selection, and manifests remain unchanged.  The physical rollout/reset core is
shared with :mod:`collect_mpc_local_carla`; only the map and preregistered state
design differ.

The full design uses 60 unique ``Town05_Opt`` base waypoints generated at 5 m
spacing: 32 train, 16 fresh outer validation, and 12 sealed test states.  Every
split is marginally balanced (count difference at most one) over four stable
rank curvature bins, speeds ``{4, 6, 8}`` m/s, and lane-center lateral offsets
``{-0.25, 0, +0.25}`` m.  Collection fails before publication on any collision,
cost mismatch, requested/actual state mismatch, reset mismatch, or control
execution mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import collect_mpc_local_carla as base  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "mpc_local_grounding_pilot_v3.yaml"
DATASET_SCHEMA = "mpc-local-carla-v3"
ACTION_PROFILE = "support_stratified_v3"
STATE_SOURCE = "map_generate_waypoints"
MAP_NAME = "/Game/Carla/Maps/Town05_Opt"
MAP_SHORT_NAME = "Town05_Opt"
COLLECTION_SEED = 37031
FULL_SPLIT_COUNTS = {"train": 32, "val": 16, "test": 12}
SMOKE_SPLIT_COUNTS = {"train": 1, "val": 1, "test": 1}
WAYPOINT_SPACING_M = 5.0
CURVATURE_BINS = 4
LATERAL_OFFSETS_M = (-0.25, 0.0, 0.25)
INITIAL_SPEEDS_MPS = (4.0, 6.0, 8.0)
SPAWN_Z_OFFSET_M = 0.30
# The requested source must retain the configured 25 m after the two neutral
# activation ticks.  Two extra metres are a conservative, outcome-independent
# source filter; the actual t=0 point is audited again against the exact 25 m.
SOURCE_FORWARD_CLEARANCE_M = 27.0
# CARLA stores transform components as float32.  Reconstructing a 0.25 m
# lateral offset at Town05-scale coordinates therefore incurs roughly 1e-5 m
# of roundoff before any physics tick.  This construction-only tolerance is
# intentionally far tighter than the preregistered 0.05 m post-warmup audit.
REQUESTED_TRANSFORM_CONSTRUCTION_TOLERANCE_M = 1e-4
OUTPUT_FILES = {
    "development_records": "development_records.npz",
    "sealed_test_records": "test_records_sealed.npz",
    "development_diagnostics": "development_diagnostics.npz",
    "sealed_test_diagnostics": "test_diagnostics_sealed.npz",
    "manifest": "manifest.json",
}
PUBLIC_MANIFEST_REDACTION = {
    "states_scope": "development_only",
    "split_state_ids": ["train", "val"],
    "require_sealed_test_states_sha256": True,
    "sealed_test_integrity_fields": [
        "states",
        "records",
        "split_code",
        "states_sha256",
        "schema_finite_passed",
        "reset_passed",
        "control_execution_passed",
        "individual_state_metadata_redacted",
        "sealed_test_stratification_passed",
    ],
    "forbid_sealed_individual_state_ids_or_covariates": True,
    "forbid_sealed_outcome_cost_collision_aggregates": True,
    "forbid_sealed_outcome_cost_collision_stdout": True,
}

PARENT_COLLECTIONS = (
    {
        "name": "mpc_local_grounding_carla_v1",
        "map": "/Game/Carla/Maps/Town10HD_Opt",
        "states_sha256": "b8c4e5cebaaf1be629308c70ef6f22c6d529b027b982d0025cc0640c74143ee3",
        "records_sha256": "f3cb11ad2ea8159427a6ff305e582dfb01bbee4860be4341f2a580f4f77f137e",
    },
    {
        "name": "mpc_local_grounding_carla_v2",
        "map": "/Game/Carla/Maps/Town10HD_Opt",
        "states_sha256": "ca435de3f2f25f3e5614989879b89dc7679cfb81e85d8bedb4719da0f897fa73",
        "records_sha256": "ecd0f8c7e9a97b6f8bf7c6b96b0994231a9649eddb83b9935c61f04246f84ff7",
    },
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _require_equal(name: str, actual: Any, expected: Any) -> None:
    if isinstance(expected, float):
        try:
            equal = math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError, OverflowError):
            equal = False
    else:
        equal = actual == expected
    if not equal:
        raise ValueError(f"V3 config {name} must be {expected!r}, found {actual!r}")


def _load_and_validate_config(path: Path) -> tuple[dict[str, Any], str]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = source.read_bytes()
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError("V3 config must be a YAML mapping")
    collection = value.get("collection")
    cem = value.get("cem")
    cost = value.get("cost")
    if not all(isinstance(item, dict) for item in (collection, cem, cost)):
        raise ValueError("V3 config requires collection, cem, and cost mappings")

    expected_collection = {
        "carla_version": base.REQUIRED_CARLA_VERSION,
        "host": "127.0.0.1",
        "port": 2100,
        "map": MAP_NAME,
        "fixed_delta_seconds": base.FIXED_DELTA_SECONDS,
        "horizon_ticks": base.HORIZON_TICKS,
        "action_segments": base.ACTION_SEGMENTS,
        "vehicle_filter": base.VEHICLE_BLUEPRINT,
        "no_rendering": True,
        "action_profile": ACTION_PROFILE,
        "seed": COLLECTION_SEED,
        "states": FULL_SPLIT_COUNTS,
        "initial_speeds_mps": list(INITIAL_SPEEDS_MPS),
        "lateral_offsets_m": list(LATERAL_OFFSETS_M),
        "target_speed_mps": base.TARGET_SPEED_MPS,
        "require_non_junction_spawn": True,
        "minimum_forward_road_m": base.MINIMUM_FORWARD_ROAD_M,
        "state_source": STATE_SOURCE,
        "waypoint_spacing_m": WAYPOINT_SPACING_M,
        "require_unique_base_waypoint_per_state": True,
        "state_identity_fields": [
            "map",
            "road_id",
            "section_id",
            "lane_id",
            "waypoint_s",
            "lateral_offset_m",
            "initial_speed_mps",
        ],
        "outputs": OUTPUT_FILES,
        "sealed_test_redaction": True,
        "public_manifest_redaction": PUBLIC_MANIFEST_REDACTION,
    }
    for key, expected in expected_collection.items():
        _require_equal(f"collection.{key}", collection.get(key), expected)

    stratification = collection.get("state_stratification")
    audit = collection.get("initial_state_audit")
    if not isinstance(stratification, dict) or not isinstance(audit, dict):
        raise ValueError(
            "V3 config requires collection.state_stratification and initial_state_audit"
        )
    expected_stratification = {
        "curvature_score": "max_abs_curvature_5m_10m_20m",
        "curvature_bins": CURVATURE_BINS,
        "binning": "stable_rank_quantiles",
        "balance_factors": [
            "curvature_bin",
            "initial_speed_mps",
            "lateral_offset_m",
        ],
        "maximum_marginal_count_difference": 1,
        "deterministic_order": "sha256_map_waypoint_seed",
    }
    for key, expected in expected_stratification.items():
        _require_equal(
            f"collection.state_stratification.{key}",
            stratification.get(key),
            expected,
        )
    for key, expected in {
        "require_requested_actual_lateral_match": True,
        "lateral_tolerance_m": 0.05,
        "require_requested_actual_speed_match": True,
        "speed_tolerance_mps": 0.20,
    }.items():
        _require_equal(
            f"collection.initial_state_audit.{key}", audit.get(key), expected
        )

    expected_cem = {
        "iterations": base.CEM_ITERATIONS,
        "population": base.FULL_POPULATION,
        "elite_count": base.FULL_ELITE_COUNT,
        "initial_mean": base.CEM_INITIAL_MEAN.tolist(),
        "initial_std": base.SAFE_LOCAL_V2_INITIAL_STD.tolist(),
        "lower": base.SAFE_LOCAL_V2_LOWER.tolist(),
        "upper": base.SAFE_LOCAL_V2_UPPER.tolist(),
        "minimum_std": base.SAFE_LOCAL_V2_MINIMUM_STD.tolist(),
    }
    for key, expected in expected_cem.items():
        _require_equal(f"cem.{key}", cem.get(key), expected)

    expected_cost = {
        "progress_weight": -0.20,
        "lateral_squared_weight": 1.50,
        "yaw_squared_weight": 0.80,
        "speed_squared_weight": 0.40,
        "steering_squared_weight": 0.02,
        "longitudinal_squared_weight": 0.01,
        "collision_weight": 10.0,
        "pair_tie_threshold": base.PAIR_TIE_THRESHOLD,
    }
    for key, expected in expected_cost.items():
        _require_equal(f"cost.{key}", cost.get(key), expected)
    return value, hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class WaypointCandidate:
    catalogue_index: int
    waypoint: Any
    road_id: int
    section_id: int
    lane_id: int
    waypoint_s: float
    identity: tuple[int, int, int, int]
    identity_sha256: str
    curvature_values: tuple[float, float, float]
    curvature_score: float
    curvature_stratum: int


@dataclass(frozen=True)
class StateSlot:
    state_id: int
    split: str
    split_code: int
    curvature_stratum: int
    initial_speed_mps: float
    lateral_offset_m: float


@dataclass(frozen=True)
class V3StateSpec:
    # The first fields intentionally match base.StateSpec's duck-typed
    # collector contract.
    state_id: int
    split: str
    split_code: int
    source_spawn_index: int
    transform: Any
    waypoint: Any
    initial_speed_mps: float
    initial_features: np.ndarray
    initial_features_raw: np.ndarray
    initial_feature_valid: np.ndarray
    source_waypoint_index: int
    source_waypoint_identity: tuple[int, int, int, int]
    source_waypoint_identity_sha256: str
    road_id: int
    section_id: int
    lane_id: int
    waypoint_s: float
    requested_lateral_offset_m: float
    speed_stratum: float
    curvature_stratum: int
    curvature_values: tuple[float, float, float]
    curvature_score: float
    nearest_spawn_distance_m: float
    state_identity_sha256: str


def _waypoint_identity(waypoint: Any) -> tuple[int, int, int, int]:
    # Millimetre quantization is far below the 5 m catalogue spacing and gives
    # a stable integer identity instead of a locale/JSON-dependent float key.
    return (
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
        int(round(float(waypoint.s) * 1000.0)),
    )


def _build_waypoint_pool(road_map: Any) -> tuple[list[WaypointCandidate], dict[str, Any]]:
    unique: dict[tuple[int, int, int, int], tuple[Any, tuple[float, ...], float]] = {}
    generated = list(road_map.generate_waypoints(WAYPOINT_SPACING_M))
    for waypoint in generated:
        if bool(waypoint.is_junction):
            continue
        if not base._has_clear_forward_road(waypoint, SOURCE_FORWARD_CLEARANCE_M):
            continue
        curvature = tuple(
            float(base._signed_curvature(waypoint, distance)[0])
            for distance in (5.0, 10.0, 20.0)
        )
        if not all(math.isfinite(value) for value in curvature):
            continue
        identity = _waypoint_identity(waypoint)
        score = max(abs(value) for value in curvature)
        previous = unique.get(identity)
        if previous is not None:
            # Duplicate map entries are only acceptable if they are physically
            # identical under the stable identity.
            previous_transform = base._transform_array(previous[0].transform)
            current_transform = base._transform_array(waypoint.transform)
            if not np.allclose(previous_transform, current_transform, rtol=0.0, atol=1e-6):
                raise RuntimeError(f"waypoint identity collision: {identity}")
            continue
        unique[identity] = (waypoint, curvature, score)
    if len(unique) < sum(FULL_SPLIT_COUNTS.values()):
        raise RuntimeError(
            f"Town05 waypoint pool has only {len(unique)} eligible unique bases"
        )

    ranked = sorted(
        unique.items(),
        key=lambda item: (float(item[1][2]), item[0]),
    )
    result: list[WaypointCandidate] = []
    for rank, (identity, (waypoint, curvature, score)) in enumerate(ranked):
        curvature_stratum = min(CURVATURE_BINS - 1, rank * CURVATURE_BINS // len(ranked))
        result.append(
            WaypointCandidate(
                catalogue_index=rank,
                waypoint=waypoint,
                road_id=identity[0],
                section_id=identity[1],
                lane_id=identity[2],
                waypoint_s=identity[3] / 1000.0,
                identity=identity,
                identity_sha256=_stable_hash(identity),
                curvature_values=tuple(float(value) for value in curvature),
                curvature_score=float(score),
                curvature_stratum=curvature_stratum,
            )
        )
    counts = {
        str(index): sum(item.curvature_stratum == index for item in result)
        for index in range(CURVATURE_BINS)
    }
    ranges = {}
    for index in range(CURVATURE_BINS):
        scores = [item.curvature_score for item in result if item.curvature_stratum == index]
        ranges[str(index)] = {"min": min(scores), "max": max(scores)}
    return result, {
        "generated_waypoints": len(generated),
        "eligible_unique_waypoints": len(result),
        "source_forward_clearance_m": SOURCE_FORWARD_CLEARANCE_M,
        "curvature_stratum_pool_counts": counts,
        "curvature_stratum_score_ranges": ranges,
    }


def _hash_permute(values: Sequence[Any], *, seed: int, label: str) -> list[Any]:
    decorated = []
    occurrence: dict[str, int] = {}
    for value in values:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
        position = occurrence.get(canonical, 0)
        occurrence[canonical] = position + 1
        digest = hashlib.sha256(
            f"{seed}|{label}|{canonical}|{position}".encode("utf-8")
        ).digest()
        decorated.append((digest, canonical, position, value))
    return [item[3] for item in sorted(decorated)]


def _balanced_levels(
    levels: Sequence[Any], count: int, *, extra_order: Sequence[int]
) -> list[Any]:
    base_count, remainder = divmod(int(count), len(levels))
    counts = [base_count] * len(levels)
    if len(extra_order) < remainder or len(set(extra_order[:remainder])) != remainder:
        raise ValueError("extra_order cannot supply balanced distinct levels")
    for index in extra_order[:remainder]:
        counts[int(index)] += 1
    return [level for level, repeats in zip(levels, counts, strict=True) for _ in range(repeats)]


def _state_slots(smoke: bool, seed: int) -> list[StateSlot]:
    if smoke:
        choices = {
            "train": (0, INITIAL_SPEEDS_MPS[0], LATERAL_OFFSETS_M[0]),
            "val": (2, INITIAL_SPEEDS_MPS[1], LATERAL_OFFSETS_M[1]),
            "test": (3, INITIAL_SPEEDS_MPS[2], LATERAL_OFFSETS_M[2]),
        }
        return [
            StateSlot(
                state_id=index,
                split=split,
                split_code=base.SPLIT_CODES[split],
                curvature_stratum=choices[split][0],
                initial_speed_mps=choices[split][1],
                lateral_offset_m=choices[split][2],
            )
            for index, split in enumerate(("train", "val", "test"))
        ]

    slots: list[StateSlot] = []
    cursor = 0
    # Rotating the remainder allocation makes the aggregate speed/offset count
    # exactly 20/20/20 while every individual split differs by at most one.
    extra_order = {
        "train": (0, 1, 2),
        "val": (2, 0, 1),
        "test": (0, 1, 2),
    }
    for split in ("train", "val", "test"):
        count = FULL_SPLIT_COUNTS[split]
        curvatures = _balanced_levels(
            tuple(range(CURVATURE_BINS)), count, extra_order=(0, 1, 2, 3)
        )
        speeds = _balanced_levels(
            INITIAL_SPEEDS_MPS, count, extra_order=extra_order[split]
        )
        offsets = _balanced_levels(
            LATERAL_OFFSETS_M, count, extra_order=extra_order[split]
        )
        curvatures = _hash_permute(
            curvatures, seed=seed, label=f"{split}|curvature"
        )
        speeds = _hash_permute(speeds, seed=seed, label=f"{split}|speed")
        offsets = _hash_permute(offsets, seed=seed, label=f"{split}|lateral")
        for curvature, speed, offset in zip(
            curvatures, speeds, offsets, strict=True
        ):
            slots.append(
                StateSlot(
                    state_id=cursor,
                    split=split,
                    split_code=base.SPLIT_CODES[split],
                    curvature_stratum=int(curvature),
                    initial_speed_mps=float(speed),
                    lateral_offset_m=float(offset),
                )
            )
            cursor += 1
    return slots


def _nearest_spawn(
    waypoint: Any, spawn_points: Sequence[Any]
) -> tuple[int, float]:
    location = waypoint.transform.location
    distances = [
        math.sqrt(
            (float(item.location.x) - float(location.x)) ** 2
            + (float(item.location.y) - float(location.y)) ** 2
            + (float(item.location.z) - float(location.z)) ** 2
        )
        for item in spawn_points
    ]
    index = min(range(len(distances)), key=lambda item: (distances[item], item))
    return int(index), float(distances[index])


def _offset_transform(carla: Any, waypoint: Any, lateral_offset_m: float) -> Any:
    lane = waypoint.transform
    yaw = math.radians(float(lane.rotation.yaw))
    right_x = -math.sin(yaw)
    right_y = math.cos(yaw)
    return carla.Transform(
        carla.Location(
            x=float(lane.location.x) + float(lateral_offset_m) * right_x,
            y=float(lane.location.y) + float(lateral_offset_m) * right_y,
            z=float(lane.location.z) + SPAWN_Z_OFFSET_M,
        ),
        carla.Rotation(
            pitch=float(lane.rotation.pitch),
            yaw=float(lane.rotation.yaw),
            roll=float(lane.rotation.roll),
        ),
    )


def _select_states(
    road_map: Any, carla: Any, *, smoke: bool, seed: int
) -> tuple[list[V3StateSpec], dict[str, Any]]:
    pool, pool_audit = _build_waypoint_pool(road_map)
    slots = _state_slots(smoke, seed)
    by_stratum = {
        index: [item for item in pool if item.curvature_stratum == index]
        for index in range(CURVATURE_BINS)
    }
    for stratum, candidates in by_stratum.items():
        candidates.sort(
            key=lambda item: (
                hashlib.sha256(
                    f"{seed}|{MAP_NAME}|{item.identity}".encode("utf-8")
                ).digest(),
                item.identity,
            )
        )
        required = sum(slot.curvature_stratum == stratum for slot in slots)
        if len(candidates) < required:
            raise RuntimeError(
                f"curvature stratum {stratum} has {len(candidates)} candidates; need {required}"
            )

    spawn_points = list(road_map.get_spawn_points())
    cursor_by_stratum = {index: 0 for index in range(CURVATURE_BINS)}
    states: list[V3StateSpec] = []
    used: set[tuple[int, int, int, int]] = set()
    for slot in slots:
        candidates = by_stratum[slot.curvature_stratum]
        position = cursor_by_stratum[slot.curvature_stratum]
        candidate = candidates[position]
        cursor_by_stratum[slot.curvature_stratum] = position + 1
        if candidate.identity in used:
            raise RuntimeError("V3 state selection reused a base waypoint")
        used.add(candidate.identity)
        transform = _offset_transform(carla, candidate.waypoint, slot.lateral_offset_m)
        normalized, raw, valid = base._initial_features(
            transform, candidate.waypoint, slot.initial_speed_mps
        )
        if not bool(valid.all()):
            raise RuntimeError(
                f"state {slot.state_id} source waypoint has incomplete initial features"
            )
        if (
            abs(float(raw[0]) - slot.lateral_offset_m)
            > REQUESTED_TRANSFORM_CONSTRUCTION_TOLERANCE_M
        ):
            raise RuntimeError(
                f"state {slot.state_id} requested lateral construction mismatch"
            )
        nearest_spawn_index, nearest_spawn_distance = _nearest_spawn(
            candidate.waypoint, spawn_points
        )
        identity_payload = {
            "map": MAP_NAME,
            "road_id": candidate.road_id,
            "section_id": candidate.section_id,
            "lane_id": candidate.lane_id,
            "waypoint_s": candidate.waypoint_s,
            "lateral_offset_m": slot.lateral_offset_m,
            "initial_speed_mps": slot.initial_speed_mps,
        }
        states.append(
            V3StateSpec(
                state_id=slot.state_id,
                split=slot.split,
                split_code=slot.split_code,
                source_spawn_index=nearest_spawn_index,
                transform=transform,
                waypoint=candidate.waypoint,
                initial_speed_mps=slot.initial_speed_mps,
                initial_features=normalized,
                initial_features_raw=raw,
                initial_feature_valid=valid,
                source_waypoint_index=candidate.catalogue_index,
                source_waypoint_identity=candidate.identity,
                source_waypoint_identity_sha256=candidate.identity_sha256,
                road_id=candidate.road_id,
                section_id=candidate.section_id,
                lane_id=candidate.lane_id,
                waypoint_s=candidate.waypoint_s,
                requested_lateral_offset_m=slot.lateral_offset_m,
                speed_stratum=slot.initial_speed_mps,
                curvature_stratum=slot.curvature_stratum,
                curvature_values=candidate.curvature_values,
                curvature_score=candidate.curvature_score,
                nearest_spawn_distance_m=nearest_spawn_distance,
                state_identity_sha256=_stable_hash(identity_payload),
            )
        )
    selection_audit = _stratification_audit(states, smoke=smoke)
    return states, {**pool_audit, **selection_audit}


def _count_levels(values: Sequence[Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _count_difference(counts: Mapping[str, int]) -> int:
    values = list(counts.values())
    return max(values) - min(values) if values else 0


def _stratification_audit(
    states: Sequence[V3StateSpec], *, smoke: bool
) -> dict[str, Any]:
    identities = [item.source_waypoint_identity for item in states]
    unique = len(set(identities)) == len(identities)
    by_split = {}
    passed = unique
    for split in ("train", "val", "test"):
        selected = [item for item in states if item.split == split]
        curvature = _count_levels([item.curvature_stratum for item in selected])
        speed = _count_levels([item.speed_stratum for item in selected])
        lateral = _count_levels([item.requested_lateral_offset_m for item in selected])
        differences = {
            "curvature_stratum": _count_difference(curvature),
            "speed_stratum": _count_difference(speed),
            "lateral_offset": _count_difference(lateral),
        }
        split_passed = bool(smoke or max(differences.values()) <= 1)
        passed &= split_passed
        by_split[split] = {
            "states": len(selected),
            "curvature_stratum_counts": curvature,
            "speed_stratum_counts": speed,
            "lateral_offset_counts": lateral,
            "maximum_marginal_count_difference": max(differences.values()),
            "passed": split_passed,
        }
    if not passed:
        raise RuntimeError(f"V3 stratification audit failed: {by_split}")
    return {
        "waypoint_state_source": STATE_SOURCE,
        "waypoint_state_source_passed": True,
        "unique_base_waypoint_required": True,
        "unique_base_waypoint_count": len(set(identities)),
        "selected_states": len(states),
        "unique_base_waypoint_passed": unique,
        "selected_base_waypoints_sha256": _stable_hash(sorted(identities)),
        "balance_is_non_evidentiary_for_smoke": bool(smoke),
        "by_split": by_split,
        "passed": passed,
    }


def _protocol(config: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    collection = config["collection"]
    split_counts = SMOKE_SPLIT_COUNTS if smoke else FULL_SPLIT_COUNTS
    population = base.SMOKE_POPULATION if smoke else int(config["cem"]["population"])
    elite_count = base.SMOKE_ELITE_COUNT if smoke else int(config["cem"]["elite_count"])
    map_disjoint = {
        "current_map": MAP_NAME,
        "parents": [dict(value) for value in PARENT_COLLECTIONS],
        "overlapping_parent_maps": [
            value["name"] for value in PARENT_COLLECTIONS if value["map"] == MAP_NAME
        ],
        "passed": all(value["map"] != MAP_NAME for value in PARENT_COLLECTIONS),
    }
    if not map_disjoint["passed"]:
        raise RuntimeError("V3 map is not disjoint from V1/V2")
    return {
        "schema_version": DATASET_SCHEMA,
        "smoke": bool(smoke),
        "seed": int(collection["seed"]),
        "action_profile": ACTION_PROFILE,
        "map": MAP_NAME,
        "carla_version": base.REQUIRED_CARLA_VERSION,
        "vehicle_blueprint": base.VEHICLE_BLUEPRINT,
        "fixed_delta_seconds": base.FIXED_DELTA_SECONDS,
        "horizon_ticks": base.HORIZON_TICKS,
        "horizon_seconds": base.HORIZON_TICKS * base.FIXED_DELTA_SECONDS,
        "action_parameterization": "[steer1,longitudinal1,steer2,longitudinal2]",
        "longitudinal_mapping": "positive=throttle; negative=brake",
        "initial_speeds_mps": list(INITIAL_SPEEDS_MPS),
        "target_speed_mps": base.TARGET_SPEED_MPS,
        "minimum_forward_non_junction_road_m": base.MINIMUM_FORWARD_ROAD_M,
        "split_counts": dict(split_counts),
        "split_codes": dict(base.SPLIT_CODES),
        "state_selection": {
            "waypoint_state_source": STATE_SOURCE,
            "waypoint_state_source_passed": True,
            "waypoint_spacing_m": WAYPOINT_SPACING_M,
            "source_forward_clearance_m": SOURCE_FORWARD_CLEARANCE_M,
            "requested_transform_construction_tolerance_m": (
                REQUESTED_TRANSFORM_CONSTRUCTION_TOLERANCE_M
            ),
            "require_unique_base_waypoint_per_state": True,
            "lateral_offsets_m": list(LATERAL_OFFSETS_M),
            "initial_speeds_mps": list(INITIAL_SPEEDS_MPS),
            "curvature_score": "max_abs_curvature_5m_10m_20m",
            "curvature_bins": CURVATURE_BINS,
            "curvature_binning": "stable_rank_quantiles",
            "deterministic_order": "sha256_map_waypoint_seed",
            "map_disjoint_from_parents": map_disjoint,
            "parent_disjoint_by_map": map_disjoint,
            "passed": True,
        },
        "cem": {
            **dict(config["cem"]),
            "population": population,
            "elite_count": elite_count,
            "sampling": (
                "numpy.PCG64 normal then componentwise clamp; stable cost/index elite order"
            ),
        },
        "cost": {
            **dict(config["cost"]),
            "formula": (
                "w_progress*outcome[0] + w_lateral*outcome[1]^2 + "
                "w_yaw*outcome[2]^2 + w_speed*outcome[3]^2 + "
                "w_steer*mean(action[0]^2,action[2]^2) + "
                "w_long*mean(action[1]^2,action[3]^2) + w_collision*collision"
            ),
            "outcome_feature_names": list(base.OUTCOME_FEATURE_NAMES),
        },
        "initial_feature_names": list(base.INITIAL_FEATURE_NAMES),
        "initial_state_audit": dict(collection["initial_state_audit"]),
        "reset": {
            "policy": "fresh vehicle and fresh attached collision sensor for every candidate",
            "pose": "two physics-disabled ticks commit and verify the exact requested transform",
            "initial_velocity": (
                "spawn-forward target velocity followed by two action-independent neutral ticks"
            ),
            "candidate_t0": (
                "actual post-neutral-warm-up transform, speed, lane, and road features; "
                "bitwise state equality audited across candidates"
            ),
            "tolerances": dict(base.RESET_TOLERANCES),
            "probe_repeats": base.RESET_PROBE_REPEATS,
        },
        "control_execution": {
            "recorded_fields": ["steer", "throttle", "brake"],
            "submission": (
                "client.apply_batch_sync(command.ApplyVehicleControl, do_tick=False) "
                "followed by world.tick"
            ),
            "audit": "vehicle.get_control after every synchronous tick",
            "max_abs_error_tolerance": base.CONTROL_EXECUTION_TOLERANCE,
        },
        "collision_policy": (
            "zero development labels and events required before atomic publish; "
            "sealed-test collision values are never summarized or used as a gate"
        ),
        "collision_gate_scope": "development_only",
        "sealed_test_redaction": True,
    }


def _state_manifest(states: Sequence[V3StateSpec]) -> list[dict[str, Any]]:
    return [
        {
            "state_id": state.state_id,
            "split": state.split,
            "split_code": state.split_code,
            "road_id": state.road_id,
            "section_id": state.section_id,
            "lane_id": state.lane_id,
            "s": state.waypoint_s,
            "waypoint_s": state.waypoint_s,
            "source_waypoint_index": state.source_waypoint_index,
            "source_waypoint_identity": list(state.source_waypoint_identity),
            "source_waypoint_identity_sha256": state.source_waypoint_identity_sha256,
            "nearest_spawn_index": state.source_spawn_index,
            "nearest_spawn_distance_m": state.nearest_spawn_distance_m,
            "requested_lateral_offset_m": state.requested_lateral_offset_m,
            "initial_speed_mps": state.initial_speed_mps,
            "speed_stratum": state.speed_stratum,
            "curvature_stratum": state.curvature_stratum,
            "curvature_score": state.curvature_score,
            "curvature_5m_10m_20m": list(state.curvature_values),
            "transform": base._transform_json(state.transform),
            "requested_spawn_initial_features": state.initial_features.tolist(),
            "state_identity_sha256": state.state_identity_sha256,
        }
        for state in states
    ]


def _diagnostic_arrays(
    states: Sequence[V3StateSpec], arrays: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    result = base._diagnostic_arrays(states, arrays)
    result.update(
        {
            "source_waypoint_index": np.asarray(
                [state.source_waypoint_index for state in states], dtype=np.int32
            ),
            "source_waypoint_identity": np.asarray(
                [state.source_waypoint_identity for state in states], dtype=np.int64
            ),
            "source_waypoint_identity_sha256": np.asarray(
                [state.source_waypoint_identity_sha256 for state in states], dtype="S64"
            ),
            "requested_lateral_offset_m": np.asarray(
                [state.requested_lateral_offset_m for state in states], dtype=np.float32
            ),
            "speed_stratum": np.asarray(
                [state.speed_stratum for state in states], dtype=np.float32
            ),
            "curvature_stratum": np.asarray(
                [state.curvature_stratum for state in states], dtype=np.int8
            ),
            "curvature_values": np.asarray(
                [state.curvature_values for state in states], dtype=np.float32
            ),
            "curvature_score": np.asarray(
                [state.curvature_score for state in states], dtype=np.float32
            ),
            "nearest_spawn_distance_m": np.asarray(
                [state.nearest_spawn_distance_m for state in states], dtype=np.float32
            ),
            "state_identity_sha256": np.asarray(
                [state.state_identity_sha256 for state in states], dtype="S64"
            ),
        }
    )
    return result


def _slice_collection_arrays(
    arrays: Mapping[str, np.ndarray], state_indices: Sequence[int]
) -> dict[str, np.ndarray]:
    selected = np.asarray(state_indices, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("state_indices must be a non-empty vector")
    state_count = int(np.asarray(arrays["real_cost"]).shape[0])
    if np.any(selected < 0) or np.any(selected >= state_count):
        raise IndexError("state collection slice is out of bounds")
    result: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.ndim < 1 or int(array.shape[0]) != state_count:
            raise RuntimeError(
                f"collector array {name} does not have the state-leading dimension"
            )
        result[name] = np.array(array[selected], copy=True)
    return result


def _split_file_payloads(
    states: Sequence[V3StateSpec], arrays: Mapping[str, np.ndarray]
) -> dict[str, tuple[list[V3StateSpec], dict[str, np.ndarray]]]:
    groups = {
        "development": [
            index for index, state in enumerate(states) if state.split in {"train", "val"}
        ],
        "sealed_test": [
            index for index, state in enumerate(states) if state.split == "test"
        ],
    }
    result = {}
    for role, indices in groups.items():
        role_states = [states[index] for index in indices]
        if not role_states:
            raise RuntimeError(f"V3 {role} state partition is empty")
        result[role] = (role_states, _slice_collection_arrays(arrays, indices))
    development_ids = {
        state.state_identity_sha256 for state in result["development"][0]
    }
    test_ids = {state.state_identity_sha256 for state in result["sealed_test"][0]}
    if development_ids & test_ids:
        raise RuntimeError("development and sealed-test states overlap")
    return result


def _states_sha256(states: Sequence[V3StateSpec]) -> str:
    return base._sha256_bytes(base._canonical_json_bytes(_state_manifest(states)))


def _public_state_selection_audit(
    audit: Mapping[str, Any], development_states: Sequence[V3StateSpec]
) -> dict[str, Any]:
    """Remove every test-specific covariate from the public selection audit."""

    public = json.loads(json.dumps(audit))
    by_split = public.get("by_split")
    if not isinstance(by_split, dict):
        raise RuntimeError("V3 state-selection audit is missing split reports")
    public["by_split"] = {
        split: by_split[split] for split in ("train", "val")
    }
    public.pop("selected_base_waypoints_sha256", None)
    public.update(
        {
            "public_state_scope": "development_only",
            "development_states": len(development_states),
            "development_base_waypoints_sha256": _stable_hash(
                sorted(state.source_waypoint_identity for state in development_states)
            ),
            "sealed_test_individual_metadata_redacted": True,
        }
    )
    return public


def _public_requested_actual_audit(
    audit: Mapping[str, Any], development_states: Sequence[V3StateSpec]
) -> dict[str, Any]:
    development_ids = {int(state.state_id) for state in development_states}
    reports = [
        dict(item)
        for item in audit.get("states", ())
        if int(item.get("state_id", -1)) in development_ids
    ]
    if len(reports) != len(development_states):
        raise RuntimeError("development requested/actual audit coverage is incomplete")
    return {
        "scope": "development_only",
        "lateral_tolerance_m": float(audit["lateral_tolerance_m"]),
        "speed_tolerance_mps": float(audit["speed_tolerance_mps"]),
        "maximum_lateral_absolute_error_m": max(
            float(item["lateral_absolute_error_m"]) for item in reports
        ),
        "maximum_speed_absolute_error_mps": max(
            float(item["speed_absolute_error_mps"]) for item in reports
        ),
        "maximum_post_warmup_position_delta_m": max(
            float(item["post_warmup_position_delta_m"]) for item in reports
        ),
        "maximum_post_warmup_rotation_delta_rad": max(
            float(item["post_warmup_rotation_delta_rad"]) for item in reports
        ),
        "states": reports,
        "passed": all(bool(item["passed"]) for item in reports),
        "sealed_test_individual_metadata_redacted": True,
    }


def _public_collection_summary(
    summary: Mapping[str, Any],
    development_states: Sequence[V3StateSpec],
    development_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Publish reset/control numbers for development rows only."""

    public = json.loads(json.dumps(summary))
    development_ids = {int(state.state_id) for state in development_states}
    paired = [
        dict(item)
        for item in public.get("paired_initial_state", ())
        if int(item.get("state_id", -1)) in development_ids
    ]
    if len(paired) != len(development_states):
        raise RuntimeError("development paired-state audit coverage is incomplete")

    reset_max = np.asarray(development_arrays["reset_diagnostics"], dtype=np.float64).max(
        axis=(0, 1, 2)
    )
    control_max = float(
        np.asarray(
            development_arrays["control_execution_max_abs_error"], dtype=np.float64
        ).max()
    )
    commands = np.asarray(
        development_arrays["initial_velocity_command"], dtype=np.float64
    )
    commanded_speeds = np.linalg.norm(commands, axis=-1)
    expected_speeds = np.asarray(
        [state.initial_speed_mps for state in development_states], dtype=np.float64
    )[:, None, None]
    velocity_error = float(np.max(np.abs(commanded_speeds - expected_speeds)))
    public.update(
        {
            "audit_numeric_scope": "development_only",
            "reset_observed_max": {
                key: float(value)
                for key, value in zip(base.RESET_TOLERANCES, reset_max, strict=True)
            },
            "control_execution_audit_passed": bool(
                control_max <= base.CONTROL_EXECUTION_TOLERANCE
            ),
            "control_execution_max_abs_error": control_max,
            "initial_velocity_command_audit_passed": bool(velocity_error <= 1e-6),
            "initial_velocity_command_max_abs_error": velocity_error,
            "paired_initial_state_passed": all(bool(item["passed"]) for item in paired),
            "paired_initial_state": paired,
            "sealed_test_individual_metadata_redacted": True,
        }
    )
    return public


def _sealed_test_integrity(
    test_states: Sequence[V3StateSpec],
    test_arrays: Mapping[str, np.ndarray],
    full_summary: Mapping[str, Any],
    full_requested_actual_audit: Mapping[str, Any],
    selection_audit: Mapping[str, Any],
    *,
    records: int,
    states_sha256: str,
) -> dict[str, Any]:
    """Return only permitted boolean/count/hash attestations for sealed rows."""

    schema_finite = True
    for value in test_arrays.values():
        array = np.asarray(value)
        if array.dtype.kind == "O" or (
            np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()
        ):
            schema_finite = False
            break

    reset_tolerances = np.asarray(
        tuple(base.RESET_TOLERANCES.values()), dtype=np.float64
    )
    reset_diagnostics = np.asarray(test_arrays["reset_diagnostics"], dtype=np.float64)
    test_ids = {int(state.state_id) for state in test_states}
    paired = [
        item
        for item in full_summary.get("paired_initial_state", ())
        if int(item.get("state_id", -1)) in test_ids
    ]
    requested_reports = [
        item
        for item in full_requested_actual_audit.get("states", ())
        if int(item.get("state_id", -1)) in test_ids
    ]
    reset_passed = bool(
        len(paired) == len(test_states)
        and len(requested_reports) == len(test_states)
        and np.all(reset_diagnostics <= reset_tolerances)
        and all(bool(item.get("passed")) for item in paired)
        and all(bool(item.get("passed")) for item in requested_reports)
    )

    control_errors = np.asarray(
        test_arrays["control_execution_max_abs_error"], dtype=np.float64
    )
    commands = np.asarray(test_arrays["initial_velocity_command"], dtype=np.float64)
    commanded_speeds = np.linalg.norm(commands, axis=-1)
    expected_speeds = np.asarray(
        [state.initial_speed_mps for state in test_states], dtype=np.float64
    )[:, None, None]
    control_passed = bool(
        np.max(control_errors) <= base.CONTROL_EXECUTION_TOLERANCE
        and np.max(np.abs(commanded_speeds - expected_speeds)) <= 1e-6
    )
    by_split = selection_audit.get("by_split")
    test_stratification = bool(
        isinstance(by_split, Mapping)
        and isinstance(by_split.get("test"), Mapping)
        and by_split["test"].get("passed") is True
    )
    integrity = {
        "states": len(test_states),
        "records": int(records),
        "split_code": int(base.SPLIT_CODES["test"]),
        "states_sha256": states_sha256,
        "schema_finite_passed": schema_finite,
        "reset_passed": reset_passed,
        "control_execution_passed": control_passed,
        "individual_state_metadata_redacted": True,
        "sealed_test_stratification_passed": test_stratification,
    }
    if not all(
        integrity[key] is True
        for key in (
            "schema_finite_passed",
            "reset_passed",
            "control_execution_passed",
            "sealed_test_stratification_passed",
        )
    ):
        raise RuntimeError("sealed-test schema/reset/control integrity failed")
    return integrity


def _collision_and_cost_audit(
    arrays: Mapping[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, Any]]:
    collision_labels = int(np.count_nonzero(arrays["collision"]))
    collision_events = int(np.asarray(arrays["collision_count"], dtype=np.int64).sum())
    collision_impulse = float(
        np.asarray(arrays["collision_impulse_sum"], dtype=np.float64).sum()
    )
    collision = {
        "scope": "development_only",
        "required_collision_labels": 0,
        "required_collision_events": 0,
        "collision_labels": collision_labels,
        "collision_events": collision_events,
        "collision_impulse_sum": collision_impulse,
        "passed": collision_labels == 0 and collision_events == 0 and collision_impulse == 0.0,
    }
    if not collision["passed"]:
        raise RuntimeError(f"V3 collision-free collection gate failed: {collision}")

    weights = np.asarray(tuple(base.COST_WEIGHTS.values()), dtype=np.float64)
    recomputed = np.tensordot(
        np.asarray(arrays["cost_terms"], dtype=np.float64), weights, axes=([-1], [0])
    )
    recorded = np.asarray(arrays["real_cost"], dtype=np.float64)
    difference = np.abs(recomputed - recorded)
    tolerance = 1e-5 + 1e-5 * np.abs(recorded)
    cost = {
        "scope": "development_only",
        "formula": "dot(recorded_physical_terms, frozen_weights)",
        "records": int(recorded.size),
        "maximum_absolute_recompute_error": float(difference.max(initial=0.0)),
        "tolerance": "1e-5 + 1e-5*abs(real_cost)",
        "passed": bool(np.all(difference <= tolerance)),
    }
    if not cost["passed"]:
        raise RuntimeError(f"V3 physical-cost audit failed: {cost}")
    return collision, cost


def _requested_actual_state_audit(
    session: base.CarlaSession,
    states: Sequence[V3StateSpec],
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if session.road_map is None:
        raise RuntimeError("CARLA road map unavailable for V3 state audit")
    lateral_tolerance = float(
        config["collection"]["initial_state_audit"]["lateral_tolerance_m"]
    )
    speed_tolerance = float(
        config["collection"]["initial_state_audit"]["speed_tolerance_mps"]
    )
    reports = []
    for state_index, state in enumerate(states):
        actual_transform = np.asarray(
            arrays["initial_actual_transform"][state_index, 0, 0], dtype=np.float64
        )
        actual_speed = float(arrays["initial_actual_speed_mps"][state_index, 0, 0])
        actual_lateral = float(
            arrays["initial_features_actual_raw"][state_index, 0, 0, 0]
        )
        location = session.carla.Location(
            x=float(actual_transform[0]),
            y=float(actual_transform[1]),
            z=float(actual_transform[2]),
        )
        waypoint = session.road_map.get_waypoint(
            location,
            project_to_road=False,
            lane_type=session.carla.LaneType.Driving,
        )
        lane_matches = bool(
            waypoint is not None
            and not bool(waypoint.is_junction)
            and int(waypoint.road_id) == state.road_id
            and int(waypoint.section_id) == state.section_id
            and int(waypoint.lane_id) == state.lane_id
        )
        forward_clear = bool(
            waypoint is not None
            and base._has_clear_forward_road(
                waypoint, float(config["collection"]["minimum_forward_road_m"])
            )
        )
        lateral_error = abs(actual_lateral - state.requested_lateral_offset_m)
        speed_error = abs(actual_speed - state.initial_speed_mps)
        requested_transform = base._transform_array(state.transform)
        position_delta = float(
            np.linalg.norm(actual_transform[:3] - requested_transform[:3])
        )
        rotation_delta = max(
            abs(base._wrap_angle_rad(actual_transform[index] - requested_transform[index]))
            for index in (3, 4, 5)
        )
        passed = bool(
            lateral_error <= lateral_tolerance
            and speed_error <= speed_tolerance
            and lane_matches
            and forward_clear
        )
        report = {
            "state_id": state.state_id,
            "road_id": state.road_id,
            "section_id": state.section_id,
            "lane_id": state.lane_id,
            "requested_lateral_offset_m": state.requested_lateral_offset_m,
            "actual_lateral_offset_m": actual_lateral,
            "lateral_absolute_error_m": lateral_error,
            "requested_speed_mps": state.initial_speed_mps,
            "actual_speed_mps": actual_speed,
            "speed_absolute_error_mps": speed_error,
            "post_warmup_position_delta_m": position_delta,
            "post_warmup_rotation_delta_rad": rotation_delta,
            "actual_lane_matches_requested": lane_matches,
            "actual_non_junction_forward_road_passed": forward_clear,
            "passed": passed,
        }
        reports.append(report)
        if not passed:
            raise RuntimeError(f"V3 requested/actual state audit failed: {report}")
    return {
        "lateral_tolerance_m": lateral_tolerance,
        "speed_tolerance_mps": speed_tolerance,
        "maximum_lateral_absolute_error_m": max(
            item["lateral_absolute_error_m"] for item in reports
        ),
        "maximum_speed_absolute_error_mps": max(
            item["speed_absolute_error_mps"] for item in reports
        ),
        "maximum_post_warmup_position_delta_m": max(
            item["post_warmup_position_delta_m"] for item in reports
        ),
        "maximum_post_warmup_rotation_delta_rad": max(
            item["post_warmup_rotation_delta_rad"] for item in reports
        ),
        "states": reports,
        "passed": all(item["passed"] for item in reports),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the fresh Town05_Opt support-stratified MPC-local V3 dataset "
            "from an existing CARLA 0.9.15 server."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default=None, help="must match frozen config if set")
    parser.add_argument("--port", type=int, default=None, help="must match frozen config if set")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="non-evidentiary 3-state, population-6 plumbing and physical audit",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be positive and finite")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, config_sha256 = _load_and_validate_config(args.config)
    collection = config["collection"]
    host = str(collection["host"])
    port = int(collection["port"])
    if args.host is not None and str(args.host) != host:
        raise ValueError("--host must match frozen V3 config")
    if args.port is not None and int(args.port) != port:
        raise ValueError("--port must match frozen V3 config")

    output, staging = base._prepare_staging(args.output)
    started_at = base._utc_now()
    monotonic_start = time.monotonic()
    protocol = _protocol(config, smoke=args.smoke)
    carla = base._import_carla()
    session = base.CarlaSession(
        carla,
        host,
        port,
        args.timeout,
        map_name=MAP_NAME,
    )
    cleanup_report: dict[str, Any] | None = None
    try:
        session.setup()
        if session.road_map is None:
            raise RuntimeError("CARLA road map was not initialized")
        states, internal_state_selection_audit = _select_states(
            session.road_map,
            carla,
            smoke=args.smoke,
            seed=int(collection["seed"]),
        )
        development_states = [
            state for state in states if state.split in {"train", "val"}
        ]
        sealed_test_states = [state for state in states if state.split == "test"]
        development_states_manifest = _state_manifest(development_states)
        development_states_sha256 = _states_sha256(development_states)
        sealed_test_states_sha256 = _states_sha256(sealed_test_states)
        print(
            f"connected client/server={base.REQUIRED_CARLA_VERSION}; "
            f"map={session.road_map.name}; profile={ACTION_PROFILE} "
            f"states={len(states)} population={protocol['cem']['population']} "
            f"output={output}",
            flush=True,
        )
        try:
            arrays, internal_collection_summary = base._collect(
                session, states, protocol
            )
            file_payloads = _split_file_payloads(states, arrays)
            development_arrays = file_payloads["development"][1]
            sealed_test_arrays = file_payloads["sealed_test"][1]
            collision_audit, cost_audit = _collision_and_cost_audit(
                development_arrays
            )
            internal_requested_actual_audit = _requested_actual_state_audit(
                session, states, arrays, config
            )
        finally:
            cleanup_report = session.cleanup()

        if not internal_collection_summary["paired_initial_state_passed"]:
            raise RuntimeError("V3 paired initial-state audit failed")
        if not internal_collection_summary["control_execution_audit_passed"]:
            raise RuntimeError("V3 applied-control audit failed")
        if not internal_collection_summary["initial_velocity_command_audit_passed"]:
            raise RuntimeError("V3 initial-velocity command audit failed")

        collection_summary = _public_collection_summary(
            internal_collection_summary,
            development_states,
            development_arrays,
        )
        requested_actual_audit = _public_requested_actual_audit(
            internal_requested_actual_audit, development_states
        )
        state_selection_audit = _public_state_selection_audit(
            internal_state_selection_audit, development_states
        )
        sealed_test_integrity = _sealed_test_integrity(
            sealed_test_states,
            sealed_test_arrays,
            internal_collection_summary,
            internal_requested_actual_audit,
            internal_state_selection_audit,
            records=int(np.asarray(sealed_test_arrays["real_cost"]).size),
            states_sha256=sealed_test_states_sha256,
        )

        required_record_keys = {
            "state_id",
            "split_code",
            "cem_iteration",
            "action_params",
            "initial_features",
            "outcome_features",
            "collision",
            "real_cost",
        }
        file_names = {
            "development": (
                OUTPUT_FILES["development_records"],
                OUTPUT_FILES["development_diagnostics"],
            ),
            "sealed_test": (
                OUTPUT_FILES["sealed_test_records"],
                OUTPUT_FILES["sealed_test_diagnostics"],
            ),
        }
        files: dict[str, Any] = {}
        dataset_files: dict[str, Any] = {}
        for role in ("development", "sealed_test"):
            role_states, role_arrays = file_payloads[role]
            record_name, diagnostic_name = file_names[role]
            record_path = staging / record_name
            diagnostic_path = staging / diagnostic_name
            role_records = base._flat_records(role_states, role_arrays)
            role_diagnostics = _diagnostic_arrays(role_states, role_arrays)
            base._atomic_write_npz(record_path, role_records)
            base._atomic_write_npz(diagnostic_path, role_diagnostics)
            files[record_name] = base._validate_npz(
                record_path, required_keys=required_record_keys
            )
            files[diagnostic_name] = base._validate_npz(diagnostic_path)
            split_codes = sorted(
                set(np.asarray(role_records["split_code"], dtype=np.int64).tolist())
            )
            expected_split_codes = [0, 1] if role == "development" else [2]
            if split_codes != expected_split_codes:
                raise RuntimeError(
                    f"V3 {role} file has split codes {split_codes}, expected "
                    f"{expected_split_codes}"
                )
            record_key = (
                "development_records" if role == "development" else "sealed_test_records"
            )
            diagnostic_key = (
                "development_diagnostics"
                if role == "development"
                else "sealed_test_diagnostics"
            )
            dataset_files[record_key] = {
                "path": record_name,
                "role": role,
                "sha256": files[record_name]["sha256"],
                "records": int(len(role_records["state_id"])),
                "states": len(role_states),
                "split_codes": split_codes,
                "sealed": role == "sealed_test",
            }
            if role == "sealed_test":
                dataset_files[record_key]["states_sha256"] = (
                    sealed_test_states_sha256
                )
            dataset_files[diagnostic_key] = {
                "path": diagnostic_name,
                "role": diagnostic_key,
                "sha256": files[diagnostic_name]["sha256"],
                "records": int(len(role_records["state_id"])),
                "states": len(role_states),
                "split_codes": split_codes,
                "sealed": role == "sealed_test",
            }
        expected_counts = {
            "development": (3456, 48),
            "sealed_test": (864, 12),
        }
        if args.smoke:
            expected_counts = {"development": (36, 2), "sealed_test": (18, 1)}
        for role, (records_expected, states_expected) in expected_counts.items():
            key = "development_records" if role == "development" else "sealed_test_records"
            item = dataset_files[key]
            if (
                item["records"] != records_expected
                or item["states"] != states_expected
            ):
                raise RuntimeError(
                    f"V3 {role} file count mismatch: {item}; expected "
                    f"records={records_expected} states={states_expected}"
                )
        map_disjoint = protocol["state_selection"]["map_disjoint_from_parents"]
        finished_at = base._utc_now()
        manifest = {
            "schema_version": 3,
            "dataset_schema": DATASET_SCHEMA,
            "action_profile": ACTION_PROFILE,
            "status": "complete",
            "smoke": bool(args.smoke),
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": time.monotonic() - monotonic_start,
            "output_directory": str(output),
            "invocation": [str(Path(__file__).resolve()), *sys.argv[1:]],
            "config_path": str(Path(args.config).expanduser().resolve()),
            "config_sha256": config_sha256,
            "public_manifest_redaction": dict(
                collection["public_manifest_redaction"]
            ),
            "source": {
                "script": str(Path(__file__).resolve()),
                "script_sha256": _sha256_file(Path(__file__).resolve()),
                "shared_rollout_core": str(Path(base.__file__).resolve()),
                "shared_rollout_core_sha256": _sha256_file(Path(base.__file__).resolve()),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pyyaml": yaml.__version__,
            },
            "protocol": protocol,
            "protocol_sha256": base._sha256_bytes(base._canonical_json_bytes(protocol)),
            "server_and_map": session.server_metadata,
            "state_selection_audit": state_selection_audit,
            "states": development_states_manifest,
            "development_states_sha256": development_states_sha256,
            "sealed_test_integrity": sealed_test_integrity,
            "fresh_state_attestation": {
                "waypoint_state_source": STATE_SOURCE,
                "unique_base_waypoint_passed": state_selection_audit[
                    "unique_base_waypoint_passed"
                ],
                "map_disjoint_from_parents": map_disjoint,
                "parent_disjoint_by_map": map_disjoint,
                "passed": bool(
                    state_selection_audit["passed"] and map_disjoint["passed"]
                ),
            },
            "requested_actual_initial_state_audit": requested_actual_audit,
            "collision_free_attestation": collision_audit,
            "cost_audit": cost_audit,
            "collection_summary": collection_summary,
            "cleanup": cleanup_report,
            "files": files,
            "dataset_files": dataset_files,
            "development_records_path": "development_records.npz",
            "development_records_sha256": files["development_records.npz"][
                "sha256"
            ],
            "development_diagnostics_path": "development_diagnostics.npz",
            "development_diagnostics_sha256": files[
                "development_diagnostics.npz"
            ]["sha256"],
            "outcome_source": "CARLA 0.9.15 simulator paired rollout",
            "same_state_reset_passed": bool(
                collection_summary["reset_probe"]["bitwise_equal"]
                and collection_summary["paired_initial_state_passed"]
            ),
            "control_execution_audit_passed": bool(
                collection_summary["control_execution_audit_passed"]
            ),
            "control_execution_max_abs_error": float(
                collection_summary["control_execution_max_abs_error"]
            ),
            "initial_velocity_command_audit_passed": bool(
                collection_summary["initial_velocity_command_audit_passed"]
            ),
            "initial_velocity_command_max_abs_error": float(
                collection_summary["initial_velocity_command_max_abs_error"]
            ),
            "split_state_ids": {
                split: [
                    state.state_id
                    for state in development_states
                    if state.split == split
                ]
                for split in ("train", "val")
            },
            "sealed_test_policy": {
                "split_code": base.SPLIT_CODES["test"],
                "physically_separate_records_file": "test_records_sealed.npz",
                "physically_separate_diagnostics_file": "test_diagnostics_sealed.npz",
                "access": (
                    "The runner may verify only manifest role/path/count/SHA format and "
                    "file existence; it must never open, hash, mmap, or decode either "
                    "sealed NPZ, and V3 outer GO never authorizes test opening"
                ),
                "outcome_collision_gate_or_summary": "prohibited",
                "smoke_is_non_evidentiary": bool(args.smoke),
            },
            "reproducibility_limit": (
                "Fresh-spawn equality is audited on this server run; CARLA physics is "
                "not claimed bitwise portable across simulator builds, GPUs, or hosts."
            ),
        }
        base._atomic_write_json(staging / "manifest.json", manifest)
        base._publish_staging(staging, output)
        print(
            f"complete: {output} records={collection_summary['candidate_count']} "
            f"development_collisions={collision_audit['collision_labels']} "
            f"development_records_sha256="
            f"{files['development_records.npz']['sha256']}",
            flush=True,
        )
        return 0
    except BaseException:
        if cleanup_report is None and session.world is not None:
            try:
                session.cleanup()
            except Exception as cleanup_exc:
                print(
                    f"cleanup also failed: {cleanup_exc}", file=sys.stderr, flush=True
                )
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
