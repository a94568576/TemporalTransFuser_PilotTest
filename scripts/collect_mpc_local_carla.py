#!/usr/bin/env python3
"""Collect paired, real-environment CEM rollouts from a CARLA 0.9.15 server.

This is deliberately a client-only collector: it connects to an already-running
server, loads Town10HD_Opt when necessary, and never starts or stops CARLA.  Each
candidate is evaluated with a newly spawned vehicle at the same state transform
and initial velocity.  No learned world-model prediction is used as a label.

The fixed full protocol is 32 disjoint non-junction states (16 train, 8 val,
8 sealed test), three CEM iterations, population 24, elite count 6, and a
20-tick/1-second horizon at 20 Hz.  ``--smoke`` retains the horizon and CEM
iterations but uses three states, population 6, and elite count 2.

The output directory must not exist.  Files are built in a sibling staging
directory and published by one directory rename only after actor/settings
cleanup succeeds.  ``records.npz`` contains exactly the eight arrays consumed
by ``temporal_tf.mpc_local_grounding.load_mpc_records``; richer reset, CEM, and
trajectory evidence is isolated in ``diagnostics.npz``.  ``manifest.json``
hashes both archives and records the complete physical protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "mpc-local-carla-v1"
REQUIRED_CARLA_VERSION = "0.9.15"
MAP_NAME = "/Game/Carla/Maps/Town10HD_Opt"
VEHICLE_BLUEPRINT = "vehicle.tesla.model3"
FIXED_DELTA_SECONDS = 0.05
HORIZON_TICKS = 20
ACTION_SEGMENTS = 2
FULL_SPLIT_COUNTS = {"train": 16, "val": 8, "test": 8}
SMOKE_SPLIT_COUNTS = {"train": 1, "val": 1, "test": 1}
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
INITIAL_SPEEDS_MPS = (4.0, 6.0, 8.0)
TARGET_SPEED_MPS = 8.0
MINIMUM_FORWARD_ROAD_M = 25.0
DEFAULT_SEED = 17031
SAFE_LOCAL_V2_SEED = 27031
DEFAULT_ACTION_PROFILE = "v1"
ACTION_PROFILE_NAMES = (DEFAULT_ACTION_PROFILE, "safe_local_v2")

CEM_ITERATIONS = 3
FULL_POPULATION = 24
FULL_ELITE_COUNT = 6
SMOKE_POPULATION = 6
SMOKE_ELITE_COUNT = 2
CEM_INITIAL_MEAN = np.asarray((0.0, 0.4, 0.0, 0.4), dtype=np.float64)
CEM_INITIAL_STD = np.asarray((0.35, 0.45, 0.35, 0.45), dtype=np.float64)
CEM_LOWER = np.asarray((-0.7, -1.0, -0.7, -1.0), dtype=np.float64)
CEM_UPPER = np.asarray((0.7, 1.0, 0.7, 1.0), dtype=np.float64)
CEM_MINIMUM_STD = np.asarray((0.03, 0.05, 0.03, 0.05), dtype=np.float64)
SAFE_LOCAL_V2_INITIAL_STD = np.asarray((0.12, 0.45, 0.12, 0.45), dtype=np.float64)
SAFE_LOCAL_V2_LOWER = np.asarray((-0.20, -1.0, -0.20, -1.0), dtype=np.float64)
SAFE_LOCAL_V2_UPPER = np.asarray((0.20, 1.0, 0.20, 1.0), dtype=np.float64)
SAFE_LOCAL_V2_MINIMUM_STD = np.asarray((0.015, 0.05, 0.015, 0.05), dtype=np.float64)

# V2 is a one-time, fresh-state remediation.  These are the immutable source
# spawn indices in the completed V1 manifest; excluding them makes all 32 V2
# states (including its sealed test split) disjoint from V1 rather than merely
# relying on a different RNG seed.  The parent identities are recorded again in
# every V2 manifest and protocol addendum.
SAFE_LOCAL_V2_EXCLUDED_V1_SPAWN_INDICES = (
    8,
    10,
    18,
    23,
    49,
    50,
    53,
    54,
    66,
    68,
    74,
    103,
    104,
    113,
    115,
    120,
    123,
    124,
    126,
    128,
    130,
    131,
    138,
    139,
    143,
    144,
    148,
    149,
    150,
    152,
    153,
    154,
)
SAFE_LOCAL_V2_PARENT_STATES_SHA256 = (
    "b8c4e5cebaaf1be629308c70ef6f22c6d529b027b982d0025cc0640c74143ee3"
)
SAFE_LOCAL_V2_PARENT_RECORDS_SHA256 = (
    "f3cb11ad2ea8159427a6ff305e582dfb01bbee4860be4341f2a580f4f77f137e"
)

# These coefficients and normalizations mirror
# configs/mpc_local_grounding_pilot_v1.yaml.  Action penalties are means over
# the two piecewise-constant segments, rather than sums.
COST_WEIGHTS = {
    "progress": -0.20,
    "lateral_squared": 1.50,
    "yaw_squared": 0.80,
    "speed_squared": 0.40,
    "steering_mean_squared": 0.02,
    "longitudinal_mean_squared": 0.01,
    "collision": 10.0,
}
PAIR_TIE_THRESHOLD = 0.005
OUTCOME_FEATURE_NAMES = (
    "progress_div_10m",
    "lateral_div_3m",
    "yaw_error_div_pi",
    "speed_error_div_target",
)
INITIAL_FEATURE_NAMES = (
    "lateral_div_3m",
    "yaw_error_div_pi",
    "initial_speed_div_target",
    "lane_width_div_4m",
    "curvature_5m_div_0p1_inv_m",
    "curvature_10m_div_0p1_inv_m",
    "curvature_20m_div_0p1_inv_m",
)
RESET_TOLERANCES = {
    "position_m": 1.0e-4,
    "rotation_rad": 1.0e-5,
    "physics_disabled_speed_mps": 1.0e-4,
    "angular_speed_rad_s": 1.0e-4,
}
RESET_PROBE_REPEATS = 5
CONTROL_EXECUTION_TOLERANCE = 1.0e-6
PAIRED_INITIAL_STATE_TOLERANCE = 1.0e-6
NEUTRAL_PHYSICS_WARMUP_TICKS = 2


def _action_profile_spec(name: str) -> dict[str, Any]:
    if name == "v1":
        return {
            "name": "v1",
            "default_seed": DEFAULT_SEED,
            "initial_mean": CEM_INITIAL_MEAN.copy(),
            "initial_std": CEM_INITIAL_STD.copy(),
            "lower": CEM_LOWER.copy(),
            "upper": CEM_UPPER.copy(),
            "minimum_std": CEM_MINIMUM_STD.copy(),
            "excluded_source_spawn_indices": (),
            "remediation_parent": None,
        }
    if name == "safe_local_v2":
        return {
            "name": "safe_local_v2",
            "default_seed": SAFE_LOCAL_V2_SEED,
            "initial_mean": CEM_INITIAL_MEAN.copy(),
            "initial_std": SAFE_LOCAL_V2_INITIAL_STD.copy(),
            "lower": SAFE_LOCAL_V2_LOWER.copy(),
            "upper": SAFE_LOCAL_V2_UPPER.copy(),
            "minimum_std": SAFE_LOCAL_V2_MINIMUM_STD.copy(),
            "excluded_source_spawn_indices": SAFE_LOCAL_V2_EXCLUDED_V1_SPAWN_INDICES,
            "remediation_parent": {
                "dataset_schema": SCHEMA_VERSION,
                "states_sha256": SAFE_LOCAL_V2_PARENT_STATES_SHA256,
                "records_sha256": SAFE_LOCAL_V2_PARENT_RECORDS_SHA256,
                "reason": "V1 collision-free precondition failed before model-performance evaluation",
            },
        }
    raise ValueError(f"unsupported action profile: {name}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = _canonical_json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _prepare_staging(output: Path) -> tuple[Path, Path]:
    output = output.expanduser().resolve(strict=False)
    if output in (Path("/"), Path.home().resolve(), PROJECT_ROOT.resolve()):
        raise ValueError(f"unsafe output directory: {output}")
    if output.exists():
        raise FileExistsError(f"output must be a new path: {output}")

    frozen_root = (PROJECT_ROOT / "frozen").resolve()
    if output == frozen_root or frozen_root in output.parents:
        raise ValueError(f"output cannot be placed in the frozen tree: {output}")

    # An output immediately below data/ is allowed, but nesting it inside an
    # existing cache/raw collection is not.  This prevents accidental writes
    # into locked evidence roots while still permitting a new collection root.
    data_root = (PROJECT_ROOT / "data").resolve()
    for ancestor in output.parents:
        if ancestor == data_root:
            break
        if ancestor == PROJECT_ROOT.parent or ancestor == Path("/"):
            break
        if ancestor.exists() and (
            (ancestor / "index.json").is_file()
            or (ancestor / "results.json.gz").is_file()
            or ancestor.name.startswith(("cache_", "raw_"))
        ):
            raise ValueError(f"output is nested inside a protected raw/cache root: {ancestor}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent)
    )
    return output, staging


def _publish_staging(staging: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output appeared during collection: {output}")
    os.rename(staging, output)
    _fsync_directory(output.parent)


def _import_carla() -> Any:
    try:
        distribution_version = importlib.metadata.version("carla")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "CARLA Python API is missing; run this script with the CARLA 0.9.15 "
            "environment (for this workspace: transfuser_test/carla_garage/.venv/bin/python)"
        ) from exc
    if distribution_version != REQUIRED_CARLA_VERSION:
        raise RuntimeError(
            f"CARLA Python API must be {REQUIRED_CARLA_VERSION}, found "
            f"{distribution_version}. Refusing to connect because an API/server mismatch "
            "can crash the Python process."
        )
    import carla  # Imported only after the distribution version is validated.

    return carla


def _wrap_angle_rad(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _vector_norm(vector: Any) -> float:
    return math.sqrt(float(vector.x) ** 2 + float(vector.y) ** 2 + float(vector.z) ** 2)


def _transform_array(transform: Any) -> np.ndarray:
    return np.asarray(
        (
            transform.location.x,
            transform.location.y,
            transform.location.z,
            math.radians(transform.rotation.roll),
            math.radians(transform.rotation.pitch),
            math.radians(transform.rotation.yaw),
        ),
        dtype=np.float64,
    )


def _transform_json(transform: Any) -> dict[str, float]:
    values = _transform_array(transform)
    return {
        "x": float(values[0]),
        "y": float(values[1]),
        "z": float(values[2]),
        "roll_rad": float(values[3]),
        "pitch_rad": float(values[4]),
        "yaw_rad": float(values[5]),
    }


def _settings_json(settings: Any) -> dict[str, Any]:
    names = (
        "synchronous_mode",
        "no_rendering_mode",
        "fixed_delta_seconds",
        "substepping",
        "max_substep_delta_time",
        "max_substeps",
        "deterministic_ragdolls",
        "tile_stream_distance",
        "actor_active_distance",
    )
    result: dict[str, Any] = {}
    for name in names:
        if hasattr(settings, name):
            value = getattr(settings, name)
            result[name] = value if value is None else float(value) if isinstance(value, float) else value
    return result


def _weather_json(weather: Any) -> dict[str, float]:
    names = (
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "wind_intensity",
        "sun_azimuth_angle",
        "sun_altitude_angle",
        "fog_density",
        "wetness",
    )
    return {name: float(getattr(weather, name)) for name in names if hasattr(weather, name)}


def _waypoint_key(waypoint: Any) -> tuple[Any, ...]:
    transform = waypoint.transform
    return (
        int(waypoint.road_id),
        int(waypoint.section_id),
        int(waypoint.lane_id),
        round(float(waypoint.s), 6),
        round(float(transform.location.x), 6),
        round(float(transform.location.y), 6),
        round(float(transform.rotation.yaw), 6),
    )


def _continuation(waypoint: Any, distance: float) -> Any | None:
    candidates = list(waypoint.next(float(distance)))
    if not candidates:
        return None
    reference_yaw = math.radians(waypoint.transform.rotation.yaw)
    return min(
        candidates,
        key=lambda candidate: (
            abs(
                _wrap_angle_rad(
                    math.radians(candidate.transform.rotation.yaw) - reference_yaw
                )
            ),
            _waypoint_key(candidate),
        ),
    )


def _has_clear_forward_road(waypoint: Any, distance: float) -> bool:
    current = waypoint
    covered = 0.0
    step = 2.0
    while covered + 1.0e-9 < distance:
        increment = min(step, distance - covered)
        current = _continuation(current, increment)
        if current is None or bool(current.is_junction):
            return False
        covered += increment
    return True


def _signed_curvature(waypoint: Any, distance: float) -> tuple[float, bool]:
    target = _continuation(waypoint, distance)
    if target is None:
        return 0.0, False
    delta = _wrap_angle_rad(
        math.radians(target.transform.rotation.yaw - waypoint.transform.rotation.yaw)
    )
    return float(delta / distance), True


@dataclass(frozen=True)
class StateSpec:
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


def _initial_features(transform: Any, waypoint: Any, speed_mps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lane_transform = waypoint.transform
    lane_yaw = math.radians(lane_transform.rotation.yaw)
    right = (-math.sin(lane_yaw), math.cos(lane_yaw))
    delta_x = float(transform.location.x - lane_transform.location.x)
    delta_y = float(transform.location.y - lane_transform.location.y)
    lateral = delta_x * right[0] + delta_y * right[1]
    yaw_error = _wrap_angle_rad(math.radians(transform.rotation.yaw) - lane_yaw)
    curvatures = [_signed_curvature(waypoint, distance) for distance in (5.0, 10.0, 20.0)]
    raw = np.asarray(
        (
            lateral,
            yaw_error,
            speed_mps,
            float(waypoint.lane_width),
            curvatures[0][0],
            curvatures[1][0],
            curvatures[2][0],
        ),
        dtype=np.float64,
    )
    normalized = np.asarray(
        (
            lateral / 3.0,
            yaw_error / math.pi,
            speed_mps / TARGET_SPEED_MPS,
            float(waypoint.lane_width) / 4.0,
            curvatures[0][0] / 0.1,
            curvatures[1][0] / 0.1,
            curvatures[2][0] / 0.1,
        ),
        dtype=np.float64,
    )
    valid = np.asarray((True, True, True, True, *(item[1] for item in curvatures)), dtype=np.bool_)
    return normalized, raw, valid


def _select_states(
    road_map: Any,
    carla: Any,
    count_by_split: Mapping[str, int],
    seed: int,
    *,
    excluded_source_spawn_indices: Sequence[int] = (),
) -> tuple[list[StateSpec], dict[str, int]]:
    valid: list[tuple[int, Any, Any]] = []
    spawn_points = list(road_map.get_spawn_points())
    for source_index, transform in enumerate(spawn_points):
        waypoint = road_map.get_waypoint(
            transform.location,
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None or bool(waypoint.is_junction):
            continue
        if not _has_clear_forward_road(waypoint, MINIMUM_FORWARD_ROAD_M):
            continue
        valid.append((source_index, transform, waypoint))
    valid.sort(
        key=lambda item: (
            round(float(item[1].location.x), 6),
            round(float(item[1].location.y), 6),
            round(float(item[1].location.z), 6),
            round(float(item[1].rotation.yaw), 6),
            item[0],
        )
    )
    valid_before_exclusion = len(valid)
    excluded = {int(value) for value in excluded_source_spawn_indices}
    valid_indices = {int(item[0]) for item in valid}
    missing_exclusions = sorted(excluded - valid_indices)
    if missing_exclusions:
        raise RuntimeError(
            "profile exclusion indices are not valid under the current map/filter: "
            f"{missing_exclusions}"
        )
    valid = [item for item in valid if int(item[0]) not in excluded]
    required = int(sum(count_by_split.values()))
    if len(valid) < required:
        raise RuntimeError(
            f"Town10HD_Opt has only {len(valid)} eligible non-junction spawn points "
            f"with {MINIMUM_FORWARD_ROAD_M:g}m clear road after {len(excluded)} "
            f"profile exclusions; need {required}"
        )
    rng = np.random.default_rng(seed)
    selected = [valid[int(index)] for index in rng.permutation(len(valid))[:required]]

    states: list[StateSpec] = []
    cursor = 0
    for split in ("train", "val", "test"):
        for local_index in range(int(count_by_split[split])):
            source_index, transform, waypoint = selected[cursor]
            speed = INITIAL_SPEEDS_MPS[local_index % len(INITIAL_SPEEDS_MPS)]
            normalized, raw, feature_valid = _initial_features(transform, waypoint, speed)
            states.append(
                StateSpec(
                    state_id=cursor,
                    split=split,
                    split_code=SPLIT_CODES[split],
                    source_spawn_index=source_index,
                    transform=transform,
                    waypoint=waypoint,
                    initial_speed_mps=speed,
                    initial_features=normalized,
                    initial_features_raw=raw,
                    initial_feature_valid=feature_valid,
                )
            )
            cursor += 1
    return states, {
        "total_spawn_points": len(spawn_points),
        "valid_spawn_points": valid_before_exclusion,
        "profile_excluded_valid_spawn_points": len(excluded),
        "eligible_spawn_points": len(valid),
    }


def _road_state(road_map: Any, carla: Any, initial_transform: Any, transform: Any, speed_mps: float) -> tuple[np.ndarray, np.ndarray, bool]:
    initial_yaw = math.radians(initial_transform.rotation.yaw)
    displacement_x = float(transform.location.x - initial_transform.location.x)
    displacement_y = float(transform.location.y - initial_transform.location.y)
    progress = displacement_x * math.cos(initial_yaw) + displacement_y * math.sin(initial_yaw)
    strict_waypoint = road_map.get_waypoint(
        transform.location,
        project_to_road=False,
        lane_type=carla.LaneType.Driving,
    )
    waypoint = road_map.get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        raise RuntimeError("CARLA map could not project terminal vehicle location to a driving lane")
    lane_yaw = math.radians(waypoint.transform.rotation.yaw)
    right = (-math.sin(lane_yaw), math.cos(lane_yaw))
    delta_x = float(transform.location.x - waypoint.transform.location.x)
    delta_y = float(transform.location.y - waypoint.transform.location.y)
    lateral = delta_x * right[0] + delta_y * right[1]
    yaw_error = _wrap_angle_rad(math.radians(transform.rotation.yaw) - lane_yaw)
    raw = np.asarray((progress, lateral, yaw_error, speed_mps), dtype=np.float64)
    normalized = np.asarray(
        (
            progress / 10.0,
            lateral / 3.0,
            yaw_error / math.pi,
            (speed_mps - TARGET_SPEED_MPS) / TARGET_SPEED_MPS,
        ),
        dtype=np.float64,
    )
    return normalized, raw, strict_waypoint is not None


def _physical_cost(outcome: np.ndarray, action: np.ndarray, collision: float) -> tuple[float, np.ndarray]:
    terms = np.asarray(
        (
            outcome[0],
            outcome[1] ** 2,
            outcome[2] ** 2,
            outcome[3] ** 2,
            0.5 * (action[0] ** 2 + action[2] ** 2),
            0.5 * (action[1] ** 2 + action[3] ** 2),
            collision,
        ),
        dtype=np.float64,
    )
    weights = np.asarray(tuple(COST_WEIGHTS.values()), dtype=np.float64)
    cost = float(np.dot(terms, weights))
    if not math.isfinite(cost):
        raise RuntimeError("physical cost is non-finite")
    return cost, terms


@dataclass
class CandidateOutcome:
    actor_id: int
    world_frames: np.ndarray
    initial_actual_transform: np.ndarray
    initial_actual_speed_mps: float
    initial_features_actual: np.ndarray
    initial_features_actual_raw: np.ndarray
    initial_velocity_command: np.ndarray
    reset_diagnostics: np.ndarray
    trajectory_world: np.ndarray
    trajectory_road: np.ndarray
    intended_controls: np.ndarray
    applied_controls: np.ndarray
    control_execution_max_abs_error: float
    outcome_features: np.ndarray
    outcome_raw: np.ndarray
    collision: float
    collision_count: int
    collision_impulse_sum: float
    cost: float
    cost_terms: np.ndarray


class CarlaSession:
    """Own temporary world settings and every actor spawned by the collector."""

    def __init__(
        self,
        carla: Any,
        host: str,
        port: int,
        timeout: float,
        *,
        map_name: str = MAP_NAME,
    ) -> None:
        self.carla = carla
        self.host = host
        self.port = port
        self.map_name = str(map_name)
        self.map_short_name = self.map_name.rstrip("/").split("/")[-1]
        if not self.map_short_name:
            raise ValueError("map_name must identify a CARLA map")
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world: Any | None = None
        self.road_map: Any | None = None
        self.original_settings: Any | None = None
        self.original_weather: Any | None = None
        self.traffic_lights: list[tuple[Any, Any, bool]] = []
        self.active_actors: dict[int, Any] = {}
        self.owned_actor_ids: set[int] = set()
        self.map_loaded = False
        self.cleanup_report: dict[str, Any] = {}
        self.server_metadata: dict[str, Any] = {}

    def setup(self) -> None:
        client_version = self.client.get_client_version()
        server_version = self.client.get_server_version()
        if client_version != REQUIRED_CARLA_VERSION or server_version != REQUIRED_CARLA_VERSION:
            raise RuntimeError(
                f"exact CARLA 0.9.15 client/server required; client={client_version}, "
                f"server={server_version}"
            )
        world = self.client.get_world()
        current_map = world.get_map().name
        if not current_map.endswith(self.map_short_name):
            available = self.client.get_available_maps()
            matches = [name for name in available if name.endswith(self.map_short_name)]
            if not matches:
                raise RuntimeError(f"server does not provide {self.map_short_name}")
            world = self.client.load_world(self.map_name)
            self.map_loaded = True
        road_map = world.get_map()
        if not road_map.name.endswith(self.map_short_name):
            raise RuntimeError(f"unexpected loaded map: {road_map.name}")

        dynamic = [
            actor
            for actor in world.get_actors()
            if actor.type_id.startswith("vehicle.")
            or actor.type_id.startswith("walker.pedestrian.")
        ]
        if dynamic:
            identities = [(actor.id, actor.type_id) for actor in dynamic[:20]]
            raise RuntimeError(
                "collector requires a traffic-free world and will not destroy foreign actors; "
                f"found {identities}"
            )

        self.world = world
        self.road_map = road_map
        self.original_settings = world.get_settings()
        self.original_weather = world.get_weather()
        run_settings = world.get_settings()
        run_settings.synchronous_mode = True
        run_settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        run_settings.no_rendering_mode = True
        if hasattr(run_settings, "substepping"):
            run_settings.substepping = True
            run_settings.max_substep_delta_time = 0.01
            run_settings.max_substeps = 5
        world.apply_settings(run_settings)
        world.set_weather(self.carla.WeatherParameters.ClearNoon)

        for light in world.get_actors().filter("traffic.traffic_light*"):
            is_frozen = bool(light.is_frozen()) if hasattr(light, "is_frozen") else False
            self.traffic_lights.append((light, light.get_state(), is_frozen))
            light.freeze(True)
        world.tick()

        self.server_metadata = {
            "host": self.host,
            "port": self.port,
            "client_version": client_version,
            "server_version": server_version,
            "map_loaded_by_collector": self.map_loaded,
            # ``map_name`` is the canonical config identity consumed by the
            # runner; CARLA's shorter runtime spelling is retained separately.
            "map_name": self.map_name,
            "actual_map_name": road_map.name,
            "opendrive_sha256": _sha256_bytes(road_map.to_opendrive().encode("utf-8")),
            "original_settings": _settings_json(self.original_settings),
            "collection_settings": _settings_json(world.get_settings()),
            "original_weather": _weather_json(self.original_weather),
            "collection_weather": _weather_json(world.get_weather()),
            "traffic_light_count_frozen": len(self.traffic_lights),
        }

    def _register(self, actor: Any) -> Any:
        self.active_actors[int(actor.id)] = actor
        self.owned_actor_ids.add(int(actor.id))
        return actor

    def _destroy(self, actor: Any | None) -> None:
        if actor is None:
            return
        actor_id = int(actor.id)
        try:
            if hasattr(actor, "is_listening") and actor.is_listening:
                actor.stop()
        finally:
            try:
                actor.destroy()
            finally:
                self.active_actors.pop(actor_id, None)

    def _submit_vehicle_control(self, vehicle: Any, control: Any) -> None:
        """Synchronously acknowledge a control RPC before advancing the world."""

        responses = self.client.apply_batch_sync(
            [self.carla.command.ApplyVehicleControl(int(vehicle.id), control)],
            False,
        )
        if len(responses) != 1 or responses[0].has_error():
            error = responses[0].error if responses else "missing CARLA response"
            raise RuntimeError(f"CARLA rejected vehicle control: {error}")

    def run_candidate(self, state: StateSpec, action: np.ndarray) -> CandidateOutcome:
        if self.world is None or self.road_map is None:
            raise RuntimeError("CARLA session is not initialized")
        world = self.world
        blueprint = world.get_blueprint_library().find(VEHICLE_BLUEPRINT)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "mpc_local_collector")
        for attribute in ("color", "driver_id"):
            if blueprint.has_attribute(attribute):
                recommended = list(blueprint.get_attribute(attribute).recommended_values)
                if recommended:
                    blueprint.set_attribute(attribute, sorted(recommended)[0])

        vehicle: Any | None = None
        collision_sensor: Any | None = None
        collision_events: list[tuple[int, float]] = []
        try:
            # CARLA synchronous mode can return a newly spawned actor before
            # its replicated transform is queryable (the pre-tick getter may
            # report the origin).  Use a private transform copy, materialize
            # the fresh actor for one physics-disabled tick, and then install
            # the exact candidate reset state.
            spawn_transform = self.carla.Transform(
                self.carla.Location(
                    x=float(state.transform.location.x),
                    y=float(state.transform.location.y),
                    z=float(state.transform.location.z),
                ),
                self.carla.Rotation(
                    pitch=float(state.transform.rotation.pitch),
                    yaw=float(state.transform.rotation.yaw),
                    roll=float(state.transform.rotation.roll),
                ),
            )
            requested = _transform_array(spawn_transform)
            vehicle = world.try_spawn_actor(blueprint, spawn_transform)
            if vehicle is None:
                world.tick()
                vehicle = world.try_spawn_actor(blueprint, spawn_transform)
            if vehicle is None:
                raise RuntimeError(
                    f"failed to freshly spawn {VEHICLE_BLUEPRINT} at state {state.state_id}"
                )
            self._register(vehicle)
            vehicle.set_simulate_physics(False)
            world.tick()
            vehicle.set_transform(spawn_transform)
            # A second disabled tick commits set_transform without integrating
            # physics.  This gives an observable exact pose and zero-velocity
            # reset.  The initial velocity is then issued as a t=0 command and
            # consumed together with the first action on the first horizon tick.
            world.tick()
            actual_transform = vehicle.get_transform()
            actual_velocity = vehicle.get_velocity()
            actual_angular_velocity = vehicle.get_angular_velocity()
            actual = _transform_array(actual_transform)
            position_error = float(np.linalg.norm(actual[:3] - requested[:3]))
            rotation_error = max(
                abs(_wrap_angle_rad(actual[index] - requested[index]))
                for index in (3, 4, 5)
            )
            # Physics is disabled during reset, so observed speed must be zero.
            # The non-zero initial speed is explicitly attested as a queued t=0
            # command below, rather than falsely comparing it with this getter.
            speed_error = _vector_norm(actual_velocity)
            angular_speed = math.radians(_vector_norm(actual_angular_velocity))
            reset_diagnostics = np.asarray(
                (position_error, rotation_error, speed_error, angular_speed),
                dtype=np.float64,
            )
            tolerances = np.asarray(tuple(RESET_TOLERANCES.values()), dtype=np.float64)
            if np.any(reset_diagnostics > tolerances):
                raise RuntimeError(
                    f"state {state.state_id} fresh-spawn reset exceeded tolerances: "
                    f"observed={reset_diagnostics.tolist()} tolerances={tolerances.tolist()} "
                    f"spawn_transform={requested.tolist()} "
                    f"actual_transform={actual.tolist()}"
                )

            forward = state.transform.get_forward_vector()
            initial_velocity_command = self.carla.Vector3D(
                x=float(forward.x) * state.initial_speed_mps,
                y=float(forward.y) * state.initial_speed_mps,
                z=float(forward.z) * state.initial_speed_mps,
            )
            initial_velocity_command_array = np.asarray(
                (
                    initial_velocity_command.x,
                    initial_velocity_command.y,
                    initial_velocity_command.z,
                ),
                dtype=np.float64,
            )
            command_speed_error = abs(
                float(np.linalg.norm(initial_velocity_command_array))
                - state.initial_speed_mps
            )
            if command_speed_error > CONTROL_EXECUTION_TOLERANCE:
                raise RuntimeError(
                    f"initial velocity command magnitude mismatch: {command_speed_error}"
                )

            # CARLA 0.9.15 may drop manual control during actor/physics
            # activation.  Use fixed action-independent neutral physics ticks
            # to acquire the initial velocity.  The post-warm-up *actual* pose,
            # speed and road features are the candidate's t=0 state.  The
            # collision sensor is attached only afterwards, so this neutral
            # initialization can never become an action-outcome label.
            vehicle.set_simulate_physics(True)
            vehicle.set_target_velocity(initial_velocity_command)
            vehicle.set_target_angular_velocity(self.carla.Vector3D(0.0, 0.0, 0.0))
            neutral_control = self.carla.VehicleControl(
                throttle=0.0,
                steer=0.0,
                brake=0.0,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
            for warmup_tick in range(NEUTRAL_PHYSICS_WARMUP_TICKS):
                self._submit_vehicle_control(vehicle, neutral_control)
                world.tick()
                warmup_control = vehicle.get_control()
                warmup_applied = np.asarray(
                    (warmup_control.steer, warmup_control.throttle, warmup_control.brake),
                    dtype=np.float64,
                )
                if float(np.max(np.abs(warmup_applied))) > CONTROL_EXECUTION_TOLERANCE:
                    raise RuntimeError(
                        f"neutral warm-up tick {warmup_tick} applied non-neutral control: "
                        f"{warmup_applied.tolist()}"
                    )

            actual_transform = vehicle.get_transform()
            actual_velocity = vehicle.get_velocity()
            actual_speed = _vector_norm(actual_velocity)
            actual = _transform_array(actual_transform)
            initial_waypoint = self.road_map.get_waypoint(
                actual_transform.location,
                project_to_road=False,
                lane_type=self.carla.LaneType.Driving,
            )
            if initial_waypoint is None or bool(initial_waypoint.is_junction):
                raise RuntimeError(
                    f"state {state.state_id} post-warm-up t=0 is not on a "
                    "non-junction driving lane"
                )
            (
                initial_features_actual,
                initial_features_actual_raw,
                initial_features_actual_valid,
            ) = _initial_features(actual_transform, initial_waypoint, actual_speed)
            if not bool(initial_features_actual_valid.all()):
                raise RuntimeError(
                    f"state {state.state_id} post-warm-up initial features are incomplete"
                )

            collision_blueprint = world.get_blueprint_library().find("sensor.other.collision")
            collision_sensor = self._register(
                world.spawn_actor(collision_blueprint, self.carla.Transform(), attach_to=vehicle)
            )

            def on_collision(event: Any) -> None:
                collision_events.append((int(event.frame), _vector_norm(event.normal_impulse)))

            collision_sensor.listen(on_collision)

            trajectory_world = np.empty((HORIZON_TICKS + 1, 5), dtype=np.float64)
            trajectory_road = np.empty((HORIZON_TICKS + 1, 5), dtype=np.float64)
            intended_controls = np.empty((HORIZON_TICKS, 3), dtype=np.float64)
            applied_controls = np.empty((HORIZON_TICKS, 3), dtype=np.float64)
            initial_normalized, initial_raw, initial_on_lane = _road_state(
                self.road_map,
                self.carla,
                actual_transform,
                actual_transform,
                actual_speed,
            )
            trajectory_world[0] = (
                actual_transform.location.x,
                actual_transform.location.y,
                actual_transform.location.z,
                math.radians(actual_transform.rotation.yaw),
                actual_speed,
            )
            trajectory_road[0] = (*initial_raw, float(initial_on_lane))
            frame_start = int(world.get_snapshot().frame)
            frame_end = frame_start

            for tick in range(HORIZON_TICKS):
                segment = 0 if tick < HORIZON_TICKS // ACTION_SEGMENTS else 1
                steer = float(action[2 * segment])
                longitudinal = float(action[2 * segment + 1])
                control = self.carla.VehicleControl(
                    throttle=max(longitudinal, 0.0),
                    steer=steer,
                    brake=max(-longitudinal, 0.0),
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False,
                )
                intended_controls[tick] = (control.steer, control.throttle, control.brake)
                self._submit_vehicle_control(vehicle, control)
                frame_end = int(world.tick())
                applied = vehicle.get_control()
                applied_controls[tick] = (applied.steer, applied.throttle, applied.brake)
                transform = vehicle.get_transform()
                speed = _vector_norm(vehicle.get_velocity())
                normalized, raw, on_lane = _road_state(
                    self.road_map, self.carla, actual_transform, transform, speed
                )
                trajectory_world[tick + 1] = (
                    transform.location.x,
                    transform.location.y,
                    transform.location.z,
                    math.radians(transform.rotation.yaw),
                    speed,
                )
                trajectory_road[tick + 1] = (*raw, float(on_lane))

            # Collision callbacks are asynchronous to the Python thread even in
            # synchronous simulation.  Give already-produced events a bounded
            # opportunity to arrive without advancing the physical world.
            time.sleep(0.005)
            control_execution_max_abs_error = float(
                np.max(np.abs(applied_controls - intended_controls))
            )
            if control_execution_max_abs_error > CONTROL_EXECUTION_TOLERANCE:
                difference = np.abs(applied_controls - intended_controls)
                worst_tick, worst_field = np.unravel_index(
                    int(np.argmax(difference)), difference.shape
                )
                raise RuntimeError(
                    "CARLA applied control differs from the intended control: "
                    f"max_abs_error={control_execution_max_abs_error:.9g}, "
                    f"tolerance={CONTROL_EXECUTION_TOLERANCE:.9g}, "
                    f"tick={worst_tick}, field={('steer', 'throttle', 'brake')[worst_field]}, "
                    f"intended={intended_controls[worst_tick].tolist()}, "
                    f"applied={applied_controls[worst_tick].tolist()}, "
                    f"action={np.asarray(action).tolist()}"
                )
            collision = float(bool(collision_events))
            outcome_features = np.asarray(
                (
                    trajectory_road[-1, 0] / 10.0,
                    trajectory_road[-1, 1] / 3.0,
                    trajectory_road[-1, 2] / math.pi,
                    (trajectory_road[-1, 3] - TARGET_SPEED_MPS) / TARGET_SPEED_MPS,
                ),
                dtype=np.float64,
            )
            outcome_raw = trajectory_road[-1, :4].copy()
            cost, cost_terms = _physical_cost(outcome_features, action, collision)
            return CandidateOutcome(
                actor_id=int(vehicle.id),
                world_frames=np.asarray((frame_start, frame_end), dtype=np.int64),
                initial_actual_transform=actual,
                initial_actual_speed_mps=actual_speed,
                initial_features_actual=initial_features_actual,
                initial_features_actual_raw=initial_features_actual_raw,
                initial_velocity_command=initial_velocity_command_array,
                reset_diagnostics=reset_diagnostics,
                trajectory_world=trajectory_world,
                trajectory_road=trajectory_road,
                intended_controls=intended_controls,
                applied_controls=applied_controls,
                control_execution_max_abs_error=control_execution_max_abs_error,
                outcome_features=outcome_features,
                outcome_raw=outcome_raw,
                collision=collision,
                collision_count=len(collision_events),
                collision_impulse_sum=float(sum(item[1] for item in collision_events)),
                cost=cost,
                cost_terms=cost_terms,
            )
        finally:
            self._destroy(collision_sensor)
            self._destroy(vehicle)
            if self.world is not None:
                self.world.tick()

    def cleanup(self) -> dict[str, Any]:
        errors: list[str] = []
        for actor in list(self.active_actors.values())[::-1]:
            try:
                self._destroy(actor)
            except Exception as exc:  # Best effort continues for all owned actors.
                errors.append(f"destroy actor {getattr(actor, 'id', '?')}: {type(exc).__name__}: {exc}")
        if self.world is not None:
            try:
                if self.world.get_settings().synchronous_mode:
                    self.world.tick()
            except Exception as exc:
                errors.append(f"cleanup tick: {type(exc).__name__}: {exc}")
            for light, state, was_frozen in self.traffic_lights:
                try:
                    light.set_state(state)
                    light.freeze(was_frozen)
                except Exception as exc:
                    errors.append(f"restore traffic light {light.id}: {type(exc).__name__}: {exc}")
            try:
                if self.original_weather is not None:
                    self.world.set_weather(self.original_weather)
            except Exception as exc:
                errors.append(f"restore weather: {type(exc).__name__}: {exc}")
            try:
                if self.original_settings is not None:
                    self.world.apply_settings(self.original_settings)
            except Exception as exc:
                errors.append(f"restore settings: {type(exc).__name__}: {exc}")

        remaining_owned: list[int] = []
        if self.world is not None:
            try:
                actor_ids = {int(actor.id) for actor in self.world.get_actors()}
                remaining_owned = sorted(self.owned_actor_ids & actor_ids)
            except Exception as exc:
                errors.append(f"verify actor cleanup: {type(exc).__name__}: {exc}")
        report = {
            "actors_remaining": remaining_owned,
            "settings_restored": not errors and self.world is not None,
            "restored_settings": _settings_json(self.world.get_settings()) if self.world is not None else None,
            "errors": errors,
        }
        self.cleanup_report = report
        if errors or remaining_owned:
            raise RuntimeError(f"CARLA cleanup failed closed: {report}")
        return report


def _protocol(
    smoke: bool, seed: int, action_profile: str = DEFAULT_ACTION_PROFILE
) -> dict[str, Any]:
    split_counts = SMOKE_SPLIT_COUNTS if smoke else FULL_SPLIT_COUNTS
    population = SMOKE_POPULATION if smoke else FULL_POPULATION
    elite_count = SMOKE_ELITE_COUNT if smoke else FULL_ELITE_COUNT
    profile = _action_profile_spec(action_profile)
    return {
        "schema_version": SCHEMA_VERSION,
        "smoke": smoke,
        "seed": seed,
        "action_profile": action_profile,
        "map": MAP_NAME,
        "carla_version": REQUIRED_CARLA_VERSION,
        "vehicle_blueprint": VEHICLE_BLUEPRINT,
        "fixed_delta_seconds": FIXED_DELTA_SECONDS,
        "horizon_ticks": HORIZON_TICKS,
        "horizon_seconds": HORIZON_TICKS * FIXED_DELTA_SECONDS,
        "action_parameterization": "[steer1,longitudinal1,steer2,longitudinal2]",
        "longitudinal_mapping": "positive=throttle; negative=brake",
        "initial_speeds_mps": list(INITIAL_SPEEDS_MPS),
        "target_speed_mps": TARGET_SPEED_MPS,
        "minimum_forward_non_junction_road_m": MINIMUM_FORWARD_ROAD_M,
        "split_counts": dict(split_counts),
        "split_codes": dict(SPLIT_CODES),
        "state_selection": {
            "excluded_source_spawn_indices": list(
                profile["excluded_source_spawn_indices"]
            ),
            "excluded_parent": profile["remediation_parent"],
            "fresh_relative_to_parent": bool(
                profile["excluded_source_spawn_indices"]
            ),
        },
        "cem": {
            "iterations": CEM_ITERATIONS,
            "population": population,
            "elite_count": elite_count,
            "initial_mean": profile["initial_mean"].tolist(),
            "initial_std": profile["initial_std"].tolist(),
            "lower": profile["lower"].tolist(),
            "upper": profile["upper"].tolist(),
            "minimum_std": profile["minimum_std"].tolist(),
            "sampling": "numpy.PCG64 normal then componentwise clamp; stable cost/index elite order",
        },
        "cost": {
            "formula": (
                "w_progress*outcome[0] + w_lateral*outcome[1]^2 + "
                "w_yaw*outcome[2]^2 + w_speed*outcome[3]^2 + "
                "w_steer*mean(action[0]^2,action[2]^2) + "
                "w_long*mean(action[1]^2,action[3]^2) + w_collision*collision"
            ),
            "weights": dict(COST_WEIGHTS),
            "pair_tie_threshold": PAIR_TIE_THRESHOLD,
            "outcome_feature_names": list(OUTCOME_FEATURE_NAMES),
        },
        "initial_feature_names": list(INITIAL_FEATURE_NAMES),
        "reset": {
            "policy": "fresh vehicle and fresh attached collision sensor for every candidate",
            "pose": "two physics-disabled ticks commit and verify the exact spawn transform",
            "initial_velocity": (
                "world-frame spawn-forward vector times state initial_speed_mps; consumed by "
                f"{NEUTRAL_PHYSICS_WARMUP_TICKS} action-independent neutral physics "
                "warm-up ticks"
            ),
            "candidate_t0": (
                "actual post-neutral-warm-up transform, speed, and road features; state-wise "
                "equality is audited across every candidate"
            ),
            "tolerances": dict(RESET_TOLERANCES),
            "probe_repeats": RESET_PROBE_REPEATS,
        },
        "control_execution": {
            "recorded_fields": ["steer", "throttle", "brake"],
            "submission": (
                "client.apply_batch_sync(command.ApplyVehicleControl, do_tick=False) "
                "followed by world.tick"
            ),
            "audit": "vehicle.get_control after every synchronous tick",
            "first_tick_policy": (
                f"{NEUTRAL_PHYSICS_WARMUP_TICKS} action-independent neutral physics "
                "warm-up ticks occur before t=0; the "
                "collision sensor and all 20 action ticks begin only after warm-up"
            ),
            "max_abs_error_tolerance": CONTROL_EXECUTION_TOLERANCE,
        },
    }


def _allocate(state_count: int, population: int, elite_count: int) -> dict[str, np.ndarray]:
    shape = (state_count, CEM_ITERATIONS, population)
    return {
        "action_params": np.empty((*shape, 4), dtype=np.float32),
        "real_cost": np.empty(shape, dtype=np.float32),
        "cost_terms": np.empty((*shape, 7), dtype=np.float32),
        "outcome_features": np.empty((*shape, 4), dtype=np.float32),
        "outcome_raw": np.empty((*shape, 4), dtype=np.float32),
        "collision": np.empty(shape, dtype=np.float32),
        "collision_count": np.empty(shape, dtype=np.int16),
        "collision_impulse_sum": np.empty(shape, dtype=np.float32),
        "actor_id": np.empty(shape, dtype=np.int64),
        "world_frames": np.empty((*shape, 2), dtype=np.int64),
        "initial_actual_transform": np.empty((*shape, 6), dtype=np.float64),
        "initial_actual_speed_mps": np.empty(shape, dtype=np.float64),
        "initial_features_actual": np.empty((*shape, len(INITIAL_FEATURE_NAMES)), dtype=np.float64),
        "initial_features_actual_raw": np.empty((*shape, len(INITIAL_FEATURE_NAMES)), dtype=np.float64),
        "initial_velocity_command": np.empty((*shape, 3), dtype=np.float32),
        "reset_diagnostics": np.empty((*shape, 4), dtype=np.float64),
        "trajectory_world": np.empty((*shape, HORIZON_TICKS + 1, 5), dtype=np.float32),
        "trajectory_road": np.empty((*shape, HORIZON_TICKS + 1, 5), dtype=np.float32),
        "intended_controls": np.empty((*shape, HORIZON_TICKS, 3), dtype=np.float32),
        "applied_controls": np.empty((*shape, HORIZON_TICKS, 3), dtype=np.float32),
        "control_execution_max_abs_error": np.empty(shape, dtype=np.float32),
        "candidate_rank": np.empty(shape, dtype=np.int16),
        "elite_mask": np.zeros(shape, dtype=np.bool_),
        "elite_indices": np.empty((state_count, CEM_ITERATIONS, elite_count), dtype=np.int16),
        "cem_mean_before": np.empty((state_count, CEM_ITERATIONS, 4), dtype=np.float32),
        "cem_std_before": np.empty((state_count, CEM_ITERATIONS, 4), dtype=np.float32),
        "cem_mean_after": np.empty((state_count, CEM_ITERATIONS, 4), dtype=np.float32),
        "cem_std_after": np.empty((state_count, CEM_ITERATIONS, 4), dtype=np.float32),
    }


def _reset_probe(
    session: CarlaSession, state: StateSpec, action: np.ndarray
) -> dict[str, Any]:
    outcomes = [session.run_candidate(state, action) for _ in range(RESET_PROBE_REPEATS)]
    trajectories = np.stack([item.trajectory_world for item in outcomes])
    terminals = np.stack([item.outcome_features for item in outcomes])
    costs = np.asarray([item.cost for item in outcomes], dtype=np.float64)
    max_trajectory_delta = float(np.max(np.abs(trajectories - trajectories[0:1])))
    max_terminal_delta = float(np.max(np.abs(terminals - terminals[0:1])))
    max_cost_delta = float(np.max(np.abs(costs - costs[0])))
    bitwise_equal = all(
        item.trajectory_world.tobytes() == outcomes[0].trajectory_world.tobytes()
        and item.outcome_features.tobytes() == outcomes[0].outcome_features.tobytes()
        and np.float64(item.cost).tobytes() == np.float64(outcomes[0].cost).tobytes()
        for item in outcomes[1:]
    )
    if not bitwise_equal:
        raise RuntimeError(
            "fresh-spawn reproducibility probe was not bitwise exact: "
            f"trajectory={max_trajectory_delta}, terminal={max_terminal_delta}, cost={max_cost_delta}"
        )
    return {
        "state_id": state.state_id,
        "action": np.asarray(action, dtype=np.float64).tolist(),
        "repeats": RESET_PROBE_REPEATS,
        "bitwise_equal": bitwise_equal,
        "max_abs_trajectory_delta": max_trajectory_delta,
        "max_abs_terminal_delta": max_terminal_delta,
        "max_abs_cost_delta": max_cost_delta,
        "trajectory_sha256": _sha256_bytes(outcomes[0].trajectory_world.tobytes()),
        "terminal_outcome": outcomes[0].outcome_features.tolist(),
        "cost": outcomes[0].cost,
        "actor_ids": [item.actor_id for item in outcomes],
        "fresh_actor_ids_unique": len({item.actor_id for item in outcomes}) == len(outcomes),
        "collision_count": int(sum(item.collision_count for item in outcomes)),
    }


def _collection_progress_message(
    state: StateSpec,
    *,
    state_count: int,
    iteration: int,
    sealed_test_redaction: bool,
    best_cost: float | None = None,
    collision_count: int | None = None,
    std: np.ndarray | None = None,
) -> str:
    if sealed_test_redaction and state.split == "test":
        return f"split=test iter={iteration} sealed_progress=true"
    if best_cost is None or collision_count is None or std is None:
        raise ValueError("unsealed progress requires cost, collision, and std values")
    return (
        f"state={state.state_id:02d}/{state_count - 1:02d} split={state.split} "
        f"iter={iteration} best={best_cost:.6f} collisions={collision_count} "
        f"std={np.round(std, 4).tolist()}"
    )


def _collect(session: CarlaSession, states: Sequence[StateSpec], protocol: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    population = int(protocol["cem"]["population"])
    elite_count = int(protocol["cem"]["elite_count"])
    initial_mean = np.asarray(protocol["cem"]["initial_mean"], dtype=np.float64)
    initial_std = np.asarray(protocol["cem"]["initial_std"], dtype=np.float64)
    lower = np.asarray(protocol["cem"]["lower"], dtype=np.float64)
    upper = np.asarray(protocol["cem"]["upper"], dtype=np.float64)
    minimum_std = np.asarray(protocol["cem"]["minimum_std"], dtype=np.float64)
    # V3 physically seals its test payload.  This opt-in flag only redacts
    # outcome-derived console/summary fields; all earlier V1/V2 protocols omit
    # it and retain their byte-for-byte control flow and reporting behavior.
    sealed_test_redaction = bool(protocol.get("sealed_test_redaction", False))
    arrays = _allocate(len(states), population, elite_count)
    probe = _reset_probe(session, states[0], initial_mean)
    print(
        f"reset probe: repeats={probe['repeats']} bitwise_equal={probe['bitwise_equal']} "
        f"collision_count={probe['collision_count']}",
        flush=True,
    )

    for state_index, state in enumerate(states):
        seed_sequence = np.random.SeedSequence((int(protocol["seed"]), state.state_id, state.source_spawn_index))
        state_seed = int(seed_sequence.generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(state_seed)
        mean = initial_mean.copy()
        std = initial_std.copy()
        for iteration in range(CEM_ITERATIONS):
            arrays["cem_mean_before"][state_index, iteration] = mean
            arrays["cem_std_before"][state_index, iteration] = std
            candidates = np.clip(
                rng.normal(loc=mean, scale=std, size=(population, 4)),
                lower,
                upper,
            )
            arrays["action_params"][state_index, iteration] = candidates
            for candidate_index, action in enumerate(candidates):
                result = session.run_candidate(state, action)
                index = (state_index, iteration, candidate_index)
                arrays["real_cost"][index] = result.cost
                arrays["cost_terms"][index] = result.cost_terms
                arrays["outcome_features"][index] = result.outcome_features
                arrays["outcome_raw"][index] = result.outcome_raw
                arrays["collision"][index] = result.collision
                arrays["collision_count"][index] = result.collision_count
                arrays["collision_impulse_sum"][index] = result.collision_impulse_sum
                arrays["actor_id"][index] = result.actor_id
                arrays["world_frames"][index] = result.world_frames
                arrays["initial_actual_transform"][index] = result.initial_actual_transform
                arrays["initial_actual_speed_mps"][index] = result.initial_actual_speed_mps
                arrays["initial_features_actual"][index] = result.initial_features_actual
                arrays["initial_features_actual_raw"][index] = (
                    result.initial_features_actual_raw
                )
                arrays["initial_velocity_command"][index] = result.initial_velocity_command
                arrays["reset_diagnostics"][index] = result.reset_diagnostics
                arrays["trajectory_world"][index] = result.trajectory_world
                arrays["trajectory_road"][index] = result.trajectory_road
                arrays["intended_controls"][index] = result.intended_controls
                arrays["applied_controls"][index] = result.applied_controls
                arrays["control_execution_max_abs_error"][index] = (
                    result.control_execution_max_abs_error
                )

            costs = arrays["real_cost"][state_index, iteration].astype(np.float64)
            order = np.argsort(costs, kind="stable")
            rank = np.empty(population, dtype=np.int16)
            rank[order] = np.arange(population, dtype=np.int16)
            elite_indices = order[:elite_count]
            arrays["candidate_rank"][state_index, iteration] = rank
            arrays["elite_mask"][state_index, iteration, elite_indices] = True
            arrays["elite_indices"][state_index, iteration] = elite_indices
            elites = candidates[elite_indices]
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0, ddof=0), minimum_std)
            arrays["cem_mean_after"][state_index, iteration] = mean
            arrays["cem_std_after"][state_index, iteration] = std
            if sealed_test_redaction and state.split == "test":
                print(
                    _collection_progress_message(
                        state,
                        state_count=len(states),
                        iteration=iteration,
                        sealed_test_redaction=True,
                    ),
                    flush=True,
                )
            else:
                collisions = int(
                    np.count_nonzero(arrays["collision"][state_index, iteration])
                )
                print(
                    _collection_progress_message(
                        state,
                        state_count=len(states),
                        iteration=iteration,
                        sealed_test_redaction=False,
                        best_cost=float(costs[order[0]]),
                        collision_count=collisions,
                        std=std,
                    ),
                    flush=True,
                )

    reset_max = arrays["reset_diagnostics"].max(axis=(0, 1, 2)).tolist()
    commanded_speeds = np.linalg.norm(
        arrays["initial_velocity_command"].astype(np.float64), axis=-1
    )
    expected_speeds = np.asarray(
        [state.initial_speed_mps for state in states], dtype=np.float64
    )[:, None, None]
    initial_velocity_command_max_abs_error = float(
        np.max(np.abs(commanded_speeds - expected_speeds))
    )
    paired_initial_state: list[dict[str, Any]] = []
    for state_index, state in enumerate(states):
        transforms = arrays["initial_actual_transform"][state_index].reshape(-1, 6)
        speeds = arrays["initial_actual_speed_mps"][state_index].reshape(-1)
        features = arrays["initial_features_actual"][state_index].reshape(
            -1, len(INITIAL_FEATURE_NAMES)
        )
        reference_transform = transforms[0]
        position_delta = np.linalg.norm(
            transforms[:, :3] - reference_transform[None, :3], axis=1
        )
        rotation_delta = np.abs(
            (transforms[:, 3:] - reference_transform[None, 3:] + math.pi)
            % (2.0 * math.pi)
            - math.pi
        )
        max_position_delta = float(position_delta.max())
        max_rotation_delta = float(rotation_delta.max())
        max_speed_delta = float(np.max(np.abs(speeds - speeds[0])))
        max_feature_delta = float(
            np.max(np.abs(features - features[0:1]))
        )
        passed = max(
            max_position_delta,
            max_rotation_delta,
            max_speed_delta,
            max_feature_delta,
        ) <= PAIRED_INITIAL_STATE_TOLERANCE
        report = {
            "state_id": state.state_id,
            "candidate_count": int(transforms.shape[0]),
            "bitwise_equal": bool(
                np.array_equal(transforms, np.broadcast_to(transforms[0], transforms.shape))
                and np.array_equal(speeds, np.broadcast_to(speeds[0], speeds.shape))
                and np.array_equal(features, np.broadcast_to(features[0], features.shape))
            ),
            "max_position_delta_m": max_position_delta,
            "max_rotation_delta_rad": max_rotation_delta,
            "max_speed_delta_mps": max_speed_delta,
            "max_initial_feature_delta": max_feature_delta,
            "passed": passed,
        }
        paired_initial_state.append(report)
        if not passed:
            raise RuntimeError(
                f"state {state.state_id} candidates do not share the same post-warm-up "
                f"initial state within {PAIRED_INITIAL_STATE_TOLERANCE}: {report}"
            )
    outcome_summary_indices = [
        index
        for index, state in enumerate(states)
        if not (sealed_test_redaction and state.split == "test")
    ]
    outcome_summary_splits = (
        ("train", "val") if sealed_test_redaction else ("train", "val", "test")
    )
    collection_summary = {
        "reset_probe": probe,
        "reset_observed_max": {
            key: float(value) for key, value in zip(RESET_TOLERANCES, reset_max, strict=True)
        },
        "candidate_count": int(len(states) * CEM_ITERATIONS * population),
        "collision_label_count": int(
            np.count_nonzero(arrays["collision"][outcome_summary_indices])
        ),
        "collision_event_count": int(
            arrays["collision_count"][outcome_summary_indices].sum()
        ),
        "collision_by_split": {
            split: int(
                np.count_nonzero(
                    arrays["collision"][[index for index, state in enumerate(states) if state.split == split]]
                )
            )
            for split in outcome_summary_splits
        },
        "control_execution_audit_passed": bool(
            float(arrays["control_execution_max_abs_error"].max())
            <= CONTROL_EXECUTION_TOLERANCE
        ),
        "control_execution_max_abs_error": float(
            arrays["control_execution_max_abs_error"].max()
        ),
        "initial_velocity_command_audit_passed": bool(
            initial_velocity_command_max_abs_error <= 1.0e-6
        ),
        "initial_velocity_command_max_abs_error": initial_velocity_command_max_abs_error,
        "paired_initial_state_tolerance": PAIRED_INITIAL_STATE_TOLERANCE,
        "paired_initial_state_passed": all(item["passed"] for item in paired_initial_state),
        "paired_initial_state": paired_initial_state,
    }
    if sealed_test_redaction:
        development_collision_labels = collection_summary.pop(
            "collision_label_count"
        )
        development_collision_events = collection_summary.pop(
            "collision_event_count"
        )
        # Even a development-only ``collision_by_split`` mapping is removed:
        # sealed protocols expose one explicitly scoped gate result, never a
        # generic field whose historical meaning included the test split.
        collection_summary.pop("collision_by_split")
        collection_summary.update(
            {
                "collision_gate_scope": "development_only",
                "development_collision_label_count": development_collision_labels,
                "development_collision_event_count": development_collision_events,
                "sealed_test_outcomes_redacted": True,
            }
        )
    return arrays, collection_summary


def _flat_records(states: Sequence[StateSpec], arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    state_count, iterations, population = arrays["real_cost"].shape
    records_per_state = iterations * population
    return {
        "state_id": np.repeat(
            np.asarray([state.state_id for state in states], dtype=np.int32), records_per_state
        ),
        "split_code": np.repeat(
            np.asarray([state.split_code for state in states], dtype=np.int8), records_per_state
        ),
        "cem_iteration": np.tile(
            np.repeat(np.arange(iterations, dtype=np.int8), population), state_count
        ),
        "action_params": np.asarray(arrays["action_params"], dtype=np.float32).reshape(-1, 4),
        "initial_features": np.asarray(
            arrays["initial_features_actual"], dtype=np.float32
        ).reshape(-1, len(INITIAL_FEATURE_NAMES)),
        "outcome_features": np.asarray(arrays["outcome_features"], dtype=np.float32).reshape(-1, 4),
        "collision": np.asarray(arrays["collision"], dtype=np.float32).reshape(-1),
        "real_cost": np.asarray(arrays["real_cost"], dtype=np.float32).reshape(-1),
    }


def _diagnostic_arrays(states: Sequence[StateSpec], arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    state_arrays = {
        "state_id": np.asarray([state.state_id for state in states], dtype=np.int32),
        "split_code": np.asarray([state.split_code for state in states], dtype=np.int8),
        "source_spawn_index": np.asarray([state.source_spawn_index for state in states], dtype=np.int32),
        "initial_speed_mps": np.asarray([state.initial_speed_mps for state in states], dtype=np.float32),
        "initial_transform": np.stack([_transform_array(state.transform) for state in states]),
        "initial_features": np.stack([state.initial_features for state in states]).astype(np.float32),
        "initial_features_raw": np.stack([state.initial_features_raw for state in states]).astype(np.float32),
        "initial_feature_valid": np.stack([state.initial_feature_valid for state in states]),
        "initial_waypoint_ids": np.asarray(
            [
                (
                    int(state.waypoint.road_id),
                    int(state.waypoint.section_id),
                    int(state.waypoint.lane_id),
                )
                for state in states
            ],
            dtype=np.int32,
        ),
        "initial_waypoint_s": np.asarray([state.waypoint.s for state in states], dtype=np.float32),
    }
    return {**state_arrays, **{key: np.asarray(value) for key, value in arrays.items()}}


def _validate_npz(path: Path, *, required_keys: set[str] | None = None) -> dict[str, Any]:
    arrays: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if required_keys is not None and keys != required_keys:
            raise RuntimeError(
                f"{path.name} array contract mismatch: expected={sorted(required_keys)} actual={sorted(keys)}"
            )
        for name in sorted(archive.files):
            value = archive[name]
            if value.dtype.kind == "O":
                raise RuntimeError(f"object array is forbidden: {path.name}:{name}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise RuntimeError(f"non-finite values in {path.name}:{name}")
            arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "arrays": arrays,
    }


def _state_manifest(states: Sequence[StateSpec]) -> list[dict[str, Any]]:
    return [
        {
            "state_id": state.state_id,
            "split": state.split,
            "split_code": state.split_code,
            "source_spawn_index": state.source_spawn_index,
            "initial_speed_mps": state.initial_speed_mps,
            "transform": _transform_json(state.transform),
            "waypoint": {
                "road_id": int(state.waypoint.road_id),
                "section_id": int(state.waypoint.section_id),
                "lane_id": int(state.waypoint.lane_id),
                "s": float(state.waypoint.s),
                "lane_width_m": float(state.waypoint.lane_width),
                "is_junction": bool(state.waypoint.is_junction),
            },
            "requested_spawn_initial_features": state.initial_features.tolist(),
        }
        for state in states
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect deterministic same-state CEM candidates from an existing "
            "CARLA 0.9.15 Town10HD_Opt server."
        )
    )
    parser.add_argument("--host", default="127.0.0.1", help="existing CARLA RPC host")
    parser.add_argument("--port", type=int, default=2100, help="existing CARLA RPC port")
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="CARLA client RPC timeout in seconds"
    )
    parser.add_argument(
        "--action-profile",
        choices=ACTION_PROFILE_NAMES,
        default=DEFAULT_ACTION_PROFILE,
        help="CEM action support; default v1 preserves the original collector",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="state/CEM RNG seed (profile default: v1=17031, safe_local_v2=27031)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new, non-existing output directory"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="3 states x 3 CEM iterations x 6 candidates (elite count 2)",
    )
    args = parser.parse_args(argv)
    if not (1 <= args.port <= 65535):
        parser.error("--port must be in [1,65535]")
    if not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("--timeout must be a positive finite number")
    profile_default_seed = int(_action_profile_spec(args.action_profile)["default_seed"])
    args.seed = profile_default_seed if args.seed is None else int(args.seed)
    if (
        args.action_profile == "safe_local_v2"
        and not args.smoke
        and args.seed != SAFE_LOCAL_V2_SEED
    ):
        parser.error(
            "non-smoke safe_local_v2 is frozen to collection seed 27031; "
            "use --smoke for development overrides"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output, staging = _prepare_staging(args.output)
    started_at = _utc_now()
    monotonic_start = time.monotonic()
    protocol = _protocol(args.smoke, args.seed, args.action_profile)
    carla = _import_carla()
    session = CarlaSession(carla, args.host, args.port, args.timeout)
    cleanup_report: dict[str, Any] | None = None
    try:
        session.setup()
        if session.road_map is None:
            raise RuntimeError("CARLA road map was not initialized")
        split_counts = SMOKE_SPLIT_COUNTS if args.smoke else FULL_SPLIT_COUNTS
        states, spawn_summary = _select_states(
            session.road_map,
            carla,
            split_counts,
            args.seed,
            excluded_source_spawn_indices=protocol["state_selection"][
                "excluded_source_spawn_indices"
            ],
        )
        states_identity = _canonical_json_bytes(_state_manifest(states))
        print(
            f"connected client/server={REQUIRED_CARLA_VERSION}; map={session.road_map.name}; "
            f"profile={args.action_profile} states={len(states)} "
            f"population={protocol['cem']['population']} output={output}",
            flush=True,
        )
        try:
            arrays, collection_summary = _collect(session, states, protocol)
        finally:
            cleanup_report = session.cleanup()

        records = _flat_records(states, arrays)
        diagnostics = _diagnostic_arrays(states, arrays)
        records_path = staging / "records.npz"
        diagnostics_path = staging / "diagnostics.npz"
        _atomic_write_npz(records_path, records)
        _atomic_write_npz(diagnostics_path, diagnostics)
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
        file_metadata = {
            "records.npz": _validate_npz(records_path, required_keys=required_record_keys),
            "diagnostics.npz": _validate_npz(diagnostics_path),
        }
        finished_at = _utc_now()
        manifest = {
            # Runner-facing aliases are intentionally top-level.  Richer file
            # metadata remains under ``files`` for independent verification.
            "schema_version": 1,
            "dataset_schema": SCHEMA_VERSION,
            "action_profile": args.action_profile,
            "status": "complete",
            "started_at_utc": started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": time.monotonic() - monotonic_start,
            "output_directory": str(output),
            "invocation": [str(Path(__file__).resolve()), *sys.argv[1:]],
            "source": {
                "script": str(Path(__file__).resolve()),
                "script_sha256": _sha256_file(Path(__file__).resolve()),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
            },
            "protocol": protocol,
            "protocol_sha256": _sha256_bytes(_canonical_json_bytes(protocol)),
            "server_and_map": session.server_metadata,
            "spawn_summary": spawn_summary,
            "states": _state_manifest(states),
            "states_sha256": _sha256_bytes(states_identity),
            "fresh_state_attestation": {
                "excluded_source_spawn_indices": protocol["state_selection"][
                    "excluded_source_spawn_indices"
                ],
                "selected_source_spawn_indices": [
                    state.source_spawn_index for state in states
                ],
                "overlap": sorted(
                    set(protocol["state_selection"]["excluded_source_spawn_indices"])
                    & {state.source_spawn_index for state in states}
                ),
                "passed": not bool(
                    set(protocol["state_selection"]["excluded_source_spawn_indices"])
                    & {state.source_spawn_index for state in states}
                ),
            },
            "collection_summary": collection_summary,
            "cleanup": cleanup_report,
            "files": file_metadata,
            "records_sha256": file_metadata["records.npz"]["sha256"],
            "diagnostics_path": "diagnostics.npz",
            "diagnostics_sha256": file_metadata["diagnostics.npz"]["sha256"],
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
                split: [state.state_id for state in states if state.split == split]
                for split in ("train", "val", "test")
            },
            "sealed_test_policy": {
                "split_code": SPLIT_CODES["test"],
                "access": (
                    "test rows share records.npz but load_mpc_records rejects test access "
                    "unless it is explicitly requested with allow_test=True"
                ),
                "smoke_is_non_evidentiary": bool(args.smoke),
            },
            "reproducibility_limit": (
                "The fresh-spawn probe is bitwise exact on this server run. CARLA physics "
                "is not claimed bitwise portable across simulator builds, GPUs, or hosts."
            ),
        }
        _atomic_write_json(staging / "manifest.json", manifest)
        _publish_staging(staging, output)
        print(
            f"complete: {output} records={collection_summary['candidate_count']} "
            f"collisions={collection_summary['collision_label_count']} "
            f"records_sha256={file_metadata['records.npz']['sha256']}",
            flush=True,
        )
        return 0
    except BaseException:
        if cleanup_report is None and session.world is not None:
            try:
                session.cleanup()
            except Exception as cleanup_exc:
                print(f"cleanup also failed: {cleanup_exc}", file=sys.stderr, flush=True)
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
