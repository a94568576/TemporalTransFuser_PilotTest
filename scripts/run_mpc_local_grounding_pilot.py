#!/usr/bin/env python3
"""Train and evaluate the sealed MPC-local grounding mechanism pilot.

The runner deliberately separates validation selection from the one-shot test
opening.  A normal run reads only train/validation records, writes frozen
checkpoints and, if all validation gates pass, emits a selection manifest.  A
second run can read test records only when both ``--open-test`` and that frozen
manifest are supplied.  Test values are never used for fitting, early stopping,
variant selection, remediation, normalization, or threshold selection.

This V1 mechanism pilot predicts four low-dimensional physical outcomes.  It
does not claim RGB/video-world-model or closed-loop driving performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn

from temporal_tf.mpc_local_grounding import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    GroundedOutcomeModel,
    MPCGroundingRecords,
    RankPairs,
    ZScoreStats,
    build_within_state_pairs,
    fit_train_zscore_stats,
    physical_cost,
    state_group_indices,
    state_macro_metrics,
    tie_aware_logistic_rank_loss,
)


VARIANTS = (
    "prediction_only",
    "global_rank",
    "elite_rank",
    "elite_rank_inverse",
)
BASELINES = ("prediction_only", "global_rank")
PROPOSED_VARIANTS = ("elite_rank", "elite_rank_inverse")
QUALIFYING_VARIANT = "elite_rank"
RECORD_ARRAYS = (
    "state_id",
    "split_code",
    "cem_iteration",
    "action_params",
    "initial_features",
    "outcome_features",
    "collision",
    "real_cost",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _experiment_name(config: Mapping[str, Any] | None) -> str:
    revision = 1
    if isinstance(config, Mapping):
        remediation = config.get("remediation")
        if isinstance(remediation, Mapping):
            revision = int(remediation.get("revision", 1))
    if revision < 1:
        raise ValueError("remediation revision must be positive")
    suffix = "_rankscale_diagnostic" if _diagnostic_only_no_test(config) else ""
    return f"mpc_local_grounding_pilot_v{revision}{suffix}"


def _diagnostic_only_no_test(config: Mapping[str, Any] | None) -> bool:
    if not isinstance(config, Mapping):
        return False
    evaluation = config.get("evaluation")
    return bool(
        isinstance(evaluation, Mapping)
        and evaluation.get("diagnostic_only_no_test") is True
    )


def _rank_logit_temperature(config: Mapping[str, Any]) -> float:
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("config training section is missing")
    try:
        value = float(training.get("rank_logit_temperature", 1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "training.rank_logit_temperature must be finite and positive"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("training.rank_logit_temperature must be finite and positive")
    return value


def _selection_manifest_allowed(config: Mapping[str, Any]) -> bool:
    return not _diagnostic_only_no_test(config)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _load_mapping(path: Path, *, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{kind} must contain a mapping")
    return value


def _load_records_restricted(
    path: Path,
    *,
    split_codes: Sequence[int],
    allow_test: bool,
) -> MPCGroundingRecords:
    """Slice allowed rows before any semantic validation or modeling.

    NPZ compression requires NumPy to decode an array before boolean slicing,
    but sealed rows are never returned, aggregated, normalized, validated as
    outcomes, or passed to a model.  Test selection is impossible unless both
    code 2 and ``allow_test=True`` are explicit here.
    """

    requested = tuple(dict.fromkeys(int(value) for value in split_codes))
    if not requested:
        raise ValueError("at least one split code is required")
    if any(value not in (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT) for value in requested):
        raise ValueError("invalid split code")
    if TEST_SPLIT in requested and not allow_test:
        raise PermissionError("test split access requires allow_test=True")
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        missing = sorted(set(RECORD_ARRAYS) - keys)
        extra = sorted(keys - set(RECORD_ARRAYS))
        if missing:
            raise ValueError(f"records NPZ is missing arrays: {missing}")
        if extra:
            raise ValueError(f"records NPZ has unexpected arrays: {extra}")
        split = np.asarray(archive["split_code"])
        if split.ndim != 1 or split.dtype.kind not in "iu":
            raise ValueError("split_code must be a one-dimensional integer array")
        selected = np.isin(split, requested)
        if not np.any(selected):
            raise ValueError(f"requested split codes are absent: {requested}")
        arrays = {
            name: np.asarray(archive[name])[selected]
            for name in RECORD_ARRAYS
        }
    return MPCGroundingRecords(**arrays)


def _top_level_alias(value: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    """Resolve aliases only in the current manifest namespace.

    Artifact identities must never be discovered recursively: remediation-parent
    provenance deliberately contains its own records/config hashes, which are not
    aliases for the current collection.
    """

    found = [value[name] for name in names if name in value]
    if not found:
        return None
    canonical = {
        json.dumps(item, sort_keys=True, ensure_ascii=False, default=str)
        for item in found
    }
    if len(canonical) != 1:
        raise ValueError(f"manifest has conflicting top-level aliases: {sorted(names)}")
    return found[0]


def _require_config(config: dict[str, Any]) -> None:
    required_sections = {
        "collection",
        "cem",
        "cost",
        "model",
        "training",
        "evaluation",
        "gates",
        "remediation",
    }
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError(f"config is missing sections: {missing}")
    variants = tuple(config["training"].get("variants", ()))
    if variants != VARIANTS:
        raise ValueError(
            "training.variants must be the fixed ordered set " + repr(VARIANTS)
        )
    if int(config["cem"]["iterations"]) != 3:
        raise ValueError("V1 runner requires exactly three CEM iterations")
    if int(config["cem"]["population"]) < 2:
        raise ValueError("CEM population must be at least two")
    if int(config["training"]["pair_budget_per_state"]) < 1:
        raise ValueError("pair_budget_per_state must be positive")
    for key in ("prediction_candidates_per_state", "rank_query_candidates_per_state"):
        if int(config["training"].get(key, 0)) < 1:
            raise ValueError(f"training.{key} must be positive")
    if (
        config["training"].get("candidate_partition_order")
        != "sha256_state_action_seed"
    ):
        raise ValueError(
            "V1 requires training.candidate_partition_order=sha256_state_action_seed"
        )
    if (
        int(config["training"]["prediction_candidates_per_state"])
        + int(config["training"]["rank_query_candidates_per_state"])
        != int(config["cem"]["population"])
    ):
        raise ValueError(
            "prediction and global-rank query candidate counts must partition the population"
        )
    tie_threshold_value = float(config["cost"]["pair_tie_threshold"])
    if not math.isfinite(tie_threshold_value) or tie_threshold_value < 0.0:
        raise ValueError("pair_tie_threshold must be non-negative")
    diagnostic_value = config["evaluation"].get("diagnostic_only_no_test", False)
    if not isinstance(diagnostic_value, bool):
        raise ValueError("evaluation.diagnostic_only_no_test must be boolean")
    temperature = _rank_logit_temperature(config)
    diagnostic_only = _diagnostic_only_no_test(config)
    if diagnostic_only:
        tie_threshold = float(config["cost"]["pair_tie_threshold"])
        if not math.isclose(temperature, tie_threshold, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "diagnostic rank-logit temperature must equal cost.pair_tie_threshold"
            )
        disclosure = config.get("diagnostic_disclosure")
        if not isinstance(disclosure, Mapping):
            raise ValueError("diagnostic-only config requires diagnostic_disclosure")
        if (
            disclosure.get("posthoc_after_official_v2_no_go") is not True
            or disclosure.get("official_v2_status") != "no_go"
            or disclosure.get("single_changed_training_field")
            != "training.rank_logit_temperature"
        ):
            raise ValueError("rank-scale diagnostic disclosure is incomplete")
    elif not math.isclose(temperature, 1.0, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(
            "non-diagnostic V1/V2 runs require rank_logit_temperature=1.0"
        )
    if int(config["gates"].get("required_collision_records", 0)) != 0:
        raise ValueError(
            "the four-outcome mechanism runner requires required_collision_records=0"
        )
    seeds = config["training"].get("seeds", ())
    if len(seeds) != 3 or len(set(int(value) for value in seeds)) != 3:
        raise ValueError("non-smoke protocol requires exactly three distinct seeds")


def _split_state_ids(manifest: Mapping[str, Any], split: str) -> list[str] | None:
    candidates: list[Any] = []
    for root_name in ("splits", "split_state_ids", "state_ids"):
        root = manifest.get(root_name)
        if not isinstance(root, Mapping) or split not in root:
            continue
        value = root[split]
        if isinstance(value, Mapping):
            for key in ("state_ids", "ids", "states"):
                if key in value:
                    candidates.append(value[key])
                    break
        else:
            candidates.append(value)
    if not candidates:
        return None
    result = candidates[0]
    if not isinstance(result, list):
        raise TypeError(f"manifest {split} state IDs must be a list")
    normalized = [str(value) for value in result]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"manifest {split} state IDs contain duplicates")
    return normalized


def _records_state_ids(records: MPCGroundingRecords) -> list[str]:
    return sorted({str(value.item() if isinstance(value, np.generic) else value) for value in records.state_id})


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    data_path: Path,
    data_sha256: str,
    config: dict[str, Any],
    config_sha256: str,
    records: MPCGroundingRecords,
    requested_split: str,
    allow_smoke_data: bool,
) -> dict[str, Any]:
    """Validate provenance without inspecting any sealed test outcome."""

    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("manifest schema_version must equal 1")
    if manifest.get("dataset_schema") != "mpc-local-carla-v1":
        raise ValueError("manifest dataset_schema must equal mpc-local-carla-v1")
    if manifest.get("status") != "complete":
        raise ValueError("collection manifest status must be complete")
    declared_data_sha = _top_level_alias(
        manifest,
        ("records_sha256", "npz_sha256", "data_sha256", "dataset_sha256"),
    )
    if declared_data_sha is None:
        raise ValueError("manifest does not declare the records NPZ SHA-256")
    if str(declared_data_sha).lower() != data_sha256:
        raise ValueError("records NPZ SHA-256 differs from manifest")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("manifest files metadata is missing")
    record_file = files.get("records.npz")
    if not isinstance(record_file, Mapping) or record_file.get("sha256") != data_sha256:
        raise ValueError("nested records.npz metadata hash is missing or inconsistent")

    declared_config_sha = _top_level_alias(
        manifest,
        ("config_sha256", "collection_config_sha256", "protocol_config_sha256"),
    )
    if declared_config_sha is not None and str(declared_config_sha).lower() != config_sha256:
        raise ValueError("runner config SHA-256 differs from collection manifest")

    expected_collection = config["collection"]
    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("manifest protocol block is missing")
    expected_action_profile = expected_collection.get("action_profile")
    actual_action_profile = protocol.get(
        "action_profile", manifest.get("action_profile")
    )
    if (
        "action_profile" in protocol
        and "action_profile" in manifest
        and str(protocol["action_profile"]) != str(manifest["action_profile"])
    ):
        raise ValueError("manifest action-profile aliases conflict")
    if expected_action_profile is not None:
        if str(actual_action_profile) != str(expected_action_profile):
            raise ValueError("manifest action profile differs from config")
    elif actual_action_profile not in (None, "v1"):
        raise ValueError(
            "a non-v1 action profile must be explicitly frozen in collection.action_profile"
        )
    comparisons = (
        ("carla_version", protocol.get("carla_version"), str(expected_collection["carla_version"]), str),
        ("map", protocol.get("map"), str(expected_collection["map"]), str),
        (
            "fixed_delta_seconds",
            protocol.get("fixed_delta_seconds"),
            float(expected_collection["fixed_delta_seconds"]),
            float,
        ),
        ("horizon_ticks", protocol.get("horizon_ticks"), int(expected_collection["horizon_ticks"]), int),
    )
    environment_checks: dict[str, Any] = {}
    for name, actual, expected, caster in comparisons:
        if actual is None:
            raise ValueError(f"manifest protocol is missing {name}")
        cast_actual = caster(actual)
        if isinstance(expected, float):
            equal = math.isclose(cast_actual, expected, rel_tol=0.0, abs_tol=1e-12)
        else:
            equal = cast_actual == expected
        if not equal:
            raise ValueError(
                f"manifest {name}={cast_actual!r}, expected {expected!r}"
            )
        environment_checks[name] = cast_actual
    server = manifest.get("server_and_map")
    if not isinstance(server, Mapping):
        raise ValueError("manifest server_and_map block is missing")
    if (
        str(server.get("client_version")) != str(expected_collection["carla_version"])
        or str(server.get("server_version")) != str(expected_collection["carla_version"])
        or str(server.get("map_name")) != str(expected_collection["map"])
    ):
        raise ValueError("CARLA client/server/map identity differs from config")

    outcome_source = _top_level_alias(
        manifest, ("outcome_source", "label_source", "real_outcome_source")
    )
    if outcome_source is None or not any(
        token in str(outcome_source).lower() for token in ("carla", "simulator")
    ):
        raise ValueError("manifest must identify CARLA/simulator as real outcome source")

    reset = _top_level_alias(
        manifest,
        ("same_state_reset_passed", "reset_determinism_passed", "reset_reproducible"),
    )
    if reset is None:
        reset_block = _top_level_alias(
            manifest, ("same_state_reset", "reset_determinism")
        )
        if isinstance(reset_block, Mapping):
            reset = reset_block.get("passed", reset_block.get("status"))
    if isinstance(reset, str):
        reset_ok = reset.lower() in {"pass", "passed", "true", "ok"}
    else:
        reset_ok = bool(reset)
    if not reset_ok:
        raise ValueError("manifest does not attest reproducible same-state reset")

    if bool(protocol.get("smoke")) and not allow_smoke_data:
        raise ValueError("collector smoke data is non-evidentiary and cannot enter this runner")
    collection_protocol_checks = (
        ("vehicle_blueprint", expected_collection["vehicle_filter"]),
        ("initial_speeds_mps", expected_collection["initial_speeds_mps"]),
        ("target_speed_mps", expected_collection["target_speed_mps"]),
        (
            "minimum_forward_non_junction_road_m",
            expected_collection["minimum_forward_road_m"],
        ),
    )
    for key, expected in collection_protocol_checks:
        actual = protocol.get(key)
        if isinstance(expected, list):
            equal = np.allclose(
                np.asarray(actual, dtype=np.float64),
                np.asarray(expected, dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            )
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            equal = math.isclose(
                float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
        else:
            equal = actual == expected
        if not equal:
            raise ValueError(f"manifest protocol {key} differs from config")
    if protocol.get("longitudinal_mapping") != "positive=throttle; negative=brake":
        raise ValueError("manifest longitudinal control mapping is unsupported")
    split_counts = protocol.get("split_counts")
    if not isinstance(split_counts, Mapping) or {
        name: int(split_counts.get(name, -1)) for name in ("train", "val", "test")
    } != {
        name: int(expected_collection["states"][name])
        for name in ("train", "val", "test")
    }:
        raise ValueError("manifest split counts differ from config")
    collection_settings = server.get("collection_settings")
    if not isinstance(collection_settings, Mapping):
        raise ValueError("manifest collection settings are missing")
    if (
        collection_settings.get("synchronous_mode") is not True
        or collection_settings.get("no_rendering_mode")
        is not bool(expected_collection["no_rendering"])
        or not math.isclose(
            float(collection_settings.get("fixed_delta_seconds", math.nan)),
            float(expected_collection["fixed_delta_seconds"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("CARLA collection settings differ from config")
    expected_cem = config["cem"]
    actual_cem = protocol.get("cem")
    if not isinstance(actual_cem, Mapping):
        raise ValueError("manifest CEM protocol is missing")
    for key in ("iterations", "population", "elite_count"):
        if int(actual_cem.get(key, -1)) != int(expected_cem[key]):
            raise ValueError(f"manifest CEM {key} differs from config")
    for key in ("initial_mean", "initial_std", "lower", "upper", "minimum_std"):
        actual = np.asarray(actual_cem.get(key), dtype=np.float64)
        expected = np.asarray(expected_cem[key], dtype=np.float64)
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"manifest CEM {key} differs from config")
    if int(protocol.get("seed", -1)) != int(config["collection"]["seed"]):
        raise ValueError("manifest collection seed differs from config")

    fresh_state_provenance: dict[str, Any] | None = None
    state_selection = protocol.get("state_selection")
    if state_selection is not None:
        if not isinstance(state_selection, Mapping):
            raise ValueError("manifest state_selection protocol must be a mapping")
        excluded = [
            int(value)
            for value in state_selection.get("excluded_source_spawn_indices", ())
        ]
        if len(excluded) != len(set(excluded)):
            raise ValueError("state_selection excluded spawn indices contain duplicates")
        is_parent_remediation = bool(state_selection.get("fresh_relative_to_parent"))
        parent = state_selection.get("excluded_parent")
        if is_parent_remediation and (not excluded or not isinstance(parent, Mapping)):
            raise ValueError(
                "fresh-parent remediation requires excluded spawn indices and parent identity"
            )
        if not is_parent_remediation and (excluded or parent is not None):
            raise ValueError("state_selection fresh-parent fields are internally inconsistent")
        expects_parent_remediation = bool(
            expected_collection.get("exclude_parent_v1_spawn_states", False)
        )
        if expects_parent_remediation != is_parent_remediation:
            raise ValueError("manifest fresh-parent state policy differs from config")
        if expects_parent_remediation:
            expected_parent = {
                "states_sha256": str(expected_collection["parent_v1_states_sha256"]),
                "records_sha256": str(expected_collection["parent_v1_records_sha256"]),
            }
            if any(str(parent.get(key)) != value for key, value in expected_parent.items()):
                raise ValueError("manifest remediation-parent identity differs from config")

        attestation = manifest.get("fresh_state_attestation")
        if not isinstance(attestation, Mapping):
            raise ValueError("manifest fresh-state attestation is missing")
        attested_excluded = [
            int(value)
            for value in attestation.get("excluded_source_spawn_indices", ())
        ]
        selected = [
            int(value)
            for value in attestation.get("selected_source_spawn_indices", ())
        ]
        overlap = [int(value) for value in attestation.get("overlap", ())]
        states = manifest.get("states")
        if not isinstance(states, list):
            raise ValueError("manifest states list is missing")
        manifest_selected = [
            int(item["source_spawn_index"])
            for item in states
            if isinstance(item, Mapping) and "source_spawn_index" in item
        ]
        if len(manifest_selected) != len(states):
            raise ValueError("manifest state source-spawn identities are incomplete")
        computed_overlap = sorted(set(excluded) & set(selected))
        if (
            attestation.get("passed") is not True
            or attested_excluded != excluded
            or selected != manifest_selected
            or overlap != computed_overlap
            or computed_overlap
        ):
            raise ValueError("fresh-state exclusion attestation failed")
        fresh_state_provenance = {
            "action_profile": actual_action_profile,
            "fresh_relative_to_parent": is_parent_remediation,
            "excluded_source_spawn_indices": excluded,
            "selected_source_spawn_indices_sha256": _stable_hash(selected),
            "selected_source_spawn_count": len(selected),
            "overlap": computed_overlap,
            "parent": dict(parent) if isinstance(parent, Mapping) else None,
            "passed": True,
        }
    protocol_cost = protocol.get("cost")
    if not isinstance(protocol_cost, Mapping):
        raise ValueError("manifest cost protocol is missing")
    manifest_weights = protocol_cost.get("weights")
    if not isinstance(manifest_weights, Mapping):
        raise ValueError("manifest cost weights are missing")
    weight_mapping = {
        "progress": "progress_weight",
        "lateral_squared": "lateral_squared_weight",
        "yaw_squared": "yaw_squared_weight",
        "speed_squared": "speed_squared_weight",
        "steering_mean_squared": "steering_squared_weight",
        "longitudinal_mean_squared": "longitudinal_squared_weight",
        "collision": "collision_weight",
    }
    for manifest_key, config_key in weight_mapping.items():
        if not math.isclose(
            float(manifest_weights.get(manifest_key, math.nan)),
            float(config["cost"][config_key]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"manifest cost weight {manifest_key} differs from config")
    if not math.isclose(
        float(protocol_cost.get("pair_tie_threshold", math.nan)),
        float(config["cost"]["pair_tie_threshold"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("manifest pair tie threshold differs from config")

    summary = manifest.get("collection_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("collection summary is missing")
    probe = summary.get("reset_probe")
    if not isinstance(probe, Mapping):
        raise ValueError("same-state reset probe is missing")
    if probe.get("bitwise_equal") is not True:
        raise ValueError("same-state reset probe was not bitwise equal")
    if probe.get("fresh_actor_ids_unique") is not True:
        raise ValueError("same-state reset probe did not use fresh actors")
    if int(probe.get("repeats", 0)) < 2:
        raise ValueError("same-state reset probe has insufficient repeats")
    reset_max = summary.get("reset_observed_max")
    reset_protocol = protocol.get("reset")
    if not isinstance(reset_max, Mapping) or not isinstance(reset_protocol, Mapping):
        raise ValueError("reset tolerance evidence is missing")
    reset_tolerances = reset_protocol.get("tolerances")
    if not isinstance(reset_tolerances, Mapping):
        raise ValueError("reset tolerances are missing")
    for key, tolerance in reset_tolerances.items():
        if key not in reset_max or float(reset_max[key]) > float(tolerance):
            raise ValueError(f"observed reset error exceeds tolerance for {key}")
    cleanup = manifest.get("cleanup")
    if not isinstance(cleanup, Mapping):
        raise ValueError("collector cleanup attestation is missing")
    if (
        cleanup.get("settings_restored") is not True
        or cleanup.get("actors_remaining") != []
        or cleanup.get("errors") != []
    ):
        raise ValueError("collector cleanup attestation failed")
    control_protocol = protocol.get("control_execution")
    if not isinstance(control_protocol, Mapping):
        raise ValueError("control execution protocol is missing")
    control_tolerance = float(
        control_protocol.get("max_abs_error_tolerance", math.nan)
    )
    control_max_error = float(
        summary.get("control_execution_max_abs_error", math.nan)
    )
    if (
        manifest.get("control_execution_audit_passed") is not True
        or summary.get("control_execution_audit_passed") is not True
        or not math.isfinite(control_tolerance)
        or not math.isfinite(control_max_error)
        or control_max_error > control_tolerance
        or not math.isclose(
            float(manifest.get("control_execution_max_abs_error", math.nan)),
            control_max_error,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("control execution audit attestation failed")
    paired_tolerance = float(
        summary.get("paired_initial_state_tolerance", math.nan)
    )
    paired_reports = summary.get("paired_initial_state")
    if (
        summary.get("paired_initial_state_passed") is not True
        or not math.isfinite(paired_tolerance)
        or paired_tolerance <= 0.0
        or not isinstance(paired_reports, list)
    ):
        raise ValueError("paired post-warm-up initial-state attestation failed")
    declared_split_ids = set(_split_state_ids(manifest, requested_split) or ())
    split_paired_reports = [
        item
        for item in paired_reports
        if isinstance(item, Mapping) and str(item.get("state_id")) in declared_split_ids
    ]
    if len(split_paired_reports) != int(expected_collection["states"][requested_split]):
        raise ValueError("paired initial-state reports do not cover the requested split")
    delta_fields = (
        "max_position_delta_m",
        "max_rotation_delta_rad",
        "max_speed_delta_mps",
        "max_initial_feature_delta",
    )
    for report in split_paired_reports:
        if report.get("passed") is not True or any(
            float(report.get(key, math.inf)) > paired_tolerance for key in delta_fields
        ):
            raise ValueError(
                f"paired initial-state report failed for state {report.get('state_id')}"
            )
    velocity_error = float(
        summary.get("initial_velocity_command_max_abs_error", math.nan)
    )
    if (
        summary.get("initial_velocity_command_audit_passed") is not True
        or manifest.get("initial_velocity_command_audit_passed") is not True
        or not math.isfinite(velocity_error)
        or velocity_error > control_tolerance
    ):
        raise ValueError("initial velocity command audit attestation failed")

    actual_ids = _records_state_ids(records)
    declared_ids = _split_state_ids(manifest, requested_split)
    if declared_ids is None:
        raise ValueError(f"manifest is missing {requested_split} state IDs")
    if sorted(declared_ids) != actual_ids:
        raise ValueError(f"manifest {requested_split} state IDs differ from NPZ")

    expected_state_count = int(expected_collection["states"][requested_split])
    if len(actual_ids) != expected_state_count:
        raise ValueError(
            f"{requested_split} has {len(actual_ids)} states; expected {expected_state_count}"
        )
    population = int(config["cem"]["population"])
    iteration_count = int(config["cem"]["iterations"])
    for group in state_group_indices(
        records,
        split_codes=(requested_split,),
        allow_test=requested_split == "test",
    ):
        per_iteration = [
            int(np.sum(records.cem_iteration[group.indices] == iteration))
            for iteration in range(iteration_count)
        ]
        if per_iteration != [population] * iteration_count:
            raise ValueError(
                f"state {group.state_id!r} candidate counts {per_iteration}; "
                f"expected {[population] * iteration_count}"
            )

    return {
        "path": str(data_path.resolve()),
        "sha256": data_sha256,
        "manifest_outcome_source": str(outcome_source),
        "same_state_reset_attestation": {
            "passed": True,
            "probe_repeats": int(probe["repeats"]),
            "bitwise_equal": True,
            "fresh_actor_ids_unique": True,
            "maximum_observed_errors": dict(reset_max),
            "tolerances": dict(reset_tolerances),
        },
        "control_execution_attestation": {
            "passed": True,
            "action_parameterization": protocol.get("action_parameterization"),
            "longitudinal_mapping": protocol.get("longitudinal_mapping"),
            "horizon_ticks": int(protocol["horizon_ticks"]),
            "fixed_delta_seconds": float(protocol["fixed_delta_seconds"]),
            "maximum_applied_vs_intended_error": control_max_error,
            "tolerance": control_tolerance,
            "tick_level_diagnostics_integrity_checked_separately": True,
        },
        "paired_initial_state_attestation": {
            "passed": True,
            "tolerance": paired_tolerance,
            "states": len(split_paired_reports),
            "reports": [dict(item) for item in split_paired_reports],
            "initial_velocity_command_max_abs_error": velocity_error,
        },
        "cleanup_attested": True,
        "environment": environment_checks,
        "action_profile": actual_action_profile or "v1",
        "fresh_state_attestation": fresh_state_provenance,
        "split": requested_split,
        "state_ids": actual_ids,
        "state_count": len(actual_ids),
        "record_count": len(records),
    }


def _validate_diagnostics_artifact(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    config: Mapping[str, Any],
    allowed_split_codes: Sequence[int],
    records: MPCGroundingRecords,
) -> dict[str, Any]:
    declared_path = _top_level_alias(
        manifest, ("diagnostics_path", "diagnostics_file", "diagnostics_npz")
    )
    declared_sha = _top_level_alias(manifest, ("diagnostics_sha256",))
    if declared_path is None or declared_sha is None:
        raise ValueError(
            "manifest must declare both diagnostics_path and diagnostics_sha256"
        )
    path = Path(str(declared_path)).expanduser()
    if not path.is_absolute():
        path = (manifest_path.parent / path).resolve()
    else:
        path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"diagnostics artifact is missing: {path}")
    actual_sha = _sha256(path)
    if actual_sha != str(declared_sha).lower():
        raise ValueError("diagnostics artifact SHA-256 differs from manifest")
    files = manifest.get("files")
    diagnostics_file = files.get("diagnostics.npz") if isinstance(files, Mapping) else None
    if not isinstance(diagnostics_file, Mapping) or diagnostics_file.get("sha256") != actual_sha:
        raise ValueError("nested diagnostics.npz metadata hash is missing or inconsistent")

    required = {
        "state_id",
        "split_code",
        "action_params",
        "initial_speed_mps",
        "initial_actual_transform",
        "initial_actual_speed_mps",
        "initial_features_actual",
        "initial_velocity_command",
        "reset_diagnostics",
        "trajectory_world",
        "world_frames",
        "intended_controls",
        "applied_controls",
        "control_execution_max_abs_error",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"diagnostics artifact is missing control arrays: {missing}")
        split_code = np.asarray(archive["split_code"])
        if split_code.ndim != 1 or split_code.dtype.kind not in "iu":
            raise ValueError("diagnostics split_code must be an integer [state] array")
        state_mask = np.isin(split_code, tuple(int(value) for value in allowed_split_codes))
        if not np.any(state_mask):
            raise ValueError("diagnostics has no states for the allowed split codes")
        state_id = np.asarray(archive["state_id"])[state_mask]
        action = np.asarray(archive["action_params"])[state_mask]
        initial_speed = np.asarray(archive["initial_speed_mps"])[state_mask]
        initial_transform = np.asarray(archive["initial_actual_transform"])[state_mask]
        initial_actual_speed = np.asarray(
            archive["initial_actual_speed_mps"]
        )[state_mask]
        initial_features_actual = np.asarray(
            archive["initial_features_actual"]
        )[state_mask]
        initial_velocity_command = np.asarray(
            archive["initial_velocity_command"]
        )[state_mask]
        reset_diagnostics = np.asarray(archive["reset_diagnostics"])[state_mask]
        trajectory_world = np.asarray(archive["trajectory_world"])[state_mask]
        world_frames = np.asarray(archive["world_frames"])[state_mask]
        intended = np.asarray(archive["intended_controls"])[state_mask]
        applied = np.asarray(archive["applied_controls"])[state_mask]
        candidate_error = np.asarray(
            archive["control_execution_max_abs_error"]
        )[state_mask]
    expected_states = sum(
        int(config["collection"]["states"][name])
        for name, code in (("train", TRAIN_SPLIT), ("val", VAL_SPLIT), ("test", TEST_SPLIT))
        if code in set(int(value) for value in allowed_split_codes)
    )
    iterations = int(config["cem"]["iterations"])
    population = int(config["cem"]["population"])
    ticks = int(config["collection"]["horizon_ticks"])
    if action.shape != (expected_states, iterations, population, 4):
        raise ValueError(f"diagnostics action_params shape mismatch: {action.shape}")
    if state_id.shape != (expected_states,) or initial_speed.shape != (expected_states,):
        raise ValueError("diagnostics state identity/speed shape mismatch")
    if initial_transform.shape != (expected_states, iterations, population, 6):
        raise ValueError("diagnostics initial actual transform shape mismatch")
    if initial_actual_speed.shape != (expected_states, iterations, population):
        raise ValueError("diagnostics initial actual speed shape mismatch")
    feature_dim = records.state_dim
    if initial_features_actual.shape != (
        expected_states,
        iterations,
        population,
        feature_dim,
    ):
        raise ValueError("diagnostics actual initial-feature shape mismatch")
    if initial_velocity_command.shape != (
        expected_states,
        iterations,
        population,
        3,
    ):
        raise ValueError("diagnostics initial velocity-command shape mismatch")
    if reset_diagnostics.shape != (expected_states, iterations, population, 4):
        raise ValueError(
            f"diagnostics reset_diagnostics shape mismatch: {reset_diagnostics.shape}"
        )
    if trajectory_world.shape != (
        expected_states,
        iterations,
        population,
        ticks + 1,
        5,
    ):
        raise ValueError("diagnostics world trajectory shape mismatch")
    if world_frames.shape != (expected_states, iterations, population, 2):
        raise ValueError("diagnostics world-frame shape mismatch")
    expected_control_shape = (expected_states, iterations, population, ticks, 3)
    if intended.shape != expected_control_shape or applied.shape != expected_control_shape:
        raise ValueError(
            "diagnostics intended/applied control shape mismatch: "
            f"{intended.shape}, {applied.shape}, expected {expected_control_shape}"
        )
    if candidate_error.shape != (expected_states, iterations, population):
        raise ValueError("diagnostics candidate control-error shape mismatch")
    for name, value in (
        ("action_params", action),
        ("initial_speed_mps", initial_speed),
        ("initial_actual_transform", initial_transform),
        ("initial_actual_speed_mps", initial_actual_speed),
        ("initial_features_actual", initial_features_actual),
        ("initial_velocity_command", initial_velocity_command),
        ("reset_diagnostics", reset_diagnostics),
        ("trajectory_world", trajectory_world),
        ("world_frames", world_frames),
        ("intended_controls", intended),
        ("applied_controls", applied),
        ("control_execution_max_abs_error", candidate_error),
    ):
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError(f"diagnostics {name} must be finite numeric values")

    segments = int(config["collection"]["action_segments"])
    if segments != 2 or ticks % segments:
        raise ValueError("V1 control audit requires two equal action segments")
    expected_intended = np.empty_like(intended, dtype=np.float64)
    segment_ticks = ticks // segments
    for segment in range(segments):
        steer = action[..., 2 * segment].astype(np.float64)
        longitudinal = action[..., 2 * segment + 1].astype(np.float64)
        control = np.stack(
            (steer, np.maximum(longitudinal, 0.0), np.maximum(-longitudinal, 0.0)),
            axis=-1,
        )
        expected_intended[..., segment * segment_ticks : (segment + 1) * segment_ticks, :] = (
            control[..., None, :]
        )
    mapping_error = float(
        np.max(np.abs(intended.astype(np.float64) - expected_intended))
    )
    applied_error_per_candidate = np.max(
        np.abs(applied.astype(np.float64) - intended.astype(np.float64)), axis=(-2, -1)
    )
    candidate_error_mismatch = float(
        np.max(
            np.abs(
                candidate_error.astype(np.float64) - applied_error_per_candidate
            )
        )
    )
    applied_error = float(np.max(applied_error_per_candidate))
    tolerance = float(
        manifest["protocol"]["control_execution"]["max_abs_error_tolerance"]
    )
    if mapping_error > tolerance:
        raise ValueError("diagnostics intended controls do not match action parameters")
    if candidate_error_mismatch > 1e-7:
        raise ValueError("diagnostics per-candidate control error is inconsistent")
    if applied_error > tolerance:
        raise ValueError("diagnostics applied controls exceed execution tolerance")
    reset_names = (
        "position_m",
        "rotation_rad",
        "physics_disabled_speed_mps",
        "angular_speed_rad_s",
    )
    reset_tolerances = manifest["protocol"]["reset"]["tolerances"]
    reset_maximum = np.max(reset_diagnostics.astype(np.float64), axis=(0, 1, 2))
    for index, name in enumerate(reset_names):
        if float(reset_maximum[index]) > float(reset_tolerances[name]):
            raise ValueError(f"diagnostics reset error exceeds tolerance for {name}")
    paired_tolerance = float(
        manifest["collection_summary"]["paired_initial_state_tolerance"]
    )
    paired_maxima = {
        "position_m": 0.0,
        "rotation_rad": 0.0,
        "speed_mps": 0.0,
        "initial_features": 0.0,
    }
    records_state_ids = np.asarray([str(value) for value in records.state_id])
    for state_index, raw_state_id in enumerate(state_id):
        transforms = initial_transform[state_index].reshape(-1, 6).astype(np.float64)
        speeds = initial_actual_speed[state_index].reshape(-1).astype(np.float64)
        features = initial_features_actual[state_index].reshape(
            -1, feature_dim
        ).astype(np.float64)
        position_delta = float(
            np.max(np.linalg.norm(transforms[:, :3] - transforms[0, :3], axis=1))
        )
        rotation_delta = float(
            np.max(
                np.abs(
                    (transforms[:, 3:] - transforms[0, 3:] + math.pi)
                    % (2.0 * math.pi)
                    - math.pi
                )
            )
        )
        speed_delta = float(np.max(np.abs(speeds - speeds[0])))
        feature_delta = float(np.max(np.abs(features - features[0])))
        for key, value in (
            ("position_m", position_delta),
            ("rotation_rad", rotation_delta),
            ("speed_mps", speed_delta),
            ("initial_features", feature_delta),
        ):
            paired_maxima[key] = max(paired_maxima[key], value)
            if value > paired_tolerance:
                raise ValueError(
                    f"diagnostics paired initial-state {key} exceeds tolerance"
                )
        record_indices = np.flatnonzero(records_state_ids == str(raw_state_id))
        if len(record_indices) != iterations * population:
            raise ValueError("records/diagnostics state candidate count mismatch")
        if not np.allclose(
            records.initial_features[record_indices].astype(np.float64),
            features,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("records initial features differ from actual t=0 diagnostics")

    commanded_speed = np.linalg.norm(
        initial_velocity_command.astype(np.float64), axis=-1
    )
    command_error = float(
        np.max(np.abs(commanded_speed - initial_speed[:, None, None]))
    )
    if command_error > tolerance:
        raise ValueError("diagnostics initial velocity command magnitude mismatch")
    if not np.allclose(
        trajectory_world[..., 0, :3].astype(np.float64),
        initial_transform[..., :3].astype(np.float64),
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("trajectory t=0 position differs from initial-state audit")
    if not np.allclose(
        trajectory_world[..., 0, 3].astype(np.float64),
        initial_transform[..., 5].astype(np.float64),
        rtol=0.0,
        atol=1e-5,
    ) or not np.allclose(
        trajectory_world[..., 0, 4].astype(np.float64),
        initial_actual_speed.astype(np.float64),
        rtol=0.0,
        atol=1e-5,
    ):
        raise ValueError("trajectory t=0 yaw/speed differs from initial-state audit")
    if not np.all(world_frames[..., 1] - world_frames[..., 0] == ticks):
        raise ValueError("diagnostics action rollout frame span differs from horizon")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "content_opened_by_runner": True,
        "arrays_read": sorted(required),
        "allowed_split_codes": [int(value) for value in allowed_split_codes],
        "selected_states": expected_states,
        "expected_control_shape": list(expected_control_shape),
        "action_to_intended_max_abs_error": mapping_error,
        "applied_to_intended_max_abs_error": applied_error,
        "candidate_error_recompute_max_abs_difference": candidate_error_mismatch,
        "reset_maximum_observed": {
            name: float(reset_maximum[index])
            for index, name in enumerate(reset_names)
        },
        "paired_initial_state_maximum_deltas": paired_maxima,
        "paired_initial_state_tolerance": paired_tolerance,
        "initial_velocity_command_max_abs_error": command_error,
        "trajectory_t0_matches_initial_state": True,
        "world_frame_span_matches_horizon": True,
        "tolerance": tolerance,
        "passed": True,
        "outcome_or_cost_arrays_read": False,
    }


def _select_split(records: MPCGroundingRecords, split_code: int) -> MPCGroundingRecords:
    return records.subset(np.flatnonzero(records.split_code == int(split_code)))


def _stats_to_json(stats: ZScoreStats) -> dict[str, Any]:
    value = asdict(stats)
    for key, item in list(value.items()):
        if isinstance(item, np.ndarray):
            value[key] = item.tolist()
        elif isinstance(item, np.generic):
            value[key] = item.item()
    return value


def _stats_from_json(value: Mapping[str, Any]) -> ZScoreStats:
    return ZScoreStats(
        state_mean=np.asarray(value["state_mean"], dtype=np.float32),
        state_std=np.asarray(value["state_std"], dtype=np.float32),
        action_mean=np.asarray(value["action_mean"], dtype=np.float32),
        action_std=np.asarray(value["action_std"], dtype=np.float32),
        outcome_mean=np.asarray(value["outcome_mean"], dtype=np.float32),
        outcome_std=np.asarray(value["outcome_std"], dtype=np.float32),
        cost_mean=float(value["cost_mean"]),
        cost_std=float(value["cost_std"]),
        train_records=int(value["train_records"]),
    )


def _normalization_provenance(
    records: MPCGroundingRecords,
    stats: ZScoreStats,
    source_indices: np.ndarray,
    prediction_iteration: int,
) -> dict[str, Any]:
    identities = [
        {
            "state_id": str(records.state_id[index]),
            "cem_iteration": int(records.cem_iteration[index]),
            "action": records.action_params[index].tolist(),
        }
        for index in source_indices.tolist()
    ]
    return {
        "fit_split": "train",
        "fit_cem_iteration": int(prediction_iteration),
        "fit_records": int(len(source_indices)),
        "record_identity_sha256": _stable_hash(identities),
        "test_values_used": False,
        "statistics": _stats_to_json(stats),
    }


def _candidate_partitions(
    records: MPCGroundingRecords,
    *,
    split_code: int,
    config: Mapping[str, Any],
    allow_test: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create the fixed label-fair candidate partition within every state."""

    training = config["training"]
    prediction_iteration = int(training["prediction_records_cem_iteration"])
    elite_iteration = int(config["evaluation"]["validation_cem_iteration"])
    prediction_count = int(training["prediction_candidates_per_state"])
    query_count = int(training["rank_query_candidates_per_state"])
    population = int(config["cem"]["population"])
    partition_seed = int(config["collection"]["seed"])
    groups_by_iteration: dict[int, dict[str, np.ndarray]] = {}
    for iteration in (prediction_iteration, elite_iteration):
        groups: dict[str, np.ndarray] = {}
        for group in state_group_indices(
            records,
            split_codes=(split_code,),
            cem_iterations=(iteration,),
            min_candidates=population,
            allow_test=allow_test,
        ):
            if len(group.indices) != population:
                raise ValueError(
                    f"state {group.state_id!r} iteration {iteration} does not have "
                    f"exactly {population} candidates"
                )
            def partition_key(index: int) -> tuple[bytes, int]:
                action_bytes = np.asarray(
                    records.action_params[index], dtype="<f4"
                ).tobytes()
                prefix = f"{partition_seed}|{group.state_id}|".encode("utf-8")
                return hashlib.sha256(prefix + action_bytes).digest(), int(index)

            ordered = sorted(group.indices.tolist(), key=partition_key)
            groups[str(group.state_id)] = np.asarray(ordered, dtype=np.int64)
        groups_by_iteration[iteration] = groups
    if set(groups_by_iteration[prediction_iteration]) != set(
        groups_by_iteration[elite_iteration]
    ):
        raise ValueError("global and elite iteration state sets differ")

    prediction: list[int] = []
    global_query: list[int] = []
    elite_query: list[int] = []
    per_state: dict[str, Any] = {}
    for state in sorted(groups_by_iteration[prediction_iteration]):
        global_order = groups_by_iteration[prediction_iteration][state]
        elite_order = groups_by_iteration[elite_iteration][state]
        state_prediction = global_order[:prediction_count]
        state_global_query = global_order[
            prediction_count : prediction_count + query_count
        ]
        state_elite_query = elite_order[:query_count]
        if np.intersect1d(state_prediction, state_global_query).size:
            raise RuntimeError("prediction and global-rank query partitions overlap")
        prediction.extend(state_prediction.tolist())
        global_query.extend(state_global_query.tolist())
        elite_query.extend(state_elite_query.tolist())
        per_state[state] = {
            "prediction_count": len(state_prediction),
            "global_rank_query_count": len(state_global_query),
            "elite_rank_query_count": len(state_elite_query),
            "prediction_action_sha256": _stable_hash(
                records.action_params[state_prediction].tolist()
            ),
            "global_rank_query_action_sha256": _stable_hash(
                records.action_params[state_global_query].tolist()
            ),
            "elite_rank_query_action_sha256": _stable_hash(
                records.action_params[state_elite_query].tolist()
            ),
        }
    partitions = {
        "prediction": np.asarray(prediction, dtype=np.int64),
        "global_query": np.asarray(global_query, dtype=np.int64),
        "elite_query": np.asarray(elite_query, dtype=np.int64),
    }
    identities = {
        name: [
            {
                "state_id": str(records.state_id[index]),
                "iteration": int(records.cem_iteration[index]),
                "action": records.action_params[index].tolist(),
            }
            for index in indices.tolist()
        ]
        for name, indices in partitions.items()
    }
    provenance = {
        "order": "sha256_state_action_seed",
        "partition_seed": partition_seed,
        "prediction_iteration": prediction_iteration,
        "elite_iteration": elite_iteration,
        "prediction_candidates_per_state": prediction_count,
        "rank_query_candidates_per_state": query_count,
        "states": len(per_state),
        "prediction_global_query_disjoint": not bool(
            np.intersect1d(partitions["prediction"], partitions["global_query"]).size
        ),
        "equal_global_elite_unique_candidate_count": bool(
            len(partitions["global_query"]) == len(partitions["elite_query"])
        ),
        "partition_identity_sha256": {
            name: _stable_hash(value) for name, value in identities.items()
        },
        "per_state": per_state,
    }
    return partitions, provenance


def _validate_cost_and_collision(
    records: MPCGroundingRecords, cost_config: Mapping[str, Any]
) -> dict[str, Any]:
    collision = records.collision.astype(np.float64)
    if not np.all(np.isin(collision, (0.0, 1.0))):
        raise ValueError("collision must be binary 0/1")
    collision_count = int(np.count_nonzero(collision))
    # V1 has no collision-prediction head.  Passing real collision labels into
    # predicted cost would be target leakage, so fail closed on any collision.
    if collision_count:
        raise ValueError(
            "This mechanism revision requires collision-free records because predicted cost has no "
            "collision head; real collision labels must never be injected"
        )
    with torch.inference_mode():
        recomputed = physical_cost(
            torch.from_numpy(np.array(records.outcome_features, copy=True)),
            torch.from_numpy(np.array(records.action_params, copy=True)),
            torch.from_numpy(np.array(records.collision, copy=True)),
            cost_config,
        ).numpy()
    difference = np.abs(recomputed.astype(np.float64) - records.real_cost.astype(np.float64))
    tolerance = 1e-5 + 1e-5 * np.abs(records.real_cost.astype(np.float64))
    if np.any(difference > tolerance):
        worst = int(np.argmax(difference - tolerance))
        raise ValueError(
            "real_cost does not match the frozen config formula: "
            f"max_abs_error={difference[worst]:.9g} at local record {worst}"
        )
    return {
        "collision_records": collision_count,
        "predicted_collision_policy": "constant_zero_fail_closed_on_any_real_collision",
        "real_cost_recomputed": True,
        "maximum_absolute_recompute_error": float(difference.max(initial=0.0)),
    }


def _pair_audit(
    records: MPCGroundingRecords,
    *,
    iteration: int,
    tie_threshold: float,
    split_code: int,
    allow_test: bool,
) -> dict[str, Any]:
    per_state: list[dict[str, Any]] = []
    all_margins: list[float] = []
    total_pairs = 0
    total_non_tied = 0
    for group in state_group_indices(
        records,
        split_codes=(split_code,),
        cem_iterations=(iteration,),
        min_candidates=2,
        allow_test=allow_test,
    ):
        costs = records.real_cost[group.indices].astype(np.float64)
        margins = []
        for left in range(len(costs) - 1):
            for right in range(left + 1, len(costs)):
                margins.append(abs(float(costs[left] - costs[right])))
        margin_array = np.asarray(margins, dtype=np.float64)
        non_tied = margin_array > tie_threshold
        per_state.append(
            {
                "state_id": str(group.state_id),
                "pairs": int(len(margin_array)),
                "non_tied_pairs": int(non_tied.sum()),
                "non_tied_fraction": float(non_tied.mean()),
            }
        )
        total_pairs += int(len(margin_array))
        total_non_tied += int(non_tied.sum())
        all_margins.extend(margin_array.tolist())
    margins = np.asarray(all_margins, dtype=np.float64)
    if not per_state or margins.size == 0:
        raise ValueError("pair audit has no candidate pairs")
    return {
        "cem_iteration": int(iteration),
        "tie_threshold": float(tie_threshold),
        "states": len(per_state),
        "pairs": total_pairs,
        "non_tied_pairs": total_non_tied,
        "non_tied_fraction": float(total_non_tied / total_pairs),
        "state_macro_non_tied_fraction": float(
            np.mean([item["non_tied_fraction"] for item in per_state])
        ),
        "margin_quantiles": {
            "min": float(np.min(margins)),
            "p10": float(np.quantile(margins, 0.10)),
            "p25": float(np.quantile(margins, 0.25)),
            "median": float(np.quantile(margins, 0.50)),
            "p75": float(np.quantile(margins, 0.75)),
            "p90": float(np.quantile(margins, 0.90)),
            "max": float(np.max(margins)),
        },
        "per_state": per_state,
    }


def _pair_provenance(pairs: RankPairs, records: MPCGroundingRecords) -> dict[str, Any]:
    counts: dict[str, int] = {}
    identities = []
    for left, right, target, state in zip(
        pairs.left_indices,
        pairs.right_indices,
        pairs.targets,
        pairs.state_ids,
        strict=True,
    ):
        state_key = str(state)
        counts[state_key] = counts.get(state_key, 0) + 1
        identities.append(
            {
                "state_id": state_key,
                "left_action": records.action_params[int(left)].tolist(),
                "right_action": records.action_params[int(right)].tolist(),
                "target": float(target),
            }
        )
    count_values = sorted(counts.values())
    return {
        "pairs": len(pairs),
        "states": len(counts),
        "pairs_per_state_min": count_values[0],
        "pairs_per_state_max": count_values[-1],
        "state_pair_counts": dict(sorted(counts.items())),
        "pair_identity_sha256": _stable_hash(identities),
        "ties": int(np.sum(pairs.targets == 0.5)),
        "non_ties": int(np.sum(pairs.targets != 0.5)),
    }


def _make_pair_sets(
    records: MPCGroundingRecords,
    *,
    partitions: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    split_code: int,
    allow_test: bool,
) -> tuple[dict[str, RankPairs], dict[str, Any]]:
    training = config["training"]
    budget = int(training["pair_budget_per_state"])
    pair_seed = int(config["collection"]["seed"])
    threshold = float(config["cost"]["pair_tie_threshold"])
    def query_pairs(iteration: int) -> RankPairs:
        partition_name = "global_query" if iteration == int(
            training["prediction_records_cem_iteration"]
        ) else "elite_query"
        query_indices = np.asarray(partitions[partition_name], dtype=np.int64)
        query_records = records.subset(query_indices)
        local_pairs = build_within_state_pairs(
            query_records,
            budget_per_state=budget,
            seed=pair_seed,
            split_code=split_code,
            cem_iterations=(iteration,),
            tie_tolerance=threshold,
            allow_test=allow_test,
        )
        # Convert query-subset-local indices back into the full split record
        # coordinates used by the training tensors.  Ties remain in the fixed
        # 64-pair budget with a 0.5 target; the loss encourages equality and
        # never invents a better/worse label.
        return RankPairs(
            left_indices=query_indices[local_pairs.left_indices],
            right_indices=query_indices[local_pairs.right_indices],
            targets=local_pairs.targets,
            state_ids=local_pairs.state_ids,
        )

    result = {
        "global": query_pairs(int(training["prediction_records_cem_iteration"])),
        "elite": query_pairs(
            int(config["evaluation"]["validation_cem_iteration"])
        ),
    }
    provenance = {name: _pair_provenance(pairs, records) for name, pairs in result.items()}
    for name, item in provenance.items():
        if item["pairs_per_state_min"] != budget or item["pairs_per_state_max"] != budget:
            raise ValueError(
                f"{name} pair pool cannot supply the fixed {budget}/state budget"
            )
    if provenance["global"]["pairs"] != provenance["elite"]["pairs"]:
        raise RuntimeError("global and elite pair label budgets differ")
    provenance["pair_seed"] = pair_seed
    provenance["budget_per_state"] = budget
    provenance["same_pairs_for_all_training_seeds"] = True
    return result, provenance


def _model_signature(model: nn.Module) -> dict[str, Any]:
    shapes = {name: list(parameter.shape) for name, parameter in model.named_parameters()}
    return {
        "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "parameter_shapes": shapes,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "sha256": _stable_hash(shapes),
    }


def _tensor_records(
    records: MPCGroundingRecords,
    stats: ZScoreStats,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    initial = torch.from_numpy(np.array(records.initial_features, copy=True)).to(device)
    action = torch.from_numpy(np.array(records.action_params, copy=True)).to(device)
    outcome = torch.from_numpy(np.array(records.outcome_features, copy=True)).to(device)
    state_mean = torch.tensor(stats.state_mean.tolist(), device=device)
    state_std = torch.tensor(stats.state_std.tolist(), device=device)
    action_mean = torch.tensor(stats.action_mean.tolist(), device=device)
    action_std = torch.tensor(stats.action_std.tolist(), device=device)
    outcome_mean = torch.tensor(stats.outcome_mean.tolist(), device=device)
    outcome_std = torch.tensor(stats.outcome_std.tolist(), device=device)
    return {
        "initial": ((initial - state_mean) / state_std).float(),
        "action": ((action - action_mean) / action_std).float(),
        "raw_action": action.float(),
        "outcome": ((outcome - outcome_mean) / outcome_std).float(),
        "raw_outcome": outcome.float(),
        "collision": torch.from_numpy(np.array(records.collision, copy=True)).to(device).float(),
        "real_cost": torch.from_numpy(np.array(records.real_cost, copy=True)).to(device).float(),
        "cem_iteration": torch.from_numpy(np.array(records.cem_iteration, copy=True)).to(device).long(),
    }


def _forward_all(
    model: GroundedOutcomeModel,
    tensors: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {
        "outcome": [],
        "latent_delta": [],
        "inverse_action": [],
    }
    size = int(tensors["initial"].shape[0])
    for start in range(0, size, batch_size):
        end = min(size, start + batch_size)
        value = model(tensors["initial"][start:end], tensors["action"][start:end])
        for key in chunks:
            chunks[key].append(value[key])
    return {key: torch.cat(value, dim=0) for key, value in chunks.items()}


def _denormalize_outcome(value: torch.Tensor, stats: ZScoreStats) -> torch.Tensor:
    mean = torch.tensor(
        stats.outcome_mean.tolist(), dtype=value.dtype, device=value.device
    )
    std = torch.tensor(
        stats.outcome_std.tolist(), dtype=value.dtype, device=value.device
    )
    return value * std + mean


def _predicted_cost(
    normalized_outcome: torch.Tensor,
    raw_action: torch.Tensor,
    stats: ZScoreStats,
    cost_config: Mapping[str, Any],
) -> torch.Tensor:
    outcome = _denormalize_outcome(normalized_outcome, stats)
    # Never substitute real collision labels.  Dataset validation has already
    # failed closed unless every collision label is zero.
    collision = torch.zeros(
        outcome.shape[:-1], dtype=outcome.dtype, device=outcome.device
    )
    return physical_cost(outcome, raw_action, collision, cost_config)


def _rank_loss(
    predicted_cost: torch.Tensor,
    real_cost: torch.Tensor,
    pairs: RankPairs,
    tie_threshold: float,
    temperature: float,
) -> torch.Tensor:
    left = torch.tensor(
        pairs.left_indices.tolist(), dtype=torch.long, device=predicted_cost.device
    )
    right = torch.tensor(
        pairs.right_indices.tolist(), dtype=torch.long, device=predicted_cost.device
    )
    return tie_aware_logistic_rank_loss(
        predicted_cost[left],
        predicted_cost[right],
        real_cost[left],
        real_cost[right],
        tie_threshold,
        temperature=temperature,
    )


def _inverse_loss(
    predicted_inverse: torch.Tensor,
    normalized_action: torch.Tensor,
    pairs: RankPairs,
) -> torch.Tensor:
    endpoints = np.unique(np.concatenate((pairs.left_indices, pairs.right_indices)))
    selected = torch.tensor(
        endpoints.tolist(), dtype=torch.long, device=predicted_inverse.device
    )
    return torch.nn.functional.mse_loss(
        predicted_inverse[selected], normalized_action[selected]
    )


def _objective(
    model: GroundedOutcomeModel,
    tensors: Mapping[str, torch.Tensor],
    *,
    stats: ZScoreStats,
    cost_config: Mapping[str, Any],
    prediction_indices: np.ndarray,
    rank_pairs: RankPairs | None,
    inverse_pairs: RankPairs | None,
    rank_weight: float,
    inverse_weight: float,
    tie_threshold: float,
    rank_temperature: float,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = _forward_all(model, tensors, batch_size=batch_size)
    prediction_index = torch.tensor(
        np.asarray(prediction_indices, dtype=np.int64).tolist(),
        dtype=torch.long,
        device=tensors["outcome"].device,
    )
    prediction_loss = torch.nn.functional.mse_loss(
        output["outcome"][prediction_index], tensors["outcome"][prediction_index]
    )
    predicted_cost = _predicted_cost(
        output["outcome"], tensors["raw_action"], stats, cost_config
    )
    if rank_pairs is None:
        rank_loss = prediction_loss.new_zeros(())
    else:
        rank_loss = _rank_loss(
            predicted_cost,
            tensors["real_cost"],
            rank_pairs,
            tie_threshold,
            rank_temperature,
        )
    if inverse_pairs is None:
        inverse_loss = prediction_loss.new_zeros(())
    else:
        inverse_loss = _inverse_loss(
            output["inverse_action"], tensors["action"], inverse_pairs
        )
    total = prediction_loss + rank_weight * rank_loss + inverse_weight * inverse_loss
    components = {
        "total": float(total.detach()),
        "prediction_mse_normalized": float(prediction_loss.detach()),
        "rank_logistic": float(rank_loss.detach()),
        "rank_logit_temperature": float(rank_temperature),
        "inverse_action_mse_normalized": float(inverse_loss.detach()),
    }
    return total, components


@torch.inference_mode()
def _common_validation_objective(
    model: GroundedOutcomeModel,
    tensors: Mapping[str, torch.Tensor],
    *,
    stats: ZScoreStats,
    cost_config: Mapping[str, Any],
    prediction_indices: np.ndarray,
    elite_pairs: RankPairs,
    rank_weight: float,
    tie_threshold: float,
    rank_temperature: float,
    batch_size: int,
) -> tuple[float, dict[str, float]]:
    # Common across all variants: global outcome fit plus late-CEM ordering.
    # The inverse auxiliary is deliberately excluded so early-stopping values
    # remain comparable across architectures that are otherwise identical.
    total, components = _objective(
        model,
        tensors,
        stats=stats,
        cost_config=cost_config,
        prediction_indices=prediction_indices,
        rank_pairs=elite_pairs,
        inverse_pairs=None,
        rank_weight=rank_weight,
        inverse_weight=0.0,
        tie_threshold=tie_threshold,
        rank_temperature=rank_temperature,
        batch_size=batch_size,
    )
    return float(total), components


def _train_variant(
    *,
    variant: str,
    seed: int,
    train_records: MPCGroundingRecords,
    val_records: MPCGroundingRecords,
    stats: ZScoreStats,
    train_partitions: Mapping[str, np.ndarray],
    val_partitions: Mapping[str, np.ndarray],
    train_pairs: Mapping[str, RankPairs],
    val_pairs: Mapping[str, RankPairs],
    config: Mapping[str, Any],
    device: torch.device,
    epochs: int,
    patience: int,
) -> tuple[GroundedOutcomeModel, dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported variant: {variant}")
    _seed_everything(seed)
    model_cfg = config["model"]
    model = GroundedOutcomeModel(
        initial_dim=train_records.state_dim,
        action_dim=4,
        outcome_dim=4,
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
    ).to(device)
    signature = _model_signature(model)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    train_tensors = _tensor_records(train_records, stats, device)
    val_tensors = _tensor_records(val_records, stats, device)
    pair_scope = {
        "prediction_only": None,
        "global_rank": "global",
        "elite_rank": "elite",
        "elite_rank_inverse": "elite",
    }[variant]
    rank_pairs = None if pair_scope is None else train_pairs[pair_scope]
    inverse_pairs = train_pairs["elite"] if variant == "elite_rank_inverse" else None
    rank_weight = float(training["rank_weight"])
    rank_temperature = _rank_logit_temperature(config)
    inverse_weight = float(training["inverse_weight"])
    threshold = float(config["cost"]["pair_tie_threshold"])
    batch_size = int(training["batch_size"])
    clip_norm = float(training["gradient_clip_norm"])

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_validation = math.inf
    stale = 0
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total, train_components = _objective(
            model,
            train_tensors,
            stats=stats,
            cost_config=config["cost"],
            prediction_indices=train_partitions["prediction"],
            rank_pairs=rank_pairs,
            inverse_pairs=inverse_pairs,
            rank_weight=rank_weight,
            inverse_weight=inverse_weight,
            tie_threshold=threshold,
            rank_temperature=rank_temperature,
            batch_size=batch_size,
        )
        train_total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        model.eval()
        validation, validation_components = _common_validation_objective(
            model,
            val_tensors,
            stats=stats,
            cost_config=config["cost"],
            prediction_indices=val_partitions["prediction"],
            elite_pairs=val_pairs["elite"],
            rank_weight=rank_weight,
            tie_threshold=threshold,
            rank_temperature=rank_temperature,
            batch_size=batch_size,
        )
        row = {
            "epoch": epoch + 1,
            "train": train_components,
            "validation_common_objective": validation,
            "validation": validation_components,
            "gradient_norm_before_clip": float(gradient_norm),
        }
        history.append(row)
        if validation < best_validation - 1e-10:
            best_validation = validation
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    final_train_total, final_train = _objective(
        model,
        train_tensors,
        stats=stats,
        cost_config=config["cost"],
        prediction_indices=train_partitions["prediction"],
        rank_pairs=rank_pairs,
        inverse_pairs=inverse_pairs,
        rank_weight=rank_weight,
        inverse_weight=inverse_weight,
        tie_threshold=threshold,
        rank_temperature=rank_temperature,
        batch_size=batch_size,
    )
    del final_train_total
    final_validation_value, final_validation = _common_validation_objective(
        model,
        val_tensors,
        stats=stats,
        cost_config=config["cost"],
        prediction_indices=val_partitions["prediction"],
        elite_pairs=val_pairs["elite"],
        rank_weight=rank_weight,
        tie_threshold=threshold,
        rank_temperature=rank_temperature,
        batch_size=batch_size,
    )
    return model, {
        "variant": variant,
        "seed": int(seed),
        "architecture": signature,
        "training_pair_scope": pair_scope,
        "rank_logit_temperature": rank_temperature,
        "rank_logit_temperature_scope": (
            "single_global_value_for_global_elite_and_common_validation"
        ),
        "inverse_loss_enabled": variant == "elite_rank_inverse",
        "epochs_requested": int(epochs),
        "epochs_completed": len(history),
        "early_stopped": len(history) < int(epochs),
        "best_epoch": best_epoch,
        "best_validation_common_objective": best_validation,
        "final_train_at_best": final_train,
        "final_validation_at_best": final_validation,
        "final_validation_common_objective": final_validation_value,
        "gradient_clipping": {
            "configured_max_norm": clip_norm,
            "norm_type": 2.0,
            "applied_after_backward_before_optimizer_step": True,
            "history_field": "gradient_norm_before_clip",
            "epochs_recorded": len(history),
            "epochs_exceeding_max_norm": int(
                sum(
                    float(row["gradient_norm_before_clip"]) > clip_norm
                    for row in history
                )
            ),
            "maximum_gradient_norm_before_clip": float(
                max(float(row["gradient_norm_before_clip"]) for row in history)
            ),
        },
        "history": history,
    }


def _bootstrap_ci(values: Sequence[float], *, samples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("bootstrap requires at least one value")
    if samples < 1:
        return [float(array.mean()), float(array.mean())]
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(array), size=(int(samples), len(array)))
    means = array[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _non_tied_order_accuracy(
    predicted: np.ndarray, real: np.ndarray, threshold: float
) -> tuple[float, int]:
    scores: list[float] = []
    for left in range(len(real) - 1):
        for right in range(left + 1, len(real)):
            real_delta = float(real[left] - real[right])
            if abs(real_delta) <= threshold:
                continue
            predicted_delta = float(predicted[left] - predicted[right])
            if abs(predicted_delta) <= 1e-12:
                scores.append(0.5)
            else:
                scores.append(float(np.sign(real_delta) == np.sign(predicted_delta)))
    if not scores:
        return 0.5, 0
    return float(np.mean(scores)), len(scores)


def _average_ranks(values: np.ndarray, tolerance: float) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        anchor = float(values[order[start]])
        while end < len(order) and abs(float(values[order[end]]) - anchor) <= tolerance:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(
    predicted: np.ndarray, real: np.ndarray, tolerance: float
) -> float | None:
    left = _average_ranks(predicted, tolerance)
    right = _average_ranks(real, tolerance)
    left -= left.mean()
    right -= right.mean()
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    # A state whose real costs have no rank variance provides no ordering
    # evidence.  Excluding it prevents all-tied states from receiving a false
    # perfect score and inflating a state-macro gate.
    if right_norm <= 1e-12:
        return None
    if left_norm <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


@torch.inference_mode()
def _evaluate_model(
    model: GroundedOutcomeModel,
    records: MPCGroundingRecords,
    *,
    split_code: int,
    split_name: str,
    stats: ZScoreStats,
    config: Mapping[str, Any],
    device: torch.device,
    bootstrap_seed: int,
    allow_test: bool,
) -> dict[str, Any]:
    tensors = _tensor_records(records, stats, device)
    output = _forward_all(
        model, tensors, batch_size=int(config["training"]["batch_size"])
    )
    outcome = _denormalize_outcome(output["outcome"], stats)
    predicted_cost = _predicted_cost(
        output["outcome"], tensors["raw_action"], stats, config["cost"]
    )
    outcome_numpy = outcome.detach().cpu().numpy().astype(np.float32)
    cost_numpy = predicted_cost.detach().cpu().numpy().astype(np.float32)
    threshold = float(config["cost"]["pair_tie_threshold"])
    iterations: dict[str, Any] = {}
    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    for iteration in range(int(config["cem"]["iterations"])):
        metrics = dict(
            state_macro_metrics(
                records,
                cost_numpy,
                outcome_numpy,
                split_code=split_code,
                cem_iterations=(iteration,),
                tie_tolerance=threshold,
                allow_test=allow_test,
            )
        )
        state_details: list[dict[str, Any]] = []
        for group in state_group_indices(
            records,
            split_codes=(split_code,),
            cem_iterations=(iteration,),
            min_candidates=2,
            allow_test=allow_test,
        ):
            indices = group.indices
            predicted_group = cost_numpy[indices].astype(np.float64)
            real_group = records.real_cost[indices].astype(np.float64)
            accuracy, non_tied_pairs = _non_tied_order_accuracy(
                predicted_group, real_group, threshold
            )
            selected = np.flatnonzero(
                np.abs(predicted_group - float(predicted_group.min())) <= threshold
            )
            regret = max(
                0.0,
                float(real_group[selected].mean()) - float(real_group.min()),
            )
            state_outcome_mse = float(
                np.mean(
                    (
                        outcome_numpy[indices].astype(np.float64)
                        - records.outcome_features[indices].astype(np.float64)
                    )
                    ** 2
                )
            )
            state_spearman = _spearman(predicted_group, real_group, threshold)
            state_details.append(
                {
                    "state_id": str(group.state_id),
                    "spearman": state_spearman,
                    "spearman_eligible": state_spearman is not None,
                    "non_tied_order_accuracy": accuracy,
                    "non_tied_pairs": non_tied_pairs,
                    "selection_regret": regret,
                    "outcome_mse": state_outcome_mse,
                }
            )
        eligible_spearman = [
            float(item["spearman"])
            for item in state_details
            if item["spearman"] is not None
        ]
        if not eligible_spearman:
            raise ValueError(
                f"{split_name} iteration {iteration} has no state with real-cost rank variance"
            )
        metrics["core_tie_inclusive_state_macro_spearman"] = metrics[
            "state_macro_spearman"
        ]
        metrics["state_macro_spearman"] = float(np.mean(eligible_spearman))
        metrics["spearman_eligible_states"] = len(eligible_spearman)
        metrics["spearman_excluded_all_tied_states"] = len(state_details) - len(
            eligible_spearman
        )
        metrics["selection_regret_predicted_tie_policy"] = (
            "mean_real_cost_across_candidates_within_pair_tie_threshold_of_predicted_min"
        )
        metrics["state_macro_non_tied_order_accuracy"] = float(
            np.mean([item["non_tied_order_accuracy"] for item in state_details])
        )
        metrics["non_tied_order_pairs"] = int(
            sum(item["non_tied_pairs"] for item in state_details)
        )
        metrics["bootstrap_95_ci"] = {
            "spearman": _bootstrap_ci(
                eligible_spearman,
                samples=bootstrap_samples,
                seed=bootstrap_seed + iteration * 101,
            ),
            "non_tied_order_accuracy": _bootstrap_ci(
                [item["non_tied_order_accuracy"] for item in state_details],
                samples=bootstrap_samples,
                seed=bootstrap_seed + iteration * 101 + 1,
            ),
            "selection_regret": _bootstrap_ci(
                [item["selection_regret"] for item in state_details],
                samples=bootstrap_samples,
                seed=bootstrap_seed + iteration * 101 + 2,
            ),
            "outcome_mse": _bootstrap_ci(
                [item["outcome_mse"] for item in state_details],
                samples=bootstrap_samples,
                seed=bootstrap_seed + iteration * 101 + 3,
            ),
        }
        metrics["state_details"] = state_details
        iterations[str(iteration)] = metrics
    return {
        "split": split_name,
        "test_values_used": split_code == TEST_SPLIT,
        "predicted_collision_source": "constant_zero_not_real_label",
        "iterations": iterations,
    }


def _fractional_reduction(baseline: float, method: float) -> float:
    if abs(baseline) <= 1e-12:
        return 0.0 if method <= baseline + 1e-12 else -1e30
    return (baseline - method) / abs(baseline)


def _paired_state_bootstrap_comparison(
    method_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    *,
    final_iteration: int,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    """Paired state bootstrap; diagnostic only and never used by a gate."""

    rng = np.random.default_rng(int(seed))

    def details(metrics: Mapping[str, Any], iteration: int) -> dict[str, Mapping[str, Any]]:
        return {
            str(item["state_id"]): item
            for item in metrics["iterations"][str(iteration)]["state_details"]
        }

    method_final = details(method_metrics, final_iteration)
    baseline_final = details(baseline_metrics, final_iteration)
    common_final = sorted(set(method_final) & set(baseline_final))
    if not common_final:
        raise ValueError("paired bootstrap has no common final-iteration states")
    regret_method = np.asarray(
        [method_final[state]["selection_regret"] for state in common_final],
        dtype=np.float64,
    )
    regret_baseline = np.asarray(
        [baseline_final[state]["selection_regret"] for state in common_final],
        dtype=np.float64,
    )
    rank_states = [
        state
        for state in common_final
        if method_final[state]["spearman"] is not None
        and baseline_final[state]["spearman"] is not None
    ]
    if not rank_states:
        raise ValueError("paired bootstrap has no common Spearman-eligible states")
    rank_method = np.asarray(
        [method_final[state]["spearman"] for state in rank_states], dtype=np.float64
    )
    rank_baseline = np.asarray(
        [baseline_final[state]["spearman"] for state in rank_states], dtype=np.float64
    )
    method_global = details(method_metrics, 0)
    baseline_global = details(baseline_metrics, 0)
    common_global = sorted(set(method_global) & set(baseline_global))
    mse_method = np.asarray(
        [method_global[state]["outcome_mse"] for state in common_global],
        dtype=np.float64,
    )
    mse_baseline = np.asarray(
        [baseline_global[state]["outcome_mse"] for state in common_global],
        dtype=np.float64,
    )

    sample_count = max(1, int(samples))
    rank_indices = rng.integers(0, len(rank_states), size=(sample_count, len(rank_states)))
    final_indices = rng.integers(
        0, len(common_final), size=(sample_count, len(common_final))
    )
    global_indices = rng.integers(
        0, len(common_global), size=(sample_count, len(common_global))
    )
    rank_gain = (rank_method[rank_indices] - rank_baseline[rank_indices]).mean(axis=1)
    regret_base_mean = regret_baseline[final_indices].mean(axis=1)
    regret_method_mean = regret_method[final_indices].mean(axis=1)
    regret_reduction = np.full_like(regret_base_mean, -1e30)
    nonzero = np.abs(regret_base_mean) > 1e-12
    regret_reduction[nonzero] = (
        regret_base_mean[nonzero] - regret_method_mean[nonzero]
    ) / np.abs(regret_base_mean[nonzero])
    zero_success = (~nonzero) & (
        regret_method_mean <= regret_base_mean + 1e-12
    )
    regret_reduction[zero_success] = 0.0
    mse_base_mean = mse_baseline[global_indices].mean(axis=1)
    mse_method_mean = mse_method[global_indices].mean(axis=1)
    mse_degradation = (mse_method_mean - mse_base_mean) / np.maximum(
        np.abs(mse_base_mean), 1e-12
    )

    def interval(value: np.ndarray) -> list[float]:
        finite = value[np.isfinite(value)]
        if finite.size == 0:
            return [-1e30, -1e30]
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    return {
        "spearman_gain_95_ci": interval(rank_gain),
        "selection_regret_reduction_fraction_95_ci": interval(regret_reduction),
        "global_outcome_mse_degradation_fraction_95_ci": interval(mse_degradation),
    }


def _seed_gate(
    metrics_by_variant: Mapping[str, Mapping[str, Any]],
    *,
    method: str,
    final_iteration: int,
    gates: Mapping[str, Any],
    test: bool,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    final = str(final_iteration)
    method_final = metrics_by_variant[method]["iterations"][final]
    baseline_final = [metrics_by_variant[name]["iterations"][final] for name in BASELINES]
    method_global = metrics_by_variant[method]["iterations"]["0"]
    baseline_global = [metrics_by_variant[name]["iterations"]["0"] for name in BASELINES]
    spearman_gains = [
        float(method_final["state_macro_spearman"])
        - float(item["state_macro_spearman"])
        for item in baseline_final
    ]
    regret_reductions = [
        _fractional_reduction(
            float(item["state_macro_selection_regret"]),
            float(method_final["state_macro_selection_regret"]),
        )
        for item in baseline_final
    ]
    mse_degradations = [
        (
            float(method_global["state_macro_outcome_mse"])
            - float(item["state_macro_outcome_mse"])
        )
        / max(abs(float(item["state_macro_outcome_mse"])), 1e-12)
        for item in baseline_global
    ]
    if test:
        required_spearman = float(gates["minimum_test_elite_spearman_gain"])
        required_regret = float(gates["minimum_test_regret_reduction_fraction"])
    else:
        required_spearman = float(gates["minimum_elite_spearman_gain"])
        required_regret = float(gates["minimum_selection_regret_reduction_fraction"])
    maximum_mse = float(gates["maximum_global_outcome_mse_degradation_fraction"])
    checks = {
        "spearman_gain_vs_each_baseline": {
            name: value for name, value in zip(BASELINES, spearman_gains, strict=True)
        },
        "minimum_spearman_gain": min(spearman_gains),
        "required_minimum_spearman_gain": required_spearman,
        "regret_reduction_vs_each_baseline": {
            name: value for name, value in zip(BASELINES, regret_reductions, strict=True)
        },
        "minimum_regret_reduction_fraction": min(regret_reductions),
        "required_minimum_regret_reduction_fraction": required_regret,
        "global_outcome_mse_degradation_vs_each_baseline": {
            name: value for name, value in zip(BASELINES, mse_degradations, strict=True)
        },
        "maximum_global_outcome_mse_degradation_fraction": max(mse_degradations),
        "allowed_maximum_global_outcome_mse_degradation_fraction": maximum_mse,
        "paired_state_bootstrap_95_ci_diagnostic_only": {
            name: _paired_state_bootstrap_comparison(
                metrics_by_variant[method],
                metrics_by_variant[name],
                final_iteration=final_iteration,
                samples=bootstrap_samples,
                seed=bootstrap_seed + index * 7919,
            )
            for index, name in enumerate(BASELINES)
        },
    }
    checks["pass"] = bool(
        min(spearman_gains) >= required_spearman
        and min(regret_reductions) >= required_regret
        and max(mse_degradations) <= maximum_mse
    )
    return checks


def _validation_gates(
    seed_results: Mapping[str, Mapping[str, Any]],
    *,
    pair_audit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    final_iteration = int(config["evaluation"]["validation_cem_iteration"])
    required = int(gates["required_validation_seeds"])
    baseline_drops: dict[str, float] = {}
    for seed, variants in seed_results.items():
        baseline = variants["prediction_only"]["metrics"]
        baseline_drops[seed] = float(
            baseline["iterations"]["0"]["state_macro_spearman"]
            - baseline["iterations"][str(final_iteration)]["state_macro_spearman"]
        )
    collapse_pass_seeds = sum(
        value >= float(gates["minimum_baseline_global_to_elite_spearman_drop"])
        for value in baseline_drops.values()
    )
    problem_checks = {
        "validation_elite_non_tied_pair_fraction": float(
            pair_audit["state_macro_non_tied_fraction"]
        ),
        "required_non_tied_pair_fraction": float(
            gates["minimum_non_tied_elite_pair_fraction"]
        ),
        "baseline_global_to_elite_spearman_drop_by_seed": baseline_drops,
        "required_spearman_drop": float(
            gates["minimum_baseline_global_to_elite_spearman_drop"]
        ),
        "collapse_pass_seeds": int(collapse_pass_seeds),
        "required_seeds": required,
    }
    problem_checks["pass"] = bool(
        problem_checks["validation_elite_non_tied_pair_fraction"]
        >= problem_checks["required_non_tied_pair_fraction"]
        and collapse_pass_seeds >= required
    )

    method_checks: dict[str, Any] = {}
    for method in PROPOSED_VARIANTS:
        by_seed = {}
        for seed, variants in seed_results.items():
            by_seed[seed] = _seed_gate(
                {name: item["metrics"] for name, item in variants.items()},
                method=method,
                final_iteration=final_iteration,
                gates=gates,
                test=False,
                bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
                bootstrap_seed=int(seed) * 3571 + (0 if method == "elite_rank" else 1),
            )
        passed = sum(item["pass"] for item in by_seed.values())
        method_checks[method] = {
            "by_seed": by_seed,
            "passing_seeds": int(passed),
            "required_seeds": required,
            "pass": bool(passed >= required),
            "eligible_to_qualify_go": method == QUALIFYING_VARIANT,
        }

    # The inverse head reconstructs an action from a latent delta that directly
    # consumed that action.  It is retained as a diagnostic ablation, but this
    # is not masked inverse dynamics evidence and it cannot rescue or qualify a
    # GO decision.
    selected: str | None = (
        QUALIFYING_VARIANT if method_checks[QUALIFYING_VARIANT]["pass"] else None
    )
    return {
        "problem_existence": problem_checks,
        "methods": method_checks,
        "selection_rule": (
            "elite_rank_only; elite_rank_inverse_is_non_qualifying_diagnostic"
        ),
        "inverse_ablation_limitation": (
            "inverse head receives latent_delta derived from the same action; "
            "it is not masked inverse-dynamics evidence and cannot qualify GO"
        ),
        "selected_variant": selected,
        "validation_go": bool(problem_checks["pass"] and selected is not None),
    }


def _test_gates(
    seed_results: Mapping[str, Mapping[str, Any]],
    *,
    selected_variant: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gates = config["gates"]
    final_iteration = int(config["evaluation"]["validation_cem_iteration"])
    by_seed = {
        seed: _seed_gate(
            {name: item["metrics"] for name, item in variants.items()},
            method=selected_variant,
            final_iteration=final_iteration,
            gates=gates,
            test=True,
            bootstrap_samples=int(config["evaluation"]["bootstrap_samples"]),
            bootstrap_seed=int(seed) * 4591,
        )
        for seed, variants in seed_results.items()
    }
    passed = sum(item["pass"] for item in by_seed.values())
    required = int(gates["required_test_seeds"])
    return {
        "selected_variant": selected_variant,
        "by_seed": by_seed,
        "passing_seeds": int(passed),
        "required_seeds": required,
        "pass": bool(passed >= required),
    }


def _diagnose(
    validation_gates: Mapping[str, Any],
    seed_results: Mapping[str, Mapping[str, Any]],
    pair_audit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if validation_gates["validation_go"]:
        return []
    diagnoses: list[dict[str, Any]] = []
    problem = validation_gates["problem_existence"]
    threshold = float(config["cost"]["pair_tie_threshold"])
    if (
        pair_audit["state_macro_non_tied_fraction"]
        < problem["required_non_tied_pair_fraction"]
    ):
        diagnoses.append(
            {
                "code": "candidate_margin_degeneracy",
                "classification": "validation_only_remediable_once",
                "evidence": {
                    "non_tied_fraction": pair_audit["state_macro_non_tied_fraction"],
                    "median_margin": pair_audit["margin_quantiles"]["median"],
                    "tie_threshold": threshold,
                },
            }
        )
    if problem["collapse_pass_seeds"] < problem["required_seeds"]:
        diagnoses.append(
            {
                "code": "baseline_late_cem_collapse_not_demonstrated",
                "classification": "structural_no_go",
                "evidence": problem["baseline_global_to_elite_spearman_drop_by_seed"],
            }
        )
    for method, checks in validation_gates["methods"].items():
        if checks["pass"]:
            continue
        rank_passes = sum(
            item["minimum_spearman_gain"]
            >= item["required_minimum_spearman_gain"]
            for item in checks["by_seed"].values()
        )
        regret_passes = sum(
            item["minimum_regret_reduction_fraction"]
            >= item["required_minimum_regret_reduction_fraction"]
            for item in checks["by_seed"].values()
        )
        if rank_passes >= checks["required_seeds"] and regret_passes < checks["required_seeds"]:
            code = "ranking_improves_without_selection_regret_reduction"
        else:
            code = "elite_query_not_better_than_equal_budget_global_query"
        diagnoses.append(
            {
                "code": code,
                "variant": method,
                "classification": "structural_no_go",
                "evidence": {
                    "rank_passing_seeds": rank_passes,
                    "regret_passing_seeds": regret_passes,
                    "joint_passing_seeds": checks["passing_seeds"],
                },
            }
        )

    # Optimization underfit is only flagged when every seed of a method barely
    # improves its own training prediction loss.  It is evidence, not an
    # automatic permission to tune or alter gates.
    for method in PROPOSED_VARIANTS:
        ratios = []
        for variants in seed_results.values():
            history = variants[method]["training"]["history"]
            first = float(history[0]["train"]["prediction_mse_normalized"])
            final = float(
                variants[method]["training"]["final_train_at_best"][
                    "prediction_mse_normalized"
                ]
            )
            ratios.append(final / max(first, 1e-12))
        if ratios and min(ratios) > 0.90:
            diagnoses.append(
                {
                    "code": "optimization_underfit",
                    "variant": method,
                    "classification": "validation_only_remediable_once",
                    "evidence": {"final_to_initial_prediction_loss_ratios": ratios},
                }
            )
    if not diagnoses:
        diagnoses.append(
            {
                "code": "joint_gate_failure",
                "classification": "structural_no_go",
                "evidence": "No permitted numerical, margin, or underfit cause was established.",
            }
        )
    return diagnoses


def _checkpoint_payload(
    model: GroundedOutcomeModel,
    *,
    variant: str,
    seed: int,
    stats: ZScoreStats,
    training_result: Mapping[str, Any],
    config_sha256: str,
    data_sha256: str,
    source_sha256: Mapping[str, str],
    prediction_record_sha256: str,
    pair_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "variant": variant,
        "seed": int(seed),
        "architecture": training_result["architecture"],
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "normalization": _stats_to_json(stats),
        "training_summary": {
            key: value for key, value in training_result.items() if key != "history"
        },
        "config_sha256": config_sha256,
        "data_sha256": data_sha256,
        "source_sha256": dict(source_sha256),
        "prediction_record_sha256": prediction_record_sha256,
        "pair_provenance": dict(pair_provenance),
        "test_values_used": False,
    }


def _load_frozen_checkpoint(
    entry: Mapping[str, Any],
    *,
    expected_variant: str,
    expected_seed: int,
    expected_config_sha256: str,
    expected_data_sha256: str,
    expected_source_sha256: Mapping[str, str],
    state_dim: int,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[GroundedOutcomeModel, ZScoreStats, dict[str, Any]]:
    path = Path(str(entry["path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen checkpoint missing: {path}")
    if _sha256(path) != str(entry["sha256"]):
        raise ValueError(f"frozen checkpoint hash mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    checks = {
        "variant": expected_variant,
        "seed": int(expected_seed),
        "config_sha256": expected_config_sha256,
        "data_sha256": expected_data_sha256,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(f"checkpoint {path} {key} mismatch")
    if payload.get("source_sha256") != dict(expected_source_sha256):
        raise ValueError(
            f"checkpoint {path} was produced by different runner/core source"
        )
    model_cfg = config["model"]
    model = GroundedOutcomeModel(
        initial_dim=state_dim,
        action_dim=4,
        outcome_dim=4,
        latent_dim=int(model_cfg["latent_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
    )
    if payload["architecture"]["sha256"] != _model_signature(model)["sha256"]:
        raise ValueError(f"checkpoint architecture mismatch: {path}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, _stats_from_json(payload["normalization"]), payload


def _source_hashes(runner_path: Path) -> dict[str, str]:
    core_path = Path(sys.modules[GroundedOutcomeModel.__module__].__file__).resolve()
    return {
        "runner": _sha256(runner_path),
        "core": _sha256(core_path),
    }


def _prediction_record_hash(
    train_records: MPCGroundingRecords, indices: np.ndarray
) -> str:
    identities = [
        {
            "state_id": str(train_records.state_id[index]),
            "action": train_records.action_params[index].tolist(),
            "outcome": train_records.outcome_features[index].tolist(),
        }
        for index in indices.tolist()
    ]
    return _stable_hash(identities)


def _results_markdown(results: Mapping[str, Any]) -> str:
    mode = results.get("mode", "unknown")
    status = results.get("status", "unknown")
    diagnostic_only = results.get("diagnostic_only_no_test") is True
    lines = [
        f"# MPC-Local Grounding Pilot Results ({results.get('experiment', 'unknown')})",
        "",
        f"- Mode: `{mode}`",
        f"- Status: **{status}**",
        f"- Generated: `{results.get('completed_at', results.get('created_at'))}`",
        "",
    ]
    if diagnostic_only:
        rank_provenance = results.get("rank_loss_provenance", {})
        lines.extend(
            [
                "## Exploratory diagnostic disclosure",
                "",
                "- Evidence class: **posthoc validation-only scale diagnostic**",
                "- The official V2 decision remains **NO-GO**.",
                "- The sealed test is prohibited and no selection manifest can be generated.",
                (
                    "- Global rank-logit temperature: "
                    f"`{rank_provenance.get('temperature')}`; the same value is used "
                    "for global/elite training queries and common validation."
                ),
                "- This diagnostic cannot qualify, rescue, or replace the official result.",
                "",
            ]
        )
    if mode == "validation":
        gates = results.get("validation_gates", {})
        problem = gates.get("problem_existence", {})
        lines.extend(
            [
                (
                    "## Exploratory gate replay (non-qualifying)"
                    if diagnostic_only
                    else "## Validation decision"
                ),
                "",
                f"- Problem-existence gate: `{problem.get('pass')}`",
                f"- Selected variant: `{gates.get('selected_variant')}`",
                f"- Validation GO: `{gates.get('validation_go')}`",
                (
                    "- Elite non-tied pair fraction: "
                    f"`{problem.get('validation_elite_non_tied_pair_fraction')}` "
                    f"(required `{problem.get('required_non_tied_pair_fraction')}`)"
                ),
                "",
                (
                    "The test split remains permanently sealed for this diagnostic; "
                    "a selection manifest is never written."
                    if diagnostic_only
                    else "The test split remained sealed. A test-selection manifest is written only "
                    "for a non-smoke validation GO."
                ),
                "`elite_rank_inverse` is a diagnostic ablation and cannot qualify or rescue GO.",
                "",
            ]
        )
        if results.get("seeds"):
            final_iteration = str(
                results["config"]["evaluation"]["validation_cem_iteration"]
            )
            lines.extend(
                [
                    (
                        "## Exploratory validation metrics (state macro)"
                        if diagnostic_only
                        else "## Validation metrics (state macro)"
                    ),
                    "",
                    "| seed | variant | elite Spearman | elite non-tied order acc. | elite regret | global outcome MSE |",
                    "|---:|---|---:|---:|---:|---:|",
                ]
            )
            for seed, variants in results["seeds"].items():
                for variant, record in variants.items():
                    metrics = record["metrics"]["iterations"]
                    elite = metrics[final_iteration]
                    global_metrics = metrics["0"]
                    lines.append(
                        "| {seed} | {variant} | {spearman:.6f} | {accuracy:.6f} | "
                        "{regret:.6f} | {mse:.6f} |".format(
                            seed=seed,
                            variant=variant,
                            spearman=float(elite["state_macro_spearman"]),
                            accuracy=float(
                                elite["state_macro_non_tied_order_accuracy"]
                            ),
                            regret=float(elite["state_macro_selection_regret"]),
                            mse=float(global_metrics["state_macro_outcome_mse"]),
                        )
                    )
            lines.append("")
        diagnoses = results.get("diagnoses", [])
        if diagnoses:
            lines.extend(["## Diagnosis", ""])
            for item in diagnoses:
                lines.append(
                    f"- `{item.get('code')}` — `{item.get('classification')}`"
                )
            lines.append("")
    elif mode == "sealed_test":
        gate = results.get("test_confirmation", {})
        lines.extend(
            [
                "## One-shot sealed-test confirmation",
                "",
                f"- Frozen selected variant: `{gate.get('selected_variant')}`",
                f"- Passing seeds: `{gate.get('passing_seeds')}/{gate.get('required_seeds')}`",
                f"- Test confirmation: `{gate.get('pass')}`",
                "",
            ]
        )
        if results.get("seeds"):
            final_iteration = str(
                results["config"]["evaluation"]["validation_cem_iteration"]
            )
            lines.extend(
                [
                    "| seed | variant | elite Spearman | elite non-tied order acc. | elite regret | global outcome MSE |",
                    "|---:|---|---:|---:|---:|---:|",
                ]
            )
            for seed, variants in results["seeds"].items():
                for variant, record in variants.items():
                    metrics = record["metrics"]["iterations"]
                    elite = metrics[final_iteration]
                    lines.append(
                        "| {seed} | {variant} | {spearman:.6f} | {accuracy:.6f} | "
                        "{regret:.6f} | {mse:.6f} |".format(
                            seed=seed,
                            variant=variant,
                            spearman=float(elite["state_macro_spearman"]),
                            accuracy=float(
                                elite["state_macro_non_tied_order_accuracy"]
                            ),
                            regret=float(elite["state_macro_selection_regret"]),
                            mse=float(
                                metrics["0"]["state_macro_outcome_mse"]
                            ),
                        )
                    )
            lines.append("")
    lines.extend(
        [
            "## Scope and leakage controls",
            "",
            "- All outcomes and real costs come from paired CARLA rollouts, not model labels.",
            "- Normalization statistics use train iteration-0 prediction records only.",
            "- Every ranking variant uses the same architecture and global prediction records.",
            "- Global and elite ranking receive the same fixed pair budget per state.",
            "- The mechanism fails closed on collisions; real collision labels are never fed to predicted cost.",
            "- This is a low-dimensional mechanism pilot, not closed-loop driving evidence.",
            "",
            "Machine-readable metrics, checkpoint hashes, and provenance are in `results.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def _selection_manifest(
    *,
    results_path: Path,
    results_sha256: str,
    validation_results: Mapping[str, Any],
    checkpoint_entries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config_path: Path,
    config_sha256: str,
    data_path: Path,
    data_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    if validation_results.get("diagnostic_only_no_test") is True:
        raise PermissionError(
            "diagnostic-only results can never create a selection manifest"
        )
    selected = validation_results["validation_gates"]["selected_variant"]
    required_variants = (*BASELINES, selected)
    checkpoints = {
        seed: {variant: variants[variant] for variant in required_variants}
        for seed, variants in checkpoint_entries.items()
    }
    return {
        "schema_version": 1,
        "status": "frozen_validation_selection",
        "created_at": _utc_now(),
        "selected_variant": selected,
        "selection_rule": validation_results["validation_gates"]["selection_rule"],
        "seeds": validation_results["training_seeds"],
        "validation_results_path": str(results_path.resolve()),
        "validation_results_sha256": results_sha256,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "data_path": str(data_path.resolve()),
        "data_sha256": data_sha256,
        "collection_manifest_path": str(manifest_path.resolve()),
        "collection_manifest_sha256": manifest_sha256,
        "checkpoints": checkpoints,
        "validation_go": True,
        "test_opened": False,
        "test_metrics_used_for_selection": False,
    }


def _validate_selection_manifest(
    selection: Mapping[str, Any],
    *,
    config_sha256: str,
    data_sha256: str,
    collection_manifest_sha256: str,
) -> None:
    if int(selection.get("schema_version", -1)) != 1:
        raise ValueError("selection manifest schema_version must equal 1")
    if selection.get("status") != "frozen_validation_selection":
        raise ValueError("selection manifest is not frozen validation selection")
    if selection.get("validation_go") is not True:
        raise ValueError("selection manifest does not attest validation GO")
    for key, expected in (
        ("config_sha256", config_sha256),
        ("data_sha256", data_sha256),
        ("collection_manifest_sha256", collection_manifest_sha256),
    ):
        if selection.get(key) != expected:
            raise ValueError(f"selection manifest {key} mismatch")
    selected = selection.get("selected_variant")
    if selected != QUALIFYING_VARIANT:
        raise ValueError("selection manifest has an invalid selected variant")
    result_path = Path(str(selection["validation_results_path"])).expanduser().resolve()
    if not result_path.is_file() or _sha256(result_path) != selection.get(
        "validation_results_sha256"
    ):
        raise ValueError("frozen validation results are missing or hash-mismatched")
    frozen_results = _load_mapping(result_path, kind="frozen validation results")
    if (
        frozen_results.get("mode") != "validation"
        or frozen_results.get("status") != "validation_go_test_sealed"
        or frozen_results.get("smoke") is not False
        or frozen_results.get("validation_gates", {}).get("validation_go") is not True
        or frozen_results.get("validation_gates", {}).get("selected_variant")
        != selected
    ):
        raise ValueError("frozen validation results do not authorize sealed test")
    if frozen_results.get("training_seeds") != selection.get("seeds"):
        raise ValueError("selection seeds differ from frozen validation results")
    result_checkpoints = frozen_results.get("checkpoint_entries")
    if not isinstance(result_checkpoints, Mapping):
        raise ValueError("frozen validation results have no checkpoint entries")
    for seed in selection["seeds"]:
        seed_key = str(seed)
        for variant in (*BASELINES, selected):
            if selection["checkpoints"][seed_key][variant] != result_checkpoints[seed_key][variant]:
                raise ValueError(
                    "selection checkpoint differs from frozen validation results: "
                    f"seed={seed_key} variant={variant}"
                )


def _preflight_frozen_checkpoints(
    selection: Mapping[str, Any],
    *,
    config_sha256: str,
    data_sha256: str,
    source_sha256: Mapping[str, str],
) -> None:
    selected = str(selection["selected_variant"])
    for seed in (int(value) for value in selection["seeds"]):
        for variant in (*BASELINES, selected):
            entry = selection["checkpoints"][str(seed)][variant]
            path = Path(str(entry["path"])).expanduser().resolve()
            if not path.is_file() or _sha256(path) != entry["sha256"]:
                raise ValueError(f"frozen checkpoint missing or hash-mismatched: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for key, expected in (
                ("variant", variant),
                ("seed", seed),
                ("config_sha256", config_sha256),
                ("data_sha256", data_sha256),
                ("source_sha256", dict(source_sha256)),
            ):
                if payload.get(key) != expected:
                    raise ValueError(f"frozen checkpoint {path} {key} mismatch")


def _claim_one_shot_test_open(selection_path: Path, output: Path) -> Path:
    """Atomically consume a frozen selection's one-shot test-open privilege."""

    receipt = selection_path.with_name(f"{selection_path.name}.test_open_receipt.json")
    payload = {
        "schema_version": 1,
        "status": "test_open_claimed",
        "claimed_at": _utc_now(),
        "selection_manifest_path": str(selection_path.resolve()),
        "selection_manifest_sha256": _sha256(selection_path),
        "test_output_path": str(output.resolve()),
        "note": (
            "This receipt is created before sealed records are accessed. A failed "
            "test run still consumes the one-shot privilege because test information "
            "may have been exposed."
        ),
    }
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            receipt,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise PermissionError(
            f"sealed test was already claimed for this selection: {receipt}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Do not remove the receipt: opening was claimed, and a failure after
        # this point must not enable exploratory reuse of the same test split.
        raise
    return receipt


def _validation_run(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    data_path: Path,
    data_sha256: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    output: Path,
    device: torch.device,
    smoke: bool,
    runner_path: Path,
) -> dict[str, Any]:
    records = _load_records_restricted(
        data_path,
        split_codes=(TRAIN_SPLIT, VAL_SPLIT),
        allow_test=False,
    )
    diagnostics_provenance = _validate_diagnostics_artifact(
        manifest,
        manifest_path=manifest_path,
        config=config,
        allowed_split_codes=(TRAIN_SPLIT, VAL_SPLIT),
        records=records,
    )
    train_records = _select_split(records, TRAIN_SPLIT)
    val_records = _select_split(records, VAL_SPLIT)
    data_provenance = {
        "train": _validate_manifest(
            manifest,
            data_path=data_path,
            data_sha256=data_sha256,
            config=config,
            config_sha256=config_sha256,
            records=train_records,
            requested_split="train",
            allow_smoke_data=smoke,
        ),
        "val": _validate_manifest(
            manifest,
            data_path=data_path,
            data_sha256=data_sha256,
            config=config,
            config_sha256=config_sha256,
            records=val_records,
            requested_split="val",
            allow_smoke_data=smoke,
        ),
    }
    cost_audit = {
        "train": _validate_cost_and_collision(train_records, config["cost"]),
        "val": _validate_cost_and_collision(val_records, config["cost"]),
    }
    prediction_iteration = int(config["training"]["prediction_records_cem_iteration"])
    train_partitions, train_partition_provenance = _candidate_partitions(
        train_records,
        split_code=TRAIN_SPLIT,
        config=config,
        allow_test=False,
    )
    val_partitions, val_partition_provenance = _candidate_partitions(
        val_records,
        split_code=VAL_SPLIT,
        config=config,
        allow_test=False,
    )
    stats_source = train_records.subset(train_partitions["prediction"])
    stats = fit_train_zscore_stats(stats_source)
    normalization = _normalization_provenance(
        train_records, stats, train_partitions["prediction"], prediction_iteration
    )
    train_pairs, train_pair_provenance = _make_pair_sets(
        train_records,
        partitions=train_partitions,
        config=config,
        split_code=TRAIN_SPLIT,
        allow_test=False,
    )
    val_pairs, val_pair_provenance = _make_pair_sets(
        val_records,
        partitions=val_partitions,
        config=config,
        split_code=VAL_SPLIT,
        allow_test=False,
    )
    final_iteration = int(config["evaluation"]["validation_cem_iteration"])
    pair_audits = {
        "train_global": _pair_audit(
            train_records,
            iteration=prediction_iteration,
            tie_threshold=float(config["cost"]["pair_tie_threshold"]),
            split_code=TRAIN_SPLIT,
            allow_test=False,
        ),
        "train_elite": _pair_audit(
            train_records,
            iteration=final_iteration,
            tie_threshold=float(config["cost"]["pair_tie_threshold"]),
            split_code=TRAIN_SPLIT,
            allow_test=False,
        ),
        "val_global": _pair_audit(
            val_records,
            iteration=prediction_iteration,
            tie_threshold=float(config["cost"]["pair_tie_threshold"]),
            split_code=VAL_SPLIT,
            allow_test=False,
        ),
        "val_elite": _pair_audit(
            val_records,
            iteration=final_iteration,
            tie_threshold=float(config["cost"]["pair_tie_threshold"]),
            split_code=VAL_SPLIT,
            allow_test=False,
        ),
    }

    training_cfg = config["training"]
    seeds = [int(value) for value in training_cfg["seeds"]]
    epochs = int(training_cfg["epochs"])
    patience = int(training_cfg["patience"])
    if smoke:
        seeds = seeds[:1]
        epochs = min(epochs, 5)
        patience = min(patience, 2)

    source_sha256 = _source_hashes(runner_path)
    prediction_sha = _prediction_record_hash(
        train_records, train_partitions["prediction"]
    )
    seed_results: dict[str, Any] = {}
    checkpoint_entries: dict[str, Any] = {}
    architecture_hash: str | None = None
    parameter_count: int | None = None
    for seed in seeds:
        seed_key = str(seed)
        seed_results[seed_key] = {}
        checkpoint_entries[seed_key] = {}
        for variant in VARIANTS:
            model, training_result = _train_variant(
                variant=variant,
                seed=seed,
                train_records=train_records,
                val_records=val_records,
                stats=stats,
                train_partitions=train_partitions,
                val_partitions=val_partitions,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                config=config,
                device=device,
                epochs=epochs,
                patience=patience,
            )
            signature = training_result["architecture"]
            if architecture_hash is None:
                architecture_hash = signature["sha256"]
                parameter_count = signature["parameter_count"]
            elif signature["sha256"] != architecture_hash or signature["parameter_count"] != parameter_count:
                raise RuntimeError("variant architecture/parameter count differs")
            metrics = _evaluate_model(
                model,
                val_records,
                split_code=VAL_SPLIT,
                split_name="val",
                stats=stats,
                config=config,
                device=device,
                bootstrap_seed=seed * 1009,
                allow_test=False,
            )
            checkpoint_path = output / "checkpoints" / f"seed_{seed}" / f"{variant}.pt"
            payload = _checkpoint_payload(
                model,
                variant=variant,
                seed=seed,
                stats=stats,
                training_result=training_result,
                config_sha256=config_sha256,
                data_sha256=data_sha256,
                source_sha256=source_sha256,
                prediction_record_sha256=prediction_sha,
                pair_provenance=train_pair_provenance,
            )
            _atomic_torch_save(checkpoint_path, payload)
            checkpoint_entries[seed_key][variant] = {
                "path": str(checkpoint_path.resolve()),
                "sha256": _sha256(checkpoint_path),
            }
            seed_results[seed_key][variant] = {
                "training": training_result,
                "metrics": metrics,
                "checkpoint": checkpoint_entries[seed_key][variant],
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    gates = _validation_gates(
        seed_results,
        pair_audit=pair_audits["val_elite"],
        config=config,
    )
    diagnoses = _diagnose(
        gates, seed_results, pair_audits["val_elite"], config
    )
    diagnostic_only = _diagnostic_only_no_test(config)
    status = (
        "smoke_plumbing_only"
        if smoke
        else (
            "exploratory_diagnostic_gate_pass"
            if gates["validation_go"]
            else "exploratory_diagnostic_gate_fail"
        )
        if diagnostic_only
        else "validation_go_test_sealed"
        if gates["validation_go"]
        else "no_go"
    )
    return {
        "schema_version": 1,
        "experiment": _experiment_name(config),
        "mode": "validation",
        "status": status,
        "created_at": _utc_now(),
        "completed_at": _utc_now(),
        "smoke": smoke,
        "smoke_is_plumbing_only": smoke,
        "diagnostic_only_no_test": diagnostic_only,
        "evidence_class": (
            "exploratory_posthoc_validation_only_diagnostic"
            if diagnostic_only
            else "preregistered_mechanism_validation"
        ),
        "official_v2_decision_unchanged": bool(diagnostic_only),
        "diagnostic_disclosure": (
            dict(config["diagnostic_disclosure"]) if diagnostic_only else None
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "config": config,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "collection_manifest_path": str(manifest_path.resolve()),
        "collection_manifest_sha256": manifest_sha256,
        "data_provenance": data_provenance,
        "diagnostics_provenance": diagnostics_provenance,
        "cost_and_collision_audit": cost_audit,
        "normalization": normalization,
        "source_sha256": source_sha256,
        "architecture_sha256": architecture_hash,
        "identical_parameter_count": parameter_count,
        "identical_global_prediction_record_sha256": prediction_sha,
        "training_seeds": seeds,
        "rank_loss_provenance": {
            "logit_formula": "(pred_cost_j - pred_cost_i) / temperature",
            "temperature": _rank_logit_temperature(config),
            "temperature_config_source": (
                "training.rank_logit_temperature"
                if "rank_logit_temperature" in config["training"]
                else "backward_compatible_default_1.0"
            ),
            "tie_threshold": float(config["cost"]["pair_tie_threshold"]),
            "tie_target": 0.5,
            "rank_weight": float(config["training"]["rank_weight"]),
            "single_temperature_for_all_training_queries": True,
            "common_validation_uses_same_temperature": True,
        },
        "optimization_provenance": {
            "gradient_clipping": {
                "configured_max_norm": float(
                    config["training"]["gradient_clip_norm"]
                ),
                "norm_type": 2.0,
                "applied_after_backward_before_optimizer_step": True,
                "per_epoch_pre_clip_norm_recorded_in_training_history": True,
            }
        },
        "statistical_interpretation": {
            "bootstrap_intervals": "diagnostic_only",
            "gate_thresholds": "predeclared_operational_heuristics",
            "limitation": (
                "Eight validation states and optimizer seeds are not independent "
                "scientific replications."
            ),
        },
        "test_policy": {
            "test_opened": False,
            "test_records_selected_or_returned": False,
            "test_labels_semantically_validated": False,
            "test_labels_used": False,
            "test_metrics_emitted": False,
            "requires_explicit_open_test_and_frozen_selection_manifest": not diagnostic_only,
            "test_open_prohibited_by_config": diagnostic_only,
            "selection_manifest_generation_prohibited": diagnostic_only,
        },
        "pair_provenance": {
            "train": {
                "candidate_partition": train_partition_provenance,
                "pairs": train_pair_provenance,
            },
            "validation": {
                "candidate_partition": val_partition_provenance,
                "pairs": val_pair_provenance,
            },
        },
        "pair_audits": pair_audits,
        "seeds": seed_results,
        "validation_gates": gates,
        "diagnoses": diagnoses,
        "remediation": {
            "rounds_used": 0,
            "automatic_tuning_performed": False,
            "gates_changed": False,
            "note": (
                "Posthoc scale diagnostic only; it cannot alter the official V2 NO-GO."
                if diagnostic_only
                else "Runner diagnoses only; any allowed repair must be disclosed and rerun from scratch."
            ),
        },
        "checkpoint_entries": checkpoint_entries,
    }


def _sealed_test_run(
    *,
    config: dict[str, Any],
    config_path: Path,
    config_sha256: str,
    data_path: Path,
    data_sha256: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    manifest_sha256: str,
    selection_path: Path,
    selection: dict[str, Any],
    output: Path,
    device: torch.device,
    runner_path: Path,
) -> dict[str, Any]:
    if _diagnostic_only_no_test(config):
        raise PermissionError(
            "evaluation.diagnostic_only_no_test permanently prohibits --open-test"
        )
    _validate_selection_manifest(
        selection,
        config_sha256=config_sha256,
        data_sha256=data_sha256,
        collection_manifest_sha256=manifest_sha256,
    )
    source_sha256 = _source_hashes(runner_path)
    _preflight_frozen_checkpoints(
        selection,
        config_sha256=config_sha256,
        data_sha256=data_sha256,
        source_sha256=source_sha256,
    )
    test_open_receipt = _claim_one_shot_test_open(selection_path, output)
    # This is the only call site in the runner that is permitted to request
    # test records, and it is reached only after validating both explicit CLI
    # consent and the frozen selection artifact.
    test_records = _load_records_restricted(
        data_path,
        split_codes=(TEST_SPLIT,),
        allow_test=True,
    )
    diagnostics_provenance = _validate_diagnostics_artifact(
        manifest,
        manifest_path=manifest_path,
        config=config,
        allowed_split_codes=(TEST_SPLIT,),
        records=test_records,
    )
    data_provenance = _validate_manifest(
        manifest,
        data_path=data_path,
        data_sha256=data_sha256,
        config=config,
        config_sha256=config_sha256,
        records=test_records,
        requested_split="test",
        allow_smoke_data=False,
    )
    cost_audit = _validate_cost_and_collision(test_records, config["cost"])
    selected = str(selection["selected_variant"])
    variants = (*BASELINES, selected)
    seeds = [int(value) for value in selection["seeds"]]
    if seeds != [int(value) for value in config["training"]["seeds"]]:
        raise ValueError("selection seed list differs from frozen config")
    checkpoint_entries = selection["checkpoints"]
    seed_results: dict[str, Any] = {}
    normalization_hashes: set[str] = set()
    for seed in seeds:
        seed_key = str(seed)
        seed_results[seed_key] = {}
        for variant in variants:
            entry = checkpoint_entries[seed_key][variant]
            model, stats, payload = _load_frozen_checkpoint(
                entry,
                expected_variant=variant,
                expected_seed=seed,
                expected_config_sha256=config_sha256,
                expected_data_sha256=data_sha256,
                expected_source_sha256=source_sha256,
                state_dim=test_records.state_dim,
                config=config,
                device=device,
            )
            normalization_hashes.add(_stable_hash(_stats_to_json(stats)))
            metrics = _evaluate_model(
                model,
                test_records,
                split_code=TEST_SPLIT,
                split_name="test",
                stats=stats,
                config=config,
                device=device,
                bootstrap_seed=seed * 2017,
                allow_test=True,
            )
            seed_results[seed_key][variant] = {
                "metrics": metrics,
                "checkpoint": dict(entry),
                "checkpoint_created_before_test_open": True,
                "training_summary": payload["training_summary"],
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(normalization_hashes) != 1:
        raise RuntimeError("frozen checkpoints disagree on train-only normalization")
    gate = _test_gates(
        seed_results,
        selected_variant=selected,
        config=config,
    )
    final_iteration = int(config["evaluation"]["validation_cem_iteration"])
    pair_audits = {
        "test_global": _pair_audit(
            test_records,
            iteration=int(config["training"]["prediction_records_cem_iteration"]),
            tie_threshold=float(config["cost"]["pair_tie_threshold"]),
            split_code=TEST_SPLIT,
            allow_test=True,
        ),
        "test_elite": _pair_audit(
            test_records,
            iteration=final_iteration,
            tie_threshold=float(config["cost"]["pair_tie_threshold"]),
            split_code=TEST_SPLIT,
            allow_test=True,
        ),
    }
    return {
        "schema_version": 1,
        "experiment": _experiment_name(config),
        "mode": "sealed_test",
        "status": "go" if gate["pass"] else "no_go",
        "created_at": _utc_now(),
        "completed_at": _utc_now(),
        "smoke": False,
        "device": str(device),
        "config": config,
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "collection_manifest_path": str(manifest_path.resolve()),
        "collection_manifest_sha256": manifest_sha256,
        "selection_manifest_path": str(selection_path.resolve()),
        "selection_manifest_sha256": _sha256(selection_path),
        "test_open_receipt_path": str(test_open_receipt.resolve()),
        "test_open_receipt_sha256": _sha256(test_open_receipt),
        "data_provenance": data_provenance,
        "diagnostics_provenance": diagnostics_provenance,
        "cost_and_collision_audit": cost_audit,
        "source_sha256": source_sha256,
        "training_seeds": seeds,
        "statistical_interpretation": {
            "bootstrap_intervals": "diagnostic_only",
            "gate_thresholds": "predeclared_operational_heuristics",
            "limitation": (
                "Eight sealed test states and optimizer seeds are not independent "
                "scientific replications."
            ),
        },
        "test_policy": {
            "test_opened": True,
            "test_records_selected_or_returned": True,
            "test_labels_semantically_validated": True,
            "test_labels_used_only_for_final_confirmation_metrics": True,
            "test_metrics_emitted": True,
            "test_used_for_training": False,
            "test_used_for_early_stopping": False,
            "test_used_for_variant_selection": False,
            "frozen_checkpoints_only": True,
        },
        "normalization_sha256": next(iter(normalization_hashes)),
        "pair_audits": pair_audits,
        "seeds": seed_results,
        "test_confirmation": gate,
        "remediation": {
            "performed_after_test_open": False,
            "gates_changed": False,
        },
    }


def _failure_result(
    *,
    mode: str,
    error: BaseException,
    config_path: Path,
    data_path: Path,
    manifest_path: Path,
    open_test: bool,
    smoke: bool,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    text = str(error)
    diagnostic_only = _diagnostic_only_no_test(config)
    rank_provenance: dict[str, Any] | None = None
    if isinstance(config, Mapping):
        try:
            rank_provenance = {
                "logit_formula": "(pred_cost_j - pred_cost_i) / temperature",
                "temperature": _rank_logit_temperature(config),
                "tie_threshold": float(config["cost"]["pair_tie_threshold"]),
                "single_temperature_for_all_training_queries": True,
                "common_validation_uses_same_temperature": True,
            }
        except (KeyError, TypeError, ValueError):
            rank_provenance = None
    if "real_cost does not match" in text:
        diagnosis = "numerical_cost_scale_error"
        classification = "validation_only_remediable_once"
    elif "pair" in text.lower() or "tie" in text.lower():
        diagnosis = "candidate_margin_degeneracy_or_pair_budget_failure"
        classification = "validation_only_remediable_once"
    elif "collision" in text.lower():
        diagnosis = "collision_head_absent_fail_closed"
        classification = f"structural_no_go_for_{_experiment_name(config).rsplit('_', 1)[-1]}"
    elif "reset" in text.lower():
        diagnosis = "same_state_reset_not_reproducible"
        classification = "structural_no_go"
    else:
        diagnosis = "preflight_or_runtime_failure"
        classification = "implementation_or_data_error"
    return {
        "schema_version": 1,
        "experiment": _experiment_name(config),
        "mode": mode,
        "status": "failed_preflight" if mode == "validation" else "sealed_test_failed",
        "created_at": _utc_now(),
        "completed_at": _utc_now(),
        "smoke": smoke,
        "diagnostic_only_no_test": diagnostic_only,
        "evidence_class": (
            "exploratory_posthoc_validation_only_diagnostic"
            if diagnostic_only
            else "preflight_failure"
        ),
        "official_v2_decision_unchanged": bool(diagnostic_only),
        "rank_loss_provenance": rank_provenance,
        "open_test_requested": open_test,
        "error": {"type": type(error).__name__, "message": text},
        "diagnoses": [
            {
                "code": diagnosis,
                "classification": classification,
                "evidence": text,
            }
        ],
        "paths": {
            "config": str(config_path.resolve()),
            "data": str(data_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        },
        "automatic_tuning_performed": False,
        "gates_changed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-resistant MPC-local grounding validation or explicitly "
            "open a sealed test with frozen checkpoints."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mpc_local_grounding_pilot_v1.yaml"),
    )
    parser.add_argument("--data", type=Path, required=True, help="paired records.npz")
    parser.add_argument("--manifest", type=Path, required=True, help="collection manifest JSON")
    parser.add_argument("--output", type=Path, required=True, help="new empty output directory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--open-test",
        action="store_true",
        help="explicitly open test using a frozen validation selection manifest",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="required with --open-test; produced only by a validation GO",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="shrink epochs/seeds for plumbing only; cannot produce GO or open test",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.smoke and args.open_test:
        raise ValueError("--smoke cannot be combined with --open-test")
    if args.open_test != (args.selection_manifest is not None):
        raise ValueError(
            "--open-test and --selection-manifest must be supplied together"
        )
    runner_path = Path(__file__).resolve()
    config_path = args.config.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    _fresh_output(output)
    config: dict[str, Any] | None = None
    try:
        config = _load_mapping(config_path, kind="config")
        _require_config(config)
        if args.open_test and _diagnostic_only_no_test(config):
            raise PermissionError(
                "evaluation.diagnostic_only_no_test permanently prohibits --open-test"
            )
        torch.set_num_threads(int(config.get("torch_num_threads", 1)))
        manifest = _load_mapping(manifest_path, kind="collection manifest")
        config_sha256 = _sha256(config_path)
        data_sha256 = _sha256(data_path)
        manifest_sha256 = _sha256(manifest_path)
        device = _resolve_device(args.device)
        _atomic_text(output / "config_frozen.yaml", config_path.read_text(encoding="utf-8"))
        _atomic_text(
            output / "collection_manifest_frozen.json",
            manifest_path.read_text(encoding="utf-8"),
        )
        if args.open_test:
            selection_path = args.selection_manifest.expanduser().resolve()
            selection = _load_mapping(selection_path, kind="selection manifest")
            _atomic_text(
                output / "selection_manifest_frozen.json",
                selection_path.read_text(encoding="utf-8"),
            )
            results = _sealed_test_run(
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                data_path=data_path,
                data_sha256=data_sha256,
                manifest_path=manifest_path,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                selection_path=selection_path,
                selection=selection,
                output=output,
                device=device,
                runner_path=runner_path,
            )
        else:
            results = _validation_run(
                config=config,
                config_path=config_path,
                config_sha256=config_sha256,
                data_path=data_path,
                data_sha256=data_sha256,
                manifest_path=manifest_path,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                output=output,
                device=device,
                smoke=args.smoke,
                runner_path=runner_path,
            )
    except Exception as error:
        results = _failure_result(
            mode=(
                "validation"
                if _diagnostic_only_no_test(config)
                else "sealed_test"
                if args.open_test
                else "validation"
            ),
            error=error,
            config_path=config_path,
            data_path=data_path,
            manifest_path=manifest_path,
            open_test=args.open_test,
            smoke=args.smoke,
            config=config,
        )
        _atomic_json(output / "results.json", results)
        _atomic_text(output / "RESULTS.md", _results_markdown(results))
        _atomic_text(output / "results.sha256", _sha256(output / "results.json") + "\n")
        print(json.dumps({"status": results["status"], "error": results["error"]}))
        return 2

    results_path = output / "results.json"
    _atomic_json(results_path, results)
    results_sha256 = _sha256(results_path)
    _atomic_text(output / "results.sha256", results_sha256 + "\n")
    _atomic_text(output / "RESULTS.md", _results_markdown(results))
    if (
        not args.open_test
        and not args.smoke
        and _selection_manifest_allowed(config)
        and results["validation_gates"]["validation_go"]
    ):
        selection = _selection_manifest(
            results_path=results_path,
            results_sha256=results_sha256,
            validation_results=results,
            checkpoint_entries=results["checkpoint_entries"],
            config_path=config_path,
            config_sha256=results["config_sha256"],
            data_path=data_path,
            data_sha256=results["data_provenance"]["train"]["sha256"],
            manifest_path=manifest_path,
            manifest_sha256=results["collection_manifest_sha256"],
        )
        _atomic_json(output / "selection_manifest.json", selection)
        _atomic_text(
            output / "selection_manifest.sha256",
            _sha256(output / "selection_manifest.json") + "\n",
        )
    print(
        json.dumps(
            {
                "status": results["status"],
                "mode": results["mode"],
                "results": str(results_path),
                "sha256": results_sha256,
            },
            ensure_ascii=False,
        )
    )
    return 0 if results["status"] not in {"failed_preflight", "sealed_test_failed"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
