#!/usr/bin/env python3
"""Run the preregistered, decision-aligned MPC-local grounding V3 study.

V3 is intentionally a development-only mechanism experiment.  It reads fresh
V3 train and outer-validation rows, creates a train-only fit/checkpoint split,
freezes every checkpoint, and only then evaluates the outer gate.  The sealed
split is inaccessible: this runner has no argument or code path that loads it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import nn

from temporal_tf.mpc_local_grounding import (
    TRAIN_SPLIT,
    VAL_SPLIT,
    GroundedOutcomeModel,
    MPCGroundingRecords,
    ZScoreStats,
    decision_state_macro_metrics,
    fit_train_zscore_stats,
    physical_cost,
    state_group_indices,
    statewise_softmin_listwise_loss,
)


VARIANTS = ("prediction_only", "global_listwise", "elite_listwise")
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
EXPERIMENT = "mpc_local_grounding_pilot_v3"
EXPECTED_MAP = "/Game/Carla/Maps/Town05_Opt"
EXPECTED_ACTION_PROFILE = "support_stratified_v3"


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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
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
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return result


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"mapping expected in {path}")
    return value


def _minimal_config_for_tests() -> dict[str, Any]:
    """Return the smallest valid shape used only by unit contract tests."""

    return {
        "schema_version": 3,
        "experiment": EXPERIMENT,
        "collection": {
            "map": EXPECTED_MAP,
            "action_profile": EXPECTED_ACTION_PROFILE,
            "seed": 37031,
            "states": {"train": 32, "val": 16, "test": 12},
            "state_source": "map_generate_waypoints",
            "require_unique_base_waypoint_per_state": True,
            "sealed_test_redaction": True,
            "public_manifest_redaction": {
                "states_scope": "development_only",
                "split_state_ids": ["train", "val"],
                "require_sealed_test_states_sha256": True,
                "sealed_test_integrity_fields": [
                    "states", "records", "split_code", "states_sha256",
                    "schema_finite_passed", "reset_passed",
                    "control_execution_passed",
                    "individual_state_metadata_redacted",
                    "sealed_test_stratification_passed",
                ],
                "forbid_sealed_individual_state_ids_or_covariates": True,
                "forbid_sealed_outcome_cost_collision_aggregates": True,
                "forbid_sealed_outcome_cost_collision_stdout": True,
            },
            "outputs": {
                "development_records": "development_records.npz",
                "sealed_test_records": "test_records_sealed.npz",
                "development_diagnostics": "development_diagnostics.npz",
                "sealed_test_diagnostics": "test_diagnostics_sealed.npz",
                "manifest": "manifest.json",
            },
        },
        "cem": {"iterations": 3, "population": 24},
        "cost": {
            "progress_weight": -0.20,
            "lateral_squared_weight": 1.50,
            "yaw_squared_weight": 0.80,
            "speed_squared_weight": 0.40,
            "steering_squared_weight": 0.02,
            "longitudinal_squared_weight": 0.01,
            "collision_weight": 10.0,
            "pair_tie_threshold": 0.005,
        },
        "model": {"latent_dim": 16, "hidden_dim": 64},
        "training": {
            "seeds": [17, 29, 43],
            "variants": list(VARIANTS),
            "inner_validation_states": 8,
            "inner_validation_order": "sha256_state_identity_collection_seed",
            "max_epochs": 1200,
            "min_epochs": 300,
            "patience": 200,
            "checkpoint_min_delta": 1e-6,
            "checkpoint_metric": "inner_common_outcome_mse_plus_weighted_top1_ce",
            "checkpoint_tie_break": "earliest_epoch",
            "checkpoint_selection_split": "inner_validation",
            "batch_size": 128,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_clip_norm": 5.0,
            "listwise_objective": "top1_cross_entropy",
            "listwise_target_mode": "deterministic_real_argmin_one_hot",
            "exact_argmin_tie_break": "sha256_raw_action_bytes",
            "listwise_weight": 0.25,
            "listwise_temperature": 0.005,
            "prediction_records_cem_iteration": 1,
            "prediction_candidates_per_state": 24,
            "rank_query_candidates_per_state": 24,
            "global_query_cem_iteration": 0,
            "elite_query_cem_iteration": 2,
            "candidate_partition_order": "full_iteration_sha256_state_action_seed",
            "require_full_iteration_queries": True,
        },
        "evaluation": {
            "validation_cem_iteration": 2,
            "primary_metric": "deterministic_exact_argmin_simple_regret",
            "exact_argmin_tie_break": "sha256_raw_action_bytes",
            "epsilon_regret": 0.005,
            "epsilon_regret_definition": "max(0, exact_simple_regret - epsilon)",
            "outer_gate_policy": "evaluate_once_after_all_inner_checkpoints_are_frozen",
            "bootstrap_samples": 5000,
            "test_access": "prohibited",
        },
        "gates": {
            "required_collision_records": 0,
            "collision_gate_scope": "development_only",
            "required_development_records": 3456,
            "required_test_records": 864,
            "required_total_records_attested": 4320,
            "required_train_states": 32,
            "required_outer_validation_states": 16,
            "required_sealed_test_states": 12,
            "require_map_disjoint_from_v1_v2": True,
            "require_unique_base_waypoint_per_state": True,
            "require_stratification_balance": True,
            "require_equal_global_elite_unique_query_rollouts": True,
            "require_full_iteration_query_coverage": True,
            "required_development_split_codes": [0, 1],
            "required_sealed_test_split_codes": [2],
            "require_separate_sealed_test_records": True,
            "require_separate_sealed_test_diagnostics": True,
            "sealed_test_runner_verification": "manifest_role_path_count_sha_format_and_file_existence_only",
            "forbid_runner_open_or_hash_sealed_payload": True,
            "require_sealed_test_states_sha256": True,
            "require_sealed_test_integrity_attestation": True,
            "require_sealed_test_individual_state_metadata_redacted": True,
            "allow_sealed_test_stratification_boolean_only": True,
            "minimum_outer_state_fraction_cost_range_gt_epsilon": 0.75,
            "minimum_exact_regret_reduction_fraction_vs_global_listwise": 0.20,
            "minimum_exact_regret_reduction_fraction_vs_prediction_only": 0.20,
            "maximum_epsilon_regret_absolute_degradation_vs_global_listwise": 0.00025,
            "maximum_global_outcome_mse_degradation_fraction": 0.10,
            "required_outer_gate_seeds": 2,
        },
        "provenance": {
            "source_revision": 3,
            "parent_revision": 2,
            "official_v2_status": "no_go",
            "official_v2_results_sha256": "a" * 64,
            "parent_v1": {"map": "/Game/Carla/Maps/Town10HD_Opt"},
            "parent_v2": {"map": "/Game/Carla/Maps/Town10HD_Opt"},
            "reuse_scope": "none",
            "fresh_data_required": True,
            "fresh_outer_validation_required": True,
            "v1_v2_test_access": "prohibited",
        },
        "terminal_policy": {"posthoc_remediation_rounds": 0},
    }


def _finite_positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}") from exc
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'non-negative' if allow_zero else 'positive'}")
    return result


def _require_config(config: dict[str, Any]) -> None:
    required = {
        "collection", "cem", "cost", "model", "training", "evaluation",
        "gates", "provenance", "terminal_policy",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"config is missing sections: {missing}")
    if int(config.get("schema_version", -1)) != 3 or config.get("experiment") != EXPERIMENT:
        raise ValueError("config must identify schema_version 3 and the V3 experiment")
    collection = config["collection"]
    if collection.get("map") != EXPECTED_MAP:
        raise ValueError(f"V3 requires fresh map {EXPECTED_MAP}")
    if collection.get("action_profile") != EXPECTED_ACTION_PROFILE:
        raise ValueError(f"V3 requires action profile {EXPECTED_ACTION_PROFILE}")
    if collection.get("state_source") != "map_generate_waypoints":
        raise ValueError("V3 requires map_generate_waypoints state source")
    if collection.get("require_unique_base_waypoint_per_state") is not True:
        raise ValueError("V3 requires unique base waypoints")
    if collection.get("sealed_test_redaction") is not True:
        raise ValueError("V3 requires sealed-test aggregate redaction")
    expected_integrity_fields = [
        "states", "records", "split_code", "states_sha256",
        "schema_finite_passed", "reset_passed", "control_execution_passed",
        "individual_state_metadata_redacted", "sealed_test_stratification_passed",
    ]
    expected_redaction = {
        "states_scope": "development_only",
        "split_state_ids": ["train", "val"],
        "require_sealed_test_states_sha256": True,
        "sealed_test_integrity_fields": expected_integrity_fields,
        "forbid_sealed_individual_state_ids_or_covariates": True,
        "forbid_sealed_outcome_cost_collision_aggregates": True,
        "forbid_sealed_outcome_cost_collision_stdout": True,
    }
    if collection.get("public_manifest_redaction") != expected_redaction:
        raise ValueError("V3 public-manifest sealed-test redaction contract differs")
    outputs = collection.get("outputs")
    expected_outputs = {
        "development_records": "development_records.npz",
        "sealed_test_records": "test_records_sealed.npz",
        "development_diagnostics": "development_diagnostics.npz",
        "sealed_test_diagnostics": "test_diagnostics_sealed.npz",
        "manifest": "manifest.json",
    }
    if outputs != expected_outputs:
        raise ValueError("V3 requires the frozen byte-separated dataset outputs")
    state_counts = {key: int(collection["states"][key]) for key in ("train", "val", "test")}
    if state_counts != {"train": 32, "val": 16, "test": 12}:
        raise ValueError("V3 requires the frozen 32/16/12 state counts")
    cem = config["cem"]
    if int(cem.get("iterations", -1)) != 3 or int(cem.get("population", -1)) != 24:
        raise ValueError("V3 requires three CEM iterations and population 24")

    training = config["training"]
    if tuple(training.get("variants", ())) != VARIANTS:
        raise ValueError(f"training.variants must equal {VARIANTS!r}")
    seeds = tuple(int(value) for value in training.get("seeds", ()))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("V3 requires exactly three distinct optimizer seeds")
    if int(training.get("inner_validation_states", -1)) != 8:
        raise ValueError("V3 requires eight train-only inner-validation states")
    if training.get("inner_validation_order") != "sha256_state_identity_collection_seed":
        raise ValueError("V3 inner-validation ordering is not frozen")
    max_epochs = int(training.get("max_epochs", 0))
    min_epochs = int(training.get("min_epochs", 0))
    patience = int(training.get("patience", 0))
    if max_epochs < 500 or min_epochs < 250 or max_epochs < min_epochs:
        raise ValueError("max_epochs/min_epochs do not provide the frozen convergence window")
    if patience < 1 or patience >= max_epochs:
        raise ValueError("patience must be positive and below max_epochs")
    if training.get("checkpoint_metric") != "inner_common_outcome_mse_plus_weighted_top1_ce":
        raise ValueError("checkpoint metric must be the common inner-validation objective")
    if training.get("checkpoint_tie_break") != "earliest_epoch":
        raise ValueError("checkpoint ties must select the earliest epoch")
    selection_split = training.get("checkpoint_selection_split", "inner_validation")
    if selection_split != "inner_validation":
        raise ValueError("checkpoint selection must use inner_validation, never outer validation")
    if training.get("listwise_objective") != "top1_cross_entropy":
        raise ValueError("V3 listwise objective must be top1_cross_entropy")
    if training.get("listwise_target_mode") != "deterministic_real_argmin_one_hot":
        raise ValueError("V3 listwise target mode is not frozen")
    if training.get("exact_argmin_tie_break") != "sha256_raw_action_bytes":
        raise ValueError("V3 listwise target tie-break is not frozen")
    _finite_positive(training.get("listwise_temperature"), "listwise_temperature")
    _finite_positive(training.get("listwise_weight"), "listwise_weight")
    for key in ("learning_rate", "gradient_clip_norm", "checkpoint_min_delta"):
        _finite_positive(training.get(key), key)
    _finite_positive(training.get("weight_decay"), "weight_decay", allow_zero=True)
    if int(training.get("batch_size", 0)) < 1:
        raise ValueError("batch_size must be positive")
    fixed_iterations = {
        "prediction_records_cem_iteration": 1,
        "global_query_cem_iteration": 0,
        "elite_query_cem_iteration": 2,
    }
    for key, expected in fixed_iterations.items():
        if int(training.get(key, -1)) != expected:
            raise ValueError(f"training.{key} must equal {expected}")
    if (
        int(training.get("prediction_candidates_per_state", -1)) != 24
        or int(training.get("rank_query_candidates_per_state", -1)) != 24
        or training.get("candidate_partition_order")
        != "full_iteration_sha256_state_action_seed"
        or training.get("require_full_iteration_queries") is not True
    ):
        raise ValueError("V3 requires deterministic full-iteration 24-candidate exposure")

    evaluation = config["evaluation"]
    if evaluation.get("test_access") != "prohibited":
        raise ValueError("evaluation.test_access must remain prohibited")
    if evaluation.get("outer_gate_policy") != "evaluate_once_after_all_inner_checkpoints_are_frozen":
        raise ValueError("outer gate policy is not frozen")
    if evaluation.get("primary_metric") != "deterministic_exact_argmin_simple_regret":
        raise ValueError("V3 primary metric is not deterministic exact-argmin regret")
    if evaluation.get("exact_argmin_tie_break") != "sha256_raw_action_bytes":
        raise ValueError("V3 evaluation tie-break is not frozen")
    if evaluation.get("epsilon_regret_definition") != "max(0, exact_simple_regret - epsilon)":
        raise ValueError("epsilon-regret definition is not frozen")
    _finite_positive(evaluation.get("epsilon_regret"), "epsilon_regret", allow_zero=True)
    if int(evaluation.get("validation_cem_iteration", -1)) != 2:
        raise ValueError("outer validation must evaluate final CEM iteration")

    gates = config["gates"]
    expected_counts = {
        "required_development_records": 3456,
        "required_test_records": 864,
        "required_total_records_attested": 4320,
        "required_train_states": 32,
        "required_outer_validation_states": 16,
        "required_sealed_test_states": 12,
        "required_collision_records": 0,
    }
    for key, expected in expected_counts.items():
        if int(gates.get(key, -1)) != expected:
            raise ValueError(f"gates.{key} must equal {expected}")
    if (
        gates.get("collision_gate_scope") != "development_only"
        or gates.get("required_development_split_codes") != [0, 1]
        or gates.get("required_sealed_test_split_codes") != [2]
        or gates.get("require_separate_sealed_test_records") is not True
        or gates.get("require_separate_sealed_test_diagnostics") is not True
        or gates.get("sealed_test_runner_verification")
        != "manifest_role_path_count_sha_format_and_file_existence_only"
        or gates.get("forbid_runner_open_or_hash_sealed_payload") is not True
        or gates.get("require_sealed_test_states_sha256") is not True
        or gates.get("require_sealed_test_integrity_attestation") is not True
        or gates.get("require_sealed_test_individual_state_metadata_redacted") is not True
        or gates.get("allow_sealed_test_stratification_boolean_only") is not True
    ):
        raise ValueError("sealed test must remain byte-separated and manifest-only")

    provenance = config["provenance"]
    if (
        int(provenance.get("source_revision", -1)) != 3
        or provenance.get("reuse_scope") != "none"
        or provenance.get("fresh_data_required") is not True
        or provenance.get("fresh_outer_validation_required") is not True
        or provenance.get("v1_v2_test_access") != "prohibited"
    ):
        raise ValueError("V3 fresh-data provenance is incomplete")
    for parent in ("parent_v1", "parent_v2"):
        value = provenance.get(parent)
        if not isinstance(value, Mapping) or value.get("map") == EXPECTED_MAP:
            raise ValueError(f"{parent} must identify a map disjoint from V3")
    if int(config["terminal_policy"].get("posthoc_remediation_rounds", -1)) != 0:
        raise ValueError("V3 posthoc remediation rounds must remain zero")


def _load_development_records(path: Path) -> tuple[MPCGroundingRecords, dict[str, int]]:
    """Load a byte-separated development NPZ and reject any sealed-test row."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if keys != set(RECORD_ARRAYS):
            raise ValueError(
                f"records arrays differ from the frozen contract; missing={sorted(set(RECORD_ARRAYS)-keys)}, "
                f"extra={sorted(keys-set(RECORD_ARRAYS))}"
            )
        split = np.asarray(archive["split_code"])
        if split.ndim != 1 or split.dtype.kind not in "iu":
            raise ValueError("split_code must be an integer [N] array")
        unexpected = sorted(set(split.tolist()) - {TRAIN_SPLIT, VAL_SPLIT})
        if unexpected:
            raise ValueError(
                "development artifact contains sealed test rows or invalid split codes: "
                f"{unexpected}"
            )
        split_summary = {
            "total": int(len(split)),
            "train": int(np.sum(split == TRAIN_SPLIT)),
            "val": int(np.sum(split == VAL_SPLIT)),
            "test": 0,
        }
        arrays = {name: np.asarray(archive[name]) for name in RECORD_ARRAYS}
    return MPCGroundingRecords(**arrays), split_summary


def _manifest_split_ids(manifest: Mapping[str, Any], split: str) -> list[str]:
    root = manifest.get("split_state_ids")
    if not isinstance(root, Mapping) or not isinstance(root.get(split), list):
        raise ValueError(f"manifest split_state_ids.{split} is missing")
    result = [str(value) for value in root[split]]
    if len(result) != len(set(result)):
        raise ValueError(f"manifest {split} state IDs contain duplicates")
    return result


def _reject_sealed_metadata_exposure(manifest: Mapping[str, Any]) -> None:
    """Fail if public provenance contains individual sealed-state metadata."""

    split_ids = manifest.get("split_state_ids")
    states = manifest.get("states")
    integrity = manifest.get("sealed_test_integrity")
    expected_integrity_keys = {
        "states", "records", "split_code", "states_sha256",
        "schema_finite_passed", "reset_passed", "control_execution_passed",
        "individual_state_metadata_redacted", "sealed_test_stratification_passed",
    }
    if not isinstance(split_ids, Mapping) or set(split_ids) != {"train", "val"}:
        raise ValueError("sealed-test individual metadata is exposed in split_state_ids")
    if not isinstance(states, list) or any(
        not isinstance(item, Mapping) or item.get("split") not in {"train", "val"}
        for item in states
    ):
        raise ValueError("sealed-test individual metadata is exposed in states")
    if not isinstance(integrity, Mapping) or set(integrity) != expected_integrity_keys:
        raise ValueError("sealed-test individual metadata/integrity schema is invalid")
    if (
        int(integrity.get("split_code", -1)) != 2
        or not isinstance(integrity.get("states_sha256"), str)
        or len(str(integrity["states_sha256"])) != 64
        or integrity.get("schema_finite_passed") is not True
        or integrity.get("reset_passed") is not True
        or integrity.get("control_execution_passed") is not True
        or integrity.get("individual_state_metadata_redacted") is not True
        or integrity.get("sealed_test_stratification_passed") is not True
    ):
        raise ValueError("sealed-test individual metadata redaction attestation failed")
    development_ids = {
        str(item["state_id"])
        for item in states
        if isinstance(item, Mapping) and "state_id" in item
    }

    def reject_unknown_report_ids(value: Any) -> None:
        if isinstance(value, Mapping):
            if "state_id" in value and not isinstance(value["state_id"], (Mapping, list)):
                if str(value["state_id"]) not in development_ids:
                    raise ValueError(
                        "sealed-test individual metadata is exposed in a public audit report"
                    )
            for nested in value.values():
                reject_unknown_report_ids(nested)
        elif isinstance(value, list):
            for nested in value:
                reject_unknown_report_ids(nested)

    reject_unknown_report_ids(manifest)
    selection_audit = manifest.get("state_selection_audit")
    if isinstance(selection_audit, Mapping):
        by_split = selection_audit.get("by_split")
        if isinstance(by_split, Mapping) and set(by_split) - {"train", "val"}:
            raise ValueError("sealed-test individual metadata/covariates are exposed in stratification audit")


def _preflight_development_path_role(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    data_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Reject a sealed path using manifest strings before any payload I/O."""

    dataset_files = manifest.get("dataset_files")
    if not isinstance(dataset_files, Mapping):
        raise ValueError("manifest dataset_files is missing")
    development = dataset_files.get("development_records")
    sealed = dataset_files.get("sealed_test_records")
    if not isinstance(development, Mapping) or not isinstance(sealed, Mapping):
        raise ValueError("manifest record artifact roles are missing")
    if development.get("role") != "development" or sealed.get("role") != "sealed_test":
        raise ValueError("manifest record artifact roles are invalid")
    configured = config["collection"]["outputs"]
    if development.get("path") != configured["development_records"]:
        raise ValueError("manifest development path differs from config")
    if sealed.get("path") != configured["sealed_test_records"]:
        raise ValueError("manifest sealed-test path differs from config")
    development_path = (manifest_path.parent / str(development["path"])).resolve()
    sealed_path = (manifest_path.parent / str(sealed["path"])).resolve()
    requested = data_path.resolve()
    if requested == sealed_path:
        raise PermissionError("sealed-test payload path is prohibited")
    if requested != development_path:
        raise PermissionError("--data must be the manifest development-role artifact")


def _state_identity(item: Mapping[str, Any], map_name: str) -> dict[str, Any]:
    waypoint = item.get("waypoint")
    if not isinstance(waypoint, Mapping):
        # The V3 collector promotes the stable generated-waypoint identity to
        # top-level fields; legacy paired-rollout manifests nested it.
        waypoint = item
    required_waypoint = ("road_id", "section_id", "lane_id", "s")
    if any(key not in waypoint for key in required_waypoint):
        raise ValueError("manifest waypoint identity is incomplete")
    lateral = item.get("requested_lateral_offset_m", item.get("lateral_offset_m"))
    if lateral is None:
        raise ValueError("manifest state is missing requested_lateral_offset_m")
    if "initial_speed_mps" not in item:
        raise ValueError("manifest state is missing initial_speed_mps")
    curvature = item.get("curvature_stratum", item.get("curvature_bin"))
    if curvature is None:
        raise ValueError("manifest state is missing curvature stratum")
    speed_stratum = item.get("speed_stratum", item.get("initial_speed_mps"))
    return {
        "map": map_name,
        "road_id": int(waypoint["road_id"]),
        "section_id": int(waypoint["section_id"]),
        "lane_id": int(waypoint["lane_id"]),
        "waypoint_s": float(waypoint["s"]),
        "lateral_offset_m": float(lateral),
        "initial_speed_mps": float(item["initial_speed_mps"]),
        "speed_stratum": str(speed_stratum),
        "curvature_stratum": str(curvature),
    }


def _balanced(values: Sequence[Any]) -> bool:
    counts: dict[str, int] = {}
    for value in values:
        key = json.dumps(value, sort_keys=True, default=str)
        counts[key] = counts.get(key, 0) + 1
    return bool(counts) and max(counts.values()) - min(counts.values()) <= 1


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    config: dict[str, Any],
    config_sha256: str,
    data_sha256: str,
    data_path: Path,
    manifest_path: Path,
    records: MPCGroundingRecords,
    split_summary: Mapping[str, int],
    smoke: bool = False,
) -> dict[str, Any]:
    _reject_sealed_metadata_exposure(manifest)
    if int(manifest.get("schema_version", -1)) != 3:
        raise ValueError("collection manifest schema_version must equal 3")
    if manifest.get("dataset_schema") != "mpc-local-carla-v3" or manifest.get("status") != "complete":
        raise ValueError("collection manifest schema/status is invalid")
    if manifest.get("smoke") is not bool(smoke):
        raise ValueError("manifest smoke status differs from runner mode")
    if manifest.get("public_manifest_redaction") != config["collection"]["public_manifest_redaction"]:
        raise ValueError("manifest public redaction attestation differs from config")
    if str(manifest.get("config_sha256", "")).lower() != config_sha256:
        raise ValueError("V3 config SHA-256 differs from collection manifest")
    dataset_files = manifest.get("dataset_files")
    expected_dataset_keys = {
        "development_records", "sealed_test_records",
        "development_diagnostics", "sealed_test_diagnostics",
    }
    if not isinstance(dataset_files, Mapping) or set(dataset_files) != expected_dataset_keys:
        raise ValueError(
            "manifest dataset_files must contain exactly the four byte-separated V3 artifacts"
        )
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("manifest files identity block is missing")
    expected_roles = {
        "development_records": "development",
        "sealed_test_records": "sealed_test",
        "development_diagnostics": "development_diagnostics",
        "sealed_test_diagnostics": "sealed_test_diagnostics",
    }
    resolved_files: dict[str, Path] = {}
    for key, role in expected_roles.items():
        item = dataset_files[key]
        if not isinstance(item, Mapping) or item.get("role") != role:
            raise ValueError(f"manifest dataset_files.{key} role is invalid")
        declared_path = Path(str(item.get("path", "")))
        if declared_path.name == "" or declared_path.is_absolute():
            raise ValueError(f"manifest dataset_files.{key}.path must be a relative file path")
        resolved = (manifest_path.parent / declared_path).resolve()
        if resolved.parent != manifest_path.parent.resolve():
            raise ValueError(f"manifest dataset_files.{key}.path escapes the dataset directory")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        declared_sha = item.get("sha256")
        if not isinstance(declared_sha, str) or len(declared_sha) != 64:
            raise ValueError(f"manifest dataset_files.{key}.sha256 is invalid")
        nested = files.get(resolved.name)
        if not isinstance(nested, Mapping) or nested.get("sha256") != declared_sha:
            raise ValueError(f"manifest nested file identity is inconsistent for {resolved.name}")
        resolved_files[key] = resolved
    development_item = dataset_files["development_records"]
    standard_dataset_fields = {
        "path", "role", "sha256", "records", "states", "split_codes", "sealed"
    }
    for key in (
        "development_records", "development_diagnostics", "sealed_test_diagnostics"
    ):
        if set(dataset_files[key]) != standard_dataset_fields:
            raise ValueError(f"manifest dataset_files.{key} field set is not frozen")
    if set(dataset_files["sealed_test_records"]) != standard_dataset_fields | {"states_sha256"}:
        raise ValueError("manifest sealed-test record identity field set is not frozen")
    if resolved_files["development_records"] != data_path.resolve():
        raise ValueError("--data must identify the manifest development artifact exactly")
    if development_item.get("sealed") is not False:
        raise ValueError("development artifact must not be marked sealed")
    if str(development_item["sha256"]).lower() != data_sha256:
        raise ValueError("development records SHA-256 differs from manifest")
    if list(development_item.get("split_codes", ())) != [0, 1]:
        raise ValueError("development artifact must declare only split codes 0 and 1")
    sealed_item = dataset_files["sealed_test_records"]
    if sealed_item.get("sealed") is not True or list(sealed_item.get("split_codes", ())) != [2]:
        raise ValueError("sealed-test artifact role/split declaration is invalid")
    # Deliberately do not open or hash the sealed artifacts.  Their identity,
    # byte size, and count are collector attestations only in this runner.
    expected_sealed_records = 18 if smoke else int(config["gates"]["required_test_records"])
    expected_sealed_states = 1 if smoke else int(config["gates"]["required_sealed_test_states"])
    if int(sealed_item.get("records", -1)) != expected_sealed_records:
        raise ValueError("sealed-test manifest record count differs from config")
    if int(sealed_item.get("states", -1)) != expected_sealed_states:
        raise ValueError("sealed-test manifest state count differs from config")
    sealed_states_sha = sealed_item.get("states_sha256")
    sealed_integrity = manifest["sealed_test_integrity"]
    if (
        not isinstance(sealed_states_sha, str)
        or len(sealed_states_sha) != 64
        or sealed_states_sha != sealed_integrity.get("states_sha256")
        or int(sealed_integrity.get("records", -1)) != expected_sealed_records
        or int(sealed_integrity.get("states", -1)) != expected_sealed_states
    ):
        raise ValueError("sealed-test state/count identity attestation is inconsistent")
    sealed_bytes = files[resolved_files["sealed_test_records"].name].get("bytes")
    if not isinstance(sealed_bytes, int) or int(sealed_bytes) <= 0:
        raise ValueError("sealed-test manifest byte-size attestation is missing")
    if any(
        "collision" in str(key).lower() or "outcome" in str(key).lower()
        for key in sealed_item
    ):
        raise ValueError("sealed-test dataset entry leaks outcome/collision aggregates")
    development_diagnostics = dataset_files["development_diagnostics"]
    diagnostics_sha = str(development_diagnostics["sha256"]).lower()
    diagnostics_path = resolved_files["development_diagnostics"]
    if development_diagnostics.get("sealed") is not False or _sha256(diagnostics_path) != diagnostics_sha:
        raise ValueError("development diagnostics identity differs from manifest")
    sealed_diagnostics = dataset_files["sealed_test_diagnostics"]
    if sealed_diagnostics.get("sealed") is not True:
        raise ValueError("sealed-test diagnostics must be marked sealed")
    sealed_diagnostics_bytes = files[
        resolved_files["sealed_test_diagnostics"].name
    ].get("bytes")
    if not isinstance(sealed_diagnostics_bytes, int) or int(sealed_diagnostics_bytes) <= 0:
        raise ValueError("sealed-test diagnostics byte-size attestation is missing")
    if any(
        "collision" in str(key).lower() or "outcome" in str(key).lower()
        for key in sealed_diagnostics
    ):
        raise ValueError("sealed-test diagnostics entry leaks outcome/collision aggregates")

    for key, path in resolved_files.items():
        nested = files.get(path.name)
        item = dataset_files[key]
        if not isinstance(nested, Mapping) or nested.get("sha256") != item.get("sha256"):
            raise ValueError(f"manifest nested file identity is inconsistent for {path.name}")

    protocol = manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("manifest protocol block is missing")
    if protocol.get("map") != EXPECTED_MAP:
        raise ValueError("manifest map is not the fresh V3 map")
    action_profile = protocol.get("action_profile", manifest.get("action_profile"))
    if action_profile != EXPECTED_ACTION_PROFILE:
        raise ValueError("manifest action profile differs from V3 config")
    if int(protocol.get("seed", -1)) != int(config["collection"]["seed"]):
        raise ValueError("manifest collection seed differs from config")
    protocol_state_selection = protocol.get("state_selection")
    if (
        not isinstance(protocol_state_selection, Mapping)
        or protocol_state_selection.get("waypoint_state_source")
        != config["collection"]["state_source"]
    ):
        raise ValueError("manifest waypoint state source differs from config")
    if int(protocol.get("horizon_ticks", -1)) != int(config["collection"]["horizon_ticks"]):
        raise ValueError("manifest horizon differs from config")
    protocol_cem = protocol.get("cem")
    if not isinstance(protocol_cem, Mapping):
        raise ValueError("manifest CEM protocol is missing")
    for key in ("iterations", "population", "elite_count", "initial_mean", "initial_std", "lower", "upper", "minimum_std"):
        expected_cem = config["cem"].get(key)
        if smoke and key == "population":
            expected_cem = 6
        if smoke and key == "elite_count":
            expected_cem = 2
        if protocol_cem.get(key) != expected_cem:
            raise ValueError(f"manifest CEM {key} differs from config")
    protocol_cost = protocol.get("cost")
    if not isinstance(protocol_cost, Mapping):
        raise ValueError("manifest physical-cost protocol is missing")
    for key in (
        "progress_weight", "lateral_squared_weight", "yaw_squared_weight",
        "speed_squared_weight", "steering_squared_weight",
        "longitudinal_squared_weight", "collision_weight", "pair_tie_threshold",
    ):
        if not math.isclose(
            float(protocol_cost.get(key, math.nan)), float(config["cost"][key]),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(f"manifest physical-cost {key} differs from config")
    server = manifest.get("server_and_map")
    if not isinstance(server, Mapping) or (
        str(server.get("client_version")) != str(config["collection"]["carla_version"])
        or str(server.get("server_version")) != str(config["collection"]["carla_version"])
        or str(server.get("map_name")) != EXPECTED_MAP
    ):
        raise ValueError("manifest CARLA client/server/map identity differs from config")

    expected_records = 36 if smoke else int(config["gates"]["required_development_records"])
    if split_summary["total"] != expected_records:
        raise ValueError(f"records contain {split_summary['total']} rows; expected {expected_records}")
    effective_population = 6 if smoke else int(config["cem"]["population"])
    effective_state_counts = (
        {"train": 1, "val": 1, "test": 1}
        if smoke
        else {key: int(config["collection"]["states"][key]) for key in ("train", "val", "test")}
    )
    population_iterations = effective_population * int(config["cem"]["iterations"])
    expected_rows = {
        "train": effective_state_counts["train"] * population_iterations,
        "val": effective_state_counts["val"] * population_iterations,
    }
    if {key: int(split_summary[key]) for key in expected_rows} != expected_rows:
        raise ValueError("record split row counts differ from frozen V3 counts")
    total_attested = int(development_item.get("records", -1)) + int(sealed_item.get("records", -1))
    expected_total_attested = 54 if smoke else int(config["gates"]["required_total_records_attested"])
    if total_attested != expected_total_attested:
        raise ValueError("manifest development+sealed record count attestation is invalid")
    split_ids = {name: _manifest_split_ids(manifest, name) for name in ("train", "val")}
    if any(
        len(split_ids[name]) != effective_state_counts[name]
        for name in split_ids
    ):
        raise ValueError("manifest split state counts differ from config")
    if set(split_ids["train"]) & set(split_ids["val"]):
        raise ValueError("manifest state IDs cross splits")
    actual_by_split = {
        "train": sorted({str(value) for value in records.state_id[records.split_code == TRAIN_SPLIT]}),
        "val": sorted({str(value) for value in records.state_id[records.split_code == VAL_SPLIT]}),
    }
    for name in ("train", "val"):
        if sorted(split_ids[name]) != actual_by_split[name]:
            raise ValueError(f"manifest {name} states differ from development records")
    for split_code in (TRAIN_SPLIT, VAL_SPLIT):
        for group in state_group_indices(records, split_codes=(split_code,)):
            counts = [
                int(np.sum(records.cem_iteration[group.indices] == iteration))
                for iteration in range(int(config["cem"]["iterations"]))
            ]
            if counts != [effective_population] * int(config["cem"]["iterations"]):
                raise ValueError(
                    f"state {group.state_id!r} candidate counts {counts} differ from full iteration"
                )

    states = manifest.get("states")
    expected_state_total = 2 if smoke else 48
    if not isinstance(states, list) or len(states) != expected_state_total:
        raise ValueError(f"manifest must contain exactly {expected_state_total} state descriptions")
    by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    base_identities: set[tuple[int, int, int, float]] = set()
    for raw in states:
        if not isinstance(raw, Mapping) or "state_id" not in raw or "split" not in raw:
            raise ValueError("manifest state description is incomplete")
        state_id = str(raw["state_id"])
        split = str(raw["split"])
        if state_id in by_id or split not in split_ids or state_id not in split_ids[split]:
            raise ValueError("manifest state identity/split is inconsistent")
        identity = _state_identity(raw, EXPECTED_MAP)
        by_id[state_id] = (split, identity)
        base = (
            int(identity["road_id"]), int(identity["section_id"]),
            int(identity["lane_id"]), round(float(identity["waypoint_s"]), 6),
        )
        if base in base_identities:
            raise ValueError("V3 base waypoint identity is not unique")
        base_identities.add(base)
    for split in ("train", "val"):
        identities = [identity for state_split, identity in by_id.values() if state_split == split]
        if smoke:
            continue
        expected_levels = {
            "initial_speed_mps": {4.0, 6.0, 8.0},
            "lateral_offset_m": {-0.25, 0.0, 0.25},
            "curvature_stratum": {"0", "1", "2", "3"},
        }
        for key, expected in expected_levels.items():
            values = [identity[key] for identity in identities]
            normalized_levels = {
                str(value) if key == "curvature_stratum" else float(value)
                for value in values
            }
            if normalized_levels != expected or not _balanced(values):
                raise ValueError(f"manifest {split} {key} strata are not balanced")

    state_selection = protocol.get("state_selection")
    if not isinstance(state_selection, Mapping):
        raise ValueError("manifest state-selection provenance is missing")
    map_disjoint = state_selection.get("map_disjoint_from_parents")
    if not isinstance(map_disjoint, Mapping) or map_disjoint.get("passed") is not True:
        raise ValueError("manifest does not attest map disjointness from V1/V2")
    if state_selection.get("waypoint_state_source") != "map_generate_waypoints":
        raise ValueError("manifest waypoint-state source attestation is missing")
    parent_values = map_disjoint.get("parents")
    if not isinstance(parent_values, list) or len(parent_values) != 2:
        raise ValueError("manifest parent-map provenance is incomplete")
    parent_by_name = {str(item.get("name")): item for item in parent_values if isinstance(item, Mapping)}
    parent_mapping = {
        "parent_v1": "mpc_local_grounding_carla_v1",
        "parent_v2": "mpc_local_grounding_carla_v2",
    }
    for config_key, manifest_key in parent_mapping.items():
        actual = parent_by_name.get(manifest_key)
        expected = config["provenance"][config_key]
        if not isinstance(actual, Mapping) or any(
            str(actual.get(key)) != str(expected.get(key))
            for key in ("map", "states_sha256", "records_sha256")
        ):
            raise ValueError(f"manifest {manifest_key} provenance differs from config")

    summary = manifest.get("collection_summary")
    if not isinstance(summary, Mapping):
        raise ValueError("manifest collection summary is missing")
    forbidden_summary_keys = {
        "collision_label_count", "collision_event_count", "collision_by_split",
        "test_collision_label_count", "test_collision_event_count",
        "test_outcome_summary", "outcome_by_split",
    }
    if forbidden_summary_keys & set(summary):
        raise ValueError("manifest exposes forbidden sealed-test outcome/collision aggregates")
    if (
        summary.get("collision_gate_scope") != "development_only"
        or int(summary.get("development_collision_label_count", -1)) != 0
        or int(summary.get("development_collision_event_count", -1)) != 0
    ):
        raise ValueError("V3 collision-free integrity gate failed")
    if summary.get("sealed_test_outcomes_redacted") is not True:
        raise ValueError("manifest does not attest sealed-test outcome redaction")
    required_true = (
        "control_execution_audit_passed", "initial_velocity_command_audit_passed",
        "paired_initial_state_passed",
    )
    if any(summary.get(key) is not True for key in required_true):
        raise ValueError("manifest V3 collection audit is incomplete")
    selection_audit = manifest.get("state_selection_audit")
    requested_audit = manifest.get("requested_actual_initial_state_audit")
    fresh_attestation = manifest.get("fresh_state_attestation")
    if (
        not isinstance(selection_audit, Mapping)
        or selection_audit.get("passed") is not True
        or selection_audit.get("waypoint_state_source_passed") is not True
        or selection_audit.get("unique_base_waypoint_passed") is not True
        or not isinstance(requested_audit, Mapping)
        or requested_audit.get("passed") is not True
        or not isinstance(fresh_attestation, Mapping)
        or fresh_attestation.get("passed") is not True
    ):
        raise ValueError("manifest V3 state-selection/requested-state audit failed")
    development_state_ids = set(split_ids["train"]) | set(split_ids["val"])
    requested_reports = requested_audit.get("states")
    paired_reports = summary.get("paired_initial_state")
    if (
        not isinstance(requested_reports, list)
        or len(requested_reports) != len(development_state_ids)
        or any(not isinstance(item, Mapping) or item.get("passed") is not True for item in requested_reports)
        or {str(item.get("state_id")) for item in requested_reports if isinstance(item, Mapping)}
        != development_state_ids
        or not isinstance(paired_reports, list)
        or len(paired_reports) != len(development_state_ids)
        or any(not isinstance(item, Mapping) or item.get("passed") is not True for item in paired_reports)
        or {str(item.get("state_id")) for item in paired_reports if isinstance(item, Mapping)}
        != development_state_ids
    ):
        raise ValueError("development-only state audit coverage is incomplete or exposes sealed IDs")
    cleanup = manifest.get("cleanup")
    if not isinstance(cleanup, Mapping) or cleanup.get("settings_restored") is not True or cleanup.get("actors_remaining") != [] or cleanup.get("errors") != []:
        raise ValueError("collector cleanup attestation failed")

    return {
        "records_sha256": data_sha256,
        "config_sha256": config_sha256,
        "diagnostics_path": str(diagnostics_path),
        "diagnostics_sha256": diagnostics_sha,
        "sealed_test": {
            "path": str(resolved_files["sealed_test_records"]),
            "sha256_declared_not_opened": sealed_item["sha256"],
            "bytes_declared": int(sealed_bytes),
            "records_declared": int(sealed_item["records"]),
            "states_sha256_declared": sealed_states_sha,
            "payload_opened": False,
        },
        "split_state_ids": split_ids,
        "state_identity_by_id": {key: value[1] for key, value in by_id.items()},
        "map_disjoint_from_v1_v2": True,
        "stratification_balance_passed": True,
        "development_unique_base_waypoints": len(base_identities),
        "sealed_test_stratification_passed_manifest_only": bool(
            sealed_integrity["sealed_test_stratification_passed"]
        ),
    }


def _select_states(records: MPCGroundingRecords, state_ids: Sequence[str]) -> MPCGroundingRecords:
    allowed = set(str(value) for value in state_ids)
    indices = np.asarray(
        [index for index, value in enumerate(records.state_id) if str(value) in allowed],
        dtype=np.int64,
    )
    if indices.size == 0:
        raise ValueError("state selection is empty")
    actual = {str(records.state_id[index]) for index in indices}
    if actual != allowed:
        raise ValueError(f"state selection is missing IDs: {sorted(allowed-actual)}")
    return records.subset(indices)


def _split_fit_inner_state_ids(
    records: MPCGroundingRecords,
    *,
    inner_validation_states: int,
    seed: int,
    identity_by_state: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    train_ids = sorted({str(value) for value in records.state_id[records.split_code == TRAIN_SPLIT]})
    count = int(inner_validation_states)
    if count < 1 or count >= len(train_ids):
        raise ValueError("inner_validation_states must leave non-empty fit and inner sets")
    identities = identity_by_state or {state_id: state_id for state_id in train_ids}
    if set(train_ids) - set(identities):
        raise ValueError("state identity provenance is incomplete for train states")
    identity_fields = (
        "map", "road_id", "section_id", "lane_id", "waypoint_s",
        "lateral_offset_m", "initial_speed_mps",
    )

    def frozen_identity(state_id: str) -> Any:
        identity = identities[state_id]
        if isinstance(identity, Mapping):
            missing = [key for key in identity_fields if key not in identity]
            if missing:
                raise ValueError(
                    f"state identity provenance is missing frozen fields: {missing}"
                )
            return {key: identity[key] for key in identity_fields}
        return identity

    ordered = sorted(
        train_ids,
        key=lambda state_id: (
            _stable_hash({"seed": int(seed), "identity": frozen_identity(state_id)}),
            state_id,
        ),
    )
    inner = sorted(ordered[:count])
    fit = sorted(ordered[count:])
    return fit, inner


def _raw_action_hash(action: np.ndarray) -> bytes:
    return hashlib.sha256(np.asarray(action, dtype="<f4").tobytes()).digest()


def _full_iteration_groups(
    records: MPCGroundingRecords,
    *,
    iteration: int,
    population: int,
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    groups: list[np.ndarray] = []
    per_state: dict[str, Any] = {}
    for group in state_group_indices(
        records,
        split_codes=(int(records.split_code[0]),),
        cem_iterations=(iteration,),
        min_candidates=population,
    ):
        if len(group.indices) != population:
            raise ValueError(f"state {group.state_id!r} iteration {iteration} lacks all {population} candidates")
        ordered = np.asarray(
            sorted(
                group.indices.tolist(),
                key=lambda index: (_raw_action_hash(records.action_params[index]), int(index)),
            ),
            dtype=np.int64,
        )
        if len(np.unique(records.action_params[ordered], axis=0)) != population:
            raise ValueError(f"state {group.state_id!r} iteration {iteration} has duplicate actions")
        groups.append(ordered)
        minimum = float(np.min(records.real_cost[ordered]))
        best = np.flatnonzero(records.real_cost[ordered].astype(np.float64) == minimum)
        per_state[str(group.state_id)] = {
            "candidates": population,
            "ordered_action_sha256": _stable_hash(records.action_params[ordered].tolist()),
            "deterministic_target_index": int(ordered[int(best[0])]),
            "exact_real_argmin_ties": int(len(best)),
        }
    if not groups:
        raise ValueError("full-iteration group construction selected no states")
    return tuple(groups), {
        "iteration": int(iteration),
        "states": len(groups),
        "candidates_per_state": population,
        "unique_query_rollouts": len(groups) * population,
        "subsampling": False,
        "exact_tie_break": "sha256_raw_action_bytes",
        "per_state": per_state,
    }


def _stats_to_json(stats: ZScoreStats) -> dict[str, Any]:
    return {
        "state_mean": stats.state_mean.tolist(), "state_std": stats.state_std.tolist(),
        "action_mean": stats.action_mean.tolist(), "action_std": stats.action_std.tolist(),
        "outcome_mean": stats.outcome_mean.tolist(), "outcome_std": stats.outcome_std.tolist(),
        "cost_mean": float(stats.cost_mean), "cost_std": float(stats.cost_std),
        "train_records": int(stats.train_records),
    }


def _tensor_records(records: MPCGroundingRecords, stats: ZScoreStats, device: torch.device) -> dict[str, torch.Tensor]:
    initial = torch.tensor(np.array(records.initial_features, copy=True), device=device)
    action = torch.tensor(np.array(records.action_params, copy=True), device=device)
    outcome = torch.tensor(np.array(records.outcome_features, copy=True), device=device)
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
        "real_cost": torch.tensor(np.array(records.real_cost, copy=True), device=device).float(),
    }


def _forward_all(model: GroundedOutcomeModel, tensors: Mapping[str, torch.Tensor], batch_size: int) -> dict[str, torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(tensors["initial"]), batch_size):
        outputs.append(model(tensors["initial"][start:start+batch_size], tensors["action"][start:start+batch_size])["outcome"])
    return {"outcome": torch.cat(outputs, dim=0)}


def _predicted_cost(
    normalized_outcome: torch.Tensor,
    raw_action: torch.Tensor,
    stats: ZScoreStats,
    cost_config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats.outcome_mean.tolist(), dtype=normalized_outcome.dtype, device=normalized_outcome.device)
    std = torch.tensor(stats.outcome_std.tolist(), dtype=normalized_outcome.dtype, device=normalized_outcome.device)
    raw_outcome = normalized_outcome * std + mean
    collision = torch.zeros(raw_outcome.shape[0], dtype=raw_outcome.dtype, device=raw_outcome.device)
    return physical_cost(raw_outcome, raw_action, collision, cost_config), raw_outcome


def _objective(
    model: GroundedOutcomeModel,
    tensors: Mapping[str, torch.Tensor],
    *,
    prediction_indices: np.ndarray,
    query_groups: Sequence[np.ndarray] | None,
    stats: ZScoreStats,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    output = _forward_all(model, tensors, int(config["training"]["batch_size"]))
    selected = torch.tensor(prediction_indices.tolist(), dtype=torch.long, device=tensors["outcome"].device)
    prediction_loss = torch.nn.functional.mse_loss(output["outcome"][selected], tensors["outcome"][selected])
    predicted_cost, _ = _predicted_cost(output["outcome"], tensors["raw_action"], stats, config["cost"])
    if query_groups is None:
        listwise = prediction_loss.new_zeros(())
    else:
        listwise = statewise_softmin_listwise_loss(
            predicted_cost,
            tensors["real_cost"],
            query_groups,
            temperature=float(config["training"]["listwise_temperature"]),
        )
    total = prediction_loss + float(config["training"]["listwise_weight"]) * listwise
    return total, {
        "total": float(total.detach()),
        "outcome_mse_normalized": float(prediction_loss.detach()),
        "top1_listwise_cross_entropy": float(listwise.detach()),
    }


def _model_signature(model: nn.Module) -> dict[str, Any]:
    shapes = {name: list(value.shape) for name, value in model.named_parameters()}
    return {
        "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "parameter_shapes": shapes,
        "sha256": _stable_hash(shapes),
    }


def _train_variant(
    *,
    variant: str,
    seed: int,
    fit_records: MPCGroundingRecords,
    inner_records: MPCGroundingRecords,
    stats: ZScoreStats,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[GroundedOutcomeModel, dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported V3 variant: {variant}")
    _seed_everything(seed)
    model = GroundedOutcomeModel(
        initial_dim=fit_records.state_dim,
        latent_dim=int(config["model"]["latent_dim"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
    ).to(device)
    signature = _model_signature(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    population = int(config["cem"]["population"])
    fit_prediction_groups, fit_prediction_audit = _full_iteration_groups(
        fit_records, iteration=1, population=population
    )
    inner_prediction_groups, inner_prediction_audit = _full_iteration_groups(
        inner_records, iteration=1, population=population
    )
    fit_global, fit_global_audit = _full_iteration_groups(fit_records, iteration=0, population=population)
    fit_elite, fit_elite_audit = _full_iteration_groups(fit_records, iteration=2, population=population)
    inner_elite, inner_elite_audit = _full_iteration_groups(inner_records, iteration=2, population=population)
    fit_prediction = np.concatenate(fit_prediction_groups)
    inner_prediction = np.concatenate(inner_prediction_groups)
    query = {
        "prediction_only": None,
        "global_listwise": fit_global,
        "elite_listwise": fit_elite,
    }[variant]
    fit_tensors = _tensor_records(fit_records, stats, device)
    inner_tensors = _tensor_records(inner_records, stats, device)
    max_epochs = int(config["training"]["max_epochs"])
    min_epochs = int(config["training"]["min_epochs"])
    patience = int(config["training"]["patience"])
    min_delta = float(config["training"]["checkpoint_min_delta"])
    clip = float(config["training"]["gradient_clip_norm"])
    best_value = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    patience_reached = False
    history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total, train_parts = _objective(
            model, fit_tensors, prediction_indices=fit_prediction,
            query_groups=query, stats=stats, config=config,
        )
        train_total.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            inner_total, inner_parts = _objective(
                # Checkpoint-only outcome MSE covers all 0/1/2 candidates.
                # These train-split inner labels never enter optimizer updates.
                model, inner_tensors,
                prediction_indices=np.arange(len(inner_records), dtype=np.int64),
                query_groups=inner_elite, stats=stats, config=config,
            )
        value = float(inner_total)
        history.append({
            "epoch": epoch, "fit": train_parts,
            "inner_common": inner_parts,
            "inner_common_objective": value,
            "gradient_norm_before_clip": gradient_norm,
        })
        if math.isfinite(value) and value < best_value - min_delta:
            best_value = value
            best_epoch = epoch
            best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch >= min_epochs and stale >= patience:
            patience_reached = True
            break
    if best_state is None:
        raise RuntimeError("training produced no finite inner-validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "variant": variant,
        "seed": int(seed),
        "architecture": signature,
        "outcome_label_scope": "fit_states_cem_iteration_1_all_24",
        "query_label_scope": {
            "prediction_only": "none",
            "global_listwise": "fit_states_cem_iteration_0_all_24",
            "elite_listwise": "fit_states_cem_iteration_2_all_24",
        }[variant],
        "outer_records_passed_to_training": False,
        "epochs_requested": max_epochs,
        "epochs_completed": len(history),
        "minimum_epochs": min_epochs,
        "early_stopped": len(history) < max_epochs,
        "best_epoch": best_epoch,
        "best_inner_common_objective": best_value,
        "checkpoint_selection_split": "train_only_inner_validation",
        "checkpoint_tie_break": "earliest_epoch",
        "converged_by_frozen_patience": patience_reached,
        "gradient_clipping": {
            "configured_max_norm": clip,
            "epochs_exceeding_max_norm": int(sum(row["gradient_norm_before_clip"] > clip for row in history)),
            "maximum_gradient_norm_before_clip": max(row["gradient_norm_before_clip"] for row in history),
        },
        "candidate_exposure": {
            "fit_prediction": fit_prediction_audit,
            "fit_global_query": fit_global_audit,
            "fit_elite_query": fit_elite_audit,
            "inner_prediction": inner_prediction_audit,
            "inner_common_elite": inner_elite_audit,
            "global_elite_equal_unique_query_rollouts": fit_global_audit["unique_query_rollouts"] == fit_elite_audit["unique_query_rollouts"],
        },
        "history": history,
    }


@torch.inference_mode()
def _evaluate(
    model: GroundedOutcomeModel,
    records: MPCGroundingRecords,
    stats: ZScoreStats,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    tensors = _tensor_records(records, stats, device)
    output = _forward_all(model, tensors, int(config["training"]["batch_size"]))
    predicted_cost, raw_outcome = _predicted_cost(output["outcome"], tensors["raw_action"], stats, config["cost"])
    decision = decision_state_macro_metrics(
        records,
        predicted_cost,
        predicted_outcome=raw_outcome,
        split_code=VAL_SPLIT,
        cem_iterations=(int(config["evaluation"]["validation_cem_iteration"]),),
        epsilon=float(config["evaluation"]["epsilon_regret"]),
        predicted_tie_tolerance=0.0,
        oracle_tie_tolerance=0.0,
    )
    global_outcome = decision_state_macro_metrics(
        records,
        predicted_cost,
        predicted_outcome=raw_outcome,
        split_code=VAL_SPLIT,
        cem_iterations=(int(config["training"]["global_query_cem_iteration"]),),
        epsilon=float(config["evaluation"]["epsilon_regret"]),
        predicted_tie_tolerance=0.0,
        oracle_tie_tolerance=0.0,
    )
    global_by_state = {
        item["state_id"]: float(item["outcome_mse"])
        for item in global_outcome["state_details"]
    }
    for item in decision["state_details"]:
        item["global_outcome_mse"] = global_by_state[item["state_id"]]
    decision["state_macro_global_outcome_mse"] = float(
        global_outcome["state_macro_outcome_mse"]
    )
    decision["global_outcome_mse_cem_iteration"] = int(
        config["training"]["global_query_cem_iteration"]
    )
    return decision


def _validate_cost(records: MPCGroundingRecords, config: Mapping[str, Any]) -> dict[str, Any]:
    if not np.all(np.isin(records.collision, (0.0, 1.0))):
        raise ValueError("collision labels must be binary")
    collisions = int(np.count_nonzero(records.collision))
    if collisions:
        raise ValueError("V3 requires collision-free records because the model has no collision head")
    with torch.inference_mode():
        recomputed = physical_cost(
            torch.tensor(np.array(records.outcome_features, copy=True)),
            torch.tensor(np.array(records.action_params, copy=True)),
            torch.tensor(np.array(records.collision, copy=True)),
            config["cost"],
        ).numpy()
    error = np.abs(recomputed.astype(np.float64) - records.real_cost.astype(np.float64))
    tolerance = 1e-5 + 1e-5 * np.abs(records.real_cost.astype(np.float64))
    if np.any(error > tolerance):
        raise ValueError("real_cost differs from the frozen physical-cost formula")
    return {
        "development_collision_records": collisions,
        "sealed_collision_status": "unknown_not_accessed",
        "real_cost_recomputed_on_supplied_scope": True,
        "maximum_absolute_recompute_error": float(np.max(error, initial=0.0)),
        "predicted_collision_policy": "constant_zero_fail_closed",
    }


def _fractional_reduction(baseline: float, method: float) -> float | None:
    if abs(baseline) <= 1e-12:
        return None
    return float((baseline - method) / abs(baseline))


def _paired_bootstrap(
    baseline: Mapping[str, Any],
    method: Mapping[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    baseline_states = {item["state_id"]: item for item in baseline["state_details"]}
    method_states = {item["state_id"]: item for item in method["state_details"]}
    if set(baseline_states) != set(method_states):
        raise ValueError("paired bootstrap state identities differ")
    ids = sorted(baseline_states)
    rng = np.random.default_rng(seed)
    output: dict[str, list[float]] = {}
    for name in ("exact_argmin_simple_regret", "epsilon_regret", "global_outcome_mse"):
        differences = np.asarray(
            [float(baseline_states[state][name]) - float(method_states[state][name]) for state in ids],
            dtype=np.float64,
        )
        draws = rng.integers(0, len(ids), size=(samples, len(ids)))
        means = differences[draws].mean(axis=1)
        output[f"baseline_minus_elite_{name}_95ci"] = [
            float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
        ]
    return output


def _problem_existence(records: MPCGroundingRecords, config: Mapping[str, Any]) -> dict[str, Any]:
    epsilon = float(config["evaluation"]["epsilon_regret"])
    groups, _ = _full_iteration_groups(
        records, iteration=int(config["evaluation"]["validation_cem_iteration"]),
        population=int(config["cem"]["population"]),
    )
    details = []
    for indices in groups:
        cost = np.sort(records.real_cost[indices].astype(np.float64))
        details.append({
            "state_id": str(records.state_id[indices[0]]),
            "real_cost_range": float(cost[-1] - cost[0]),
            "best_second_gap": float(cost[1] - cost[0]),
            "range_gt_epsilon": bool(cost[-1] - cost[0] > epsilon),
        })
    fraction = float(np.mean([item["range_gt_epsilon"] for item in details]))
    threshold = float(config["gates"]["minimum_outer_state_fraction_cost_range_gt_epsilon"])
    return {
        "epsilon": epsilon, "states": len(details),
        "fraction_cost_range_gt_epsilon": fraction,
        "required_fraction": threshold, "pass": fraction >= threshold,
        "state_details": details,
    }


def _outer_gates(evaluations: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    gates = config["gates"]
    seed_checks: dict[str, Any] = {}
    for seed in config["training"]["seeds"]:
        key = str(int(seed))
        prediction = evaluations[key]["prediction_only"]
        global_value = evaluations[key]["global_listwise"]
        elite = evaluations[key]["elite_listwise"]
        exact_global = _fractional_reduction(
            float(global_value["state_macro_exact_argmin_simple_regret"]),
            float(elite["state_macro_exact_argmin_simple_regret"]),
        )
        exact_prediction = _fractional_reduction(
            float(prediction["state_macro_exact_argmin_simple_regret"]),
            float(elite["state_macro_exact_argmin_simple_regret"]),
        )
        epsilon_degradation = float(elite["state_macro_epsilon_regret"]) - float(global_value["state_macro_epsilon_regret"])
        prediction_mse = float(prediction["state_macro_global_outcome_mse"])
        mse_degradation = None if prediction_mse <= 1e-12 else (
            float(elite["state_macro_global_outcome_mse"]) - prediction_mse
        ) / abs(prediction_mse)
        checks = {
            "exact_regret_reduction_vs_global_listwise": exact_global,
            "exact_regret_reduction_vs_prediction_only": exact_prediction,
            "epsilon_regret_absolute_degradation_vs_global_listwise": epsilon_degradation,
            "outcome_mse_degradation_vs_prediction_only": mse_degradation,
            "exact_vs_global_pass": exact_global is not None and exact_global >= float(gates["minimum_exact_regret_reduction_fraction_vs_global_listwise"]),
            "exact_vs_prediction_pass": exact_prediction is not None and exact_prediction >= float(gates["minimum_exact_regret_reduction_fraction_vs_prediction_only"]),
            "epsilon_pass": epsilon_degradation <= float(gates["maximum_epsilon_regret_absolute_degradation_vs_global_listwise"]),
            "outcome_mse_pass": mse_degradation is not None and mse_degradation <= float(gates["maximum_global_outcome_mse_degradation_fraction"]),
        }
        checks["pass"] = bool(all(checks[name] for name in (
            "exact_vs_global_pass", "exact_vs_prediction_pass", "epsilon_pass", "outcome_mse_pass"
        )))
        checks["paired_bootstrap_report_only"] = {
            "vs_global_listwise": _paired_bootstrap(
                global_value, elite, samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["collection"]["seed"]) + int(seed) * 101,
            ),
            "vs_prediction_only": _paired_bootstrap(
                prediction, elite, samples=int(config["evaluation"]["bootstrap_samples"]),
                seed=int(config["collection"]["seed"]) + int(seed) * 101 + 1,
            ),
        }
        seed_checks[key] = checks
    passes = sum(bool(item["pass"]) for item in seed_checks.values())
    required = int(gates["required_outer_gate_seeds"])
    return {
        "seed_checks": seed_checks, "passing_seeds": passes,
        "required_passing_seeds": required, "pass": passes >= required,
        "bootstrap_is_report_only_not_rescue_gate": True,
    }


def _results_markdown(results: Mapping[str, Any]) -> str:
    if results["status"] == "non_evidentiary_smoke_preflight_pass":
        return (
            "# MPC-local grounding V3 smoke preflight\n\n"
            "- Status: **non_evidentiary_smoke_preflight_pass**\n"
            "- Development artifact schema/provenance/cost: passed\n"
            "- Sealed test payload: not opened, hashed, mapped, or decoded\n"
            "- Training and outer performance evaluation: not run\n"
        )
    lines = [
        "# MPC-local grounding V3 results", "",
        f"- Status: **{results['status']}**",
        "- Primary decision: deterministic exact predicted argmin",
        "- Test access: **prohibited; no sealed outcome was evaluated**",
        "- Checkpoints: train-only fit/inner validation; outer performance metrics were evaluated only after freeze",
        "",
        "| Seed | Variant | Exact simple regret | Epsilon regret | Top-1 accuracy | Outcome MSE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed, variants in results.get("outer_validation", {}).items():
        for variant in VARIANTS:
            item = variants[variant]
            lines.append(
                f"| {seed} | {variant} | {item['state_macro_exact_argmin_simple_regret']:.8f} | "
                f"{item['state_macro_epsilon_regret']:.8f} | {item.get('exact_top1_accuracy', float('nan')):.6f} | "
                f"{item['state_macro_global_outcome_mse']:.8f} |"
            )
    lines.extend(["", "## Gate", ""])
    convergence = results.get("convergence_gate", {})
    lines.append(
        f"- Inner convergence gate: {convergence.get('pass')} "
        f"({convergence.get('converged_models', 0)}/{convergence.get('required_models', 0)} models)"
    )
    problem = results.get("problem_existence", {})
    lines.append(
        f"- Cost-range problem gate: {problem.get('pass')} "
        f"({problem.get('fraction_cost_range_gt_epsilon', float('nan')):.3f})"
    )
    gate = results.get("outer_gate", {})
    lines.append(
        f"- Elite method gate: {gate.get('pass')} "
        f"({gate.get('passing_seeds', 0)}/{gate.get('required_passing_seeds', 0)} seeds)"
    )
    lines.extend([
        "", "This is a low-dimensional paired-dynamics mechanism result, not visual-world-model or closed-loop MPC evidence.",
    ])
    return "\n".join(lines) + "\n"


def _run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).resolve()
    data_path = Path(args.data).resolve()
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    config = _load_mapping(config_path)
    _require_config(config)
    manifest = _load_mapping(manifest_path)
    # This role check is intentionally before hashing or loading --data.  A
    # caller cannot cause this process to touch the sealed payload by passing
    # its path on the command line.
    _preflight_development_path_role(
        manifest, manifest_path=manifest_path, data_path=data_path, config=config
    )
    config_sha = _sha256(config_path)
    data_sha = _sha256(data_path)
    manifest_sha = _sha256(manifest_path)
    records, split_summary = _load_development_records(data_path)
    provenance = _validate_manifest(
        manifest, config=config, config_sha256=config_sha, data_sha256=data_sha,
        data_path=data_path, manifest_path=manifest_path, records=records,
        split_summary=split_summary,
        smoke=bool(args.smoke_preflight_only),
    )
    _fresh_output(output)
    _atomic_text(output / "config.frozen.yaml", config_path.read_text(encoding="utf-8"))

    if args.smoke_preflight_only:
        development_cost_audit = _validate_cost(records, config)
        results = {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "status": "non_evidentiary_smoke_preflight_pass",
            "completed_at_utc": _utc_now(),
            "evidentiary": False,
            "training_run": False,
            "outer_performance_evaluated": False,
            "test_access": {
                "policy": "prohibited",
                "sealed_payload_opened_or_hashed": False,
                "selection_manifest_created": False,
            },
            "artifact_identity": {
                "config_path": str(config_path), "config_sha256": config_sha,
                "data_path": str(data_path), "development_records_sha256": data_sha,
                "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
                "runner_sha256": _sha256(Path(__file__).resolve()),
            },
            "integrity": {
                "manifest": provenance,
                "development_cost": development_cost_audit,
            },
        }
        _atomic_json(output / "results.json", results)
        _atomic_text(output / "RESULTS.md", _results_markdown(results))
        return results

    fit_ids, inner_ids = _split_fit_inner_state_ids(
        records,
        inner_validation_states=int(config["training"]["inner_validation_states"]),
        seed=int(config["collection"]["seed"]),
        identity_by_state=provenance["state_identity_by_id"],
    )
    outer_ids = sorted({str(value) for value in records.state_id[records.split_code == VAL_SPLIT]})
    if set(fit_ids) & set(inner_ids) or set(fit_ids + inner_ids) & set(outer_ids):
        raise RuntimeError("fit/inner/outer state separation failed")
    fit_records = _select_states(records, fit_ids)
    inner_records = _select_states(records, inner_ids)
    train_records = _select_states(records, fit_ids + inner_ids)
    train_cost_audit = _validate_cost(train_records, config)
    prediction_groups, _ = _full_iteration_groups(
        fit_records, iteration=1, population=int(config["cem"]["population"])
    )
    normalization_records = fit_records.subset(np.concatenate(prediction_groups))
    stats = fit_train_zscore_stats(normalization_records)

    device = _resolve_device(args.device)
    torch.set_num_threads(int(config.get("torch_num_threads", 1)))
    models: dict[str, dict[str, GroundedOutcomeModel]] = {}
    training_results: dict[str, Any] = {}
    checkpoint_entries: list[dict[str, Any]] = []
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        seed_key = str(seed)
        models[seed_key] = {}
        training_results[seed_key] = {}
        for variant in VARIANTS:
            model, training = _train_variant(
                variant=variant, seed=seed, fit_records=fit_records,
                inner_records=inner_records, stats=stats, config=config, device=device,
            )
            models[seed_key][variant] = model
            checkpoint_path = output / "checkpoints" / f"seed_{seed}_{variant}.pt"
            _atomic_torch_save(checkpoint_path, {
                "schema_version": 3, "experiment": EXPERIMENT,
                "seed": seed, "variant": variant,
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "normalization": _stats_to_json(stats), "training": training,
                "fit_state_ids": fit_ids, "inner_validation_state_ids": inner_ids,
                "outer_performance_metrics_evaluated": False,
                "config_sha256": config_sha, "records_sha256": data_sha,
                "manifest_sha256": manifest_sha,
            })
            checkpoint_sha = _sha256(checkpoint_path)
            checkpoint_entries.append({
                "seed": seed, "variant": variant, "path": str(checkpoint_path),
                "sha256": checkpoint_sha, "best_epoch": training["best_epoch"],
                "outer_performance_metrics_evaluated": False,
            })
            training_results[seed_key][variant] = training

    freeze = {
        "schema_version": 1, "experiment": EXPERIMENT,
        "frozen_at_utc": _utc_now(), "checkpoints": checkpoint_entries,
        "fit_state_ids": fit_ids, "inner_validation_state_ids": inner_ids,
        "outer_validation_state_ids_sha256": _stable_hash(outer_ids),
        "outer_validation_used_for_fitting_or_checkpoint_selection": False,
        "test_access": "prohibited",
        "config_sha256": config_sha, "records_sha256": data_sha,
        "manifest_sha256": manifest_sha,
    }
    _atomic_json(output / "CHECKPOINTS_FROZEN_BEFORE_OUTER.json", freeze)
    freeze_sha = _sha256(output / "CHECKPOINTS_FROZEN_BEFORE_OUTER.json")

    convergence_models = [
        {
            "seed": int(seed),
            "variant": variant,
            "converged": bool(training_results[str(int(seed))][variant]["converged_by_frozen_patience"]),
            "epochs_completed": int(training_results[str(int(seed))][variant]["epochs_completed"]),
            "best_epoch": int(training_results[str(int(seed))][variant]["best_epoch"]),
        }
        for seed in config["training"]["seeds"]
        for variant in VARIANTS
    ]
    convergence_gate = {
        "definition": "frozen_inner_validation_patience_reached",
        "models": convergence_models,
        "converged_models": int(sum(item["converged"] for item in convergence_models)),
        "required_models": len(convergence_models),
    }
    convergence_gate["pass"] = (
        convergence_gate["converged_models"] == convergence_gate["required_models"]
    )
    source_identity = {
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "core_path": str((Path(__file__).resolve().parents[1] / "src/temporal_tf/mpc_local_grounding.py").resolve()),
        "core_sha256": _sha256((Path(__file__).resolve().parents[1] / "src/temporal_tf/mpc_local_grounding.py").resolve()),
    }
    if not convergence_gate["pass"]:
        results = {
            "schema_version": 1,
            "experiment": EXPERIMENT,
            "status": "terminal_no_go",
            "completed_at_utc": _utc_now(),
            "scope": "low_dimensional_paired_dynamics_mechanism_only",
            "failure_stage": "inner_convergence_precondition",
            "test_access": {
                "policy": "prohibited", "test_outcomes_loaded": False,
                "sealed_payload_opened_or_hashed": False,
                "selection_manifest_created": False,
            },
            "artifact_identity": {
                "config_path": str(config_path), "config_sha256": config_sha,
                "data_path": str(data_path), "records_sha256": data_sha,
                "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
                "checkpoint_freeze_sha256": freeze_sha,
                "source": source_identity,
            },
            "integrity": {
                "manifest": provenance,
                "train_cost_before_checkpoint_freeze": train_cost_audit,
            },
            "state_roles": {
                "fit": fit_ids, "inner_checkpoint_validation": inner_ids,
                "outer_gate_validation_sha256": _stable_hash(outer_ids),
                "outer_used_for_checkpoint_selection": False,
                "outer_performance_evaluated": False,
            },
            "normalization": {
                **_stats_to_json(stats),
                "scope": "fit_states_cem_iteration_1_common_outcome_labels_only",
            },
            "training": training_results,
            "checkpoints_frozen_before_outer": checkpoint_entries,
            "convergence_gate": convergence_gate,
            "problem_existence": {"pass": False, "skipped": "inner convergence precondition failed"},
            "outer_validation": {},
            "outer_gate": {
                "pass": False, "passing_seeds": 0,
                "required_passing_seeds": int(config["gates"]["required_outer_gate_seeds"]),
                "skipped": "inner convergence precondition failed",
            },
            "terminal_policy": {
                "posthoc_remediation_rounds": 0,
                "same_outer_validation_reuse_for_tuning": "forbidden",
            },
        }
        _atomic_json(output / "results.json", results)
        _atomic_text(output / "RESULTS.md", _results_markdown(results))
        return results

    # This is the sole outer-validation evaluation point.  No optimizer,
    # checkpoint, normalization statistic, or variant selection follows it.
    outer_records = _select_states(records, outer_ids)
    outer_cost_audit = _validate_cost(outer_records, config)
    problem = _problem_existence(outer_records, config)
    evaluations: dict[str, Any] = {}
    for seed_key, variants in models.items():
        evaluations[seed_key] = {}
        for variant, model in variants.items():
            evaluations[seed_key][variant] = _evaluate(
                model, outer_records, stats, config, device
            )
    outer_gate = _outer_gates(evaluations, config)
    status = "mechanism_go_no_test" if problem["pass"] and outer_gate["pass"] else "terminal_no_go"
    results = {
        "schema_version": 1, "experiment": EXPERIMENT, "status": status,
        "completed_at_utc": _utc_now(),
        "scope": "low_dimensional_paired_dynamics_mechanism_only",
        "test_access": {
            "policy": "prohibited", "test_outcomes_loaded": False,
            "sealed_payload_opened_or_hashed": False,
            "selection_manifest_created": False,
        },
        "artifact_identity": {
            "config_path": str(config_path), "config_sha256": config_sha,
            "data_path": str(data_path), "records_sha256": data_sha,
            "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
            "checkpoint_freeze_sha256": freeze_sha,
            "source": source_identity,
        },
        "integrity": {
            "manifest": provenance,
            "train_cost_before_checkpoint_freeze": train_cost_audit,
            "outer_cost_after_checkpoint_freeze": outer_cost_audit,
        },
        "state_roles": {
            "fit": fit_ids, "inner_checkpoint_validation": inner_ids,
            "outer_gate_validation": outer_ids,
            "fit_states": len(fit_ids), "inner_states": len(inner_ids),
            "outer_states": len(outer_ids),
            "outer_used_for_checkpoint_selection": False,
        },
        "normalization": {
            **_stats_to_json(stats),
            "scope": "fit_states_cem_iteration_1_common_outcome_labels_only",
        },
        "training": training_results,
        "checkpoints_frozen_before_outer": checkpoint_entries,
        "convergence_gate": convergence_gate,
        "problem_existence": problem,
        "outer_validation": evaluations,
        "outer_gate": outer_gate,
        "terminal_policy": {
            "posthoc_remediation_rounds": 0,
            "same_outer_validation_reuse_for_tuning": "forbidden",
        },
    }
    _atomic_json(output / "results.json", results)
    _atomic_text(output / "RESULTS.md", _results_markdown(results))
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--smoke-preflight-only",
        action="store_true",
        help="validate a non-evidentiary collector smoke without training",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        results = _run(args)
    except Exception as exc:
        print(f"V3 run failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": results["status"],
        "passing_seeds": results.get("outer_gate", {}).get("passing_seeds"),
        "test_outcomes_loaded": False,
    }, indent=2))
    return 0 if results["status"] in {
        "mechanism_go_no_test", "non_evidentiary_smoke_preflight_pass"
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
