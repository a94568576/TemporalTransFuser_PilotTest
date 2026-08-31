"""Focused provenance regressions for the MPC-local grounding runner."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np

from temporal_tf.mpc_local_grounding import MPCGroundingRecords, TRAIN_SPLIT


def _load_runner() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_mpc_local_grounding_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_mpc_local_grounding_pilot_manifest_test", path
    )
    if spec is None or spec.loader is None:  # pragma: no cover - import invariant
        raise RuntimeError(f"cannot import runner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records() -> MPCGroundingRecords:
    size = 6
    return MPCGroundingRecords(
        state_id=np.asarray(["train_0"] * size, dtype="U16"),
        split_code=np.full(size, TRAIN_SPLIT, dtype=np.int64),
        cem_iteration=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        action_params=np.asarray(
            [
                [-0.10, 0.20, -0.05, 0.30],
                [0.10, 0.10, 0.05, 0.20],
                [-0.08, 0.25, -0.04, 0.35],
                [0.08, 0.15, 0.04, 0.25],
                [-0.06, 0.30, -0.03, 0.40],
                [0.06, 0.20, 0.03, 0.30],
            ],
            dtype=np.float32,
        ),
        initial_features=np.tile(
            np.asarray([[0.0, 0.0, 0.75, 0.9, 0.1, 0.1, 0.1]], dtype=np.float32),
            (size, 1),
        ),
        outcome_features=np.zeros((size, 4), dtype=np.float32),
        collision=np.zeros(size, dtype=np.float32),
        real_cost=np.linspace(0.0, 0.05, size, dtype=np.float32),
    )


def _config(*, parent_records_sha256: str, parent_states_sha256: str) -> dict:
    return {
        "collection": {
            "carla_version": "0.9.15",
            "map": "/Game/Carla/Maps/Town10HD_Opt",
            "fixed_delta_seconds": 0.05,
            "horizon_ticks": 20,
            "vehicle_filter": "vehicle.tesla.model3",
            "no_rendering": True,
            "action_profile": "safe_local_v2",
            "seed": 27031,
            "states": {"train": 1, "val": 0, "test": 0},
            "initial_speeds_mps": [4.0, 6.0, 8.0],
            "target_speed_mps": 8.0,
            "minimum_forward_road_m": 25.0,
            "exclude_parent_v1_spawn_states": True,
            "parent_v1_states_sha256": parent_states_sha256,
            "parent_v1_records_sha256": parent_records_sha256,
        },
        "cem": {
            "iterations": 3,
            "population": 2,
            "elite_count": 1,
            "initial_mean": [0.0, 0.4, 0.0, 0.4],
            "initial_std": [0.12, 0.45, 0.12, 0.45],
            "lower": [-0.2, -1.0, -0.2, -1.0],
            "upper": [0.2, 1.0, 0.2, 1.0],
            "minimum_std": [0.015, 0.05, 0.015, 0.05],
        },
        "cost": {
            "progress_weight": -0.2,
            "lateral_squared_weight": 1.5,
            "yaw_squared_weight": 0.8,
            "speed_squared_weight": 0.4,
            "steering_squared_weight": 0.02,
            "longitudinal_squared_weight": 0.01,
            "collision_weight": 10.0,
            "pair_tie_threshold": 0.005,
        },
    }


def _manifest(
    *, current_records_sha256: str, parent_records_sha256: str, parent_states_sha256: str
) -> dict:
    reset_tolerances = {
        "position_m": 1.0e-4,
        "rotation_rad": 1.0e-5,
        "physics_disabled_speed_mps": 1.0e-4,
        "angular_speed_rad_s": 1.0e-4,
    }
    paired_report = {
        "state_id": "train_0",
        "passed": True,
        "max_position_delta_m": 0.0,
        "max_rotation_delta_rad": 0.0,
        "max_speed_delta_mps": 0.0,
        "max_initial_feature_delta": 0.0,
    }
    return {
        "schema_version": 1,
        "dataset_schema": "mpc-local-carla-v1",
        "status": "complete",
        "action_profile": "safe_local_v2",
        "records_sha256": current_records_sha256,
        "files": {"records.npz": {"sha256": current_records_sha256}},
        "protocol": {
            "smoke": False,
            "seed": 27031,
            "action_profile": "safe_local_v2",
            "carla_version": "0.9.15",
            "map": "/Game/Carla/Maps/Town10HD_Opt",
            "fixed_delta_seconds": 0.05,
            "horizon_ticks": 20,
            "vehicle_blueprint": "vehicle.tesla.model3",
            "initial_speeds_mps": [4.0, 6.0, 8.0],
            "target_speed_mps": 8.0,
            "minimum_forward_non_junction_road_m": 25.0,
            "longitudinal_mapping": "positive=throttle; negative=brake",
            "action_parameterization": (
                "[steer1,longitudinal1,steer2,longitudinal2]"
            ),
            "split_counts": {"train": 1, "val": 0, "test": 0},
            "state_selection": {
                "excluded_source_spawn_indices": [7],
                "fresh_relative_to_parent": True,
                "excluded_parent": {
                    "dataset_schema": "mpc-local-carla-v1",
                    "states_sha256": parent_states_sha256,
                    "records_sha256": parent_records_sha256,
                },
            },
            "cem": {
                "iterations": 3,
                "population": 2,
                "elite_count": 1,
                "initial_mean": [0.0, 0.4, 0.0, 0.4],
                "initial_std": [0.12, 0.45, 0.12, 0.45],
                "lower": [-0.2, -1.0, -0.2, -1.0],
                "upper": [0.2, 1.0, 0.2, 1.0],
                "minimum_std": [0.015, 0.05, 0.015, 0.05],
            },
            "cost": {
                "weights": {
                    "progress": -0.2,
                    "lateral_squared": 1.5,
                    "yaw_squared": 0.8,
                    "speed_squared": 0.4,
                    "steering_mean_squared": 0.02,
                    "longitudinal_mean_squared": 0.01,
                    "collision": 10.0,
                },
                "pair_tie_threshold": 0.005,
            },
            "reset": {"tolerances": reset_tolerances},
            "control_execution": {"max_abs_error_tolerance": 1.0e-6},
        },
        "server_and_map": {
            "client_version": "0.9.15",
            "server_version": "0.9.15",
            "map_name": "/Game/Carla/Maps/Town10HD_Opt",
            "collection_settings": {
                "synchronous_mode": True,
                "no_rendering_mode": True,
                "fixed_delta_seconds": 0.05,
            },
        },
        "outcome_source": "CARLA simulator paired rollout",
        "same_state_reset_passed": True,
        "states": [{"state_id": "train_0", "source_spawn_index": 99}],
        "fresh_state_attestation": {
            "excluded_source_spawn_indices": [7],
            "selected_source_spawn_indices": [99],
            "overlap": [],
            "passed": True,
        },
        "collection_summary": {
            "reset_probe": {
                "bitwise_equal": True,
                "fresh_actor_ids_unique": True,
                "repeats": 2,
            },
            "reset_observed_max": {key: 0.0 for key in reset_tolerances},
            "control_execution_audit_passed": True,
            "control_execution_max_abs_error": 0.0,
            "paired_initial_state_passed": True,
            "paired_initial_state_tolerance": 1.0e-6,
            "paired_initial_state": [paired_report],
            "initial_velocity_command_audit_passed": True,
            "initial_velocity_command_max_abs_error": 0.0,
        },
        "cleanup": {
            "settings_restored": True,
            "actors_remaining": [],
            "errors": [],
        },
        "control_execution_audit_passed": True,
        "control_execution_max_abs_error": 0.0,
        "initial_velocity_command_audit_passed": True,
        "split_state_ids": {"train": ["train_0"], "val": [], "test": []},
    }


class RemediationParentHashScopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()

    def test_current_records_hash_coexists_with_distinct_parent_hash(self) -> None:
        current_hash = "a" * 64
        parent_hash = "b" * 64
        parent_states_hash = "c" * 64
        records = _records()
        config = _config(
            parent_records_sha256=parent_hash,
            parent_states_sha256=parent_states_hash,
        )
        manifest = _manifest(
            current_records_sha256=current_hash,
            parent_records_sha256=parent_hash,
            parent_states_sha256=parent_states_hash,
        )

        provenance = self.runner._validate_manifest(
            manifest,
            data_path=Path("records.npz"),
            data_sha256=current_hash,
            config=config,
            config_sha256="d" * 64,
            records=records,
            requested_split="train",
            allow_smoke_data=False,
        )

        self.assertEqual(provenance["sha256"], current_hash)
        self.assertEqual(
            provenance["fresh_state_attestation"]["parent"]["records_sha256"],
            parent_hash,
        )

        wrong_current = copy.deepcopy(manifest)
        wrong_current["records_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "records NPZ SHA-256 differs"):
            self.runner._validate_manifest(
                wrong_current,
                data_path=Path("records.npz"),
                data_sha256=current_hash,
                config=config,
                config_sha256="d" * 64,
                records=records,
                requested_split="train",
                allow_smoke_data=False,
            )


class RankScaleDiagnosticConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner()
        cls.config_root = Path(__file__).resolve().parents[1] / "configs"

    def test_default_reproduction_and_diagnostic_temperature_contracts(self) -> None:
        official = self.runner._load_mapping(
            self.config_root / "mpc_local_grounding_pilot_v2.yaml",
            kind="official V2 config",
        )
        diagnostic = self.runner._load_mapping(
            self.config_root / "mpc_local_grounding_pilot_v2_rankscale_diagnostic.yaml",
            kind="rank-scale diagnostic config",
        )

        self.runner._require_config(official)
        self.runner._require_config(diagnostic)
        self.assertEqual(self.runner._rank_logit_temperature(official), 1.0)
        self.assertEqual(self.runner._rank_logit_temperature(diagnostic), 0.005)
        self.assertFalse(self.runner._diagnostic_only_no_test(official))
        self.assertTrue(self.runner._diagnostic_only_no_test(diagnostic))
        self.assertTrue(self.runner._selection_manifest_allowed(official))
        self.assertFalse(self.runner._selection_manifest_allowed(diagnostic))

    def test_scaled_temperature_cannot_enter_official_or_mismatch_tie_scale(self) -> None:
        official = self.runner._load_mapping(
            self.config_root / "mpc_local_grounding_pilot_v2.yaml",
            kind="official V2 config",
        )
        official["training"]["rank_logit_temperature"] = 0.005
        with self.assertRaisesRegex(ValueError, "non-diagnostic V1/V2"):
            self.runner._require_config(official)

        diagnostic = self.runner._load_mapping(
            self.config_root / "mpc_local_grounding_pilot_v2_rankscale_diagnostic.yaml",
            kind="rank-scale diagnostic config",
        )
        diagnostic["training"]["rank_logit_temperature"] = 0.01
        with self.assertRaisesRegex(ValueError, "must equal cost.pair_tie_threshold"):
            self.runner._require_config(diagnostic)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
