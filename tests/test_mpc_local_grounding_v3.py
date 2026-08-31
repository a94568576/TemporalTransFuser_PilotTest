"""Contract tests for the decision-aligned V3 MPC-local grounding pilot."""

from __future__ import annotations

import copy
import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from temporal_tf.mpc_local_grounding import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    MPCGroundingRecords,
    decision_state_macro_metrics,
    statewise_softmin_listwise_loss,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_mpc_local_grounding_pilot_v3.py"


def _load_runner():
    specification = importlib.util.spec_from_file_location("mpc_v3_runner", RUNNER_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError("could not import V3 runner")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _records() -> MPCGroundingRecords:
    state_ids: list[str] = []
    splits: list[int] = []
    iterations: list[int] = []
    actions: list[list[float]] = []
    initial: list[list[float]] = []
    outcomes: list[list[float]] = []
    collisions: list[float] = []
    costs: list[float] = []
    ordinal = 0
    specifications = (
        ("train_0", TRAIN_SPLIT),
        ("train_1", TRAIN_SPLIT),
        ("train_2", TRAIN_SPLIT),
        ("train_3", TRAIN_SPLIT),
        ("val_0", VAL_SPLIT),
        ("test_0", TEST_SPLIT),
    )
    for state_id, split in specifications:
        for iteration in (0, 1, 2):
            for candidate in range(4):
                state_ids.append(state_id)
                splits.append(split)
                iterations.append(iteration)
                actions.append([candidate / 10, 0.2, -candidate / 20, 0.3])
                initial.append([float(split), float(len(state_id))])
                outcomes.append([candidate / 5, 0.01 * candidate, 0.0, 0.0])
                collisions.append(0.0)
                costs.append(float(candidate) + 0.1 * iteration + 0.001 * ordinal)
                ordinal += 1
    return MPCGroundingRecords(
        state_id=np.asarray(state_ids, dtype="U16"),
        split_code=np.asarray(splits, dtype=np.int64),
        cem_iteration=np.asarray(iterations, dtype=np.int64),
        action_params=np.asarray(actions, dtype=np.float32),
        initial_features=np.asarray(initial, dtype=np.float32),
        outcome_features=np.asarray(outcomes, dtype=np.float32),
        collision=np.asarray(collisions, dtype=np.float32),
        real_cost=np.asarray(costs, dtype=np.float32),
    )


class ListwiseLossTest(unittest.TestCase):
    def test_softmin_cross_entropy_is_state_macro_and_decision_aligned(self):
        true = torch.tensor([0.0, 1.0, 2.0, 5.0, 5.5, 6.0])
        good = true.clone().requires_grad_(True)
        bad = (-true).clone().requires_grad_(True)
        groups = (np.asarray([0, 1, 2]), np.asarray([3, 4, 5]))

        good_loss = statewise_softmin_listwise_loss(
            good, true, groups, temperature=0.25
        )
        bad_loss = statewise_softmin_listwise_loss(
            bad, true, groups, temperature=0.25
        )
        self.assertLess(float(good_loss), float(bad_loss))
        good_loss.backward()
        self.assertGreater(float(good.grad.abs().sum()), 0.0)

        per_state = statewise_softmin_listwise_loss(
            good.detach(), true, groups, temperature=0.25, reduction="none"
        )
        self.assertEqual(tuple(per_state.shape), (2,))
        self.assertAlmostEqual(float(per_state.mean()), float(good_loss), places=7)

    def test_softmin_loss_rejects_invalid_temperature_and_groups(self):
        predicted = torch.tensor([0.0, 1.0, 2.0])
        true = predicted.clone()
        with self.assertRaisesRegex(ValueError, "temperature"):
            statewise_softmin_listwise_loss(predicted, true, ([0, 1],), temperature=0)
        with self.assertRaisesRegex(ValueError, "at least two"):
            statewise_softmin_listwise_loss(predicted, true, ([0],), temperature=1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            statewise_softmin_listwise_loss(predicted, true, ([0, 0],), temperature=1)


class DecisionMetricTest(unittest.TestCase):
    def test_exact_argmin_is_deterministic_and_epsilon_regret_is_secondary(self):
        records = _records()
        predicted = np.full(len(records), 100.0, dtype=np.float32)
        val_indices = np.flatnonzero(
            (records.split_code == VAL_SPLIT) & (records.cem_iteration == 2)
        )
        predicted[val_indices] = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        metrics = decision_state_macro_metrics(
            records,
            predicted,
            split_code=VAL_SPLIT,
            cem_iterations=(2,),
            epsilon=0.5,
            predicted_tie_tolerance=0.0,
        )
        self.assertEqual(metrics["states"], 1)
        self.assertEqual(metrics["state_macro_exact_argmin_simple_regret"], 0.0)
        self.assertEqual(metrics["state_macro_epsilon_regret"], 0.0)
        self.assertEqual(metrics["epsilon_success_rate"], 1.0)
        detail = metrics["state_details"][0]
        self.assertEqual(detail["selected_index"], int(val_indices[0]))
        self.assertEqual(detail["predicted_argmin_tie_count"], 1)
        self.assertFalse(detail["deterministic_tie_break_applied"])

        # An exact predicted tie is resolved by the frozen raw-action SHA, not
        # NPZ row order or averaging the tied candidates' real costs.
        predicted[val_indices] = np.asarray([0.0, 0.0, 2.0, 3.0], dtype=np.float32)
        tied = decision_state_macro_metrics(
            records,
            predicted,
            split_code=VAL_SPLIT,
            cem_iterations=(2,),
            epsilon=0.5,
            predicted_tie_tolerance=0.0,
        )
        detail = tied["state_details"][0]
        expected_tie_winner = min(
            val_indices[:2],
            key=lambda index: hashlib.sha256(
                np.asarray(records.action_params[index], dtype="<f4").tobytes()
            ).digest(),
        )
        self.assertEqual(detail["selected_index"], int(expected_tie_winner))
        self.assertEqual(detail["predicted_argmin_tie_count"], 2)
        self.assertTrue(detail["deterministic_tie_break_applied"])

        predicted[val_indices] = np.asarray([3.0, 2.0, 0.0, 1.0], dtype=np.float32)
        worse = decision_state_macro_metrics(
            records,
            predicted,
            split_code="val",
            cem_iterations=(2,),
            epsilon=0.5,
        )
        regret = worse["state_macro_exact_argmin_simple_regret"]
        self.assertGreater(regret, 0.5)
        self.assertAlmostEqual(
            worse["state_macro_epsilon_regret"], regret - 0.5, places=6
        )
        self.assertEqual(worse["epsilon_success_rate"], 0.0)

    def test_decision_metric_keeps_test_explicitly_sealed(self):
        records = _records()
        predicted = np.arange(len(records), dtype=np.float32)
        with self.assertRaisesRegex(PermissionError, "allow_test"):
            decision_state_macro_metrics(
                records,
                predicted,
                split_code=TEST_SPLIT,
                cem_iterations=(2,),
                epsilon=0.1,
            )


class RunnerProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = _load_runner()

    def test_variants_and_test_prohibition_are_hard_coded(self):
        self.assertEqual(
            self.runner.VARIANTS,
            ("prediction_only", "global_listwise", "elite_listwise"),
        )
        self.assertNotIn("open-test", RUNNER_PATH.read_text(encoding="utf-8"))
        config = self.runner._minimal_config_for_tests()
        self.runner._require_config(config)
        config["evaluation"]["test_access"] = "allowed"
        with self.assertRaisesRegex(ValueError, "prohibited"):
            self.runner._require_config(config)

    def test_inner_validation_split_is_hash_deterministic_and_train_only(self):
        records = _records()
        first = self.runner._split_fit_inner_state_ids(
            records, inner_validation_states=2, seed=731
        )
        second = self.runner._split_fit_inner_state_ids(
            records.subset(np.arange(len(records) - 1, -1, -1)),
            inner_validation_states=2,
            seed=731,
        )
        self.assertEqual(first, second)
        fit, inner = first
        self.assertEqual(len(fit), 2)
        self.assertEqual(len(inner), 2)
        self.assertFalse(set(fit) & set(inner))
        self.assertTrue(all(value.startswith("train_") for value in fit + inner))

    def test_development_loader_rejects_any_sealed_test_row(self):
        records = _records()
        arrays = {
            name: np.asarray(getattr(records, name))
            for name in (
                "state_id", "split_code", "cem_iteration", "action_params",
                "initial_features", "outcome_features", "collision", "real_cost",
            )
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root / "combined.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "sealed test rows"):
                self.runner._load_development_records(root / "combined.npz")

            development = records.split_code != TEST_SPLIT
            np.savez(
                root / "development_records.npz",
                **{key: value[development] for key, value in arrays.items()},
            )
            loaded, summary = self.runner._load_development_records(
                root / "development_records.npz"
            )
            self.assertNotIn(TEST_SPLIT, loaded.split_code.tolist())
            self.assertEqual(summary["test"], 0)

    def test_sealed_role_is_rejected_before_payload_access(self):
        config = self.runner._minimal_config_for_tests()
        manifest = {
            "dataset_files": {
                "development_records": {
                    "path": "development_records.npz", "role": "development"
                },
                "sealed_test_records": {
                    "path": "test_records_sealed.npz", "role": "sealed_test"
                },
            }
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            # The sealed file deliberately does not exist: rejection must use
            # only the public role/path string and cannot stat/open/hash it.
            with self.assertRaisesRegex(PermissionError, "prohibited"):
                self.runner._preflight_development_path_role(
                    manifest,
                    manifest_path=manifest_path,
                    data_path=root / "test_records_sealed.npz",
                    config=config,
                )

    def test_manifest_rejects_individual_sealed_test_metadata(self):
        base = {
            "states": [
                {"state_id": 0, "split": "train"},
                {"state_id": 1, "split": "val"},
            ],
            "split_state_ids": {"train": [0], "val": [1]},
            "sealed_test_integrity": {
                "states": 12,
                "records": 864,
                "split_code": 2,
                "states_sha256": "a" * 64,
                "schema_finite_passed": True,
                "reset_passed": True,
                "control_execution_passed": True,
                "individual_state_metadata_redacted": True,
                "sealed_test_stratification_passed": True,
            },
        }
        self.runner._reject_sealed_metadata_exposure(base)

        leaking_ids = copy.deepcopy(base)
        leaking_ids["split_state_ids"]["test"] = [2]
        with self.assertRaisesRegex(ValueError, "sealed-test.*metadata"):
            self.runner._reject_sealed_metadata_exposure(leaking_ids)

        leaking_state = copy.deepcopy(base)
        leaking_state["states"].append(
            {"state_id": 2, "split": "test", "initial_speed_mps": 8.0}
        )
        with self.assertRaisesRegex(ValueError, "sealed-test.*metadata"):
            self.runner._reject_sealed_metadata_exposure(leaking_state)

        leaking_outcome = copy.deepcopy(base)
        leaking_outcome["sealed_test_integrity"]["collision_count"] = 0
        with self.assertRaisesRegex(ValueError, "sealed-test.*metadata"):
            self.runner._reject_sealed_metadata_exposure(leaking_outcome)

    def test_config_requires_enough_epochs_and_outer_val_checkpoint_prohibition(self):
        config = self.runner._minimal_config_for_tests()
        config["training"]["max_epochs"] = config["training"]["min_epochs"] - 1
        with self.assertRaisesRegex(ValueError, "max_epochs"):
            self.runner._require_config(config)

        config = self.runner._minimal_config_for_tests()
        config["training"]["checkpoint_selection_split"] = "outer_validation"
        with self.assertRaisesRegex(ValueError, "inner_validation"):
            self.runner._require_config(config)

    def test_training_smoke_uses_equal_full_iteration_query_exposure(self):
        records = _records()
        fit = self.runner._select_states(records, ["train_0", "train_1"])
        inner = self.runner._select_states(records, ["train_2", "train_3"])
        prediction_groups, _ = self.runner._full_iteration_groups(
            fit, iteration=1, population=4
        )
        from temporal_tf.mpc_local_grounding import fit_train_zscore_stats

        stats = fit_train_zscore_stats(fit.subset(np.concatenate(prediction_groups)))
        config = self.runner._minimal_config_for_tests()
        config["cem"]["population"] = 4
        config["training"].update(
            max_epochs=3,
            min_epochs=1,
            patience=1,
            checkpoint_min_delta=1e-6,
            batch_size=128,
        )
        signatures = set()
        for variant in self.runner.VARIANTS:
            _, audit = self.runner._train_variant(
                variant=variant,
                seed=17,
                fit_records=fit,
                inner_records=inner,
                stats=stats,
                config=config,
                device=torch.device("cpu"),
            )
            signatures.add(audit["architecture"]["sha256"])
            self.assertFalse(audit["outer_records_passed_to_training"])
            exposure = audit["candidate_exposure"]
            self.assertTrue(exposure["global_elite_equal_unique_query_rollouts"])
            self.assertEqual(exposure["fit_global_query"]["candidates_per_state"], 4)
            self.assertEqual(exposure["fit_elite_query"]["candidates_per_state"], 4)
            self.assertFalse(exposure["fit_global_query"]["subsampling"])
        self.assertEqual(len(signatures), 1)


if __name__ == "__main__":
    unittest.main()
