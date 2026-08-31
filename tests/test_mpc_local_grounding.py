"""Contract tests for the pure MPC-local grounding core."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from temporal_tf.mpc_local_grounding import (
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
    GroundedOutcomeModel,
    LatentDynamicsModel,
    MPCGroundingRecords,
    PhysicalCostWeights,
    RankPairs,
    build_within_state_pairs,
    filter_state_groups,
    fit_train_zscore_stats,
    load_mpc_records,
    pairwise_logistic_rank_loss,
    physical_cost,
    state_group_indices,
    state_macro_metrics,
    tie_aware_logistic_rank_loss,
)


def _record_arrays() -> dict[str, np.ndarray]:
    specifications = (
        ("train_a", TRAIN_SPLIT, (0, 1, 2, 2), (0.0, 1.0, 1.0, 2.0), (1.0, 2.0, 3.0)),
        ("train_b", TRAIN_SPLIT, (0, 1, 2, 2), (2.0, 1.0, 0.0, 0.0), (4.0, 5.0, 6.0)),
        ("val_a", VAL_SPLIT, (0, 1, 2), (0.0, 0.5, 1.0), (10.0, 11.0, 12.0)),
        ("val_b", VAL_SPLIT, (0, 1, 2), (1.5, 1.0, 0.5), (13.0, 14.0, 15.0)),
        ("test_a", TEST_SPLIT, (0, 1, 2), (0.0, 1.0, 2.0), (20.0, 21.0, 22.0)),
    )
    state_ids = []
    split_codes = []
    iterations = []
    actions = []
    initial = []
    outcomes = []
    collision = []
    costs = []
    ordinal = 0
    for state_id, split, state_iterations, state_costs, feature in specifications:
        for local_index, (iteration, cost) in enumerate(
            zip(state_iterations, state_costs, strict=True)
        ):
            state_ids.append(state_id)
            split_codes.append(split)
            iterations.append(iteration)
            actions.append(
                [
                    -0.2 + 0.03 * ordinal,
                    0.1 + 0.02 * local_index,
                    -0.1 + 0.01 * ordinal,
                    0.2 - 0.01 * local_index,
                ]
            )
            initial.append(feature)
            outcomes.append(
                [
                    0.1 * ordinal,
                    0.01 * local_index,
                    -0.02 * local_index,
                    0.03 * local_index,
                ]
            )
            collision.append(float(local_index == len(state_iterations) - 1))
            costs.append(cost)
            ordinal += 1
    return {
        "state_id": np.asarray(state_ids, dtype="U16"),
        "split_code": np.asarray(split_codes, dtype=np.int64),
        "cem_iteration": np.asarray(iterations, dtype=np.int64),
        "action_params": np.asarray(actions, dtype=np.float32),
        "initial_features": np.asarray(initial, dtype=np.float32),
        "outcome_features": np.asarray(outcomes, dtype=np.float32),
        "collision": np.asarray(collision, dtype=np.float32),
        "real_cost": np.asarray(costs, dtype=np.float32),
    }


def _records() -> MPCGroundingRecords:
    return MPCGroundingRecords(**_record_arrays())


class NPZContractTest(unittest.TestCase):
    def test_loader_defaults_to_train_val_and_requires_explicit_test_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "records.npz"
            np.savez(path, **_record_arrays())

            development = load_mpc_records(path)
            self.assertEqual(set(development.split_code.tolist()), {TRAIN_SPLIT, VAL_SPLIT})
            self.assertNotIn(TEST_SPLIT, development.split_code)

            with self.assertRaisesRegex(PermissionError, "allow_test"):
                load_mpc_records(path, split_codes=("test",))
            sealed = load_mpc_records(
                path, split_codes=("test",), allow_test=True
            )
            self.assertTrue(np.all(sealed.split_code == TEST_SPLIT))
            self.assertEqual(len(sealed), 3)

    def test_loader_rejects_missing_extra_and_object_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = _record_arrays()

            missing = dict(arrays)
            del missing["collision"]
            np.savez(root / "missing.npz", **missing)
            with self.assertRaisesRegex(ValueError, "missing required"):
                load_mpc_records(root / "missing.npz")

            extra = dict(arrays, oracle_future=np.zeros(len(arrays["state_id"])))
            np.savez(root / "extra.npz", **extra)
            with self.assertRaisesRegex(ValueError, "unexpected"):
                load_mpc_records(root / "extra.npz")

            objects = dict(arrays)
            objects["state_id"] = objects["state_id"].astype(object)
            np.savez(root / "object.npz", **objects)
            with self.assertRaises(ValueError):
                load_mpc_records(root / "object.npz")

    def test_records_reject_shape_finite_split_iteration_and_state_leakage(self):
        mutations = []
        bad_action = _record_arrays()
        bad_action["action_params"] = bad_action["action_params"][:, :3]
        mutations.append((bad_action, "action_params"))

        nonfinite = _record_arrays()
        nonfinite["outcome_features"][0, 0] = np.nan
        mutations.append((nonfinite, "non-finite"))

        bad_split = _record_arrays()
        bad_split["split_code"][0] = 7
        mutations.append((bad_split, "invalid"))

        bad_iteration = _record_arrays()
        bad_iteration["cem_iteration"][0] = 3
        mutations.append((bad_iteration, "invalid"))

        leaking = _record_arrays()
        leaking["split_code"][1] = VAL_SPLIT
        mutations.append((leaking, "crosses split"))

        inconsistent = _record_arrays()
        inconsistent["initial_features"][1, 0] += 1.0
        mutations.append((inconsistent, "inconsistent"))

        bad_collision = _record_arrays()
        bad_collision["collision"][0] = -1.0
        mutations.append((bad_collision, "non-negative"))

        for values, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(
                (ValueError, TypeError), message
            ):
                MPCGroundingRecords(**values)


class StateGroupingAndStatisticsTest(unittest.TestCase):
    def test_filters_are_state_atomic_deterministic_and_test_gated(self):
        records = _records()
        groups = state_group_indices(
            records,
            split_codes=("train",),
            cem_iterations=(2,),
            min_candidates=2,
        )
        self.assertEqual([group.state_id for group in groups], ["train_a", "train_b"])
        self.assertTrue(all(len(group.indices) == 2 for group in groups))
        filtered = filter_state_groups(
            records,
            split_codes=(TRAIN_SPLIT,),
            cem_iterations=(2,),
            min_candidates=2,
        )
        self.assertEqual(filtered.state_id.tolist(), ["train_a"] * 2 + ["train_b"] * 2)
        self.assertTrue(np.all(filtered.cem_iteration == 2))

        reversed_records = records.subset(np.arange(len(records) - 1, -1, -1))
        reversed_groups = state_group_indices(
            reversed_records,
            split_codes=(TRAIN_SPLIT,),
            cem_iterations=(2,),
            min_candidates=2,
        )
        self.assertEqual(
            [group.state_id for group in reversed_groups],
            [group.state_id for group in groups],
        )
        for left_group, right_group in zip(groups, reversed_groups, strict=True):
            np.testing.assert_allclose(
                records.action_params[left_group.indices],
                reversed_records.action_params[right_group.indices],
            )

        with self.assertRaisesRegex(PermissionError, "allow_test"):
            state_group_indices(records, split_codes=(TEST_SPLIT,))
        test_groups = state_group_indices(
            records, split_codes=(TEST_SPLIT,), allow_test=True
        )
        self.assertEqual(len(test_groups), 1)

    def test_zscore_statistics_use_only_train_rows(self):
        records = _records()
        statistics = fit_train_zscore_stats(records)
        train = records.split_code == TRAIN_SPLIT
        np.testing.assert_allclose(
            statistics.state_mean,
            records.initial_features[train].mean(axis=0),
        )
        np.testing.assert_allclose(
            statistics.action_mean,
            records.action_params[train].mean(axis=0),
        )
        self.assertEqual(statistics.train_records, int(train.sum()))

        changed = _record_arrays()
        changed["initial_features"][changed["split_code"] != TRAIN_SPLIT] += 1e6
        changed["action_params"][changed["split_code"] != TRAIN_SPLIT] -= 1e6
        changed["outcome_features"][changed["split_code"] != TRAIN_SPLIT] += 1e6
        changed["real_cost"][changed["split_code"] != TRAIN_SPLIT] += 1e6
        changed_records = MPCGroundingRecords(**changed)
        changed_statistics = fit_train_zscore_stats(changed_records)
        for name in (
            "state_mean",
            "state_std",
            "action_mean",
            "action_std",
            "outcome_mean",
            "outcome_std",
        ):
            np.testing.assert_allclose(
                getattr(statistics, name), getattr(changed_statistics, name)
            )
        self.assertEqual(statistics.cost_mean, changed_statistics.cost_mean)
        self.assertEqual(statistics.cost_std, changed_statistics.cost_std)

        normalized = statistics.normalize_action(records.action_params[train])
        np.testing.assert_allclose(normalized.mean(axis=0), np.zeros(4), atol=1e-6)


class DynamicsAndCostTest(unittest.TestCase):
    def test_models_have_exact_shapes_aliases_and_gradients(self):
        for model in (
            LatentDynamicsModel(state_dim=3),
            GroundedOutcomeModel(initial_dim=3),
        ):
            with self.subTest(model=type(model).__name__):
                state = torch.randn(5, 3)
                action = torch.randn(5, 4)
                output = model(state, action)
                self.assertEqual(output["state_latent"].shape, (5, 16))
                self.assertEqual(output["latent_delta"].shape, (5, 16))
                self.assertEqual(output["outcome"].shape, (5, 4))
                self.assertEqual(output["inverse_action"].shape, (5, 4))
                self.assertIs(output["outcome"], output["predicted_outcome"])
                self.assertIs(output["latent_delta"], output["delta_z"])
                self.assertIs(output["inverse_action"], output["reconstructed_action"])
                loss = output["outcome"].square().mean() + (
                    output["inverse_action"] - action
                ).square().mean()
                loss.backward()
                for name in (
                    "state_encoder",
                    "action_encoder",
                    "transition",
                    "outcome_decoder",
                    "inverse_head",
                ):
                    gradients = [
                        parameter.grad
                        for parameter in getattr(model, name).parameters()
                        if parameter.grad is not None
                    ]
                    self.assertTrue(gradients, name)
                    self.assertGreater(sum(value.abs().sum().item() for value in gradients), 0.0)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            GroundedOutcomeModel(3)(
                torch.tensor([[float("nan"), 0.0, 0.0]]), torch.zeros(1, 4)
            )

    def test_physical_cost_matches_authoritative_formula_and_is_differentiable(self):
        outcome = torch.tensor([[2.0, 0.5, -0.25, 0.1]], requires_grad=True)
        action = torch.tensor([[0.2, 0.3, -0.1, -0.2]], requires_grad=True)
        collision = torch.tensor([1.0])
        config = {
            "progress_weight": -0.20,
            "lateral_squared_weight": 1.50,
            "yaw_squared_weight": 0.80,
            "speed_squared_weight": 0.40,
            "steering_squared_weight": 0.02,
            "longitudinal_squared_weight": 0.01,
            "collision_weight": 10.0,
            "pair_tie_threshold": 0.005,
        }
        value = physical_cost(outcome, action, collision, config)
        expected = (
            -0.20 * 2.0
            + 1.50 * 0.5**2
            + 0.80 * 0.25**2
            + 0.40 * 0.1**2
            + 0.02 * ((0.2**2 + 0.1**2) / 2.0)
            + 0.01 * ((0.3**2 + 0.2**2) / 2.0)
            + 10.0
        )
        self.assertAlmostEqual(float(value), expected, places=6)
        value.sum().backward()
        self.assertGreater(float(outcome.grad.abs().sum()), 0.0)
        self.assertGreater(float(action.grad.abs().sum()), 0.0)

        defaults = physical_cost(
            outcome.detach(), action.detach(), collision, PhysicalCostWeights()
        )
        torch.testing.assert_close(defaults, value.detach())
        with self.assertRaisesRegex(ValueError, "non-negative"):
            physical_cost(outcome.detach(), action.detach(), -collision, config)


class PairAndRankLossTest(unittest.TestCase):
    def test_pair_builder_is_seeded_tie_aware_per_state_and_never_crosses_state(self):
        records = _records()
        first = build_within_state_pairs(
            records,
            budget_per_state=3,
            seed=17,
            split_code=TRAIN_SPLIT,
        )
        second = build_within_state_pairs(
            records,
            budget_per_state=3,
            seed=17,
            split_code=TRAIN_SPLIT,
        )
        np.testing.assert_array_equal(first.left_indices, second.left_indices)
        np.testing.assert_array_equal(first.right_indices, second.right_indices)
        np.testing.assert_array_equal(first.targets, second.targets)
        self.assertEqual(len(first), 6)
        self.assertEqual(first.state_ids.count("train_a"), 3)
        self.assertEqual(first.state_ids.count("train_b"), 3)
        for left, right, state_id in zip(
            first.left_indices, first.right_indices, first.state_ids, strict=True
        ):
            self.assertEqual(records.state_id[left], records.state_id[right])
            self.assertEqual(records.state_id[left], state_id)

        all_pairs = build_within_state_pairs(
            records,
            budget_per_state=64,
            seed=29,
            split_code="train",
        )
        self.assertEqual(len(all_pairs), 12)
        self.assertIn(0.5, all_pairs.targets)

        with self.assertRaisesRegex(PermissionError, "allow_test"):
            build_within_state_pairs(
                records, budget_per_state=2, seed=1, split_code="test"
            )

    def test_pairwise_and_direct_tie_aware_logistic_losses(self):
        pairs = RankPairs(
            left_indices=np.asarray([0, 1]),
            right_indices=np.asarray([1, 2]),
            targets=np.asarray([1.0, 0.5], dtype=np.float32),
            state_ids=("s", "s"),
        )
        correct = torch.tensor([0.0, 2.0, 2.0], requires_grad=True)
        wrong = torch.tensor([2.0, 0.0, 4.0])
        correct_loss = pairwise_logistic_rank_loss(correct, pairs)
        wrong_loss = pairwise_logistic_rank_loss(wrong, pairs)
        self.assertLess(float(correct_loss), float(wrong_loss))
        correct_loss.backward()
        self.assertGreater(float(correct.grad.abs().sum()), 0.0)

        true_i = torch.tensor([0.0, 1.0])
        true_j = torch.tensor([1.0, 1.001])
        direct_good = tie_aware_logistic_rank_loss(
            torch.tensor([0.0, 2.0]),
            torch.tensor([2.0, 2.0]),
            true_i,
            true_j,
            0.005,
        )
        direct_bad = tie_aware_logistic_rank_loss(
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, 3.0]),
            true_i,
            true_j,
            0.005,
        )
        self.assertLess(float(direct_good), float(direct_bad))

    def test_tie_aware_logistic_temperature_matches_exact_bce(self):
        pred_i = torch.tensor([0.010, 0.004, -0.003], dtype=torch.float64)
        pred_j = torch.tensor([0.000, 0.009, 0.002], dtype=torch.float64)
        true_i = torch.tensor([0.0, 1.000, 2.0], dtype=torch.float64)
        true_j = torch.tensor([1.0, 1.001, 1.0], dtype=torch.float64)
        temperature = 0.005

        actual = tie_aware_logistic_rank_loss(
            pred_i,
            pred_j,
            true_i,
            true_j,
            tie_threshold=0.005,
            temperature=temperature,
            reduction="none",
        )
        target = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            (pred_j - pred_i) / temperature,
            target,
            reduction="none",
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        default = tie_aware_logistic_rank_loss(
            pred_i,
            pred_j,
            true_i,
            true_j,
            tie_threshold=0.005,
            reduction="none",
        )
        explicit_v1_v2 = tie_aware_logistic_rank_loss(
            pred_i,
            pred_j,
            true_i,
            true_j,
            tie_threshold=0.005,
            temperature=1.0,
            reduction="none",
        )
        torch.testing.assert_close(default, explicit_v1_v2, rtol=0.0, atol=0.0)

    def test_tie_aware_logistic_temperature_must_be_finite_and_positive(self):
        costs = torch.tensor([0.0, 1.0])
        for temperature in (
            0.0,
            -0.1,
            float("nan"),
            float("inf"),
            -float("inf"),
            None,
            "invalid",
        ):
            with self.subTest(temperature=temperature), self.assertRaisesRegex(
                ValueError, "temperature must be finite and positive"
            ):
                tie_aware_logistic_rank_loss(
                    costs,
                    costs,
                    costs,
                    costs,
                    tie_threshold=0.005,
                    temperature=temperature,
                )


class StateMacroMetricsTest(unittest.TestCase):
    def _metric_records(self) -> MPCGroundingRecords:
        state_id = np.asarray(["a", "a", "a", "b", "b"])
        return MPCGroundingRecords(
            state_id=state_id,
            split_code=np.full(5, VAL_SPLIT, dtype=np.int64),
            cem_iteration=np.full(5, 2, dtype=np.int64),
            action_params=np.zeros((5, 4), dtype=np.float32),
            initial_features=np.asarray(
                [[0.0, 0.0]] * 3 + [[1.0, 1.0]] * 2, dtype=np.float32
            ),
            outcome_features=np.zeros((5, 4), dtype=np.float32),
            collision=np.zeros(5, dtype=np.float32),
            real_cost=np.asarray([0.0, 1.0, 1.0, 0.0, 2.0], dtype=np.float32),
        )

    def test_state_macro_metrics_handle_ties_and_do_not_micro_weight_states(self):
        records = self._metric_records()
        predicted_cost = np.asarray([0.0, 1.0, 1.0, 2.0, 0.0], dtype=np.float32)
        predicted_outcome = np.zeros((5, 4), dtype=np.float32)
        predicted_outcome[3:] = 1.0

        metrics = state_macro_metrics(
            records,
            predicted_cost,
            predicted_outcome,
            split_code=VAL_SPLIT,
            cem_iterations=(2,),
        )

        self.assertAlmostEqual(metrics["state_macro_spearman"], 0.0, places=6)
        self.assertAlmostEqual(metrics["state_macro_tie_accuracy"], 0.5, places=6)
        self.assertAlmostEqual(metrics["state_macro_selection_regret"], 1.0, places=6)
        self.assertAlmostEqual(metrics["state_macro_outcome_mse"], 0.5, places=6)
        self.assertEqual(metrics["states"], 2)
        self.assertEqual(metrics["pairs"], 4)

    def test_constant_costs_do_not_count_as_perfect_rank_agreement(self):
        records = self._metric_records()
        metrics = state_macro_metrics(
            records,
            np.zeros(5, dtype=np.float32),
            np.zeros((5, 4), dtype=np.float32),
            split_code=VAL_SPLIT,
            cem_iterations=(2,),
        )
        self.assertEqual(metrics["state_macro_spearman"], 0.0)

    def test_state_macro_metrics_enforce_test_gate_and_finite_predictions(self):
        records = _records()
        costs = records.real_cost.copy()
        outcomes = records.outcome_features.copy()
        with self.assertRaisesRegex(PermissionError, "allow_test"):
            state_macro_metrics(
                records, costs, outcomes, split_code=TEST_SPLIT
            )
        metrics = state_macro_metrics(
            records,
            costs,
            outcomes,
            split_code=TEST_SPLIT,
            allow_test=True,
        )
        self.assertAlmostEqual(metrics["state_macro_spearman"], 1.0)

        invalid = costs.copy()
        invalid[0] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            state_macro_metrics(
                records, invalid, outcomes, split_code=VAL_SPLIT
            )


if __name__ == "__main__":
    unittest.main()
