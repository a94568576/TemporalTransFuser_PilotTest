"""Contract tests for the counterfactual action-influence pilot."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch

from temporal_tf.action_influence import (
    ActionFiLMResidualAdapter,
    ActionTransitionDataset,
    PassiveFuturePredictor,
    build_action_candidates,
    center_action_residuals,
    denormalize_latent,
    deterministic_derangement,
    normalize_actions,
    normalize_latent,
    train_normalization_statistics,
)
from temporal_tf.bev_geometry import warp_bev_to_current
from temporal_tf.cache import CacheWriter, SCHEMA_VERSION, load_index


def _cache_record(
    route_id: str, route_index: int, frame_id: int, *, spatial_size: int
) -> dict:
    value = float(route_index * 10 + frame_id)
    trajectory = torch.tensor(
        [[value + 1.0, 0.0], [value + 2.0, 0.25]], dtype=torch.float32
    )
    return {
        "bev_feature": torch.full(
            (4, spatial_size, spatial_size), value, dtype=torch.float32
        ),
        "pred_trajectory": trajectory,
        "gt_trajectory": trajectory + 0.5,
        "ego_pose": torch.tensor([value, 0.0, 0.0], dtype=torch.float64),
        "route_id": route_id,
        "frame_id": frame_id,
        "timestamp": frame_id * 0.25,
        "trajectory_source": "frozen_model_prediction",
    }


def _measurement(route_index: int, frame_id: int) -> dict:
    return {
        "steer": route_index * 0.01 + frame_id * 0.1,
        "throttle": 0.2 + frame_id * 0.05,
        "brake": frame_id == 0,
        "control_brake": frame_id % 2 == 1,
        "speed": 3.0 + frame_id,
    }


def _write_measurement(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)


def _make_transition_fixture(
    cache_root: Path, measurements_root: Path, *, spatial_size: int = 8
) -> dict:
    """Create one route per split, each with exactly one contiguous H=2 pair."""

    writer = CacheWriter(cache_root)
    route_indices = {f"collection_{index}/route_{index}": index for index in range(3)}
    # 0 -> 2 is valid for H=2.  A naive endpoint-only implementation would
    # incorrectly also admit 2 -> 4 despite the missing frame 3.
    frame_ids = (0, 1, 2, 4)
    for route_id, route_index in route_indices.items():
        for frame_id in frame_ids:
            writer.add(
                _cache_record(
                    route_id, route_index, frame_id, spatial_size=spatial_size
                )
            )
            _write_measurement(
                measurements_root
                / Path(route_id).name
                / "measurements"
                / f"{frame_id:04d}.json.gz",
                _measurement(route_index, frame_id),
            )
    writer.finalize(
        source={
            "raw_sources": [
                {
                    "route_directory_paths": [
                        str(measurements_root / Path(route_id).name)
                        for route_id in route_indices
                    ]
                }
            ]
        },
        split_ratios={"train": 1.0, "val": 1.0, "test": 1.0},
        split_seed=11,
    )
    index = load_index(cache_root)
    if index["schema_version"] != 3 or SCHEMA_VERSION != 3:
        raise AssertionError("the action-transition fixture must use cache schema v3")
    return {
        entry["split"]: (entry["route_id"], route_indices[entry["route_id"]])
        for entry in index["records"]
    }


class _ExactStatisticsDataset:
    """Small dataset with hand-computable channel and action statistics."""

    def __init__(self, split: str = "train") -> None:
        self.split = split
        self._samples = [
            {
                "future_latent": torch.tensor(
                    [[[1.0, 3.0]], [[2.0, 4.0]]], dtype=torch.float32
                )
            },
            {
                "future_latent": torch.tensor(
                    [[[5.0, 7.0]], [[6.0, 8.0]]], dtype=torch.float32
                )
            },
        ]
        self._actions = torch.tensor(
            [
                [[0.0, 0.2, 0.0], [0.2, 0.4, 1.0]],
                [[0.4, 0.6, 0.0], [0.6, 0.8, 1.0]],
            ],
            dtype=torch.float32,
        )

    def __iter__(self):
        return iter(self._samples)

    @property
    def actions(self) -> torch.Tensor:
        return self._actions


class ActionInfluenceGeometryAndNormalizationTest(unittest.TestCase):
    def test_native_source_to_future_warp_has_correct_forward_motion_direction(self):
        source = torch.zeros(1, 2, 64, 64)
        # Width is ego x.  The ramp exposes sampling direction while the marker
        # makes the one-cell displacement visually unambiguous.
        source[0, 0] = torch.arange(64, dtype=torch.float32).view(1, 64)
        source[0, 1, 32, 40] = 1.0
        source_pose = torch.tensor([[0.0, 0.0, 0.0]])
        target_pose = torch.tensor([[1.0, 0.0, 0.0]])

        in_target, target_validity = warp_bev_to_current(
            source, source_pose, target_pose
        )
        in_reverse, reverse_validity = warp_bev_to_current(
            source, target_pose, source_pose
        )

        # When ego advances +1 m, a static source marker has target-frame x one
        # cell smaller.  Reversing the pose arguments moves it the other way.
        self.assertAlmostEqual(float(in_target[0, 1, 32, 39]), 1.0, places=5)
        self.assertAlmostEqual(float(in_reverse[0, 1, 32, 41]), 1.0, places=5)
        self.assertAlmostEqual(float(in_target[0, 0, 32, 20]), 21.0, places=5)
        self.assertAlmostEqual(float(in_reverse[0, 0, 32, 20]), 19.0, places=5)
        self.assertFalse(torch.equal(in_target, in_reverse))

        self.assertEqual(target_validity.shape, (1, 1, 64, 64))
        self.assertTrue(torch.all((target_validity >= 0.0) & (target_validity <= 1.0)))
        self.assertTrue(torch.all(target_validity[0, 0, :, -1] == 0.0))
        self.assertTrue(torch.all(target_validity[0, 0, :, 0] == 1.0))
        self.assertTrue(torch.all(reverse_validity[0, 0, :, 0] == 0.0))
        self.assertTrue(torch.all(reverse_validity[0, 0, :, -1] == 1.0))

    def test_train_normalization_statistics_exact_values_and_round_trip(self):
        dataset = _ExactStatisticsDataset(split="train")

        statistics = train_normalization_statistics(dataset)

        root_five = torch.sqrt(torch.tensor(5.0))
        torch.testing.assert_close(
            statistics["latent_mean"], torch.tensor([4.0, 5.0])
        )
        torch.testing.assert_close(
            statistics["latent_std"], torch.tensor([root_five, root_five])
        )
        torch.testing.assert_close(
            statistics["action_mean"], torch.tensor([0.3, 0.5, 0.5])
        )
        torch.testing.assert_close(
            statistics["action_std"],
            torch.tensor([0.05**0.5, 0.05**0.5, 0.5]),
        )

        latent = torch.stack(
            [sample["future_latent"] for sample in dataset._samples]
        )
        normalized = normalize_latent(latent, statistics)
        torch.testing.assert_close(
            denormalize_latent(normalized, statistics), latent
        )

        raw_literal_zero = torch.zeros(2, 4, 3)
        normalized_zero = normalize_actions(raw_literal_zero, statistics)
        expected_zero = -statistics["action_mean"] / statistics["action_std"]
        torch.testing.assert_close(
            normalized_zero,
            expected_zero.reshape(1, 1, 3).expand_as(normalized_zero),
        )

    def test_normalization_statistics_reject_validation_split(self):
        with self.assertRaisesRegex(ValueError, "train split"):
            train_normalization_statistics(_ExactStatisticsDataset(split="val"))


class ActionCandidateTest(unittest.TestCase):
    def test_derangement_is_seeded_long_permutation_without_fixed_points(self):
        for size in (2, 3, 17):
            with self.subTest(size=size):
                first = deterministic_derangement(size, seed=29)
                second = deterministic_derangement(size, seed=29)
                self.assertEqual(first.dtype, torch.long)
                self.assertEqual(first.shape, (size,))
                torch.testing.assert_close(first, second)
                torch.testing.assert_close(first.sort().values, torch.arange(size))
                self.assertFalse(torch.any(first == torch.arange(size)))

    def test_derangement_rejects_impossible_sizes(self):
        for size in (-1, 0, 1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                deterministic_derangement(size, seed=0)

    def test_candidate_construction_has_six_exact_controls(self):
        actions = torch.arange(4 * 3 * 3, dtype=torch.float32).reshape(4, 3, 3)
        shuffled = torch.tensor([1, 0, 3, 2], dtype=torch.long)
        other = torch.tensor([2, 3, 0, 1], dtype=torch.long)

        candidates, names = build_action_candidates(actions, shuffled, other)

        self.assertEqual(
            names, ("true", "shuffled", "zero", "hold", "reverse", "other")
        )
        self.assertEqual(candidates.shape, (4, 6, 3, 3))
        self.assertEqual(candidates.dtype, actions.dtype)
        torch.testing.assert_close(candidates[:, 0], actions)
        torch.testing.assert_close(candidates[:, 1], actions[shuffled])
        torch.testing.assert_close(candidates[:, 2], torch.zeros_like(actions))
        torch.testing.assert_close(
            candidates[:, 3], actions[:, :1].expand_as(actions)
        )
        torch.testing.assert_close(candidates[:, 4], actions.flip(1))
        torch.testing.assert_close(candidates[:, 5], actions[other])

    def test_candidate_construction_rejects_bad_shapes_and_indices(self):
        valid = torch.zeros(3, 4, 3)
        indices = torch.tensor([1, 2, 0], dtype=torch.long)
        invalid_calls = (
            (torch.zeros(3, 4, 2), indices, indices),
            (valid, torch.tensor([1, 0]), indices),
            (valid, indices.float(), indices),
            (valid, indices, torch.tensor([1, 2, 3])),
        )
        for arguments in invalid_calls:
            with self.subTest(shapes=[tuple(item.shape) for item in arguments]):
                with self.assertRaises((ValueError, TypeError, IndexError)):
                    build_action_candidates(*arguments)

    def test_centered_residuals_have_zero_candidate_mean_and_keep_gradients(self):
        raw = torch.randn(2, 6, 4, 3, 5, dtype=torch.float64, requires_grad=True)
        centered = center_action_residuals(raw)

        self.assertEqual(centered.shape, raw.shape)
        torch.testing.assert_close(centered, raw - raw.mean(dim=1, keepdim=True))
        torch.testing.assert_close(
            centered.mean(dim=1),
            torch.zeros_like(centered[:, 0]),
            atol=1e-12,
            rtol=0.0,
        )
        centered.square().sum().backward()
        self.assertIsNotNone(raw.grad)

    def test_centering_rejects_non_bkchw_input(self):
        with self.assertRaises(ValueError):
            center_action_residuals(torch.zeros(2, 6, 4, 5))


class ActionInfluenceModelTest(unittest.TestCase):
    def test_passive_predictor_preserves_bchw_shape(self):
        predictor = PassiveFuturePredictor(channels=4, hidden_channels=7)
        current = torch.randn(3, 4, 8, 6, requires_grad=True)

        future = predictor(current)

        self.assertEqual(future.shape, current.shape)
        self.assertEqual(future.dtype, current.dtype)
        self.assertTrue(torch.isfinite(future).all())
        future.square().mean().backward()
        self.assertIsNotNone(current.grad)
        self.assertTrue(any(parameter.grad is not None for parameter in predictor.parameters()))

    def test_passive_predictor_rejects_bad_input_shapes(self):
        predictor = PassiveFuturePredictor(channels=4, hidden_channels=7)
        for tensor in (torch.zeros(2, 4, 8), torch.zeros(2, 5, 8, 8)):
            with self.subTest(shape=tuple(tensor.shape)):
                with self.assertRaises((ValueError, RuntimeError)):
                    predictor(tensor)

    def test_centered_film_adapter_shapes_equation_and_frozen_base(self):
        batch, candidates_count, channels = 3, 6, 4
        action_horizon, height, width = 3, 5, 7
        current = torch.randn(batch, channels, height, width, requires_grad=True)

        passive = PassiveFuturePredictor(channels=channels, hidden_channels=8)
        passive.requires_grad_(False)
        base_future = passive(current).detach()
        self.assertFalse(base_future.requires_grad)

        action_candidates = torch.randn(
            batch, candidates_count, action_horizon, 3
        )
        adapter = ActionFiLMResidualAdapter(
            channels=channels,
            action_horizon=action_horizon,
            hidden_channels=8,
            action_hidden_dim=11,
            centered=True,
        )

        output = adapter(current, base_future, action_candidates)

        self.assertIsInstance(output, dict)
        self.assertTrue({"prediction", "residual", "gate"}.issubset(output))
        self.assertEqual(
            output["prediction"].shape,
            (batch, candidates_count, channels, height, width),
        )
        self.assertEqual(output["residual"].shape, output["prediction"].shape)
        self.assertIsInstance(output["gate"], torch.Tensor)
        torch.testing.assert_close(
            output["residual"].mean(dim=1),
            torch.zeros_like(output["residual"][:, 0]),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output["prediction"],
            base_future.unsqueeze(1) + output["gate"] * output["residual"],
        )

        loss = output["prediction"].square().mean() + output["residual"].square().mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in adapter.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in passive.parameters()))

    def test_centered_single_query_requires_nonempty_references(self):
        adapter = ActionFiLMResidualAdapter(
            channels=3,
            action_horizon=2,
            hidden_channels=5,
            action_hidden_dim=7,
            centered=True,
        )
        current = torch.randn(2, 3, 4, 5)
        base = torch.randn_like(current)
        one_query = torch.randn(2, 1, 2, 3)

        with self.assertRaisesRegex(ValueError, "reference"):
            adapter(current, base, one_query)
        with self.assertRaisesRegex(ValueError, "reference"):
            adapter(
                current,
                base,
                one_query,
                reference_actions=torch.empty(2, 0, 2, 3),
            )

    def test_fixed_references_make_query_prediction_candidate_set_independent(self):
        torch.manual_seed(73)
        adapter = ActionFiLMResidualAdapter(
            channels=3,
            action_horizon=2,
            hidden_channels=5,
            action_hidden_dim=7,
            centered=True,
        ).eval()
        # Defeat the production-safe zero initialization so this test exercises
        # real action-dependent values rather than equality of all-zero outputs.
        with torch.no_grad():
            adapter.output_head[-1].weight.normal_(mean=0.0, std=0.2)
            adapter.output_head[-1].bias.normal_(mean=0.0, std=0.1)

        current = torch.randn(2, 3, 4, 5)
        base = torch.randn_like(current)
        query = torch.randn(2, 1, 2, 3)
        unrelated = 3.0 * torch.randn(2, 5, 2, 3)
        references = torch.randn(2, 4, 2, 3)

        with torch.no_grad():
            alone = adapter(
                current,
                base,
                query,
                reference_actions=references,
            )
            with_unrelated = adapter(
                current,
                base,
                torch.cat((query, unrelated), dim=1),
                reference_actions=references,
            )

        torch.testing.assert_close(
            alone["raw_residual"][:, 0], with_unrelated["raw_residual"][:, 0]
        )
        torch.testing.assert_close(
            alone["residual"][:, 0], with_unrelated["residual"][:, 0]
        )
        torch.testing.assert_close(
            alone["prediction"][:, 0], with_unrelated["prediction"][:, 0]
        )

    def test_film_adapter_rejects_inconsistent_shapes(self):
        adapter = ActionFiLMResidualAdapter(
            channels=4,
            action_horizon=3,
            hidden_channels=8,
            action_hidden_dim=11,
            centered=True,
        )
        current = torch.zeros(2, 4, 5, 6)
        base = torch.zeros_like(current)
        actions = torch.zeros(2, 6, 3, 3)
        invalid_calls = (
            (current[:, :, :, :-1], base, actions),
            (current, torch.zeros(3, 4, 5, 6), actions),
            (current, base, torch.zeros(2, 6, 2, 3)),
            (current, base, torch.zeros(2, 6, 3, 2)),
        )
        for arguments in invalid_calls:
            with self.subTest(shapes=[tuple(item.shape) for item in arguments]):
                with self.assertRaises((ValueError, RuntimeError)):
                    adapter(*arguments)


class ActionTransitionDatasetTest(unittest.TestCase):
    def test_dataset_builds_contiguous_route_safe_action_transitions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            measurements_root = root / "raw"
            split_routes = _make_transition_fixture(cache_root, measurements_root)

            dataset = ActionTransitionDataset(
                cache_root,
                measurements_root,
                split="train",
                action_horizon=2,
            )

            # Per route, only frames 0,1,2 form a contiguous transition.  The
            # endpoint pair 2 -> 4 must be rejected because frame 3 is absent.
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            route_id, route_index = split_routes["train"]
            self.assertEqual(sample["route_id"], route_id)
            self.assertEqual(sample["frame_id"], 0)
            self.assertEqual(sample["future_frame_id"], 2)
            self.assertEqual(sample["current_latent"].shape, (4, 8, 8))
            self.assertEqual(sample["future_latent"].shape, (4, 8, 8))
            torch.testing.assert_close(
                sample["current_latent"],
                torch.full_like(sample["current_latent"], route_index * 10.0),
            )
            torch.testing.assert_close(
                sample["future_latent"],
                torch.full_like(sample["future_latent"], route_index * 10.0 + 2.0),
            )
            expected_actions = torch.tensor(
                [
                    [route_index * 0.01, 0.2, 1.0],
                    [route_index * 0.01 + 0.1, 0.25, 1.0],
                ],
                dtype=torch.float32,
            )
            self.assertEqual(sample["actions"].shape, (2, 3))
            self.assertEqual(sample["actions"].dtype, torch.float32)
            torch.testing.assert_close(sample["actions"], expected_actions)
            torch.testing.assert_close(
                sample["current_pose"],
                torch.tensor([route_index * 10.0, 0.0, 0.0], dtype=torch.float64),
            )
            torch.testing.assert_close(
                sample["future_pose"],
                torch.tensor(
                    [route_index * 10.0 + 2.0, 0.0, 0.0], dtype=torch.float64
                ),
            )

    def test_dataset_accepts_native64_and_resolves_raw_paths_from_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            measurements_root = root / "raw"
            split_routes = _make_transition_fixture(
                cache_root, measurements_root, spatial_size=64
            )

            dataset = ActionTransitionDataset(
                cache_root,
                split="train",
                action_horizon=2,
            )

            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(sample["route_id"], split_routes["train"][0])
            self.assertEqual(sample["current_latent"].shape, (4, 64, 64))
            self.assertEqual(sample["future_latent"].shape, (4, 64, 64))

    def test_test_split_is_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            measurements_root = root / "raw"
            split_routes = _make_transition_fixture(cache_root, measurements_root)

            with self.assertRaisesRegex(ValueError, "allow_test|test split"):
                ActionTransitionDataset(
                    cache_root,
                    measurements_root,
                    split="test",
                    action_horizon=2,
                )

            dataset = ActionTransitionDataset(
                cache_root,
                measurements_root,
                split="test",
                action_horizon=2,
                allow_test=True,
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]["route_id"], split_routes["test"][0])

    def test_dataset_rejects_route_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            measurements_root = root / "raw"
            _make_transition_fixture(cache_root, measurements_root)
            index_path = cache_root / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            route_id = index["records"][0]["route_id"]
            route_entries = [
                entry for entry in index["records"] if entry["route_id"] == route_id
            ]
            route_entries[-1]["split"] = (
                "val" if route_entries[0]["split"] != "val" else "train"
            )
            index_path.write_text(json.dumps(index), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "route split leakage"):
                ActionTransitionDataset(
                    cache_root,
                    measurements_root,
                    split="train",
                    action_horizon=2,
                )

    def test_dataset_excludes_transition_with_non_quarter_second_cadence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            measurements_root = root / "raw"
            split_routes = _make_transition_fixture(cache_root, measurements_root)
            train_route = split_routes["train"][0]

            index_path = cache_root / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            middle = next(
                entry
                for entry in index["records"]
                if entry["route_id"] == train_route and int(entry["frame_id"]) == 1
            )
            middle["timestamp"] = 0.30
            index_path.write_text(json.dumps(index), encoding="utf-8")

            # The sole frame-contiguous train window now has 0.30/0.20 second
            # deltas instead of 0.25/0.25 and must not enter the dataset.
            with self.assertRaisesRegex(ValueError, "contiguous transitions"):
                ActionTransitionDataset(
                    cache_root,
                    measurements_root,
                    split="train",
                    action_horizon=2,
                    expected_cadence_seconds=0.25,
                )

    def test_dataset_rejects_invalid_horizon_and_missing_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            measurements_root = root / "raw"
            split_routes = _make_transition_fixture(cache_root, measurements_root)

            with self.assertRaises(ValueError):
                ActionTransitionDataset(
                    cache_root,
                    measurements_root,
                    split="train",
                    action_horizon=0,
                )

            route_id, route_index = split_routes["train"]
            broken_path = (
                measurements_root
                / Path(route_id).name
                / "measurements"
                / "0001.json.gz"
            )
            payload = _measurement(route_index, 1)
            del payload["brake"]
            del payload["control_brake"]
            _write_measurement(broken_path, payload)
            with self.assertRaises((KeyError, ValueError)):
                dataset = ActionTransitionDataset(
                    cache_root,
                    measurements_root,
                    split="train",
                    action_horizon=2,
                )
                _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
