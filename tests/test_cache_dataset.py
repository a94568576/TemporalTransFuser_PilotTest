import json
import tempfile
import unittest
from pathlib import Path

import torch

from temporal_tf.cache import SCHEMA_VERSION, CacheWriter, canonical_record, load_index
from temporal_tf.data import TemporalCacheDataset


def make_record(route_id: str, frame_id: int, *, with_optional_inputs: bool = False):
    prediction = torch.tensor([[2.0 + frame_id, 0.0], [3.0 + frame_id, 0.1]])
    result = {
        "bev_feature": torch.full((3, 4, 4), float(frame_id)),
        "pred_trajectory": prediction,
        "gt_trajectory": prediction + 0.25,
        "ego_pose": torch.tensor([float(frame_id), 0.0, 0.0]),
        "route_id": route_id,
        "frame_id": frame_id,
        "trajectory_source": "frozen_model_prediction",
    }
    if with_optional_inputs:
        result["speed_t"] = torch.tensor([2.5 + frame_id], dtype=torch.float64)
        result["command_t"] = torch.nn.functional.one_hot(
            torch.tensor(frame_id % 6), num_classes=6
        )
    return result


class CacheDatasetTest(unittest.TestCase):
    def _cache(self, root: Path, *, with_optional_inputs: bool = False):
        writer = CacheWriter(root)
        for route_index in range(6):
            for frame_id in range(7):
                writer.add(
                    make_record(
                        f"route_{route_index}",
                        frame_id,
                        with_optional_inputs=with_optional_inputs,
                    )
                )
        writer.finalize(split_seed=3)

    def test_route_level_splits_and_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._cache(root)
            index = load_index(root)
            route_splits = {}
            for entry in index["records"]:
                previous = route_splits.setdefault(entry["route_id"], entry["split"])
                self.assertEqual(previous, entry["split"])
            self.assertEqual(set(route_splits.values()), {"train", "val", "test"})

            for split in ("train", "val", "test"):
                dataset = TemporalCacheDataset(root, split=split, history_length=3)
                sample = dataset[0]
                torch.testing.assert_close(
                    sample["current_bev"],
                    torch.full_like(sample["current_bev"], float(sample["frame_id"])),
                )
                self.assertEqual(dataset.sample_shape["current_bev"], (3, 4, 4))
                for window in dataset.windows:
                    self.assertEqual(len({entry["route_id"] for entry in window}), 1)
                    self.assertEqual(len({entry["split"] for entry in window}), 1)

    def test_history_uses_prediction_not_gt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._cache(root)
            dataset = TemporalCacheDataset(
                root,
                split="train",
                history_length=3,
                align_past_trajectories=False,
            )
            before = dataset[0]["past_trajectory"].clone()
            past_entry = dataset.windows[0][0]
            path = root / past_entry["path"]
            record = torch.load(path, weights_only=True)
            record["gt_trajectory"] = torch.full_like(record["gt_trajectory"], 9999.0)
            torch.save(record, path)
            after = dataset[0]["past_trajectory"]
            torch.testing.assert_close(after, before)

    def test_dataset_rejects_route_split_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._cache(root)
            index_path = root / "index.json"
            index = json.loads(index_path.read_text())
            route = index["records"][0]["route_id"]
            same_route = [entry for entry in index["records"] if entry["route_id"] == route]
            same_route[-1]["split"] = "test" if same_route[0]["split"] != "test" else "val"
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "route split leakage"):
                TemporalCacheDataset(root, split="train", history_length=3)

    def test_index_rejects_invalid_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._cache(root)
            index_path = root / "index.json"
            index = json.loads(index_path.read_text())
            index["records"][0]["split"] = "tset"
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "invalid split"):
                load_index(root)

    def test_schema_rejects_uncertified_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = CacheWriter(Path(directory))
            record = make_record("route", 0)
            record["trajectory_source"] = "ground_truth"
            with self.assertRaisesRegex(ValueError, "frozen_model_prediction"):
                writer.add(record)

    def test_schema_rejects_privileged_extra_input(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = CacheWriter(Path(directory))
            record = make_record("route", 0)
            record["future_waypoints"] = torch.ones(2, 2)
            with self.assertRaisesRegex(ValueError, "non-allowlisted"):
                writer.add(record)

    def test_schema_rejects_stale_record_version(self):
        record = make_record("route", 0)
        record["schema_version"] = SCHEMA_VERSION - 1
        with self.assertRaisesRegex(ValueError, "record schema"):
            canonical_record(record)

    def test_optional_planner_inputs_are_canonical_and_not_adapter_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._cache(root, with_optional_inputs=True)
            index = load_index(root)
            self.assertEqual(index["schema_version"], SCHEMA_VERSION)

            cached = torch.load(root / index["records"][0]["path"], weights_only=True)
            self.assertEqual(cached["schema_version"], SCHEMA_VERSION)
            self.assertEqual(cached["speed_t"].dtype, torch.float32)
            self.assertEqual(cached["speed_t"].shape, torch.Size([]))
            self.assertEqual(cached["command_t"].dtype, torch.float32)
            self.assertEqual(cached["command_t"].shape, torch.Size([6]))

            dataset = TemporalCacheDataset(root, split="train", history_length=3)
            sample = dataset[0]
            self.assertNotIn("speed_t", sample)
            self.assertNotIn("command_t", sample)

    def test_optional_planner_inputs_may_be_absent(self):
        normalized = canonical_record(make_record("route", 0))
        self.assertNotIn("speed_t", normalized)
        self.assertNotIn("command_t", normalized)

    def test_signed_forward_speed_is_preserved(self):
        record = make_record("route", 0)
        record["speed_t"] = -0.25
        normalized = canonical_record(record)
        self.assertEqual(float(normalized["speed_t"]), -0.25)

    def test_optional_planner_inputs_are_strictly_validated(self):
        invalid_values = {
            "non-scalar speed": {"speed_t": torch.ones(2)},
            "non-finite speed": {"speed_t": float("nan")},
            "wrong command shape": {"command_t": torch.ones(5)},
            "soft command": {"command_t": torch.full((6,), 1.0 / 6.0)},
            "multi-hot command": {"command_t": torch.tensor([1, 1, 0, 0, 0, 0])},
        }
        for name, invalid in invalid_values.items():
            with self.subTest(name=name):
                record = make_record("route", 0)
                record.update(invalid)
                with self.assertRaisesRegex(ValueError, "speed_t|command_t"):
                    canonical_record(record)

    def test_cache_cannot_be_finalized_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = CacheWriter(Path(directory))
            writer.add(make_record("route", 0))
            writer.finalize()
            with self.assertRaises(FileExistsError):
                writer.finalize()


if __name__ == "__main__":
    unittest.main()
