import json
import tempfile
import unittest
from pathlib import Path

import torch

from temporal_tf.audit import audit_cache
from temporal_tf.cache import CacheWriter, load_index, sha256_file


def record(route: str, frame: int, bev_size: int = 4, *, with_optional_inputs: bool = False):
    path = torch.zeros(3, 2)
    result = {
        "bev_feature": torch.zeros(2, bev_size, bev_size),
        "pred_trajectory": path,
        "gt_trajectory": path,
        "ego_pose": torch.zeros(3),
        "route_id": route,
        "frame_id": frame,
        "trajectory_source": "frozen_model_prediction",
    }
    if with_optional_inputs:
        result["speed_t"] = float(route.rsplit("_", maxsplit=1)[-1]) + 0.25
        result["command_t"] = torch.nn.functional.one_hot(
            torch.tensor(int(route.rsplit("_", maxsplit=1)[-1]) % 6), num_classes=6
        )
    return result


class AuditTest(unittest.TestCase):
    def _write_three_routes(
        self,
        root: Path,
        heterogeneous: bool = False,
        *,
        with_optional_inputs: bool = False,
    ):
        writer = CacheWriter(root)
        for route_index in range(3):
            size = 5 if heterogeneous and route_index == 2 else 4
            writer.add(
                record(
                    f"route_{route_index}",
                    0,
                    size,
                    with_optional_inputs=with_optional_inputs,
                )
            )
        writer.finalize(split_seed=1)

    def test_corrupt_record_is_reported_not_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_three_routes(root)
            first = load_index(root)["records"][0]
            (root / first["path"]).write_bytes(b"not a torch record")
            report = audit_cache(root, deep=True)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(
                any(
                    "SHA256 mismatch" in error
                    or "torch record" in error
                    or "Weights only load" in error
                    for error in report["errors"]
                )
            )

    def test_heterogeneous_shapes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_three_routes(root, heterogeneous=True)
            report = audit_cache(root, deep=True)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("heterogeneous BEV" in error for error in report["errors"]))

    def test_optional_inputs_are_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_three_routes(root, with_optional_inputs=True)
            report = audit_cache(root, deep=True)
            self.assertEqual(report["status"], "pass")
            analysis = report["input_analysis"]
            self.assertEqual(analysis["checked_records"], 3)
            self.assertEqual(analysis["presence"]["both"], 3)
            self.assertEqual(analysis["speed_t"]["present"], 3)
            self.assertEqual(analysis["speed_t"]["missing"], 0)
            self.assertEqual(analysis["speed_t"]["min"], 0.25)
            self.assertEqual(analysis["speed_t"]["max"], 2.25)
            self.assertEqual(
                analysis["command_t"]["class_counts_zero_based"],
                {"0": 1, "1": 1, "2": 1},
            )

    def test_deep_audit_rejects_invalid_optional_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_three_routes(root, with_optional_inputs=True)
            index_path = root / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            entry = index["records"][0]
            record_path = root / entry["path"]
            cached = torch.load(record_path, weights_only=True)
            cached["command_t"] = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            torch.save(cached, record_path)
            entry["sha256"] = sha256_file(record_path)
            index_path.write_text(json.dumps(index), encoding="utf-8")

            report = audit_cache(root, deep=True)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("command_t" in error for error in report["errors"]))

    def test_split_filter_excludes_test_tensor_and_preserves_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_three_routes(root, with_optional_inputs=True)
            index = load_index(root)
            test_entry = next(entry for entry in index["records"] if entry["split"] == "test")
            (root / test_entry["path"]).write_bytes(b"corrupt test tensor")
            markers = {
                root / "test_opened.marker.json": "child marker\n",
                root / "study_test_opened.marker.json": "study marker\n",
            }
            for marker, content in markers.items():
                marker.write_text(content, encoding="utf-8")

            filtered = audit_cache(root, deep=True, splits={"train", "val"})
            self.assertEqual(filtered["status"], "pass")
            self.assertEqual(filtered["selected_splits"], ["train", "val"])
            self.assertEqual(filtered["records"], 2)
            self.assertEqual(filtered["total_index_records"], 3)
            self.assertEqual(filtered["input_analysis"]["checked_records"], 2)
            self.assertNotIn("test", filtered["split_record_counts"])
            for marker, content in markers.items():
                self.assertEqual(marker.read_text(encoding="utf-8"), content)

            unfiltered = audit_cache(root, deep=True)
            self.assertEqual(unfiltered["status"], "fail")

    def test_split_filter_rejects_invalid_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_three_routes(root)
            with self.assertRaisesRegex(ValueError, "unsupported splits"):
                audit_cache(root, deep=True, splits={"holdout"})


if __name__ == "__main__":
    unittest.main()
