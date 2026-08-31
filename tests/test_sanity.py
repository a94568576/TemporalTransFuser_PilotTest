import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.visualize_cache import _resolve_rgb, render_cache_visualizations
from temporal_tf.cache import CacheWriter, load_index, sha256_file
from temporal_tf.sanity import (
    analyze_cache,
    deterministic_sample_entries,
    raw_frame_metrics,
    route_safe_history_entries,
    write_json_artifact,
)


def cache_record(route_id: str, frame_id: int, error: float, *, bev_size: int = 4):
    target = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    prediction = target + torch.tensor([error, 0.0])
    return {
        "bev_feature": torch.full((3, bev_size, bev_size), float(frame_id + 1)),
        "pred_trajectory": prediction,
        "gt_trajectory": target,
        "ego_pose": torch.tensor([float(frame_id), 0.0, 0.0]),
        "route_id": route_id,
        "frame_id": frame_id,
        "timestamp": frame_id * 0.1,
        "trajectory_source": "frozen_model_prediction",
    }


class SanityTest(unittest.TestCase):
    def _write_cache(
        self,
        root: Path,
        *,
        route_count: int = 3,
        frames=(0, 1, 2, 3),
        heterogeneous: bool = False,
    ):
        writer = CacheWriter(root)
        for route_index in range(route_count):
            for frame_id in frames:
                bev_size = 5 if heterogeneous and route_index == route_count - 1 else 4
                writer.add(
                    cache_record(
                        f"route_{route_index}",
                        frame_id,
                        float(route_index),
                        bev_size=bev_size,
                    )
                )
        writer.finalize(split_seed=7)

    def test_raw_frame_metric_definitions(self):
        target = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        prediction = torch.tensor([[1.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
        metrics = raw_frame_metrics(prediction, target)
        self.assertAlmostEqual(metrics["ade"], 5.0 / 3.0)
        self.assertAlmostEqual(metrics["fde"], 3.0)
        self.assertAlmostEqual(metrics["point_l1"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["smoothness"], 2.0)

    def test_micro_and_equal_route_macro_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)
            report = analyze_cache(root)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["index"]["records"], 12)
            self.assertEqual(report["index"]["routes"], 3)
            self.assertAlmostEqual(report["metrics"]["sample_micro"]["ade"]["mean"], 1.0)
            self.assertAlmostEqual(
                report["metrics"]["sample_micro"]["point_l1"]["mean"], 0.5
            )
            macro = report["metrics"]["equal_route_macro"]["ade"]
            self.assertAlmostEqual(macro["mean"], 1.0)
            self.assertAlmostEqual(macro["p90"], 1.8)
            self.assertAlmostEqual(macro["p95"], 1.9)

    def test_single_route_empty_validation_splits_are_warning_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root, route_count=1)
            report = analyze_cache(root)
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["index"]["split_record_counts"]["val"], 0)
            self.assertEqual(report["index"]["split_record_counts"]["test"], 0)
            self.assertTrue(any("single-route" in warning for warning in report["warnings"]))

    def test_too_few_routes_for_split_count_is_warning_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root, route_count=2)
            report = analyze_cache(root)
            self.assertEqual(report["status"], "pass")
            self.assertTrue(any("only 2 routes" in warning for warning in report["warnings"]))

    def test_explicit_split_filter_requires_every_requested_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root, route_count=1)
            report = analyze_cache(root, splits={"train", "val"})
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("empty selected splits" in error for error in report["errors"]))

    def test_unequal_route_lengths_distinguish_micro_and_macro(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = CacheWriter(root)
            for route_index, frame_count in enumerate((1, 2, 5)):
                for frame_id in range(frame_count):
                    writer.add(
                        cache_record(
                            f"route_{route_index}", frame_id, float(route_index)
                        )
                    )
            writer.finalize(split_seed=7)
            report = analyze_cache(root)
            self.assertEqual(report["status"], "pass")
            self.assertAlmostEqual(
                report["metrics"]["sample_micro"]["ade"]["mean"], 1.5
            )
            self.assertAlmostEqual(
                report["metrics"]["equal_route_macro"]["ade"]["mean"], 1.0
            )

    def test_frame_gap_is_a_sanity_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root, route_count=1, frames=(0, 2, 3))
            report = analyze_cache(root)
            self.assertEqual(report["status"], "fail")
            route = report["cadence"]["routes"]["route_0"]
            self.assertEqual(route["frame"]["gap_count"], 1)
            self.assertTrue(any("frame_gap_count" in error for error in report["errors"]))

    def test_corrupt_record_and_nonfinite_record_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)
            index = load_index(root)
            first_path = root / index["records"][0]["path"]
            first_path.write_bytes(b"corrupt")
            report = analyze_cache(root)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["checks"]["hash_failures"], 1)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)
            index_path = root / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            entry = index["records"][0]
            record_path = root / entry["path"]
            record = torch.load(record_path, weights_only=True)
            record["pred_trajectory"][0, 0] = float("nan")
            torch.save(record, record_path)
            entry["sha256"] = sha256_file(record_path)
            index_path.write_text(json.dumps(index), encoding="utf-8")
            report = analyze_cache(root)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("NaN or Inf" in error for error in report["errors"]))

    def test_nonmonotonic_index_and_heterogeneous_shapes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)
            index_path = root / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["records"][0], index["records"][1] = (
                index["records"][1],
                index["records"][0],
            )
            index_path.write_text(json.dumps(index), encoding="utf-8")
            report = analyze_cache(root)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("frame_nonmonotonic" in error for error in report["errors"]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root, heterogeneous=True)
            report = analyze_cache(root)
            self.assertEqual(report["status"], "fail")
            self.assertIn("bev_feature", report["checks"]["heterogeneous_shapes"])

    def test_duplicate_and_nonmonotonic_timestamps_fail_without_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)
            index_path = root / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["records"][1]["timestamp"] = index["records"][0]["timestamp"]
            index["records"][2]["timestamp"] = index["records"][0]["timestamp"] - 0.1
            index_path.write_text(json.dumps(index), encoding="utf-8")
            report = analyze_cache(root)
            self.assertEqual(report["status"], "fail")
            counts = report["cadence"]["aggregate_counts"]
            self.assertEqual(counts["timestamp_duplicate_count"], 1)
            self.assertEqual(counts["timestamp_nonmonotonic_count"], 1)

    def test_timestamp_cadence_outlier_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)
            index_path = root / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["records"][1]["timestamp"] = 0.15
            index["records"][2]["timestamp"] = 0.25
            index_path.write_text(json.dumps(index), encoding="utf-8")
            report = analyze_cache(root)
            self.assertEqual(report["status"], "fail")
            self.assertEqual(
                report["cadence"]["aggregate_counts"]["timestamp_cadence_mismatch_count"],
                2,
            )

    def test_duplicate_index_identity_is_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)
            index_path = root / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["records"].append(copy.deepcopy(index["records"][0]))
            index_path.write_text(json.dumps(index), encoding="utf-8")
            report = analyze_cache(root)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("duplicate" in error for error in report["errors"]))

    def test_reference_tolerance_passes_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)
            baseline = analyze_cache(root)
            matching = analyze_cache(
                root,
                reference={"metrics": baseline["metrics"]},
                reference_tolerance=1e-12,
            )
            self.assertEqual(matching["status"], "pass")
            self.assertEqual(matching["reference_comparison"]["status"], "pass")

            mismatched_reference = copy.deepcopy({"metrics": baseline["metrics"]})
            mismatched_reference["metrics"]["sample_micro"]["ade"]["mean"] += 0.1
            mismatched = analyze_cache(
                root,
                reference=mismatched_reference,
                reference_tolerance=0.01,
            )
            self.assertEqual(mismatched["status"], "fail")
            self.assertEqual(mismatched["reference_comparison"]["status"], "fail")

            malformed_reference = copy.deepcopy({"metrics": baseline["metrics"]})
            malformed_reference["metrics"]["sample_micro"]["ad_typo"] = {
                "mean": 1.0
            }
            malformed = analyze_cache(root, reference=malformed_reference)
            self.assertEqual(malformed["status"], "fail")
            self.assertTrue(
                any("unknown metric" in error for error in malformed["errors"])
            )

    def test_deterministic_selection_and_route_safe_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_cache(root)
            index = load_index(root)
            first = deterministic_sample_entries(index, max_samples=5)
            second = deterministic_sample_entries(index, max_samples=5)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 5)

            current = next(
                entry
                for entry in index["records"]
                if entry["route_id"] == "route_0" and entry["frame_id"] == 3
            )
            history = route_safe_history_entries(index, current, history_length=2)
            self.assertEqual([entry["frame_id"] for entry in history], [1, 2])
            self.assertEqual({entry["route_id"] for entry in history}, {"route_0"})

    def test_split_filter_scopes_metrics_sampling_and_visualization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            self._write_cache(root)
            index = load_index(root)
            test_entry = next(entry for entry in index["records"] if entry["split"] == "test")
            (root / test_entry["path"]).write_bytes(b"corrupt test tensor")
            markers = {
                root / "test_opened.marker.json": "child marker\n",
                root / "study_test_opened.marker.json": "study marker\n",
            }
            for marker, content in markers.items():
                marker.write_text(content, encoding="utf-8")

            report = analyze_cache(root, splits={"train", "val"})
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["parameters"]["selected_splits"], ["train", "val"])
            self.assertEqual(report["index"]["records"], 8)
            self.assertEqual(report["index"]["total_index_records"], 12)
            self.assertEqual(report["index"]["routes"], 2)
            self.assertEqual(report["index"]["split_record_counts"]["test"], 0)
            self.assertEqual(report["metrics"]["sample_micro"]["ade"]["count"], 8)

            selected = deterministic_sample_entries(
                index, max_samples=5, splits={"train", "val"}
            )
            self.assertEqual(len(selected), 5)
            self.assertEqual({entry["split"] for entry in selected}, {"train", "val"})

            visual = render_cache_visualizations(
                root,
                Path(directory) / "visualizations",
                max_samples=4,
                history_length=1,
                splits={"train", "val"},
            )
            self.assertEqual(visual["status"], "pass")
            self.assertEqual(visual["selection"]["selected_splits"], ["train", "val"])
            self.assertEqual({figure["split"] for figure in visual["figures"]}, {"train", "val"})
            self.assertNotIn(
                "test", visual["selection"]["selected_sample_split_counts"]
            )
            for marker, content in markers.items():
                self.assertEqual(marker.read_text(encoding="utf-8"), content)

            unfiltered = analyze_cache(root)
            self.assertEqual(unfiltered["status"], "fail")

    def test_artifact_writer_refuses_permanent_test_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in (
                "test_opened.marker.json",
                "study_test_opened.marker.json",
            ):
                marker = root / filename
                marker.write_text("permanent\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "permanent test marker"):
                    write_json_artifact({"status": "pass"}, marker)
                self.assertEqual(marker.read_text(encoding="utf-8"), "permanent\n")

    def test_ambiguous_rgb_match_disables_overlay(self):
        route_id = "dataset/root_00/shared_route_name"
        lookup = {
            ("shared_route_name", 12): [
                Path("/raw/a/shared_route_name/rgb/0012.png"),
                Path("/raw/b/shared_route_name/rgb/0012.png"),
            ]
        }
        path, warning = _resolve_rgb(lookup, route_id, 12)
        self.assertIsNone(path)
        self.assertIn("ambiguous", warning)


if __name__ == "__main__":
    unittest.main()
