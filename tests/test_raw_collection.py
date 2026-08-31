import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from temporal_tf.raw_collection import (
    assert_exact_identity_set,
    assert_collection_key_matches_route,
    audit_raw_collection,
    evaluate_tfpp_route_acceptance,
    load_collection_root_manifest,
)


def _write_collection(
    container: Path,
    *,
    leaf_name: str = "Accident_1044_0",
    route_name: str = "Town13_Rep0_1044_0_route0_08_24_19_59_08",
    frames: int = 3,
    route_composed: float = 100.0,
    route_status: str | None = None,
    route_infractions: dict[str, list[str]] | None = None,
) -> Path:
    leaf = container / leaf_name
    route = leaf / route_name
    for modality, extension in {
        "rgb": ".jpg",
        "lidar": ".laz",
        "measurements": ".json.gz",
        "boxes": ".json.gz",
    }.items():
        directory = route / modality
        directory.mkdir(parents=True, exist_ok=True)
        for frame in range(frames):
            (directory / f"{frame:04d}{extension}").write_bytes(f"{modality}-{frame}".encode())

    timestamp = route_name.removeprefix("Town13_Rep0_")
    route_infractions = route_infractions or {
        "collisions_vehicle": [],
        "min_speed_infractions": [],
    }
    route_status = route_status or ("Perfect" if route_composed >= 100.0 else "Completed")
    route_result = {
        "timestamp": timestamp,
        "status": route_status,
        "num_infractions": sum(len(values) for values in route_infractions.values()),
        "infractions": route_infractions,
        "scores": {
            "score_composed": route_composed,
            "score_route": 100.0,
            "score_penalty": route_composed / 100.0,
        },
    }
    with gzip.open(route / "results.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(route_result, stream)
    result = {
        "_checkpoint": {
            "global_record": {
                "status": route_status,
                "scores_mean": {
                    "score_composed": route_composed,
                    "score_route": 100.0,
                    "score_penalty": route_composed / 100.0,
                },
                "meta": {"exceptions": []},
            },
            "records": [{"timestamp": timestamp, "status": route_status, "meta": {}}],
        },
        "entry_status": "Finished",
    }
    (leaf / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return leaf


class RawCollectionAuditTest(unittest.TestCase):
    def test_complete_collection_passes_and_reports_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory)
            leaf = _write_collection(container)

            report = audit_raw_collection([leaf])

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["summary"]["routes"], 1)
            self.assertEqual(report["summary"]["frames"], 3)
            self.assertGreater(report["summary"]["bytes"], 0)
            self.assertEqual(report["summary"]["scenario_labels"], {"Accident": 1})
            self.assertEqual(
                report["summary"]["tfpp_loader_acceptance"],
                {
                    "accepted": 1,
                    "rejected": 0,
                    "acceptance_bases": {"perfect_composed_score": 1},
                },
            )
            audited = report["results"][0]
            self.assertEqual(len(audited["result_sha256"]), 64)
            self.assertEqual(audited["global"]["score_composed"], 100.0)
            self.assertTrue(audited["record_route_name_match"]["matches"])

    def test_misaligned_required_and_auxiliary_frames_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            leaf = _write_collection(Path(directory))
            route = next(path.parent for path in leaf.rglob("measurements"))
            (route / "lidar" / "0002.laz").unlink()
            (route / "boxes" / "0002.json.gz").unlink()

            report = audit_raw_collection([leaf])

            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("required modality frame counts differ" in e for e in report["errors"]))
            self.assertTrue(any("auxiliary modality 'boxes'" in e for e in report["errors"]))

    def test_incomplete_result_metadata_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            leaf = _write_collection(Path(directory))
            result_path = leaf / "result.json"
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload["entry_status"] = "Started"
            payload["_checkpoint"]["global_record"]["scores_mean"]["score_route"] = 99.0
            payload["_checkpoint"]["global_record"]["meta"]["exceptions"] = ["timeout"]
            result_path.write_text(json.dumps(payload), encoding="utf-8")

            report = audit_raw_collection([leaf])

            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("entry_status" in error for error in report["errors"]))
            self.assertTrue(any("100%" in error for error in report["errors"]))
            self.assertTrue(any("exceptions" in error for error in report["errors"]))

    def test_malformed_result_is_reported_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            leaf = _write_collection(Path(directory))
            (leaf / "result.json").write_text("{broken", encoding="utf-8")

            report = audit_raw_collection([leaf])

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["summary"]["routes"], 1)
            self.assertTrue(any("malformed result.json" in error for error in report["errors"]))

    def test_overlapping_inputs_are_detected_without_double_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory)
            leaf = _write_collection(container)

            report = audit_raw_collection([container, leaf])

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["discovery"]["result_occurrences"], 2)
            self.assertEqual(report["discovery"]["unique_result_paths"], 1)
            self.assertEqual(report["summary"]["routes"], 1)
            self.assertEqual(report["summary"]["frames"], 3)
            self.assertEqual(len(report["duplicates"]["result_paths"]), 1)

    def test_duplicate_route_directory_names_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory)
            common_name = "Town13_Rep0_1044_0_route0_08_24_19_59_08"
            _write_collection(container, leaf_name="Accident_1044_0", route_name=common_name)
            _write_collection(container, leaf_name="AccidentTwoWays_1153_0", route_name=common_name)

            report = audit_raw_collection([container])

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["summary"]["routes"], 2)
            self.assertEqual(len(report["duplicates"]["route_names"]), 1)

    def test_duplicate_physical_route_is_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory)
            first = _write_collection(container, leaf_name="Accident_1044_0")
            second = container / "AccidentTwoWays_1153_0"
            second.mkdir()
            (second / "result.json").write_bytes((first / "result.json").read_bytes())
            source_route = next(path.parent for path in first.rglob("measurements"))
            (second / source_route.name).symlink_to(source_route, target_is_directory=True)

            report = audit_raw_collection([container])

            self.assertEqual(report["status"], "fail")
            self.assertEqual(report["summary"]["routes"], 1)
            self.assertEqual(report["summary"]["frames"], 3)
            self.assertEqual(len(report["duplicates"]["route_directories"]), 1)

    def test_upstream_min_speed_only_exception_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            leaf = _write_collection(
                Path(directory),
                route_composed=95.0,
                route_infractions={
                    "collisions_vehicle": [],
                    "min_speed_infractions": ["slow"],
                },
            )

            report = audit_raw_collection([leaf])

            self.assertEqual(report["status"], "pass")
            loader = report["results"][0]["tfpp_loader_acceptance"]
            self.assertTrue(loader["accepted"])
            self.assertEqual(loader["acceptance_basis"], "sub_100_min_speed_only_exception")

    def test_upstream_collision_route_is_rejected_despite_full_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            leaf = _write_collection(
                Path(directory),
                route_composed=60.0,
                route_infractions={
                    "collisions_vehicle": ["collision"],
                    "min_speed_infractions": [],
                },
            )
            route = next(path.parent for path in leaf.rglob("measurements"))

            direct = evaluate_tfpp_route_acceptance(route)
            report = audit_raw_collection([leaf])

            self.assertFalse(direct["accepted"])
            self.assertEqual(report["status"], "fail")
            self.assertTrue(
                any("not all reported infractions" in reason for reason in direct["reasons"])
            )
            self.assertTrue(any("CARLA_Data would reject" in error for error in report["errors"]))

    def test_upstream_failed_status_and_malformed_route_result_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory)
            failed = _write_collection(container, route_status="Failed - Agent crashed")
            failed_route = next(path.parent for path in failed.rglob("measurements"))
            self.assertFalse(evaluate_tfpp_route_acceptance(failed_route)["accepted"])

            with gzip.open(failed_route / "results.json.gz", "wb") as stream:
                stream.write(b"not-json")
            malformed = evaluate_tfpp_route_acceptance(failed_route)
            self.assertFalse(malformed["accepted"])
            self.assertTrue(any("malformed" in reason for reason in malformed["reasons"]))

    def test_root_manifest_is_explicit_unique_and_retry_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "Accident_1044_0").mkdir(parents=True)
            (root / "data" / "HazardAtSideLane_1619_0_retry1").mkdir()
            manifest = root / "roots.txt"
            manifest.write_text(
                "Accident_1044_0 data/Accident_1044_0\n"
                "HazardAtSideLane_1619_0 data/HazardAtSideLane_1619_0_retry1\n",
                encoding="utf-8",
            )

            entries = load_collection_root_manifest(manifest)

            self.assertEqual([entry["key"] for entry in entries], [
                "Accident_1044_0", "HazardAtSideLane_1619_0"
            ])
            self.assertTrue(entries[1]["root"].endswith("HazardAtSideLane_1619_0_retry1"))

            manifest.write_text(
                "Accident_1044_0 data/Accident_1044_0\n"
                "Accident_1044_0 data/HazardAtSideLane_1619_0_retry1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate logical key"):
                load_collection_root_manifest(manifest)

    def test_exact_route_set_assertion_reports_silent_omission(self):
        assert_exact_identity_set({"a", "b"}, {"b", "a"}, stage="loader")
        with self.assertRaisesRegex(RuntimeError, r"missing=\['b'\]"):
            assert_exact_identity_set({"a", "b"}, {"a", "c"}, stage="loader")

    def test_collection_key_is_bound_to_route_seed_and_index(self):
        assert_collection_key_matches_route(
            "HazardAtSideLane_1619_0",
            "Town13_Rep0_1619_0_route0_08_24_20_41_13",
        )
        with self.assertRaisesRegex(RuntimeError, "expects route fragment"):
            assert_collection_key_matches_route(
                "HazardAtSideLane_1619_0",
                "Town13_Rep0_9999_0_route0_08_24_20_41_13",
            )

    def test_extractor_preflight_rejects_upstream_filtered_route(self):
        with tempfile.TemporaryDirectory() as directory:
            leaf = _write_collection(
                Path(directory),
                route_composed=60.0,
                route_infractions={
                    "collisions_vehicle": ["collision"],
                    "min_speed_infractions": [],
                },
            )
            script = Path(__file__).parents[1] / "scripts" / "cache_tfpp_dataset.py"
            spec = importlib.util.spec_from_file_location("cache_tfpp_dataset_test", script)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)

            with self.assertRaisesRegex(ValueError, "CARLA_Data policy"):
                module._raw_source_provenance(
                    [leaf.resolve()],
                    collection_keys=["Accident_1044_0"],
                    require_successful_results=True,
                )


if __name__ == "__main__":
    unittest.main()
