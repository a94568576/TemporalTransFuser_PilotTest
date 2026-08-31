import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from temporal_tf.config import DEFAULTS
from temporal_tf.engine import (
    CHECKPOINT_SELECTION_METRIC,
    SELECTION_MANIFEST,
    TEST_OPEN_MARKER,
    _STUDY_FINALIZE_CAPABILITY,
    finalize_selection,
)
from temporal_tf.model import VARIANTS as MODEL_VARIANTS
from temporal_tf.selection_compare import CHOICE_JSON, write_selection_choice
from temporal_tf.study import (
    STUDY_MANIFEST,
    STUDY_REPORT,
    STUDY_RESULTS,
    STUDY_TEST_OPEN_MARKER,
    _file_hash,
    _stable_hash,
    finalize_multiseed_study,
    run_multiseed_selection,
)


SEEDS = (17, 29, 43)
VARIANTS = ("current_only",)


def _write_fake_selection(call_log, **kwargs):
    cache_root = Path(kwargs["cache_root"])
    output_dir = Path(kwargs["output_dir"])
    config = deepcopy(kwargs["config"])
    variants = tuple(kwargs["variants"])
    seed = int(config["seed"])
    selection_id = f"selection-{seed}"
    selection_owner = deepcopy(kwargs.get("selection_owner"))
    call_log.append(kwargs)
    if kwargs.get("evaluation_mode") != "selection":
        raise AssertionError("study selection attempted a non-selection engine mode")

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoints = {}
    for variant in variants:
        checkpoint_path = checkpoint_dir / f"{variant}.pt"
        checkpoint_path.write_bytes(f"locked:{seed}:{variant}".encode("utf-8"))
        checkpoints[variant] = {
            "path": f"checkpoints/{variant}.pt",
            "sha256": _file_hash(checkpoint_path),
        }
    results = {
        "status": "completed",
        "evaluation_mode": "selection",
        "test_opened": False,
        "selection_id": selection_id,
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "configuration": config,
        "variants": {
            variant: {"splits": {"val": {"route_macro": {"ade": 1.0}}}}
            for variant in variants
        },
    }
    if selection_owner is not None:
        results["selection_owner"] = selection_owner
    (output_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "selection_id": selection_id,
        "status": "selection_complete",
        "test_opened": False,
        "config_sha256": _stable_hash(config),
        "cache_index_sha256": _file_hash(cache_root / "index.json"),
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "variants": list(variants),
        "variant_specs": {
            variant: MODEL_VARIANTS[variant].input_semantics() for variant in variants
        },
        "checkpoints": checkpoints,
        "results": "results.json",
    }
    if selection_owner is not None:
        manifest["selection_owner"] = selection_owner
    (output_dir / SELECTION_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return results


def _split(metric, *, gate, residual, latency, improved=None, harmed=None):
    result = {
        "route_macro": {"ade": float(metric)},
        "mean_gate": float(gate),
        "mean_raw_residual_l1": float(residual) * 2.0,
        "mean_applied_residual_l1": float(residual),
        "mean_latency_ms": float(latency),
    }
    if improved is not None:
        result["paired_vs_baseline"] = {
            "route_fractions": {
                "improved": float(improved),
                "harmed": float(harmed),
                "unchanged": 1.0 - float(improved) - float(harmed),
            },
            "sample_fractions": {
                "improved": float(improved) - 0.1,
                "harmed": float(harmed) + 0.1,
                "unchanged": 1.0 - float(improved) - float(harmed),
            },
        }
    return result


def _fake_final(**kwargs):
    selection_dir = Path(kwargs["selection_dir"])
    seed = int(selection_dir.name.removeprefix("seed_"))
    ordinal = SEEDS.index(seed) + 1
    Path(kwargs["output_dir"]).mkdir(parents=True)
    if kwargs.get("config") is not None:
        raise AssertionError("study final attempted to override a locked child config")
    locked_config = json.loads(
        (selection_dir / "results.json").read_text(encoding="utf-8")
    )["configuration"]
    return {
        "selection_id": f"selection-{seed}",
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "configuration": locked_config,
        "selection_owner": json.loads(
            (selection_dir / "results.json").read_text(encoding="utf-8")
        )["selection_owner"],
        "baseline": {
            "test": _split(10 + ordinal, gate=0.0, residual=0.0, latency=10 + ordinal)
        },
        "variants": {
            "current_only": {
                "splits": {
                    "test": _split(
                        ordinal,
                        gate=0.1 * ordinal,
                        residual=0.01 * ordinal,
                        latency=4 + ordinal,
                        improved=0.4 + 0.1 * ordinal,
                        harmed=0.3 - 0.1 * ordinal,
                    )
                }
            }
        },
    }


class MultiSeedStudyTest(unittest.TestCase):
    def _selection(self, root, *, name="study_selection", residual_weight=None):
        cache = root / "cache"
        cache.mkdir()
        (cache / "index.json").write_text('{"records": []}', encoding="utf-8")
        study_dir = root / name
        config = deepcopy(DEFAULTS)
        if residual_weight is not None:
            config["training"]["residual_weight"] = residual_weight
        original = deepcopy(config)
        calls = []
        with patch(
            "temporal_tf.study.run_pilot",
            side_effect=lambda **kwargs: _write_fake_selection(calls, **kwargs),
        ), patch("temporal_tf.study.finalize_selection") as final_mock:
            manifest = run_multiseed_selection(
                cache_root=cache,
                study_dir=study_dir,
                config=config,
                seeds=SEEDS,
                variants=VARIANTS,
                device_name="cpu",
            )
            final_mock.assert_not_called()
        self.assertEqual(config, original)
        self.assertEqual([call["config"]["seed"] for call in calls], list(SEEDS))
        for call in calls:
            seed_config = deepcopy(call["config"])
            seed_config["seed"] = original["seed"]
            self.assertEqual(seed_config, original)
            self.assertEqual(tuple(call["variants"]), VARIANTS)
            self.assertEqual(call["evaluation_mode"], "selection")
            self.assertEqual(call["selection_owner"]["study_owner_id"], manifest["study_owner_id"])
        self.assertFalse(manifest["test_opened"])
        self.assertEqual(manifest["seeds"], list(SEEDS))
        self.assertTrue((study_dir / STUDY_MANIFEST).is_file())
        self.assertFalse((study_dir / STUDY_TEST_OPEN_MARKER).exists())
        for seed in SEEDS:
            child = study_dir / "selections" / f"seed_{seed}"
            self.assertFalse((child / TEST_OPEN_MARKER).exists())
            child_manifest = json.loads(
                (child / SELECTION_MANIFEST).read_text(encoding="utf-8")
            )
            self.assertEqual(
                child_manifest["selection_owner"]["study_owner_id"],
                manifest["study_owner_id"],
            )
        return cache, study_dir

    def test_selection_hash_lock_final_aggregation_and_reopen_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, study_dir = self._selection(root)
            manifest = json.loads((study_dir / STUDY_MANIFEST).read_text(encoding="utf-8"))
            checkpoint_info = manifest["selections"][0]["checkpoints"]["current_only"]
            checkpoint_path = (
                study_dir
                / manifest["selections"][0]["selection_dir"]
                / checkpoint_info["path"]
            )
            locked_bytes = checkpoint_path.read_bytes()
            checkpoint_path.write_bytes(b"tampered")
            with patch("temporal_tf.study.finalize_selection") as final_mock:
                with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                    finalize_multiseed_study(
                        study_selection=study_dir,
                        cache_root=cache,
                        output_dir=root / "tampered_final",
                        device_name="cpu",
                    )
                final_mock.assert_not_called()
            self.assertFalse((study_dir / STUDY_TEST_OPEN_MARKER).exists())
            checkpoint_path.write_bytes(locked_bytes)

            with patch("temporal_tf.study.finalize_selection", side_effect=_fake_final) as final_mock:
                result = finalize_multiseed_study(
                    study_selection=study_dir,
                    cache_root=cache,
                    output_dir=root / "final",
                    device_name="cpu",
                )
            self.assertEqual(final_mock.call_count, 3)
            for call in final_mock.call_args_list:
                self.assertEqual(
                    call.kwargs["_study_owner_id"], manifest["study_owner_id"]
                )
                self.assertIs(
                    call.kwargs["_study_finalize_capability"],
                    _STUDY_FINALIZE_CAPABILITY,
                )
            self.assertTrue((study_dir / STUDY_TEST_OPEN_MARKER).is_file())
            self.assertTrue(result["single_test_open_event"])
            self.assertEqual(set(result["seed_results"]), {"17", "29", "43"})
            primary = result["method_aggregates"]["current_only"]["route_macro_primary"]
            self.assertAlmostEqual(primary["mean"], 2.0)
            self.assertAlmostEqual(primary["std"], 1.0)
            self.assertEqual(primary["min"], 1.0)
            self.assertEqual(primary["max"], 3.0)
            diagnostics = result["method_aggregates"]["current_only"]
            self.assertAlmostEqual(diagnostics["mean_gate"]["mean"], 0.2)
            self.assertAlmostEqual(diagnostics["mean_applied_residual_l1"]["mean"], 0.02)
            self.assertAlmostEqual(diagnostics["latency_ms"]["mean"], 6.0)
            harm = diagnostics["harm_improvement"]
            self.assertAlmostEqual(harm["route_improved_fraction"]["mean"], 0.6)
            self.assertAlmostEqual(harm["route_harmed_fraction"]["mean"], 0.1)
            self.assertTrue((root / "final" / STUDY_RESULTS).is_file())
            report_path = root / "final" / STUDY_REPORT
            self.assertTrue(report_path.is_file())
            self.assertIn("one permanent test-open event", report_path.read_text(encoding="utf-8"))

            with patch("temporal_tf.study.finalize_selection") as final_mock:
                with self.assertRaisesRegex(RuntimeError, "already been opened"):
                    finalize_multiseed_study(
                        study_selection=study_dir,
                        cache_root=cache,
                        output_dir=root / "second_final",
                        device_name="cpu",
                    )
                final_mock.assert_not_called()

    def test_study_child_refuses_standalone_finalize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, study_dir = self._selection(root)
            child = study_dir / "selections" / "seed_17"

            with self.assertRaisesRegex(RuntimeError, "owned by a multi-seed study"):
                finalize_selection(
                    selection_dir=child,
                    cache_root=cache,
                    output_dir=root / "forbidden_child_final",
                    device_name="cpu",
                )
            self.assertFalse((child / TEST_OPEN_MARKER).exists())

    def test_choice_binding_rejects_wrong_or_tampered_choice_and_finalizes_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            low_root = root / "low"
            high_root = root / "high"
            low_root.mkdir()
            high_root.mkdir()
            low_cache, low_study = self._selection(
                low_root, name="study", residual_weight=0.01
            )
            _, high_study = self._selection(
                high_root, name="study", residual_weight=0.10
            )
            choice_dir = root / "choice"
            choice = write_selection_choice(
                [high_study, low_study],
                output_dir=choice_dir,
                primary_variant="current_only",
            )
            choice_path = choice_dir / CHOICE_JSON
            self.assertEqual(choice["chosen"]["study_dir"], str(low_study.resolve()))

            with patch("temporal_tf.study.finalize_selection") as final_mock:
                with self.assertRaisesRegex(ValueError, "does not match.*chosen study"):
                    finalize_multiseed_study(
                        study_selection=high_study,
                        cache_root=low_cache,
                        output_dir=root / "wrong_study_final",
                        device_name="cpu",
                        selection_choice=choice_path,
                    )
                final_mock.assert_not_called()
            self.assertFalse((low_study / STUDY_TEST_OPEN_MARKER).exists())

            tampered = deepcopy(choice)
            losing = next(
                candidate
                for candidate in tampered["candidates"]
                if candidate["study_dir"] != tampered["chosen"]["study_dir"]
            )
            tampered["chosen"] = {
                "study_id": losing["study_id"],
                "study_dir": losing["study_dir"],
                "study_manifest": losing["study_manifest"],
                "study_manifest_sha256": losing["study_manifest_sha256"],
                "training_residual_weight": losing["training_residual_weight"],
                "primary_validation_route_macro_ade": losing[
                    "primary_validation_route_macro_ade"
                ],
            }
            tampered_dir = root / "tampered_choice"
            tampered_dir.mkdir()
            tampered_path = tampered_dir / CHOICE_JSON
            tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
            with patch("temporal_tf.study.finalize_selection") as final_mock:
                with self.assertRaisesRegex(ValueError, "does not match recomputation"):
                    finalize_multiseed_study(
                        study_selection=None,
                        cache_root=low_cache,
                        output_dir=root / "tampered_choice_final",
                        device_name="cpu",
                        selection_choice=tampered_path,
                    )
                final_mock.assert_not_called()
            self.assertFalse((low_study / STUDY_TEST_OPEN_MARKER).exists())

            chosen_manifest = low_study / STUDY_MANIFEST
            locked_manifest_bytes = chosen_manifest.read_bytes()
            chosen_manifest.write_bytes(locked_manifest_bytes + b"\n")
            with patch("temporal_tf.study.finalize_selection") as final_mock:
                with self.assertRaisesRegex(ValueError, "does not match recomputation"):
                    finalize_multiseed_study(
                        study_selection=None,
                        cache_root=low_cache,
                        output_dir=root / "manifest_hash_tamper_final",
                        device_name="cpu",
                        selection_choice=choice_path,
                    )
                final_mock.assert_not_called()
            self.assertFalse((low_study / STUDY_TEST_OPEN_MARKER).exists())
            chosen_manifest.write_bytes(locked_manifest_bytes)

            with patch("temporal_tf.study.finalize_selection", side_effect=_fake_final):
                result = finalize_multiseed_study(
                    study_selection=None,
                    cache_root=low_cache,
                    output_dir=root / "choice_locked_final",
                    device_name="cpu",
                    selection_choice=choice_path,
                )
            self.assertEqual(
                result["selection_choice"]["chosen_study_manifest_sha256"],
                choice["chosen"]["study_manifest_sha256"],
            )
            self.assertEqual(
                result["selection_choice"]["training_residual_weight"], 0.01
            )

    def test_study_marker_survives_child_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, study_dir = self._selection(root)

            def fail_second(**kwargs):
                if Path(kwargs["selection_dir"]).name == "seed_29":
                    raise RuntimeError("injected child failure")
                return _fake_final(**kwargs)

            with patch("temporal_tf.study.finalize_selection", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "injected child failure"):
                    finalize_multiseed_study(
                        study_selection=study_dir,
                        cache_root=cache,
                        output_dir=root / "failed_final",
                        device_name="cpu",
                    )
            self.assertTrue((study_dir / STUDY_TEST_OPEN_MARKER).is_file())
            manifest = json.loads((study_dir / STUDY_MANIFEST).read_text(encoding="utf-8"))
            self.assertTrue(manifest["test_opened"])

    def test_seed_validation_requires_three_unique_values(self):
        config = deepcopy(DEFAULTS)
        with self.assertRaisesRegex(ValueError, "at least three"):
            run_multiseed_selection(
                cache_root="unused",
                study_dir="unused",
                config=config,
                seeds=(1, 2),
                variants=VARIANTS,
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            run_multiseed_selection(
                cache_root="unused",
                study_dir="unused",
                config=config,
                seeds=(1, 1, 2),
                variants=VARIANTS,
            )


if __name__ == "__main__":
    unittest.main()
