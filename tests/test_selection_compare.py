import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from temporal_tf.config import DEFAULTS
from temporal_tf.engine import (
    CHECKPOINT_SELECTION_METRIC,
    SELECTION_MANIFEST,
    STUDY_SELECTION_OWNER_KIND,
    TEST_OPEN_MARKER,
)
from temporal_tf.model import VARIANTS as MODEL_VARIANTS
from temporal_tf.selection_compare import (
    CHOICE_JSON,
    CHOICE_REPORT,
    SelectionComparisonError,
    build_selection_choice,
    write_selection_choice,
)
from temporal_tf.study import (
    STUDY_MANIFEST,
    STUDY_TEST_OPEN_MARKER,
    _file_hash,
    _stable_hash,
    _study_id_payload,
    _study_owner_payload,
)


SEEDS = (17, 29, 43)
VARIANTS = ("current_only", "past_bev")
CACHE_HASH = hashlib.sha256(b"shared-cache-index").hexdigest()


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _make_study(
    root: Path,
    name: str,
    *,
    residual_weight: float,
    primary_values,
    diagnostic_values=(9.0, 9.0, 9.0),
    learning_rate=None,
) -> Path:
    study_dir = root / name
    variant_specs = {
        variant: MODEL_VARIANTS[variant].input_semantics() for variant in VARIANTS
    }
    owner_config = deepcopy(DEFAULTS)
    owner_config["training"]["residual_weight"] = residual_weight
    if learning_rate is not None:
        owner_config["training"]["learning_rate"] = learning_rate
    owner_config.pop("seed")
    base_config_hash = _stable_hash(owner_config)
    study_owner_id = _stable_hash(
        _study_owner_payload(
            cache_index_sha256=CACHE_HASH,
            base_config_without_seed_sha256=base_config_hash,
            seeds=SEEDS,
            variants=VARIANTS,
            variant_specs=variant_specs,
            primary_metric="ade",
        )
    )
    selection_owner = {
        "kind": STUDY_SELECTION_OWNER_KIND,
        "study_owner_id": study_owner_id,
    }
    entries = []
    for seed, primary, diagnostic in zip(SEEDS, primary_values, diagnostic_values):
        selection_dir = study_dir / "selections" / f"seed_{seed}"
        checkpoint_dir = selection_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        config = deepcopy(DEFAULTS)
        config["seed"] = seed
        config["training"]["residual_weight"] = residual_weight
        if learning_rate is not None:
            config["training"]["learning_rate"] = learning_rate
        config_without_seed = deepcopy(config)
        config_without_seed.pop("seed")
        assert base_config_hash == _stable_hash(config_without_seed)

        selection_id = f"{name}-selection-{seed}"
        results = {
            "status": "completed",
            "evaluation_mode": "selection",
            "test_opened": False,
            "selection_id": selection_id,
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "configuration": config,
            "selection_owner": selection_owner,
            "variants": {
                "current_only": {
                    "splits": {"val": {"route_macro": {"ade": float(diagnostic)}}}
                },
                "past_bev": {
                    "splits": {"val": {"route_macro": {"ade": float(primary)}}}
                },
            },
        }
        results_path = selection_dir / "results.json"
        _write_json(results_path, results)

        checkpoints = {}
        for variant in VARIANTS:
            checkpoint_path = checkpoint_dir / f"{variant}.pt"
            checkpoint_path.write_bytes(f"{name}:{seed}:{variant}".encode("utf-8"))
            checkpoints[variant] = {
                "path": f"checkpoints/{variant}.pt",
                "sha256": _file_hash(checkpoint_path),
            }
        child_manifest = {
            "schema_version": 2,
            "selection_id": selection_id,
            "status": "selection_complete",
            "test_opened": False,
            "config_sha256": _stable_hash(config),
            "cache_index_sha256": CACHE_HASH,
            "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
            "variants": list(VARIANTS),
            "variant_specs": variant_specs,
            "checkpoints": checkpoints,
            "results": "results.json",
            "selection_owner": selection_owner,
        }
        child_manifest_path = selection_dir / SELECTION_MANIFEST
        _write_json(child_manifest_path, child_manifest)
        entries.append(
            {
                "seed": seed,
                "selection_id": selection_id,
                "selection_dir": f"selections/seed_{seed}",
                "selection_manifest": {
                    "path": SELECTION_MANIFEST,
                    "sha256": _file_hash(child_manifest_path),
                },
                "selection_results": {
                    "path": "results.json",
                    "sha256": _file_hash(results_path),
                },
                "config_sha256": _stable_hash(config),
                "cache_index_sha256": CACHE_HASH,
                "checkpoints": checkpoints,
            }
        )

    manifest = {
        "schema_version": 1,
        "status": "selection_complete",
        "test_opened": False,
        "study_owner_id": study_owner_id,
        "cache_index_sha256": CACHE_HASH,
        "base_config_without_seed_sha256": base_config_hash,
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "variant_specs": variant_specs,
        "primary_metric": "ade",
        "checkpoint_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "selections": entries,
        "test_protocol": "locked test remains unopened",
    }
    manifest["study_id"] = _stable_hash(_study_id_payload(manifest))[:20]
    _write_json(study_dir / STUDY_MANIFEST, manifest)
    return study_dir


class SelectionComparisonTest(unittest.TestCase):
    def test_correct_primary_choice_and_artifacts_are_validation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lower_primary = _make_study(
                root,
                "weight_010",
                residual_weight=0.10,
                primary_values=(0.7, 0.9, 0.8),
                diagnostic_values=(50.0, 50.0, 50.0),
            )
            better_diagnostic_only = _make_study(
                root,
                "weight_001",
                residual_weight=0.01,
                primary_values=(0.9, 1.0, 1.1),
                diagnostic_values=(0.01, 0.01, 0.01),
            )
            before = {
                path: path.read_bytes()
                for path in (
                    lower_primary / STUDY_MANIFEST,
                    better_diagnostic_only / STUDY_MANIFEST,
                )
            }

            output = root / "choice"
            choice = write_selection_choice(
                [better_diagnostic_only / STUDY_MANIFEST, lower_primary],
                output_dir=output,
            )

            self.assertEqual(choice["chosen"]["study_dir"], str(lower_primary.resolve()))
            self.assertEqual(choice["chosen"]["training_residual_weight"], 0.10)
            self.assertAlmostEqual(
                choice["chosen"]["primary_validation_route_macro_ade"]["mean"], 0.8
            )
            self.assertAlmostEqual(
                choice["chosen"]["primary_validation_route_macro_ade"]["std"], 0.1
            )
            self.assertFalse(choice["test_data_accessed"])
            self.assertFalse(choice["diagnostic_variants_influence_choice"])
            self.assertTrue((output / CHOICE_JSON).is_file())
            report = (output / CHOICE_REPORT).read_text(encoding="utf-8")
            self.assertIn("No test data was read", report)
            self.assertIn(choice["chosen"]["study_manifest_sha256"], report)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
                self.assertFalse((path.parent / STUDY_TEST_OPEN_MARKER).exists())

    def test_tampered_result_and_test_open_state_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _make_study(
                root, "first", residual_weight=0.01, primary_values=(1.0, 1.0, 1.0)
            )
            second = _make_study(
                root, "second", residual_weight=0.02, primary_values=(1.1, 1.1, 1.1)
            )
            tampered = first / "selections" / "seed_17" / "results.json"
            tampered.write_text(tampered.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(SelectionComparisonError, "results hash mismatch"):
                build_selection_choice([first, second])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _make_study(
                root, "first", residual_weight=0.01, primary_values=(1.0, 1.0, 1.0)
            )
            second = _make_study(
                root, "second", residual_weight=0.02, primary_values=(1.1, 1.1, 1.1)
            )
            (second / STUDY_TEST_OPEN_MARKER).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(SelectionComparisonError, "test-open marker"):
                build_selection_choice([first, second])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _make_study(
                root, "first", residual_weight=0.01, primary_values=(1.0, 1.0, 1.0)
            )
            second = _make_study(
                root, "second", residual_weight=0.02, primary_values=(1.1, 1.1, 1.1)
            )
            child_marker = second / "selections" / "seed_29" / TEST_OPEN_MARKER
            child_marker.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(SelectionComparisonError, "test-open marker"):
                build_selection_choice([first, second])

    def test_non_weight_configuration_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _make_study(
                root,
                "first",
                residual_weight=0.01,
                primary_values=(1.0, 1.0, 1.0),
                learning_rate=0.001,
            )
            second = _make_study(
                root,
                "second",
                residual_weight=0.02,
                primary_values=(0.9, 0.9, 0.9),
                learning_rate=0.002,
            )
            with self.assertRaisesRegex(
                SelectionComparisonError, "differ outside training.residual_weight"
            ):
                build_selection_choice([first, second])

    def test_sample_micro_checkpoint_study_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = _make_study(
                root, "first", residual_weight=0.01, primary_values=(1.0, 1.0, 1.0)
            )
            second = _make_study(
                root, "second", residual_weight=0.02, primary_values=(1.1, 1.1, 1.1)
            )
            manifest_path = first / STUDY_MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["checkpoint_selection_metric"]["aggregation"] = "sample_micro"
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(
                SelectionComparisonError, "validation equal-route macro ADE"
            ):
                build_selection_choice([first, second])

    def test_ties_use_lower_weight_then_stable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            higher_weight = _make_study(
                root,
                "z_higher_weight",
                residual_weight=0.05,
                primary_values=(0.8, 1.0, 1.2),
            )
            stable_first = _make_study(
                root,
                "a_lower_weight",
                residual_weight=0.02,
                primary_values=(1.2, 1.0, 0.8),
            )
            stable_second = _make_study(
                root,
                "b_lower_weight",
                residual_weight=0.02,
                primary_values=(1.0, 1.0, 1.0),
            )

            first_order = build_selection_choice(
                [higher_weight, stable_second, stable_first]
            )
            reverse_order = build_selection_choice(
                [stable_first, stable_second, higher_weight]
            )
            expected = str(stable_first.resolve())
            self.assertEqual(first_order["chosen"]["study_dir"], expected)
            self.assertEqual(reverse_order["chosen"]["study_dir"], expected)
            self.assertEqual(first_order["chosen"]["training_residual_weight"], 0.02)
            self.assertAlmostEqual(
                first_order["chosen"]["primary_validation_route_macro_ade"]["std"],
                0.2,
            )


if __name__ == "__main__":
    unittest.main()
