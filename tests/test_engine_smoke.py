import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import torch

from temporal_tf.config import DEFAULTS
from temporal_tf.engine import (
    CHECKPOINT_SELECTION_METRIC,
    SELECTION_MANIFEST,
    STUDY_SELECTION_OWNER_KIND,
    TEST_OPEN_MARKER,
    _STUDY_FINALIZE_CAPABILITY,
    finalize_selection,
    run_pilot,
)
from temporal_tf.synthetic import generate_synthetic_cache


class EngineSmokeTest(unittest.TestCase):
    def test_study_owner_requires_capability_but_authorized_final_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            selection = root / "owned_selection"
            generate_synthetic_cache(
                cache,
                num_routes=6,
                frames_per_route=6,
                num_waypoints=3,
                bev_channels=8,
                bev_size=4,
                seed=7,
            )
            config = deepcopy(DEFAULTS)
            config["seed"] = 7
            config["data"]["history_length"] = 2
            config["adapter"].update(
                {
                    "bev_compressed_channels": 2,
                    "bev_pooled_size": 2,
                    "bev_token_dim": 4,
                    "hidden_dim": 6,
                    "query_dim": 4,
                }
            )
            config["training"].update(
                {"epochs": 1, "batch_size": 8, "torch_num_threads": 1, "num_workers": 0}
            )
            owner = {
                "kind": STUDY_SELECTION_OWNER_KIND,
                "study_owner_id": "a" * 64,
            }
            run_pilot(
                cache_root=cache,
                output_dir=selection,
                config=config,
                device_name="cpu",
                variants=("current_only",),
                selection_owner=owner,
            )

            with self.assertRaisesRegex(RuntimeError, "cannot be finalized standalone"):
                finalize_selection(
                    selection_dir=selection,
                    cache_root=cache,
                    output_dir=root / "standalone_final",
                    device_name="cpu",
                )
            self.assertFalse((selection / TEST_OPEN_MARKER).exists())

            result = finalize_selection(
                selection_dir=selection,
                cache_root=cache,
                output_dir=root / "authorized_final",
                device_name="cpu",
                _study_owner_id=owner["study_owner_id"],
                _study_finalize_capability=_STUDY_FINALIZE_CAPABILITY,
            )
            self.assertTrue(result["test_opened"])
            self.assertEqual(result["selection_owner"], owner)

    def test_one_epoch_controls_and_locked_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            output = root / "output"
            generate_synthetic_cache(
                cache,
                num_routes=6,
                frames_per_route=8,
                num_waypoints=4,
                bev_channels=8,
                bev_size=6,
                seed=5,
            )
            config = deepcopy(DEFAULTS)
            config["seed"] = 5
            config["data"]["history_length"] = 3
            config["adapter"].update(
                {
                    "bev_compressed_channels": 4,
                    "bev_pooled_size": 2,
                    "bev_token_dim": 8,
                    "hidden_dim": 12,
                    "query_dim": 8,
                }
            )
            config["training"].update(
                {"epochs": 1, "batch_size": 8, "torch_num_threads": 1, "num_workers": 0}
            )
            control_variants = [
                "current_only_matched",
                "repeat_current",
                "current_bev",
                "shuffled_past_bev",
            ]
            result = run_pilot(
                cache_root=cache,
                output_dir=output,
                config=config,
                device_name="cpu",
                variants=control_variants,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["evidence_level"], "synthetic_smoke")
            self.assertFalse(result["test_opened"])
            self.assertNotIn("test", result["baseline"])
            self.assertTrue((output / "results.json").is_file())
            for variant_name in control_variants:
                self.assertTrue((output / "checkpoints" / f"{variant_name}.pt").is_file())
                self.assertIn("input_semantics", result["variants"][variant_name])
                history = result["variants"][variant_name]["history"]
                self.assertAlmostEqual(
                    history[-1]["val_route_macro_ade"],
                    result["variants"][variant_name]["splits"]["val"]["route_macro"]["ade"],
                )
                checkpoint = torch.load(
                    output / "checkpoints" / f"{variant_name}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                self.assertEqual(
                    checkpoint["selection_metric"], CHECKPOINT_SELECTION_METRIC
                )
                self.assertEqual(
                    checkpoint["best_val_route_macro_ade"], checkpoint["best_val_ade"]
                )
            matched = result["variants"]["current_only_matched"]
            self.assertEqual(matched["capacity_match"]["target_variant"], "past_bev")
            self.assertLess(
                abs(matched["capacity_match"]["relative_parameter_difference"]), 0.01
            )
            self.assertTrue(
                result["variants"]["repeat_current"]["capacity_match"][
                    "exact_state_shape_match"
                ]
            )
            self.assertTrue((output / SELECTION_MANIFEST).is_file())
            with self.assertRaises(FileExistsError):
                run_pilot(
                    cache_root=cache,
                    output_dir=output,
                    config=config,
                    device_name="cpu",
                    variants=control_variants,
                )

            mismatched_config = deepcopy(config)
            mismatched_config["training"]["residual_weight"] = 0.01
            with self.assertRaisesRegex(ValueError, "config.*hash"):
                finalize_selection(
                    selection_dir=output,
                    cache_root=cache,
                    output_dir=root / "mismatched_final",
                    config=mismatched_config,
                    device_name="cpu",
                )
            self.assertFalse((output / TEST_OPEN_MARKER).exists())

            manifest_path = output / SELECTION_MANIFEST
            locked_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            altered_manifest = deepcopy(locked_manifest)
            altered_manifest["variant_specs"]["current_bev"]["bev_history"] = "past"
            manifest_path.write_text(json.dumps(altered_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input semantics"):
                finalize_selection(
                    selection_dir=output,
                    cache_root=cache,
                    output_dir=root / "semantics_mismatched_final",
                    config=config,
                    device_name="cpu",
                )
            self.assertFalse((output / TEST_OPEN_MARKER).exists())
            manifest_path.write_text(json.dumps(locked_manifest), encoding="utf-8")

            altered_manifest = deepcopy(locked_manifest)
            altered_manifest["checkpoint_selection_metric"]["aggregation"] = (
                "sample_micro"
            )
            manifest_path.write_text(json.dumps(altered_manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "equal-route macro ADE"):
                finalize_selection(
                    selection_dir=output,
                    cache_root=cache,
                    output_dir=root / "metric_mismatched_final",
                    config=config,
                    device_name="cpu",
                )
            self.assertFalse((output / TEST_OPEN_MARKER).exists())
            manifest_path.write_text(json.dumps(locked_manifest), encoding="utf-8")

            final_result = finalize_selection(
                selection_dir=output,
                cache_root=cache,
                output_dir=root / "final_output",
                config=config,
                device_name="cpu",
            )
            self.assertTrue(final_result["test_opened"])
            self.assertIn("test", final_result["baseline"])
            self.assertEqual(set(final_result["variants"]), set(control_variants))
            self.assertEqual(
                final_result["checkpoint_selection_metric"],
                CHECKPOINT_SELECTION_METRIC,
            )
            self.assertTrue((output / TEST_OPEN_MARKER).is_file())
            with self.assertRaisesRegex(RuntimeError, "already been opened"):
                finalize_selection(
                    selection_dir=output,
                    cache_root=cache,
                    output_dir=root / "second_final_output",
                    config=config,
                    device_name="cpu",
                )


if __name__ == "__main__":
    unittest.main()
