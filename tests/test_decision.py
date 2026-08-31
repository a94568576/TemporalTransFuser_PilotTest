import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from temporal_tf.decision import (
    DecisionInputError,
    evaluate_go_no_go,
    load_and_evaluate,
    render_decision_markdown,
    write_decision_outputs,
)


SEEDS = (17, 29, 43)


def _split(*, ade, fde, smoothness, routes=12, paired_upper=None):
    result = {
        "route_count": routes,
        "route_macro": {
            "ade": float(ade),
            "fde": float(fde),
            "smoothness": float(smoothness),
        },
    }
    if paired_upper is not None:
        result["paired_vs_baseline"] = {
            "delta_definition": "adapter_minus_baseline",
            "route_count": routes,
            "route_bootstrap_ci": {
                "ade": {
                    "upper": float(paired_upper),
                    "confidence_level": 0.95,
                    "cluster_unit": "route",
                }
            },
        }
    return result


def _study(*, past_ade=0.80, current_bev_ade=0.90, shuffled_ade=0.92, routes=12):
    seed_results = {}
    for index, seed in enumerate(SEEDS):
        adjustment = 0.01 * index
        baseline = _split(
            ade=1.00 + adjustment,
            fde=1.10 + adjustment,
            smoothness=0.100,
            routes=routes,
        )
        variants = {
            "current_only_matched": {
                "splits": {
                    "test": _split(
                        ade=0.95 + adjustment,
                        fde=1.00 + adjustment,
                        smoothness=0.100,
                        routes=routes,
                    )
                }
            },
            "current_bev": {
                "splits": {
                    "test": _split(
                        ade=current_bev_ade + adjustment,
                        fde=0.97 + adjustment,
                        smoothness=0.101,
                        routes=routes,
                    )
                }
            },
            "past_bev": {
                "splits": {
                    "test": _split(
                        ade=past_ade + adjustment,
                        fde=0.85 + adjustment,
                        smoothness=0.104,
                        routes=routes,
                        paired_upper=-0.02,
                    )
                }
            },
            "shuffled_past_bev": {
                "splits": {
                    "test": _split(
                        ade=shuffled_ade + adjustment,
                        fde=0.96 + adjustment,
                        smoothness=0.102,
                        routes=routes,
                    )
                }
            },
        }
        seed_results[str(seed)] = {
            "status": "completed",
            "evaluation_mode": "final",
            "test_opened": True,
            "evidence_level": "offline_cache_evaluation",
            "cache_source": {"kind": "frozen_tfpp", "target_semantics": "geometric_path"},
            "dataset_windows": {"test": 240},
            "baseline": {"test": baseline},
            "variants": variants,
        }
    return {
        "schema_version": 1,
        "status": "completed",
        "evaluation_mode": "final_multiseed",
        "single_test_open_event": True,
        "study_id": "study-fixed",
        "cache_index_sha256": "a" * 64,
        "seeds": list(SEEDS),
        "variants": [
            "current_only_matched",
            "current_bev",
            "past_bev",
            "shuffled_past_bev",
        ],
        "primary_metric": "ade",
        "checkpoint_selection_metric": {
            "split": "val",
            "metric": "ade",
            "aggregation": "equal_route_macro",
        },
        "seed_results": seed_results,
        # Deliberately wrong: the evaluator must recompute from seed results.
        "method_aggregates": {"past_bev": {"route_macro_primary": {"mean": 999.0}}},
    }


class GoNoGoDecisionTest(unittest.TestCase):
    def test_go_when_every_fixed_gate_passes(self):
        decision = evaluate_go_no_go(_study())
        self.assertEqual(decision["status"], "go")
        self.assertTrue(decision["summary"]["all_hard_gates_pass"])
        self.assertAlmostEqual(decision["method_means"]["past_bev"]["ade"], 0.81)
        self.assertNotEqual(decision["method_means"]["past_bev"]["ade"], 999.0)
        self.assertEqual(decision["study"]["test_route_count_per_seed"], 12)
        self.assertEqual(decision["verdict_scope"], "exploratory_offline_adapter_pilot")
        self.assertEqual(decision["authoritative_primary_method"], "past_bev")
        self.assertEqual(decision["combined_role"], "diagnostic_only")
        self.assertFalse(decision["paper_go"])
        self.assertFalse(decision["closed_loop_go"])
        markdown = render_decision_markdown(decision)
        self.assertIn("Exploratory pilot decision: `go`", markdown)
        self.assertIn("not a paper or closed-loop GO", markdown)

    def test_three_route_go_is_explicitly_not_paper_evidence(self):
        decision = evaluate_go_no_go(_study(routes=3))
        self.assertEqual(decision["status"], "go")
        self.assertFalse(decision["paper_go"])
        self.assertTrue(
            any(
                "Exactly three independent test routes" in limitation
                for limitation in decision["limitations"]
            )
        )

    def test_ambiguous_when_core_bev_gates_pass_but_temporal_controls_fail(self):
        decision = evaluate_go_no_go(
            _study(past_ade=0.88, current_bev_ade=0.86, shuffled_ade=0.87)
        )
        self.assertEqual(decision["status"], "ambiguous")
        self.assertTrue(decision["summary"]["core_bev_gates_pass"])
        self.assertFalse(decision["summary"]["temporal_specific_gates_pass"])
        self.assertEqual(
            set(decision["summary"]["failed_gates"]),
            {"ade_vs_current_bev", "ade_vs_shuffled_past_bev"},
        )

    def test_no_go_when_non_temporal_core_gate_fails(self):
        decision = evaluate_go_no_go(_study(past_ade=0.94))
        self.assertEqual(decision["status"], "no_go")
        self.assertFalse(
            decision["gates"]["ade_3pct_vs_current_only_matched"]["passed"]
        )

    def test_no_go_when_any_seed_direction_or_bootstrap_ci_fails(self):
        study = _study()
        seed = str(SEEDS[-1])
        past_split = study["seed_results"][seed]["variants"]["past_bev"]["splits"]["test"]
        past_split["route_macro"]["ade"] = 1.02
        past_split["paired_vs_baseline"]["route_bootstrap_ci"]["ade"]["upper"] = 0.001
        decision = evaluate_go_no_go(study)
        self.assertEqual(decision["status"], "no_go")
        self.assertFalse(decision["gates"]["ade_direction_every_seed"]["passed"])
        self.assertFalse(decision["gates"]["bootstrap_ci_every_seed"]["passed"])

    def test_missing_or_invalid_inputs_raise_specific_errors(self):
        missing = _study()
        del missing["seed_results"]["29"]["variants"]["current_bev"]
        with self.assertRaisesRegex(DecisionInputError, "current_bev"):
            evaluate_go_no_go(missing)

        malformed_ci = _study()
        interval = malformed_ci["seed_results"]["17"]["variants"]["past_bev"]["splits"][
            "test"
        ]["paired_vs_baseline"]["route_bootstrap_ci"]["ade"]
        interval["cluster_unit"] = "window"
        with self.assertRaisesRegex(DecisionInputError, "cluster_unit must be route"):
            evaluate_go_no_go(malformed_ci)

        synthetic = _study()
        synthetic["seed_results"]["17"]["evidence_level"] = "synthetic_smoke"
        with self.assertRaisesRegex(DecisionInputError, "synthetic smoke is forbidden"):
            evaluate_go_no_go(synthetic)

        too_few = _study()
        too_few["seeds"] = [17, 29]
        too_few["seed_results"] = {
            key: value for key, value in too_few["seed_results"].items() if key != "43"
        }
        with self.assertRaisesRegex(DecisionInputError, "at least 3 seeds"):
            evaluate_go_no_go(too_few)

        sample_micro = _study()
        sample_micro["checkpoint_selection_metric"]["aggregation"] = "sample_micro"
        with self.assertRaisesRegex(DecisionInputError, "equal-route macro ADE"):
            evaluate_go_no_go(sample_micro)

    def test_file_outputs_do_not_touch_marker_and_refuse_marker_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "study_results.json"
            source.write_text(json.dumps(_study()), encoding="utf-8")
            marker = root / "study_test_opened.marker.json"
            marker.write_text("locked\n", encoding="utf-8")

            decision, source_hash = load_and_evaluate(source)
            self.assertEqual(decision["source"]["sha256"], source_hash)
            json_path, markdown_path = write_decision_outputs(
                decision,
                json_output=root / "GO_NO_GO.json",
                markdown_output=root / "GO_NO_GO.md",
            )
            self.assertEqual(marker.read_text(encoding="utf-8"), "locked\n")
            self.assertEqual(json.loads(json_path.read_text())["status"], "go")
            self.assertIn("Limitations", markdown_path.read_text())

            with self.assertRaisesRegex(ValueError, "test marker"):
                write_decision_outputs(
                    decision,
                    json_output=marker,
                    markdown_output=root / "other.md",
                    overwrite=True,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "locked\n")

    def test_input_object_is_not_mutated(self):
        study = _study()
        original = deepcopy(study)
        evaluate_go_no_go(study)
        self.assertEqual(study, original)


if __name__ == "__main__":
    unittest.main()
