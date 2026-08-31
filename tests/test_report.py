import unittest

from temporal_tf.report import render_markdown


def _metrics(ade: float) -> dict[str, float]:
    return {
        "ade": ade,
        "fde": ade + 1.0,
        "waypoint_l1": ade / 2.0,
        "smoothness": 0.1,
        "second_difference_error": 0.2,
    }


class ReportTest(unittest.TestCase):
    def _base_result(self) -> dict:
        baseline_split = {
            "overall": _metrics(9.0),
            "route_count": 2,
            "route_macro": _metrics(1.0),
            "route_quantiles": {"ade": {"p90": 2.0, "p95": 3.0}},
            "sample_quantiles": {"ade": {"p90": 8.0, "p95": 9.0}},
            "baseline_worst_slice": _metrics(10.0),
        }
        variant_split = {
            "overall": _metrics(8.0),
            "route_count": 2,
            "route_macro": _metrics(0.8),
            "route_quantiles": {"ade": {"p90": 1.8, "p95": 2.8}},
            "sample_quantiles": {"ade": {"p90": 7.0, "p95": 8.0}},
            "baseline_worst_slice": _metrics(9.0),
            "paired_vs_baseline": {
                "primary_metric": "ade",
                "route_macro_delta": {"ade": -0.2},
                "route_bootstrap_ci": {"ade": {"lower": -0.3, "upper": -0.1}},
                "route_fractions": {"improved": 0.75, "harmed": 0.25, "unchanged": 0.0},
                "sample_fractions": {"improved": 0.6, "harmed": 0.3, "unchanged": 0.1},
            },
        }
        return {
            "evidence_level": "offline_cache_evaluation",
            "cache_source": {"kind": "frozen_tfpp", "target_semantics": "geometric_path"},
            "device": "cpu",
            "evaluation_mode": "final",
            "test_opened": True,
            "selection_id": "selection",
            "dataset_windows": {"test": 10},
            "baseline": {"test": baseline_split},
            "variants": {
                "combined": {
                    "trainable_parameters": 123,
                    "splits": {"test": variant_split},
                }
            },
        }

    def test_route_macro_and_paired_statistics_are_rendered(self):
        rendered = render_markdown(self._base_result())
        self.assertIn("Primary aggregation: equally weighted route macro", rendered)
        self.assertIn("| Baseline | 0 | 2 | 1.0000 |", rendered)
        self.assertIn("| combined | -0.2000 | [-0.3000, -0.1000] | 75.0% | 25.0% | 60.0% | 30.0% |", rendered)
        self.assertIn("Sample-micro ADE", rendered)
        self.assertIn("Route-mean ADE P90", rendered)
        self.assertIn("Route-cluster bootstrap 95% CI", rendered)

    def test_legacy_result_schema_still_renders(self):
        result = self._base_result()
        baseline = result["baseline"]["test"]
        variant = result["variants"]["combined"]["splits"]["test"]
        for split_result in (baseline, variant):
            split_result.pop("route_count")
            split_result.pop("route_macro")
            split_result.pop("route_quantiles")
            split_result.pop("sample_quantiles")
        variant.pop("paired_vs_baseline")
        variant["paired_delta_ade_vs_baseline"] = -1.0
        variant["fraction_improved_ade"] = 0.6

        rendered = render_markdown(result)
        self.assertIn("legacy sample micro", rendered)
        self.assertIn("| combined | -1.0000 | — | — | — | 60.0% | — |", rendered)


if __name__ == "__main__":
    unittest.main()
