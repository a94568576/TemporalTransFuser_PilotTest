import unittest

import torch

from temporal_tf.metrics import quantile_metrics
from temporal_tf.route_stats import paired_route_comparison, summarize_route_metrics


class RouteStatisticsTest(unittest.TestCase):
    def test_route_macro_gives_each_route_equal_weight(self):
        per_sample = {
            "ade": torch.tensor([4.0, 0.0, 2.0]),
            "fde": torch.tensor([5.0, 1.0, 3.0]),
        }
        summary = summarize_route_metrics(per_sample, ["route_b", "route_a", "route_a"])

        self.assertEqual(summary["route_count"], 2)
        self.assertAlmostEqual(summary["route_macro"]["ade"], 2.5)
        self.assertAlmostEqual(summary["route_macro"]["fde"], 3.5)
        self.assertAlmostEqual(summary["per_route_metrics"]["route_a"]["ade"], 1.0)
        self.assertAlmostEqual(summary["per_route_metrics"]["route_b"]["ade"], 4.0)
        self.assertAlmostEqual(summary["route_quantiles"]["ade"]["p90"], 3.7)
        self.assertAlmostEqual(summary["route_quantiles"]["ade"]["p95"], 3.85)
        self.assertAlmostEqual(summary["sample_quantiles"]["ade"]["p90"], 3.6)
        self.assertNotAlmostEqual(
            summary["route_macro"]["ade"], float(per_sample["ade"].mean().item())
        )

    def test_paired_route_delta_bootstrap_and_fractions(self):
        baseline = {
            "ade": torch.tensor([1.0, 3.0, 5.0]),
            "fde": torch.tensor([2.0, 4.0, 6.0]),
        }
        adapter = {
            "ade": torch.tensor([0.0, 4.0, 4.0]),
            "fde": torch.tensor([1.0, 5.0, 5.0]),
        }
        route_ids = ["route_a", "route_a", "route_b"]

        first = paired_route_comparison(
            adapter,
            baseline,
            route_ids,
            bootstrap_seed=23,
            bootstrap_resamples=2_000,
        )
        second = paired_route_comparison(
            adapter,
            baseline,
            route_ids,
            bootstrap_seed=23,
            bootstrap_resamples=2_000,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["delta_definition"], "adapter_minus_baseline")
        self.assertAlmostEqual(first["per_route_delta"]["route_a"]["ade"], 0.0)
        self.assertAlmostEqual(first["per_route_delta"]["route_b"]["ade"], -1.0)
        self.assertAlmostEqual(first["route_macro_delta"]["ade"], -0.5)
        self.assertEqual(first["route_bootstrap_ci"]["ade"]["cluster_unit"], "route")
        self.assertAlmostEqual(first["route_bootstrap_ci"]["ade"]["lower"], -1.0)
        self.assertAlmostEqual(first["route_bootstrap_ci"]["ade"]["upper"], 0.0)
        self.assertAlmostEqual(first["sample_fractions"]["improved"], 2.0 / 3.0)
        self.assertAlmostEqual(first["sample_fractions"]["harmed"], 1.0 / 3.0)
        self.assertAlmostEqual(first["route_fractions"]["improved"], 0.5)
        self.assertAlmostEqual(first["route_fractions"]["harmed"], 0.0)
        self.assertAlmostEqual(first["route_fractions"]["unchanged"], 0.5)

    def test_bootstrap_is_invariant_to_sample_order(self):
        baseline = {"ade": torch.tensor([1.0, 3.0, 5.0, 9.0])}
        adapter = {"ade": torch.tensor([0.0, 4.0, 4.0, 7.0])}
        routes = ["route_a", "route_a", "route_b", "route_c"]
        permutation = torch.tensor([3, 1, 0, 2])

        original = paired_route_comparison(
            adapter,
            baseline,
            routes,
            bootstrap_seed=7,
            bootstrap_resamples=1_000,
        )
        permuted = paired_route_comparison(
            {"ade": adapter["ade"][permutation]},
            {"ade": baseline["ade"][permutation]},
            [routes[index] for index in permutation.tolist()],
            bootstrap_seed=7,
            bootstrap_resamples=1_000,
        )

        self.assertEqual(original, permuted)

    def test_metric_mapping_order_does_not_break_pairing(self):
        result = paired_route_comparison(
            {
                "ade": torch.tensor([0.5]),
                "fde": torch.tensor([1.5]),
            },
            {
                "fde": torch.tensor([2.0]),
                "ade": torch.tensor([1.0]),
            },
            ["route"],
            bootstrap_seed=11,
            bootstrap_resamples=10,
        )
        self.assertAlmostEqual(result["route_macro_delta"]["ade"], -0.5)
        self.assertAlmostEqual(result["route_macro_delta"]["fde"], -0.5)

    def test_single_route_has_degenerate_interval(self):
        result = paired_route_comparison(
            {"ade": torch.tensor([0.5, 1.5])},
            {"ade": torch.tensor([1.0, 2.0])},
            ["only", "only"],
            bootstrap_seed=5,
            bootstrap_resamples=50,
        )
        interval = result["route_bootstrap_ci"]["ade"]
        self.assertAlmostEqual(interval["estimate"], -0.5)
        self.assertAlmostEqual(interval["lower"], -0.5)
        self.assertAlmostEqual(interval["upper"], -0.5)

    def test_quantile_and_input_validation(self):
        quantiles = quantile_metrics({"ade": torch.tensor([0.0, 10.0])})
        self.assertAlmostEqual(quantiles["ade"]["p90"], 9.0)
        self.assertAlmostEqual(quantiles["ade"]["p95"], 9.5)
        with self.assertRaisesRegex(ValueError, "3 samples"):
            summarize_route_metrics({"ade": torch.tensor([1.0, 2.0])}, ["a", "b", "c"])
        with self.assertRaisesRegex(ValueError, "metric names"):
            paired_route_comparison(
                {"ade": torch.tensor([1.0])},
                {"fde": torch.tensor([1.0])},
                ["a"],
                bootstrap_seed=1,
            )
        with self.assertRaisesRegex(ValueError, "num_resamples"):
            paired_route_comparison(
                {"ade": torch.tensor([1.0])},
                {"ade": torch.tensor([1.0])},
                ["a"],
                bootstrap_seed=1,
                bootstrap_resamples=0,
            )


if __name__ == "__main__":
    unittest.main()
