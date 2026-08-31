import os
import random
import unittest

import numpy as np
import torch

from temporal_tf.engine import (
    CHECKPOINT_SELECTION_METRIC,
    DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
    _validation_route_macro_ade,
    deterministic_runtime_metadata,
    seed_everything,
)


class ValidationCheckpointMetricTest(unittest.TestCase):
    def test_equal_route_macro_not_sample_micro_and_locked_order(self):
        collected = {
            "sample_ids": ["a0", "a1", "a2", "b0"],
            "route_ids": ["route_a", "route_a", "route_a", "route_b"],
            "per_sample": {"ade": torch.tensor([0.0, 0.0, 0.0, 6.0])},
        }
        route_macro = _validation_route_macro_ade(
            collected,
            expected_sample_ids=collected["sample_ids"],
            expected_route_ids=collected["route_ids"],
        )
        self.assertEqual(route_macro, 3.0)
        self.assertEqual(float(collected["per_sample"]["ade"].mean()), 1.5)
        self.assertEqual(CHECKPOINT_SELECTION_METRIC["aggregation"], "equal_route_macro")

        with self.assertRaisesRegex(RuntimeError, "sample IDs/order"):
            _validation_route_macro_ade(
                collected,
                expected_sample_ids=["a1", "a0", "a2", "b0"],
                expected_route_ids=collected["route_ids"],
            )
        with self.assertRaisesRegex(RuntimeError, "route IDs/order"):
            _validation_route_macro_ade(
                collected,
                expected_sample_ids=collected["sample_ids"],
                expected_route_ids=["route_a", "route_b", "route_a", "route_a"],
            )


class SeedPolicyTest(unittest.TestCase):
    def test_seed_everything_enables_documented_deterministic_policy(self):
        seed_everything(123)
        first = (random.random(), float(np.random.rand()), torch.rand(3))
        seed_everything(123)
        second = (random.random(), float(np.random.rand()), torch.rand(3))

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], rtol=0.0, atol=0.0)
        metadata = deterministic_runtime_metadata()
        self.assertTrue(metadata["deterministic_algorithms_enabled"])
        self.assertTrue(metadata["deterministic_algorithms_warn_only"])
        self.assertTrue(metadata["cudnn_deterministic"])
        self.assertFalse(metadata["cudnn_benchmark"])
        self.assertFalse(metadata["cuda_matmul_allow_tf32"])
        self.assertFalse(metadata["cudnn_allow_tf32"])
        self.assertEqual(
            os.environ["CUBLAS_WORKSPACE_CONFIG"],
            DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
        )


if __name__ == "__main__":
    unittest.main()
