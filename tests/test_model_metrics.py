import unittest

import torch

from temporal_tf.losses import residual_adapter_loss
from temporal_tf.metrics import per_sample_metrics, worst_fraction_indices
from temporal_tf.model import TemporalResidualAdapter


class ModelMetricsTest(unittest.TestCase):
    def test_identity_initialization_all_modalities(self):
        current = torch.randn(2, 5, 2)
        past_trajectory = torch.randn(2, 4, 5, 2)
        past_bev = torch.randn(2, 4, 6, 8, 8)
        for use_trajectory, use_bev in ((False, False), (True, False), (False, True), (True, True)):
            model = TemporalResidualAdapter(
                num_waypoints=5,
                bev_channels=6,
                use_past_trajectory=use_trajectory,
                use_past_bev=use_bev,
                bev_compressed_channels=4,
                bev_pooled_size=2,
                bev_token_dim=8,
                hidden_dim=12,
                query_dim=10,
            )
            output = model(
                current_trajectory=current,
                past_trajectory=past_trajectory if use_trajectory else None,
                past_bev=past_bev if use_bev else None,
            )
            torch.testing.assert_close(output["trajectory"], current)
            self.assertTrue(torch.all((output["gate"] >= 0) & (output["gate"] <= 1)))

    def test_adapter_gets_gradients(self):
        model = TemporalResidualAdapter(
            num_waypoints=3,
            bev_channels=2,
            use_past_trajectory=True,
            use_past_bev=False,
            hidden_dim=8,
            query_dim=8,
        )
        output = model(
            current_trajectory=torch.zeros(2, 3, 2),
            past_trajectory=torch.randn(2, 4, 3, 2),
        )
        losses = residual_adapter_loss(output, torch.ones(2, 3, 2), residual_weight=0.05)
        losses["loss"].backward()
        self.assertGreater(model.delta_head.weight.grad.abs().sum().item(), 0.0)

    def test_metrics(self):
        target = torch.zeros(2, 3, 2)
        prediction = target.clone()
        prediction[1, :, 0] = torch.tensor([1.0, 2.0, 3.0])
        metrics = per_sample_metrics(prediction, target)
        self.assertEqual(metrics["ade"][0].item(), 0.0)
        self.assertAlmostEqual(metrics["ade"][1].item(), 2.0)
        self.assertAlmostEqual(metrics["fde"][1].item(), 3.0)
        self.assertIn("second_difference_error", metrics)
        indices = worst_fraction_indices(metrics["ade"], fraction=0.2)
        self.assertEqual(indices.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
