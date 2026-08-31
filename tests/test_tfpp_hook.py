import unittest

import torch
from torch import nn

from temporal_tf.tfpp_hook import FeatureCapture, select_base_path


class FakeBackbone(nn.Module):
    def forward(self, x):
        return x + 1.0, x + 2.0, x + 3.0


class FakeTFPP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = FakeBackbone()
        self.change_channel = nn.Conv2d(4, 5, kernel_size=1)

    def forward(self, x):
        bev, fused, _ = self.backbone(x)
        _planner = self.change_channel(fused)
        path = torch.zeros(x.shape[0], 10, 2)
        return None, torch.zeros(x.shape[0], 8), path


class TFPPHookTest(unittest.TestCase):
    def test_released_checkpoint_output_selection(self):
        outputs = (None, torch.zeros(1, 8), torch.ones(1, 10, 2))
        path, name = select_base_path(outputs)
        self.assertEqual(name, "pred_checkpoint")
        self.assertEqual(tuple(path.shape), (1, 10, 2))

    def test_old_waypoint_output_selection(self):
        outputs = (torch.ones(1, 4, 2), None, None)
        path, name = select_base_path(outputs)
        self.assertEqual(name, "pred_wp")
        self.assertEqual(tuple(path.shape), (1, 4, 2))

    def test_backbone_and_planner_hooks(self):
        model = FakeTFPP()
        x = torch.zeros(2, 4, 8, 8)
        with FeatureCapture(model, "backbone_bev", cache_spatial_size=4) as capture:
            model(x)
            feature = capture.pop()
        self.assertEqual(tuple(feature.shape), (2, 4, 4, 4))
        self.assertTrue(torch.all(feature == 1.0))

        with FeatureCapture(model, "planner_grid", cache_spatial_size=None) as capture:
            model(x)
            feature = capture.pop()
        self.assertEqual(tuple(feature.shape), (2, 5, 8, 8))

        with self.assertRaisesRegex(ValueError, "refusing to upsample"):
            with FeatureCapture(model, "planner_grid", cache_spatial_size=16):
                model(x)


if __name__ == "__main__":
    unittest.main()
