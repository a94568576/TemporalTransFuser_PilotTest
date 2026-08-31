import math
import unittest

import torch

from temporal_tf.geometry import transform_trajectory_between_egos


class GeometryTest(unittest.TestCase):
    def test_identity(self):
        trajectory = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
        pose = torch.tensor([4.0, 5.0, 0.7])
        actual = transform_trajectory_between_egos(trajectory, pose, pose)
        torch.testing.assert_close(actual, trajectory, atol=1e-6, rtol=1e-6)

    def test_translation(self):
        trajectory = torch.tensor([[2.0, 0.0]])
        actual = transform_trajectory_between_egos(
            trajectory,
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        )
        torch.testing.assert_close(actual, torch.tensor([[1.0, 0.0]]))

    def test_ninety_degree_target(self):
        trajectory = torch.tensor([[1.0, 0.0]])
        actual = transform_trajectory_between_egos(
            trajectory,
            torch.tensor([0.0, 0.0, 0.0]),
            torch.tensor([0.0, 0.0, math.pi / 2]),
        )
        torch.testing.assert_close(actual, torch.tensor([[0.0, -1.0]]), atol=1e-6, rtol=1e-6)

    def test_round_trip(self):
        trajectory = torch.tensor([[1.2, -0.4], [3.0, 2.0]])
        source = torch.tensor([2.0, -3.0, 0.4])
        target = torch.tensor([-1.0, 5.0, -1.1])
        transformed = transform_trajectory_between_egos(trajectory, source, target)
        restored = transform_trajectory_between_egos(transformed, target, source)
        torch.testing.assert_close(restored, trajectory, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
