import unittest
from copy import deepcopy

from temporal_tf.config import DEFAULTS, validate_config
from temporal_tf.synthetic import generate_synthetic_cache


class ConfigSyntheticTest(unittest.TestCase):
    def test_post_override_config_validation(self):
        config = deepcopy(DEFAULTS)
        config["training"]["epochs"] = 0
        with self.assertRaisesRegex(ValueError, "epochs"):
            validate_config(config)
        config = deepcopy(DEFAULTS)
        config["training"]["residual_weight"] = -0.1
        with self.assertRaisesRegex(ValueError, "residual_weight"):
            validate_config(config)
        config = deepcopy(DEFAULTS)
        config["adapter"]["dropout"] = 1.0
        with self.assertRaisesRegex(ValueError, "dropout"):
            validate_config(config)
        config = deepcopy(DEFAULTS)
        config["training"]["torch_num_threads"] = 0
        with self.assertRaisesRegex(ValueError, "torch_num_threads"):
            validate_config(config)

    def test_synthetic_dimensions_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "bev_channels"):
            generate_synthetic_cache("unused", bev_channels=4)
        with self.assertRaisesRegex(ValueError, "num_waypoints"):
            generate_synthetic_cache("unused", num_waypoints=0)


if __name__ == "__main__":
    unittest.main()
