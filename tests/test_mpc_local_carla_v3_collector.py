"""Offline contract tests for the fresh Town05 V3 CARLA collector."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_PATH = ROOT / "scripts" / "collect_mpc_local_carla_v3.py"
CONFIG_PATH = ROOT / "configs" / "mpc_local_grounding_pilot_v3.yaml"


def _load_collector():
    specification = importlib.util.spec_from_file_location(
        "mpc_local_carla_v3_collector_test", COLLECTOR_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not import V3 collector")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class CollectorConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collector = _load_collector()

    def test_frozen_config_and_protocol_match_collector_contract(self):
        config, digest = self.collector._load_and_validate_config(CONFIG_PATH)
        self.assertEqual(digest, hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest())
        self.assertEqual(config["collection"]["outputs"], self.collector.OUTPUT_FILES)
        self.assertTrue(config["collection"]["sealed_test_redaction"])
        self.assertEqual(
            config["collection"]["public_manifest_redaction"],
            self.collector.PUBLIC_MANIFEST_REDACTION,
        )

        protocol = self.collector._protocol(config, smoke=False)
        self.assertEqual(protocol["map"], self.collector.MAP_NAME)
        self.assertEqual(protocol["action_profile"], "support_stratified_v3")
        self.assertTrue(protocol["sealed_test_redaction"])
        self.assertEqual(protocol["collision_gate_scope"], "development_only")
        self.assertTrue(
            protocol["state_selection"]["map_disjoint_from_parents"]["passed"]
        )
        self.assertTrue(protocol["state_selection"]["waypoint_state_source_passed"])

    def test_base_collector_keeps_town10_default(self):
        parameter = inspect.signature(
            self.collector.base.CarlaSession.__init__
        ).parameters["map_name"]
        self.assertEqual(parameter.default, self.collector.base.MAP_NAME)

        v1 = self.collector.base._protocol(
            False,
            self.collector.base.DEFAULT_SEED,
            self.collector.base.DEFAULT_ACTION_PROFILE,
        )
        v2 = self.collector.base._protocol(
            False,
            self.collector.base.SAFE_LOCAL_V2_SEED,
            "safe_local_v2",
        )
        self.assertEqual(v1["map"], self.collector.base.MAP_NAME)
        self.assertEqual(v2["map"], self.collector.base.MAP_NAME)
        self.assertEqual(v1["action_profile"], "v1")
        self.assertEqual(v2["action_profile"], "safe_local_v2")
        self.assertEqual(v1["cem"]["lower"], self.collector.base.CEM_LOWER.tolist())
        self.assertEqual(
            v2["cem"]["lower"], self.collector.base.SAFE_LOCAL_V2_LOWER.tolist()
        )
        self.assertEqual(v1["state_selection"]["excluded_source_spawn_indices"], [])
        self.assertEqual(
            v2["state_selection"]["excluded_source_spawn_indices"],
            list(self.collector.base.SAFE_LOCAL_V2_EXCLUDED_V1_SPAWN_INDICES),
        )
        self.assertNotIn("sealed_test_redaction", v1)
        self.assertNotIn("sealed_test_redaction", v2)

    def test_sealed_progress_redacts_id_cost_collision_and_distribution(self):
        state = SimpleNamespace(state_id=59, split="test")
        message = self.collector.base._collection_progress_message(
            state,
            state_count=60,
            iteration=2,
            sealed_test_redaction=True,
        )
        self.assertEqual(message, "split=test iter=2 sealed_progress=true")
        for forbidden in ("59", "best", "cost", "collision", "std", "outcome"):
            self.assertNotIn(forbidden, message.lower())


class StateDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collector = _load_collector()

    def test_full_slots_are_unique_and_balanced_in_every_split(self):
        slots = self.collector._state_slots(False, self.collector.COLLECTION_SEED)
        self.assertEqual(len(slots), 60)
        self.assertEqual([slot.state_id for slot in slots], list(range(60)))

        for split, expected in self.collector.FULL_SPLIT_COUNTS.items():
            selected = [slot for slot in slots if slot.split == split]
            self.assertEqual(len(selected), expected)
            for values, levels in (
                ([slot.curvature_stratum for slot in selected], range(4)),
                ([slot.initial_speed_mps for slot in selected], (4.0, 6.0, 8.0)),
                (
                    [slot.lateral_offset_m for slot in selected],
                    (-0.25, 0.0, 0.25),
                ),
            ):
                counts = Counter(values)
                self.assertEqual(set(counts), set(levels))
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

        self.assertEqual(
            Counter(slot.initial_speed_mps for slot in slots),
            Counter({4.0: 20, 6.0: 20, 8.0: 20}),
        )
        self.assertEqual(
            Counter(slot.lateral_offset_m for slot in slots),
            Counter({-0.25: 20, 0.0: 20, 0.25: 20}),
        )

    def test_smoke_spans_three_splits_and_support_extremes(self):
        slots = self.collector._state_slots(True, self.collector.COLLECTION_SEED)
        self.assertEqual(
            [
                (
                    slot.split,
                    slot.curvature_stratum,
                    slot.initial_speed_mps,
                    slot.lateral_offset_m,
                )
                for slot in slots
            ],
            [
                ("train", 0, 4.0, -0.25),
                ("val", 2, 6.0, 0.0),
                ("test", 3, 8.0, 0.25),
            ],
        )

    def test_development_and_sealed_payloads_are_physically_partitioned(self):
        states = [
            SimpleNamespace(split="train", state_identity_sha256="train"),
            SimpleNamespace(split="val", state_identity_sha256="val"),
            SimpleNamespace(split="test", state_identity_sha256="test"),
        ]
        arrays = {
            "real_cost": np.arange(18, dtype=np.float32).reshape(3, 3, 2),
            "action_params": np.arange(72, dtype=np.float32).reshape(3, 3, 2, 4),
        }
        partitions = self.collector._split_file_payloads(states, arrays)
        development_states, development = partitions["development"]
        sealed_states, sealed = partitions["sealed_test"]

        self.assertEqual([state.split for state in development_states], ["train", "val"])
        self.assertEqual([state.split for state in sealed_states], ["test"])
        np.testing.assert_array_equal(development["real_cost"], arrays["real_cost"][:2])
        np.testing.assert_array_equal(sealed["real_cost"], arrays["real_cost"][2:])
        self.assertFalse(np.shares_memory(development["real_cost"], arrays["real_cost"]))
        self.assertFalse(np.shares_memory(sealed["real_cost"], arrays["real_cost"]))


if __name__ == "__main__":
    unittest.main()
