from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TRANSFORMER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRANSFORMER_DIR))

from labels import LabelSchema, TARGET_NAMES


class LabelSchemaTests(unittest.TestCase):
    def test_builds_configurable_horizon_with_three_absolute_changes(self) -> None:
        schema = LabelSchema.from_rates(
            present_average_rate=1_500.0,
            future_average_rates=[1_512.0, 1_490.0],
            future_high_rates=[1_520.0, 1_505.0],
            future_low_rates=[1_498.0, 1_480.0],
        )

        self.assertEqual(schema.horizon_days, 2)
        self.assertEqual(TARGET_NAMES, ("avg_change", "high_change", "low_change"))
        np.testing.assert_array_equal(
            schema.as_array(),
            np.array([[12.0, 8.0, -14.0], [-10.0, 15.0, -10.0]], dtype=np.float32),
        )

    def test_rejects_empty_or_mismatched_future_rates(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one future day"):
            LabelSchema.from_rates(1_500.0, [], [], [])

        with self.assertRaisesRegex(ValueError, "same length"):
            LabelSchema.from_rates(
                1_500.0,
                [1_510.0],
                [1_520.0, 1_521.0],
                [1_500.0],
            )


if __name__ == "__main__":
    unittest.main()
