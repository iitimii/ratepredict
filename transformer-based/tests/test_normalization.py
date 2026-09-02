from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TRANSFORMER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRANSFORMER_DIR))

from normalization import DatasetNormalizers, NamedStandardizer


class NamedStandardizerTests(unittest.TestCase):
    def test_ignores_missing_values_zero_fills_and_inverse_transforms(self) -> None:
        values = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 10.0]])
        mask = np.isfinite(values)
        scaler = NamedStandardizer.fit(values, ("moving", "constant"), mask)

        transformed = scaler.transform(values, ("moving", "constant"), mask)

        np.testing.assert_allclose(scaler.mean, [3.0, 10.0])
        np.testing.assert_allclose(scaler.scale, [np.sqrt(8.0 / 3.0), 1.0])
        self.assertEqual(transformed[1, 1], 0.0)
        np.testing.assert_allclose(
            scaler.inverse_transform(transformed[[0, 2]], ("moving", "constant")),
            values[[0, 2]],
        )

    def test_state_round_trip_and_name_validation(self) -> None:
        scaler = NamedStandardizer.fit(np.array([[1.0], [3.0]]), ("rate",))
        restored = NamedStandardizer.from_state_dict(scaler.state_dict())
        np.testing.assert_allclose(restored.mean, scaler.mean)
        np.testing.assert_allclose(restored.scale, scaler.scale)

        with self.assertRaisesRegex(ValueError, "feature names"):
            restored.transform(np.array([[2.0]]), ("wrong",))

        pair = DatasetNormalizers(quantitative=scaler, target=scaler)
        restored_pair = DatasetNormalizers.from_state_dict(pair.state_dict())
        self.assertEqual(restored_pair.quantitative.names, ("rate",))
        self.assertEqual(restored_pair.target.names, ("rate",))

    def test_rejects_feature_without_finite_training_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "no finite training values"):
            NamedStandardizer.fit(np.array([[np.nan], [np.nan]]), ("empty",))


if __name__ == "__main__":
    unittest.main()
