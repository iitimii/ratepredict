from __future__ import annotations

import sys
import unittest
from pathlib import Path


TRANSFORMER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRANSFORMER_DIR))

from build_dataset_csv import market_quality, stable_id, timestamp_value


class CanonicalCsvTests(unittest.TestCase):
    def test_stable_id_is_deterministic_and_sensitive_to_input(self) -> None:
        first = stable_id("cbn", "2026-01-01", "mpr")
        self.assertEqual(first, stable_id("cbn", "2026-01-01", "mpr"))
        self.assertNotEqual(first, stable_id("cbn", "2026-01-02", "mpr"))
        self.assertEqual(len(first), 24)

    def test_marks_known_market_quality_problems(self) -> None:
        self.assertEqual(
            market_quality(
                "quidax_usdtngn_btcngn_2h",
                {"high": 729_000, "low": 450, "close": 500},
                "high",
            ),
            "suspicious_bad_tick",
        )
        self.assertEqual(
            market_quality("quidax_runtime_2h", {"trade_count": 0}, "trade_count"),
            "unavailable_placeholder",
        )

    def test_normalizes_timestamp_to_utc(self) -> None:
        self.assertEqual(
            timestamp_value({"date": "2026-09-02"}),
            "2026-09-02T00:00:00+00:00",
        )
        self.assertEqual(timestamp_value({"date": float("nan")}), "")


if __name__ == "__main__":
    unittest.main()
