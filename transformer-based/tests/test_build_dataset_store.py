from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


TRANSFORMER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRANSFORMER_DIR))

from build_dataset_store import optional_year, typed_record, valid_quidax_rows


class ParquetDatasetStoreTests(unittest.TestCase):
    def test_typed_record_converts_nullable_values(self) -> None:
        record = typed_record(
            {
                "record_id": "abc",
                "observed_at": "2026-09-02T00:00:00+00:00",
                "period_year": "2026",
                "modality": "quantitative",
                "source_id": "example",
                "record_type": "feature_observation",
                "feature_name": "example.value",
                "numeric_value": "1.25",
                "text_value": "",
                "title": "",
                "url": "",
                "unit": "percent",
                "frequency": "daily",
                "quality_status": "unreviewed",
                "source_path": "example.csv",
                "metadata_json": "{}",
            }
        )

        self.assertEqual(record["period_year"], 2026)
        self.assertEqual(record["numeric_value"], 1.25)
        self.assertIsNone(record["text_value"])
        self.assertEqual(record["observed_at"].isoformat(), "2026-09-02T00:00:00+00:00")

    def test_rejects_invalid_period_years(self) -> None:
        self.assertIsNone(optional_year(""))
        self.assertIsNone(optional_year("not-a-year"))
        self.assertIsNone(optional_year("1200"))

    def test_filters_malformed_market_bars(self) -> None:
        frame = pd.DataFrame(
            {
                "bucket_2h": ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"],
                "open": [500, 501],
                "high": [729_000, 510],
                "low": [490, 495],
                "close": [500, 505],
            }
        )

        filtered = valid_quidax_rows(frame)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["high"], 510)


if __name__ == "__main__":
    unittest.main()
