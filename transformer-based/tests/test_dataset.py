from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


TRANSFORMER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRANSFORMER_DIR))

from dataset import (
    DailyMultimodalDataset,
    DatasetConfig,
    discover_valid_sample_end_dates,
    fit_train_normalizers,
    prepare_daily_data,
)
from labels import TARGET_NAMES
from normalization import DatasetNormalizers, NamedStandardizer


class DailyDataPreparationTests(unittest.TestCase):
    def base_quantitative(self, periods: int = 8) -> pd.DataFrame:
        index = pd.date_range("2026-01-01", periods=periods, freq="D", tz="UTC")
        return pd.DataFrame(
            {
                "mpr": np.linspace(25.0, 26.0, periods),
                "volume": np.linspace(100.0, 200.0, periods),
                "avg_rate": np.linspace(1_500.0, 1_507.0, periods),
                "high_rate": np.linspace(1_510.0, 1_517.0, periods),
                "low_rate": np.linspace(1_490.0, 1_497.0, periods),
            },
            index=index,
        )

    def test_aligns_daily_modalities_masks_missing_inputs_and_discovers_origins(self) -> None:
        complete = self.base_quantitative()
        quantitative = complete.drop(complete.index[2]).copy()
        quantitative.loc[complete.index[3], "mpr"] = np.nan
        quantitative.loc[complete.index[6], ["avg_rate", "high_rate", "low_rate"]] = np.nan

        businessday = pd.DataFrame(
            np.arange(16, dtype=float).reshape(8, 2),
            index=complete.index,
            columns=["e0", "e1"],
        )
        vanguard = businessday.drop(complete.index[1]).copy() + 100.0

        config = DatasetConfig(lookback_days=2, horizon_days=2)
        prepared = prepare_daily_data(
            quantitative=quantitative,
            text_embeddings={"vanguard": vanguard, "businessday": businessday},
            quantitative_features=("mpr", "volume"),
            config=config,
        )

        self.assertEqual(prepared.quantitative_feature_names, ("mpr", "volume"))
        self.assertEqual(prepared.text_source_names, ("businessday", "vanguard"))
        self.assertEqual(prepared.embedding_columns, ("e0", "e1"))
        self.assertEqual(prepared.quantitative_values.shape, (8, 2))
        self.assertEqual(prepared.text_values.shape, (8, 2, 2))
        self.assertFalse(prepared.quantitative_mask[2].any())
        self.assertFalse(prepared.quantitative_mask[3, 0])
        self.assertFalse(prepared.text_mask[1, 1])
        np.testing.assert_array_equal(prepared.text_values[1, 1], [0.0, 0.0])

        valid_dates = discover_valid_sample_end_dates(prepared, config)

        self.assertEqual(valid_dates.tolist(), [complete.index[3]])

    def test_supports_quantitative_only_inputs(self) -> None:
        quantitative = self.base_quantitative(periods=5)

        prepared = prepare_daily_data(
            quantitative=quantitative,
            text_embeddings={},
            quantitative_features=("mpr",),
            config=DatasetConfig(lookback_days=2, horizon_days=1),
        )

        self.assertEqual(prepared.text_values.shape, (5, 0, 0))
        self.assertEqual(prepared.text_mask.shape, (5, 0))
        self.assertEqual(prepared.text_source_names, ())
        self.assertEqual(prepared.embedding_columns, ())

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "lookback_days"):
            DatasetConfig(lookback_days=0)
        with self.assertRaisesRegex(ValueError, "horizon_days"):
            DatasetConfig(horizon_days=0)

    def test_rejects_invalid_quantitative_indices(self) -> None:
        cases = {
            "DatetimeIndex": self.base_quantitative().set_axis(range(8)),
            "timezone-aware UTC": self.base_quantitative().tz_localize(None),
            "timezone must be UTC": self.base_quantitative().tz_convert("Africa/Lagos"),
            "midnight": self.base_quantitative().set_axis(
                self.base_quantitative().index + pd.Timedelta(hours=1)
            ),
            "duplicate": pd.concat(
                [self.base_quantitative(), self.base_quantitative().iloc[[0]]]
            ),
        }

        for message, frame in cases.items():
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    prepare_daily_data(
                        quantitative=frame,
                        text_embeddings={},
                        quantitative_features=("mpr",),
                        config=DatasetConfig(lookback_days=2, horizon_days=1),
                    )

    def test_rejects_missing_columns_and_incompatible_text_embeddings(self) -> None:
        quantitative = self.base_quantitative()
        config = DatasetConfig(lookback_days=2, horizon_days=1)

        with self.assertRaisesRegex(ValueError, "missing columns.*unknown"):
            prepare_daily_data(quantitative, {}, ("unknown",), config)

        with self.assertRaisesRegex(ValueError, "missing columns.*high_rate"):
            prepare_daily_data(
                quantitative.drop(columns="high_rate"),
                {},
                ("mpr",),
                config,
            )

        first = pd.DataFrame(
            1.0,
            index=quantitative.index,
            columns=["e0", "e1"],
        )
        second = pd.DataFrame(
            1.0,
            index=quantitative.index,
            columns=["e1", "e0"],
        )
        with self.assertRaisesRegex(ValueError, "ordered embedding columns"):
            prepare_daily_data(
                quantitative,
                {"first": first, "second": second},
                ("mpr",),
                config,
            )

        duplicate_text = pd.concat([first, first.iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "text source 'first'.*duplicate"):
            prepare_daily_data(
                quantitative,
                {"first": duplicate_text},
                ("mpr",),
                    config,
                )


class LazyDatasetTests(unittest.TestCase):
    def make_inputs(
        self,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], DatasetConfig]:
        dates = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
        average = np.arange(100.0, 112.0)
        quantitative = pd.DataFrame(
            {
                "feature": np.arange(1.0, 13.0),
                "avg_rate": average,
                "high_rate": average + 2.0,
                "low_rate": average - 3.0,
            },
            index=dates,
        )
        quantitative.loc[dates[9], "feature"] = 10_000.0
        quantitative.loc[dates[9], ["avg_rate", "high_rate", "low_rate"]] = [
            50_000.0,
            75_000.0,
            25_000.0,
        ]
        base_embeddings = np.arange(24, dtype=float).reshape(12, 2)
        text_embeddings = {
            "vanguard": pd.DataFrame(
                base_embeddings + 100.0,
                index=dates,
                columns=["e0", "e1"],
            ),
            "businessday": pd.DataFrame(
                base_embeddings,
                index=dates,
                columns=["e0", "e1"],
            ),
        }
        return (
            quantitative,
            text_embeddings,
            DatasetConfig(lookback_days=3, horizon_days=2),
        )

    def test_fits_normalizers_from_unique_training_dates_and_targets_only(self) -> None:
        quantitative, text_embeddings, config = self.make_inputs()
        prepared = prepare_daily_data(
            quantitative,
            text_embeddings,
            ("feature",),
            config,
        )

        normalizers = fit_train_normalizers(
            prepared,
            training_sample_end_dates=prepared.dates[[2, 3, 4]],
            config=config,
        )

        self.assertEqual(normalizers.quantitative.mean[0], 3.0)
        np.testing.assert_allclose(normalizers.target.mean, [1.5, 2.0, -3.0])

    def test_returns_exact_lazy_windows_as_typed_tensors(self) -> None:
        quantitative, text_embeddings, config = self.make_inputs()
        prepared = prepare_daily_data(
            quantitative,
            text_embeddings,
            ("feature",),
            config,
        )
        normalizers = fit_train_normalizers(
            prepared,
            training_sample_end_dates=prepared.dates[[2, 3, 4]],
            config=config,
        )
        training_dataset = DailyMultimodalDataset(
            prepared,
            sample_end_dates=prepared.dates[[4]],
            normalizers=normalizers,
            config=config,
        )

        sample = training_dataset[0]

        self.assertEqual(len(training_dataset), 1)
        self.assertEqual(
            set(sample),
            {"quantitative", "quantitative_mask", "text", "text_mask", "target", "as_of"},
        )
        self.assertEqual(tuple(sample["quantitative"].shape), (3, 1))
        self.assertEqual(tuple(sample["quantitative_mask"].shape), (3, 1))
        self.assertEqual(tuple(sample["text"].shape), (3, 2, 2))
        self.assertEqual(tuple(sample["text_mask"].shape), (3, 2))
        self.assertEqual(tuple(sample["target"].shape), (2, 3))
        self.assertEqual(sample["quantitative"].dtype, torch.float32)
        self.assertEqual(sample["quantitative_mask"].dtype, torch.bool)
        self.assertEqual(sample["text"].dtype, torch.float32)
        self.assertEqual(sample["text_mask"].dtype, torch.bool)
        self.assertEqual(sample["target"].dtype, torch.float32)
        self.assertEqual(sample["as_of"].dtype, torch.int64)
        self.assertEqual(int(sample["as_of"]), prepared.dates[4].value)
        np.testing.assert_allclose(
            normalizers.quantitative.inverse_transform(
                sample["quantitative"].numpy(),
                prepared.quantitative_feature_names,
            ),
            [[3.0], [4.0], [5.0]],
        )
        np.testing.assert_allclose(
            normalizers.target.inverse_transform(
                sample["target"].numpy(),
                TARGET_NAMES,
            ),
            [[1.0, 2.0, -3.0], [2.0, 2.0, -3.0]],
        )
        self.assertEqual(training_dataset.prepared.quantitative_values.ndim, 2)
        self.assertEqual(training_dataset.prepared.text_values.ndim, 3)

    def test_rejects_empty_invalid_or_normalizer_incompatible_origins(self) -> None:
        quantitative, text_embeddings, config = self.make_inputs()
        prepared = prepare_daily_data(
            quantitative,
            text_embeddings,
            ("feature",),
            config,
        )
        normalizers = fit_train_normalizers(
            prepared,
            training_sample_end_dates=prepared.dates[[2, 3, 4]],
            config=config,
        )

        with self.assertRaisesRegex(ValueError, "at least one sample end date"):
            DailyMultimodalDataset(prepared, [], normalizers, config)

        with self.assertRaisesRegex(ValueError, "invalid sample end dates"):
            DailyMultimodalDataset(
                prepared,
                [prepared.dates[-1]],
                normalizers,
                config,
            )

        incompatible = DatasetNormalizers(
            quantitative=NamedStandardizer.fit(
                np.array([[1.0], [2.0]]),
                ("wrong",),
            ),
            target=normalizers.target,
        )
        with self.assertRaisesRegex(ValueError, "quantitative normalizer"):
            DailyMultimodalDataset(
                prepared,
                [prepared.dates[4]],
                incompatible,
                config,
            )


if __name__ == "__main__":
    unittest.main()
