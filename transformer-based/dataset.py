from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from labels import LabelSchema, TARGET_NAMES
from normalization import DatasetNormalizers, NamedStandardizer


@dataclass(frozen=True)
class DatasetConfig:
    lookback_days: int = 365
    horizon_days: int = 7
    avg_rate_column: str = "avg_rate"
    high_rate_column: str = "high_rate"
    low_rate_column: str = "low_rate"

    def __post_init__(self) -> None:
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be a positive integer")
        if self.horizon_days <= 0:
            raise ValueError("horizon_days must be a positive integer")
        if len(set(self.target_rate_columns)) != 3:
            raise ValueError("target rate column names must be unique")

    @property
    def target_rate_columns(self) -> tuple[str, str, str]:
        return (
            self.avg_rate_column,
            self.high_rate_column,
            self.low_rate_column,
        )


@dataclass(frozen=True)
class PreparedDailyData:
    dates: pd.DatetimeIndex
    quantitative_values: np.ndarray
    quantitative_mask: np.ndarray
    target_rates: np.ndarray
    target_rate_mask: np.ndarray
    text_values: np.ndarray
    text_mask: np.ndarray
    quantitative_feature_names: tuple[str, ...]
    text_source_names: tuple[str, ...]
    embedding_columns: tuple[str, ...]
    target_rate_names: tuple[str, str, str]


def _validate_daily_index(index: pd.Index, source: str) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{source} index must be a DatetimeIndex")
    if index.tz is None:
        raise ValueError(f"{source} index must be timezone-aware UTC")
    if str(index.tz) != "UTC":
        raise ValueError(f"{source} index timezone must be UTC")
    if index.has_duplicates:
        raise ValueError(f"{source} index contains duplicate dates")
    if not index.equals(index.normalize()):
        raise ValueError(f"{source} index timestamps must be midnight UTC")


def _numeric_values(frame: pd.DataFrame, source: str) -> np.ndarray:
    try:
        return frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} values must be numeric") from exc


def prepare_daily_data(
    quantitative: pd.DataFrame,
    text_embeddings: Mapping[str, pd.DataFrame],
    quantitative_features: Sequence[str],
    config: DatasetConfig,
) -> PreparedDailyData:
    """Align quantitative values and precomputed embeddings to one UTC calendar."""
    if quantitative.empty:
        raise ValueError("quantitative data must not be empty")
    _validate_daily_index(quantitative.index, "quantitative")

    feature_names = tuple(str(name) for name in quantitative_features)
    if not feature_names:
        raise ValueError("quantitative feature names must not be empty")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("quantitative feature names must be unique")

    required_columns = feature_names + config.target_rate_columns
    missing_columns = [
        column for column in required_columns if column not in quantitative.columns
    ]
    if missing_columns:
        raise ValueError(
            f"quantitative data is missing columns: {', '.join(missing_columns)}"
        )

    ordered_quantitative = quantitative.sort_index()
    dates = pd.date_range(
        ordered_quantitative.index.min(),
        ordered_quantitative.index.max(),
        freq="D",
        tz="UTC",
    )
    aligned_quantitative = ordered_quantitative.reindex(dates)
    quantitative_values = _numeric_values(
        aligned_quantitative.loc[:, feature_names], "quantitative"
    )
    quantitative_mask = np.isfinite(quantitative_values)
    target_rates = _numeric_values(
        aligned_quantitative.loc[:, config.target_rate_columns], "target rate"
    )
    target_rate_mask = np.isfinite(target_rates)

    if any(not isinstance(name, str) or not name for name in text_embeddings):
        raise ValueError("text source names must be non-empty strings")
    source_names = tuple(sorted(text_embeddings))
    embedding_columns: tuple[str, ...] = ()
    aligned_text_values: list[np.ndarray] = []
    aligned_text_masks: list[np.ndarray] = []

    for source_name in source_names:
        frame = text_embeddings[source_name]
        _validate_daily_index(frame.index, f"text source '{source_name}'")
        columns = tuple(str(column) for column in frame.columns)
        if not columns:
            raise ValueError(f"text source '{source_name}' has no embedding columns")
        if not embedding_columns:
            embedding_columns = columns
        elif columns != embedding_columns:
            raise ValueError(
                "text sources must use identical ordered embedding columns"
            )

        aligned_frame = frame.sort_index().reindex(dates)
        values = _numeric_values(aligned_frame, f"text source '{source_name}'")
        valid_rows = np.isfinite(values).all(axis=1)
        aligned_text_values.append(np.where(valid_rows[:, None], values, 0.0))
        aligned_text_masks.append(valid_rows)

    if source_names:
        text_values = np.stack(aligned_text_values, axis=1)
        text_mask = np.stack(aligned_text_masks, axis=1)
    else:
        text_values = np.empty((len(dates), 0, 0), dtype=np.float64)
        text_mask = np.empty((len(dates), 0), dtype=bool)

    return PreparedDailyData(
        dates=dates,
        quantitative_values=quantitative_values,
        quantitative_mask=quantitative_mask,
        target_rates=target_rates,
        target_rate_mask=target_rate_mask,
        text_values=text_values,
        text_mask=text_mask,
        quantitative_feature_names=feature_names,
        text_source_names=source_names,
        embedding_columns=embedding_columns,
        target_rate_names=config.target_rate_columns,
    )


def discover_valid_sample_end_dates(
    prepared: PreparedDailyData,
    config: DatasetConfig,
) -> pd.DatetimeIndex:
    """Return forecast origins with a full calendar window and complete targets."""
    if prepared.target_rate_names != config.target_rate_columns:
        raise ValueError("dataset target rate columns do not match prepared data")

    positions: list[int] = []
    first_position = config.lookback_days - 1
    stop_position = len(prepared.dates) - config.horizon_days
    for position in range(first_position, stop_position):
        current_average_available = prepared.target_rate_mask[position, 0]
        future_targets_available = prepared.target_rate_mask[
            position + 1 : position + 1 + config.horizon_days
        ].all()
        if current_average_available and future_targets_available:
            positions.append(position)

    return prepared.dates.take(positions)


def _validated_origin_positions(
    prepared: PreparedDailyData,
    sample_end_dates: Sequence[pd.Timestamp],
    config: DatasetConfig,
    *,
    context: str,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    requested_values = list(sample_end_dates)
    if not requested_values:
        raise ValueError(f"{context} requires at least one sample end date")

    try:
        requested = pd.DatetimeIndex(requested_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} sample end dates must be timestamps") from exc
    if requested.tz is None or str(requested.tz) != "UTC":
        raise ValueError(f"{context} sample end dates must be timezone-aware UTC")
    if requested.has_duplicates:
        raise ValueError(f"{context} sample end dates must be unique")
    if not requested.equals(requested.normalize()):
        raise ValueError(f"{context} sample end dates must be midnight UTC")
    requested = requested.sort_values()

    valid_dates = discover_valid_sample_end_dates(prepared, config)
    invalid = requested.difference(valid_dates)
    if len(invalid):
        rendered = ", ".join(timestamp.isoformat() for timestamp in invalid)
        raise ValueError(f"invalid sample end dates for {context}: {rendered}")

    positions = prepared.dates.get_indexer(requested)
    return requested, positions.astype(np.int64, copy=False)


def _raw_target_for_position(
    prepared: PreparedDailyData,
    position: int,
    config: DatasetConfig,
) -> np.ndarray:
    future = prepared.target_rates[
        position + 1 : position + 1 + config.horizon_days
    ]
    return LabelSchema.from_rates(
        present_average_rate=float(prepared.target_rates[position, 0]),
        future_average_rates=future[:, 0],
        future_high_rates=future[:, 1],
        future_low_rates=future[:, 2],
    ).as_array(dtype=np.float64)


def fit_train_normalizers(
    prepared: PreparedDailyData,
    training_sample_end_dates: Sequence[pd.Timestamp],
    config: DatasetConfig,
) -> DatasetNormalizers:
    """Fit quantitative and target statistics from explicit training origins."""
    _, origin_positions = _validated_origin_positions(
        prepared,
        training_sample_end_dates,
        config,
        context="training normalizer fitting",
    )
    input_positions = np.unique(
        np.concatenate(
            [
                np.arange(
                    position - config.lookback_days + 1,
                    position + 1,
                    dtype=np.int64,
                )
                for position in origin_positions
            ]
        )
    )
    quantitative = NamedStandardizer.fit(
        prepared.quantitative_values[input_positions],
        prepared.quantitative_feature_names,
        prepared.quantitative_mask[input_positions],
    )
    raw_targets = np.stack(
        [
            _raw_target_for_position(prepared, int(position), config)
            for position in origin_positions
        ]
    )
    target = NamedStandardizer.fit(raw_targets, TARGET_NAMES)
    return DatasetNormalizers(quantitative=quantitative, target=target)


class DailyMultimodalDataset(Dataset):
    """Lazy daily quantitative/text windows with normalized future targets."""

    def __init__(
        self,
        prepared: PreparedDailyData,
        sample_end_dates: Sequence[pd.Timestamp],
        normalizers: DatasetNormalizers,
        config: DatasetConfig,
    ) -> None:
        if normalizers.quantitative.names != prepared.quantitative_feature_names:
            raise ValueError(
                "quantitative normalizer feature names do not match prepared data"
            )
        if normalizers.target.names != TARGET_NAMES:
            raise ValueError("target normalizer feature names do not match targets")

        validated_dates, positions = _validated_origin_positions(
            prepared,
            sample_end_dates,
            config,
            context="dataset",
        )
        self.prepared = prepared
        self.sample_end_dates = validated_dates
        self.normalizers = normalizers
        self.config = config
        self._origin_positions = positions

    def __len__(self) -> int:
        return len(self._origin_positions)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        origin = int(self._origin_positions[index])
        start = origin - self.config.lookback_days + 1
        stop = origin + 1

        quantitative_mask = self.prepared.quantitative_mask[start:stop]
        quantitative = self.normalizers.quantitative.transform(
            self.prepared.quantitative_values[start:stop],
            self.prepared.quantitative_feature_names,
            quantitative_mask,
        )
        text = self.prepared.text_values[start:stop]
        text_mask = self.prepared.text_mask[start:stop]
        raw_target = _raw_target_for_position(self.prepared, origin, self.config)
        target = self.normalizers.target.transform(
            raw_target,
            TARGET_NAMES,
        )

        return {
            "quantitative": torch.from_numpy(
                np.ascontiguousarray(quantitative, dtype=np.float32)
            ),
            "quantitative_mask": torch.from_numpy(
                np.ascontiguousarray(quantitative_mask, dtype=bool)
            ),
            "text": torch.from_numpy(np.ascontiguousarray(text, dtype=np.float32)),
            "text_mask": torch.from_numpy(
                np.ascontiguousarray(text_mask, dtype=bool)
            ),
            "target": torch.from_numpy(
                np.ascontiguousarray(target, dtype=np.float32)
            ),
            "as_of": torch.tensor(
                self.prepared.dates[origin].value,
                dtype=torch.int64,
            ),
        }
