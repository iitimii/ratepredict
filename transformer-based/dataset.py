from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


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
