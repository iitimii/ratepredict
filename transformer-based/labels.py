from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


TARGET_NAMES = ("avg_change", "high_change", "low_change")


@dataclass(frozen=True)
class DailyLabel:
    """USDT/NGN changes for one day in the seven-day forecast horizon."""

    day: int
    avg_change: float
    high_change: float
    low_change: float

    @classmethod
    def from_rates(
        cls,
        *,
        day: int,
        present_average_rate: float,
        future_average_rate: float,
        future_high_rate: float,
        future_low_rate: float,
    ) -> DailyLabel:
        """Calculate one day's labels as absolute USDT/NGN rate differences."""
        return cls(
            day=day,
            avg_change=future_average_rate - present_average_rate,
            high_change=future_high_rate - future_average_rate,
            low_change=future_low_rate - future_average_rate,
        )


@dataclass(frozen=True)
class LabelSchema:
    """Complete set of daily targets for a configurable forecast horizon."""

    labels: tuple[DailyLabel, ...]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("labels must contain at least one future day")
        expected_days = list(range(1, len(self.labels) + 1))
        actual_days = [label.day for label in self.labels]
        if actual_days != expected_days:
            raise ValueError("labels must contain ordered consecutive future days")

    @property
    def horizon_days(self) -> int:
        return len(self.labels)

    @classmethod
    def from_rates(
        cls,
        present_average_rate: float,
        future_average_rates: Sequence[float],
        future_high_rates: Sequence[float],
        future_low_rates: Sequence[float],
    ) -> LabelSchema:
        lengths = {
            len(future_average_rates),
            len(future_high_rates),
            len(future_low_rates),
        }
        if lengths == {0}:
            raise ValueError("rates must contain at least one future day")
        if len(lengths) != 1:
            raise ValueError("future rate sequences must have the same length")

        return cls(
            tuple(
                DailyLabel.from_rates(
                    day=index + 1,
                    present_average_rate=present_average_rate,
                    future_average_rate=float(average),
                    future_high_rate=float(high),
                    future_low_rate=float(low),
                )
                for index, (average, high, low) in enumerate(
                    zip(future_average_rates, future_high_rates, future_low_rates)
                )
            )
        )

    def as_array(self, *, dtype: np.dtype = np.float32) -> np.ndarray:
        return np.asarray(
            [
                [label.avg_change, label.high_change, label.low_change]
                for label in self.labels
            ],
            dtype=dtype,
        )
