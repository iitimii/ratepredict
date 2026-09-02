from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


def _names_tuple(names: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(name) for name in names)
    if not result:
        raise ValueError("feature names must not be empty")
    if len(set(result)) != len(result):
        raise ValueError("feature names must be unique")
    return result


def _values_array(values: np.ndarray, feature_count: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0 or array.shape[-1] != feature_count:
        raise ValueError(
            f"values must have {feature_count} features in the final dimension"
        )
    return array


@dataclass(frozen=True)
class NamedStandardizer:
    """Mean/std normalization whose feature order is part of the contract."""

    names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        names: Sequence[str],
        valid_mask: np.ndarray | None = None,
    ) -> NamedStandardizer:
        feature_names = _names_tuple(names)
        array = _values_array(values, len(feature_names))
        valid = np.isfinite(array)
        if valid_mask is not None:
            supplied_mask = np.asarray(valid_mask, dtype=bool)
            if supplied_mask.shape != array.shape:
                raise ValueError("valid_mask must have the same shape as values")
            valid &= supplied_mask

        flat_values = array.reshape(-1, len(feature_names))
        flat_valid = valid.reshape(-1, len(feature_names))
        counts = flat_valid.sum(axis=0)
        if (counts == 0).any():
            missing = [
                name for name, count in zip(feature_names, counts) if count == 0
            ]
            raise ValueError(
                f"features have no finite training values: {', '.join(missing)}"
            )

        sums = np.where(flat_valid, flat_values, 0.0).sum(axis=0)
        mean = sums / counts
        centered = np.where(flat_valid, flat_values - mean, 0.0)
        variance = (centered * centered).sum(axis=0) / counts
        scale = np.sqrt(variance)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(names=feature_names, mean=mean, scale=scale)

    def _validate_names(self, names: Sequence[str]) -> None:
        if tuple(names) != self.names:
            raise ValueError(
                f"feature names must match fitted order {self.names}, got {tuple(names)}"
            )

    def transform(
        self,
        values: np.ndarray,
        names: Sequence[str],
        valid_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        self._validate_names(names)
        array = _values_array(values, len(self.names))
        valid = np.isfinite(array)
        if valid_mask is not None:
            supplied_mask = np.asarray(valid_mask, dtype=bool)
            if supplied_mask.shape != array.shape:
                raise ValueError("valid_mask must have the same shape as values")
            valid &= supplied_mask
        normalized = (array - self.mean) / self.scale
        return np.where(valid, normalized, 0.0)

    def inverse_transform(
        self,
        values: np.ndarray,
        names: Sequence[str],
    ) -> np.ndarray:
        self._validate_names(names)
        array = _values_array(values, len(self.names))
        return array * self.scale + self.mean

    def state_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> NamedStandardizer:
        names = _names_tuple(state["names"])
        mean = np.asarray(state["mean"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        expected_shape = (len(names),)
        if mean.shape != expected_shape or scale.shape != expected_shape:
            raise ValueError("normalizer state dimensions do not match feature names")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("normalizer state must contain finite values")
        if (scale <= 0.0).any():
            raise ValueError("normalizer scale values must be positive")
        return cls(names=names, mean=mean, scale=scale)


@dataclass(frozen=True)
class DatasetNormalizers:
    quantitative: NamedStandardizer
    target: NamedStandardizer

    def state_dict(self) -> dict[str, object]:
        return {
            "quantitative": self.quantitative.state_dict(),
            "target": self.target.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> DatasetNormalizers:
        return cls(
            quantitative=NamedStandardizer.from_state_dict(state["quantitative"]),
            target=NamedStandardizer.from_state_dict(state["target"]),
        )
