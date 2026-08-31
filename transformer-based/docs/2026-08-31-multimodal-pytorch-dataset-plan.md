# Multimodal Daily PyTorch Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lazy, daily, multimodal PyTorch dataset that maps a configurable historical quantitative/text-embedding window to configurable future USDT/NGN average/high/low change targets.

**Architecture:** Align quantitative data and per-source precomputed text embeddings once on a complete UTC daily calendar, retain missingness masks, and store only valid forecast-origin positions. Fit named quantitative and target standardizers from explicit training origins, then slice and normalize overlapping windows lazily in a map-style PyTorch dataset.

**Tech Stack:** Python 3.10+, pandas, NumPy, PyTorch, standard-library `unittest`

**Spec:** `transformer-based/docs/2026-08-31-multimodal-pytorch-dataset-design.md`

## Global Constraints

- All implementation, dependency declarations, and tests stay under `transformer-based/`.
- Root application code and the existing 42-feature model pipeline remain unchanged.
- Input timestamps are unique UTC midnight calendar dates.
- Default `lookback_days` is `365`; default `horizon_days` is `7`; both must accept any positive integer.
- Quantitative and target normalization is fitted only from explicit training forecast origins.
- Text inputs are precomputed embeddings with identical ordered embedding columns for every source.
- Missing inputs remain usable through masks; incomplete target windows are excluded.
- Labels are absolute rate differences, not percentages.
- Follow strict red-green-refactor TDD and commit after each task.

---

## File Structure

- Modify `transformer-based/labels.py`: configurable horizon label objects and array conversion.
- Create `transformer-based/normalization.py`: named train-only standardizers and serializable combined state.
- Create `transformer-based/dataset.py`: configuration, aligned base storage, valid-origin discovery, normalizer fitting, and lazy PyTorch dataset.
- Create `transformer-based/requirements.txt`: isolated NumPy, pandas, and PyTorch dependencies.
- Create `transformer-based/tests/test_labels.py`: configurable label formula tests.
- Create `transformer-based/tests/test_normalization.py`: fit/transform/inverse/state tests.
- Create `transformer-based/tests/test_dataset.py`: alignment, masks, split safety, shape, and lazy-window tests.

---

### Task 1: Configurable Multi-Horizon Labels

**Files:**
- Modify: `transformer-based/labels.py`
- Create: `transformer-based/tests/test_labels.py`

**Interfaces:**
- Produces: `TARGET_NAMES: tuple[str, str, str]`
- Produces: `DailyLabel.from_rates(day, present_average_rate, future_average_rate, future_high_rate, future_low_rate) -> DailyLabel`
- Produces: `LabelSchema.from_rates(present_average_rate, future_average_rates, future_high_rates, future_low_rates) -> LabelSchema`
- Produces: `LabelSchema.horizon_days -> int`
- Produces: `LabelSchema.as_array(dtype=np.float32) -> np.ndarray` with shape `[horizon_days, 3]`

- [ ] **Step 1: Create the isolated label tests with local module imports**

```python
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
            LabelSchema.from_rates(1_500.0, [1_510.0], [1_520.0, 1_521.0], [1_500.0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the label tests and verify the configurable API fails**

Run:

```bash
.venv/bin/python -m unittest discover -s transformer-based/tests -p 'test_labels.py'
```

Expected: FAIL because `TARGET_NAMES`, `LabelSchema.from_rates`, and `LabelSchema.as_array` do not exist and the current horizon is fixed at seven.

- [ ] **Step 3: Implement the minimal configurable label schema**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

TARGET_NAMES = ("avg_change", "high_change", "low_change")


@dataclass(frozen=True)
class DailyLabel:
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
    ) -> "DailyLabel":
        return cls(
            day=day,
            avg_change=future_average_rate - present_average_rate,
            high_change=future_high_rate - future_average_rate,
            low_change=future_low_rate - future_average_rate,
        )


@dataclass(frozen=True)
class LabelSchema:
    labels: tuple[DailyLabel, ...]

    def __post_init__(self) -> None:
        expected = list(range(1, len(self.labels) + 1))
        if not self.labels:
            raise ValueError("labels must contain at least one future day")
        if [label.day for label in self.labels] != expected:
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
    ) -> "LabelSchema":
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
                    future_average_rate=float(avg),
                    future_high_rate=float(high),
                    future_low_rate=float(low),
                )
                for index, (avg, high, low) in enumerate(
                    zip(future_average_rates, future_high_rates, future_low_rates)
                )
            )
        )

    def as_array(self, *, dtype: np.dtype = np.float32) -> np.ndarray:
        return np.asarray(
            [[label.avg_change, label.high_change, label.low_change] for label in self.labels],
            dtype=dtype,
        )
```

- [ ] **Step 4: Run the label tests and verify they pass**

Run:

```bash
.venv/bin/python -m unittest discover -s transformer-based/tests -p 'test_labels.py'
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit the configurable label component**

```bash
git add transformer-based/labels.py transformer-based/tests/test_labels.py
git commit -m "feat: add configurable daily forecast labels"
```

---

### Task 2: Serializable Named Train-Only Normalizers

**Files:**
- Create: `transformer-based/normalization.py`
- Create: `transformer-based/tests/test_normalization.py`

**Interfaces:**
- Produces: `NamedStandardizer.fit(values, names, valid_mask=None) -> NamedStandardizer`
- Produces: `NamedStandardizer.transform(values, names, valid_mask=None) -> np.ndarray`
- Produces: `NamedStandardizer.inverse_transform(values, names) -> np.ndarray`
- Produces: `NamedStandardizer.state_dict() -> dict[str, object]`
- Produces: `NamedStandardizer.from_state_dict(state) -> NamedStandardizer`
- Produces: `DatasetNormalizers(quantitative: NamedStandardizer, target: NamedStandardizer)` with serializable state

- [ ] **Step 1: Write normalizer behavior tests**

```python
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
        with self.assertRaisesRegex(ValueError, "feature names"):
            restored.transform(np.array([[2.0]]), ("wrong",))

        pair = DatasetNormalizers(quantitative=scaler, target=scaler)
        restored_pair = DatasetNormalizers.from_state_dict(pair.state_dict())
        self.assertEqual(restored_pair.quantitative.names, ("rate",))
        self.assertEqual(restored_pair.target.names, ("rate",))

    def test_rejects_feature_without_finite_training_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "no finite training values"):
            NamedStandardizer.fit(np.array([[np.nan], [np.nan]]), ("empty",))
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```bash
.venv/bin/python -m unittest discover -s transformer-based/tests -p 'test_normalization.py'
```

Expected: FAIL because `normalization.py` does not exist.

- [ ] **Step 3: Implement named standardization and state serialization**

Implement `NamedStandardizer` as an immutable dataclass containing `names`, `mean`, and `scale`. Flatten all leading dimensions during fitting while preserving the final feature dimension. Combine `valid_mask` with `np.isfinite(values)`, compute population standard deviation (`ddof=0`), replace zero scales with `1.0`, and raise with the offending feature names when a column has no valid training values.

The transformation must execute this exact behavior:

```python
valid = np.isfinite(values)
if valid_mask is not None:
    valid &= np.asarray(valid_mask, dtype=bool)
normalized = (values - self.mean) / self.scale
return np.where(valid, normalized, 0.0)
```

Serialize arrays as lists so state remains JSON-compatible:

```python
def state_dict(self) -> dict[str, object]:
    return {
        "names": list(self.names),
        "mean": self.mean.tolist(),
        "scale": self.scale.tolist(),
    }
```

`DatasetNormalizers.state_dict()` returns `{"quantitative": ..., "target": ...}` and `from_state_dict()` restores both named standardizers.

- [ ] **Step 4: Run the normalizer and label suites**

Run:

```bash
.venv/bin/python -m unittest discover -s transformer-based/tests -p 'test_*.py'
```

Expected: all label and normalization tests pass.

- [ ] **Step 5: Commit the normalizer component**

```bash
git add transformer-based/normalization.py transformer-based/tests/test_normalization.py
git commit -m "feat: add train-only dataset normalizers"
```

---

### Task 3: Daily Alignment and Valid Forecast-Origin Discovery

**Files:**
- Create: `transformer-based/dataset.py`
- Create: `transformer-based/tests/test_dataset.py`

**Interfaces:**
- Produces: `DatasetConfig(lookback_days=365, horizon_days=7, avg_rate_column="avg_rate", high_rate_column="high_rate", low_rate_column="low_rate")`
- Produces: `PreparedDailyData` with aligned base arrays and ordered metadata
- Produces: `prepare_daily_data(quantitative, text_embeddings, quantitative_features, config) -> PreparedDailyData`
- Produces: `discover_valid_sample_end_dates(prepared, config) -> pd.DatetimeIndex`

- [ ] **Step 1: Write alignment, mask, ordering, and origin-discovery tests**

Use eight UTC-midnight dates, omit one quantitative calendar date, and provide text mappings in reverse alphabetical order. Assert:

```python
config = DatasetConfig(lookback_days=2, horizon_days=2)
prepared = prepare_daily_data(
    quantitative=quantitative,
    text_embeddings={"vanguard": vanguard_embeddings, "businessday": businessday_embeddings},
    quantitative_features=("mpr", "volume"),
    config=config,
)

self.assertEqual(prepared.quantitative_feature_names, ("mpr", "volume"))
self.assertEqual(prepared.text_source_names, ("businessday", "vanguard"))
self.assertEqual(prepared.text_values.shape, (8, 2, 2))
self.assertFalse(prepared.quantitative_mask[2].any())
self.assertFalse(prepared.text_mask[1, 1])
```

Set all three target-source columns to missing on day six. For a two-day lookback and two-day horizon, assert that only origins with complete future targets are returned and that missing non-target features do not remove an otherwise valid origin.

Add rejection tests for:

```python
DatasetConfig(lookback_days=0)
DatasetConfig(horizon_days=0)
```

Also assert descriptive rejection of:

- a non-datetime quantitative index;
- a timezone-naive, non-UTC, non-midnight, or duplicate quantitative index;
- a duplicate text index;
- a missing selected quantitative or target-source column; and
- text sources whose ordered embedding columns differ.

Add a quantitative-only case with `text_embeddings={}` and assert shapes `[days, 0, 0]` and `[days, 0]`.

- [ ] **Step 2: Run the dataset tests and verify the preparation API is missing**

Run:

```bash
.venv/bin/python -m unittest discover -s transformer-based/tests -p 'test_dataset.py'
```

Expected: FAIL because `dataset.py` does not exist.

- [ ] **Step 3: Implement configuration and aligned base storage**

Create immutable dataclasses with these fields:

```python
@dataclass(frozen=True)
class DatasetConfig:
    lookback_days: int = 365
    horizon_days: int = 7
    avg_rate_column: str = "avg_rate"
    high_rate_column: str = "high_rate"
    low_rate_column: str = "low_rate"


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
```

`DatasetConfig.__post_init__` rejects non-positive lengths. `prepare_daily_data` performs these steps in order:

1. Validate a `DatetimeIndex`, UTC timezone, midnight timestamps, uniqueness, and selected columns.
2. Sort the quantitative input and create `pd.date_range(min_date, max_date, freq="D", tz="UTC")`.
3. Reindex quantitative values onto that calendar and create elementwise finite masks.
4. Sort text source names alphabetically.
5. Validate identical ordered embedding columns for every text source.
6. Reindex each embedding frame, require every component to be finite for a source/day mask to be true, and zero the entire vector otherwise.
7. Store float arrays without constructing sample windows.

When the text mapping is empty, use `text_values.shape == [days, 0, 0]`, `text_mask.shape == [days, 0]`, and empty source/embedding metadata so the quantitative-only case remains valid.

- [ ] **Step 4: Implement valid-origin discovery**

Iterate candidate positions from `lookback_days - 1` through `len(dates) - horizon_days - 1`. Keep an origin only when current `avg_rate` is finite and every future target-rate triplet is finite:

```python
current_avg_ok = prepared.target_rate_mask[position, 0]
future_ok = prepared.target_rate_mask[
    position + 1 : position + 1 + config.horizon_days
].all()
```

Return the matching dates as a chronological `pd.DatetimeIndex`. Do not reject origins for missing quantitative feature or text inputs.

- [ ] **Step 5: Run all isolated tests and commit alignment/discovery**

Run:

```bash
.venv/bin/python -m unittest discover -s transformer-based/tests -p 'test_*.py'
```

Expected: all label, normalization, and dataset-preparation tests pass.

Commit:

```bash
git add transformer-based/dataset.py transformer-based/tests/test_dataset.py
git commit -m "feat: align daily multimodal training data"
```

---

### Task 4: Train-Only Fitting and Lazy PyTorch Dataset

**Files:**
- Modify: `transformer-based/dataset.py`
- Modify: `transformer-based/tests/test_dataset.py`
- Create: `transformer-based/requirements.txt`

**Interfaces:**
- Consumes: `LabelSchema`, `TARGET_NAMES`, `NamedStandardizer`, `DatasetNormalizers`, `PreparedDailyData`, `DatasetConfig`
- Produces: `fit_train_normalizers(prepared, training_sample_end_dates, config) -> DatasetNormalizers`
- Produces: `DailyMultimodalDataset(prepared, sample_end_dates, normalizers, config)` implementing `torch.utils.data.Dataset`

- [ ] **Step 1: Declare isolated runtime dependencies**

Create `transformer-based/requirements.txt`:

```text
numpy>=1.24.0
pandas>=1.5.0,<3
torch>=2.2.0,<3
```

Install only if the active environment lacks PyTorch:

```bash
uv pip install --python .venv/bin/python -r transformer-based/requirements.txt
```

- [ ] **Step 2: Add tests for train-only fitting with a leakage sentinel**

Create twelve daily rows with one quantitative feature equal to `1, 2, ..., 12`, `avg_rate` equal to `100, 101, ..., 111`, `high_rate = avg_rate + 2`, and `low_rate = avg_rate - 3`. Replace the day-ten quantitative value and day-ten target-source values with extreme validation-only values. Use training origins whose union of input positions covers only days one through five:

```python
normalizers = fit_train_normalizers(
    prepared,
    training_sample_end_dates=prepared.dates[[2, 3, 4]],
    config=DatasetConfig(lookback_days=3, horizon_days=2),
)

self.assertEqual(normalizers.quantitative.mean[0], 3.0)
np.testing.assert_allclose(normalizers.target.mean, [1.5, 2.0, -3.0])
```

These assertions must fail if fitting reads the full timeline, includes validation targets, or weights overlapping quantitative input windows repeatedly.

- [ ] **Step 3: Add tests for lazy slicing, output shapes, and exact boundaries**

Build a dataset for one origin at day five with `lookback_days=3` and `horizon_days=2`. Assert:

```python
sample = dataset[0]
self.assertEqual(tuple(sample["quantitative"].shape), (3, num_features))
self.assertEqual(tuple(sample["quantitative_mask"].shape), (3, num_features))
self.assertEqual(tuple(sample["text"].shape), (3, num_sources, embedding_dim))
self.assertEqual(tuple(sample["text_mask"].shape), (3, num_sources))
self.assertEqual(tuple(sample["target"].shape), (2, 3))
self.assertEqual(sample["quantitative"].dtype, torch.float32)
self.assertEqual(sample["quantitative_mask"].dtype, torch.bool)
self.assertEqual(sample["as_of"].dtype, torch.int64)
```

Inverse-transform the quantitative slice and assert it contains raw days three, four, and five. Inverse-transform the target and assert it contains labels derived from days six and seven relative to day five.

Assert base storage remains daily rather than sample-expanded:

```python
self.assertEqual(dataset.prepared.quantitative_values.ndim, 2)
self.assertEqual(dataset.prepared.text_values.ndim, 3)
```

Add tests that constructor input rejects an origin absent from `discover_valid_sample_end_dates`, a normalizer with mismatched feature names, and an empty origin list.

- [ ] **Step 4: Implement train-only normalizer fitting**

Validate every training origin against discovered valid dates and translate dates to integer positions. Compute the union of all input positions:

```python
input_positions = np.unique(
    np.concatenate(
        [np.arange(pos - config.lookback_days + 1, pos + 1) for pos in origin_positions]
    )
)
```

Fit the quantitative normalizer once on those unique daily rows. Build one raw target array per training origin with `LabelSchema.from_rates`, stack to `[samples, horizon_days, 3]`, and fit the target standardizer with `TARGET_NAMES` across the first two dimensions. Return `DatasetNormalizers`.

- [ ] **Step 5: Implement the map-style lazy dataset**

Subclass `torch.utils.data.Dataset`. Constructor behavior:

1. Require at least one requested origin.
2. Require exact normalizer names: prepared quantitative names and `TARGET_NAMES`.
3. Require every requested origin to exist in the discovered valid-origin index.
4. Store only prepared base data, configuration, normalizers, and integer origin positions.

`__getitem__` must:

1. Slice raw quantitative values and masks from `origin-lookback+1` through `origin`.
2. Normalize quantitative values and zero missing elements through the named standardizer.
3. Slice already-zeroed text values and text masks for the same dates.
4. Construct raw future labels through `LabelSchema.from_rates`.
5. Normalize the `[horizon_days, 3]` target.
6. Return contiguous PyTorch tensors and `torch.tensor(prepared.dates[origin].value, dtype=torch.int64)` for nanosecond `as_of`.

The returned dictionary keys are exactly:

```python
{
    "quantitative",
    "quantitative_mask",
    "text",
    "text_mask",
    "target",
    "as_of",
}
```

- [ ] **Step 6: Run the complete isolated suite and commit**

Run:

```bash
.venv/bin/python -m unittest discover -s transformer-based/tests -p 'test_*.py'
```

Expected: all isolated tests pass with no root application imports or services.

Commit:

```bash
git add transformer-based/dataset.py transformer-based/tests/test_dataset.py transformer-based/requirements.txt
git commit -m "feat: add lazy multimodal pytorch dataset"
```

---

### Task 5: Isolated Usage Contract and Final Verification

**Files:**
- Create: `transformer-based/README.md`
- Modify only if verification exposes a defect: files under `transformer-based/`

**Interfaces:**
- Documents: preparation, explicit training origins, train-only fitting, train/validation construction, tensor shapes, normalizer persistence, and test command.

- [ ] **Step 1: Write a runnable usage example**

Document this sequence with concrete variable names:

```python
from dataset import (
    DailyMultimodalDataset,
    DatasetConfig,
    discover_valid_sample_end_dates,
    fit_train_normalizers,
    prepare_daily_data,
)

config = DatasetConfig(lookback_days=365, horizon_days=7)
prepared = prepare_daily_data(
    quantitative=daily_quantitative,
    text_embeddings=embeddings_by_source,
    quantitative_features=quantitative_feature_names,
    config=config,
)
valid_dates = discover_valid_sample_end_dates(prepared, config)
train_dates = valid_dates[valid_dates <= "2024-12-31"]
validation_dates = valid_dates[valid_dates >= "2025-01-08"]
normalizers = fit_train_normalizers(prepared, train_dates, config)
train_dataset = DailyMultimodalDataset(prepared, train_dates, normalizers, config)
validation_dataset = DailyMultimodalDataset(prepared, validation_dates, normalizers, config)
```

Explain that the eight-day date separation shown for a seven-day horizon prevents overlapping target windows, and show saving `normalizers.state_dict()` with the model checkpoint.

- [ ] **Step 2: Verify syntax and isolated tests from the repository root**

Run:

```bash
.venv/bin/python -m py_compile \
  transformer-based/labels.py \
  transformer-based/normalization.py \
  transformer-based/dataset.py
.venv/bin/python -m unittest discover -s transformer-based/tests -p 'test_*.py'
```

Expected: compilation succeeds and every isolated test passes.

- [ ] **Step 3: Run an import-and-batch smoke test**

From a short inline script, construct at least 375 synthetic UTC daily rows, two quantitative features, and two text sources with four-dimensional embeddings. Fit normalizers on valid training origins, construct `DataLoader(dataset, batch_size=2)`, and assert the first batch shapes are:

```text
quantitative:      [2, 365, 2]
quantitative_mask: [2, 365, 2]
text:              [2, 365, 2, 4]
text_mask:         [2, 365, 2]
target:            [2, 7, 3]
as_of:             [2]
```

- [ ] **Step 4: Check scope and worktree state**

Run:

```bash
git diff --check
git status --short
git diff --stat HEAD~4..HEAD
```

Confirm that implementation commits touch only `transformer-based/`. Preserve the pre-existing user modification to `data/latest/external_daily.csv` without staging or editing it.

- [ ] **Step 5: Commit the usage documentation**

```bash
git add transformer-based/README.md
git commit -m "docs: explain multimodal dataset usage"
```
