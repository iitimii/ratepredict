# Multimodal Daily Forecast Dataset

This folder contains an isolated daily PyTorch dataset for a different modeling approach from the root application. It consumes daily quantitative features and precomputed daily text embeddings, then maps a configurable historical window to future USDT/NGN labels.

The defaults are:

```text
X[t-364 ... t] -> y[t+1 ... t+7]
```

## Setup

Before starting any training run, go over the [mandatory pre-training checklist](../docs/pre_training_checklist.md) and record the review outcome. See [current dataset completeness](../data/COMPLETENESS.md) and the [proposed embedding workflow](docs/unstructured_embedding_plan.md).

From the repository root:

```bash
uv pip install --python .venv/bin/python -r transformer-based/requirements.txt
```

Run the isolated tests with:

```bash
.venv/bin/python -m unittest discover -s transformer-based/tests -p 'test_*.py'
```

Build the canonical partitioned Parquet store from all acquired sources with:

```bash
.venv/bin/python transformer-based/build_dataset_store.py
```

The output is `data/dataset/`; see `data/DATASET_STORAGE.md` for its typed
schema, partitions, quality flags, and daily model view. A CSV exporter remains
available only for interoperability.

When importing these top-level modules from another script, add `transformer-based/` to `PYTHONPATH` or run the script from this directory.

## Input contract

`daily_quantitative` is a pandas DataFrame with a unique UTC-midnight daily index. It contains selected model features plus explicit `avg_rate`, `high_rate`, and `low_rate` columns.

`embeddings_by_source` maps each text source name to a DataFrame with the same kind of daily index. Every source must have identical ordered embedding columns and dimensions:

```python
embeddings_by_source = {
    "businessday_ng": businessday_daily_embeddings,
    "cbn_circulars": cbn_circular_daily_embeddings,
    "nairametrics": nairametrics_daily_embeddings,
}
```

Missing quantitative values and embeddings remain usable through masks. A sample is excluded only when the current average rate or one of its future target rates is missing.

## Dataset construction

```python
import sys

sys.path.insert(0, "transformer-based")

from dataset import (
    DailyMultimodalDataset,
    DatasetConfig,
    discover_valid_sample_end_dates,
    fit_train_normalizers,
    prepare_daily_data,
)

quantitative_feature_names = (
    "usdtngn_open",
    "usdtngn_high",
    "usdtngn_low",
    "usdtngn_close",
    "usdtngn_volume",
    "btcngn_close",
    "cbn_usdngn",
    "parallel_usdngn",
    "mpr",
    "ngx_asi_performance",
    "nigeria_inflation",
    "bonny_oil_price",
    "foreign_reserves",
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

normalizers = fit_train_normalizers(
    prepared,
    training_sample_end_dates=train_dates,
    config=config,
)
train_dataset = DailyMultimodalDataset(
    prepared,
    sample_end_dates=train_dates,
    normalizers=normalizers,
    config=config,
)
validation_dataset = DailyMultimodalDataset(
    prepared,
    sample_end_dates=validation_dates,
    normalizers=normalizers,
    config=config,
)
```

For a seven-day target horizon, the example leaves eight calendar days between the final training origin and first validation origin. Training targets end on January 7 and validation targets begin on January 9, so target windows cannot overlap.

Choose split dates explicitly for the actual training period. Always reuse the training normalizers for validation, testing, and inference.

## Sample tensors

Each dataset item is:

```text
quantitative:      float32 [lookback_days, quantitative_features]
quantitative_mask: bool    [lookback_days, quantitative_features]
text:              float32 [lookback_days, text_sources, embedding_dim]
text_mask:         bool    [lookback_days, text_sources]
target:            float32 [horizon_days, 3]
as_of:             int64   scalar Unix timestamp in nanoseconds
```

The target channels are ordered as:

```text
avg_change, high_change, low_change
```

and calculated for every future day `d` as:

```python
avg_change[d] = avg_rate[t+d] - avg_rate[t]
high_change[d] = high_rate[t+d] - avg_rate[t+d]
low_change[d] = low_rate[t+d] - avg_rate[t+d]
```

## Checkpoint persistence

Normalizer state contains only names and numeric lists, so it can be stored directly in a PyTorch checkpoint:

```python
import torch

torch.save(
    {
        "model_state": model.state_dict(),
        "normalizers": normalizers.state_dict(),
        "dataset_config": {
            "lookback_days": config.lookback_days,
            "horizon_days": config.horizon_days,
        },
    },
    "checkpoint.pt",
)
```

Restore the fitted statistics with:

```python
from normalization import DatasetNormalizers

checkpoint = torch.load("checkpoint.pt", weights_only=False)
normalizers = DatasetNormalizers.from_state_dict(checkpoint["normalizers"])
```

Use `normalizers.target.inverse_transform(predictions, target_names)` to convert normalized model predictions back to absolute NGN rate differences.
