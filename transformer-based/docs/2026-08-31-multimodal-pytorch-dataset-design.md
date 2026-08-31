# Multimodal Daily PyTorch Dataset Design

## Objective

Build an isolated PyTorch dataset under `transformer-based/` for daily multimodal forecasting. Each sample maps a configurable historical window ending at forecast origin `t` to configurable future daily USDT/NGN targets:

```text
X[t-lookback_days+1 ... t] -> y[t+1 ... t+horizon_days]
```

The defaults are a 365-calendar-day input window and a seven-calendar-day forecast horizon. The primary repository remains read-only and supplies raw data only.

## Scope

All new or modified implementation, dependency, documentation, and test files belong under `transformer-based/`:

```text
transformer-based/
├── data.py
├── labels.py
├── normalization.py
├── dataset.py
├── requirements.txt
├── docs/
└── tests/
```

The implementation will not use the primary repository's existing 42 engineered features. It will not modify root application, model, dependency, or test files.

## Modalities

### Daily quantitative data

The quantitative input is a pandas DataFrame with a unique UTC daily index. Columns are selected explicitly and may expand without changing dataset code. Expected families include:

- USDT/NGN, BTC/NGN, CBN USD/NGN, and parallel-market USD/NGN;
- OHLCV and trade aggregates;
- Monetary Policy Rate;
- NGX All-Share Index performance;
- Nigerian and other relevant inflation rates;
- Bonny oil prices;
- foreign reserves; and
- additional point-in-time quantitative sources added later.

The DataFrame must also contain explicit daily `avg_rate`, `high_rate`, and `low_rate` columns. These values are supplied by the caller; the dataset does not derive them from OHLC data.

### Daily text embeddings

Precomputed text embeddings are supplied as:

```python
Mapping[str, pd.DataFrame]
```

Each mapping key is a stable source name, such as `nairametrics` or `cbn_circulars`. Each DataFrame has a unique UTC daily index and embedding dimensions as columns. All sources must use the same ordered embedding columns and dimension.

Expected sources include Nairametrics, BusinessDay NG, Google News, Vanguard NG, IMF News, Premium Times, CBN press releases, ThisDay Live, Punch NG, CBN circulars, and FGN bond auction results. New sources require only another mapping entry.

Upstream embedding generation combines multiple same-source documents for a day into one daily source document before embedding. Embedding generation itself is outside this dataset's scope.

## Sample Contract

For a sample whose forecast origin is daily timestamp `t`, the inclusive input window is:

```text
t - lookback_days + 1, ..., t
```

The target window is:

```text
t + 1, ..., t + horizon_days
```

The dataset returns a dictionary with these tensors:

```text
quantitative:      float32 [lookback_days, num_quantitative_features]
quantitative_mask: bool    [lookback_days, num_quantitative_features]
text:              float32 [lookback_days, num_text_sources, embedding_dim]
text_mask:         bool    [lookback_days, num_text_sources]
target:            float32 [horizon_days, 3]
as_of:             int64   scalar Unix timestamp in nanoseconds
```

The three target channels are ordered and named:

```text
avg_change, high_change, low_change
```

The dataset exposes the ordered quantitative feature names, text source names, and target names so every tensor position is auditable.

## Label Semantics

For each future horizon day `d`:

```python
avg_change[d] = avg_rate[t+d] - avg_rate[t]
high_change[d] = high_rate[t+d] - avg_rate[t+d]
low_change[d] = low_rate[t+d] - avg_rate[t+d]
```

Labels are absolute rate differences, not percentages. `labels.py` will support a configurable positive horizon rather than fixing the schema to seven days.

## Alignment and Missing Data

The quantitative calendar is the canonical calendar. It is reindexed to a complete sequence of UTC calendar days. Text sources are aligned to that calendar.

Missing quantitative inputs are allowed:

- the mask is `False` at each missing or non-finite element;
- missing values are excluded from normalizer fitting; and
- normalized missing values are replaced with `0.0`.

Missing text inputs are allowed:

- a source/day embedding is valid only when every embedding component is finite;
- an absent or incomplete embedding becomes a zero vector; and
- its source/day text mask is `False`.

A forecast origin is invalid and excluded when:

- `avg_rate[t]` is missing or non-finite;
- any future `avg_rate`, `high_rate`, or `low_rate` required by its horizon is missing or non-finite; or
- the requested complete calendar lookback or forecast horizon is unavailable.

Missing non-target inputs do not invalidate a sample.

## Lazy Indexed Dataset

Daily quantitative and text arrays are aligned and stored once. The dataset stores only integer positions for valid forecast origins. `__getitem__` slices the base arrays on demand and converts the slices to PyTorch tensors. It never pre-materializes overlapping sample windows.

Samples remain chronological inside the dataset. Training shuffle is controlled by `torch.utils.data.DataLoader`.

The constructor accepts explicit `sample_end_dates`. This makes train, validation, and test membership auditable and prevents a dataset from silently choosing split boundaries. Requested dates that are not valid forecast origins raise an error.

## Train-Only Normalization

Normalization is isolated in `normalization.py` and cannot be fitted by validation or test datasets.

### Quantitative normalizer

The quantitative normalizer:

- fits one mean and standard deviation per selected feature;
- uses the union of input dates referenced by training samples, so overlapping windows do not overweight central dates;
- ignores missing and non-finite values;
- uses scale `1.0` for constant features;
- raises an error when a selected feature has no finite training observations; and
- validates feature names and order during transformation.

### Target normalizer

The target normalizer:

- fits separate mean and standard deviation values for `avg_change`, `high_change`, and `low_change`;
- uses only targets belonging to training forecast origins;
- transforms targets during dataset access; and
- inverse-transforms model predictions back to absolute NGN differences.

Both normalizers have serializable state dictionaries for reuse in validation, testing, checkpointing, and inference. Text embeddings are not normalized by the dataset.

## Split Safety

The caller supplies train, validation, and test forecast-origin dates. The library will provide a helper to discover valid forecast origins and a helper to fit normalizers from training origins only.

Adjacent chronological splits must separate forecast origins by at least `horizon_days`, ensuring target windows do not overlap across split boundaries. The dataset validates requested origins but does not choose business-specific split proportions.

## Public API

The primary API consists of:

- a dataset configuration carrying `lookback_days`, `horizon_days`, and target-column names;
- configurable daily label construction;
- quantitative and target normalizers with serialization and inverse transformation;
- discovery of valid forecast-origin dates;
- train-only normalizer fitting from explicit training origins; and
- `DailyMultimodalDataset`, a map-style `torch.utils.data.Dataset`.

## Validation and Errors

Construction fails with a descriptive error for:

- non-datetime, duplicate, non-UTC, or non-midnight quantitative indices;
- duplicate text indices;
- missing selected quantitative or target-source columns;
- inconsistent text embedding dimensions;
- non-positive lookback or horizon values;
- unfitted or feature-incompatible normalizers;
- requested invalid forecast origins; or
- an empty valid-sample set.

## Testing

Tests remain under `transformer-based/tests/` and cover:

- exact inclusive input and exclusive future target boundaries;
- default and configurable lookback/horizon lengths;
- all three target formulas;
- output tensor shapes and dtypes;
- deterministic feature and source ordering;
- quantitative element masks and text source/day masks;
- zero filling after normalization;
- incomplete-target exclusion;
- invalid embedding-dimension rejection;
- train-only fitting using extreme validation values as a leakage sentinel;
- target inverse transformation;
- invalid requested forecast origins; and
- lazy storage without a pre-materialized sample tensor.

The isolated `transformer-based/requirements.txt` will declare PyTorch, pandas, and NumPy. Verification will run the isolated test suite without requiring the root application's model artifacts or services.
