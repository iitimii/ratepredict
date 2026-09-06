# Dataset storage

The authoritative model-data store is `data/dataset/`, a typed, Zstandard-
compressed Apache Parquet dataset. Raw API responses, HTML, and future PDF
documents remain in `data/raw/` and are linked through checksummed provenance
records rather than duplicated inside the analytical store.

## Layout

```text
data/dataset/
├── _dataset.json
├── records/
│   ├── modality=quantitative/source_id=.../*.parquet
│   ├── modality=unstructured/source_id=.../*.parquet
│   ├── modality=unstructured_index/source_id=.../*.parquet
│   └── modality=metadata/source_id=.../*.parquet
└── views/
    ├── daily_quantitative_observed.parquet
    └── daily_text_embeddings.parquet
```

`_dataset.json` records row counts, schemas, partitions, build time, feature-view
coverage, and a SHA-256 digest of the complete store.

## Current contents

- 932,464 canonical records with unique IDs
- 925,469 quantitative records
- 3,755 document records
- 3,117 publisher archive-index records
- 123 source/provenance records
- 34 source partitions
- 7,550 rows and 64 columns in the daily quantitative view
- Approximately 22 MB on disk

## Record schema

| Field | Arrow type | Meaning |
|---|---|---|
| `record_id` | string | Stable record identifier |
| `observed_at` | UTC timestamp | Observation, publication, or acquisition time |
| `period_year` | int16 | Year when no precise publication date is available |
| `modality` | partition string | Quantitative, unstructured, index, or metadata |
| `source_id` | partition string | Stable source namespace |
| `record_type` | string | Feature observation, document, sitemap, manifest, etc. |
| `feature_name` | string | Globally namespaced feature identifier |
| `numeric_value` | float64 | Quantitative value |
| `text_value` | string | Available document title/description/keywords |
| `title` | string | Document title |
| `url` | string | Original source URL |
| `unit` | string | Source-provided unit |
| `frequency` | string | Source cadence |
| `quality_status` | string | Review/missingness/acquisition status |
| `source_path` | string | Repository-relative originating file |
| `metadata_json` | string | Source-specific dimensions and provenance |

## Loading records selectively

```python
import pyarrow.dataset as ds

records = ds.dataset(
    "data/dataset/records",
    format="parquet",
    partitioning="hive",
)
cbn_quant = records.to_table(
    filter=(ds.field("modality") == "quantitative")
    & (ds.field("source_id") == "cbn_reserves"),
    columns=["observed_at", "feature_name", "numeric_value", "quality_status"],
)
```

Partition and column filters avoid reading unrelated sources or fields.

## Loading the daily model view

```python
import pandas as pd

daily_quantitative = pd.read_parquet(
    "data/dataset/views/daily_quantitative_observed.parquet"
).set_index("date")
```

The view uses feature-aware aggregation for Quidax bars: first open, maximum
high, minimum low, last close, and summed activity fields. Known malformed bars
are excluded. Monthly and annual macro series are not forward-filled because
their historical release dates are not yet available; doing so from period
start could leak future information.

`daily_text_embeddings.parquet` currently contains the final typed schema but
zero rows. When document bodies and an embedding model are ready, each record
will contain date, source, model identity, a float32 list, document count, and a
missing-text flag.

## Rebuilding

```bash
source .venv/bin/activate
python transformer-based/build_dataset_store.py
```

The builder streams canonical records, builds in a temporary directory, and
replaces the prior store only after all outputs and the dataset digest have been
written successfully.

For systems that require CSV, `transformer-based/build_dataset_csv.py` remains
an optional export path. CSV is not retained as the primary store because it is
untyped, unpartitioned, and over twelve times larger for this corpus.
