# Unified Inference Data Source Design

## Purpose

`transformer-based/data.py` will expose every data input currently available in the repository through one model-agnostic class API. The API will not classify fields as model inputs or auxiliary context because the consuming model has not been selected. It will instead return one point-in-time-correct, two-hourly dataset with an explicit schema and source-health information.

The implementation will preserve the repository's existing ingestion and feature semantics. It will compose the service classes in `app/services` instead of duplicating their network protocols or feature formulas.

## Public API

The module will define these immutable dataclasses:

```python
@dataclass(frozen=True)
class FieldSchema:
    name: str
    dtype: str
    nullable: bool
    source: str
    frequency: str
    description: str
    unit: str | None = None


@dataclass(frozen=True)
class DataSourceStatus:
    source_id: str
    status: str
    latest_timestamp: datetime | None = None
    message: str | None = None


@dataclass(frozen=True)
class InferenceDataset:
    records: pd.DataFrame
    schema: tuple[FieldSchema, ...]
    source_statuses: tuple[DataSourceStatus, ...]
    generated_at: datetime
    schema_version: str = "1.0"

    def latest(self) -> dict[str, object]: ...
    def to_dict(self) -> dict[str, object]: ...
    def validate(self) -> None: ...
```

The entry point will be:

```python
class DataSource:
    def __init__(self, settings: Settings | None = None, *, dependencies: DataDependencies | None = None): ...

    def get_data(
        self,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        *,
        market_notes: str = "",
        refresh: bool = True,
        strict: bool = False,
    ) -> InferenceDataset: ...

    def get_schema(self) -> tuple[FieldSchema, ...]: ...
```

`get_data()` is the only method callers need for normal use. `refresh=True` attempts the live sources so a default call returns all currently obtainable inputs. `refresh=False` performs deterministic local/cache assembly without network access. `strict=False` retains the complete schema and represents unavailable values as null; `strict=True` raises `DataSourceError` when a requested source cannot produce data and no cache is usable.

`DataDependencies` is an internal dependency-injection container for the existing loaders and services. It makes tests deterministic and lets callers substitute production-approved providers without changing the returned contract.

## Canonical Record Shape

`InferenceDataset.records` is the single data collection. It is not divided into model and context sections. It has one row per UTC two-hour bucket, is sorted oldest to newest, and has a unique `timestamp` column plus a stable `series_id` value of `usdtngn`.

The table contains the following field families together in every record.

### Quidax USDT/NGN history

The best available historical bar source is selected in the same order as `ExportLoader`: `data/latest/quidax_runtime_2h.csv`, then the newest matching BigQuery export under `data/latest`, then the newest matching export under `artifacts`.

All available historical fields are retained:

- OHLCV: `open`, `high`, `low`, `close`, and `volume`.
- Trade flow: `trade_count`, `buy_volume`, `sell_volume`, `buy_count`, and `sell_count`.
- Trade-size and concentration observations: `avg_trade_size`, `max_trade_size`, `stddev_trade_size`, `large_trade_count`, `large_trade_volume`, `unique_buyers`, and `unique_sellers`.
- Intrabar observations: `intrabar_price_stddev` and `intrabar_range`.
- Joined Quidax BTC/NGN observations: `btcngn_close`, `btcngn_volume`, `btcngn_trade_count`, and `implied_btcusd_quidax`.

Missing historical fields in the public runtime-bar file remain nullable. They are not filled with invented measurements.

### Live quotes and public k-lines

When refresh is enabled, the newest two-hour bucket receives separately named live fields so the raw historical close is not silently overwritten:

- QBOT USDT/NGN: `live_usdtngn_bid`, `live_usdtngn_ask`, `live_usdtngn_last`, `live_usdtngn_low`, `live_usdtngn_high`, `live_usdtngn_open`, `live_usdtngn_provider`, `live_usdtngn_source`, and `live_usdtngn_observed_at`.
- Quidax BTC/NGN ticker: `live_btcngn_bid`, `live_btcngn_ask`, `live_btcngn_last`, `live_btcngn_low`, `live_btcngn_high`, `live_btcngn_open`, `live_btcngn_volume`, and `live_btcngn_observed_at`.
- Quidax public closed k-lines for `usdtngn` and `btcngn` are used by the existing refresh workflow and remain represented through the historical bar fields.

The feature-building view applies the existing live-overlay behavior internally when there is no fresh runtime-bar file. Both the original bar fields and live fields remain visible in the returned records.

### External daily markets

Daily fields are joined by effective date and forward-filled to the two-hour timeline without using future observations:

- `brent` from `BZ=F`.
- `dxy` from `DX-Y.NYB`.
- `vix` from `^VIX`.
- `usdzar` from `USDZAR=X`.
- `usdngn_official` from `USDNGN=X`.
- `usdghs` from `USDGHS=X`.
- `usdkes` from `USDKES=X`.
- `btcusd_global` from `BTC-USD`.

The existing `data/latest/external_daily.csv` cache is preferred when it is current. With refresh enabled, the existing Yahoo-backed service may update or fill that cache according to `Settings.external_live_fallback_enabled`.

### Engineered features

All columns in `artifacts/feature_cols.json` are returned under their existing names and in artifact order. This currently means the 42 momentum, cross-venue, volatility, calendar, and regional-FX features produced by `PublicFeatureBuilder`. The API also retains all raw columns from which those features were computed.

If the artifact feature list and constructed features disagree, validation fails with a message naming each missing column. Feature formulas, forward-fill behavior, and null-to-zero behavior remain identical to the current runtime inference path.

### News

Available news items from `NewsAggregatorService` are assigned to the UTC two-hour bucket containing their publication time. Each bucket contains:

- `news_items`: a JSON-compatible list of dictionaries containing title, summary, source, URL, publication time, category, and relevance score.
- `news_item_count`.
- `news_high_relevance_count`.
- `news_categories`: a JSON-compatible category-to-count mapping.

News is never forward-filled. A cache-only call uses the existing memory or disk cache if available. A failed refresh retains empty collections and records source failures rather than fabricating headlines.

### Macro calendar

The static event catalogue and schedules in `app/macro_calendar.py` are evaluated for every date represented by the requested range. Each row contains:

- `macro_events`: a JSON-compatible list of complete event dictionaries applicable to that timestamp's date or month-long window.
- `macro_event_count`.
- `macro_max_magnitude`, using the original textual magnitude.

Known dated events are present for every bucket on their scheduled date. Seasonal month-wide events are present throughout the applicable month. Irregular events without a known effective date are not attached to a historical timestamp; they remain discoverable through the schema/source description and are not represented as having occurred.

### Desk notes

The `market_notes` argument is placed only on the newest returned record as `market_notes`. It is never backfilled into historical rows.

### Signal history

`app/signal_log.csv` is treated as a lagged input because recent resolved accuracy participates in confidence scoring. Signals are assigned to their timestamp's two-hour bucket. The returned fields are:

- `signal_history`: a JSON-compatible list preserving every signal-log row in that bucket.
- `signal_history_count`.
- `resolved_signal_count_trailing_10`.
- `signal_accuracy_trailing_10`.
- `signal_pnl_bps_mean_trailing_10`.

Rolling statistics use only signal outcomes whose resolution was already present in the stored log and never look ahead to later rows in the requested dataset.

## Schema and Serialization

Every returned column has one `FieldSchema` entry. The schema records its pandas-compatible type, nullability, origin, native update frequency, semantic description, and unit where relevant. Structured collection cells use Python lists or dictionaries in memory and become JSON arrays or objects in `to_dict()`.

`InferenceDataset.latest()` returns the newest record as a JSON-compatible dictionary. `InferenceDataset.to_dict()` returns `schema_version`, `generated_at`, `schema`, `source_statuses`, and `records`. Datetimes serialize as ISO-8601 UTC strings, NumPy scalars become Python scalars, and missing pandas values become `None`.

## Source Health and Failure Handling

The implementation normalizes status objects from the market and news services into `DataSourceStatus`. Source identifiers are stable and include the active file name or provider identifier where useful.

In non-strict mode:

- A live-source failure is captured as `error` or `degraded`.
- A usable cache continues to populate the associated fields.
- A source with no usable data retains its declared columns as null.
- One source failure does not prevent other sources from being returned.

In strict mode, any unavailable requested live source or invalid required local history raises `DataSourceError`. Credentials are never included in returned records, status messages, exceptions, or serialized output.

The historical Quidax timeline is the minimum viable backbone. If no historical export or runtime-bar file exists, `get_data()` raises even in non-strict mode because there is no timestamp grid on which to assemble inference data.

## Validation

`InferenceDataset.validate()` checks:

- The dataset is not empty.
- `timestamp` exists, contains timezone-aware UTC values, is monotonic, and is unique.
- `series_id` exists and is non-null.
- Every record column has exactly one schema entry and every schema entry has a record column.
- Feature columns declared by the artifact contract are present.
- Numeric schema fields do not contain non-numeric values other than nulls.
- Structured fields contain only their declared list or dictionary types.

Validation runs before `get_data()` returns.

## Compatibility and Scope

The implementation is confined to `transformer-based/data.py` plus focused tests. Existing application services and current Streamlit inference behavior are not changed. The new module consumes their public interfaces and preserves the uncommitted `DataSource` work already present in the target file by evolving that class rather than replacing the user's intent.

`transformer-based/run_chronos.py` remains outside this change. Its sample electricity Parquet URLs are demonstration inputs, not repository production data sources. A later model integration can select and rename any fields from `InferenceDataset.records` without requiring `DataSource` to decide the model contract.

Documentation-only candidate sources such as Binance P2P, AbokiFX, FMDQ, and licensed institutional feeds are not fetched because the repository has no active provider implementation or endpoint for them. The dependency-injection boundary allows an approved implementation to be added later without changing the dataset API.

## Testing Strategy

Tests will load `transformer-based/data.py` through `importlib` because the directory name contains a hyphen. They will use temporary CSV fixtures and injected fake services; no test will require network access, credentials, Streamlit, or the pickled production models.

The tests will prove:

- Local history, external daily values, engineered features, news, macro events, desk notes, live quotes, and signal history all appear in one `records` table.
- Time boundaries are inclusive and normalized to UTC.
- Daily inputs forward-fill without future leakage.
- Snapshot values occur only on the newest appropriate bucket.
- News is bucketed by publication time and is not forward-filled.
- Signal statistics use only earlier stored outcomes.
- A failed optional source produces null fields and a source status in non-strict mode.
- The same failure raises in strict mode.
- Serialization is JSON-compatible and excludes credentials.
- Schema/record mismatches fail validation.

The final verification will run the focused new tests and the repository's complete test suite.
