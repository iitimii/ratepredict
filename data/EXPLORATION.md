# Historical multimodal data exploration

Target window: **2006-01-01 through 2026-09-02** (7,550 calendar days).

The authoritative normalized store is the partitioned Parquet dataset under
`dataset/`. Its schema, daily model view, and loading examples are documented in
`DATASET_STORAGE.md`.

This directory now separates source payloads (`raw/`), normalized acquisition
outputs (`curated/`), and provenance (`metadata/`). Existing project files were
left in place and `latest/external_daily.csv` was not overwritten.

## Original inventory

| File | Shape | Coverage | What it is | Training decision |
|---|---:|---|---|---|
| `Data Sources - Data.csv` | 19 x 5 | n/a | Desired quantitative/document source list | Keep as planning input; superseded operationally by `metadata/source_catalog.json` |
| `Data Sources - Events.csv` | 18 x 13 | mainly 2015-2026 holiday entries | CBN/FOMC/BoE links plus manually entered Chinese holidays | Do not train directly; replace with dated, sourced event records |
| `Dt - New (1).csv` | 1,217 x 23 | 2022-01-03 to 2026-09-01 | Engineered USD/NGN macro table | Reference only; do not use as raw X because it mixes hand-filled/derived columns and the project explicitly dropped engineered features |
| `latest/external_daily.csv` | 2,436 x 9 | 2019-12-29 to 2026-08-31 | Yahoo-derived daily global markets | Retain for comparison/current continuity, but use authoritative long-history replacements where available |
| `latest/quidax_runtime_2h.csv` | 999 x 24 | 2025-12-17 to 2026-03-10 | Quidax OHLCV plus placeholder trade aggregates | Keep OHLCV; treat all-zero trade aggregate fields as unavailable, not real zero activity |
| `usdngn training data - usdngn training data.csv` | 26,252 x 24 | 2020-01-08 to 2026-03-10 | Reverse-ordered 2-hour USDT/NGN/BTC-NGN history | Keep after quality filtering; it cannot supply a 20-year crypto history |

## Material quality findings

- `Dt - New (1).csv` has 1,216 unique dates across 1,217 rows and already
  contains returns, moving averages, lagged rates, and proxy features. It is not
  a clean source table.
- `latest/external_daily.csv` is a calendar-day frame with only about 69% of
  Brent/DXY/VIX observations populated and contains a suspicious USD/GHS value
  of 573. Source-vintage metadata is absent.
- `latest/quidax_runtime_2h.csv` has 999 consecutive two-hour buckets, but every
  trade-count, buy/sell, trade-size, intrabar-statistic, and unique-trader value
  is zero. Those columns must be masked as missing.
- The 26,252-row historical Quidax export contains a USDT/NGN `high` of 729,000
  and corresponding extreme intrabar values. It needs an explicit bad-tick rule
  before aggregation to daily OHLCV.
- CBN endpoints sometimes contain repeated natural observations, including
  corrected values. Raw JSON retains every source row; curated tables keep the
  highest source `id` for each date (and currency where applicable).

## Acquired quantitative data

| Output | Curated rows | Coverage | Notes |
|---|---:|---|---|
| `curated/quantitative/cbn_exchange_rates.csv` | 55,382 | 2006-01-03 to 2026-09-02 | Long form, 18 currency labels |
| `curated/quantitative/cbn_usd_ngn_official.csv` | 5,053 | 2006-01-03 to 2026-09-02 | Official CBN USD/NGN buying/central/selling rates |
| `curated/quantitative/cbn_money_market.csv` | 247 | 2006-01 to 2026-07 | MPR and nine other money-market/deposit/lending rates |
| `curated/quantitative/cbn_money_credit.csv` | 246 | 2006-01 to 2026-07 | Monthly M1/M2/M3, reserve money, domestic/foreign assets and credit aggregates |
| `curated/quantitative/cbn_inflation.csv` | 246 | 2006-01 to 2026-06 | Headline, food and core inflation rates and averages |
| `curated/quantitative/cbn_government_securities.csv` | 4,179 | 2006-01-03 to 2026-09-02 | NTB, OMO, FGN bond and CBN bill auction observations |
| `curated/quantitative/cbn_reserves.csv` | 4,874 | 2006-04-03 to 2026-09-01 | Gross, liquid, blocked reserves and blocked share |
| `curated/quantitative/cbn_daily_crude.csv` | 3,939 | 2009-10-23 to 2026-09-02 | CBN Nigeria crude series; not a full 20 years |
| `curated/quantitative/cbn_nfem_rates.csv` | 432 | 2024-12-02 to 2026-09-02 | NFEM rates, turnover and deal counts |
| `curated/quantitative/fred/*.csv` | one file per series | mostly 2006 to 2026 | 2Y/10Y yields, Fed funds, broad USD index, Brent, VIX, CPI and 10Y inflation expectations |
| `curated/quantitative/world_bank_nigeria.csv` | 300 | annual 2006 to 2025 | 15 indicators; 230 published values and 70 unavailable country-years retained as null |

`BAMLH0A0HYM2` is only returned from September 2023 in the current FRED export,
so it is marked partial rather than being presented as a 20-year feature.

## Acquired unstructured indexes

The current acquisition stores metadata and URLs, not copyrighted article/PDF
bodies. This is enough to schedule controlled document downloads and preserve
source identity before text extraction and embedding.

| Output | Rows | Coverage/status |
|---|---:|---|
| `cbn_circulars_index.csv` | 2,336 | 2006-01-03 to 2026-08-12 |
| `cbn_press_releases_index.csv` | 252 | 2006-01-03 to 2026-08-11 |
| `dmo_fgn_bond_auction_results_index.csv` | 223 | 2008-01-03 to 2026-08-17 |
| `fomc_document_index.csv` | 605 | 2006 to 2026 statements/minutes/projections links |
| `boe_mpc_minutes_index.csv` | 339 | 2006 to 2026; HTML/PDF alternatives may both appear |
| `publisher_sitemaps/*.csv` | 3,117 child sitemaps | Nairametrics, BusinessDay, Vanguard and Premium Times |

Observed publisher archive starts are materially later than 2006: BusinessDay
and Vanguard around 2009, Nairametrics around 2012, and Premium Times around
2012. ThisDay currently disallows crawling in `robots.txt`; Punch's declared
sitemap returns HTTP 500. Google News RSS is a recent-query surface, not a
reproducible twenty-year bulk archive.

## Dataset construction rules

1. Build a daily calendar from 2006-01-01 onward.
2. Aggregate genuine intraday market observations to daily OHLCV/trade fields.
3. Join low-frequency macro data on **release availability date**, then carry
   forward with an age/missingness mask. Do not backfill from future releases.
4. Store one text-document table with source, publication time, title, URL,
   checksum, extracted text, language, and extraction status.
5. Compute precomputed embeddings per source/day and keep document counts and
   empty-source masks beside every embedding column.
6. Fit normalization only on the training split, as required by the dataset
   implementation under `transformer-based/`.

## Remaining high-priority gaps

- Official/licensed daily NGX ASI history and market turnover.
- Authoritative parallel-market USD/NGN history; do not scrape a commercial
  quote archive without permission and stable terms.
- Monthly NBS headline/core/food CPI files and their original release dates.
- CBN monthly M2, capital importation (FDI/FPI), OMO and NTB auction tables.
- Document-body download, PDF/HTML extraction, deduplication and embeddings for
  the acquired CBN/DMO/Fed/BoE indexes.
- IMF archive indexing and permitted article retrieval.
- Publisher article-URL enumeration in bounded, resumable batches, followed by
  rights/robots-aware text retrieval. A 20-year history is impossible for
  publishers whose online archives begin later; institutional and licensed
  archives must cover the early years.

Run acquisition with:

```bash
source .venv/bin/activate
python transformer-based/acquire_history.py --group all \
  --start-date 2006-01-01 --end-date 2026-09-02
```

Use the narrower groups (`cbn-quant`, `fred`, `world-bank`, `documents`, or
`news-indexes`) for resumable refreshes.
