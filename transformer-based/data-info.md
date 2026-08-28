# Data Sources

## Already implemented

| Item | Format / dimensions | Frequency | Source |
|---|---|---|---|
| USDT/NGN history | CSV/DataFrame, `N x 23` | 2-hour bars | BigQuery exports and [Quidax](https://app.quidax.io/api/v1/markets/usdtngn/k?period=120&limit=1000) |
| BTC/NGN history | DataFrame, `N x 3` (`close`, `volume`, `trade_count`) | 2-hour bars | [Quidax](https://app.quidax.io/api/v1/markets/btcngn/k?period=120&limit=1000) |
| External markets | CSV/DataFrame, `N x 8` | Daily | Yahoo Finance |
| Model features | DataFrame, `N x 42` | Every 2 hours | Derived from market data |
| Live USDT/NGN rate | JSON object | Live/on request | QBOT rates API |
| Live BTC/NGN rate | JSON object | Live/on request | [Quidax ticker](https://app.quidax.io/api/v1/markets/tickers/btcngn) |
| Market news | List of JSON objects, up to 60 | On request; 15-minute cache | Google News, Nigerian business feeds, CBN, OilPrice and IMF |
| Macro calendar | List of 69 JSON objects | Updated manually | `app/macro_calendar.py` |
| Signal history | CSV/DataFrame, `N x 11` | Each model run | `app/signal_log.csv` |

Yahoo Finance provides these eight daily columns:

`brent`, `dxy`, `vix`, `usdzar`, `usdngn_official`, `usdghs`, `usdkes`, `btcusd_global`.

All historical timestamps are UTC. Calling `DataSchema.to_dict()` converts DataFrames to JSON
records, timestamps to ISO-8601 strings, and missing values to `null`.

## To implement

| Item | Expected format |
|---|---|
| CBN circulars and reports | PDF/HTML |
| Coinbase | API JSON |
| NGX All-Share Index (ASI) | CSV/API |
| SEC Nigeria | HTML/PDF/API |
| FMDQ | CSV/HTML/API |
| Government auction results | PDF/CSV/HTML |
| FGN bonds | CSV |
| NUPRC oil production / Bonny data | PDF/XLSX |
