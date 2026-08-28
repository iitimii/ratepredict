from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "transformer-based" / "data.py"
SPEC = importlib.util.spec_from_file_location("transformer_data", MODULE_PATH)
assert SPEC and SPEC.loader
DATA_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DATA_MODULE
SPEC.loader.exec_module(DATA_MODULE)

DataSource = DATA_MODULE.DataSource
format_snapshot = DATA_MODULE.format_snapshot


class DataSourceTests(unittest.TestCase):
    def make_settings(self, root: Path) -> SimpleNamespace:
        data_dir = root / "data"
        artifacts_dir = root / "artifacts"
        runtime_dir = root / "runtime"
        for directory in (data_dir, artifacts_dir, runtime_dir):
            directory.mkdir()
        return SimpleNamespace(
            base_dir=root,
            data_dir=data_dir,
            artifacts_dir=artifacts_dir,
            runtime_dir=runtime_dir,
            runtime_bars_filename="quidax_runtime_2h.csv",
            external_daily_filename="external_daily.csv",
            export_glob="bquxjob_*.csv",
            feature_lookback_bars=480,
            quidax_kline_period_minutes=120,
            quidax_kline_limit=1000,
            external_live_fallback_enabled=True,
            yahoo_tickers={},
            http_timeout_seconds=5.0,
            news_cache_ttl_seconds=900,
            news_max_age_hours=72,
            news_fetch_timeout_seconds=1.0,
        )

    def test_historical_usdt_ngn_combines_archive_and_runtime_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self.make_settings(root)
            pd.DataFrame(
                {
                    "bucket_2h": ["2026-01-01T00:00:00Z", "2026-01-01T02:00:00Z"],
                    "close": [1400.0, 1401.0],
                }
            ).to_csv(settings.artifacts_dir / "bquxjob_history.csv", index=False)
            pd.DataFrame(
                {
                    "bucket_2h": ["2026-01-01T02:00:00Z", "2026-01-01T04:00:00Z"],
                    "close": [1410.0, 1411.0],
                }
            ).to_csv(settings.data_dir / settings.runtime_bars_filename, index=False)

            frame = DataSource(settings).get_historical_usdt_ngn()

            self.assertEqual(len(frame), 3)
            self.assertEqual(float(frame.loc[pd.Timestamp("2026-01-01T02:00:00Z"), "close"]), 1410.0)
            self.assertEqual(str(frame.index.tz), "UTC")

    def test_serve_returns_every_source_in_one_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = DataSource(self.make_settings(root))
            index = pd.date_range("2026-01-01", periods=2, freq="2h", tz=UTC)
            usdt_ngn = pd.DataFrame({"close": [1400.0, 1401.0]}, index=index)
            btc_ngn = pd.DataFrame({"close": [90_000_000.0, 91_000_000.0]}, index=index)
            external = pd.DataFrame({"brent": [75.0]}, index=pd.date_range("2026-01-01", periods=1, tz=UTC))
            features = pd.DataFrame({"return_2h": [0.0, 0.001]}, index=index)
            live = {"usdt_ngn": {"last": 1402.0}, "btc_ngn": {"last": 91_500_000.0}}
            news = [{"title": "CBN holds rates"}]
            macro = [{"event": "CBN Monetary Policy Committee (MPC)"}]
            signals = pd.DataFrame({"signal": ["UP"]})

            with (
                patch.object(source, "get_historical_usdt_ngn", return_value=usdt_ngn),
                patch.object(source, "get_historical_btc_ngn", return_value=btc_ngn),
                patch.object(source, "get_external_market_data", return_value=external),
                patch.object(source, "get_features", return_value=features),
                patch.object(source, "get_live_rates", return_value=live),
                patch.object(source, "get_news", return_value=news),
                patch.object(source, "get_macro_calendar", return_value=macro),
                patch.object(source, "get_signal_history", return_value=signals),
            ):
                data = source.serve(refresh=True, market_notes="Large client order")

            pd.testing.assert_frame_equal(data.usdt_ngn, usdt_ngn)
            pd.testing.assert_frame_equal(data.btc_ngn, btc_ngn)
            pd.testing.assert_frame_equal(data.external_markets, external)
            pd.testing.assert_frame_equal(data.features, features)
            self.assertEqual(data.live_rates, live)
            self.assertEqual(data.news, news)
            self.assertEqual(data.macro_calendar, macro)
            pd.testing.assert_frame_equal(data.signal_history, signals)
            self.assertEqual(data.market_notes, "Large client order")
            self.assertEqual(data.as_of.tzinfo, UTC)

            serialized = data.to_dict()
            self.assertEqual(serialized["schema_version"], "1.0")
            self.assertEqual(serialized["market_notes"], "Large client order")
            self.assertEqual(len(serialized["usdt_ngn"]), 2)

    def test_direct_run_snapshot_represents_every_data_source(self) -> None:
        index = pd.date_range("2026-01-01", periods=2, freq="2h", tz=UTC, name="bucket_2h")
        daily_index = pd.date_range("2026-01-01", periods=1, tz=UTC, name="date")
        snapshot = DataSource.DataSchema(
            as_of=datetime(2026, 1, 1, 4, tzinfo=UTC),
            usdt_ngn=pd.DataFrame({"close": [1400.0, 1401.0]}, index=index),
            btc_ngn=pd.DataFrame({"close": [90_000_000.0, 91_000_000.0]}, index=index),
            external_markets=pd.DataFrame({"brent": [75.0]}, index=daily_index),
            features=pd.DataFrame({"return_2h": [0.0, 0.001]}, index=index),
            live_rates={"usdt_ngn": {"last": 1402.0}},
            news=[
                {
                    "title": "CBN holds rates",
                    "source": "CBN",
                    "published": datetime(2026, 1, 1, 3, tzinfo=UTC),
                    "relevance": 1.0,
                }
            ],
            macro_calendar=[{"category": "Nigeria", "event": "MPC"}],
            signal_history=pd.DataFrame({"signal": ["HOLD"], "forecast_price": [1403.0]}),
            market_notes="Large client order",
            source_statuses=[
                {
                    "source_id": "quidax_kline_usdtngn",
                    "status": "ok",
                    "latest_timestamp": index[-1],
                    "message": "2 bars",
                }
            ],
        )

        output = format_snapshot(snapshot)

        for section in (
            "USDT/NGN HISTORY",
            "BTC/NGN HISTORY",
            "EXTERNAL MARKETS",
            "MODEL FEATURES",
            "LIVE RATES",
            "NEWS",
            "MACRO CALENDAR",
            "SIGNAL HISTORY",
            "MARKET NOTES",
            "SOURCE STATUSES",
        ):
            self.assertIn(section, output)
        self.assertIn("time=t: 2026-01-01T04:00:00+00:00", output)
        self.assertIn("CBN holds rates", output)
        self.assertIn("Large client order", output)
        self.assertIn("quidax_kline_usdtngn", output)

        fake_source = SimpleNamespace(source=lambda **kwargs: snapshot)
        printed = io.StringIO()
        with redirect_stdout(printed):
            result = DATA_MODULE.main(fake_source)

        self.assertEqual(result, 0)
        self.assertIn("RATEPREDICT DATA SNAPSHOT", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
