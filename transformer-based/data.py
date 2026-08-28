from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class DataSource:
    """Single entry point for every data input available in this repository."""

    @dataclass(frozen=True)
    class DataSchema:
        as_of: datetime
        usdt_ngn: pd.DataFrame
        btc_ngn: pd.DataFrame
        external_markets: pd.DataFrame
        features: pd.DataFrame
        live_rates: dict[str, Any]
        news: list[dict[str, Any]]
        macro_calendar: list[dict[str, Any]]
        signal_history: pd.DataFrame
        market_notes: str
        source_statuses: list[dict[str, Any]]
        schema_version: str = "1.0"

        @property
        def usdt_btc(self) -> pd.DataFrame:
            """Compatibility alias for the name used in the original class stub."""
            return self.btc_ngn

        def to_dict(self) -> dict[str, Any]:
            """Return a JSON-serializable version of the complete dataset."""
            return {
                "schema_version": self.schema_version,
                "as_of": self.as_of.isoformat(),
                "usdt_ngn": DataSource._frame_to_records(self.usdt_ngn),
                "btc_ngn": DataSource._frame_to_records(self.btc_ngn),
                "external_markets": DataSource._frame_to_records(self.external_markets),
                "features": DataSource._frame_to_records(self.features),
                "live_rates": DataSource._json_value(self.live_rates),
                "news": DataSource._json_value(self.news),
                "macro_calendar": DataSource._json_value(self.macro_calendar),
                "signal_history": DataSource._frame_to_records(self.signal_history),
                "market_notes": self.market_notes,
                "source_statuses": DataSource._json_value(self.source_statuses),
            }

    def __init__(self, settings: Any | None = None):
        if settings is None:
            from app.config import get_settings

            settings = get_settings()
        self.settings = settings
        self._statuses: list[dict[str, Any]] = []
        self._data: DataSource.DataSchema | None = None

    def source(
        self,
        *,
        refresh: bool = True,
        market_notes: str = "",
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> DataSchema:
        """Collect all repository data sources into one predictable schema."""
        self._statuses = []
        usdt_ngn = self.get_historical_usdt_ngn(refresh=refresh)
        btc_ngn = self.get_historical_btc_ngn(usdt_ngn)

        start_at = self._utc_timestamp(start) if start is not None else usdt_ngn.index.min()
        end_at = self._utc_timestamp(end) if end is not None else pd.Timestamp.now(tz=UTC) + pd.Timedelta(days=1)
        usdt_ngn = usdt_ngn.loc[(usdt_ngn.index >= start_at) & (usdt_ngn.index <= end_at)]
        btc_ngn = btc_ngn.loc[(btc_ngn.index >= start_at) & (btc_ngn.index <= end_at)]

        external = self._optional(
            "external_markets",
            lambda: self.get_external_market_data(
                start=start_at - pd.Timedelta(days=10),
                end=end_at,
                refresh=refresh,
            ),
            pd.DataFrame(),
        )
        features = self._optional(
            "features",
            lambda: self.get_features(usdt_ngn, external),
            pd.DataFrame(index=usdt_ngn.index),
        )
        live_rates = self._optional("live_rates", self.get_live_rates, {}) if refresh else {}
        news = self._optional("news", lambda: self.get_news(refresh=refresh), [])
        macro_calendar = self._optional("macro_calendar", self.get_macro_calendar, [])
        signal_history = self._optional("signal_history", self.get_signal_history, pd.DataFrame())

        self._data = self.DataSchema(
            as_of=datetime.now(UTC),
            usdt_ngn=usdt_ngn,
            btc_ngn=btc_ngn,
            external_markets=external,
            features=features,
            live_rates=live_rates,
            news=news,
            macro_calendar=macro_calendar,
            signal_history=signal_history,
            market_notes=market_notes.strip(),
            source_statuses=list(self._statuses),
        )
        return self._data

    def serve(
        self,
        *,
        refresh: bool = True,
        market_notes: str = "",
        start: datetime | str | None = None,
        end: datetime | str | None = None,
    ) -> DataSchema:
        """Get data ready immediately before inference."""
        return self.source(refresh=refresh, market_notes=market_notes, start=start, end=end)

    def get_historical_usdt_ngn(self, *, refresh: bool = False) -> pd.DataFrame:
        """Combine all local BigQuery/runtime USDT-NGN bars and optional live k-lines."""
        frames: list[pd.DataFrame] = []
        paths: list[Path] = []

        for directory in (self.settings.artifacts_dir, self.settings.data_dir):
            paths.extend(sorted(directory.glob(self.settings.export_glob)))
        runtime_path = self.settings.data_dir / self.settings.runtime_bars_filename
        if runtime_path.exists():
            paths.append(runtime_path)

        for path in dict.fromkeys(paths):
            try:
                frame = self._read_bar_file(path)
                frames.append(frame)
                self._status(
                    f"file:{path.name}",
                    "ok",
                    latest_timestamp=frame.index.max(),
                    message=f"{len(frame)} rows",
                )
            except Exception as exc:
                self._status(f"file:{path.name}", "error", message=str(exc))

        if refresh:
            try:
                live_bars = self.get_quidax_klines()
                if not live_bars.empty:
                    frames.append(live_bars)
            except Exception as exc:
                self._status("quidax_klines", "error", message=str(exc))

        if not frames:
            raise FileNotFoundError("No historical or runtime USDT/NGN bars were available.")

        combined = pd.concat(frames)
        combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
        combined.index.name = "bucket_2h"
        return combined

    def get_quidax_klines(self) -> pd.DataFrame:
        """Fetch the current closed USDT/NGN and BTC/NGN public k-lines."""
        from app.services.market_data import QuidaxKlineService
        from scripts.refresh_runtime_data import build_runtime_bars, drop_open_bar

        service = QuidaxKlineService(self.settings)
        usdtngn = service.fetch("usdtngn")
        btcngn = service.fetch("btcngn")
        self._add_market_status(usdtngn.status)
        self._add_market_status(btcngn.status)

        frame = build_runtime_bars(usdtngn.frame, btcngn.frame)
        return drop_open_bar(frame, period_minutes=self.settings.quidax_kline_period_minutes)

    def get_historical_btc_ngn(self, usdt_ngn: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return the BTC/NGN observations already carried by the Quidax history."""
        frame = usdt_ngn if usdt_ngn is not None else self.get_historical_usdt_ngn()
        columns = {
            "btcngn_close": "close",
            "btcngn_volume": "volume",
            "btcngn_trade_count": "trade_count",
        }
        available = [column for column in columns if column in frame.columns]
        if not available:
            return pd.DataFrame(index=frame.index)
        result = frame[available].rename(columns=columns).copy()
        result.index.name = "bucket_2h"
        return result

    def get_external_market_data(
        self,
        *,
        start: datetime | pd.Timestamp,
        end: datetime | pd.Timestamp,
        refresh: bool = True,
    ) -> pd.DataFrame:
        """Return Brent, DXY, VIX, regional FX, official USD/NGN and BTC/USD."""
        from app.services.market_data import ExternalDailyMarketDataService

        service = ExternalDailyMarketDataService(self.settings)
        if refresh:
            result = service.fetch(
                start=self._utc_timestamp(start).to_pydatetime(),
                end=self._utc_timestamp(end).to_pydatetime(),
            )
            for status in result.statuses:
                self._add_market_status(status)
            return result.frame

        cached = service.load_cached()
        if cached is None:
            self._status("external_daily_cache", "missing", message="No cached external market data")
            return pd.DataFrame(columns=list(self.settings.yahoo_tickers.values()))
        frame, statuses = cached
        for status in statuses:
            self._add_market_status(status)
        return frame.loc[
            (frame.index >= self._utc_timestamp(start)) & (frame.index <= self._utc_timestamp(end))
        ]

    def get_features(self, usdt_ngn: pd.DataFrame, external_markets: pd.DataFrame) -> pd.DataFrame:
        """Build the complete saved feature set without loading any prediction model."""
        from app.services.features import PublicFeatureBuilder

        feature_columns = json.loads((self.settings.artifacts_dir / "feature_cols.json").read_text())
        result = PublicFeatureBuilder().build(
            export_frame=usdt_ngn,
            external_daily=external_markets,
            feature_columns=feature_columns,
        )
        self._status(
            "engineered_features",
            "ok",
            latest_timestamp=result.features.index.max(),
            message=f"{len(feature_columns)} features",
        )
        return result.features

    def get_live_rates(self) -> dict[str, Any]:
        """Fetch the live QBOT USDT/NGN and Quidax BTC/NGN tickers."""
        from app.services.market_data import LiveQuoteService

        snapshot = LiveQuoteService(self.settings).fetch()
        for status in snapshot.statuses:
            self._add_market_status(status)
        return {"usdt_ngn": asdict(snapshot.usdtngn), "btc_ngn": asdict(snapshot.btcngn)}

    def get_news(self, *, refresh: bool = True) -> list[dict[str, Any]]:
        """Return all recent news supplied to the system's AI assessment."""
        from app.services.news_aggregator import NewsAggregatorService

        service = NewsAggregatorService(self.settings)
        digest = service.fetch() if refresh else service._load_disk_cache()
        if digest is None:
            self._status("news", "missing", message="No current news cache")
            return []
        for status in digest.source_statuses:
            self._status(
                f"news:{status.name}",
                "ok" if status.ok else "error",
                message=status.error or f"{status.item_count} items",
            )
        return [asdict(item) for item in digest.items]

    def get_macro_calendar(self) -> list[dict[str, Any]]:
        """Return the repository's complete recurring macro-event catalogue."""
        from app.macro_calendar import EVENTS

        self._status("macro_calendar", "ok", message=f"{len(EVENTS)} events")
        return [dict(event) for event in EVENTS]

    def get_signal_history(self) -> pd.DataFrame:
        """Return prior model signals and resolved outcomes used by confidence scoring."""
        path = self.settings.base_dir / "app" / "signal_log.csv"
        if not path.exists():
            self._status("signal_history", "missing", message=str(path))
            return pd.DataFrame()
        frame = pd.read_csv(path)
        if "datetime" in frame.columns:
            frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        self._status(
            "signal_history",
            "ok",
            latest_timestamp=frame["datetime"].max() if "datetime" in frame.columns and not frame.empty else None,
            message=f"{len(frame)} signals",
        )
        return frame

    def _optional(self, source_id: str, getter: Any, fallback: Any) -> Any:
        try:
            return getter()
        except Exception as exc:
            self._status(source_id, "error", message=str(exc))
            return fallback

    def _status(
        self,
        source_id: str,
        status: str,
        *,
        latest_timestamp: Any | None = None,
        message: str | None = None,
    ) -> None:
        self._statuses.append(
            {
                "source_id": source_id,
                "status": status,
                "latest_timestamp": latest_timestamp,
                "message": message,
            }
        )

    def _add_market_status(self, status: Any) -> None:
        self._status(
            status.source_id,
            status.status,
            latest_timestamp=status.latest_timestamp,
            message=status.message,
        )

    @staticmethod
    def _read_bar_file(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if "bucket_2h" not in frame.columns:
            raise ValueError(f"{path} is missing the bucket_2h column")
        frame["bucket_2h"] = pd.to_datetime(frame["bucket_2h"], utc=True)
        frame = frame.sort_values("bucket_2h").set_index("bucket_2h")
        for column in frame.columns:
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().sum() == frame[column].notna().sum():
                frame[column] = converted
        return frame

    @staticmethod
    def _utc_timestamp(value: datetime | str | pd.Timestamp) -> pd.Timestamp:
        timestamp = pd.Timestamp(value)
        return timestamp.tz_localize(UTC) if timestamp.tzinfo is None else timestamp.tz_convert(UTC)

    @staticmethod
    def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        output = frame.copy()
        if not isinstance(output.index, pd.RangeIndex):
            index_name = output.index.name or "timestamp"
            output = output.reset_index().rename(columns={index_name: "timestamp"})
        return [DataSource._json_value(record) for record in output.to_dict(orient="records")]

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): DataSource._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [DataSource._json_value(item) for item in value]
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value.item() if hasattr(value, "item") else value
