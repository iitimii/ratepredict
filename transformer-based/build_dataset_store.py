"""Build the authoritative partitioned Parquet store for the multimodal dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from build_dataset_csv import all_rows


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "dataset"
STORE_VERSION = "1.0"
BATCH_SIZE = 75_000

RECORD_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("observed_at", pa.timestamp("us", tz="UTC")),
        pa.field("period_year", pa.int16()),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("record_type", pa.string(), nullable=False),
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("numeric_value", pa.float64()),
        pa.field("text_value", pa.string()),
        pa.field("title", pa.string()),
        pa.field("url", pa.string()),
        pa.field("unit", pa.string()),
        pa.field("frequency", pa.string()),
        pa.field("quality_status", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("metadata_json", pa.string(), nullable=False),
    ],
    metadata={
        b"dataset": b"ratepredict_multimodal",
        b"schema_version": STORE_VERSION.encode(),
        b"time_basis": b"UTC",
    },
)

PARTITION_SCHEMA = pa.schema(
    [pa.field("modality", pa.string()), pa.field("source_id", pa.string())]
)


def optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def optional_year(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2200 else None


def optional_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def typed_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": str(row["record_id"]),
        "observed_at": optional_timestamp(row.get("observed_at")),
        "period_year": optional_year(row.get("period_year")),
        "modality": str(row["modality"]),
        "source_id": str(row["source_id"]),
        "record_type": str(row["record_type"]),
        "feature_name": str(row["feature_name"]),
        "numeric_value": optional_number(row.get("numeric_value")),
        "text_value": optional_text(row.get("text_value")),
        "title": optional_text(row.get("title")),
        "url": optional_text(row.get("url")),
        "unit": optional_text(row.get("unit")),
        "frequency": optional_text(row.get("frequency")),
        "quality_status": str(row["quality_status"]),
        "source_path": str(row["source_path"]),
        "metadata_json": str(row["metadata_json"]),
    }


def batches(rows: Iterable[dict[str, Any]], size: int = BATCH_SIZE) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(typed_record(row))
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_partitioned_records(
    data_root: Path, output_root: Path
) -> tuple[int, Counter[str], Counter[str]]:
    records_root = output_root / "records"
    partitioning = ds.partitioning(PARTITION_SCHEMA, flavor="hive")
    total = 0
    modality_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for batch_number, materialized in enumerate(batches(all_rows(data_root))):
        table = pa.Table.from_pylist(materialized, schema=RECORD_SCHEMA)
        modality_counts.update(table["modality"].to_pylist())
        source_counts.update(table["source_id"].to_pylist())
        total += table.num_rows
        ds.write_dataset(
            table,
            records_root,
            format=ds.ParquetFileFormat(),
            partitioning=partitioning,
            basename_template=f"batch-{batch_number:05d}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            file_options=ds.ParquetFileFormat().make_write_options(
                compression="zstd",
                compression_level=6,
                use_dictionary=True,
                write_statistics=True,
            ),
            max_rows_per_file=250_000,
            max_rows_per_group=64_000,
        )
    return total, modality_counts, source_counts


def valid_quidax_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if column != "bucket_2h":
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result["bucket_2h"] = pd.to_datetime(result["bucket_2h"], errors="coerce", utc=True)
    result = result.dropna(subset=["bucket_2h", "open", "high", "low", "close"])
    bad_tick = (result["high"] > result["close"] * 3) | (result["high"] < result["low"])
    return result.loc[~bad_tick].sort_values("bucket_2h")


def daily_quidax(data_root: Path) -> pd.DataFrame:
    path = data_root / "usdngn training data - usdngn training data.csv"
    frame = valid_quidax_rows(pd.read_csv(path, low_memory=False))
    frame["date"] = frame["bucket_2h"].dt.floor("D").dt.tz_localize(None)
    additive = [
        column
        for column in (
            "volume",
            "trade_count",
            "buy_volume",
            "sell_volume",
            "buy_count",
            "sell_count",
            "large_trade_count",
            "large_trade_volume",
            "btcngn_volume",
            "btcngn_trade_count",
        )
        if column in frame
    ]
    aggregations: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "btcngn_close": "last",
        "implied_btcusd_quidax": "last",
    }
    aggregations.update({column: "sum" for column in additive})
    daily = frame.groupby("date", sort=True).agg(aggregations)
    daily = daily.rename(
        columns={
            "open": "usdtngn_open",
            "high": "usdtngn_high",
            "low": "usdtngn_low",
            "close": "usdtngn_close",
            "volume": "usdtngn_volume",
            "trade_count": "usdtngn_trade_count",
            "buy_volume": "usdtngn_buy_volume",
            "sell_volume": "usdtngn_sell_volume",
            "buy_count": "usdtngn_buy_count",
            "sell_count": "usdtngn_sell_count",
            "large_trade_count": "usdtngn_large_trade_count",
            "large_trade_volume": "usdtngn_large_trade_volume",
        }
    )
    return daily


def read_date_index(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.tz_localize(None)
    return frame.dropna(subset=["date"]).drop_duplicates("date", keep="last").set_index("date")


def daily_quantitative_view(data_root: Path, output_root: Path) -> dict[str, Any]:
    catalog = json.loads((data_root / "metadata" / "source_catalog.json").read_text())
    start = catalog["target_window"]["start"]
    end = catalog["target_window"]["end"]
    calendar = pd.DataFrame(index=pd.date_range(start, end, freq="D", name="date"))

    usd = read_date_index(data_root / "curated/quantitative/cbn_usd_ngn_official.csv")
    reserves = read_date_index(data_root / "curated/quantitative/cbn_reserves.csv")
    crude = read_date_index(data_root / "curated/quantitative/cbn_daily_crude.csv")
    money = read_date_index(data_root / "curated/quantitative/cbn_money_market.csv")
    money_credit = read_date_index(
        data_root / "curated/quantitative/cbn_money_credit.csv"
    )
    inflation = read_date_index(data_root / "curated/quantitative/cbn_inflation.csv")
    securities = pd.read_csv(
        data_root / "curated/quantitative/cbn_government_securities.csv",
        low_memory=False,
    )
    securities["date"] = pd.to_datetime(securities["date"], errors="coerce")
    for column in (
        "totalSubscription",
        "totalSuccessful",
        "amtOffered",
        "netValue",
        "rate",
    ):
        securities[column] = pd.to_numeric(securities[column], errors="coerce")
    securities = securities.dropna(subset=["date"])
    auction_daily = securities.groupby("date", sort=True).agg(
        cbn_auction_count=("id", "count"),
        cbn_auction_total_subscription=("totalSubscription", "sum"),
        cbn_auction_total_successful=("totalSuccessful", "sum"),
        cbn_auction_amount_offered=("amtOffered", "sum"),
        cbn_auction_net_value=("netValue", "sum"),
        cbn_auction_rate_mean=("rate", "mean"),
    )
    quidax = daily_quidax(data_root)

    selected = calendar.join(
        usd[["buyingrate", "centralrate", "sellingrate"]].rename(
            columns=lambda column: f"cbn_usdngn_{column}"
        )
    )
    selected = selected.join(
        reserves[["gross", "liquid", "blocked", "blockPercent"]].rename(
            columns={
                "gross": "foreign_reserves_gross",
                "liquid": "foreign_reserves_liquid",
                "blocked": "foreign_reserves_blocked",
                "blockPercent": "foreign_reserves_blocked_pct",
            }
        )
    )
    selected = selected.join(
        crude[["crudeOilPrice"]].rename(columns={"crudeOilPrice": "nigeria_crude_price"})
    )
    money_columns = [
        column
        for column in ("mpr", "interBankCallRate", "treasuryBill", "primeLending", "maxLending")
        if column in money
    ]
    selected = selected.join(
        money[money_columns].rename(
            columns={
                "interBankCallRate": "cbn_interbank_call_rate",
                "treasuryBill": "cbn_treasury_bill_rate",
                "primeLending": "cbn_prime_lending_rate",
                "maxLending": "cbn_max_lending_rate",
            }
        )
    )
    money_credit_columns = [
        column
        for column in (
            "moneySupply_M2",
            "moneySupply_M3",
            "narrowMoney",
            "quasiMoney",
            "netForeignAssets",
            "netDomesticAssets",
            "netDomesticCredit",
            "creditToGovernment",
            "creditToPrivateSector",
            "baseMoney",
            "currencyInCirculation",
        )
        if column in money_credit
    ]
    selected = selected.join(
        money_credit[money_credit_columns].rename(
            columns={column: f"cbn_{column}" for column in money_credit_columns}
        )
    )
    inflation_columns = [
        column
        for column in (
            "allItemsYearOn",
            "allItemsAverage",
            "foodYearOn",
            "foodAverage",
            "allItemsLessFrmProdYearOn",
            "allItemsLessFrmProdAverage",
            "allItemsLessFrmProdAndEnergyYearOn",
            "allItemsLessFrmProdAndEnergyAvg",
        )
        if column in inflation
    ]
    selected = selected.join(
        inflation[inflation_columns].rename(
            columns={column: f"nigeria_inflation_{column}" for column in inflation_columns}
        )
    )
    selected = selected.join(auction_daily)
    selected = selected.join(quidax)

    fred_frames = []
    for path in sorted((data_root / "curated/quantitative/fred").glob("*.csv")):
        frame = read_date_index(path)
        series_id = str(frame["series_id"].dropna().iloc[0]) if not frame.empty else path.stem
        fred_frames.append(frame[["value"]].rename(columns={"value": f"fred_{series_id}"}))
    if fred_frames:
        selected = selected.join(pd.concat(fred_frames, axis=1))

    selected.insert(0, "date", selected.index)
    table = pa.Table.from_pandas(selected.reset_index(drop=True), preserve_index=False)
    path = output_root / "views" / "daily_quantitative_observed.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=7_550,
    )
    return {
        "path": str(path.relative_to(output_root)),
        "rows": table.num_rows,
        "columns": table.num_columns,
        "start": selected.index.min().date().isoformat(),
        "end": selected.index.max().date().isoformat(),
    }


def write_empty_embedding_schema(output_root: Path) -> None:
    schema = pa.schema(
        [
            pa.field("date", pa.date32(), nullable=False),
            pa.field("source_id", pa.string(), nullable=False),
            pa.field("model_id", pa.string(), nullable=False),
            pa.field("model_revision", pa.string(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32()), nullable=False),
            pa.field("document_count", pa.int32(), nullable=False),
            pa.field("has_text", pa.bool_(), nullable=False),
        ],
        metadata={b"status": b"awaiting_document_bodies_and_embedding_model"},
    )
    path = output_root / "views" / "daily_text_embeddings.parquet"
    pq.write_table(pa.Table.from_pylist([], schema=schema), path, compression="zstd")


def directory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "_dataset.json":
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def write_dataset_manifest(
    output_root: Path,
    *,
    rows: int,
    modality_counts: Counter[str],
    source_counts: Counter[str],
    daily_view: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "dataset": "ratepredict_multimodal",
        "schema_version": STORE_VERSION,
        "built_at": datetime.now(UTC).isoformat(),
        "format": "Apache Parquet",
        "compression": "zstd level 6",
        "partitioning": ["modality", "source_id"],
        "record_rows": rows,
        "rows_by_modality": dict(sorted(modality_counts.items())),
        "rows_by_source": dict(sorted(source_counts.items())),
        "daily_quantitative_view": daily_view,
        "record_schema": str(RECORD_SCHEMA),
        "dataset_sha256": directory_sha256(output_root),
    }
    (output_root / "_dataset.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def replace_directory(temporary: Path, destination: Path) -> None:
    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.replace(backup)
    try:
        temporary.replace(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def build_store(data_root: Path, output: Path) -> dict[str, Any]:
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        rows, modalities, sources = write_partitioned_records(data_root, temporary)
        daily_view = daily_quantitative_view(data_root, temporary)
        existing_embeddings = output / "views" / "daily_text_embeddings.parquet"
        if existing_embeddings.exists():
            destination = temporary / "views" / existing_embeddings.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing_embeddings, destination)
        else:
            write_empty_embedding_schema(temporary)
        manifest = write_dataset_manifest(
            temporary,
            rows=rows,
            modality_counts=modalities,
            source_counts=sources,
            daily_view=daily_view,
        )
        replace_directory(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "output": str(output),
        "record_rows": rows,
        "partitions": len(sources),
        "daily_view": daily_view,
        "dataset_sha256": manifest["dataset_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_store(args.data_root.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
