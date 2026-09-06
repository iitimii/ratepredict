"""Build one canonical long-form CSV for all currently acquired dataset records.

Each row is one quantitative feature observation, document, sitemap, or metadata
record. This avoids a very sparse wide CSV while retaining enough information to
pivot quantitative features by date and source for model training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "ratepredict_dataset.csv"

OUTPUT_COLUMNS = [
    "record_id",
    "observed_at",
    "period_year",
    "modality",
    "source_id",
    "record_type",
    "feature_name",
    "numeric_value",
    "text_value",
    "title",
    "url",
    "unit",
    "frequency",
    "quality_status",
    "source_path",
    "metadata_json",
]

QUANT_FILES = {
    "cbn_exchange_rates": Path("curated/quantitative/cbn_exchange_rates.csv"),
    "cbn_usd_ngn_official": Path(
        "curated/quantitative/cbn_usd_ngn_official.csv"
    ),
    "cbn_money_market": Path("curated/quantitative/cbn_money_market.csv"),
    "cbn_money_credit": Path("curated/quantitative/cbn_money_credit.csv"),
    "cbn_inflation": Path("curated/quantitative/cbn_inflation.csv"),
    "cbn_government_securities": Path(
        "curated/quantitative/cbn_government_securities.csv"
    ),
    "cbn_reserves": Path("curated/quantitative/cbn_reserves.csv"),
    "cbn_daily_crude": Path("curated/quantitative/cbn_daily_crude.csv"),
    "cbn_nfem": Path("curated/quantitative/cbn_nfem_rates.csv"),
    "world_bank_nigeria": Path(
        "curated/quantitative/world_bank_nigeria.csv"
    ),
    "legacy_external_daily": Path("latest/external_daily.csv"),
    "quidax_runtime_2h": Path("latest/quidax_runtime_2h.csv"),
    "quidax_usdtngn_btcngn_2h": Path(
        "usdngn training data - usdngn training data.csv"
    ),
}

FREQUENCIES = {
    "cbn_exchange_rates": "business-day-currency",
    "cbn_usd_ngn_official": "business-day",
    "cbn_money_market": "monthly",
    "cbn_money_credit": "monthly",
    "cbn_inflation": "monthly",
    "cbn_government_securities": "auction-event",
    "cbn_reserves": "business-day",
    "cbn_daily_crude": "business-day",
    "cbn_nfem": "business-day",
    "world_bank_nigeria": "annual",
    "legacy_external_daily": "calendar-day-reference-only",
    "quidax_runtime_2h": "2-hour",
    "quidax_usdtngn_btcngn_2h": "2-hour",
    "fred_global_macro": "daily-or-monthly",
}

DOCUMENT_FILES = {
    "cbn_circulars": Path("curated/unstructured/cbn_circulars_index.csv"),
    "cbn_press_releases": Path(
        "curated/unstructured/cbn_press_releases_index.csv"
    ),
    "dmo_fgn_bond_auction_results": Path(
        "curated/unstructured/dmo_fgn_bond_auction_results_index.csv"
    ),
    "fomc_documents": Path("curated/unstructured/fomc_document_index.csv"),
    "boe_mpc_minutes": Path("curated/unstructured/boe_mpc_minutes_index.csv"),
    "nbs_monthly_cpi": Path("curated/unstructured/nbs_cpi_resources_index.csv"),
}

SITEMAP_FILES = {
    "businessday_ng": Path(
        "curated/unstructured/publisher_sitemaps/businessday_ng.csv"
    ),
    "nairametrics": Path(
        "curated/unstructured/publisher_sitemaps/nairametrics.csv"
    ),
    "premium_times": Path(
        "curated/unstructured/publisher_sitemaps/premium_times.csv"
    ),
    "vanguard_ng": Path(
        "curated/unstructured/publisher_sitemaps/vanguard_ng.csv"
    ),
}

NON_FEATURE_COLUMNS = {
    "date",
    "bucket_2h",
    "id",
    "ratedate",
    "postdate",
    "movedate",
    "period",
    "tyear",
    "tmonth",
    "currency",
    "country",
    "indicator_id",
    "indicator",
    "description",
    "series_id",
    "unit",
    "obs_status",
    "decimal",
    "auctiondate",
    "maturitydate",
    "securitytype",
    "tenor",
    "auctionno",
    "auction",
    "week",
    "rangebid",
    "successfulbidrates",
    "ratedescription",
    "nettype",
}

RUNTIME_PLACEHOLDERS = {
    "trade_count",
    "buy_volume",
    "sell_volume",
    "buy_count",
    "sell_count",
    "avg_trade_size",
    "max_trade_size",
    "stddev_trade_size",
    "large_trade_count",
    "large_trade_volume",
    "intrabar_price_stddev",
    "intrabar_range",
    "unique_buyers",
    "unique_sellers",
    "btcngn_trade_count",
}


def clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    return value


def json_text(value: dict[str, Any]) -> str:
    clean = {key: clean_scalar(item) for key, item in value.items()}
    return json.dumps(clean, ensure_ascii=False, separators=(",", ":"))


def stable_id(*parts: Any) -> str:
    material = "\x1f".join(str(part or "") for part in parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:24]


def timestamp_value(row: dict[str, Any]) -> str:
    value = clean_scalar(row.get("date"))
    if value is None:
        value = clean_scalar(row.get("bucket_2h"))
    if value is None:
        value = clean_scalar(row.get("auctionDate"))
    if value is None:
        value = clean_scalar(row.get("publication_date"))
    if value is None:
        value = clean_scalar(row.get("period_date"))
    if value is None or str(value).strip() == "":
        return ""
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return str(value)
    return parsed.isoformat()


def numeric_value(value: Any) -> float | int | None:
    value = clean_scalar(value)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def market_quality(source_id: str, row: dict[str, Any], feature: str) -> str:
    if source_id == "quidax_runtime_2h" and feature in RUNTIME_PLACEHOLDERS:
        if numeric_value(row.get(feature)) == 0:
            return "unavailable_placeholder"
    if source_id == "quidax_usdtngn_btcngn_2h":
        high = numeric_value(row.get("high"))
        low = numeric_value(row.get("low"))
        close = numeric_value(row.get("close"))
        if high is not None and close not in (None, 0) and high > close * 3:
            return "suspicious_bad_tick"
        if high is not None and low is not None and high < low:
            return "invalid_range"
    if source_id.startswith("legacy_"):
        return "reference_only"
    return "unreviewed"


def feature_name(source_id: str, row: dict[str, Any], column: str) -> str:
    if source_id in {"cbn_exchange_rates", "cbn_usd_ngn_official"}:
        currency = slug(row.get("currency") or "us_dollar")
        return f"{source_id}.{currency}.{column}"
    if source_id == "cbn_government_securities":
        security = slug(row.get("securityType") or "security")
        tenor = slug(row.get("tenor") or "unknown_tenor")
        return f"{source_id}.{security}.{tenor}.{column}"
    return f"{source_id}.{column}"


def quant_rows(
    data_root: Path, files: dict[str, Path] | None = None
) -> Iterator[dict[str, Any]]:
    for source_id, relative_path in (files or QUANT_FILES).items():
        path = data_root / relative_path
        frame = pd.read_csv(path, low_memory=False)
        for row_number, raw_row in enumerate(frame.to_dict(orient="records"), start=1):
            observed_at = timestamp_value(raw_row)
            metadata = {
                key: clean_scalar(value)
                for key, value in raw_row.items()
                if key.lower() in NON_FEATURE_COLUMNS and clean_scalar(value) is not None
            }
            for column, value in raw_row.items():
                if column.lower() in NON_FEATURE_COLUMNS:
                    continue
                number = numeric_value(value)
                if number is None:
                    continue
                name = feature_name(source_id, raw_row, column)
                yield {
                    "record_id": stable_id(source_id, row_number, column, observed_at),
                    "observed_at": observed_at,
                    "period_year": metadata.get("tyear") or "",
                    "modality": "quantitative",
                    "source_id": source_id,
                    "record_type": "feature_observation",
                    "feature_name": name,
                    "numeric_value": number,
                    "text_value": "",
                    "title": "",
                    "url": "",
                    "unit": metadata.get("unit") or "",
                    "frequency": FREQUENCIES[source_id],
                    "quality_status": market_quality(source_id, raw_row, column),
                    "source_path": str(relative_path),
                    "metadata_json": json_text(metadata),
                }


def fred_rows(data_root: Path) -> Iterator[dict[str, Any]]:
    directory = data_root / "curated" / "quantitative" / "fred"
    for path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        relative_path = path.relative_to(data_root)
        for row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
            number = numeric_value(row.get("value"))
            if number is None:
                continue
            observed_at = timestamp_value(row)
            series_id = str(row.get("series_id") or path.stem)
            yield {
                "record_id": stable_id("fred", series_id, row_number, observed_at),
                "observed_at": observed_at,
                "period_year": "",
                "modality": "quantitative",
                "source_id": "fred_global_macro",
                "record_type": "feature_observation",
                "feature_name": f"fred.{series_id}",
                "numeric_value": number,
                "text_value": "",
                "title": "",
                "url": f"https://fred.stlouisfed.org/series/{series_id}",
                "unit": "",
                "frequency": FREQUENCIES["fred_global_macro"],
                "quality_status": (
                    "partial_window" if series_id == "BAMLH0A0HYM2" else "unreviewed"
                ),
                "source_path": str(relative_path),
                "metadata_json": json_text(
                    {"series_id": series_id, "description": row.get("description")}
                ),
            }


def world_bank_rows(data_root: Path) -> Iterator[dict[str, Any]]:
    relative_path = QUANT_FILES["world_bank_nigeria"]
    frame = pd.read_csv(data_root / relative_path, low_memory=False)
    for row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
        number = numeric_value(row.get("value"))
        observed_at = timestamp_value(row)
        indicator_id = str(row.get("indicator_id") or "unknown")
        metadata = {
            "country": clean_scalar(row.get("country")),
            "indicator": clean_scalar(row.get("indicator")),
            "obs_status": clean_scalar(row.get("obs_status")),
            "decimal": clean_scalar(row.get("decimal")),
        }
        yield {
            "record_id": stable_id("world_bank", indicator_id, row_number, observed_at),
            "observed_at": observed_at,
            "period_year": observed_at[:4] if observed_at else "",
            "modality": "quantitative",
            "source_id": "world_bank_nigeria",
            "record_type": "feature_observation",
            "feature_name": f"world_bank.{indicator_id}",
            "numeric_value": "" if number is None else number,
            "text_value": "",
            "title": "",
            "url": f"https://data.worldbank.org/indicator/{indicator_id}?locations=NG",
            "unit": clean_scalar(row.get("unit")) or "",
            "frequency": "annual",
            "quality_status": "missing_at_source" if number is None else "unreviewed",
            "source_path": str(relative_path),
            "metadata_json": json_text(metadata),
        }


def combined_document_text(row: dict[str, Any]) -> str:
    parts = [row.get("title"), row.get("description"), row.get("keywords")]
    return "\n".join(str(part).strip() for part in parts if clean_scalar(part))


def document_rows(data_root: Path) -> Iterator[dict[str, Any]]:
    for source_id, relative_path in DOCUMENT_FILES.items():
        frame = pd.read_csv(data_root / relative_path, low_memory=False)
        for row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
            observed_at = timestamp_value(row)
            period_year = clean_scalar(row.get("year")) or (
                observed_at[:4] if observed_at else ""
            )
            title = clean_scalar(row.get("title")) or ""
            url = clean_scalar(row.get("url")) or ""
            metadata = {
                key: clean_scalar(value)
                for key, value in row.items()
                if key not in {"date", "year", "title", "url"}
                and clean_scalar(value) is not None
            }
            yield {
                "record_id": stable_id(source_id, row_number, title, url),
                "observed_at": observed_at,
                "period_year": period_year,
                "modality": "unstructured",
                "source_id": source_id,
                "record_type": clean_scalar(row.get("document_type")) or "document_index",
                "feature_name": f"{source_id}.document",
                "numeric_value": "",
                "text_value": combined_document_text(row),
                "title": title,
                "url": url,
                "unit": "",
                "frequency": "event",
                "quality_status": (
                    "body_acquired"
                    if clean_scalar(row.get("package_path"))
                    else "body_download_pending"
                ),
                "source_path": str(relative_path),
                "metadata_json": json_text(metadata),
            }


def sitemap_rows(data_root: Path) -> Iterator[dict[str, Any]]:
    for source_id, relative_path in SITEMAP_FILES.items():
        frame = pd.read_csv(data_root / relative_path, low_memory=False)
        for row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
            url = clean_scalar(row.get("sitemap_url")) or ""
            last_modified = clean_scalar(row.get("last_modified")) or ""
            yield {
                "record_id": stable_id(source_id, "sitemap", row_number, url),
                "observed_at": last_modified,
                "period_year": last_modified[:4] if last_modified else "",
                "modality": "unstructured_index",
                "source_id": source_id,
                "record_type": "publisher_sitemap",
                "feature_name": f"{source_id}.sitemap",
                "numeric_value": "",
                "text_value": "",
                "title": "",
                "url": url,
                "unit": "",
                "frequency": "archive-index",
                "quality_status": "article_enumeration_pending",
                "source_path": str(relative_path),
                "metadata_json": "{}",
            }


def metadata_rows(data_root: Path) -> Iterator[dict[str, Any]]:
    catalog_path = data_root / "metadata" / "source_catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    for source in catalog["sources"]:
        yield {
            "record_id": stable_id("source_catalog", source["id"]),
            "observed_at": "",
            "period_year": "",
            "modality": "metadata",
            "source_id": source["id"],
            "record_type": "source_catalog",
            "feature_name": "source.metadata",
            "numeric_value": "",
            "text_value": source.get("feature_family", ""),
            "title": source.get("publisher", ""),
            "url": source.get("url") or "",
            "unit": "",
            "frequency": source.get("frequency", ""),
            "quality_status": source.get("status", ""),
            "source_path": "metadata/source_catalog.json",
            "metadata_json": json_text(source),
        }

    manifest_path = data_root / "metadata" / "acquisition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        yield {
            "record_id": stable_id("manifest", item["path"], item["sha256"]),
            "observed_at": item.get("fetched_at", ""),
            "period_year": "",
            "modality": "metadata",
            "source_id": "acquisition_manifest",
            "record_type": "raw_file_manifest",
            "feature_name": "file.provenance",
            "numeric_value": item.get("bytes", ""),
            "text_value": "",
            "title": item["path"],
            "url": item.get("url", ""),
            "unit": "bytes",
            "frequency": "acquisition",
            "quality_status": "checksummed",
            "source_path": "metadata/acquisition_manifest.json",
            "metadata_json": json_text(item),
        }

    for plan_name in ("Data Sources - Data.csv", "Data Sources - Events.csv"):
        path = data_root / plan_name
        frame = pd.read_csv(path, low_memory=False)
        for row_number, row in enumerate(frame.to_dict(orient="records"), start=1):
            yield {
                "record_id": stable_id("legacy_plan", plan_name, row_number),
                "observed_at": "",
                "period_year": "",
                "modality": "metadata",
                "source_id": "legacy_source_plan",
                "record_type": "planning_row",
                "feature_name": "source.plan",
                "numeric_value": "",
                "text_value": "",
                "title": "",
                "url": "",
                "unit": "",
                "frequency": "static",
                "quality_status": "reference_only",
                "source_path": plan_name,
                "metadata_json": json_text(row),
            }


def all_rows(data_root: Path) -> Iterable[dict[str, Any]]:
    # World Bank requires one row per indicator/year, including missing-at-source
    # records. Exclude it from generic numeric expansion to avoid duplicates.
    quant_without_world_bank = dict(QUANT_FILES)
    quant_without_world_bank.pop("world_bank_nigeria")
    yield from quant_rows(data_root, quant_without_world_bank)
    yield from world_bank_rows(data_root)
    yield from fred_rows(data_root)
    yield from document_rows(data_root)
    yield from sitemap_rows(data_root)
    yield from metadata_rows(data_root)


def build_csv(data_root: Path, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    counts: dict[str, int] = {}
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            total = 0
            for row in all_rows(data_root):
                writer.writerow(row)
                total += 1
                counts[row["modality"]] = counts.get(row["modality"], 0) + 1
        Path(temporary_name).replace(output)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return {
        "output": str(output),
        "rows": total,
        "bytes": output.stat().st_size,
        "rows_by_modality": counts,
        "built_at": datetime.now(UTC).isoformat(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_csv(args.data_root.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
