"""Acquire long-horizon quantitative and document indexes for the transformer dataset.

The script deliberately separates immutable-ish source responses under ``data/raw``
from normalized tables under ``data/curated``.  It only uses public endpoints and
does not download publisher article bodies or bypass access controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urljoin

import httpx
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_START_DATE = date(2006, 1, 1)
# Some public data CDNs silently deprioritize non-browser user agents.  Keep a
# transparent project token while presenting a conventional browser prefix.
USER_AGENT = (
    "Mozilla/5.0 (compatible; ratepredict-research/0.1; "
    "+https://github.com/oxowolabi/ratepredict)"
)

CBN_QUANT_ENDPOINTS = {
    "exchange_rates": ("https://www.cbn.gov.ng/api/GetAllExchangeRates", "ratedate"),
    "daily_crude": ("https://www.cbn.gov.ng/api/GetAllDailyCrude", "postDate"),
    "reserves": ("https://www.cbn.gov.ng/api/GetAllReserves", "moveDate"),
    "money_market": (
        "https://www.cbn.gov.ng/api/GetAllMoneyMarketIndicators",
        "period",
    ),
    "money_credit": (
        "https://www.cbn.gov.ng/api/GetAllMoneyAndCreditStats",
        "period",
    ),
    "inflation": ("https://www.cbn.gov.ng/api/GetAllInflationRates", "period"),
    "government_securities": (
        "https://www.cbn.gov.ng/api/GetAllSecurities",
        "auctionDate",
    ),
    "nfem_rates": ("https://www.cbn.gov.ng/api/GetAllNFEM_Rates", "ratedate"),
}

CBN_DOCUMENT_ENDPOINTS = {
    "cbn_circulars": "https://www.cbn.gov.ng/api/GetAllCirculars",
    "cbn_press_releases": "https://www.cbn.gov.ng/api/GetAllPressReleases",
}

FRED_SERIES = {
    "DGS2": "US 2-year Treasury constant-maturity yield",
    "DGS10": "US 10-year Treasury constant-maturity yield",
    "DFF": "Effective federal funds rate",
    "DTWEXBGS": "Nominal broad US dollar index",
    "DCOILBRENTEU": "Brent spot price",
    "VIXCLS": "CBOE VIX close",
    "CPIAUCSL": "US consumer price index",
    "T10YIE": "US 10-year breakeven inflation rate",
    "BAMLH0A0HYM2": "US high-yield corporate option-adjusted spread",
}

WORLD_BANK_INDICATORS = {
    "PA.NUS.FCRF": "Official exchange rate (LCU per US$, period average)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "NY.GDP.MKTP.CD": "GDP (current US$)",
    "FI.RES.TOTL.CD": "Total reserves including gold (current US$)",
    "FM.LBL.BMNY.ZG": "Broad money growth (annual %)",
    "BX.KLT.DINV.CD.WD": "Foreign direct investment, net inflows",
    "BX.PEF.TOTL.CD.WD": "Portfolio equity, net inflows",
    "BN.CAB.XOKA.CD": "Current account balance",
    "NY.GDP.PETR.RT.ZS": "Oil rents (% of GDP)",
    "NE.EXP.GNFS.CD": "Exports of goods and services",
    "NE.IMP.GNFS.CD": "Imports of goods and services",
    "BX.TRF.PWKR.CD.DT": "Personal remittances received",
    "CM.MKT.INDX.ZG": "S&P Global Equity Indices annual change",
    "GC.DOD.TOTL.GD.ZS": "Central government debt (% of GDP)",
}

NEWS_SITEMAPS = {
    "nairametrics": "https://nairametrics.com/sitemap_index.xml",
    "businessday_ng": "https://businessday.ng/sitemap_index.xml",
    "vanguard_ng": "https://www.vanguardngr.com/sitemap_index.xml",
    "premium_times": "https://www.premiumtimesng.com/sitemap_index.xml",
}

NBS_CPI_CATALOG_URL = (
    "https://microdata.nigerianstat.gov.ng/index.php/catalog/154/related-materials"
)

MONTH_NUMBERS = {
    month.lower(): number
    for number, month in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}


@dataclass(frozen=True)
class FetchResult:
    url: str
    content: bytes
    content_type: str
    fetched_at: str
    sha256: str


class AnchorParser(HTMLParser):
    """Collect links and their visible text without an HTML dependency."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        self._href = dict(attrs).get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        self.links.append(
            {"url": self._href, "title": " ".join("".join(self._text).split())}
        )
        self._href = None
        self._text = []


class AcquisitionClient:
    def __init__(self, data_root: Path, *, pause_seconds: float = 0.15) -> None:
        self.data_root = data_root
        self.pause_seconds = pause_seconds
        self.http = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
            follow_redirects=True,
            timeout=httpx.Timeout(300.0, connect=30.0),
        )
        self.manifest: list[dict[str, Any]] = []

    def close(self) -> None:
        self.http.close()

    def fetch(self, url: str, *, attempts: int = 4) -> FetchResult:
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.http.get(url)
                response.raise_for_status()
                break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt == attempts:
                    raise
                time.sleep(min(2 ** (attempt - 1), 8))
        else:  # pragma: no cover - the loop either succeeds or raises
            raise RuntimeError(f"Unable to fetch {url}") from last_error
        content = response.content
        result = FetchResult(
            url=str(response.url),
            content=content,
            content_type=response.headers.get("content-type", ""),
            fetched_at=datetime.now(UTC).isoformat(),
            sha256=hashlib.sha256(content).hexdigest(),
        )
        time.sleep(self.pause_seconds)
        return result

    def record(self, result: FetchResult, path: Path, *, rows: int | None = None) -> None:
        self.manifest.append(
            {
                "url": result.url,
                "path": str(path.relative_to(self.data_root)),
                "content_type": result.content_type,
                "fetched_at": result.fetched_at,
                "sha256": result.sha256,
                "bytes": len(result.content),
                "rows": rows,
            }
        )


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)
        temporary = Path(handle.name)
    temporary.replace(path)
    return len(materialized)


def parse_date(value: Any, source_field: str = "") -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    formats = ["%Y-%m-%d", "%d/%m/%Y", "%B-%d-%Y", "%B %Y", "%Y-%m"]
    if source_field == "ratedate" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        formats = ["%Y-%m-%d"] + formats
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date().replace(day=1 if "%d" not in fmt else datetime.strptime(text, fmt).day)
        except ValueError:
            continue
    return None


def normalize_number(value: Any) -> str | float | int | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {".", "..", "...", "-"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return str(value).strip()
    return int(number) if number.is_integer() else number


def acquire_cbn_quant(
    client: AcquisitionClient, start_date: date, end_date: date
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, (url, date_field) in CBN_QUANT_ENDPOINTS.items():
        result = client.fetch(url)
        payload = json.loads(result.content)
        raw_path = client.data_root / "raw" / "quantitative" / "cbn" / f"{name}.json"
        atomic_write_bytes(raw_path, result.content)
        client.record(result, raw_path, rows=len(payload))

        normalized: list[dict[str, Any]] = []
        for source_row in payload:
            row = dict(source_row)
            if name in {"money_market", "money_credit", "inflation"}:
                year = int(row["tyear"])
                month = int(row["tmonth"])
                observed = date(year, month if 1 <= month <= 12 else 1, 1)
            else:
                observed = parse_date(row.get(date_field), date_field)
            if observed is None or not start_date <= observed <= end_date:
                continue
            clean = {key: normalize_number(value) for key, value in row.items()}
            clean["date"] = observed.isoformat()
            normalized.append(clean)

        # CBN occasionally republishes the same natural observation with a
        # higher database id.  Raw responses retain every version; curated
        # tables keep the last id so one date maps to one training value.
        if name == "exchange_rates":
            natural_key = ["date", "currency"]
        elif name == "government_securities":
            natural_key = ["id"]
        else:
            natural_key = ["date"]
        deduplicated: dict[tuple[str, ...], dict[str, Any]] = {}
        for row in sorted(normalized, key=lambda item: int(item.get("id") or 0)):
            key = tuple(str(row.get(column, "")) for column in natural_key)
            deduplicated[key] = row
        normalized = list(deduplicated.values())
        normalized.sort(key=lambda row: (row["date"], str(row.get("currency", ""))))
        fields = ["date"] + sorted(
            {key for row in normalized for key in row if key != "date"}
        )
        output = client.data_root / "curated" / "quantitative" / f"cbn_{name}.csv"
        counts[name] = write_csv(output, normalized, fields)

        if name == "exchange_rates":
            usd_rows = [
                row
                for row in normalized
                if str(row.get("currency", "")).strip().upper() == "US DOLLAR"
            ]
            write_csv(
                client.data_root
                / "curated"
                / "quantitative"
                / "cbn_usd_ngn_official.csv",
                usd_rows,
                fields,
            )
    return counts


def acquire_fred(
    client: AcquisitionClient, start_date: date, end_date: date
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for series_id, description in FRED_SERIES.items():
        url = (
            "https://fred.stlouisfed.org/graph/fredgraph.csv"
            f"?id={series_id}&cosd={start_date.isoformat()}&coed={end_date.isoformat()}"
        )
        result = client.fetch(url)
        raw_path = (
            client.data_root / "raw" / "quantitative" / "fred" / f"{series_id}.csv"
        )
        atomic_write_bytes(raw_path, result.content)

        reader = csv.DictReader(result.content.decode("utf-8-sig").splitlines())
        rows: list[dict[str, Any]] = []
        for source_row in reader:
            observed = source_row.get("observation_date") or source_row.get("DATE")
            value = source_row.get(series_id)
            if observed is None:
                continue
            rows.append(
                {
                    "date": observed,
                    "series_id": series_id,
                    "value": normalize_number(value),
                    "description": description,
                }
            )
        client.record(result, raw_path, rows=len(rows))
        output = (
            client.data_root
            / "curated"
            / "quantitative"
            / "fred"
            / f"{series_id}.csv"
        )
        counts[series_id] = write_csv(
            output, rows, ["date", "series_id", "value", "description"]
        )
    return counts


def acquire_world_bank(
    client: AcquisitionClient, start_date: date, end_date: date
) -> dict[str, int]:
    counts: dict[str, int] = {}
    combined: list[dict[str, Any]] = []
    for indicator_id, description in WORLD_BANK_INDICATORS.items():
        url = (
            f"https://api.worldbank.org/v2/country/NGA/indicator/{indicator_id}"
            f"?date={start_date.year}:{end_date.year}&format=json&per_page=1000&footnote=y"
        )
        result = client.fetch(url)
        payload = json.loads(result.content)
        observations = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        raw_path = (
            client.data_root
            / "raw"
            / "quantitative"
            / "world_bank"
            / f"{indicator_id}.json"
        )
        atomic_write_bytes(raw_path, result.content)
        client.record(result, raw_path, rows=len(observations or []))

        rows = []
        for observation in observations or []:
            row = {
                "date": f"{observation['date']}-01-01",
                "country": observation.get("countryiso3code", "NGA"),
                "indicator_id": indicator_id,
                "indicator": description,
                "value": observation.get("value"),
                "unit": observation.get("unit", ""),
                "obs_status": observation.get("obs_status", ""),
                "decimal": observation.get("decimal"),
            }
            rows.append(row)
            combined.append(row)
        rows.sort(key=lambda row: row["date"])
        counts[indicator_id] = len(rows)

    combined.sort(key=lambda row: (row["date"], row["indicator_id"]))
    write_csv(
        client.data_root / "curated" / "quantitative" / "world_bank_nigeria.csv",
        combined,
        [
            "date",
            "country",
            "indicator_id",
            "indicator",
            "value",
            "unit",
            "obs_status",
            "decimal",
        ],
    )
    return counts


def html_text(fragment: str) -> str:
    return " ".join(
        html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split()
    ).strip()


def infer_nbs_period(title: str, filename: str) -> tuple[str, str]:
    """Infer the statistic's reference month separately from publication time."""

    text = f"{title} {filename}".replace("_", " ").replace("-", " ")
    months = "|".join(MONTH_NUMBERS)
    match = re.search(
        rf"\b({months})\s*(20\d{{2}}|\d{{2}})\b", text, re.I
    ) or re.search(rf"\b(20\d{{2}})\s*({months})\b", text, re.I)
    if match:
        first, second = match.groups()
        if first.lower() in MONTH_NUMBERS:
            month_text, year_text = first, second
        else:
            year_text, month_text = first, second
        year = int(year_text)
        if year < 100:
            year += 2000
        return date(year, MONTH_NUMBERS[month_text.lower()], 1).isoformat(), "month"
    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        return date(int(year_match.group(1)), 1, 1).isoformat(), "year"
    return "", "unknown"


def parse_nbs_cpi_resources(page: str) -> list[dict[str, str]]:
    """Parse public CPI resources from the NBS NADA catalog page."""

    rows: list[dict[str, str]] = []
    for block in re.split(r'<div class="colx resource[^>]*">', page, flags=re.I)[1:]:
        info = re.search(
            r'<span class="resource-info"[\s\S]*?id="(\d+)">([\s\S]*?)</span>',
            block,
            re.I,
        )
        download = re.search(
            r'<a\s+target="_blank"[\s\S]*?href="([^"]+/download/\d+)"'
            r'[\s\S]*?title="([^"]+)"[\s\S]*?data-extension="([^"]+)"'
            r'[\s\S]*?>([\s\S]*?)</a>',
            block,
            re.I,
        )
        if not info or not download:
            continue
        published = re.search(
            r'<td\s+class="caption"\s*>Date</td>\s*<td>(.*?)</td>',
            block,
            re.I | re.S,
        )
        preview = [
            html_text(value)
            for value in re.findall(
                r'<div class="file">([\s\S]*?)</div>', block, re.I
            )
        ]
        title = html_text(info.group(2))
        filename = html.unescape(download.group(2)).strip()
        period_date, period_precision = infer_nbs_period(title, filename)
        published_text = html_text(published.group(1)) if published else ""
        parsed_published = parse_date(published_text)
        rows.append(
            {
                "resource_id": info.group(1),
                "period_date": period_date,
                "period_precision": period_precision,
                "publication_date": (
                    parsed_published.isoformat() if parsed_published else ""
                ),
                "source": "National Bureau of Statistics, Nigeria",
                "document_type": (
                    "cpi_package"
                    if download.group(3).lower() == "zip"
                    else "cpi_table"
                    if download.group(3).lower() in {"xls", "xlsx", "csv"}
                    else "cpi_report"
                ),
                "title": title,
                "filename": filename,
                "extension": download.group(3).lower(),
                "download_label": html_text(download.group(4)),
                "url": html.unescape(download.group(1)),
                "preview_files": json.dumps(preview, ensure_ascii=False),
            }
        )
    return rows


def safe_filename(value: str) -> str:
    name = Path(value.replace("\\", "/")).name.strip()
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return cleaned or "resource.bin"


def extract_nbs_zip(content: bytes, destination: Path) -> list[Path]:
    """Extract bounded, model-relevant files without trusting archive paths."""

    allowed = {".csv", ".pdf", ".txt", ".xls", ".xlsx"}
    extracted: list[Path] = []
    total_size = 0
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.infolist():
            archive_path = PurePosixPath(member.filename.replace("\\", "/"))
            if member.is_dir() or archive_path.is_absolute() or ".." in archive_path.parts:
                continue
            if member.flag_bits & 0x1:
                continue
            suffix = Path(archive_path.name).suffix.lower()
            if suffix not in allowed or member.file_size > 100 * 1024 * 1024:
                continue
            total_size += member.file_size
            if total_size > 250 * 1024 * 1024:
                raise ValueError("NBS archive exceeds the 250 MB extraction limit")
            target = destination / safe_filename(archive_path.name)
            atomic_write_bytes(target, archive.read(member))
            extracted.append(target)
    return extracted


def nbs_feature_slug(value: Any) -> str:
    text = str(value or "").lower().replace("&", " and ").replace("%", " percent ")
    text = text.replace("allitems", "all items").replace("accomodation", "accommodation")
    text = text.replace("non alcoholic bev ", "non alcoholic beverages ")
    normalized = re.sub(r"[^a-z0-9]+", " ", text).strip()
    if normalized.startswith("month on"):
        return "headline_month_on_percent"
    if normalized.startswith("year on"):
        return "headline_year_on_percent"
    if normalized.startswith("12 month average"):
        return "headline_12_month_average_percent"
    normalized = re.sub(r"\bindex\b", "", normalized)
    normalized = " ".join(normalized.split())
    aliases = {
        "all items": "all_items_index",
        "all items less farm produce": "all_items_less_farm_produce_index",
        "all items less farm produce and energy": "core_index",
        "core all items less farm produce and energy": "core_index",
        "core index all items less farm produce and energy": "core_index",
    }
    return aliases.get(normalized, normalized.replace(" ", "_") + "_index")


def parse_nbs_cpi_sheet(frame: pd.DataFrame, *, historical: bool) -> dict[str, dict[str, float]]:
    """Parse the NBS national CPI category table into month-keyed features."""

    if frame.shape[0] < 4 or frame.shape[1] < 3:
        return {}
    selected_columns: list[tuple[int, str]] = []
    if historical:
        # The historical worksheet pairs the former base with the re-referenced
        # 2024=100 series. The right-hand member of each pair is comparable to
        # the current worksheet and is the one retained for model features.
        for column in range(2, frame.shape[1] - 1, 2):
            label = frame.iat[1, column + 1]
            if pd.isna(label):
                label = frame.iat[1, column]
            if pd.isna(label):
                continue
            selected_columns.append((column + 1, nbs_feature_slug(label)))
    else:
        for column in range(2, frame.shape[1]):
            label = frame.iat[1, column]
            if pd.isna(label):
                continue
            selected_columns.append((column, nbs_feature_slug(label)))

    aliases = {name[:3].lower(): number for name, number in MONTH_NUMBERS.items()}
    aliases.update(MONTH_NUMBERS)
    rows: dict[str, dict[str, float]] = {}
    year: int | None = None
    for row_index in range(3, frame.shape[0]):
        raw_year = normalize_number(frame.iat[row_index, 0])
        if isinstance(raw_year, int) and 1900 <= raw_year <= 2100:
            year = raw_year
        raw_month = str(frame.iat[row_index, 1] or "").strip().lower()
        month = aliases.get(raw_month) or aliases.get(raw_month[:3])
        if year is None or month is None:
            continue
        observed = date(year, month, 1).isoformat()
        values = rows.setdefault(observed, {})
        for column, feature in selected_columns:
            value = normalize_number(frame.iat[row_index, column])
            if isinstance(value, (int, float)):
                values[feature] = float(value)
    return rows


def normalize_nbs_cpi_workbook(
    workbook: Path,
    output: Path,
    *,
    start_date: date,
    end_date: date,
    publication_dates: dict[str, str],
    source_path: str,
) -> dict[str, int]:
    excel = pd.ExcelFile(workbook)
    combined: dict[str, dict[str, float]] = {}
    if "Table2 (2)" in excel.sheet_names:
        combined.update(
            parse_nbs_cpi_sheet(
                pd.read_excel(workbook, sheet_name="Table2 (2)", header=None),
                historical=True,
            )
        )
    if "Table2" not in excel.sheet_names:
        raise ValueError(f"NBS workbook has no Table2 sheet: {workbook}")
    current = parse_nbs_cpi_sheet(
        pd.read_excel(workbook, sheet_name="Table2", header=None), historical=False
    )
    for observed, values in current.items():
        combined.setdefault(observed, {}).update(values)

    feature_columns = sorted({feature for values in combined.values() for feature in values})
    rows: list[dict[str, Any]] = []
    for observed, values in sorted(combined.items()):
        observed_date = date.fromisoformat(observed)
        if not start_date <= observed_date <= end_date:
            continue
        rows.append(
            {
                "date": observed,
                "publication_date": publication_dates.get(observed, ""),
                "base_year": 2024,
                "source_workbook": source_path,
                **values,
            }
        )
    write_csv(
        output,
        rows,
        ["date", "publication_date", "base_year", "source_workbook"]
        + feature_columns,
    )
    return {"rows": len(rows), "features": len(feature_columns)}


def acquire_nbs_cpi(
    client: AcquisitionClient, start_date: date, end_date: date
) -> dict[str, int]:
    """Acquire every publicly listed NBS CPI report/table in the requested window."""

    catalog = client.fetch(NBS_CPI_CATALOG_URL)
    catalog_path = (
        client.data_root
        / "raw"
        / "quantitative"
        / "nbs_cpi"
        / "catalog_154_related_materials.html"
    )
    atomic_write_bytes(catalog_path, catalog.content)
    resources = parse_nbs_cpi_resources(
        catalog.content.decode("utf-8", errors="replace")
    )
    client.record(catalog, catalog_path, rows=len(resources))

    selected = []
    for row in resources:
        observed = parse_date(row["period_date"])
        if observed is not None and not start_date <= observed <= end_date:
            continue
        result = client.fetch(row["url"])
        filename = safe_filename(row["filename"])
        package_path = (
            client.data_root
            / "raw"
            / "quantitative"
            / "nbs_cpi"
            / "packages"
            / f"{row['resource_id']}_{filename}"
        )
        atomic_write_bytes(package_path, result.content)

        extracted: list[Path] = []
        if row["extension"] == "zip":
            extracted = extract_nbs_zip(
                result.content,
                client.data_root
                / "raw"
                / "quantitative"
                / "nbs_cpi"
                / "extracted"
                / row["resource_id"],
            )
        client.record(result, package_path, rows=len(extracted) or 1)
        enriched = dict(row)
        enriched["package_path"] = str(package_path.relative_to(client.data_root))
        enriched["extracted_paths"] = json.dumps(
            [str(path.relative_to(client.data_root)) for path in extracted],
            ensure_ascii=False,
        )
        enriched["sha256"] = result.sha256
        enriched["bytes"] = str(len(result.content))
        selected.append(enriched)

    selected.sort(key=lambda row: (row["period_date"], row["resource_id"]))
    fields = [
        "resource_id",
        "period_date",
        "period_precision",
        "publication_date",
        "source",
        "document_type",
        "title",
        "filename",
        "extension",
        "download_label",
        "url",
        "preview_files",
        "package_path",
        "extracted_paths",
        "sha256",
        "bytes",
    ]
    count = write_csv(
        client.data_root
        / "curated"
        / "unstructured"
        / "nbs_cpi_resources_index.csv",
        selected,
        fields,
    )
    workbook_candidates: list[tuple[str, Path]] = []
    publication_dates = {
        row["period_date"]: row["publication_date"]
        for row in selected
        if row["period_date"] and row["publication_date"]
    }
    for row in selected:
        paths = [Path(value) for value in json.loads(row["extracted_paths"])]
        if row["extension"] in {"xls", "xlsx"}:
            paths.append(Path(row["package_path"]))
        for relative in paths:
            if relative.suffix.lower() in {".xls", ".xlsx"}:
                workbook_candidates.append((row["period_date"], client.data_root / relative))
    if not workbook_candidates:
        raise ValueError("NBS CPI catalog contained no downloadable workbook")
    _, latest_workbook = max(workbook_candidates, key=lambda item: item[0])
    normalized = normalize_nbs_cpi_workbook(
        latest_workbook,
        client.data_root / "curated" / "quantitative" / "nbs_cpi_monthly.csv",
        start_date=start_date,
        end_date=end_date,
        publication_dates=publication_dates,
        source_path=str(latest_workbook.relative_to(client.data_root)),
    )
    return {
        "catalog_resources": len(resources),
        "downloaded_resources": count,
        "extracted_files": sum(
            len(json.loads(row["extracted_paths"])) for row in selected
        ),
        "monthly_rows": normalized["rows"],
        "monthly_features": normalized["features"],
    }


def acquire_cbn_documents(
    client: AcquisitionClient, start_date: date, end_date: date
) -> dict[str, int]:
    counts: dict[str, int] = {}
    fields = [
        "date",
        "source",
        "document_type",
        "title",
        "description",
        "keywords",
        "ref_no",
        "url",
        "filesize",
        "source_id",
    ]
    for source, url in CBN_DOCUMENT_ENDPOINTS.items():
        result = client.fetch(url)
        payload = json.loads(result.content)
        raw_path = client.data_root / "raw" / "unstructured" / "cbn" / f"{source}.json"
        atomic_write_bytes(raw_path, result.content)
        client.record(result, raw_path, rows=len(payload))
        rows = []
        for item in payload:
            observed = parse_date(item.get("documentDate"))
            if observed is None or not start_date <= observed <= end_date:
                continue
            rows.append(
                {
                    "date": observed.isoformat(),
                    "source": "Central Bank of Nigeria",
                    "document_type": source.removeprefix("cbn_"),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "keywords": item.get("keywords", ""),
                    "ref_no": item.get("refNo", ""),
                    "url": urljoin("https://www.cbn.gov.ng", item.get("link", "")),
                    "filesize": item.get("filesize", ""),
                    "source_id": item.get("id", ""),
                }
            )
        rows.sort(key=lambda row: (row["date"], row["title"]))
        output = client.data_root / "curated" / "unstructured" / f"{source}_index.csv"
        counts[source] = write_csv(output, rows, fields)
    return counts


def parse_dmo_rows(page: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for block in re.findall(r'<tr class="docman_item".*?</tr>', page, re.I | re.S):
        link = re.search(
            r'<a[^>]+href="([^"]+/file)"[^>]+data-title="([^"]+)"[^>]*>',
            block,
            re.I | re.S,
        )
        published = re.search(r'<time[^>]+datetime="([^"]+)"', block, re.I | re.S)
        if not link:
            continue
        rows.append(
            {
                "date": (published.group(1).split()[0] if published else ""),
                "source": "Debt Management Office Nigeria",
                "document_type": "fgn_bond_auction_result",
                "title": html.unescape(link.group(2)).strip(),
                "url": urljoin("https://www.dmo.gov.ng", html.unescape(link.group(1))),
            }
        )
    return rows


def acquire_dmo_index(
    client: AcquisitionClient, start_date: date, end_date: date
) -> int:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for offset in range(0, 1000, 100):
        url = (
            "https://www.dmo.gov.ng/fgn-bonds/bonds-auction-results"
            f"?limit=100&limitstart={offset}&direction=asc"
        )
        result = client.fetch(url)
        raw_path = (
            client.data_root
            / "raw"
            / "unstructured"
            / "dmo"
            / f"bond_auction_index_{offset:04d}.html"
        )
        atomic_write_bytes(raw_path, result.content)
        page_rows = parse_dmo_rows(result.content.decode("utf-8", errors="replace"))
        client.record(result, raw_path, rows=len(page_rows))
        new_rows = [row for row in page_rows if row["url"] not in seen]
        if not new_rows:
            break
        for row in new_rows:
            seen.add(row["url"])
            observed = parse_date(row["date"])
            if observed is not None and start_date <= observed <= end_date:
                rows.append(row)
        if len(page_rows) < 100:
            break
    rows.sort(key=lambda row: (row["date"], row["title"]))
    return write_csv(
        client.data_root
        / "curated"
        / "unstructured"
        / "dmo_fgn_bond_auction_results_index.csv",
        rows,
        ["date", "source", "document_type", "title", "url"],
    )


def parse_links(page: str, base_url: str) -> list[dict[str, str]]:
    parser = AnchorParser()
    parser.feed(page)
    return [
        {"url": urljoin(base_url, row["url"]), "title": row["title"]}
        for row in parser.links
        if row["url"]
    ]


def acquire_fomc_index(
    client: AcquisitionClient, start_date: date, end_date: date
) -> int:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for year in range(start_date.year, min(end_date.year, 2020) + 1):
        url = f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
        result = client.fetch(url)
        raw_path = (
            client.data_root
            / "raw"
            / "unstructured"
            / "federal_reserve"
            / f"fomc_{year}.html"
        )
        atomic_write_bytes(raw_path, result.content)
        client.record(result, raw_path)
        for link in parse_links(result.content.decode("utf-8", errors="replace"), url):
            target = link["url"].lower()
            title = link["title"].lower()
            if not any(
                token in target or token in title
                for token in ("statement", "minute", "press conference", "projection")
            ):
                continue
            if link["url"] in seen:
                continue
            seen.add(link["url"])
            rows.append(
                {
                    "year": str(year),
                    "source": "Federal Reserve",
                    "document_type": "fomc",
                    "title": link["title"],
                    "url": link["url"],
                }
            )

    if end_date.year >= 2021:
        url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
        result = client.fetch(url)
        raw_path = (
            client.data_root
            / "raw"
            / "unstructured"
            / "federal_reserve"
            / "fomc_current_calendars.html"
        )
        atomic_write_bytes(raw_path, result.content)
        client.record(result, raw_path)
        for link in parse_links(result.content.decode("utf-8", errors="replace"), url):
            target = link["url"].lower()
            match = re.search(r"(20(?:2[1-9]|3\d))", target + " " + link["title"])
            if not match or not any(
                token in target for token in ("fomcminutes", "fomcstatement", "fomcprojtabl")
            ):
                continue
            if link["url"] in seen:
                continue
            seen.add(link["url"])
            rows.append(
                {
                    "year": match.group(1),
                    "source": "Federal Reserve",
                    "document_type": "fomc",
                    "title": link["title"],
                    "url": link["url"],
                }
            )

    rows.sort(key=lambda row: (row["year"], row["url"]))
    return write_csv(
        client.data_root / "curated" / "unstructured" / "fomc_document_index.csv",
        rows,
        ["year", "source", "document_type", "title", "url"],
    )


def acquire_boe_index(
    client: AcquisitionClient, start_date: date, end_date: date
) -> int:
    url = "https://www.bankofengland.co.uk/sitemap/minutes"
    result = client.fetch(url)
    raw_path = client.data_root / "raw" / "unstructured" / "boe" / "minutes.html"
    atomic_write_bytes(raw_path, result.content)
    client.record(result, raw_path)
    rows = []
    seen: set[str] = set()
    for link in parse_links(result.content.decode("utf-8", errors="replace"), url):
        match = re.search(r"/(20\d{2})/", link["url"])
        if not match or link["url"] in seen:
            continue
        year = int(match.group(1))
        if not start_date.year <= year <= end_date.year:
            continue
        title_lower = link["title"].lower()
        if not any(
            token in link["url"].lower()
            for token in ("/minutes/", "monetary-policy-summary-and-minutes")
        ):
            continue
        if "mpc" not in title_lower and "monetary policy" not in title_lower:
            continue
        seen.add(link["url"])
        rows.append(
            {
                "year": str(year),
                "source": "Bank of England",
                "document_type": "mpc_minutes",
                "title": link["title"],
                "url": link["url"],
            }
        )
    rows.sort(key=lambda row: (row["year"], row["url"]))
    return write_csv(
        client.data_root / "curated" / "unstructured" / "boe_mpc_minutes_index.csv",
        rows,
        ["year", "source", "document_type", "title", "url"],
    )


def acquire_news_sitemap_indexes(client: AcquisitionClient) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source, url in NEWS_SITEMAPS.items():
        result = client.fetch(url)
        raw_path = (
            client.data_root
            / "raw"
            / "unstructured"
            / "publisher_sitemaps"
            / f"{source}.xml"
        )
        atomic_write_bytes(raw_path, result.content)
        text = result.content.decode("utf-8", errors="replace")
        locations = [html.unescape(value.strip()) for value in re.findall(r"<loc>(.*?)</loc>", text, re.I | re.S)]
        lastmods = [value.strip() for value in re.findall(r"<lastmod>(.*?)</lastmod>", text, re.I | re.S)]
        rows = [
            {
                "source": source,
                "sitemap_url": location,
                "last_modified": lastmods[index] if index < len(lastmods) else "",
            }
            for index, location in enumerate(locations)
        ]
        client.record(result, raw_path, rows=len(rows))
        counts[source] = write_csv(
            client.data_root
            / "curated"
            / "unstructured"
            / "publisher_sitemaps"
            / f"{source}.csv",
            rows,
            ["source", "sitemap_url", "last_modified"],
        )
    return counts


def write_manifest(client: AcquisitionClient) -> Path:
    path = client.data_root / "metadata" / "acquisition_manifest.json"
    previous_files: list[dict[str, Any]] = []
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous_files = list(previous.get("files", []))
        except (OSError, json.JSONDecodeError, TypeError):
            previous_files = []
    merged = {str(item.get("path")): item for item in previous_files}
    merged.update({str(item.get("path")): item for item in client.manifest})
    write_json(
        path,
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "user_agent": USER_AGENT,
            "files": sorted(merged.values(), key=lambda item: str(item.get("path", ""))),
        },
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        choices=(
            "quant",
            "cbn-quant",
            "fred",
            "nbs",
            "world-bank",
            "documents",
            "news-indexes",
            "all",
        ),
        default="quant",
        help="Source group to acquire (default: quant).",
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=date.today())
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pause-seconds", type=float, default=0.15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.end_date < args.start_date:
        raise SystemExit("--end-date must not be earlier than --start-date")
    client = AcquisitionClient(args.data_root.resolve(), pause_seconds=args.pause_seconds)
    summary: dict[str, Any] = {
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
    }
    try:
        if args.group in {"quant", "cbn-quant", "all"}:
            summary["cbn_quant"] = acquire_cbn_quant(client, args.start_date, args.end_date)
        if args.group in {"quant", "fred", "all"}:
            summary["fred"] = acquire_fred(client, args.start_date, args.end_date)
        if args.group in {"quant", "world-bank", "all"}:
            summary["world_bank"] = acquire_world_bank(client, args.start_date, args.end_date)
        if args.group in {"quant", "nbs", "all"}:
            summary["nbs_cpi"] = acquire_nbs_cpi(
                client, args.start_date, args.end_date
            )
        if args.group in {"documents", "all"}:
            summary["cbn_documents"] = acquire_cbn_documents(
                client, args.start_date, args.end_date
            )
            summary["dmo_documents"] = acquire_dmo_index(
                client, args.start_date, args.end_date
            )
            summary["fomc_documents"] = acquire_fomc_index(
                client, args.start_date, args.end_date
            )
            summary["boe_documents"] = acquire_boe_index(
                client, args.start_date, args.end_date
            )
        if args.group in {"news-indexes", "all"}:
            summary["publisher_sitemaps"] = acquire_news_sitemap_indexes(client)
        summary["manifest"] = str(write_manifest(client))
    finally:
        client.close()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
