from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path


TRANSFORMER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRANSFORMER_DIR))

from acquire_history import (
    AcquisitionClient,
    extract_nbs_zip,
    parse_date,
    parse_dmo_rows,
    parse_nbs_cpi_resources,
    write_manifest,
)


class HistoricalAcquisitionTests(unittest.TestCase):
    def test_parses_source_date_formats(self) -> None:
        self.assertEqual(parse_date("2026-09-01", "ratedate"), date(2026, 9, 1))
        self.assertEqual(parse_date("01/09/2026", "postDate"), date(2026, 9, 1))
        self.assertEqual(parse_date("September-01-2026", "ratedate"), date(2026, 9, 1))
        self.assertEqual(parse_date("July 2026", "period"), date(2026, 7, 1))
        self.assertIsNone(parse_date(""))

    def test_parses_dmo_document_row(self) -> None:
        page = """
        <tr class="docman_item">
          <time itemprop="datePublished" datetime="2008-01-03 00:00:00"></time>
          <a href="/fgn-bonds/bonds-auction-results/422-result/file"
             data-title="Summary &amp; Result" data-id="422"></a>
        </tr>
        """

        self.assertEqual(
            parse_dmo_rows(page),
            [
                {
                    "date": "2008-01-03",
                    "source": "Debt Management Office Nigeria",
                    "document_type": "fgn_bond_auction_result",
                    "title": "Summary & Result",
                    "url": "https://www.dmo.gov.ng/fgn-bonds/bonds-auction-results/422-result/file",
                }
            ],
        )

    def test_parses_nbs_cpi_resource_and_reference_month(self) -> None:
        page = """
        <div class="colx resource alternate">
          <span class="resource-info" id="1432">
            <i class="far fa-plus-square"></i> July 2026 CPI Report
          </span>
          <a target="_blank"
             href="https://microdata.nigerianstat.gov.ng/index.php/catalog/154/download/1432"
             title="CPI_JULY_2026.zip" data-extension="zip">
             Download [ZIP, 2.38 MB]
          </a>
          <td class="caption">Date</td><td>2026-08-17</td>
          <div class="file">CPI_JULY_2026.pdf</div>
          <div class="file">cpi_1New_July2026.xlsx</div>
        </div>
        """

        self.assertEqual(
            parse_nbs_cpi_resources(page),
            [
                {
                    "resource_id": "1432",
                    "period_date": "2026-07-01",
                    "period_precision": "month",
                    "publication_date": "2026-08-17",
                    "source": "National Bureau of Statistics, Nigeria",
                    "document_type": "cpi_package",
                    "title": "July 2026 CPI Report",
                    "filename": "CPI_JULY_2026.zip",
                    "extension": "zip",
                    "download_label": "Download [ZIP, 2.38 MB]",
                    "url": "https://microdata.nigerianstat.gov.ng/index.php/catalog/154/download/1432",
                    "preview_files": '["CPI_JULY_2026.pdf", "cpi_1New_July2026.xlsx"]',
                }
            ],
        )

    def test_safely_extracts_only_supported_nbs_files(self) -> None:
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.writestr("tables.xlsx", b"sheet")
            archive.writestr("report.pdf", b"pdf")
            archive.writestr("../escape.csv", b"escape")
            archive.writestr("script.exe", b"binary")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = extract_nbs_zip(archive_bytes.getvalue(), root)
            self.assertEqual(
                sorted(path.name for path in paths), ["report.pdf", "tables.xlsx"]
            )
            self.assertFalse((root.parent / "escape.csv").exists())

    def test_manifest_merges_existing_source_records_by_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "metadata" / "acquisition_manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps({"files": [{"path": "raw/old.json", "sha256": "old"}]}),
                encoding="utf-8",
            )
            client = AcquisitionClient(root, pause_seconds=0)
            client.manifest.append({"path": "raw/new.json", "sha256": "new"})
            try:
                write_manifest(client)
            finally:
                client.close()

            written = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["path"] for item in written["files"]],
                ["raw/new.json", "raw/old.json"],
            )


if __name__ == "__main__":
    unittest.main()
