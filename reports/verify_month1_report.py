"""Perform structural checks on the generated Month 1 PDF report."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


REPORT_PATH = Path(__file__).resolve().parents[1] / "output" / "pdf" / "month1_experiment_report.pdf"
EXPECTED_PAGES = 8
FORBIDDEN_TEXT = ("TODO", "PLACEHOLDER", "codex-file-citation")


def main() -> None:
    if not REPORT_PATH.exists():
        raise FileNotFoundError(f"Report not found: {REPORT_PATH}")

    reader = PdfReader(str(REPORT_PATH))
    if len(reader.pages) != EXPECTED_PAGES:
        raise AssertionError(f"Expected {EXPECTED_PAGES} pages, found {len(reader.pages)}")

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if len(text.strip()) < 80:
            raise AssertionError(f"Page {page_number} contains too little extractable text")
        if f"Page {page_number} of {EXPECTED_PAGES}" not in text:
            raise AssertionError(f"Page {page_number} is missing its expected page label")
        for forbidden in FORBIDDEN_TEXT:
            if forbidden.lower() in text.lower():
                raise AssertionError(f"Page {page_number} contains forbidden text: {forbidden}")

    print(f"Verified {REPORT_PATH}")
    print(f"Pages: {len(reader.pages)}")
    print(f"Size: {REPORT_PATH.stat().st_size:,} bytes")
    print(f"Title: {reader.metadata.title}")


if __name__ == "__main__":
    main()
