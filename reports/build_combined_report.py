"""Build one single-file PDF covering Month 1 through Month 2-3 progress.

This does not replace the two gate-evidence PDFs
(output/pdf/month1_experiment_report.pdf and
output/pdf/month2_3_progress_report.pdf); those stay exactly as they are
because reports/verify_gate1_artifacts.py hard-requires the Month 1 file to
exist as a tracked Git artifact for Gate 1's evidence trail. This script
reuses the same section content, unmodified, and concatenates it into one
continuously paginated document for easier day-to-day tracking and sharing.

Run from the repository root:
    python reports/build_combined_report.py

The PDF is written to output/pdf/federated_unlearning_progress_report.pdf.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, SimpleDocTemplate
from reportlab.pdfgen import canvas as canvas_module

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.build_month1_report import (  # noqa: E402
    BODY_FONT,
    GRID,
    MID_GRAY,
    build_month1_story,
)
from reports.build_progress_report import build_month2_3_story  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "output" / "pdf" / "federated_unlearning_progress_report.pdf"

HEADER_LEFT = "FEDERATED UNLEARNING THESIS"
HEADER_RIGHT = "PROGRESS REPORT (MONTHS 1-3)"


class NumberedCanvas(canvas_module.Canvas):
    """Draws 'Page X of Y' by buffering pages until the true total is known."""

    def __init__(self, *args, **kwargs):
        canvas_module.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_header_footer(total_pages)
            canvas_module.Canvas.showPage(self)
        canvas_module.Canvas.save(self)

    def _draw_header_footer(self, total_pages: int) -> None:
        page_number = self._pageNumber
        self.saveState()
        if page_number > 1:
            self.setStrokeColor(GRID)
            self.setLineWidth(0.5)
            self.line(0.65 * inch, 10.48 * inch, 7.85 * inch, 10.48 * inch)
            self.setFont(BODY_FONT, 7.5)
            self.setFillColor(MID_GRAY)
            self.drawString(0.65 * inch, 10.57 * inch, HEADER_LEFT)
            self.drawRightString(7.85 * inch, 10.57 * inch, HEADER_RIGHT)
        self.setFont(BODY_FONT, 7.5)
        self.setFillColor(MID_GRAY)
        self.drawCentredString(
            4.25 * inch, 0.38 * inch, f"Page {page_number} of {total_pages}"
        )
        self.restoreState()


def build_report() -> None:
    story = build_month1_story() + [PageBreak()] + build_month2_3_story()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=LETTER,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.62 * inch,
        title="Federated Unlearning Thesis - Progress Report",
        author="Federated Unlearning Thesis Project",
        subject="Month 1 centralized ML/DL foundations through Month 2-3 FedAvg/FedProx experiments",
    )
    document.build(story, canvasmaker=NumberedCanvas)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    build_report()
