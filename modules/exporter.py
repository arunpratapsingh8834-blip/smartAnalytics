"""
exporter.py
-------------
Produces downloadable artifacts: CSV, Excel, PDF summary report, and a
combined ZIP package. All functions return bytes (or a file path for the
ZIP), ready to hand to st.download_button in app.py.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime

import pandas as pd
from fpdf import FPDF


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buffer.getvalue()


class _ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Smart Analytics - Project Summary Report", ln=True, align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, datetime.utcnow().strftime("Generated %Y-%m-%d %H:%M UTC"), ln=True, align="C")
        self.ln(4)

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(235, 235, 245)
        self.cell(0, 8, title, ln=True, fill=True)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 6, text)
        self.ln(2)


def generate_pdf_report(
    dataset_overview: dict,
    cleaning_summary: list[str],
    kpis: dict,
    forecast_summary: dict,
    business_insights: dict,
    limitations: list[str],
) -> bytes:
    pdf = _ReportPDF()
    pdf.add_page()

    pdf.section_title("1. Dataset Overview")
    pdf.body_text(
        f"Rows: {dataset_overview.get('rows', 'n/a')}  |  "
        f"Columns: {dataset_overview.get('columns', 'n/a')}  |  "
        f"Missing: {dataset_overview.get('total_missing_pct', 'n/a')}%  |  "
        f"Duplicates: {dataset_overview.get('duplicate_rows', 'n/a')}"
    )

    pdf.section_title("2. Cleaning & Transformation")
    pdf.body_text("\n".join(f"- {line}" for line in cleaning_summary) or "No cleaning actions recorded.")

    pdf.section_title("3. Key Performance Indicators")
    pdf.body_text("\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in kpis.items()) or "n/a")

    pdf.section_title("4. Forecast Summary")
    pdf.body_text(
        f"Recommended model: {forecast_summary.get('recommended_model', 'n/a')}\n"
        f"Horizon: {forecast_summary.get('horizon', 'n/a')}\n"
        f"Validation MAE: {forecast_summary.get('mae', 'n/a')}  "
        f"RMSE: {forecast_summary.get('rmse', 'n/a')}  "
        f"MAPE: {forecast_summary.get('mape', 'n/a')}"
    )

    pdf.section_title("5. Business Insights")
    insight_lines = []
    if business_insights.get("profit"):
        insight_lines.append(business_insights["profit"])
    insight_lines += business_insights.get("trends", [])
    insight_lines += business_insights.get("risks", [])
    pdf.body_text("\n".join(f"- {line}" for line in insight_lines) or "No notable insights generated.")

    pdf.section_title("6. Limitations")
    default_limitations = [
        "Forecast accuracy is not guaranteed and depends on data quality, seasonality, volatility and horizon.",
        "Model-estimated relationships (e.g. profit drivers) describe association, not proven causation.",
    ]
    pdf.body_text("\n".join(f"- {line}" for line in (limitations or default_limitations)))

    return bytes(pdf.output())


def build_export_zip(
    cleaned_csv: bytes,
    forecast_csv: bytes,
    model_comparison_csv: bytes,
    data_quality_csv: bytes,
    summary_pdf: bytes,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("cleaned_data.csv", cleaned_csv)
        zf.writestr("forecast.csv", forecast_csv)
        zf.writestr("model_comparison.csv", model_comparison_csv)
        zf.writestr("data_quality_report.csv", data_quality_csv)
        zf.writestr("summary_report.pdf", summary_pdf)
    return buffer.getvalue()
