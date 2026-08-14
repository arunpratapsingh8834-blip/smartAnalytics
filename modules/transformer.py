"""
transformer.py
---------------
Turns CLEANED data into ANALYSIS-READY and FORECASTING-READY data:
date sorting/aggregation, duplicate-date handling, frequency detection,
and financial derived columns (profit, margin, growth).
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class ForecastReadinessCheck:
    ok: bool
    frequency: str | None
    history_length: int
    warnings: list[str]


def detect_frequency(dates: pd.Series) -> str:
    """Very small heuristic frequency detector based on median gap between sorted dates."""
    d = pd.to_datetime(dates.dropna()).sort_values().unique()
    if len(d) < 2:
        return "unknown"
    diffs = pd.Series(d[1:] - d[:-1]).dt.days
    median_gap = diffs.median()

    if median_gap <= 1:
        return "daily"
    if median_gap <= 8:
        return "weekly"
    if median_gap <= 32:
        return "monthly"
    if median_gap <= 95:
        return "quarterly"
    return "yearly"


def aggregate_duplicate_dates(df: pd.DataFrame, date_col: str, agg: str = "sum") -> pd.DataFrame:
    """Collapses multiple rows on the same date into one, summing (or averaging) numeric columns."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        return df.drop_duplicates(subset=[date_col]).reset_index(drop=True)

    agg_map = {c: agg for c in numeric_cols}
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != date_col]
    for c in non_numeric_cols:
        agg_map[c] = "first"

    grouped = df.groupby(date_col, as_index=False).agg(agg_map)
    return grouped.sort_values(date_col).reset_index(drop=True)


def prepare_forecast_ready_data(
    df: pd.DataFrame, date_col: str, target_col: str
) -> tuple[pd.DataFrame, ForecastReadinessCheck]:
    """
    Sorts by date, removes invalid dates, aggregates duplicate dates,
    and runs sufficiency checks. Returns (prepared_df, readiness_report).
    """
    warnings: list[str] = []
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    n_invalid = int(df[date_col].isna().sum())
    if n_invalid:
        warnings.append(f"{n_invalid} rows had invalid/unparseable dates and were removed.")
    df = df[df[date_col].notna()]

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    n_invalid_target = int(df[target_col].isna().sum())
    if n_invalid_target:
        warnings.append(f"{n_invalid_target} rows had non-numeric target values and were removed.")
    df = df[df[target_col].notna()]

    df = df.sort_values(date_col)

    n_dupe_dates = int(df[date_col].duplicated().sum())
    if n_dupe_dates:
        warnings.append(f"{n_dupe_dates} duplicate dates found and aggregated (summed).")
        df = aggregate_duplicate_dates(df, date_col)

    frequency = detect_frequency(df[date_col])
    history_length = len(df)

    ok = True
    if history_length < 10:
        warnings.append(
            f"Only {history_length} data points available. Forecasts will be unreliable below ~10 periods."
        )
        ok = False
    elif history_length < 30:
        warnings.append(
            f"Only {history_length} data points available. Forecasts are possible but confidence intervals will be wide."
        )

    if frequency == "unknown":
        warnings.append("Could not confidently detect data frequency (daily/weekly/monthly).")
        ok = False

    return df.reset_index(drop=True), ForecastReadinessCheck(
        ok=ok, frequency=frequency, history_length=history_length, warnings=warnings
    )


def convert_horizon_to_periods(horizon_label: str, frequency: str) -> int:
    """
    Converts a user-facing horizon like '30 days' / '3 months' / '6 months' into
    the correct number of PERIODS for the detected frequency, so e.g. monthly data
    with a '6 months' horizon forecasts 6 periods, not 180.
    """
    horizon_label = horizon_label.lower().strip()

    # extract number
    num = int("".join(ch for ch in horizon_label if ch.isdigit()) or 0)
    is_months = "month" in horizon_label
    is_days = "day" in horizon_label
    is_weeks = "week" in horizon_label

    if frequency == "daily":
        if is_days:
            return num
        if is_weeks:
            return num * 7
        if is_months:
            return num * 30
    elif frequency == "weekly":
        if is_weeks:
            return num
        if is_days:
            return max(1, round(num / 7))
        if is_months:
            return round(num * 4.33)
    elif frequency == "monthly":
        if is_months:
            return num
        if is_days:
            return max(1, round(num / 30))
        if is_weeks:
            return max(1, round(num / 4.33))
    elif frequency in ("quarterly", "yearly"):
        if is_months:
            return max(1, round(num / 3)) if frequency == "quarterly" else max(1, round(num / 12))

    # fallback: treat the raw number as periods
    return max(num, 1)


# --------------------------------------------------------- financial cols --

def compute_financial_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Adds total_expenses / profit / profit_margin / revenue_growth / expense_growth
    ONLY where the required source columns exist. Returns (df, columns_added).
    """
    df = df.copy()
    added = []

    cost_cols = [c for c in ["cost", "operating_expenses", "marketing_expenses"] if c in df.columns]
    if cost_cols:
        df["total_expenses"] = df[cost_cols].sum(axis=1)
        added.append("total_expenses")

    if "revenue" in df.columns and "total_expenses" in df.columns:
        df["profit"] = df["revenue"] - df["total_expenses"]
        added.append("profit")

        # avoid divide-by-zero
        df["profit_margin"] = np.where(
            df["revenue"] != 0, (df["profit"] / df["revenue"]) * 100, np.nan
        )
        added.append("profit_margin")

    if "revenue" in df.columns:
        df["revenue_growth"] = df["revenue"].pct_change() * 100
        added.append("revenue_growth")

    if "total_expenses" in df.columns:
        df["expense_growth"] = df["total_expenses"].pct_change() * 100
        added.append("expense_growth")

    return df, added
