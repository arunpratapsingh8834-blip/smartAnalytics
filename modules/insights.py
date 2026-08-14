"""
insights.py
-------------
Rule-based (non-AI) business insight generation. Everything here is derived
directly from computed numbers -- nothing is invented. This is what feeds
ai_engine.py's summary, and also what the app falls back to if no AI API
key is configured.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def _pct_change(series: pd.Series, periods: int) -> float | None:
    if len(series) <= periods:
        return None
    start, end = series.iloc[-periods - 1], series.iloc[-1]
    if start == 0:
        return None
    return round((end - start) / abs(start) * 100, 2)


def generate_trend_insights(df: pd.DataFrame, date_col: str, value_cols: list[str], recent_periods: int = 3) -> list[str]:
    """Compares the most recent `recent_periods` points against the ones before them."""
    insights = []
    df = df.sort_values(date_col)

    for col in value_cols:
        if col not in df.columns:
            continue
        change = _pct_change(df[col], recent_periods)
        if change is None:
            continue
        direction = "increased" if change > 0 else "decreased" if change < 0 else "stayed flat"
        insights.append(f"{col.replace('_', ' ').title()} has {direction} by {abs(change)}% over the last {recent_periods} periods.")

    return insights


def generate_expense_insight(df: pd.DataFrame) -> str | None:
    expense_cols = [c for c in ["cost", "operating_expenses", "marketing_expenses"] if c in df.columns]
    if not expense_cols:
        return None
    totals = df[expense_cols].sum().sort_values(ascending=False)
    largest = totals.index[0]
    share = round(totals.iloc[0] / totals.sum() * 100, 1) if totals.sum() else 0
    return f"'{largest.replace('_', ' ').title()}' is the largest expense category, accounting for {share}% of total tracked expenses."


def generate_profit_insight(df: pd.DataFrame) -> str | None:
    if "profit" not in df.columns:
        return None
    recent_change = _pct_change(df["profit"], min(3, len(df) - 1)) if len(df) > 1 else None
    avg_margin = round(df["profit_margin"].mean(), 2) if "profit_margin" in df.columns else None

    parts = []
    if recent_change is not None:
        trend = "declined" if recent_change < 0 else "improved"
        parts.append(f"Profit has {trend} by {abs(recent_change)}% recently")
    if avg_margin is not None:
        parts.append(f"average profit margin is {avg_margin}%")

    return ". ".join(p.capitalize() for p in parts) + "." if parts else None


def generate_data_quality_insight(profile_summary: dict) -> str:
    missing_pct = profile_summary.get("total_missing_pct", 0)
    dup_rows = profile_summary.get("duplicate_rows", 0)

    if missing_pct == 0 and dup_rows == 0:
        return "The dataset is clean: no missing values or duplicate rows were detected."

    parts = []
    if missing_pct > 0:
        parts.append(f"{missing_pct}% of all cells were missing before cleaning")
    if dup_rows > 0:
        parts.append(f"{dup_rows} duplicate rows were found and removed")
    return "; ".join(parts).capitalize() + "."


def generate_risk_flags(df: pd.DataFrame) -> list[str]:
    flags = []
    if "profit_margin" in df.columns:
        recent_margin = df["profit_margin"].tail(3).mean()
        if pd.notna(recent_margin) and recent_margin < 5:
            flags.append(f"Recent average profit margin ({recent_margin:.1f}%) is low -- consider reviewing pricing or cost structure.")
    if "expense_growth" in df.columns and "revenue_growth" in df.columns:
        exp_growth = df["expense_growth"].tail(3).mean()
        rev_growth = df["revenue_growth"].tail(3).mean()
        if pd.notna(exp_growth) and pd.notna(rev_growth) and exp_growth > rev_growth:
            flags.append("Expenses are growing faster than revenue in the recent period -- a potential margin risk.")
    return flags


def build_full_insight_summary(
    df: pd.DataFrame, date_col: str, profile_summary: dict
) -> dict:
    """Aggregates all rule-based insights into one structured dict, ready for
    either direct display or as input to ai_engine.py."""
    value_cols = [c for c in ["revenue", "cost", "profit", "total_expenses"] if c in df.columns]

    return {
        "data_quality": generate_data_quality_insight(profile_summary),
        "trends": generate_trend_insights(df, date_col, value_cols),
        "expense": generate_expense_insight(df),
        "profit": generate_profit_insight(df),
        "risks": generate_risk_flags(df),
    }
