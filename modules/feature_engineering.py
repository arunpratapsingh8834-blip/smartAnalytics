"""
feature_engineering.py
------------------------
Time-based feature generation and lag features used by the
tree-based forecasting model in forecasting.py.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Adds year/month/week/day/day_of_week/quarter columns derived from date_col."""
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["week"] = dt.dt.isocalendar().week.astype(int)
    df["day"] = dt.dt.day
    df["day_of_week"] = dt.dt.dayofweek
    df["quarter"] = dt.dt.quarter
    return df


def add_lag_features(df: pd.DataFrame, target_col: str, lags: list[int] = (1, 7, 30)) -> pd.DataFrame:
    """Adds lag_N columns (previous values of the target) for use as regressors."""
    df = df.copy()
    for lag in lags:
        if lag < len(df):
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str, windows: list[int] = (7, 30)) -> pd.DataFrame:
    """Adds rolling mean/std columns, useful signals for tree-based models."""
    df = df.copy()
    for w in windows:
        if w < len(df):
            df[f"{target_col}_roll_mean_{w}"] = df[target_col].rolling(w).mean()
            df[f"{target_col}_roll_std_{w}"] = df[target_col].rolling(w).std()
    return df


def build_ml_feature_matrix(
    df: pd.DataFrame, date_col: str, target_col: str
) -> pd.DataFrame:
    """Combines time + lag + rolling features and drops rows with NaNs created by shifting."""
    out = add_time_features(df, date_col)
    out = add_lag_features(out, target_col)
    out = add_rolling_features(out, target_col)
    return out.dropna().reset_index(drop=True)
