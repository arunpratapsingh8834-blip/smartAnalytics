"""
backtesting.py
----------------
Time-series train/validation split and evaluation metrics.
Never shuffle time-series data -- always split chronologically.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    model_name: str
    mae: float
    rmse: float
    mape: float | None  # None if any actuals were zero (MAPE undefined)
    n_validation_points: int


def train_validation_split(df: pd.DataFrame, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split -- the last `validation_fraction` of rows become the validation set."""
    n = len(df)
    split_idx = max(1, int(n * (1 - validation_fraction)))
    return df.iloc[:split_idx].reset_index(drop=True), df.iloc[split_idx:].reset_index(drop=True)


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))

    if np.any(actual == 0):
        mape = None  # undefined / would be misleading
    else:
        mape = float(np.mean(np.abs((actual - predicted) / actual)) * 100)

    return {"mae": mae, "rmse": rmse, "mape": mape}


def backtest_model(
    model_fn, full_df: pd.DataFrame, frequency: str, validation_fraction: float = 0.2, **model_kwargs
) -> BacktestResult | None:
    """
    Trains model_fn on the earlier portion of full_df and evaluates against the
    held-out later portion (chronological -- this IS the "historical backtesting"
    referenced throughout the spec).
    """
    train, validation = train_validation_split(full_df, validation_fraction)
    if len(validation) == 0 or len(train) < 5:
        return None

    forecast = model_fn(train, len(validation), frequency, **model_kwargs)
    if forecast is None or len(forecast) < len(validation):
        return None

    predicted = forecast["yhat"].values[: len(validation)]
    actual = validation["y"].values

    metrics = compute_metrics(actual, predicted)
    return BacktestResult(
        model_name=model_fn.__name__,
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        mape=metrics["mape"],
        n_validation_points=len(validation),
    )


def rolling_backtest(
    model_fn, full_df: pd.DataFrame, frequency: str, n_splits: int = 3, min_train_size: int = 10, **model_kwargs
) -> list[BacktestResult]:
    """
    Expanding-window backtest: repeatedly grows the training set and validates on the
    next block, giving a more robust estimate than a single train/validation split.
    """
    n = len(full_df)
    if n < min_train_size + n_splits:
        return []

    fold_size = max(1, (n - min_train_size) // n_splits)
    results = []

    for i in range(n_splits):
        train_end = min_train_size + i * fold_size
        val_end = min(train_end + fold_size, n)
        if val_end <= train_end:
            continue

        train = full_df.iloc[:train_end].reset_index(drop=True)
        validation = full_df.iloc[train_end:val_end].reset_index(drop=True)
        if len(validation) == 0:
            continue

        forecast = model_fn(train, len(validation), frequency, **model_kwargs)
        if forecast is None or len(forecast) < len(validation):
            continue

        metrics = compute_metrics(validation["y"].values, forecast["yhat"].values[: len(validation)])
        results.append(
            BacktestResult(
                model_name=model_fn.__name__,
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                mape=metrics["mape"],
                n_validation_points=len(validation),
            )
        )

    return results
