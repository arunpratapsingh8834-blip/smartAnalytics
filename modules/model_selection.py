"""
model_selection.py
---------------------
Runs every applicable forecasting model through backtesting, ranks them,
recommends the best one, and (only after selection) produces the actual
future forecast on the FULL history using the winning model.

Nothing here hardcodes "Prophet is best" or similar -- the ranking is
always computed from backtest metrics on the user's own data.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from modules import forecasting, backtesting
from modules.feature_engineering import build_ml_feature_matrix


@dataclass
class ModelComparisonRow:
    model_name: str
    mae: float
    rmse: float
    mape: float | None
    rank: int


@dataclass
class ForecastRunResult:
    recommended_model: str
    comparison_table: list[ModelComparisonRow]
    forecast: pd.DataFrame  # ds, yhat, yhat_lower, yhat_upper
    warnings: list[str]


def _applicable_models(n_points: int) -> dict:
    """Only include models that make sense given how much history is available."""
    models = {
        "Naive": forecasting.naive_forecast,
        "Seasonal Naive": forecasting.seasonal_naive_forecast,
    }
    if n_points >= 10:
        models["Exponential Smoothing"] = forecasting.exponential_smoothing_forecast
        models["ARIMA"] = forecasting.arima_forecast
        models["Prophet"] = forecasting.prophet_forecast
    if n_points >= 40:
        models["Random Forest"] = forecasting.random_forest_forecast
    return models


def compare_models(df: pd.DataFrame, frequency: str) -> tuple[list[ModelComparisonRow], dict]:
    """
    df must have columns ['ds', 'y'] already sorted chronologically.
    Returns (ranked comparison rows, {model_name: model_fn}) for models that
    successfully produced a backtest result.
    """
    models = _applicable_models(len(df))
    results = []
    working_fns = {}

    for name, fn in models.items():
        kwargs = {"feature_builder": build_ml_feature_matrix} if name == "Random Forest" else {}
        result = backtesting.backtest_model(fn, df, frequency, **kwargs)
        if result is not None:
            results.append((name, result))
            working_fns[name] = (fn, kwargs)

    if not results:
        return [], {}

    # Rank by MAE primarily (works even when MAPE is undefined due to zero actuals)
    results.sort(key=lambda r: r[1].mae)

    comparison = [
        ModelComparisonRow(
            model_name=name, mae=round(res.mae, 3), rmse=round(res.rmse, 3),
            mape=round(res.mape, 2) if res.mape is not None else None, rank=i + 1,
        )
        for i, (name, res) in enumerate(results)
    ]
    return comparison, working_fns


def run_forecast(
    df: pd.DataFrame, frequency: str, periods: int, manual_model: str | None = None
) -> ForecastRunResult:
    """
    Full pipeline: backtest all applicable models, pick the winner (or honor
    manual_model override), then fit that model on the FULL history to produce
    the actual forward-looking forecast for `periods` future points.
    """
    warnings: list[str] = []
    comparison, working_fns = compare_models(df, frequency)

    if not comparison:
        warnings.append("No model could be validated on this data. Falling back to a naive forecast.")
        forecast = forecasting.naive_forecast(df, periods, frequency)
        return ForecastRunResult(
            recommended_model="Naive", comparison_table=[], forecast=forecast, warnings=warnings
        )

    recommended = manual_model if manual_model in working_fns else comparison[0].model_name
    if manual_model and manual_model not in working_fns:
        warnings.append(f"'{manual_model}' could not be validated on this dataset; using {recommended} instead.")

    fn, kwargs = working_fns[recommended]
    forecast = fn(df, periods, frequency, **kwargs)

    if forecast is None:
        warnings.append(f"{recommended} failed on full-history refit; falling back to Naive.")
        recommended = "Naive"
        forecast = forecasting.naive_forecast(df, periods, frequency)

    if len(df) < 30:
        warnings.append(
            "Fewer than 30 historical points were available -- treat this forecast as indicative only."
        )

    return ForecastRunResult(
        recommended_model=recommended,
        comparison_table=comparison,
        forecast=forecast,
        warnings=warnings,
    )
