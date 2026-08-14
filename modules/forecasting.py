"""
forecasting.py
----------------
Individual forecasting model implementations, all exposed through one
common interface: fit on a training series, produce `periods` future
points with (yhat, yhat_lower, yhat_upper).

Every model function returns a pandas DataFrame with columns:
    ds, yhat, yhat_lower, yhat_upper

Models degrade gracefully: if a library (e.g. Prophet) isn't installed,
model_selection.py simply skips it rather than crashing the app.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


FREQ_MAP = {"daily": "D", "weekly": "W", "monthly": "MS", "quarterly": "QS", "yearly": "YS"}


def _future_dates(last_date: pd.Timestamp, periods: int, frequency: str) -> pd.DatetimeIndex:
    freq_code = FREQ_MAP.get(frequency, "D")
    return pd.date_range(start=last_date, periods=periods + 1, freq=freq_code)[1:]


# --------------------------------------------------------------- models ----

def naive_forecast(train: pd.DataFrame, periods: int, frequency: str) -> pd.DataFrame:
    """Repeats the last observed value. Always-available baseline."""
    last_value = train["y"].iloc[-1]
    last_date = train["ds"].iloc[-1]
    std = train["y"].std() if len(train) > 1 else 0.0

    dates = _future_dates(last_date, periods, frequency)
    return pd.DataFrame({
        "ds": dates,
        "yhat": [last_value] * periods,
        "yhat_lower": [last_value - 1.28 * std] * periods,
        "yhat_upper": [last_value + 1.28 * std] * periods,
    })


def seasonal_naive_forecast(train: pd.DataFrame, periods: int, frequency: str, season_length: int = 7) -> pd.DataFrame:
    """Repeats the value from one season ago (e.g. same weekday last week for daily data)."""
    if len(train) < season_length * 2:
        return naive_forecast(train, periods, frequency)

    last_season = train["y"].iloc[-season_length:].values
    std = train["y"].std()
    last_date = train["ds"].iloc[-1]
    dates = _future_dates(last_date, periods, frequency)

    values = [last_season[i % season_length] for i in range(periods)]
    return pd.DataFrame({
        "ds": dates,
        "yhat": values,
        "yhat_lower": [v - 1.28 * std for v in values],
        "yhat_upper": [v + 1.28 * std for v in values],
    })


def exponential_smoothing_forecast(train: pd.DataFrame, periods: int, frequency: str) -> pd.DataFrame | None:
    """Holt-Winters exponential smoothing, with damped trend and optional seasonality."""
    if not STATSMODELS_AVAILABLE or len(train) < 4:
        return None
    try:
        seasonal_periods = {"daily": 7, "weekly": 52, "monthly": 12}.get(frequency)
        use_seasonal = seasonal_periods is not None and len(train) >= 2 * seasonal_periods

        model = ExponentialSmoothing(
            train["y"].values,
            trend="add",
            damped_trend=True,
            seasonal="add" if use_seasonal else None,
            seasonal_periods=seasonal_periods if use_seasonal else None,
        ).fit()

        forecast = model.forecast(periods)
        resid_std = np.std(model.resid) if hasattr(model, "resid") else train["y"].std()
        dates = _future_dates(train["ds"].iloc[-1], periods, frequency)

        return pd.DataFrame({
            "ds": dates,
            "yhat": forecast,
            "yhat_lower": forecast - 1.28 * resid_std,
            "yhat_upper": forecast + 1.28 * resid_std,
        })
    except Exception:
        return None


def arima_forecast(train: pd.DataFrame, periods: int, frequency: str) -> pd.DataFrame | None:
    if not STATSMODELS_AVAILABLE or len(train) < 10:
        return None
    try:
        model = ARIMA(train["y"].values, order=(1, 1, 1)).fit()
        forecast_result = model.get_forecast(periods)
        mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.20)  # ~80% interval
        dates = _future_dates(train["ds"].iloc[-1], periods, frequency)

        return pd.DataFrame({
            "ds": dates,
            "yhat": mean,
            "yhat_lower": conf_int[:, 0],
            "yhat_upper": conf_int[:, 1],
        })
    except Exception:
        return None


def prophet_forecast(train: pd.DataFrame, periods: int, frequency: str) -> pd.DataFrame | None:
    if not PROPHET_AVAILABLE or len(train) < 10:
        return None
    try:
        model = Prophet(interval_width=0.8)
        model.fit(train[["ds", "y"]])
        freq_code = FREQ_MAP.get(frequency, "D")
        future = model.make_future_dataframe(periods=periods, freq=freq_code)
        forecast = model.predict(future)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
        return result.reset_index(drop=True)
    except Exception:
        return None


def random_forest_forecast(
    train: pd.DataFrame, periods: int, frequency: str, feature_builder
) -> pd.DataFrame | None:
    """
    Tree-based forecast using lag/rolling/time features (feature_builder from
    feature_engineering.py). Forecasts recursively: predict one step, feed it
    back in as the newest lag, predict the next step, etc.
    """
    if not SKLEARN_AVAILABLE or len(train) < 40:
        return None
    try:
        working = train.rename(columns={"y": "target"}).copy()
        feature_df = feature_builder(working, "ds", "target")
        if len(feature_df) < 20:
            return None

        feature_cols = [c for c in feature_df.columns if c not in ("ds", "target")]
        X_train = feature_df[feature_cols]
        y_train = feature_df["target"]

        model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
        model.fit(X_train, y_train)

        history = working.copy()
        preds = []
        for _ in range(periods):
            feat = feature_builder(history, "ds", "target")
            if feat.empty:
                break
            latest_features = feat[feature_cols].iloc[[-1]]
            next_value = float(model.predict(latest_features)[0])
            next_date = _future_dates(history["ds"].iloc[-1], 1, frequency)[0]
            preds.append((next_date, next_value))
            history = pd.concat(
                [history, pd.DataFrame({"ds": [next_date], "target": [next_value]})],
                ignore_index=True,
            )

        if not preds:
            return None

        resid_std = float(np.std(y_train - model.predict(X_train)))
        dates, values = zip(*preds)
        return pd.DataFrame({
            "ds": dates,
            "yhat": values,
            "yhat_lower": [v - 1.28 * resid_std for v in values],
            "yhat_upper": [v + 1.28 * resid_std for v in values],
        })
    except Exception:
        return None


MODEL_REGISTRY = {
    "Naive": naive_forecast,
    "Seasonal Naive": seasonal_naive_forecast,
    "Exponential Smoothing": exponential_smoothing_forecast,
    "ARIMA": arima_forecast,
    "Prophet": prophet_forecast,
    "Random Forest": random_forest_forecast,  # requires feature_builder kwarg
}
