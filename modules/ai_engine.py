"""
ai_engine.py
--------------
Optional AI explanation layer. IMPORTANT: this module never computes any
statistics, forecasts, or metrics itself -- it only turns structured
summaries (already computed by profiler/insights/model_selection) into
natural-language text.

If ANTHROPIC_API_KEY is not set, or the API call fails for any reason,
every function falls back to a template-based summary so the app keeps
working fully offline.
"""

from __future__ import annotations

import os
import json

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False


def _get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not ANTHROPIC_SDK_AVAILABLE:
        return None
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception:
        return None


def _call_claude(system_prompt: str, user_content: str, max_tokens: int = 600) -> str | None:
    client = _get_client()
    if client is None:
        return None
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return "".join(block.text for block in response.content if hasattr(block, "text")).strip()
    except Exception:
        return None


SYSTEM_PROMPT = (
    "You are a business data analyst assistant embedded in a student's college "
    "analytics project. You will be given structured JSON summaries of data "
    "quality, cleaning actions, forecasts, and model comparisons that have "
    "ALREADY been computed by Python/pandas/statsmodels/sklearn. Your only job "
    "is to explain these numbers in clear, plain business language. Do not "
    "invent numbers that are not present in the JSON. Never claim a forecast "
    "is guaranteed to be accurate -- always mention that reliability depends "
    "on data quality, seasonality, volatility and forecast horizon. Keep "
    "responses concise (under 200 words) and use simple formatting."
)


def explain_data_quality(profile_summary: dict, insight_summary: dict) -> str:
    payload = json.dumps({"profile": profile_summary, "insights": insight_summary})
    result = _call_claude(SYSTEM_PROMPT, f"Explain this dataset's data-quality situation:\n{payload}")
    if result:
        return result
    # fallback: template based on insights.py output, which is already text
    return insight_summary.get("data_quality", "No major data quality issues detected.")


def explain_cleaning_recommendation(column: str, method: str, reason: str) -> str:
    result = _call_claude(
        SYSTEM_PROMPT,
        f"In one short paragraph, explain to a student why '{method}' was recommended "
        f"for missing values in column '{column}'. Technical reason: {reason}",
    )
    return result or reason


def explain_forecast(comparison_table: list[dict], recommended_model: str, forecast_summary: dict) -> str:
    payload = json.dumps({
        "comparison_table": comparison_table,
        "recommended_model": recommended_model,
        "forecast_summary": forecast_summary,
    })
    result = _call_claude(SYSTEM_PROMPT, f"Explain this forecast result to a business user:\n{payload}")
    if result:
        return result

    best = comparison_table[0] if comparison_table else {}
    return (
        f"Based on backtesting against historical data, '{recommended_model}' had the lowest "
        f"error (MAE={best.get('mae', 'n/a')}) among the models tested. Forecast reliability "
        f"depends on historical data quality, seasonality, volatility and forecast horizon -- "
        f"treat this as a data-informed estimate, not a guarantee."
    )


def generate_executive_summary(profile_summary: dict, insight_summary: dict, forecast_summary: dict) -> str:
    payload = json.dumps({
        "profile": profile_summary,
        "insights": insight_summary,
        "forecast": forecast_summary,
    })
    result = _call_claude(
        SYSTEM_PROMPT,
        f"Write a short executive summary (bullet points) covering data quality, "
        f"key business trends, and the forecast outlook:\n{payload}",
        max_tokens=500,
    )
    if result:
        return result

    lines = [f"- {insight_summary.get('data_quality', '')}"]
    lines += [f"- {t}" for t in insight_summary.get("trends", [])]
    if insight_summary.get("profit"):
        lines.append(f"- {insight_summary['profit']}")
    for risk in insight_summary.get("risks", []):
        lines.append(f"- Risk: {risk}")
    lines.append(
        f"- Forecast: recommended model is {forecast_summary.get('recommended_model', 'n/a')}. "
        f"Forecast reliability depends on data quality, seasonality, volatility and horizon."
    )
    return "\n".join(lines)


def is_ai_available() -> bool:
    return _get_client() is not None
