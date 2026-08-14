"""
cleaner.py
-----------
Missing-value recommendation engine + actual cleaning operations
(duplicates, whitespace, column-name normalization, numeric/date coercion,
category-spelling suggestions via fuzzy matching).

Design rule: functions that RECOMMEND return plain data (dict/dataclass);
functions that APPLY changes always take the recommendation/choice as an
explicit argument and return a new DataFrame. Nothing here silently
decides on the user's behalf.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process

from modules.profiler import normalize_missing_tokens, _looks_numeric_as_text


@dataclass
class MissingValueRecommendation:
    column: str
    missing_pct: float
    inferred_type: str  # numeric / categorical / date
    recommended_method: str
    reason: str
    options: list[str]


NUMERIC_OPTIONS = ["Median", "Mean", "Forward Fill", "Backward Fill", "Interpolation", "Drop Rows", "Custom Value"]
CATEGORICAL_OPTIONS = ["Mode", "Unknown", "Forward Fill", "Backward Fill", "Drop Rows"]
DATE_OPTIONS = ["Drop Rows", "Forward Fill", "Flag Only"]


# ------------------------------------------------------- recommendation ----

def recommend_missing_value_treatment(
    df: pd.DataFrame, column: str, inferred_type: str
) -> MissingValueRecommendation:
    series = df[column]
    n = len(series)
    missing_count = int(series.isna().sum())
    missing_pct = round(missing_count / n * 100, 2) if n else 0.0

    if inferred_type == "date":
        return MissingValueRecommendation(
            column=column,
            missing_pct=missing_pct,
            inferred_type=inferred_type,
            recommended_method="Flag Only",
            reason="Dates should not be guessed. Rows are flagged so you can decide manually.",
            options=DATE_OPTIONS,
        )

    if inferred_type == "numeric":
        numeric_series = pd.to_numeric(
            series.astype(str).str.replace(r"[,$%\s]", "", regex=True), errors="coerce"
        )
        skew = numeric_series.dropna().skew() if numeric_series.dropna().shape[0] > 2 else 0.0
        is_skewed = abs(skew) > 1.0

        if missing_pct > 40:
            method, reason = "Drop Rows", f"{missing_pct}% missing is too high to impute reliably."
        elif is_skewed:
            method, reason = "Median", f"Median imputation recommended because this numeric column contains {missing_pct}% missing values and is skewed (skew={skew:.2f})."
        elif missing_pct < 5:
            method, reason = "Mean", f"Mean imputation recommended: only {missing_pct}% missing and the distribution is roughly symmetric."
        else:
            method, reason = "Interpolation", f"Interpolation recommended: {missing_pct}% missing values, moderate gap size suitable for linear interpolation."

        return MissingValueRecommendation(
            column=column, missing_pct=missing_pct, inferred_type=inferred_type,
            recommended_method=method, reason=reason, options=NUMERIC_OPTIONS,
        )

    # categorical / text
    if missing_pct > 50:
        method, reason = "Drop Rows", f"{missing_pct}% missing is too high; the column carries little information."
    elif missing_pct < 5:
        method, reason = "Mode", f"Mode (most frequent value) recommended: only {missing_pct}% missing."
    else:
        method, reason = "Unknown", f"Labeling as 'Unknown' recommended to avoid biasing category frequencies with {missing_pct}% missing."

    return MissingValueRecommendation(
        column=column, missing_pct=missing_pct, inferred_type=inferred_type,
        recommended_method=method, reason=reason, options=CATEGORICAL_OPTIONS,
    )


def recommend_all(df: pd.DataFrame, column_types: dict[str, str]) -> list[MissingValueRecommendation]:
    """column_types: {col_name: 'numeric'|'categorical'|'date'} typically from profiler.DatasetProfile"""
    recs = []
    for col, inferred_type in column_types.items():
        if df[col].isna().sum() > 0:
            recs.append(recommend_missing_value_treatment(df, col, inferred_type))
    return recs


# ------------------------------------------------------------- applying ----

def apply_missing_value_method(df: pd.DataFrame, column: str, method: str, custom_value=None) -> pd.DataFrame:
    df = df.copy()
    series = df[column]

    if method == "Drop Rows":
        df = df[series.notna()].reset_index(drop=True)
    elif method == "Mean":
        numeric_series = pd.to_numeric(series, errors="coerce")
        df[column] = numeric_series.fillna(numeric_series.mean())
    elif method == "Median":
        numeric_series = pd.to_numeric(series, errors="coerce")
        df[column] = numeric_series.fillna(numeric_series.median())
    elif method == "Mode":
        mode_val = series.mode(dropna=True)
        df[column] = series.fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")
    elif method == "Unknown":
        df[column] = series.fillna("Unknown")
    elif method == "Forward Fill":
        df[column] = series.ffill()
    elif method == "Backward Fill":
        df[column] = series.bfill()
    elif method == "Interpolation":
        numeric_series = pd.to_numeric(series, errors="coerce")
        df[column] = numeric_series.interpolate(method="linear", limit_direction="both")
    elif method == "Custom Value":
        df[column] = series.fillna(custom_value)
    elif method == "Flag Only":
        df[f"{column}_missing_flag"] = series.isna()
    # "Keep Original" / unrecognized -> no-op

    return df


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    return cleaned, before - len(cleaned)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns
    ]
    return df


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Converts strings like '1,200', '$45.00', '12%' into real floats."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        if _looks_numeric_as_text(df[col]) or pd.api.types.is_numeric_dtype(df[col]):
            cleaned = df[col].astype(str).str.replace(r"[,$%\s]", "", regex=True)
            df[col] = pd.to_numeric(cleaned, errors="coerce")
    return df


def coerce_date_column(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, int]:
    """Returns (df with parsed dates, count of rows that failed to parse)."""
    df = df.copy()
    parsed = pd.to_datetime(df[column], errors="coerce", format="mixed")
    invalid_count = int(parsed.isna().sum() - df[column].isna().sum())
    df[column] = parsed
    return df, max(invalid_count, 0)


def full_clean_pipeline(
    raw_df: pd.DataFrame,
    missing_value_choices: dict[str, dict],
    date_column: str | None = None,
    numeric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper that runs the standard cleaning sequence.
    missing_value_choices: {column: {"method": str, "custom_value": Any}}
    """
    def _norm(name: str) -> str:
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    df = normalize_missing_tokens(raw_df)
    df = strip_whitespace(df)
    df = normalize_column_names(df)
    df, _ = remove_duplicates(df)

    if numeric_columns:
        numeric_columns = [_norm(c) for c in numeric_columns]
        df = coerce_numeric_columns(df, numeric_columns)

    if date_column:
        date_column = _norm(date_column)
        if date_column in df.columns:
            df, _ = coerce_date_column(df, date_column)

    for raw_column, choice in missing_value_choices.items():
        column = _norm(raw_column)
        if column in df.columns:
            df = apply_missing_value_method(
                df, column, choice.get("method", "Leave Missing"), choice.get("custom_value")
            )

    return df


# ------------------------------------------------ category standardization

def detect_category_inconsistencies(
    df: pd.DataFrame, column: str, similarity_threshold: int = 80
) -> list[dict]:
    """
    Fuzzy-matches distinct values against each other and flags likely typos/variants.
    NEVER auto-corrects -- only returns suggestions for the UI to present.
    """
    if column not in df.columns:
        return []

    values = df[column].dropna().astype(str).str.strip()
    counts = values.value_counts()
    unique_values = counts.index.tolist()

    if len(unique_values) < 2:
        return []

    flagged = []
    checked = set()

    for value in unique_values:
        key = value.lower()
        if key in checked:
            continue
        # Compare against more frequent values only, so the "canonical" form is the common one
        candidates = [v for v in unique_values if v != value]
        matches = process.extract(
            value, candidates, scorer=fuzz.ratio, limit=3, score_cutoff=similarity_threshold
        )
        # Only flag if the value is markedly less frequent than its match (likely the typo)
        real_matches = [
            (m, score) for m, score, _ in matches if counts[m] >= counts[value] and m.lower() != key
        ]
        if real_matches:
            flagged.append(
                {
                    "value": value,
                    "count": int(counts[value]),
                    "possible_matches": [
                        {"match": m, "confidence": round(score, 1)} for m, score in real_matches
                    ],
                }
            )
        checked.add(key)

    return flagged


def apply_category_correction(df: pd.DataFrame, column: str, mapping: dict[str, str]) -> pd.DataFrame:
    """mapping: {original_value: corrected_value}, applied only for entries the user confirmed."""
    df = df.copy()
    df[column] = df[column].replace(mapping)
    return df
