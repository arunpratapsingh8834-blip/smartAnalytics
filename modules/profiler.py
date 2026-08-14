"""
profiler.py
------------
Pure-pandas dataset profiling. No Streamlit calls here so it can be unit
tested and reused (e.g. by ai_engine.py, exporter.py) without a UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd
import numpy as np

MISSING_TOKENS = {"", "na", "n/a", "n.a.", "null", "none", "-", "--", "?", "nan"}

# Column name synonyms -> canonical name. Used only to SUGGEST, never to
# silently rename (see profiler.suggest_column_mapping).
COLUMN_SYNONYMS: dict[str, list[str]] = {
    "date": ["date", "order_date", "transaction_date", "invoice_date", "day"],
    "revenue": ["revenue", "sales", "income", "turnover", "total_sales", "amount"],
    "cost": ["cost", "cogs", "cost_of_goods_sold", "unit_cost"],
    "operating_expenses": ["operating_expenses", "opex", "op_ex"],
    "marketing_expenses": ["marketing_expenses", "marketing_spend", "advertising_cost", "ad_spend"],
    "quantity": ["quantity", "units_sold", "qty", "units"],
    "product": ["product", "item", "sku", "product_name"],
    "category": ["category", "product_category", "segment"],
    "region": ["region", "state", "area", "zone"],
    "customer": ["customer", "customer_name", "client"],
}


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    is_constant: bool
    is_high_cardinality: bool
    is_potential_id: bool
    inferred_type: str  # "numeric" | "categorical" | "date" | "text"


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    duplicate_rows: int
    total_missing_pct: float
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    high_cardinality_columns: list[str] = field(default_factory=list)
    potential_id_columns: list[str] = field(default_factory=list)
    column_profiles: dict[str, ColumnProfile] = field(default_factory=dict)


def _looks_like_date(series: pd.Series, sample_size: int = 50) -> bool:
    """
    Heuristic: try parsing a sample; if most parse successfully, treat as a date column.
    Guards against false positives where plain numeric strings (e.g. '7516.54') get
    misread by pandas as years/timestamps -- a real date string should contain date
    separators ('-', '/', '.') between multiple numeric groups, or month names, or
    be a clean 8-digit YYYYMMDD code.
    """
    if series.dtype == "datetime64[ns]":
        return True
    sample = series.dropna().astype(str).str.strip().head(sample_size)
    if sample.empty:
        return False

    date_like_pattern = sample.str.match(
        r"^\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}$"          # 2024-01-15, 15/01/2024, etc.
        r"|^\d{8}$"                                       # 20240115
        r"|^\d{1,2}[-\s][A-Za-z]{3,9}[-\s]\d{2,4}$"       # 15-Jan-2024
        r"|^[A-Za-z]{3,9}\s\d{1,2},?\s\d{2,4}$"           # January 15, 2024
    )
    if date_like_pattern.mean() < 0.6:
        return False

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def _looks_numeric_as_text(series: pd.Series, sample_size: int = 50) -> bool:
    """Detects numeric values that were loaded as strings, e.g. '1,200' or '$45.00'."""
    if pd.api.types.is_numeric_dtype(series):
        return False
    sample = series.dropna().astype(str).head(sample_size)
    if sample.empty:
        return False
    cleaned = sample.str.replace(r"[,$%\s]", "", regex=True)
    numeric_like = pd.to_numeric(cleaned, errors="coerce")
    return numeric_like.notna().mean() > 0.8


def normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Replaces common string placeholders for missingness (e.g. 'N/A', '-') with actual NaN."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda v: np.nan if isinstance(v, str) and v.strip().lower() in MISSING_TOKENS else v
        )
    return df


def profile_dataset(raw_df: pd.DataFrame) -> DatasetProfile:
    """Builds a full DatasetProfile. Does not mutate raw_df."""
    df = normalize_missing_tokens(raw_df)
    n_rows, n_cols = df.shape

    profile = DatasetProfile(
        rows=n_rows,
        columns=n_cols,
        duplicate_rows=int(df.duplicated().sum()),
        total_missing_pct=round(float(df.isna().sum().sum()) / (n_rows * n_cols) * 100, 2) if n_rows and n_cols else 0.0,
    )

    for col in df.columns:
        series = df[col]
        missing_count = int(series.isna().sum())
        missing_pct = round(missing_count / n_rows * 100, 2) if n_rows else 0.0
        unique_count = int(series.nunique(dropna=True))
        is_constant = unique_count <= 1
        is_high_cardinality = n_rows > 0 and unique_count / n_rows > 0.9 and not pd.api.types.is_numeric_dtype(series)
        is_potential_id = bool(
            unique_count == n_rows and n_rows > 0 and ("id" in col.lower() or is_high_cardinality)
        )

        if _looks_like_date(series):
            inferred = "date"
            profile.date_columns.append(col)
        elif pd.api.types.is_numeric_dtype(series) or _looks_numeric_as_text(series):
            inferred = "numeric"
            profile.numeric_columns.append(col)
        elif unique_count <= max(20, int(n_rows * 0.05)):
            inferred = "categorical"
            profile.categorical_columns.append(col)
        else:
            inferred = "text"
            profile.categorical_columns.append(col)

        if is_constant:
            profile.constant_columns.append(col)
        if is_high_cardinality:
            profile.high_cardinality_columns.append(col)
        if is_potential_id:
            profile.potential_id_columns.append(col)

        profile.column_profiles[col] = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            missing_count=missing_count,
            missing_pct=missing_pct,
            unique_count=unique_count,
            is_constant=is_constant,
            is_high_cardinality=is_high_cardinality,
            is_potential_id=is_potential_id,
            inferred_type=inferred,
        )

    return profile


def suggest_column_mapping(df: pd.DataFrame) -> dict[str, dict]:
    """
    Suggests canonical-name matches for ambiguous columns (e.g. 'turnover' -> 'revenue').
    Returns {actual_column_name: {"suggested": canonical_name, "confidence": float}}
    Never renames automatically -- caller/UI must confirm.
    """
    suggestions: dict[str, dict] = {}
    lower_cols = {c: c.lower().strip().replace(" ", "_") for c in df.columns}

    for actual_col, normalized in lower_cols.items():
        for canonical, synonyms in COLUMN_SYNONYMS.items():
            if normalized == canonical:
                continue  # already canonical, no suggestion needed
            if normalized in synonyms:
                confidence = 1.0 if normalized == synonyms[0] else 0.85
                suggestions[actual_col] = {"suggested": canonical, "confidence": confidence}
                break

    return suggestions


def summary_dict(profile: DatasetProfile) -> dict:
    """Small JSON-serializable summary, e.g. for the AI explanation layer."""
    return {
        "rows": profile.rows,
        "columns": profile.columns,
        "duplicate_rows": profile.duplicate_rows,
        "total_missing_pct": profile.total_missing_pct,
        "numeric_columns": profile.numeric_columns,
        "categorical_columns": profile.categorical_columns,
        "date_columns": profile.date_columns,
        "constant_columns": profile.constant_columns,
        "potential_id_columns": profile.potential_id_columns,
    }
