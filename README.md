# Smart Analytics & AI-Powered Sales Forecasting Platform

A college Major Project: upload a sales CSV, get automatic data profiling,
guided data cleaning, transformation, multi-model forecasting with
backtesting, profit-driver analysis, and an optional AI explanation layer.

## 1. Folder structure

```
SmartAnalytics/
├── app.py                     # Streamlit entry point (UI only)
├── requirements.txt
├── README.md
├── .env.example
│
├── modules/
│   ├── auth.py                # register/login/logout, bcrypt hashing
│   ├── database.py             # SQLite connection + schema + CRUD
│   ├── profiler.py             # dataset profiling
│   ├── cleaner.py              # missing value engine + cleaning
│   ├── transformer.py          # type conversion, date handling, aggregation
│   ├── feature_engineering.py  # time features, profit/margin features
│   ├── forecasting.py          # individual forecasting models
│   ├── backtesting.py          # time-series train/validation split + metrics
│   ├── model_selection.py      # runs all models, ranks them, recommends one
│   ├── insights.py             # rule-based business insight generation
│   ├── ai_engine.py            # optional Anthropic API explanation layer
│   └── exporter.py             # CSV / Excel / PDF / ZIP export
│
├── database/
│   └── app.db                  # created automatically on first run
│
├── assets/
│   └── style.css
│
├── reports/                    # generated PDF reports land here
│
├── utils/
│   └── sample_data.py          # generates a messy sample CSV for testing
│
└── tests/
    ├── test_cleaner.py
    ├── test_transformer.py
    └── test_forecasting.py
```

## 2. Setup

```bash
cd SmartAnalytics
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then edit .env if you want AI insights
```

Prophet can be slow to install on some systems. If it fails, the app still
works — `forecasting.py` automatically skips Prophet if it's not importable
and just uses the other models.

## 3. Run

```bash
streamlit run app.py
```

First run auto-creates `database/app.db` with all required tables.

## 4. Generate a test CSV

```bash
python utils/sample_data.py
```

This creates `sample_sales_data.csv` with intentional missing values,
duplicates, inconsistent capitalization, numbers stored as text, and bad
date formats — good for demonstrating the cleaning engine in your viva.

## 5. Run tests

```bash
pytest tests/
```

## 6. Architecture notes (for viva)

- **Separation of concerns**: `app.py` only handles UI/session state. All
  logic (cleaning, forecasting, stats) lives in `modules/`, so it can be
  unit-tested without Streamlit running.
- **RAW → CLEANED → TRANSFORMED → FORECASTED**: the raw upload is never
  mutated in place; each stage returns a new DataFrame stored separately in
  `st.session_state`, so "before vs after" comparisons are always possible.
- **Model selection is data-driven**: `model_selection.py` backtests every
  candidate model on held-out history and ranks by MAE/RMSE/MAPE — nothing
  is hardcoded as "the best model."
- **AI layer is explanation-only**: `ai_engine.py` never computes metrics or
  forecasts itself. It receives a small JSON summary of numbers already
  computed by pandas/statsmodels/sklearn and only turns them into
  natural-language text. If no API key is set, it falls back to a
  template-based summary so the app still works offline.
