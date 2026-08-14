"""
app.py
-------
Streamlit entry point. UI ONLY -- all logic lives in modules/.
Run with: streamlit run app.py
"""

from __future__ import annotations

import os
import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from modules import database, auth, profiler, cleaner, transformer, insights, ai_engine, exporter
from modules.model_selection import run_forecast

load_dotenv()

st.set_page_config(
    page_title="Smart Analytics & AI Forecasting",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

database.init_db()

CSS_PATH = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------ session state

DEFAULT_STATE = {
    "logged_in": False,
    "user": None,
    "raw_data": None,
    "current_filename": None,
    "dataset_id": None,
    "profile": None,
    "column_types": {},
    "missing_choices": {},
    "cleaned_data": None,
    "transformed_data": None,
    "forecast_result": None,
    "date_col": None,
    "target_col": None,
}
for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ------------------------------------------------------------------ helpers

def metric_row(items: list[tuple[str, str]]):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        with col:
            st.markdown(f'<div class="metric-card"><b>{label}</b><br><span style="font-size:1.4rem">{value}</span></div>', unsafe_allow_html=True)


def require_login():
    if not st.session_state.logged_in:
        st.warning("Please log in from the sidebar to use the platform.")
        st.stop()
         
def normalize_col_name(name:str|None) -> str|None:
    if name is None:
        return None
    return name.strip().lower().replace(" ", "_").replace("-", "_")
    
       

# --------------------------------------------------------------- auth panel

def render_auth_sidebar():
    st.sidebar.title("\U0001F464 Account")

    if st.session_state.logged_in:
        st.sidebar.success(f"Logged in as **{st.session_state.user['username']}**")
        if st.sidebar.button("Logout"):
            auth.logout_user(st.session_state)
            st.session_state.logged_in = False
            st.rerun()
        return

    tab_login, tab_register = st.sidebar.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            try:
                user = auth.login_user(username, password)
                st.session_state.user = user
                st.session_state.logged_in = True
                st.rerun()
            except auth.AuthError as e:
                st.sidebar.error(str(e))

    with tab_register:
        with st.form("register_form"):
            r_username = st.text_input("Choose a username")
            r_email = st.text_input("Email")
            r_password = st.text_input("Password", type="password")
            r_confirm = st.text_input("Confirm password", type="password")
            r_submitted = st.form_submit_button("Register")
        if r_submitted:
            try:
                auth.register_user(r_username, r_email, r_password, r_confirm)
                st.sidebar.success("Account created! Please log in.")
            except auth.AuthError as e:
                st.sidebar.error(str(e))


# --------------------------------------------------------------------- Home

def render_home():
    st.title("\U0001F3E0 Smart Analytics & AI-Powered Sales Forecasting Platform")
    st.write(
        "Upload a sales CSV to get automatic data profiling, guided cleaning, "
        "multi-model forecasting with backtesting, and AI-generated business insights."
    )
    metric_row([
        ("Modules", "12"),
        ("Forecast Models", "Up to 6"),
        ("Export Formats", "CSV / Excel / PDF / ZIP"),
    ])
    st.info(
        "\u26A0\uFE0F Forecast reliability depends on historical data quality, seasonality, "
        "volatility and forecast horizon. No forecast is guaranteed to be accurate."
    )


# ------------------------------------------------------------------- Upload

def render_upload():
    st.header("\U0001F4C2 Upload Dataset")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"\u26A0\uFE0F Could not read this CSV: {e}")
            return

        if df.empty:
            st.error("\u26A0\uFE0F The uploaded file has no rows.")
            return

        st.session_state.raw_data = df
        st.session_state.current_filename = uploaded.name
        st.session_state.cleaned_data = None
        st.session_state.transformed_data = None
        st.session_state.forecast_result = None

        if st.session_state.logged_in:
            dataset_id = database.create_dataset(
                st.session_state.user["id"], uploaded.name, df.shape[0], df.shape[1]
            )
            st.session_state.dataset_id = dataset_id

        st.success(f"Loaded **{uploaded.name}** -- {df.shape[0]} rows x {df.shape[1]} columns.")
        st.dataframe(df.head(20), use_container_width=True)

    elif st.session_state.raw_data is not None:
        st.info(f"Currently loaded: **{st.session_state.current_filename}**")
        st.dataframe(st.session_state.raw_data.head(20), use_container_width=True)


# ------------------------------------------------------------------ Profile

def render_profile():
    st.header("\U0001F50E Data Profile")
    if st.session_state.raw_data is None:
        st.warning("Please upload a CSV first (Upload tab).")
        return

    profile = profiler.profile_dataset(st.session_state.raw_data)
    st.session_state.profile = profile
    st.session_state.column_types = {
        col: cp.inferred_type for col, cp in profile.column_profiles.items()
    }

    metric_row([
        ("Rows", f"{profile.rows:,}"),
        ("Columns", f"{profile.columns}"),
        ("Missing", f"{profile.total_missing_pct}%"),
        ("Duplicate Rows", f"{profile.duplicate_rows}"),
    ])

    st.subheader("Column Details")
    rows = [
        {
            "Column": cp.name,
            "Inferred Type": cp.inferred_type,
            "Dtype": cp.dtype,
            "Missing %": cp.missing_pct,
            "Unique": cp.unique_count,
            "Constant?": cp.is_constant,
            "Potential ID?": cp.is_potential_id,
        }
        for cp in profile.column_profiles.values()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    suggestions = profiler.suggest_column_mapping(st.session_state.raw_data)
    if suggestions:
        st.subheader("\U0001F50D Possible Column Matches")
        for actual, info in suggestions.items():
            st.write(f"**{actual}** -> possibly **{info['suggested']}** (confidence: {info['confidence']*100:.0f}%)")

    if profile.date_columns:
        st.session_state.date_col = st.selectbox("Detected date column", profile.date_columns, key="profile_date_col")
    if profile.numeric_columns:
        default_target = "revenue" if "revenue" in profile.numeric_columns else profile.numeric_columns[0]
        st.session_state.target_col = st.selectbox(
            "Forecast target column", profile.numeric_columns,
            index=profile.numeric_columns.index(default_target), key="profile_target_col",
        )


# ------------------------------------------------------------------ Cleaning

def render_cleaning():
    st.header("\U0001F9F9 Data Cleaning")
    if st.session_state.profile is None:
        st.warning("Please run Data Profile first.")
        return

    df = st.session_state.raw_data
    profile = st.session_state.profile

    recs = cleaner.recommend_all(df, st.session_state.column_types)
    if not recs:
        st.success("No missing values detected -- nothing to clean here!")
    else:
        st.subheader("Missing Value Recommendations")
        for rec in recs:
            with st.expander(f"**{rec.column}** -- {rec.missing_pct}% missing -- recommended: {rec.recommended_method}"):
                st.write(f"**Why:** {rec.reason}")
                choice = st.selectbox(
                    "Method", rec.options,
                    index=rec.options.index(rec.recommended_method) if rec.recommended_method in rec.options else 0,
                    key=f"method_{rec.column}",
                )
                custom_value = None
                if choice == "Custom Value":
                    custom_value = st.text_input(f"Custom value for {rec.column}", key=f"custom_{rec.column}")
                st.session_state.missing_choices[rec.column] = {"method": choice, "custom_value": custom_value}

                if st.session_state.dataset_id:
                    database.log_cleaning_action(
                        st.session_state.dataset_id, rec.column, "missing_values",
                        rec.recommended_method, choice,
                    )

    st.subheader("Category Spelling Check")
    cat_col = st.selectbox(
        "Check a categorical column for inconsistencies",
        ["(none)"] + profile.categorical_columns, key="cat_check_col",
    )
    if cat_col != "(none)":
        flagged = cleaner.detect_category_inconsistencies(df, cat_col)
        if not flagged:
            st.success(f"No inconsistencies detected in '{cat_col}'.")
        else:
            corrections = {}
            for item in flagged:
                best_match = item["possible_matches"][0]["match"]
                st.write(f"**'{item['value']}'** ({item['count']} rows) -- possible match: **{best_match}** ({item['possible_matches'][0]['confidence']}% similar)")
                action = st.radio(
                    f"Action for '{item['value']}'", [f"Use '{best_match}'", "Keep Original", "Ignore"],
                    key=f"cat_action_{cat_col}_{item['value']}", horizontal=True,
                )
                if action.startswith("Use"):
                    corrections[item["value"]] = best_match
            if st.button("Apply Category Corrections", key=f"apply_cat_{cat_col}"):
                st.session_state.raw_data = cleaner.apply_category_correction(df, cat_col, corrections)
                st.success("Corrections applied.")
                st.rerun()

    st.divider()
    if st.button("\u2705 Run Full Cleaning Pipeline", type="primary"):
        cleaned = cleaner.full_clean_pipeline(
            df,
            st.session_state.missing_choices,
            date_column=st.session_state.date_col,
            numeric_columns=profile.numeric_columns,
        )
        st.session_state.cleaned_data = cleaned
        st.session_state.date_col = normalize_col_name(st.session_state.date_col)
        st.session_state.target_col = normalize_col_name(st.session_state.target_col)
        st.success(f"Cleaning complete. {len(df)} rows -> {len(cleaned)} rows.")

    if st.session_state.cleaned_data is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Before (raw)")
            st.dataframe(df.head(10), use_container_width=True)
        with col2:  
            st.caption("After (cleaned)")
            st.dataframe(st.session_state.cleaned_data.head(10), use_container_width=True)

        st.download_button(
            "\u2B07\uFE0F Download Cleaned CSV",
            exporter.dataframe_to_csv_bytes(st.session_state.cleaned_data),
            file_name="cleaned_data.csv", mime="text/csv",
        )


# --------------------------------------------------------------- Transform

def render_transformation():
    st.header("\U0001F504 Data Transformation")
    if st.session_state.cleaned_data is None:
        st.warning("Please run Data Cleaning first.")
        return

    df = st.session_state.cleaned_data
    transformed, added_cols = transformer.compute_financial_columns(df)
    st.session_state.transformed_data = transformed

    if added_cols:
        st.success(f"Derived columns added: {', '.join(added_cols)}")
    else:
        st.info("No revenue/cost columns found -- skipping financial feature generation.")

    st.dataframe(transformed.head(20), use_container_width=True)
    st.download_button(
        "\u2B07\uFE0F Download Transformed CSV",
        exporter.dataframe_to_csv_bytes(transformed),
        file_name="transformed_data.csv", mime="text/csv",
    )


# -------------------------------------------------------------------- EDA

def render_analysis():
    st.header("\U0001F4CA Exploratory Analysis")
    df = st.session_state.transformed_data if st.session_state.transformed_data is not None else st.session_state.cleaned_data
    if df is None:
        st.warning("Please run Data Cleaning (and ideally Transformation) first.")
        return

    date_col = st.session_state.date_col
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if date_col and date_col in df.columns and numeric_cols:
        y_col = st.selectbox("Column to plot over time", numeric_cols, key="eda_y_col")
        plot_df = df[[date_col, y_col]].dropna().sort_values(date_col)
        fig = px.line(plot_df, x=date_col, y=y_col, title=f"{y_col} over time")
        st.plotly_chart(fig, use_container_width=True)

    if numeric_cols:
        st.subheader("Correlation Heatmap")
        corr = df[numeric_cols].corr()
        fig2 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0))
        st.plotly_chart(fig2, use_container_width=True)

    profile_summary = profiler.summary_dict(st.session_state.profile) if st.session_state.profile else {}
    insight_summary = insights.build_full_insight_summary(df, date_col or df.columns[0], profile_summary)
    st.session_state.insight_summary = insight_summary

    st.subheader("Automatic Insights")
    for line in insight_summary.get("trends", []):
        st.write(f"- {line}")
    if insight_summary.get("expense"):
        st.write(f"- {insight_summary['expense']}")
    if insight_summary.get("profit"):
        st.write(f"- {insight_summary['profit']}")
    for risk in insight_summary.get("risks", []):
        st.warning(risk)


# ------------------------------------------------------------------ Forecast

def render_forecast():
    st.header("\U0001F52E Forecast")
    df = st.session_state.transformed_data if st.session_state.transformed_data is not None else st.session_state.cleaned_data
    if df is None:
        st.warning("Please run Data Cleaning first.")
        return
    if not st.session_state.date_col or not st.session_state.target_col:
        st.warning("Please select a date column and target column in the Data Profile tab.")
        return

    date_col, target_col = st.session_state.date_col, st.session_state.target_col

    prepared, readiness = transformer.prepare_forecast_ready_data(df, date_col, target_col)
    for w in readiness.warnings:
        st.warning(w)

    if not readiness.ok:
        st.error("\u26A0\uFE0F Data is not sufficient/consistent enough for a reliable forecast. See warnings above.")
        return

    st.success(f"Detected frequency: **{readiness.frequency}** -- {readiness.history_length} historical points.")

    horizon_options = {"daily": ["30 days", "60 days", "90 days"], "weekly": ["4 weeks", "12 weeks", "26 weeks"],
                        "monthly": ["3 months", "6 months", "12 months"]}.get(readiness.frequency, ["30 days"])
    horizon_label = st.selectbox("Forecast horizon", horizon_options)
    periods = transformer.convert_horizon_to_periods(horizon_label, readiness.frequency)
    st.caption(f"This will forecast **{periods}** periods at **{readiness.frequency}** frequency.")

    manual_model = st.selectbox(
        "Model (leave on Auto to let backtesting choose)",
        ["Auto"] + ["Naive", "Seasonal Naive", "Exponential Smoothing", "ARIMA", "Prophet", "Random Forest"],
    )

    if st.button("\U0001F680 Run Forecast", type="primary"):
        model_df = prepared[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
        with st.spinner("Backtesting models and generating forecast..."):
            result = run_forecast(
                model_df, readiness.frequency, periods,
                manual_model=None if manual_model == "Auto" else manual_model,
            )
        st.session_state.forecast_result = result
        st.session_state.forecast_horizon_label = horizon_label

        if st.session_state.dataset_id and result.comparison_table:
            best = result.comparison_table[0]
            database.log_forecast_run(
                st.session_state.dataset_id, target_col, result.recommended_model, periods,
                {"mae": best.mae, "rmse": best.rmse, "mape": best.mape},
            )

    result = st.session_state.forecast_result
    if result is not None:
        for w in result.warnings:
            st.info(w)

        if result.comparison_table:
            st.subheader("Model Comparison (backtested)")
            comp_df = pd.DataFrame([vars(r) for r in result.comparison_table])
            st.dataframe(comp_df, use_container_width=True)

        st.success(f"Recommended model: **{result.recommended_model}**")

        st.subheader(f"Forecast -- next {periods} periods")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prepared[date_col], y=prepared[target_col], name="Historical", mode="lines"))
        fig.add_trace(go.Scatter(x=result.forecast["ds"], y=result.forecast["yhat"], name="Forecast", mode="lines"))
        fig.add_trace(go.Scatter(
            x=list(result.forecast["ds"]) + list(result.forecast["ds"])[::-1],
            y=list(result.forecast["yhat_upper"]) + list(result.forecast["yhat_lower"])[::-1],
            fill="toself", fillcolor="rgba(99,110,250,0.15)", line=dict(width=0), name="Confidence interval",
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "\u26A0\uFE0F Prediction intervals represent uncertainty and are not guarantees. "
            "Forecast reliability depends on historical data quality, seasonality, volatility and horizon."
        )

        st.dataframe(result.forecast, use_container_width=True)
        st.download_button(
            "\u2B07\uFE0F Download Forecast CSV",
            exporter.dataframe_to_csv_bytes(result.forecast),
            file_name="forecast.csv", mime="text/csv",
        )


# --------------------------------------------------------------- AI Insights

def render_ai_insights():
    st.header("\U0001F916 AI Insights")

    if not ai_engine.is_ai_available():
        st.info(
            "No ANTHROPIC_API_KEY configured -- showing rule-based summaries instead. "
            "Set the key in your .env file to enable live AI explanations."
        )

    if st.session_state.transformed_data is None and st.session_state.cleaned_data is None:
        st.warning("Please clean/transform your data first (and ideally run a forecast).")
        return

    df = st.session_state.transformed_data if st.session_state.transformed_data is not None else st.session_state.cleaned_data
    profile_summary = profiler.summary_dict(st.session_state.profile) if st.session_state.profile else {}
    insight_summary = getattr(st.session_state, "insight_summary", None) or insights.build_full_insight_summary(
        df, st.session_state.date_col or df.columns[0], profile_summary
    )

    st.subheader("Data Quality Explanation")
    st.write(ai_engine.explain_data_quality(profile_summary, insight_summary))

    result = st.session_state.forecast_result
    if result is not None:
        st.subheader("Forecast Explanation")
        comparison_dicts = [vars(r) for r in result.comparison_table]
        forecast_summary = {
            "recommended_model": result.recommended_model,
            "horizon": getattr(st.session_state, "forecast_horizon_label", "n/a"),
        }
        st.write(ai_engine.explain_forecast(comparison_dicts, result.recommended_model, forecast_summary))

    st.subheader("Executive Summary")
    forecast_summary = {"recommended_model": result.recommended_model} if result else {}
    st.write(ai_engine.generate_executive_summary(profile_summary, insight_summary, forecast_summary))


# -------------------------------------------------------------------- Export

def render_export():
    st.header("\U0001F4E5 Export")
    if st.session_state.cleaned_data is None:
        st.warning("Please run Data Cleaning first.")
        return

    cleaned = st.session_state.cleaned_data
    transformed = st.session_state.transformed_data if st.session_state.transformed_data is not None else cleaned
    result = st.session_state.forecast_result
    profile = st.session_state.profile

    forecast_df = result.forecast if result else pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    comparison_df = pd.DataFrame([vars(r) for r in result.comparison_table]) if result and result.comparison_table else pd.DataFrame()
    quality_df = pd.DataFrame([
        {"column": cp.name, "missing_pct": cp.missing_pct, "unique": cp.unique_count, "type": cp.inferred_type}
        for cp in profile.column_profiles.values()
    ]) if profile else pd.DataFrame()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("\u2B07\uFE0F Cleaned CSV", exporter.dataframe_to_csv_bytes(cleaned), "cleaned_data.csv", "text/csv")
    with col2:
        st.download_button("\u2B07\uFE0F Forecast CSV", exporter.dataframe_to_csv_bytes(forecast_df), "forecast.csv", "text/csv")
    with col3:
        excel_bytes = exporter.dataframe_to_excel_bytes({"Cleaned": cleaned, "Forecast": forecast_df, "Model Comparison": comparison_df})
        st.download_button("\u2B07\uFE0F Excel Report", excel_bytes, "report.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    insight_summary = getattr(st.session_state, "insight_summary", {}) or {}
    kpis = {}
    if "revenue" in transformed.columns:
        kpis["total_revenue"] = round(transformed["revenue"].sum(), 2)
    if "profit" in transformed.columns:
        kpis["total_profit"] = round(transformed["profit"].sum(), 2)
        kpis["avg_profit_margin_pct"] = round(transformed["profit_margin"].mean(), 2) if "profit_margin" in transformed.columns else "n/a"

    pdf_bytes = exporter.generate_pdf_report(
        dataset_overview=profiler.summary_dict(profile) if profile else {},
        cleaning_summary=[f"{col}: {choice['method']}" for col, choice in st.session_state.missing_choices.items()],
        kpis=kpis,
        forecast_summary={
            "recommended_model": result.recommended_model if result else "n/a",
            "horizon": getattr(st.session_state, "forecast_horizon_label", "n/a"),
            "mae": result.comparison_table[0].mae if result and result.comparison_table else "n/a",
            "rmse": result.comparison_table[0].rmse if result and result.comparison_table else "n/a",
            "mape": result.comparison_table[0].mape if result and result.comparison_table else "n/a",
        },
        business_insights=insight_summary,
        limitations=[],
    )
    st.download_button("\u2B07\uFE0F PDF Summary Report", pdf_bytes, "summary_report.pdf", "application/pdf")

    zip_bytes = exporter.build_export_zip(
        exporter.dataframe_to_csv_bytes(cleaned),
        exporter.dataframe_to_csv_bytes(forecast_df),
        exporter.dataframe_to_csv_bytes(comparison_df),
        exporter.dataframe_to_csv_bytes(quality_df),
        pdf_bytes,
    )
    st.download_button("\U0001F4E6 Download Full ZIP Package", zip_bytes, "project_results.zip", "application/zip", type="primary")


# ------------------------------------------------------------------- Router

PAGES = {
    "\U0001F3E0 Home": render_home,
    "\U0001F4C2 Upload": render_upload,
    "\U0001F50E Data Profile": render_profile,
    "\U0001F9F9 Data Cleaning": render_cleaning,
    "\U0001F504 Transformation": render_transformation,
    "\U0001F4CA Analysis": render_analysis,
    "\U0001F52E Forecast": render_forecast,
    "\U0001F916 AI Insights": render_ai_insights,
    "\U0001F4E5 Export": render_export,
}


def main():
    render_auth_sidebar()
    st.sidebar.divider()
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()), label_visibility="collapsed")

    if choice != "\U0001F3E0 Home":
        require_login()

    PAGES[choice]()


if __name__ == "__main__":
    main()
