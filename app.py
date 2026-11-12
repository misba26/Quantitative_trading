from __future__ import annotations

import hashlib
import io
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATASET_NAME = "dow_jones_index.data"
INITIAL_CAPITAL = 100_000.0

st.set_page_config(
    page_title="QuantML | Trading Research Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark, trading-terminal-inspired styling. Streamlit's theme remains functional.
st.markdown(
    """
    <style>
      .stApp { background: #0b1120; color: #e5e7eb; }
      [data-testid="stHeader"] { background: rgba(11,17,32,0.96); }
      [data-testid="stSidebar"] { background: #101827; border-right: 1px solid #243247; }
      [data-testid="stMetric"] { background: linear-gradient(135deg,#131f31,#101827); border: 1px solid #26364c; padding: 16px 18px; border-radius: 14px; }
      [data-testid="stMetricLabel"] { color: #9fb0c7; }
      [data-testid="stMetricValue"] { color: #f8fafc; }
      .hero { padding: 22px 24px; border: 1px solid #26364c; border-radius: 18px; background: linear-gradient(115deg,#14253a,#101827 65%,#102b2b); margin-bottom: 18px; }
      .hero h1 { margin: 0; color: #f8fafc; font-size: 2rem; }
      .hero p { color: #a9bad0; margin: 8px 0 0 0; }
      .eyebrow { color: #4ade80; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
      .section-note { color: #9fb0c7; font-size: .9rem; }
      .status-pill { display:inline-block; padding:4px 9px; border-radius:999px; background:#153a32; color:#6ee7b7; border:1px solid #236354; font-size:.78rem; }
      div[data-testid="stTabs"] button { color: #9fb0c7; }
      div[data-testid="stTabs"] button[aria-selected="true"] { color: #6ee7b7; border-bottom-color: #34d399; }
      .small-muted { color:#94a3b8; font-size:.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_dataset(uploaded_file) -> pd.DataFrame:
    """Load the user-provided historical Dow Jones dataset."""
    if uploaded_file is not None:
        raw = uploaded_file.getvalue()
        return pd.read_csv(io.BytesIO(raw), skipinitialspace=True)
    local_path = BASE_DIR / DATASET_NAME
    if local_path.exists():
        return pd.read_csv(local_path, skipinitialspace=True)
    raise FileNotFoundError(
        f"Could not find {DATASET_NAME}. Upload it in the sidebar or place it beside app.py."
    )


def prepare_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = df.columns.astype(str).str.strip()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    required = {"date", "stock", "percent_change_next_weeks_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "The uploaded dataset is missing required columns: " + ", ".join(sorted(missing))
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in df.columns:
        if col not in ("date", "stock"):
            df[col] = pd.to_numeric(
                df[col].astype(str)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip(),
                errors="coerce",
            )

    df = df.dropna(subset=["date", "stock", "percent_change_next_weeks_price"]).copy()
    if df.empty:
        raise ValueError("No usable rows remain after cleaning the dataset.")
    df["Target"] = (df["percent_change_next_weeks_price"] > 0).astype(int)
    df = df.sort_values(["date", "stock"]).reset_index(drop=True)
    return df


def evaluate_strategy(dates, signals, actual_returns, initial_capital=INITIAL_CAPITAL):
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "signal": np.asarray(signals, dtype=int),
            "actual_return": np.asarray(actual_returns, dtype=float),
        }
    )
    rows["selected_return"] = rows["signal"] * rows["actual_return"]
    grouped = rows.groupby("date").agg(
        selected_return_sum=("selected_return", "sum"),
        selected_count=("signal", "sum"),
    )
    grouped["strategy_return"] = np.where(
        grouped["selected_count"] > 0,
        grouped["selected_return_sum"] / grouped["selected_count"],
        0.0,
    )
    grouped = grouped.sort_index()
    if grouped.empty:
        raise ValueError("The selected period has no rows to evaluate.")
    portfolio_values = initial_capital * np.cumprod(1 + grouped["strategy_return"].to_numpy())
    values_with_initial = np.insert(portfolio_values, 0, initial_capital)
    running_peaks = np.maximum.accumulate(values_with_initial)
    drawdowns = values_with_initial / running_peaks - 1
    total_return = (portfolio_values[-1] / initial_capital - 1) * 100
    max_drawdown = drawdowns.min() * 100
    period_returns = grouped["strategy_return"].to_numpy()
    if len(period_returns) > 1 and np.std(period_returns, ddof=1) > 0:
        sharpe_ratio = (np.mean(period_returns) / np.std(period_returns, ddof=1)) * np.sqrt(52)
    else:
        sharpe_ratio = np.nan
    grouped["Portfolio_Value"] = portfolio_values
    grouped["Drawdown_Percent"] = (grouped["Portfolio_Value"] / np.maximum.accumulate(values_with_initial)[1:] - 1) * 100
    return {
        "history": grouped,
        "final_value": float(portfolio_values[-1]),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe_ratio) if np.isfinite(sharpe_ratio) else np.nan,
    }


@st.cache_data(show_spinner=False)
def train_and_evaluate(csv_bytes: bytes, file_hash: str):
    """Fit models using chronological splits; cached by the actual uploaded data."""
    del file_hash  # Included as an explicit cache key for uploaded/local dataset changes.
    raw = pd.read_csv(io.BytesIO(csv_bytes), skipinitialspace=True)
    df = prepare_dataframe(raw)

    future_columns = [
        "next_weeks_open",
        "next_weeks_close",
        "percent_change_next_weeks_price",
        "percent_return_next_dividend",
    ]
    drop_columns = ["date", "Target"] + [c for c in future_columns if c in df.columns]
    X = df.drop(columns=drop_columns, errors="ignore").dropna(axis=1, how="all")
    y = df["Target"].copy()
    if "stock" in X.columns:
        X["stock"] = X["stock"].astype(str)
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_columns = [c for c in X.columns if c not in categorical_columns]

    unique_dates = np.array(sorted(df["date"].unique()))
    if len(unique_dates) < 5:
        raise ValueError("At least five distinct dates are required for a chronological split.")
    train_end = int(len(unique_dates) * 0.60)
    validation_end = int(len(unique_dates) * 0.80)
    if train_end < 1 or validation_end <= train_end or validation_end >= len(unique_dates):
        raise ValueError("Could not create non-empty chronological train, validation, and test sets.")

    train_dates = unique_dates[:train_end]
    validation_dates = unique_dates[train_end:validation_end]
    test_dates = unique_dates[validation_end:]
    train_mask = df["date"].isin(train_dates)
    val_mask = df["date"].isin(validation_dates)
    test_mask = df["date"].isin(test_dates)

    X_train, X_val, X_test = X.loc[train_mask].copy(), X.loc[val_mask].copy(), X.loc[test_mask].copy()
    y_train, y_val, y_test = y.loc[train_mask].copy(), y.loc[val_mask].copy(), y.loc[test_mask].copy()
    if y_train.nunique() < 2:
        raise ValueError("Training split contains only one target class; the models need both classes.")

    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    transformers = []
    if numerical_columns:
        transformers.append(("numeric", numeric_pipeline, numerical_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    preprocessor = ColumnTransformer(transformers=transformers)
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)
    X_test_p = preprocessor.transform(X_test)

    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1
    )
    logistic_model.fit(X_train_p, y_train)
    rf_model.fit(X_train_p, y_train)

    logistic_pred = logistic_model.predict(X_test_p)
    rf_default_pred = rf_model.predict(X_test_p)

    model_rows = []
    for name, pred in [("Logistic Regression", logistic_pred), ("Random Forest", rf_default_pred)]:
        model_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1 score": f1_score(y_test, pred, zero_division=0),
            }
        )

    validation_returns = df.loc[val_mask, "percent_change_next_weeks_price"].to_numpy(dtype=float) / 100.0
    test_returns = df.loc[test_mask, "percent_change_next_weeks_price"].to_numpy(dtype=float) / 100.0
    val_prob = rf_model.predict_proba(X_val_p)[:, 1]
    test_prob = rf_model.predict_proba(X_test_p)[:, 1]

    test_rows = df.loc[test_mask].copy()
    test_rows["Actual_Target"] = y_test.to_numpy(dtype=int)
    test_rows["Predicted_Probability"] = test_prob
    test_rows["Predicted_Target"] = (test_prob >= 0.50).astype(int)
    test_rows["Signal"] = test_rows["Predicted_Target"]
    test_rows["Actual_Return_Decimal"] = test_returns
    test_rows["Strategy_Return_Per_Asset"] = test_rows["Signal"] * test_rows["Actual_Return_Decimal"]

    return {
        "df": df,
        "model_comparison": pd.DataFrame(model_rows),
        "validation": {
            "dates": df.loc[val_mask, "date"].to_numpy(),
            "labels": y_val.to_numpy(dtype=int),
            "returns": validation_returns,
            "probabilities": val_prob,
            "rows": df.loc[val_mask].copy(),
        },
        "test": {
            "dates": df.loc[test_mask, "date"].to_numpy(),
            "labels": y_test.to_numpy(dtype=int),
            "returns": test_returns,
            "probabilities": test_prob,
            "rows": test_rows,
        },
        "models": {"Logistic Regression": logistic_model, "Random Forest": rf_model},
        "preprocessor": preprocessor,
        "feature_count": int(X.shape[1]),
        "features": X.columns.tolist(),
        "split_counts": {"train": int(train_mask.sum()), "validation": int(val_mask.sum()), "test": int(test_mask.sum())},
        "split_dates": {
            "train_start": str(df.loc[train_mask, "date"].min().date()),
            "train_end": str(df.loc[train_mask, "date"].max().date()),
            "validation_start": str(df.loc[val_mask, "date"].min().date()),
            "validation_end": str(df.loc[val_mask, "date"].max().date()),
            "test_start": str(df.loc[test_mask, "date"].min().date()),
            "test_end": str(df.loc[test_mask, "date"].max().date()),
        },
        "class_counts": df["Target"].value_counts().sort_index().to_dict(),
    }


def money(value: float) -> str:
    return f"₹{value:,.2f}"


def metric_line(label: str, value: float, percent: bool = False):
    return f"{value:.2f}%" if percent else f"{value:.4f}"


# Sidebar controls
with st.sidebar:
    st.markdown("<div class='eyebrow'>QUANTML TERMINAL</div>", unsafe_allow_html=True)
    st.title("Research controls")
    uploaded_file = st.file_uploader(
        "Historical dataset (.data or .csv)", type=["data", "csv"],
        help="Upload the original dow_jones_index.data file. This app does not fetch live market prices.",
    )
    st.caption("Or place dow_jones_index.data next to app.py.")
    st.divider()
    threshold = st.slider("Random Forest probability threshold", 0.30, 0.70, 0.50, 0.05)
    st.caption("A stock is selected when its predicted probability is at least this threshold.")
    st.divider()
    st.markdown("**Model configuration**")
    st.write("Train / validation / test: 60% / 20% / 20% by date")
    st.write("Random Forest: 200 trees · max depth 10")
    st.write("Initial simulated capital: ₹1,00,000")
    st.caption("Historical educational simulation only — not investment advice.")

try:
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        dataset_label = uploaded_file.name
    else:
        local_file = BASE_DIR / DATASET_NAME
        if not local_file.exists():
            st.markdown(
                "<div class='hero'><div class='eyebrow'>HISTORICAL MODEL WORKBENCH</div>"
                "<h1>Quantitative Trading with Machine Learning</h1>"
                "<p>Explore model quality, threshold-based signals, and historical portfolio simulations.</p></div>",
                unsafe_allow_html=True,
            )
            st.warning(f"Dataset not found. Upload `{DATASET_NAME}` using the sidebar, or copy it into this folder:")
            st.code(str(BASE_DIR))
            st.info("No market values or model results are fabricated. The dashboard appears after a valid historical dataset is loaded.")
            st.stop()
        raw_bytes = local_file.read_bytes()
        dataset_label = DATASET_NAME

    digest = hashlib.sha256(raw_bytes).hexdigest()
    with st.spinner("Preparing historical data and training Logistic Regression and Random Forest…"):
        result = train_and_evaluate(raw_bytes, digest)
except Exception as exc:
    st.error(f"Could not load or model the dataset: {exc}")
    st.markdown("Check that you are using the original Dow Jones dataset and that its required columns include `date`, `stock`, and `percent_change_next_weeks_price`.")
    st.stop()

# Threshold-dependent historical validation/test results.
val = result["validation"]
test = result["test"]
val_predictions = (val["probabilities"] >= threshold).astype(int)
test_predictions = (test["probabilities"] >= threshold).astype(int)
val_strategy = evaluate_strategy(val["dates"], val_predictions, val["returns"])
test_strategy = evaluate_strategy(test["dates"], test_predictions, test["returns"])

st.markdown(
    "<div class='hero'><div class='eyebrow'>HISTORICAL MODEL WORKBENCH</div>"
    "<h1>Quantitative Trading <span style='color:#6ee7b7'>×</span> Machine Learning</h1>"
    "<p>Model comparison, probability-based signals, and historical portfolio analysis — driven by your dataset.</p>"
    "<span class='status-pill'>● Dataset loaded</span> &nbsp; "
    f"<span class='small-muted'>Source: {dataset_label} · SHA-256: {digest[:12]}</span></div>",
    unsafe_allow_html=True,
)

# Top-level overview metrics use computed results only.
row1 = st.columns(4)
row1[0].metric("Historical rows", f"{len(result['df']):,}")
row1[1].metric("Stocks", f"{result['df']['stock'].nunique():,}")
row1[2].metric("Test accuracy · RF @ 0.50", f"{result['model_comparison'].loc[result['model_comparison']['Model'] == 'Random Forest', 'Accuracy'].iloc[0] * 100:.1f}%")
row1[3].metric("Test period return", f"{test_strategy['total_return']:+.2f}%", help="Computed from the selected threshold on historical test-period returns; excludes trading costs.")

# Navigation tabs
tab_dashboard, tab_predictions, tab_models, tab_portfolio = st.tabs(
    ["◈ Dashboard", "⌁ Predictions", "▦ Model comparison", "↗ Portfolio analytics"]
)

with tab_dashboard:
    st.subheader("Market research overview")
    st.markdown("<p class='section-note'>Dataset coverage and computed validation metrics. These are historical records, not live quotes.</p>", unsafe_allow_html=True)
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Dataset coverage")
        date_summary = result["df"].groupby("date").size().rename("Stocks recorded").reset_index()
        fig, ax = plt.subplots(figsize=(8, 3.3))
        ax.plot(date_summary["date"], date_summary["Stocks recorded"], marker="o", linewidth=1.8)
        ax.set_xlabel("Historical date")
        ax.set_ylabel("Rows / stocks")
        ax.grid(alpha=0.25)
        fig.autofmt_xdate()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("#### Target distribution")
        target_counts = result["df"]["Target"].map({0: "Non-positive next-week change", 1: "Positive next-week change"}).value_counts()
        st.bar_chart(target_counts)
        st.caption("Target labels are derived from the dataset's `percent_change_next_weeks_price` column.")

    st.markdown("#### Chronological data split")
    split_df = pd.DataFrame([
        {"Split": "Training", "Rows": result["split_counts"]["train"], "Date range": f"{result['split_dates']['train_start']} → {result['split_dates']['train_end']}"},
        {"Split": "Validation", "Rows": result["split_counts"]["validation"], "Date range": f"{result['split_dates']['validation_start']} → {result['split_dates']['validation_end']}"},
        {"Split": "Final test", "Rows": result["split_counts"]["test"], "Date range": f"{result['split_dates']['test_start']} → {result['split_dates']['test_end']}"},
    ])
    st.dataframe(split_df, hide_index=True, use_container_width=True)
    st.info("The probability threshold slider changes validation and test signals. Model fitting is cached and is not repeated when only the threshold changes. For unbiased model reporting, use the fixed 0.50 model-comparison table and treat the threshold-adjusted test strategy as exploratory.")

with tab_predictions:
    st.subheader("Historical stock predictions")
    st.markdown(f"<p class='section-note'>Random Forest probability threshold: <b>{threshold:.2f}</b>. Rows shown are from the held-out historical test period, not future/live predictions.</p>", unsafe_allow_html=True)
    pred_df = test["rows"].copy()
    pred_df["Threshold"] = threshold
    pred_df["Predicted_Target"] = test_predictions
    pred_df["Signal"] = np.where(test_predictions == 1, "SELECT / LONG", "CASH / SKIP")
    pred_df["Probability"] = test["probabilities"]
    pred_df["Actual next-week change (%)"] = pred_df["percent_change_next_weeks_price"]
    pred_df["Actual outcome"] = np.where(pred_df["Actual_Target"] == 1, "Positive", "Non-positive")
    display_cols = [c for c in ["date", "stock", "Probability", "Signal", "Actual outcome", "Actual next-week change (%)"] if c in pred_df.columns]
    search_col1, search_col2 = st.columns([1, 1])
    with search_col1:
        stock_options = ["All stocks"] + sorted(pred_df["stock"].astype(str).unique().tolist())
        selected_stock = st.selectbox("Filter stock", stock_options)
    with search_col2:
        signal_filter = st.selectbox("Filter signal", ["All signals", "SELECT / LONG", "CASH / SKIP"])
    filtered = pred_df.copy()
    if selected_stock != "All stocks":
        filtered = filtered[filtered["stock"].astype(str) == selected_stock]
    if signal_filter != "All signals":
        filtered = filtered[filtered["Signal"] == signal_filter]
    show = filtered[display_cols].copy()
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
    show["Probability"] = show["Probability"].map(lambda x: f"{x:.1%}")
    show["Actual next-week change (%)"] = show["Actual next-week change (%)"].map(lambda x: f"{x:.2f}%")
    st.caption(f"Showing {len(show):,} of {len(pred_df):,} held-out test rows.")
    st.dataframe(show, use_container_width=True, hide_index=True)
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered predictions (CSV)", data=csv_bytes, file_name="historical_stock_predictions.csv", mime="text/csv")
    st.warning("This is a historical classification demonstration. It is not a recommendation to buy or sell any security.")

with tab_models:
    st.subheader("Model performance")
    st.markdown("<p class='section-note'>Logistic Regression and Random Forest are evaluated on the same chronological final test set using each model's default 0.50 decision threshold.</p>", unsafe_allow_html=True)
    model_table = result["model_comparison"].copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 score"]:
        model_table[col] = model_table[col].map(lambda x: f"{x:.3f}")
    st.dataframe(model_table, use_container_width=True, hide_index=True)
    numeric_metrics = result["model_comparison"].set_index("Model")[["Accuracy", "Precision", "Recall", "F1 score"]]
    st.markdown("#### Metric comparison")
    st.bar_chart(numeric_metrics)

    rf_default = result["models"]["Random Forest"]
    rf_default_pred = (test["probabilities"] >= 0.50).astype(int)
    cm = confusion_matrix(test["labels"], rf_default_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual non-positive", "Actual positive"], columns=["Predicted non-positive", "Predicted positive"])
    st.markdown("#### Random Forest confusion matrix · threshold 0.50")
    st.dataframe(cm_df, use_container_width=True)
    st.markdown("#### Classification report")
    report = classification_report(test["labels"], rf_default_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)
    st.download_button("Download model comparison (CSV)", data=result["model_comparison"].to_csv(index=False).encode("utf-8"), file_name="model_comparison.csv", mime="text/csv")
    st.caption(f"Features used: {result['feature_count']}. Preprocessing is fitted on training data only; future-information columns are excluded from the input features.")

with tab_portfolio:
    st.subheader("Historical portfolio simulation")
    st.markdown(f"<p class='section-note'>Equal-weight long-or-flat simulation using Random Forest signals at threshold {threshold:.2f}. Initial capital is ₹1,00,000.</p>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final portfolio value", money(test_strategy["final_value"]))
    c2.metric("Total return", f"{test_strategy['total_return']:+.2f}%")
    c3.metric("Maximum drawdown", f"{test_strategy['max_drawdown']:.2f}%")
    sharpe_text = f"{test_strategy['sharpe_ratio']:.3f}" if np.isfinite(test_strategy["sharpe_ratio"]) else "N/A"
    c4.metric("Annualized Sharpe ratio", sharpe_text)

    hist = test_strategy["history"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(hist.index, hist["Portfolio_Value"], marker="o", linewidth=2, label="Simulated portfolio")
    ax.axhline(INITIAL_CAPITAL, linestyle="--", linewidth=1.4, label="Initial capital")
    ax.set_ylabel("Portfolio value (₹)")
    ax.set_xlabel("Historical date")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("#### Period-by-period portfolio history")
    hist_view = hist.reset_index().rename(columns={"date": "Date", "strategy_return": "Strategy return", "Portfolio_Value": "Portfolio value (₹)", "selected_count": "Selected assets"})
    if "Date" not in hist_view.columns:
        hist_view = hist_view.rename(columns={hist_view.columns[0]: "Date"})
    st.dataframe(hist_view, use_container_width=True, hide_index=True)
    st.download_button("Download portfolio history (CSV)", data=hist.reset_index().to_csv(index=False).encode("utf-8"), file_name="portfolio_history.csv", mime="text/csv")

    with st.expander("Simulation assumptions and limitations"):
        st.markdown(
            "- Uses the dataset's historical next-week percentage change for backtesting.\n"
            "- Signals are long-or-flat; selected stocks receive equal weight for each date.\n"
            "- If no stocks pass the threshold, the simulated strategy holds cash for that period.\n"
            "- Does not include transaction costs, slippage, taxes, liquidity limits, or market impact.\n"
            "- The dataset is historical; this app does not provide live prices or financial advice.\n"
            "- Changing the threshold on test data is exploratory. Prefer choosing thresholds on validation data before final test reporting."
        )

st.divider()
st.markdown(
    "<div class='small-muted'>QUANTML · Educational research dashboard · All displayed results are computed from the loaded historical dataset. No live market data is fetched.</div>",
    unsafe_allow_html=True,
)
