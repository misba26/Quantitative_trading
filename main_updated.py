# ============================================================
# QUANTITATIVE TRADING WITH MACHINE LEARNING
# Complete pipeline + Dynamic Prediction Threshold
# ============================================================

# STEP 1: Import libraries
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)

# Keep input/output paths relative to this script's folder.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

print("Libraries imported successfully!")


# ============================================================
# STEP 2: Load local dataset
# ============================================================

file_path = os.path.join(BASE_DIR, "dow_jones_index.data")

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"Dataset not found: {file_path}\\n"
        "Place dow_jones_index.data in the same folder as main.py."
    )

print(f"Dataset found: {file_path}")
df = pd.read_csv(file_path, skipinitialspace=True)

# Clean column names
df.columns = df.columns.str.strip()

print("\\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\\nColumns:")
print(df.columns.tolist())
print(df.head())



# STEP 3: Clean data
# ============================================================

# Remove extra spaces from text columns
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.strip()

# Check required columns
required_columns = ["date", "stock", "percent_change_next_weeks_price"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}. "
        "Check that you uploaded the original Dow Jones dataset."
    )

# Parse dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Convert financial columns to numeric
# Handles values such as $15.82, 2.50%, and numbers with commas.
for col in df.columns:
    if col not in ["date", "stock"]:
        cleaned = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        df[col] = pd.to_numeric(cleaned, errors="coerce")

# Remove rows without essential information
df = df.dropna(
    subset=["date", "stock", "percent_change_next_weeks_price"]
).copy()

# Target:
# 1 = next week's price change is positive
# 0 = next week's price change is zero or negative
df["Target"] = (
    df["percent_change_next_weeks_price"] > 0
).astype(int)

# Sort chronologically
df = df.sort_values(["date", "stock"]).reset_index(drop=True)

print("\nCleaned dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["Target"].value_counts().sort_index())

print(df.head())


# ============================================================
# STEP 4: Remove future-information columns
# ============================================================

# These columns must not be used to predict the target.
future_columns = [
    "next_weeks_open",
    "next_weeks_close",
    "percent_change_next_weeks_price",
    "percent_return_next_dividend"
]

# Date is used for chronological splitting, not as a feature.
columns_to_drop = ["date", "Target"]

for col in future_columns:
    if col in df.columns:
        columns_to_drop.append(col)

X = df.drop(columns=columns_to_drop, errors="ignore")
y = df["Target"].copy()

# Remove columns that contain no usable values
X = X.dropna(axis=1, how="all")

# Make sure the stock identifier is treated as categorical
if "stock" in X.columns:
    X["stock"] = X["stock"].astype(str)

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

print("\nNumerical features:", numerical_columns)
print("\nCategorical features:", categorical_columns)
print("\nTotal input features:", X.shape[1])


# ============================================================
# STEP 5: Chronological train/validation/test split
# ============================================================

# Earlier dates for training, later dates for validation and testing.
unique_dates = np.array(sorted(df["date"].unique()))

if len(unique_dates) < 5:
    raise ValueError(
        "Not enough distinct dates for a chronological split."
    )

train_end = int(len(unique_dates) * 0.60)
validation_end = int(len(unique_dates) * 0.80)

train_dates = unique_dates[:train_end]
validation_dates = unique_dates[train_end:validation_end]
test_dates = unique_dates[validation_end:]

train_mask = df["date"].isin(train_dates)
validation_mask = df["date"].isin(validation_dates)
test_mask = df["date"].isin(test_dates)

X_train = X.loc[train_mask].copy()
X_validation = X.loc[validation_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_validation = y.loc[validation_mask].copy()
y_test = y.loc[test_mask].copy()

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_validation))
print("Test rows:", len(X_test))

print("\nTraining dates:",
      df.loc[train_mask, "date"].min(),
      "to", df.loc[train_mask, "date"].max())

print("Validation dates:",
      df.loc[validation_mask, "date"].min(),
      "to", df.loc[validation_mask, "date"].max())

print("Test dates:",
      df.loc[test_mask, "date"].min(),
      "to", df.loc[test_mask, "date"].max())


# ============================================================
# STEP 6: Build preprocessing pipeline
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numerical_columns:
    transformers.append(
        ("numeric", numeric_pipeline, numerical_columns)
    )

if categorical_columns:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_columns)
    )

preprocessor = ColumnTransformer(transformers=transformers)

# Fit preprocessing only on training data
X_train_processed = preprocessor.fit_transform(X_train)

X_validation_processed = preprocessor.transform(X_validation)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed!")
print("Processed training shape:", X_train_processed.shape)
print("Processed validation shape:", X_validation_processed.shape)
print("Processed test shape:", X_test_processed.shape)


# ============================================================
# STEP 7: Train Logistic Regression
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_processed, y_train)

logistic_predictions = logistic_model.predict(X_test_processed)

print("\nLogistic Regression trained successfully!")


# ============================================================
# STEP 8: Train Random Forest
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

random_forest.fit(X_train_processed, y_train)

rf_test_predictions = random_forest.predict(X_test_processed)

print("Random Forest trained successfully!")


# ============================================================
# STEP 9: Compare models on the final test set
# ============================================================

model_results = []

for model_name, predictions in [
    ("Logistic Regression", logistic_predictions),
    ("Random Forest", rf_test_predictions)
]:
    model_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

model_comparison = pd.DataFrame(model_results)

print("\nMODEL COMPARISON — FINAL TEST SET")
print(model_comparison.round(4).to_string(index=False))

os.makedirs("results", exist_ok=True)

model_comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)


# ============================================================
# STEP 10: Get Random Forest probabilities
# ============================================================

# These probabilities are generated by the already-trained model.
# Moving the slider will change the decision threshold, not retrain it.

rf_validation_probabilities = random_forest.predict_proba(
    X_validation_processed
)[:, 1]

rf_test_probabilities = random_forest.predict_proba(
    X_test_processed
)[:, 1]

# Actual next-week percentage returns for backtesting.
# The original dataset stores these values in percentage points.
validation_actual_returns = (
    df.loc[validation_mask, "percent_change_next_weeks_price"]
    .to_numpy(dtype=float) / 100.0
)

test_actual_returns = (
    df.loc[test_mask, "percent_change_next_weeks_price"]
    .to_numpy(dtype=float) / 100.0
)

validation_actual_labels = y_validation.to_numpy(dtype=int)
test_actual_labels = y_test.to_numpy(dtype=int)

validation_dates_aligned = (
    df.loc[validation_mask, "date"].to_numpy()
)

test_dates_aligned = (
    df.loc[test_mask, "date"].to_numpy()
)


# ============================================================
# STEP 11: Helper function for strategy evaluation
# ============================================================

def evaluate_strategy(dates, signals, actual_returns, initial_capital=100000):
    """
    Long-or-flat strategy.

    For each date:
    - Buy/hold the assets with positive signals.
    - Give selected assets equal weight.
    - If there are no positive signals, hold cash for that period.

    This is a simplified historical simulation. It excludes transaction
    costs, slippage, taxes and other practical trading constraints.
    """

    rows = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "signal": np.asarray(signals, dtype=int),
        "actual_return": np.asarray(actual_returns, dtype=float)
    })

    # Return contribution from selected assets
    rows["selected_return"] = (
        rows["signal"] * rows["actual_return"]
    )

    # Aggregate asset returns into one portfolio return per date
    grouped = rows.groupby("date").agg(
        selected_return_sum=("selected_return", "sum"),
        selected_count=("signal", "sum")
    )

    grouped["strategy_return"] = np.where(
        grouped["selected_count"] > 0,
        grouped["selected_return_sum"] / grouped["selected_count"],
        0.0
    )

    grouped = grouped.sort_index()

    portfolio_values = initial_capital * np.cumprod(
        1 + grouped["strategy_return"].to_numpy()
    )

    values_with_initial = np.insert(
        portfolio_values, 0, initial_capital
    )

    running_peaks = np.maximum.accumulate(values_with_initial)

    drawdowns = (
        values_with_initial / running_peaks
    ) - 1

    total_return = (
        portfolio_values[-1] / initial_capital - 1
    ) * 100

    max_drawdown = drawdowns.min() * 100

    period_returns = grouped["strategy_return"].to_numpy()

    if len(period_returns) > 1 and np.std(period_returns, ddof=1) > 0:
        sharpe_ratio = (
            np.mean(period_returns) /
            np.std(period_returns, ddof=1)
        ) * np.sqrt(52)
    else:
        sharpe_ratio = np.nan

    grouped["Portfolio_Value"] = portfolio_values

    return {
        "history": grouped,
        "final_value": portfolio_values[-1],
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio
    }


# ============================================================
# STEP 12: Dynamic threshold selection in the terminal
# ============================================================

print("\\nDYNAMIC PREDICTION DASHBOARD")
print("Choose a threshold between 0.30 and 0.70.")
print("Default threshold is 0.50.")

while True:
    threshold_input = input("Enter validation threshold (0.30-0.70, press Enter for 0.50): ").strip()
    if threshold_input == "":
        validation_threshold = 0.50
        break
    try:
        validation_threshold = float(threshold_input)
        if 0.30 <= validation_threshold <= 0.70:
            break
        print("Please enter a value from 0.30 to 0.70.")
    except ValueError:
        print("Invalid input. Enter a number such as 0.45 or 0.60.")

validation_predictions = (
    rf_validation_probabilities >= validation_threshold
).astype(int)

print(f"\\nSelected threshold: {validation_threshold:.2f}")
print("\\nVALIDATION CLASSIFICATION METRICS")
print(f"Accuracy:  {accuracy_score(validation_actual_labels, validation_predictions):.4f}")
print(f"Precision: {precision_score(validation_actual_labels, validation_predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(validation_actual_labels, validation_predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(validation_actual_labels, validation_predictions, zero_division=0):.4f}")

print("\\nVALIDATION TRADING SIMULATION")
validation_result = evaluate_strategy(
    validation_dates_aligned,
    validation_predictions,
    validation_actual_returns,
)
print(f"Final portfolio value: ₹{validation_result['final_value']:,.2f}")
print(f"Total return: {validation_result['total_return']:.2f}%")
print(f"Maximum drawdown: {validation_result['max_drawdown']:.2f}%")
print(f"Annualized Sharpe ratio (52 periods/year): {validation_result['sharpe_ratio']:.4f}")
print(f"Positive signals: {int(validation_predictions.sum())} out of {len(validation_predictions)}")

validation_history = validation_result["history"]
plt.figure(figsize=(10, 4))
plt.plot(validation_history.index, validation_history["Portfolio_Value"], marker="o")
plt.axhline(100000, linestyle="--", label="Initial capital")
plt.title(f"Validation Portfolio — Threshold {validation_threshold:.2f}")
plt.xlabel("Date")
plt.ylabel("Portfolio value (₹)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# STEP 13: Final test evaluation at a fixed 0.50 threshold
# ============================================================

# Do not tune the threshold using the final test results.
# This provides a separate, fixed-threshold test evaluation.

FINAL_THRESHOLD = 0.50

final_test_predictions = (
    rf_test_probabilities >= FINAL_THRESHOLD
).astype(int)

print("\nFINAL TEST EVALUATION")
print("Fixed threshold:", FINAL_THRESHOLD)

print("\nClassification report:")
print(
    classification_report(
        test_actual_labels,
        final_test_predictions,
        zero_division=0
    )
)

final_test_strategy = evaluate_strategy(
    test_dates_aligned,
    final_test_predictions,
    test_actual_returns
)

print("\nFINAL TEST STRATEGY RESULTS")
print(
    f"Final portfolio value: ₹{final_test_strategy['final_value']:,.2f}"
)
print(
    f"Total return: {final_test_strategy['total_return']:.2f}%"
)
print(
    f"Maximum drawdown: {final_test_strategy['max_drawdown']:.2f}%"
)
print(
    f"Sharpe ratio: {final_test_strategy['sharpe_ratio']:.4f}"
)

plt.figure(figsize=(10, 4))

plt.plot(
    final_test_strategy["history"].index,
    final_test_strategy["history"]["Portfolio_Value"],
    marker="o"
)

plt.axhline(
    100000,
    linestyle="--",
    label="Initial capital"
)

plt.title("Final Test Portfolio — Fixed Threshold 0.50")
plt.xlabel("Date")
plt.ylabel("Portfolio value (₹)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# STEP 14: Save test predictions and results
# ============================================================

test_results = df.loc[test_mask].copy()

test_results["Actual_Target"] = test_actual_labels
test_results["Predicted_Probability"] = rf_test_probabilities
test_results["Predicted_Target"] = final_test_predictions
test_results["Signal"] = final_test_predictions

test_results["Actual_Return_Decimal"] = test_actual_returns
test_results["Strategy_Return_Per_Asset"] = (
    test_results["Signal"] *
    test_results["Actual_Return_Decimal"]
)

test_results.to_csv(
    "results/trading_results.csv",
    index=False
)

final_test_strategy["history"].to_csv(
    "results/portfolio_history.csv"
)

# Save trained model and preprocessing pipeline
import joblib

joblib.dump(
    random_forest,
    "results/random_forest_model.pkl"
)

joblib.dump(
    logistic_model,
    "results/logistic_regression_model.pkl"
)

joblib.dump(
    preprocessor,
    "results/preprocessor.pkl"
)

print("\nAll results saved inside the results folder.")

print("\nFiles created:")
for root, dirs, filenames in os.walk("results"):
    for file in filenames:
        print(os.path.join(root, file))


# ============================================================
# STEP 15: Completion
# ============================================================

print("\\nResult files are saved in the results folder.")
print("Open the results folder next to main.py to view the CSV and model files.")
print("\\nPROJECT EXECUTION COMPLETED!")
