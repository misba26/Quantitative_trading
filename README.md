# Quantitative Trading With Machine Learning

## 1. Project Overview

This project uses Machine Learning to predict stock trading outcomes from historical market data. The predictions are converted into trading signals and evaluated using a simple backtesting strategy.

**Models used:**

* Logistic Regression
* Random Forest

## 2. Problem Statement

The objective is to develop a Machine Learning-based system that learns patterns from historical stock data and predicts potential trading opportunities.

## 3. Dataset

The dataset contains historical stock information such as:

* Open, High, Low, Close prices
* Trading Volume
* Stock and Date information
* Price changes
* Dividend-related features
* Target variable

After preprocessing, **720 usable records** were obtained.

## 4. Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Google Colab / Jupyter Notebook

## 5. Project Workflow

```text
Historical Data
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
ML Model Training
      ↓
Prediction
      ↓
Trading Signals
      ↓
Backtesting
      ↓
Performance Evaluation
```

## 6. How to Run

### Step 1

Open:

```text
notebooks/quantitative_trading_ml.ipynb
```

in Google Colab or Jupyter Notebook.

### Step 2

Upload the dataset when prompted.

### Step 3

Install required libraries if needed:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

### Step 4

Run all notebook cells from top to bottom.

The notebook will:

1. Load and preprocess the data.
2. Train Logistic Regression and Random Forest models.
3. Generate predictions.
4. Generate trading signals.
5. Perform backtesting.
6. Display model and trading performance.

## 7. Results

| Model               | Accuracy | F1-Score |
| ------------------- | -------: | -------: |
| Logistic Regression |   47.33% |   50.93% |
| Random Forest       |   44.00% |   50.00% |

Backtesting started with **₹100,000** and resulted in approximately **₹100,007.38**.

## 8. Conclusion

The project demonstrates how Machine Learning can be integrated into a quantitative trading pipeline. Although the models generated trading signals, the small backtested return shows that financial prediction is challenging and that model accuracy alone does not guarantee profitable trading.
