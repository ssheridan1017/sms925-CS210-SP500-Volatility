# Forecasting S&P500 Volatility: A Machine Learning Approach using Macroeconomic Indicators

**CS 210: Data Management for Data Science**
**Sam Sheridan**

---

## Overview
This project builds an end-to-end data management and machine learning pipeline to analyze how macroeconomic indicators influence S&P 500 daily volatility. The dataset covers S&P 500 prices and macroeconomic data from 2000–2008. The pipeline covers data cleaning, SQL database design, exploratory data analysis, and a comparison of a Random Forest Regressor against a Linear Regression baseline.

---

## Project Goals
- Clean and filter raw S&P 500 and macroeconomic data using Pandas
- Store data in a SQLite database with two relational tables joined by date key
- Perform exploratory data analysis with 9 Matplotlib visualizations
- Engineer a daily volatility target variable and lagged macro features to capture the delayed effect of indicators on market behavior
- Train and compare a Random Forest Regressor vs Linear Regression baseline using MAE and RMSE
- Identify which macroeconomic indicators most influence S&P 500 volatility via feature importance

---

## Project Structure
```
cs210-project/
├── data/
│   ├── stock_macro_data.csv        # Raw dataset (Kaggle)
│   ├── sp500_clean.csv             # Cleaned S&P 500 only dataset
│   └── sp500_with_lags.csv         # Dataset with lagged macro features
├── charts/                         # All 9 generated visualizations
├── data_cleaning.py                # Step 1: Clean and prepare data
├── eda.py                          # Step 2: Exploratory data analysis
├── sql_database.py                 # Step 3: SQL database and queries
├── model.py                        # Step 4: ML models and evaluation
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Target Variable
```
daily_volatility_pct = (High - Low) / Open × 100
```
Measures how much the S&P 500 swung intraday as a percentage — a standard measure of daily market volatility.

---

## Features (12 total)
**Same-day macro indicators:** GDP Growth, Inflation Rate, Unemployment Rate, Interest Rate, Consumer Confidence, Crude Oil Price, Gold Price, Consumer Spending

**Lagged indicators (previous day):** GDP Growth, Inflation Rate, Unemployment Rate, Consumer Confidence

---

## Results

### Model Performance
| Metric | Linear Regression | Random Forest |
|--------|------------------|---------------|
| MAE    | 1.51%            | 1.54%         |
| RMSE   | 1.95%            | 2.00%         |

Both models performed similarly. Linear Regression marginally outperformed Random Forest, consistent with the synthetic nature of the dataset which limits non-linear pattern detection.

### Feature Importance (Top 5)
| Rank | Indicator | Importance |
|------|-----------|------------|
| 1 | unemployment_rate_pct_lag1 | 10.3% |
| 2 | consumer_spending_bln | 9.4% |
| 3 | inflation_rate_pct | 9.2% |
| 4 | crude_oil_price | 8.8% |
| 5 | inflation_rate_pct_lag1 | 8.7% |

Lagged unemployment ranked #1, outperforming same-day unemployment (8.4%) — directly supporting the lagging effect hypothesis that macro indicators affect volatility with a time delay.

### Key SQL Findings
- Recession signal days (low GDP + high unemployment) had nearly 2.5x worse average daily returns (-1.66 vs -0.69)
- Best macro regime: Low inflation + low unemployment (avg return +1.39)
- Worst macro regime: High inflation + high unemployment (avg return -2.52)
- 7 of the 10 most volatile days were bearish

---

## Dataset
- **Source:** Kaggle — "Finance & Economics Dataset (2000-Present)" by Khushi Yadav
- **Raw size:** 3,000 rows × 24 columns (S&P 500, Dow Jones, NASDAQ)
- **After cleaning:** 1,036 rows (S&P 500 only)
- **Note:** Dataset is synthetic and covers 2000–2008. Results represent a proof-of-concept pipeline.

---

## Limitations
- Synthetic dataset limits real-world generalizability
- 1-day lag may be insufficient — real macro effects can take weeks or months
- No sentiment indicators (VIX, bond yields) which are known volatility predictors

---

## Dependencies
```
pip install -r requirements.txt
```
- `pandas` — data cleaning and manipulation
- `matplotlib` — visualizations
- `scikit-learn` — machine learning models
- `numpy` — numerical operations
- `sqlite3` — built into Python, no install needed

---

## Data Sources
- Yadav, Khushi. "Finance & Economics Dataset (2000-Present)." Kaggle, 2025.
- Federal Reserve Board: www.federalreserve.gov
- FRED Economic Data: fred.stlouisfed.org
