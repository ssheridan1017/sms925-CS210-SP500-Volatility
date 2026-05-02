# sms925-CS210-SP500-Volatility
# Forecasting S&P 500 Volatility: A Machine Learning Approach using Macroeconomic Indicators (2000-2024)
**CS 210: Data Management for Data Science**
**Sam Sheridan**

---

## Project Overview
This project builds an end-to-end data management and machine learning pipeline to analyze how macroeconomic indicators influence S&P 500 closing prices. Using a dataset of S&P 500 prices and macroeconomic data from 2000–2008, the project cleans raw data, stores it in a SQL database, performs exploratory data analysis, and compares a Random Forest Regressor against a Linear Regression baseline to predict closing price.

---

## Repository Structure
```
cs210-project/
├── data/
│   ├── stock_macro_data.csv        # Raw dataset (source: Kaggle)
│   ├── sp500_clean.csv             # Cleaned S&P 500 only dataset
│   └── sp500_with_lags.csv        # Dataset with lagged macro features
├── charts/
│   ├── chart1_close_price.png      # S&P 500 closing price over time
│   ├── chart2_volatility_over_time.png
│   ├── chart3_volatility_distribution.png
│   ├── chart4_macro_indicators.png
│   ├── chart5_correlation_heatmap.png
│   ├── chart6_scatter_plots.png
│   ├── chart7_feature_importance.png
│   ├── chart8_actual_vs_predicted.png
│   └── chart9_model_comparison.png
├── data_cleaning.py                # Step 1: Clean and prepare data
├── eda.py                          # Step 2: Exploratory data analysis
├── sql_database.py                 # Step 3: SQL database and queries
├── model.py                        # Step 4: ML models and evaluation
├── .gitignore
├── README.md
└── requirements.txt
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run scripts in order
Each script builds on the output of the previous one. Run them in this sequence:

```bash
# Step 1 — Clean the raw data
python data_cleaning.py

# Step 2 — Generate EDA charts
python eda.py

# Step 3 — Build SQL database and run analytical queries
python sql_database.py

# Step 4 — Train models and evaluate
python model.py
```

### 3. Outputs
- Cleaned datasets saved to `data/`
- Charts saved to `charts/`
- SQL database saved as `sp500_project.db` (excluded from repo via .gitignore)

---

## Dataset
- **Source:** Kaggle — "Finance & Economics Dataset (2000-Present)" by Khushi Yadav
- **Original size:** 3,000 rows × 24 columns (S&P 500, Dow Jones, NASDAQ)
- **After cleaning:** 1,036 rows × 24 columns (S&P 500 only)
- **Note:** Dataset covers 2000–2008 and is synthetic in nature. Results are intended as a proof-of-concept pipeline rather than real-world predictions.

---

## Methodology
| Step | Tool | Description |
|------|------|-------------|
| Data Cleaning | Pandas | Filter to S&P 500, fix types, engineer volatility feature, create lag features |
| EDA | Matplotlib | 6 charts exploring price trends, volatility, and macro indicator relationships |
| SQL Database | SQLite | Two tables (prices + macro indicators) joined by date key, 5 analytical queries |
| Modeling | scikit-learn | Random Forest Regressor vs Linear Regression baseline, evaluated with MAE and RMSE |

---

## Key Findings
- **Feature Importance:** GDP growth and consumer spending ranked highest in predicting closing price, though all indicators contributed relatively equally (~0.12–0.13 importance score each)
- **Model Performance:** Linear Regression (MAE: $1,013) and Random Forest (MAE: $1,016) performed similarly, suggesting the synthetic dataset lacks the non-linear relationships that would give Random Forest an advantage over real data
- **Macro Regimes:** SQL analysis showed the best S&P 500 returns occurred during low inflation + low unemployment periods (avg daily return +1.39), while high inflation + high unemployment produced the worst returns (-2.52)
- **Volatility:** The 10 most volatile trading days were predominantly bearish (7 out of 10 closed lower than they opened)

---

## Dependencies
See `requirements.txt` for full list. Main libraries:
- `pandas` — data cleaning and manipulation
- `matplotlib` — visualizations
- `scikit-learn` — machine learning models
- `sqlite3` — SQL database (built into Python, no install needed)

---

## Data Sources
- Yadav, Khushi. "Finance & Economics Dataset (2000-Present)." Kaggle, 2025.
- Federal Reserve Board: www.federalreserve.gov
- FRED Economic Data: fred.stlouisfed.org
