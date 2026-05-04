"""
CS210 Final Project - Predicting S&P 500 Volatility
Sam Sheridan

Goal: Measure the effect that macroeconomic indicators have on S&P 500 daily volatility (engineered feature)
      To achieve this we use two models, a linear regression model and a Random Forest Regressor. We evaluate
      the skill and accuracy of these models with MAE and RMSE, we can use this to isolate indicators that have
      the most influence on daily volatility percent.

Run AFTER data_cleaning.py — this script reads sp500_clean.csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# load data
df = pd.read_csv("sp500_with_lags.csv", parse_dates=["date"])
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# list features
# X: all macro indicators fed in simultaneously
# y: daily_volatility_pct — what we are trying to predict
# We deliberately exclude open, high, low, volume from features
# because those are same-day values — using them would be overfitting.
FEATURE_COLS = [
    "gdp_growth_pct",
    "inflation_rate_pct",
    "unemployment_rate_pct",
    "interest_rate_pct",
    "consumer_confidence",
    "crude_oil_price",
    "gold_price",
    "consumer_spending_bln",

      # lagged features - measures effect of 1 day delay between price changes and indicators
      "gdp_growth_pct_lag1",
    "inflation_rate_pct_lag1",
    "unemployment_rate_pct_lag1",
    "consumer_confidence_lag1",
]
TARGET_COL = "daily_volatility_pct"

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"\nFeatures : {len(FEATURE_COLS)} macro indicators")
print(f"Target   : {TARGET_COL}")
print(f"Samples  : {len(X):,}")

# training/testing data, use shuffle = False to train the model on earlier dates (2000-2006), and train the model on later data (2006-2008)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"\nTrain set: {len(X_train):,} rows")
print(f"Test set : {len(X_test):,} rows")

# scaling features (for linear regression models only)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# LINEAR REGRESSION MODEL

print("MODEL 1: LINEAR REGRESSION (Baseline)")

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

lr_mae  = mean_absolute_error(y_test, lr_preds)
lr_rmse = root_mean_squared_error(y_test, lr_preds)

print(f"MAE  (Mean Absolute Error) : {lr_mae:.2f}")
print(f"RMSE (Root Mean Squared Error) : {lr_rmse:.2f}")
print(f"Interpretation: On average, Linear Regression predicted")
print(f"closing price within ${lr_mae:.2f} of the actual value.")

# RANDOM FOREST REGRESSOR 
# n_estimators=100 means 100 decision trees vote on the prediction.
# random_state=42 ensures reproducible results every time you run it.

print("MODEL 2: RANDOM FOREST REGRESSOR")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1       # use all CPU cores to speed up training
)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)

rf_mae  = mean_absolute_error(y_test, rf_preds)
rf_rmse = root_mean_squared_error(y_test, rf_preds)

print(f"MAE  (Mean Absolute Error)      : {rf_mae:.2f}")
print(f"RMSE (Root Mean Squared Error)  : {rf_rmse:.2f}")
print(f"Interpretation: On average, Random Forest predicted")
print(f"closing price within ${rf_mae:.2f} of the actual value.")

# Comparing Results 

print("MODEL COMPARISON")

print(f"{'Metric':<10} {'Linear Regression':>20} {'Random Forest':>20} {'Improvement':>15}")
print("-" * 70)
mae_improvement  = ((lr_mae  - rf_mae)  / lr_mae)  * 100
rmse_improvement = ((lr_rmse - rf_rmse) / lr_rmse) * 100
print(f"{'MAE':<10} {lr_mae:>20.2f} {rf_mae:>20.2f} {mae_improvement:>14.1f}%")
print(f"{'RMSE':<10} {lr_rmse:>20.2f} {rf_rmse:>20.2f} {rmse_improvement:>14.1f}%")

if rf_mae < lr_mae:
    print(f"\n Random Forest outperforms Linear Regression on both metrics.")
    print(f"  This confirms that non-linear modeling captures patterns")
    print(f"  that linear regression misses — supporting the project hypothesis.")
else:
    print(f"\n Note: Linear Regression performed similarly or better.")
    print(f"  This may be due to the synthetic nature of the dataset.")

# Feature Analysis

print("FEATURE IMPORTANCE (Random Forest)")
importance_df = pd.DataFrame({
    "indicator" : FEATURE_COLS,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False).reset_index(drop=True)

importance_df["rank"] = importance_df.index + 1

for _, row in importance_df.iterrows():
    bar = "█" * int(row["importance"] * 100)
    print(f"  {row['rank']}. {row['indicator']:<30} {row['importance']:.3f}  {bar}")

# CHART 1: FEATURE IMPORTANCE BAR CHART 
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#1f77b4" if i == 0 else "#aec7e8" for i in range(len(importance_df))]
ax.barh(importance_df["indicator"][::-1],
        importance_df["importance"][::-1],
        color=colors[::-1], edgecolor="white")
ax.set_xlabel("Feature Importance Score")
ax.set_title("Which Macro Indicators Most Influence S&P 500 Daily Volatility?\n(Random Forest Feature Importance)", fontsize=12)
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig("chart7_feature_importance.png", dpi=150)
plt.close()
print("\n Saved chart7_feature_importance.png")

# CHART 2: ACTUAL vs PREDICTED (RANDOM FOREST)
# Shows how well the model tracks real closing prices on the test set
test_dates = df["date"].iloc[len(X_train):].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(test_dates, y_test.values, label="Actual Close",    color="#1f77b4", linewidth=1.5)
ax.plot(test_dates, rf_preds,      label="RF Predicted",    color="#d62728", linewidth=1, linestyle="--")
ax.plot(test_dates, lr_preds,      label="LR Predicted",    color="#2ca02c", linewidth=1, linestyle=":")
ax.set_title("Actual vs Predicted S&P 500 Daily Volatility (Test Set)", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (USD)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("chart8_actual_vs_predicted.png", dpi=150)
plt.close()
print(" Saved chart8_actual_vs_predicted.png")

# CHART 3: MODEL COMPARISON BAR CHART 
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

metrics = ["MAE", "RMSE"]
lr_scores = [lr_mae, lr_rmse]
rf_scores = [rf_mae, rf_rmse]
x = np.arange(len(metrics))
width = 0.35

for ax, metric, lr_val, rf_val in zip(axes, metrics, lr_scores, rf_scores):
    bars = ax.bar(["Linear Regression", "Random Forest"],
                  [lr_val, rf_val],
                  color=["#aec7e8", "#1f77b4"], edgecolor="white", width=0.5)
    ax.set_title(f"{metric} Comparison", fontsize=12)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, [lr_val, rf_val]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10)

plt.suptitle("Model Performance Comparison", fontsize=13)
plt.tight_layout()
plt.savefig("chart9_model_comparison.png", dpi=150)
plt.close()
print(" Saved chart9_model_comparison.png")
