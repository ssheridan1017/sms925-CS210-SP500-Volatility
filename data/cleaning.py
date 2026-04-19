"""
CS210 Final Project - Data Cleaning
Sam Sheridan

Goal: Clean the raw data to filter only for S&P500 data,
type all colums, validate the quality, and export
to a new .csv file to analyze.
"""
import pandas as pd
import numpy as np

# Raw Data
raw = "data/stock_macro_data.csv"
clean = "sp500_clean.csv"
lag = "sp500_with_lags.csv"

df_raw = pd.read)_csv(raw)

print (f"Shape: {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
print (f"Stock Indexes: {df_raw['Stock Index'].value_counts().to_dict()}")
print (f"Null Values: {df_raw.isnull().sum().sum()} total")
print f("Duplicates : {df_raw.duplicated().sum()} rows")

#Filter S&P500 Only
df = df_raw[df_raw['Stock Index"] == "S&P 500"].copy()
print(f"After Filtering to S&P500 ONLY: {len(df):,} rows")

#Column Names
rename_map = {
    "Date": "date",
    "Stock Index": "stock_index",
    "Open Price": "open",
    "Close Price": "close",
    "Daily High": "high",
    "Daily Low": "low",
    "Trading Volume": "volume",
    "GDP Growth (%)": "gdp_growth_pct",
    "Inflation Rate (%)": "inflation_rate_pct",
    "Unemployment Rate (%)": "unemployment_rate_pct",
    "Interest Rate (%)": "interest_rate_pct",
    "Consumer Confidence Index": "consumer_confidence",
    "Government Debt (Billion USD)": "govt_debt_bln",
    "Corporate Profits (Billion USD)": "corp_profits_bln",
    "Forex USD/EUR": "forex_usd_eur",
    "Forex USD/JPY": "forex_usd_jpy",
    "Crude Oil Price (USD per Barrel)": "crude_oil_price",
    "Gold Price (USD per Ounce)": "gold_price",
    "Real Estate Index": "real_estate_index",
    "Retail Sales (Billion USD)": "retail_sales_bln",
    "Bankruptcy Rate (%)": "bankruptcy_rate_pct",
    "Mergers & Acquisitions Deals": "ma_deals",
    "Venture Capital Funding (Billion USD)": "vc_funding_bln",
    "Consumer Spending (Billion USD)": "consumer_spending_bln",
}
df.rename(columns=rename_map, inplace = True)

#Fix Data Types
df["date"] = pd.to_datetime(df["date"])
df.drop(columns=["stock_index"], inplace=True)

#Sort Data
df.sort_values("date", inplace = True)
df.reset_index(drop=True, inplace = True)

#Check Price Integrity
bad_hl = df[df['high'] < df['low']]
print(f"Rows where High < Low: {len(bad_hl)}")

#Check if open and close are within high and low for every row
bad_open  = df[(df["open"]  < df["low"]) | (df["open"]  > df["high"])]
bad_close = df[(df["close"] < df["low"]) | (df["close"] > df["high"])]
print(f"Rows where Open is outside [Low, High]: {len(bad_open)}")
print(f"Rows where Close is outside [Low, High]: {len(bad_close)}")

#Check if volume is always positive
bad_vol = df[df['volume']<=0]
print(f"Rows with non-positive volume: {len(bad_vol)}")

#Check for nulls
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
       print("No nulls after filtering")
else:
      print("Null Values Detected")
      print(null_counts[null_counts>0])

#Feature Engineering for Volatility
df['daily_volatility_pct'] = (df['high'] - df['low']) / df['open'] * 100

#Lag Features (account for downtime in change of indicator and prices for this we will look at a 1 day lag)
lag_cols= [
    "gdp_growth_pct",
    "inflation_rate_pct",
    "unemployment_rate_pct",
    "consumer_confidence",
]
df_lag = df.copy()
for col in lag_cols:
        df_lag[f"{col}_lag1"] = df_lag[col].shift(1)
df_lag.dropna(inplace=True)
df_lag.reset_index(drop=True, inplace = True)

#Data Summary
print("CLEAN DATA SUMMARY (sp500_clean.csv)")
print(f"Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Date range : {df['date'].min().date()}  →  {df['date'].max().date()}")
print(f"Null values: {df.isnull().sum().sum()}")
 
print()
print("LAG DATASET SUMMARY (sp500_with_lags.csv)")
print(f"Shape      : {df_lag.shape[0]:,} rows × {df_lag.shape[1]} columns")
print(f"Date range : {df_lag['date'].min().date()}  →  {df_lag['date'].max().date()}")
print(f"Null values: {df_lag.isnull().sum().sum()}")
print()
print("Added lag columns:")
for col in lag_cols:
    print(f"  {col}_lag1")

#Export
df.to_csv(clean, index=False)
df_lag.to_csv(lag, index=False)


      
