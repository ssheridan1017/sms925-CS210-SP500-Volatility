"""
CS210 Final Project - SQL Database
Sam Sheridan

Goal: Load the cleaned S&P 500 data into a SQLite database with two
      separate tables (prices and macro indicators), then join them
      by date to create a unified analysis table.

Run AFTER data_cleaning.py — this script reads sp500_clean.csv.
"""

import pandas as pd
import sqlite3
import os

# ── LOAD CLEAN DATA ───────────────────────────────────────────────────────────
CLEAN_PATH = "sp500_clean.csv"
DB_PATH    = "sp500_project.db"

df = pd.read_csv(CLEAN_PATH, parse_dates=["date"])
print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── SPLIT INTO TWO TABLES ─────────────────────────────────────────────────────
# Per the proposal: separate tables for price data and macro indicators,
# joined by date key.

# Table 1: sp500_prices — daily financial/trading data
price_cols = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "daily_volatility_pct",
]
df_prices = df[price_cols].copy()

# Table 2: macro_indicators — macroeconomic data per date
macro_cols = [
    "date",
    "gdp_growth_pct",
    "inflation_rate_pct",
    "unemployment_rate_pct",
    "interest_rate_pct",
    "consumer_confidence",
    "govt_debt_bln",
    "corp_profits_bln",
    "forex_usd_eur",
    "forex_usd_jpy",
    "crude_oil_price",
    "gold_price",
    "real_estate_index",
    "retail_sales_bln",
    "bankruptcy_rate_pct",
    "ma_deals",
    "vc_funding_bln",
    "consumer_spending_bln",
]
df_macro = df[macro_cols].copy()

# ── CONNECT TO SQLITE DATABASE ────────────────────────────────────────────────
# This creates the .db file if it doesn't exist yet
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
print(f"\nConnected to database: {DB_PATH}")

# ── CREATE TABLES ─────────────────────────────────────────────────────────────
# Drop tables first so the script is re-runnable without errors
cursor.execute("DROP TABLE IF EXISTS sp500_prices")
cursor.execute("DROP TABLE IF EXISTS macro_indicators")
cursor.execute("DROP TABLE IF EXISTS sp500_analysis")

cursor.execute("""
    CREATE TABLE sp500_prices (
        date                TEXT PRIMARY KEY,
        open                REAL,
        close               REAL,
        high                REAL,
        low                 REAL,
        volume              REAL,
        daily_volatility_pct REAL
    )
""")

cursor.execute("""
    CREATE TABLE macro_indicators (
        date                    TEXT PRIMARY KEY,
        gdp_growth_pct          REAL,
        inflation_rate_pct      REAL,
        unemployment_rate_pct   REAL,
        interest_rate_pct       REAL,
        consumer_confidence     REAL,
        govt_debt_bln           REAL,
        corp_profits_bln        REAL,
        forex_usd_eur           REAL,
        forex_usd_jpy           REAL,
        crude_oil_price         REAL,
        gold_price              REAL,
        real_estate_index       REAL,
        retail_sales_bln        REAL,
        bankruptcy_rate_pct     REAL,
        ma_deals                REAL,
        vc_funding_bln          REAL,
        consumer_spending_bln   REAL
    )
""")

print("✓ Tables created: sp500_prices, macro_indicators")

# ── INSERT DATA ───────────────────────────────────────────────────────────────
# Convert date to string for SQLite storage
df_prices_load  = df_prices.copy()
df_macro_load   = df_macro.copy()
df_prices_load["date"] = df_prices_load["date"].dt.strftime("%Y-%m-%d")
df_macro_load["date"]  = df_macro_load["date"].dt.strftime("%Y-%m-%d")

df_prices_load.to_sql("sp500_prices",     conn, if_exists="append", index=False)
df_macro_load.to_sql("macro_indicators",  conn, if_exists="append", index=False)

print(f"✓ Inserted {len(df_prices_load):,} rows into sp500_prices")
print(f"✓ Inserted {len(df_macro_load):,} rows into macro_indicators")

# ── JOIN TABLES BY DATE ───────────────────────────────────────────────────────
# This is the core SQL step from the proposal — join prices and macro data
# using date as the key to create one unified analysis table.
join_query = """
    CREATE TABLE sp500_analysis AS
    SELECT
        p.date,
        p.open,
        p.close,
        p.high,
        p.low,
        p.volume,
        p.daily_volatility_pct,
        m.gdp_growth_pct,
        m.inflation_rate_pct,
        m.unemployment_rate_pct,
        m.interest_rate_pct,
        m.consumer_confidence,
        m.crude_oil_price,
        m.gold_price,
        m.consumer_spending_bln,
        m.unemployment_rate_pct
    FROM sp500_prices p
    INNER JOIN macro_indicators m
        ON p.date = m.date
    ORDER BY p.date ASC
"""
cursor.execute(join_query)
conn.commit()
print("✓ Created sp500_analysis table via JOIN on date")

# ── EXAMPLE QUERIES ───────────────────────────────────────────────────────────
# These show you can pull meaningful insights from the database.
# Great to include in your final report to demonstrate SQL usage.

print()
print("=" * 60)
print("EXAMPLE QUERIES")
print("=" * 60)

# Query 1: High volatility days (top 10)
print("\n--- Top 10 Most Volatile Days ---")
q1 = pd.read_sql_query("""
    SELECT date, close, daily_volatility_pct, inflation_rate_pct, unemployment_rate_pct
    FROM sp500_analysis
    ORDER BY daily_volatility_pct DESC
    LIMIT 10
""", conn)
print(q1.to_string(index=False))

# Query 2: Average volatility by year
print("\n--- Average Volatility by Year ---")
q2 = pd.read_sql_query("""
    SELECT
        SUBSTR(date, 1, 4) AS year,
        ROUND(AVG(daily_volatility_pct), 3) AS avg_volatility,
        ROUND(AVG(inflation_rate_pct), 3)   AS avg_inflation,
        ROUND(AVG(unemployment_rate_pct), 3) AS avg_unemployment
    FROM sp500_analysis
    GROUP BY year
    ORDER BY year
""", conn)
print(q2.to_string(index=False))

# Query 3: High inflation + high volatility days
print("\n--- Days with High Inflation (>3%) AND High Volatility (>5%) ---")
q3 = pd.read_sql_query("""
    SELECT date, close, daily_volatility_pct, inflation_rate_pct, interest_rate_pct
    FROM sp500_analysis
    WHERE inflation_rate_pct > 3.0
      AND daily_volatility_pct > 5.0
    ORDER BY daily_volatility_pct DESC
    LIMIT 10
""", conn)
print(q3.to_string(index=False))

# Query 4: Row count verification
cursor.execute("SELECT COUNT(*) FROM sp500_analysis")
count = cursor.fetchone()[0]
print(f"\n--- sp500_analysis total rows: {count:,} ---")

# ── CLOSE CONNECTION ──────────────────────────────────────────────────────────
conn.close()
print()
print(f"✓ Database saved to: {DB_PATH}")
print("  (Add sp500_project.db to your .gitignore — commit only this script)")
