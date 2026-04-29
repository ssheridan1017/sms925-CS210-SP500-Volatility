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
# Each query answers a real analytical question combining price action
# and macro indicators — directly supporting the project's goal of
# understanding how macro conditions affect S&P 500 behavior.

print()
print("=" * 60)
print("ANALYTICAL QUERIES")
print("=" * 60)

# ── QUERY 1: BULLISH vs BEARISH HIGH-VOLATILITY DAYS ─────────────────────────
# On the most volatile days, did the market close up or down?
# close > open = bullish (buyers won), close < open = bearish (sellers won)
# This shows volatility is not just about magnitude — direction matters too.
print("\n--- Top 10 Most Volatile Days: Bullish or Bearish? ---")
q1 = pd.read_sql_query("""
    SELECT
        date,
        ROUND(open, 2)                          AS open,
        ROUND(close, 2)                         AS close,
        ROUND(high, 2)                          AS high,
        ROUND(low, 2)                           AS low,
        ROUND(close - open, 2)                  AS daily_return,
        CASE
            WHEN close > open THEN 'Bullish'
            ELSE 'Bearish'
        END                                     AS direction,
        ROUND(daily_volatility_pct, 3)          AS volatility_pct,
        ROUND(inflation_rate_pct, 2)            AS inflation,
        ROUND(unemployment_rate_pct, 2)         AS unemployment
    FROM sp500_analysis
    ORDER BY daily_volatility_pct DESC
    LIMIT 10
""", conn)
print(q1.to_string(index=False))

# ── QUERY 2: MACRO ENVIRONMENT vs MARKET PERFORMANCE BY YEAR ─────────────────
# For each year, how did the macro environment (inflation, unemployment,
# interest rates) relate to average price levels and volatility?
# This is the core question of the project — do macro conditions predict behavior?
print("\n--- Yearly Macro Environment vs Market Performance ---")
q2 = pd.read_sql_query("""
    SELECT
        SUBSTR(date, 1, 4)                          AS year,
        ROUND(AVG(close), 2)                        AS avg_close,
        ROUND(MAX(high) - MIN(low), 2)              AS annual_price_range,
        ROUND(AVG(close - open), 3)                 AS avg_daily_return,
        ROUND(AVG(daily_volatility_pct), 3)         AS avg_volatility,
        ROUND(AVG(inflation_rate_pct), 2)           AS avg_inflation,
        ROUND(AVG(unemployment_rate_pct), 2)        AS avg_unemployment,
        ROUND(AVG(interest_rate_pct), 2)            AS avg_interest_rate,
        ROUND(AVG(gdp_growth_pct), 2)               AS avg_gdp_growth
    FROM sp500_analysis
    GROUP BY year
    ORDER BY year
""", conn)
print(q2.to_string(index=False))

# ── QUERY 3: HIGH INTEREST RATE ENVIRONMENT vs LOW ───────────────────────────
# Compare average price action and volatility when interest rates are
# above vs below the median. Tests whether rate environment affects market behavior.
print("\n--- High vs Low Interest Rate Environment (split at median) ---")
q3 = pd.read_sql_query("""
    SELECT
        CASE
            WHEN interest_rate_pct > (SELECT AVG(interest_rate_pct) FROM sp500_analysis)
            THEN 'High Interest Rate'
            ELSE 'Low Interest Rate'
        END                                         AS rate_environment,
        COUNT(*)                                    AS trading_days,
        ROUND(AVG(close), 2)                        AS avg_close,
        ROUND(AVG(close - open), 3)                 AS avg_daily_return,
        ROUND(AVG(high - low), 3)                   AS avg_intraday_range,
        ROUND(AVG(daily_volatility_pct), 3)         AS avg_volatility
    FROM sp500_analysis
    GROUP BY rate_environment
""", conn)
print(q3.to_string(index=False))

# ── QUERY 4: RECESSION SIGNAL DAYS ───────────────────────────────────────────
# Flag days where both GDP growth was below 2% AND unemployment was rising
# (above its yearly average) — a basic recession signal.
# Then check: were those days more volatile and more bearish?
print("\n--- Recession Signal Days vs Normal Days ---")
q4 = pd.read_sql_query("""
    SELECT
        CASE
            WHEN gdp_growth_pct < 2.0
             AND unemployment_rate_pct > (SELECT AVG(unemployment_rate_pct) FROM sp500_analysis)
            THEN 'Recession Signal'
            ELSE 'Normal'
        END                                         AS market_condition,
        COUNT(*)                                    AS trading_days,
        ROUND(AVG(close), 2)                        AS avg_close,
        ROUND(AVG(close - open), 3)                 AS avg_daily_return,
        ROUND(AVG(high - low), 3)                   AS avg_intraday_range,
        ROUND(AVG(daily_volatility_pct), 3)         AS avg_volatility,
        SUM(CASE WHEN close < open THEN 1 ELSE 0 END) AS bearish_days
    FROM sp500_analysis
    GROUP BY market_condition
""", conn)
print(q4.to_string(index=False))

# ── QUERY 5: BEST AND WORST MACRO ENVIRONMENTS FOR THE S&P 500 ───────────────
# Group days into 4 macro regimes based on inflation and unemployment
# being above or below their averages. Which combo produced the best returns?
print("\n--- S&P 500 Performance by Macro Regime ---")
q5 = pd.read_sql_query("""
    SELECT
        CASE
            WHEN inflation_rate_pct <= (SELECT AVG(inflation_rate_pct) FROM sp500_analysis)
             AND unemployment_rate_pct <= (SELECT AVG(unemployment_rate_pct) FROM sp500_analysis)
            THEN 'Low Inflation + Low Unemployment'
            WHEN inflation_rate_pct > (SELECT AVG(inflation_rate_pct) FROM sp500_analysis)
             AND unemployment_rate_pct <= (SELECT AVG(unemployment_rate_pct) FROM sp500_analysis)
            THEN 'High Inflation + Low Unemployment'
            WHEN inflation_rate_pct <= (SELECT AVG(inflation_rate_pct) FROM sp500_analysis)
             AND unemployment_rate_pct > (SELECT AVG(unemployment_rate_pct) FROM sp500_analysis)
            THEN 'Low Inflation + High Unemployment'
            ELSE 'High Inflation + High Unemployment'
        END                                         AS macro_regime,
        COUNT(*)                                    AS trading_days,
        ROUND(AVG(close), 2)                        AS avg_close,
        ROUND(AVG(close - open), 3)                 AS avg_daily_return,
        ROUND(AVG(daily_volatility_pct), 3)         AS avg_volatility,
        SUM(CASE WHEN close > open THEN 1 ELSE 0 END) AS bullish_days,
        SUM(CASE WHEN close < open THEN 1 ELSE 0 END) AS bearish_days
    FROM sp500_analysis
    GROUP BY macro_regime
    ORDER BY avg_daily_return DESC
""", conn)
print(q5.to_string(index=False))

# ── ROW COUNT VERIFICATION ────────────────────────────────────────────────────
cursor.execute("SELECT COUNT(*) FROM sp500_analysis")
count = cursor.fetchone()[0]
print(f"\n--- sp500_analysis total rows: {count:,} ---")

# ── CLOSE CONNECTION ──────────────────────────────────────────────────────────
conn.close()
print()
print(f"✓ Database saved to: {DB_PATH}")
