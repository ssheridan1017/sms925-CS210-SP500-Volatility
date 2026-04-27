"""
CS210 Final Project - Predicting S&P500 Volatility

Goal: This file will cover simple basic data analysis and create visualizations
that I will use in my oral and final report. This is where we will build relationships
between daily volatility and macroeconomic indicators.

*This script is ran after the dataset is cleaned in "cleaning.py"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#load clean data
df = pd.read_csv("sp500_clean.csv"), parse_dates=['date'])

print(f"{df.shape[0]:,} rows x {df.shape[1]} colums")
print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")

# ── CHART 1: S&P 500 CLOSING PRICE OVER TIME ─────────────────────────────────
# Shows the overall price trend across the dataset period.
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["date"], df["close"], color="#1f77b4", linewidth=1)
ax.set_title("S&P 500 Closing Price (2000–2008)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (USD)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("chart1_close_price.png", dpi=150)
plt.close()
print("✓ Saved chart1_close_price.png")
 
# ── CHART 2: DAILY VOLATILITY OVER TIME ──────────────────────────────────────
# This is your TARGET VARIABLE. You should see spikes around major events.
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["date"], df["daily_volatility_pct"], color="#d62728", linewidth=0.8, alpha=0.8)
ax.set_title("S&P 500 Daily Volatility % Over Time (2000–2008)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Volatility % (High-Low / Open)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("chart2_volatility_over_time.png", dpi=150)
plt.close()
print("✓ Saved chart2_volatility_over_time.png")
 
# ── CHART 3: VOLATILITY DISTRIBUTION ─────────────────────────────────────────
# Histogram showing what range of volatility is most common.
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["daily_volatility_pct"], bins=40, color="#ff7f0e", edgecolor="white")
ax.axvline(df["daily_volatility_pct"].mean(), color="black", linestyle="--",
           label=f"Mean: {df['daily_volatility_pct'].mean():.2f}%")
ax.set_title("Distribution of Daily Volatility %", fontsize=14)
ax.set_xlabel("Volatility %")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("chart3_volatility_distribution.png", dpi=150)
plt.close()
print("✓ Saved chart3_volatility_distribution.png")
 
# ── CHART 4: MACRO INDICATORS OVER TIME ──────────────────────────────────────
# Four subplots showing how each key indicator changed over the period.
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
 
indicators = [
    ("inflation_rate_pct",    "Inflation Rate (%)",    "#2ca02c"),
    ("unemployment_rate_pct", "Unemployment Rate (%)", "#9467bd"),
    ("gdp_growth_pct",        "GDP Growth (%)",        "#8c564b"),
    ("interest_rate_pct",     "Interest Rate (%)",     "#e377c2"),
]
 
for ax, (col, label, color) in zip(axes, indicators):
    ax.plot(df["date"], df[col], color=color, linewidth=1)
    ax.set_ylabel(label, fontsize=9)
    ax.grid(True, alpha=0.3)
 
axes[0].set_title("Key Macro Indicators Over Time (2000–2008)", fontsize=14)
axes[-1].set_xlabel("Date")
plt.tight_layout()
plt.savefig("chart4_macro_indicators.png", dpi=150)
plt.close()
print("✓ Saved chart4_macro_indicators.png")
 
# ── CHART 5: CORRELATION HEATMAP ─────────────────────────────────────────────
# Shows which indicators correlate most with daily volatility.
# This directly informs which features you use in your Random Forest.
macro_cols = [
    "gdp_growth_pct",
    "inflation_rate_pct",
    "unemployment_rate_pct",
    "interest_rate_pct",
    "consumer_confidence",
    "crude_oil_price",
    "gold_price",
    "consumer_spending_bln",
    "daily_volatility_pct",
]
 
corr_matrix = df[macro_cols].corr()
 
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
 
# Label axes
labels = [c.replace("_pct", "").replace("_", " ").title() for c in macro_cols]
ax.set_xticks(range(len(macro_cols)))
ax.set_yticks(range(len(macro_cols)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
 
# Add correlation values inside cells
for i in range(len(macro_cols)):
    for j in range(len(macro_cols)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
 
ax.set_title("Correlation Heatmap — Macro Indicators vs Volatility", fontsize=13)
plt.tight_layout()
plt.savefig("chart5_correlation_heatmap.png", dpi=150)
plt.close()
print("✓ Saved chart5_correlation_heatmap.png")
 
# ── CHART 6: TOP INDICATORS vs VOLATILITY (SCATTER) ──────────────────────────
# Scatter plots for the indicators with the strongest correlation to volatility.
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
scatter_cols = [
    ("gdp_growth_pct",        "GDP Growth (%)"),
    ("inflation_rate_pct",    "Inflation Rate (%)"),
    ("unemployment_rate_pct", "Unemployment Rate (%)"),
    ("interest_rate_pct",     "Interest Rate (%)"),
]
 
for ax, (col, label) in zip(axes.flat, scatter_cols):
    ax.scatter(df[col], df["daily_volatility_pct"],
               alpha=0.3, s=10, color="#1f77b4")
    # Add a trend line
    m, b = np.polyfit(df[col], df["daily_volatility_pct"], 1)
    x_line = np.linspace(df[col].min(), df[col].max(), 100)
    ax.plot(x_line, m * x_line + b, color="red", linewidth=1.5)
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("Volatility %", fontsize=9)
    ax.set_title(f"{label} vs Volatility", fontsize=10)
    ax.grid(True, alpha=0.3)
 
plt.suptitle("Macro Indicators vs Daily Volatility", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("chart6_scatter_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved chart6_scatter_plots.png")
 
# ── PRINT CORRELATION SUMMARY ─────────────────────────────────────────────────
print()
print("=" * 50)
print("CORRELATION WITH DAILY VOLATILITY (sorted)")
print("=" * 50)
corr_with_vol = corr_matrix["daily_volatility_pct"].drop("daily_volatility_pct").sort_values()
for col, val in corr_with_vol.items():
    label = col.replace("_pct", "").replace("_", " ").title()
    bar = "█" * int(abs(val) * 20)
    direction = "+" if val > 0 else "-"
    print(f"  {label:<30} {direction}{abs(val):.3f}  {bar}")
 
print()
print("All charts saved. Review them before moving to modeling.")
ax.set_title("S&P 500 Closing Price (2000–2008)", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (USD)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("chart1_close_price.png", dpi=150)
plt.close()
print("Saved chart1_close_price.png")
