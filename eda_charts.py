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
