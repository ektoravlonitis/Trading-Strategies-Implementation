# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 18:49:39 2024

@author: he_98
"""

# Time Series Preparation

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Setting the font sizes
plt.rcParams['axes.titlesize'] = 20  # Title font size
plt.rcParams['axes.labelsize'] = 18  # Axis label font size
plt.rcParams['xtick.labelsize'] = 16  # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 16  # Y-axis tick label size
plt.rcParams['legend.fontsize'] = 16  # Legend font size

# Download SPTL data
sptl = yf.download('SPTL', start='2014-01-01', end='2019-12-31')

# EFFR data
file_path = 'Search.xlsx'

df = pd.read_excel(file_path)

effective_rate_column = 'Rate (%)'

df.set_index(df.columns[0], inplace=True)

effr_daily_adjusted = df[effective_rate_column] / 252

# Converting the index to datetime to ensure compatibility
effr_daily_adjusted.index = pd.to_datetime(effr_daily_adjusted.index)

# Reindexing EFFR data to match SPTL data dates, forward-filling missing values
effr_daily_adjusted = effr_daily_adjusted.reindex(sptl.index, method='ffill')
print(effr_daily_adjusted.head())

# Calculating daily returns for SPTL
sptl['Daily Return'] = sptl['Adj Close'].pct_change()

# Calculating daily excess return
sptl['Excess Return'] = sptl['Daily Return'] - effr_daily_adjusted

plt.figure(figsize=(14, 10))

# Plotting SPTL Daily Returns
plt.subplot(3, 1, 1)
plt.plot(sptl.index, sptl['Daily Return'], label='SPTL Daily Return')
plt.title('SPTL Daily Returns')
plt.legend()

# Plotting EFFR Daily Rate
plt.subplot(3, 1, 2)
plt.plot(sptl.index, effr_daily_adjusted, label='EFFR Daily Rate', color='orange')
plt.title('EFFR Daily Rate')
plt.legend()

# Plotting Excess Return
plt.subplot(3, 1, 3)
plt.plot(sptl.index, sptl['Excess Return'], label='Excess Return', color='green')
plt.title('Excess Returns of SPTL')
plt.legend()

plt.tight_layout()
plt.show()

