# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:32:11 2024

@author: he_98
"""

# TRADING STRATEGIES

# 3 - Volatility Threshold Strategy

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Setting the font sizes
plt.rcParams['axes.titlesize'] = 20  # Title font size
plt.rcParams['axes.labelsize'] = 18  # Axis label font size
plt.rcParams['xtick.labelsize'] = 16  # X-axis tick label size
plt.rcParams['ytick.labelsize'] = 16  # Y-axis tick label size
plt.rcParams['legend.fontsize'] = 16  # Legend font size

# Downloading SPTL data
sptl = yf.download('SPTL', start='2014-01-01', end='2019-12-31')
sptl['Daily Return'] = sptl['Adj Close'].pct_change()

# EFFR data
file_path = 'Search.xlsx'

df = pd.read_excel(file_path)

effective_rate_column = 'Rate (%)'

df.set_index(df.columns[0], inplace=True)

effr_daily_adjusted = df[effective_rate_column] / 252
#effr_daily_adjusted = 1.5* effr_daily_adjusted

# Converting the index to datetime to ensure compatibility
effr_daily_adjusted.index = pd.to_datetime(effr_daily_adjusted.index)

# Reindexing EFFR data to match SPTL data dates, forward-filling missing values
effr_daily_adjusted = effr_daily_adjusted.reindex(sptl.index, method='ffill')

# Calculating the annualized volatility of daily returns over a 30-day rolling period
sptl['Volatility'] = sptl['Daily Return'].rolling(window=30).std() * np.sqrt(252)

# Initial setup
V0 = 200000  # Initial capital
L = 10  # Leverage

# Splitting data into training (70%) and testing (30%) sets
split = int(len(sptl) * 0.7)
train, test = sptl[:split], sptl[split:]

# Aligning risk-free rate series with the dataset
train_rf_rate = effr_daily_adjusted.reindex(train.index, method='ffill')
test_rf_rate = effr_daily_adjusted.reindex(test.index, method='ffill')

def calculate_sharpe_ratio(returns, risk_free_rate):
    excess_returns = returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)  # Annualizing
    return annualized_sharpe_ratio

# Finding the best volatility threshold based on Sharpe Ratio for the training set
best_sharpe_ratio = -np.inf
best_threshold = None
vol_thresholds = np.linspace(train['Volatility'].min(), train['Volatility'].max(), 100)

for threshold in vol_thresholds:
    # Defining signals based on threshold
    train['Signal'] = np.where(train['Volatility'] > threshold, -1, 
                                   np.where(train['Volatility'] < threshold, 1, 0))
    # Calculating daily pnl based on signals
    train['Theta_t'] = V0 * L * train['Signal']
    train['Daily_PnL'] = (train['Daily Return'] - effr_daily_adjusted.reindex(train.index, method='ffill')) * train['Theta_t']

    # Calculating Sharpe Ratio
    sr = calculate_sharpe_ratio(train['Daily_PnL'], train_rf_rate)
    if sr > best_sharpe_ratio:
        best_sharpe_ratio = sr
        best_threshold = threshold


# Applying best threshold to the test set
test['Signal'] = np.where(test['Volatility'] > best_threshold, -1, 
                              np.where(test['Volatility'] < best_threshold, 1, 0))

test['Theta_t'] = V0*L*test['Signal']
test['Daily PnL'] = test['Theta_t'] * (test['Daily Return'] - test_rf_rate)
test['Cumulative PnL'] = test['Daily PnL'].cumsum()

# Plotting the adjusted Cumulative PnL for the test set
plt.figure(figsize=(14, 7))
plt.plot(test.index, test['Cumulative PnL'], label='Cumulative PnL Using Best Volatility Threshold', color='blue')
plt.title(f'Adjusted Cumulative PnL Using Best Volatility Threshold: {best_threshold}')
plt.xlabel('Date')
plt.ylabel('Cumulative PnL ($)')
plt.legend()
plt.show()

# QUESTION B

# Calculating upper and lower bounds
upper_bound = [V0 * L] * len(test)
lower_bound = [-V0 * L] * len(test)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(test.index, test['Theta_t'], label='Theta_t Over Time', color='green')
plt.plot(test.index, upper_bound, label='Upper Bound', linestyle='--', color='red')
plt.plot(test.index, lower_bound, label='Lower Bound', linestyle='--', color='blue')
plt.title('Position Size Over Time (\Theta_t) with Bounds')
plt.xlabel('Date')
plt.ylabel('Position Size ($)')
plt.legend()
plt.show()

# Calculating the daily absolute change in position size to find daily turnover
test['Daily_Dollar_Turnover'] = abs(test['Theta_t'].diff())

# Calculating the cumulative sum of these absolute changes to get cumulative turnover
test['Cumulative_Dollar_Turnover'] = test['Daily_Dollar_Turnover'].cumsum()

# Plotting the cumulative dollar turnover
plt.figure(figsize=(14, 7))
plt.plot(test.index, test['Cumulative_Dollar_Turnover'], label='Cumulative Dollar Turnover', color='orange')
plt.title('Cumulative Dollar Turnover Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Dollar Turnover ($)')
plt.legend()
plt.show()


# Calculating the ratio of position size over adjusted close price for each day
test['Theta_over_AdjClose'] = test['Theta_t'] / test['Adj Close']

# Calculating the daily change in this ratio, and take the absolute value
test['Daily_Unit_Turnover'] = abs(test['Theta_over_AdjClose'].diff())

# Calculating the cumulative sum of these daily unit turnovers to get the cumulative turnover in units
test['Cumulative_Unit_Turnover'] = test['Daily_Unit_Turnover'].cumsum()

# Plotting the cumulative turnover in units
plt.figure(figsize=(14, 7))
plt.plot(test.index, test['Cumulative_Unit_Turnover'], label='Cumulative Unit Turnover', color='blue')
plt.title('Cumulative Unit Turnover Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Units Traded')
plt.legend()
plt.show()

# Calculating the total turnover in dollars by summing the daily changes
total_dollar_turnover = test['Daily_Dollar_Turnover'].sum()
total_unit_turnover = test['Daily_Unit_Turnover'].sum()

print(f"Total Dollar Turnover: {total_dollar_turnover}")
print(f"Total Unit Turnover: {total_unit_turnover}")

# Calculating a 30-day moving average for dollar turnover
test['MA_Dollar_Turnover'] = test['Daily_Dollar_Turnover'].rolling(window=30).mean()

# Calculating a 30-day moving average for unit turnover
test['MA_Unit_Turnover'] = test['Daily_Unit_Turnover'].rolling(window=30).mean()

test['Volatility_not_annualized'] = test['Daily Return'].rolling(window=30).std()

fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'navy'
ax1.set_xlabel('Date')
ax1.set_ylabel('30-Day MA Dollar Turnover', color=color)
ax1.plot(test.index, test['MA_Dollar_Turnover'], color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='x')

ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel('30-Day Volatility of Daily Returns', color=color)
ax2.plot(test.index, test['Volatility_not_annualized'], color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('30-Day Moving Average Dollar Turnover and Volatility')
plt.show()

# QUESTION C

# Calculating daily PnL based on Theta_t and adjusted daily return
test['Daily_PnL'] = (test['Daily Return'] - effr_daily_adjusted.reindex(test.index, method='ffill')) * test['Theta_t']

test['V_total'] = V0
test['Delta_V_cap'] = 0.0

# Calculating M_t as the total margin used
test['M_t'] = abs(test['Theta_t']) / L

for i in range(1, len(test)):
    if i == 1:  # For the first day, there's no previous day to look back on, so we use V0 directly
        test.loc[test.index[i], 'V_total'] = V0
    else:
        test.loc[test.index[i], 'V_total'] = test.loc[test.index[i-1], 'V_total'] + test.loc[test.index[i-1], 'Daily_PnL'] + test.loc[test.index[i-1], 'Delta_V_cap']

    # Calculating available capital for the money market, subtracting the margin from the total portfolio value
    available_capital = test.loc[test.index[i], 'V_total'] - test.loc[test.index[i], 'M_t']

    # Calculating the growth of the money-market account for the day
    test.loc[test.index[i], 'Delta_V_cap'] = available_capital * effr_daily_adjusted.loc[test.index[i]]

test['Cumsum_Delta_V'] = test['Daily_PnL'].cumsum()
test['Cumsum_Delta_V_cap'] = test['Delta_V_cap'].cumsum()
test['Cumsum_Delta_V_total'] = test['Cumsum_Delta_V'] + test['Cumsum_Delta_V_cap']

# Plotting all three cumulative sums after accurate calculations
plt.figure(figsize=(14, 7))
plt.plot(test.index, test['Cumsum_Delta_V'], label='Cumulative ΔV (Trading PnL)', color='blue')
plt.plot(test.index, test['Cumsum_Delta_V_cap'], label='Cumulative ΔV^cap (Money-Market Growth)', color='green')
plt.plot(test.index, test['Cumsum_Delta_V_total'], label='Cumulative ΔV^total (Total Portfolio Change)', color='red')
plt.title('Cumulative Changes in Portfolio Value with Accurate Calculations')
plt.xlabel('Date')
plt.ylabel('Cumulative Value ($)')
plt.legend(loc='best')
plt.show()


# 3. PERFORMANCE INDICATORS

# (a)

import numpy as np

def calculate_sharpe_ratio(daily_pnl, effr_daily_adjusted):
    excess_returns = daily_pnl - effr_daily_adjusted
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)  # Annualizing
    return annualized_sharpe_ratio

def calculate_sortino_ratio(daily_pnl, effr_daily_adjusted):
    excess_returns = daily_pnl - effr_daily_adjusted
    downside_returns = excess_returns[excess_returns < 0]
    sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) if np.std(downside_returns) > 0 else np.nan
    return sortino_ratio

# Aligning effr_daily_adjusted with the test dataset
aligned_effr_daily_adjusted = effr_daily_adjusted.reindex(test.index, method='ffill')

# Sharpe Ratio
sharpe_ratio = calculate_sharpe_ratio(test['Daily_PnL'], aligned_effr_daily_adjusted)

# Sortino Ratio
sortino_ratio = calculate_sortino_ratio(test['Daily_PnL'], aligned_effr_daily_adjusted)

test['Daily Returns'] = test['Daily_PnL'] / test['Theta_t'].shift().abs()
test['Daily Returns'].fillna(0, inplace=True)

# Calculating aggregate net returns
aggregate_net_returns = test['Daily Returns'].cumsum()

# Calculating Maximum Drawdown
peak = aggregate_net_returns.expanding(min_periods=1).max()
drawdowns = (aggregate_net_returns - peak) / peak.where(peak > 0)
drawdowns.fillna(0, inplace=True)
max_drawdown = drawdowns.min()

# Calculating Annualized Return based on the average of daily returns
annualized_return = test['Daily Returns'].mean() * 252

# Calculating Calmar Ratio
calmar_ratio = annualized_return / abs(max_drawdown)

print("Test Set Performance Metrics:")
print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Sortino Ratio: {sortino_ratio}")
print(f"Maximum Drawdown: {max_drawdown}")
print(f"Calmar Ratio: {calmar_ratio}")

# TRAIN set

train['Signal'] = np.where(train['Volatility'] > best_threshold, -1, 
                              np.where(train['Volatility'] < best_threshold, 1, 0))

# Calculating Theta_t for the training set based on the signal
train['Theta_t'] = V0 * L * train['Signal']

# Calculating Daily PnL for the training set
train['Daily_PnL'] = (train['Daily Return'] - effr_daily_adjusted.reindex(train.index, method='ffill')) * train['Theta_t']

# Calculating Cumulative PnL for the training set
train['Cumulative_PnL'] = train['Daily_PnL'].cumsum()

# Calculating Performance Metrics for the Training Set

# Sharpe Ratio for the Training Set
sharpe_ratio_train = calculate_sharpe_ratio(train['Daily_PnL'], effr_daily_adjusted.reindex(train.index, method='ffill'))

# Sortino Ratio for the Training Set
sortino_ratio_train = calculate_sortino_ratio(train['Daily_PnL'], effr_daily_adjusted.reindex(train.index, method='ffill'))

train['Daily Returns'] = train['Daily_PnL'] / train['Theta_t'].shift().abs()
train['Daily Returns'] = train['Daily Returns'].replace([np.inf, -np.inf], np.nan)
train['Daily Returns'].fillna(0, inplace=True)

# Calculating aggregate net returns
aggregate_net_returns_train = train['Daily Returns'].cumsum()

# Calculating Maximum Drawdown
peak = aggregate_net_returns_train.expanding(min_periods=1).max()
drawdowns = (aggregate_net_returns_train - peak) / peak.where(peak > 0)
drawdowns.fillna(0, inplace=True)
max_drawdown_train = drawdowns.min()

# Calculating Annualized Return based on the average of daily returns
annualized_return_train = train['Daily Returns'].mean() * 252

# Calculating Calmar Ratio
calmar_ratio_train = annualized_return_train / abs(max_drawdown_train)

print("Training Set Performance Metrics:")
print(f"Sharpe Ratio: {sharpe_ratio_train}")
print(f"Sortino Ratio: {sortino_ratio_train}")
print(f"Maximum Drawdown: {max_drawdown_train}")
print(f"Calmar Ratio: {calmar_ratio_train}")

# (b)
# Defining the function to calculate the rolling Sharpe Ratio
def rolling_sharpe_ratio(returns, risk_free_rate, window=252):
    # Calculating rolling mean of returns
    rolling_mean = returns.rolling(window=window).mean()
    # Calculating rolling standard deviation of returns
    rolling_std = returns.rolling(window=window).std()
    # Calculating rolling Sharpe Ratio
    rolling_sharpe = (rolling_mean - risk_free_rate) / rolling_std
    return rolling_sharpe

# Combining the training and test sets for a continuous series
combined_pnl = pd.concat([train['Daily_PnL'], test['Daily_PnL']])

# Calculating the combined rolling Sharpe Ratio
combined_rolling_sr = rolling_sharpe_ratio(combined_pnl, effr_daily_adjusted.reindex(sptl.index, method='ffill'), window=252)

# Splitting the combined rolling SR for separate plotting
train_rolling_sr = combined_rolling_sr[:len(train)]
test_rolling_sr = combined_rolling_sr[len(train):]

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(train_rolling_sr.index, train_rolling_sr, label='Training Set Rolling Sharpe Ratio', color='blue')
plt.plot(test_rolling_sr.index, test_rolling_sr, label='Test Set Rolling Sharpe Ratio', color='orange')
plt.title('Continuous Rolling Sharpe Ratio Over Time')
plt.xlabel('Date')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.show()


# (c)

def calculate_drawdown(cumulative_pnl):
    peak = cumulative_pnl.expanding(min_periods=1).max()
    drawdown = peak - cumulative_pnl 
    return drawdown

def calculate_rolling_volatility(daily_return, window=90):
    return daily_return.rolling(window=window).std() * np.sqrt(window)

underlying_returns = test['Daily Return']

strategy_drawdown = calculate_drawdown(aggregate_net_returns)
strategy_rolling_volatility = calculate_rolling_volatility(underlying_returns)

# Plotting
fig, ax1 = plt.subplots(figsize=(14, 7))

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Drawdown', color=color)
ax1.plot(strategy_drawdown.index, strategy_drawdown, color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('90-day Rolling Volatility', color=color)
ax2.plot(strategy_rolling_volatility.index, strategy_rolling_volatility, color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('Drawdown and 90-day Rolling Volatility')
plt.show()

# Risk investigation - VAR

confidence_level = 0.95

# Calculating VaR for the training set
train_var_95 = train['Daily_PnL'].quantile(1 - confidence_level)

# Calculating VaR for the test set
test_var_95 = test['Daily_PnL'].quantile(1 - confidence_level)

# Printing the results
print(f"95% Confidence Level VaR for Training Set: {train_var_95}")
print(f"95% Confidence Level VaR for Test Set: {test_var_95}")
