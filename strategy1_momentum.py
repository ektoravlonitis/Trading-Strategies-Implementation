# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:09:03 2024

@author: he_98
"""

# TRADING STRATEGIES

# 1 - Momentum Strategy

import yfinance as yf
import numpy as np
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
#effr_daily_adjusted = 1.5* effr_daily_adjusted

# Converting the index to datetime
effr_daily_adjusted.index = pd.to_datetime(effr_daily_adjusted.index)

# Reindexing EFFR data to match SPTL data dates, forward-filling missing values
effr_daily_adjusted = effr_daily_adjusted.reindex(sptl.index, method='ffill')

# Calculate daily returns for SPTL
sptl['Daily Return'] = sptl['Adj Close'].pct_change()

def calculate_sharpe_ratio(daily_pnl, effr_daily_adjusted):
    excess_returns = daily_pnl - effr_daily_adjusted
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)  # Annualizing
    return annualized_sharpe_ratio


# Initial setup
V0 = 200000  # Initial capital
L = 10  # Leverage

amplification_factor = 100

# Splitting data into training (70%) and testing (30%) sets
split_index = int(len(sptl) * 0.7)
train = sptl.iloc[:split_index]
test = sptl.iloc[split_index:]

# Optimize Parameters

best_cumulative_pnl = -np.inf
best_sharpe_ratio = -np.inf
best_window = None
best_holding_period = None

# Defining 'momentum_windows' and 'holding_periods'
momentum_windows = [5, 10, 15, 20, 25, 30]
holding_periods = [1, 5, 10, 15, 20]

for window in momentum_windows:
    for holding_period in holding_periods:
        temp_train = train.copy()  # Working on a copy of the training data
        
        # Calculating rolling momentum based on current 'window'
        temp_train['Momentum'] = temp_train['Daily Return'].rolling(window=window).mean()
        temp_train['Amplified Momentum'] = temp_train['Momentum'].apply(lambda x: np.nan if pd.isna(x) else max(-1, min(1, amplification_factor * x)))

        # Initializing signals with the first value and setting the initial days since last change
        initial_signal = temp_train['Amplified Momentum'].iloc[window - 1]
        temp_train['Signal'] = initial_signal
        
        days_since_last_change = 0
        
        # Looping through the DataFrame to assign signals considering the holding period
        for i in range(window, len(temp_train)):
            days_since_last_change += 1
            if days_since_last_change >= holding_period:
                new_signal = temp_train['Amplified Momentum'].iloc[i]

                if new_signal != temp_train['Signal'].iloc[i - 1]:
                    temp_train['Signal'].iloc[i] = new_signal
                    days_since_last_change = 0
                else:
                    temp_train['Signal'].iloc[i] = temp_train['Signal'].iloc[i - 1]
            else:
                temp_train['Signal'].iloc[i] = temp_train['Signal'].iloc[i - 1]
        
        temp_train['Theta_t'] = V0 * L * temp_train['Signal']
        temp_train['Daily_PnL'] = (temp_train['Daily Return'] - effr_daily_adjusted.reindex(temp_train.index, method='ffill')) * temp_train['Theta_t']
        temp_train['Cumulative_PnL'] = temp_train['Daily_PnL'].cumsum()
        temp_train['Excess_Daily_Return'] = temp_train['Daily_PnL'] / (V0 * L) - effr_daily_adjusted.reindex(temp_train.index, method='ffill')
        daily_returns = temp_train['Excess_Daily_Return'].dropna()  # Removing NaN values

        # Calculating Sharpe Ratio and updating best parameters if needed
        if len(daily_returns) > 0:
            sharpe_ratio = calculate_sharpe_ratio(temp_train['Daily_PnL'], effr_daily_adjusted.reindex(temp_train.index, method='ffill'))
            if sharpe_ratio > best_sharpe_ratio:
                best_sharpe_ratio = sharpe_ratio
                best_window = window
                best_holding_period = holding_period
        

print(f"Best Momentum Window: {best_window} Days, Best Holding Period: {best_holding_period} Days with Sharpe Ratio: {best_sharpe_ratio}")

# Calculating Momentum and Signals using the best window, without a for-loop
test['Momentum'] = test['Daily Return'].rolling(window=best_window).mean()

test['Amplified Momentum'] = test['Momentum'].apply(lambda x: -1 if pd.isna(x) else max(-1, min(1, amplification_factor * x)))


# Setting the holding period
holding_period = best_holding_period
window = best_window

# Initializing the signal column with the first value based on initial momentum
initial_signal = test['Amplified Momentum'].iloc[0]

test['Signal'] = initial_signal

# Initializing the day since last signal change
days_since_last_change = 0

# Looping through the momentum to assign signals considering the holding period
for i in range(1, len(test)):
    days_since_last_change += 1
    
    if days_since_last_change >= holding_period:
        # Updating the signal based on the momentum after the holding period
        test['Signal'].iloc[i] = test['Amplified Momentum'].iloc[i]
        days_since_last_change = 0
    else:
        test['Signal'].iloc[i] = test['Signal'].iloc[i - 1]

# Calculating Theta_t based on constant V0 and leverage
test['Theta_t'] = V0 * L * test['Signal']

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

# Calculating the daily change in this ratio, and taking the absolute value
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


# Calculating the 30-day moving average for dollar turnover
test['MA_Dollar_Turnover'] = test['Daily_Dollar_Turnover'].rolling(window=30).mean()

# Calculating the 30-day moving average for unit turnover
test['MA_Unit_Turnover'] = test['Daily_Unit_Turnover'].rolling(window=30).mean()


# Calculating the 30-day standard deviation (volatility) of daily returns
test['Volatility'] = test['Daily Return'].rolling(window=30).std()

# Plotting the 30-Day Moving Average Dollar Turnover and Volatility
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
ax2.plot(test.index, test['Volatility'], color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('30-Day Moving Average Dollar Turnover and Volatility')
plt.show()


# TRAIN set

train['Momentum'] = train['Daily Return'].rolling(window=best_window).mean()

train['Amplified Momentum'] = train['Momentum'].apply(lambda x: -1 if pd.isna(x) else max(-1, min(1, amplification_factor * x)))

# Setting the holding period
holding_period = best_holding_period
window = best_window

# Initializing the signal column with the first value based on initial momentum
initial_signal = train['Amplified Momentum']

train['Signal'] = initial_signal

days_since_last_change = 0

# Looping through the momentum to assign signals considering the holding period
for i in range(1, len(train)):
    days_since_last_change += 1
    
    if days_since_last_change >= holding_period:
        train['Signal'].iloc[i] = train['Amplified Momentum'].iloc[i]
        days_since_last_change = 0
    else:
        # If within the holding period, retain the previous signal
        train['Signal'].iloc[i] = train['Signal'].iloc[i - 1]


# Calculating Theta_t based on constant V0 and leverage
train['Theta_t'] = V0 * L * train['Signal']

# QUESTION C

# Calculating daily PnL based on Theta_t and adjusted daily return
test['Daily_PnL'] = (test['Daily Return'] - effr_daily_adjusted.reindex(test.index, method='ffill')) * test['Theta_t']

# Initializing V_total with the initial capital and preparing Delta_V_cap
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


# TRAIN set

# Calculating daily PnL based on Theta_t and adjusted daily return
train['Daily_PnL'] = (train['Daily Return'] - effr_daily_adjusted.reindex(train.index, method='ffill')) * train['Theta_t']

# 3. PERFORMANCE INDICATORS

# (a)

import numpy as np

def calculate_sharpe_ratio(daily_pnl, effr_daily_adjusted):
    excess_returns = daily_pnl - effr_daily_adjusted
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)  # Annualizing
    return annualized_sharpe_ratio
    #return sharpe_ratio

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
drawdowns = (aggregate_net_returns - peak) / peak
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

# Calculating Performance Metrics for the Training Set

# Sharpe Ratio for the Training Set
sharpe_ratio_train = calculate_sharpe_ratio(train['Daily_PnL'], effr_daily_adjusted.reindex(train.index, method='ffill'))

# Sortino Ratio for the Training Set
sortino_ratio_train = calculate_sortino_ratio(train['Daily_PnL'], effr_daily_adjusted.reindex(train.index, method='ffill'))

train['Daily Returns'] = train['Daily_PnL'] / train['Theta_t'].shift().abs()
train['Daily Returns'].fillna(0, inplace=True)

# Calculating aggregate net returns
aggregate_net_returns_train = train['Daily Returns'].cumsum()

# Calculating Maximum Drawdown
peak = aggregate_net_returns_train.expanding(min_periods=1).max()
drawdowns = (aggregate_net_returns_train - peak) / peak
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
combined_rolling_sr.fillna(method='ffill', inplace=True)

train_rolling_sr = combined_rolling_sr.loc[train.index]
test_rolling_sr = combined_rolling_sr.loc[test.index]

# Plotting the Continuous Rolling Sharpe Ratio Over Time 
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
