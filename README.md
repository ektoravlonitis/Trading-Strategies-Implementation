# Trading-Strategies-Implementation
Algorithmic Trading Project

## Introduction
This project focuses on defining and implementing trading strategies using the SPTL ETF and the Effective Federal Funds Rate (EFFR) as a benchmark for the risk-free rate.
The goal of the analysis is to explore different trading strategies, starting with a significant amount of initial capital to try and maximize returns while dealing with the usual risks and market fluctuations.
The main part of the project involved analyzing three specific leveraged trading strategies. These are tested using a dataset divided into two parts: 70% for training and 30% for testing. The results are compared by performing tests under different financial conditions. This report outlines the methods used, presents the results, and offers a critical review of what was found. The aim is to understand algorithmic trading and its effects on today’s financial markets.

## Strategies
1. strategy1_momentum.py - Momentum Leveraged Trading Strategy: This strategy leverages the concept of momentum, predicting that the SPTL ETF’s price will continue to move in its current direction after significant price changes.
2. strategy2_ml_mean_reversion.py - Machine Learning-Enhanced Mean Reversion Trading Strategy: This approach integrates machine learning to predict market movements and produce signals based on mean reversion.
3. strategy3_volatility_threshold.py - Volatility Threshold-based Leveraged Trading Strategy: Focused on market volatility, this strategy uses a calculated volatility threshold to guide trading decisions.
