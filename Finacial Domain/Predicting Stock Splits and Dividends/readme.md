# Predicting Stock Splits and Dividends

## Project Overview

This project aims to develop predictive models for stock splits and dividends using deep learning techniques. We focus on forecasting stock splits and dividend payouts based on historical financial indicators. The dataset includes stock price, earnings per share, P/E ratio, dividend yield, market cap, stock split, and dividend payout.

## Dataset

The dataset is synthetic and contains the following columns:
- Date
- Stock_Price
- Earnings_Per_Share
- P_E_Ratio
- Dividend_Yield
- Market_Cap
- Stock_Split (0 or 1)
- Dividend_Payout

## EDA (Exploratory Data Analysis)

The EDA process includes:
1. Loading and displaying the dataset.
2. Checking for missing values.
3. Statistical description of the dataset.
4. Visualizing the time series of each feature.
5. Plotting the correlation matrix.
6. Plotting histograms and pairplots to understand the distribution and relationships between variables.

## Deep Learning Models

We implemented three deep learning algorithms for predicting stock splits and dividends:
1. **LSTM (Long Short-Term Memory)**
2. **GRU (Gated Recurrent Unit)**
3. **Dense Neural Network (DNN)**

### LSTM Model

An LSTM model is built to capture long-term dependencies in the time series data.

### GRU Model

A GRU model is implemented as an alternative to the LSTM, focusing on reducing the computational cost.

### Dense Neural Network (DNN)

A simple DNN model is used as a baseline for comparison with the sequential models.

## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow


