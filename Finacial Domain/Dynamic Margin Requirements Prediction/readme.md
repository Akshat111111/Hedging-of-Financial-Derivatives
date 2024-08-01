# Dynamic Margin Requirements Prediction

## Project Overview

This project aims to develop a predictive model for dynamic margin requirements using deep learning techniques. We focus on forecasting margin requirements based on historical financial indicators. The dataset includes stock index return, interest rate, volatility index, trading volume, credit spread, and the margin requirement.

## Dataset

The dataset is synthetic and contains the following columns:
- Date
- Stock_Index_Return
- Interest_Rate
- Volatility_Index
- Trading_Volume
- Credit_Spread
- Margin_Requirement

## EDA (Exploratory Data Analysis)

The EDA process includes:
1. Loading and displaying the dataset.
2. Checking for missing values.
3. Statistical description of the dataset.
4. Visualizing the time series of each feature.
5. Plotting the correlation matrix.
6. Plotting histograms and pairplots to understand the distribution and relationships between variables.

## Deep Learning Models

We implemented three deep learning algorithms:
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

