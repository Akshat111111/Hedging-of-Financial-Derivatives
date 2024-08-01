# Asset Correlation Analysis Streamlit App

## Project Overview

This project aims to analyze the correlations between different assets using historical price data. The analysis includes calculating daily returns, performing correlation analysis, and visualizing the results using a Streamlit app.

## Dataset

The dataset is synthetic and contains the following columns:
- Date
- Asset_A
- Asset_B
- Asset_C
- Asset_D

## EDA (Exploratory Data Analysis)

The EDA process includes:
1. Loading and displaying the dataset.
2. Checking for missing values.
3. Statistical description of the dataset.
4. Visualizing the time series of each asset.
5. Plotting the correlation matrix of asset prices.

## Correlation Analysis

The correlation analysis includes:
1. Calculating daily returns for each asset.
2. Plotting the correlation matrix of daily returns.
3. Calculating and plotting rolling correlations between selected assets.

## Requirements

- pandas
- numpy==1.22.0
- matplotlib
- seaborn
- streamlit

