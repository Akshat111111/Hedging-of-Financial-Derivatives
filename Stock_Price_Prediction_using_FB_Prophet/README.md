# Stock Price Prediction Using Facebook Prophet

Welcome to the Stock Price Prediction project! This project utilizes the Facebook Prophet model to forecast stock prices 30 days into the future. Additionally, the project involves data visualization using Plotly Express and stock analysis and forecast evaluation using Google Finance in Google Sheets.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Modeling and Forecasting](#modeling-and-forecasting)
- [Data Visualization](#data-visualization)
- [Stock Analysis & Forecast Evaluation](#stock-analysis--forecast-evaluation)
- [Conclusion](#conclusion)

## Introduction

This project aims to predict the future stock prices of a chosen company using the Facebook Prophet forecasting tool. The forecast is extended to 30 days beyond the last available data point. The data used in this project can be obtained from Yahoo Finance for the desired date range.

## Project Overview

1. **Creating a Facebook Prophet Model & Forecasting**: Build and train a Facebook Prophet model to predict stock prices for the next 30 days.
2. **Data Visualization using Plotly Express**: Create interactive visualizations to represent the stock data and forecast results.
3. **Stock Analysis & Forecast Evaluation**: Use Google Finance in Google Sheets for further analysis and evaluation of the forecast accuracy.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- `pandas`
- `prophet` (formerly `fbprophet`)
- `plotly`
- `yfinance`

You can install these dependencies using pip:

    ```bash
    pip install pandas prophet plotly yfinance

## Usage

**Data Collection:** Fetch the stock data from Yahoo Finance. You can use any dataset for the desired period.

**Data Preparation:** Prepare the data for modeling by formatting it appropriately for Facebook Prophet.

**Model Training and Forecasting:** Train the Facebook Prophet model on the prepared data and forecast the stock prices for the next 30 days.

**Visualization:** Use Plotly Express to create visualizations of the historical stock data and the forecasted values.

**Analysis and Evaluation:** Analyze the stock performance and evaluate the forecast using Google Finance in Google Sheets.

## Data

The data for this project is sourced from Yahoo Finance. You can download the historical stock data for any company for your desired date range. For example, data could span from March 15, 2020, to March 15, 2021.

## Modeling and Forecasting
Using the Facebook Prophet model, this project involves the following steps:

- Import the data and preprocess it.
- Create and configure the Prophet model.
- Fit the model to the data.
- Generate future dates and predict the stock prices for those dates.
- Evaluate the model's performance.

## Data Visualization
Interactive visualizations are created using Plotly Express to help understand the stock trends and the model's forecasts. The visualizations include:

- Historical stock prices
- Forecasted stock prices
- Comparison between actual and predicted prices

## Stock Analysis & Forecast Evaluation
Further analysis and evaluation of the forecast can be performed using Google Finance in Google Sheets. This includes:

- Importing the forecasted data into Google Sheets
- Comparing forecasted prices with actual prices (if available)
- Calculating evaluation metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE)

You can use the following formula in Google Sheets to retrieve historical stock prices:

    ```bash
    =GOOGLEFINANCE("STOCK_SYMBOL","price",DATE(START_YEAR,START_MONTH,START_DAY),DATE(END_YEAR,END_MONTH,END_DAY))



For example:



    ```bash
    =GOOGLEFINANCE("TSLA","price",DATE(2020,3,15),DATE(2021,3,15))
This will provide the date and closing stock price for the specified period.

## Conclusion

This project provides a comprehensive approach to forecasting stock prices using Facebook Prophet. It includes data collection, model training, visualization, and evaluation. The methodologies used can be applied to any stock data for similar predictive analysis.