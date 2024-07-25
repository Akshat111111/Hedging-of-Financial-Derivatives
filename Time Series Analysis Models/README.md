# **Time Series Analysis**

### 🎯 Goal
To analyze and model time series data using various statistical models to understand patterns, trends, and make predictions.

### Purpose
The purpose of this project is to explore time series data using autoregressive models, moving average models, and combined autoregressive moving average models. This analysis helps in understanding the underlying patterns in the data and making informed predictions.

### 🧵 Dataset
The datasets used in this project is: 
- /kaggle/input/stock-time-series-20050101-to-20171231
- /kaggle/input/historical-hourly-weather-data
- /kaggle/input/dummy-truck-sales-for-time-series/Truck_sales.csv

### 🧾 Description
The stock dataset contains historical stock price data, including features such as Open, High, Low, Close prices, and Volume. The weather dataset includes historical hourly weather data with features such as temperature, humidity, and wind speed. These datasets are used to train and evaluate the performance of the time series models.

### 🚀 Models Implemented
1. AR (Autoregressive)
2. MA (Moving Average)
3. ARMA (Autoregressive Moving Average)
4. ARIMA (Auto Regressive Integrated Moving Average)
5. SARIMA (Seasonal Autoregressive Moving Average)

### 📚 Libraries Needed
- Statsmodels: For building and analyzing statistical models.
- Pandas: For data manipulation and analysis.
- NumPy: For numerical computations and array operations.
- Matplotlib: For plotting and visualizing data.
- Seaborn: For enhanced data visualization.

### 📊 Data Visualization
- Stock Prices Visualization: Plotting the historical stock prices to understand the trends and patterns.
- Weather Data Visualization: Plotting temperature, humidity, and wind speed over time to observe seasonal variations.
- Truck Data Visualization: Plotting monthly sales data for trucks from a specific company throughout the year.

### 📢 Conclusion
| Model | RMSE |
|-------|------|
| AR | 7.21 |
| MA | 11.34 |
| ARMA | 38038336.51 |
| ARIMA | 209.161604  |
| SARIMA |  84.272155 |



## ✒️ Contributor
- Name: Khushi Kalra
- Github: https://www.github.com/abckhush
