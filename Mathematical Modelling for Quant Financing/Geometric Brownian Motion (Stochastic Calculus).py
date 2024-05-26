# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

"""
This script demonstrates the application of Geometric Brownian Motion (GBM) for stock price prediction

The script performs the following steps:
1. Load and preprocess the stock price dataset.
2. Perform exploratory data analysis (EDA) on the dataset.
3. Fit a Gaussian HMM to the log returns of the stock prices.
4. Predict the stock price using a Geometric Brownian Motion (GBM)

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
"""


# Set the aesthetics for the plots
# This line configures seaborn to enhance the visual appeal of the plots.
sns.set(style="whitegrid")

# Load the dataset
# Here, we're loading a CSV file into a pandas DataFrame. This is a common way to load data for analysis.
data = pd.read_csv('Dataset/US Stock Market Data & Technical Indicators/AAPL.csv')

# Rename the 'Close(t)' column to 'Close' for simplicity
# Renaming columns can make them easier to reference in your code.
df_Stock = data.rename(columns={'Close(t)':'Close'})

# Display the first few rows of the dataset
# This is useful for getting a quick look at your data to ensure it's loaded correctly.
print(df_Stock.head())

# Display the last 5 rows of the dataset
# Similar to head(), but shows the end of the DataFrame.
print(df_Stock.tail(5))

# Display the shape of the dataset (number of rows and columns)
# This gives you an idea of the dataset's size.
print(df_Stock.shape)

# Display the column names of the dataset
# Knowing the column names is essential for data manipulation and analysis.
print(df_Stock.columns)

# Display basic information about the dataset
# The info() method provides a concise summary of the DataFrame, including the data type of each column.
print(df_Stock.info())

# Check for missing values in the dataset
# Missing data can significantly affect your analysis, so it's crucial to identify and handle it appropriately.
print(df_Stock.isnull().sum())

# Summary statistics for numerical features
# The describe() method gives a quick overview of the statistical distribution of numerical columns.
print(df_Stock.describe())

# Summary statistics for categorical features
# Including 'object' in describe() allows you to see statistics for non-numeric columns.
print(df_Stock.describe(include=['object']))

# Plot histograms for each numerical feature
# Histograms are a great way to visualize the distribution of your data.
df_Stock.hist(bins=50, figsize=(20,15))
plt.show()

# Select only numeric columns for correlation
# Correlation matrices are a powerful tool for understanding the relationships between numerical variables.
numeric_df = df_Stock.select_dtypes(include=[np.number])

# Compute the correlation matrix
# This step is crucial for feature selection and understanding the relationships between variables.
corr = numeric_df.corr()

# Generate a heatmap to visualize the correlation matrix
# Heatmaps make it easier to identify highly correlated variables at a glance.
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

# Plot the closing prices over time
# Time series plots are essential for financial data analysis, allowing you to observe trends, cycles, and volatility.
df_Stock['Close'].plot(figsize=(10, 7))
plt.title("Stock Price", fontsize=17)
plt.ylabel('Price', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()

# Calculate the mean price of the stock
# The mean gives you a sense of the "central" value of the dataset.
mean_price = df_Stock['Close'].mean()
print("Mean price:", mean_price)

# Calculate the standard deviation of the stock price
# Standard deviation measures the amount of variation or dispersion in a set of values.
std_deviation = df_Stock['Close'].std()
print("Standard Deviation:", std_deviation)

# Calculate logarithmic returns
# Log returns are often used in financial analysis because they are time-additive and can be interpreted as continuously compounded returns.
log_returns = np.log(df_Stock['Close'] / df_Stock['Close'].shift(1))

# Calculate volatility as the standard deviation of log returns
# Volatility is a measure of the price movements of a stock or the stock market as a whole.
volatility = log_returns.std() * np.sqrt(252)  # Annualizing the volatility
print("Annualized Volatility:", volatility)

# Calculate the variance of the stock price
# Variance measures how far a set of numbers is spread out from their average value.
variance = df_Stock['Close'].var()
print("Variance:", variance)

# Calculate the median price of the stock
# The median is the value separating the higher half from the lower half of a data sample.
median_price = df_Stock['Close'].median()
print("Median price:", median_price)

# Describe to get a summary of statistics
# This provides a quick overview of the central tendency, dispersion, and shape of the dataset's distribution.
summary_stats = df_Stock['Close'].describe()
print("Summary Statistics:\n", summary_stats)

# Plot the closing prices over time
# This is a repeat of an earlier plot, reinforcing the importance of visualizing your data.
plt.figure(figsize=(10, 7))
plt.plot(df_Stock['Close'], label='Close Price')
plt.title('Stock Price Over Time')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Assuming 'Close' is the column with closing prices
# Here we calculate the log returns, which are useful for various financial analyses.
df_Stock['Log_Returns'] = np.log(df_Stock['Close'] / df_Stock['Close'].shift(1))

# Calculate the drift coefficient (mean of log returns)
# The drift is the expected value of the returns, which can be used in pricing models and risk management.
drift_coefficient = df_Stock['Log_Returns'].mean()
print("Drift coefficient (mu):", drift_coefficient)

# Annualize the drift coefficient
# Annualizing helps compare the drift coefficient across different time periods.
annual_drift = drift_coefficient * 252
print("Annualized Drift Coefficient:", annual_drift)

# Plotting the closing prices with the drift
# This plot can help visualize how the drift coefficient affects the stock price over time.
df_Stock['Close'].plot(figsize=(10, 7))
plt.title("Stock Price with Drift Coefficient: {:.5f}".format(annual_drift), fontsize=17)
plt.ylabel('Price', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()

# Parameters for Geometric Brownian Motion (GBM)
# GBM is a mathematical model for predicting the future price movements of financial instruments.
S0 = 6.60        # initial stock price
T = 15           # time horizon in years
mu = annual_drift       # drift coefficient
sigma = volatility     # volatility
dt =  0.00396825396      # time step (1 trading day)
N = round(T/dt) # number of time steps
t = np.linspace(0, T, N) # vector of times

def generate_gbm(S0, mu, sigma, dt, N):
    """
    Generate stock prices using Geometric Brownian Motion (GBM).
    
    Parameters:
    S0 (float): Initial stock price
    mu (float): Drift coefficient
    sigma (float): Volatility
    dt (float): Time step
    N (int): Number of time steps
    
    Returns:
    np.ndarray: Simulated stock prices
    """
    W = np.random.standard_normal(size = N)
    W = np.cumsum(W)*np.sqrt(dt)  # standard Brownian motion
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)  # geometric Brownian motion
    return S

# Generate stock prices using GBM
# This function simulates possible future stock price paths based on the specified parameters.
stock_price = generate_gbm(S0, mu, sigma, dt, N)

# Plot the simulated stock prices
# Visualizing the simulated stock prices can help assess the model's realism and potential future scenarios.
plt.figure(figsize=(10, 5))
plt.plot(t, stock_price)
plt.title('Geometric Brownian Motion - Stock Price')
plt.xlabel('Time in years')
plt.ylabel('Stock Price')
plt.show()
