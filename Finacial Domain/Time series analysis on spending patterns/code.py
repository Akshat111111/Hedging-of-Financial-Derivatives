import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

# Load the dataset
file_path = '/mnt/data/sd254_users.csv'
data = pd.read_csv(file_path)

# Convert 'date' column to datetime
data['date'] = pd.to_datetime(data['date'])

# Ensure data is sorted by date
data = data.sort_values('date')

# Set the 'date' column as the index
data.set_index('date', inplace=True)

# Display the first few rows of the dataset
print(data.head())

# Resample data by month and calculate average spending score
monthly_spending = data.resample('M')['spending_score'].mean()

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(monthly_spending, marker='o', linestyle='-')
plt.title('Average Monthly Spending Score')
plt.xlabel('Date')
plt.ylabel('Average Spending Score')
plt.grid(True)
plt.show()

# Decompose the time series
result = seasonal_decompose(monthly_spending, model='additive')

# Plot the decomposition
result.plot()
plt.show()

# Perform STL decomposition
stl = STL(monthly_spending, seasonal=13)
result = stl.fit()

# Plot the STL decomposition
plt.figure(figsize=(12, 8))
plt.subplot(4, 1, 1)
plt.plot(result.observed)
plt.title('Observed')

plt.subplot(4, 1, 2)
plt.plot(result.trend)
plt.title('Trend')

plt.subplot(4, 1, 3)
plt.plot(result.seasonal)
plt.title('Seasonal')

plt.subplot(4, 1, 4)
plt.plot(result.resid)
plt.title('Residual')

plt.tight_layout()
plt.show()

# Plot the trend component
plt.figure(figsize=(12, 6))
plt.plot(result.trend, marker='o', linestyle='-')
plt.title('Trend Component')
plt.xlabel('Date')
plt.ylabel('Spending Score Trend')
plt.grid(True)
plt.show()

# Perform statistical analysis on the trend component
trend = result.trend.dropna()
slope, intercept, r_value, p_value, std_err = stats.linregress(trend.index.astype(int), trend.values)
print(f'Trend slope: {slope}')
print(f'R-squared: {r_value**2}')

# Detect anomalies using the residual component
residual = result.resid
threshold = 3 * residual.std()
anomalies = residual[abs(residual) > threshold]

# Plot the anomalies
plt.figure(figsize=(12, 6))
plt.plot(monthly_spending, marker='o', linestyle='-', label='Monthly Spending Score')
plt.scatter(anomalies.index, anomalies.values, color='red', label='Anomalies')
plt.title('Anomaly Detection in Spending Score')
plt.xlabel('Date')
plt.ylabel('Spending Score')
plt.legend()
plt.grid(True)
plt.show()

# Fit an ARIMA model
model = ARIMA(monthly_spending, order=(5, 1, 0))
model_fit = model.fit()

# Forecast the next 12 months
forecast = model_fit.forecast(steps=12)

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(monthly_spending, marker='o', linestyle='-', label='Historical')
plt.plot(forecast, marker='o', linestyle='-', color='red', label='Forecast')
plt.title('Spending Score Forecast')
plt.xlabel('Date')
plt.ylabel('Spending Score')
plt.legend()
plt.grid(True)
plt.show()

# Print forecast values
print(forecast)
