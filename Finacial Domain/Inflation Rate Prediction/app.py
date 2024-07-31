

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('inflation_data.csv', parse_dates=['Date'])

# EDA Section
st.title("Inflation Rate Prediction")
st.write("### Exploratory Data Analysis")

st.write("#### Time Series of Economic Indicators")
plt.figure(figsize=(14, 8))
plt.plot(df['Date'].values, df['Inflation_Rate'], label='Inflation Rate')
plt.plot(df['Date'].values, df['GDP_Growth_Rate'], label='GDP Growth Rate')
plt.plot(df['Date'].values, df['Unemployment_Rate'], label='Unemployment Rate')
plt.plot(df['Date'].values, df['Interest_Rate'], label='Interest Rate')
plt.plot(df['Date'].values, df['Money_Supply'], label='Money Supply')
plt.legend()
st.pyplot(plt)

st.write("#### Correlation Matrix")
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)

# ML Section
st.write("### Inflation Rate Prediction")

# Prepare the data
X = df[['GDP_Growth_Rate', 'Unemployment_Rate', 'Interest_Rate', 'Money_Supply']]
y = df['Inflation_Rate']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
st.write(f"#### Mean Squared Error: {mse}")

# Plot the predictions
st.write("#### True vs Predicted Inflation Rate")
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='True Inflation Rate')
plt.plot(y_pred, label='Predicted Inflation Rate')
plt.legend()
st.pyplot(plt)
