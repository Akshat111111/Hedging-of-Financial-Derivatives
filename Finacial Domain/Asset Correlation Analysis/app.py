import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (GRU, LSTM, Conv1D, Dense, Flatten,
                                     MaxPooling1D)
from tensorflow.keras.models import Sequential


# Function to load data from a file
@st.cache_data
def load_data(file):
    data = pd.read_csv(file, parse_dates=['Date'])
    return data

# Function to calculate returns
def calculate_returns(data):
    returns = data.set_index('Date').pct_change().dropna()
    return returns

# Function to plot time series
def plot_time_series(data):
    plt.figure(figsize=(10, 6))
    for col in data.columns[1:]:
        plt.plot(data['Date'], data[col], label=col)
    plt.title('Time Series of Asset Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

# Function to plot correlation matrix
def plot_correlation_matrix(data):
    plt.figure(figsize=(10, 6))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    st.pyplot(plt)

# Function to plot rolling correlation
def plot_rolling_correlation(returns, asset1, asset2, window):
    rolling_corr = returns[asset1].rolling(window).corr(returns[asset2])
    plt.figure(figsize=(10, 6))
    plt.plot(rolling_corr)
    plt.title(f'Rolling Correlation between {asset1} and {asset2}')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    st.pyplot(plt)

# Function to create datasets
def create_dataset(data, time_step=1):
    dataX, dataY = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(data[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Function to plot predictions
def plot_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('Actual vs Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    st.pyplot(plt)

# Streamlit app
st.title('Asset Correlation Analysis')

# File upload
uploaded_file = st.file_uploader("Upload your asset prices CSV file", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("First few rows of the dataset:")
    st.write(df.head())

    # EDA
    st.header("Exploratory Data Analysis")
    st.write("Statistical description of the dataset:")
    st.write(df.describe())
    
    st.subheader("Time Series of Asset Prices")
    plot_time_series(df)

    st.subheader("Correlation Matrix of Asset Prices")
    plot_correlation_matrix(df)

    # Correlation Analysis
    st.header("Correlation Analysis")
    returns = calculate_returns(df)

    st.subheader("Correlation Matrix of Daily Returns")
    plot_correlation_matrix(returns)

    st.subheader("Rolling Correlation Analysis")
    assets = df.columns[1:]
    asset1 = st.selectbox("Select first asset", assets)
    asset2 = st.selectbox("Select second asset", assets)
    window = st.slider("Select rolling window size (days)", min_value=5, max_value=100, value=30)
    if asset1 != asset2:
        plot_rolling_correlation(returns, asset1, asset2, window)
    else:
        st.write("Please select two different assets.")
    
    # Deep Learning Models
    st.header("Deep Learning Models")
    
    # Preprocess data for deep learning models
    df[asset1] = df[asset1].fillna(df[asset1].mean())
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[asset1].values.reshape(-1, 1))

    training_size = int(len(scaled_data) * 0.8)
    test_size = len(scaled_data) - training_size
    train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:len(scaled_data), :]

    time_step = 10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # LSTM Model
    st.subheader("LSTM Model")
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    lstm_model.add(LSTM(50, return_sequences=False))
    lstm_model.add(Dense(25))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, batch_size=1, epochs=1)
    
    train_predict = lstm_model.predict(X_train)
    test_predict = lstm_model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    
    st.write("LSTM Model Predictions")
    plot_predictions(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict)

    # GRU Model
    st.subheader("GRU Model")
    gru_model = Sequential()
    gru_model.add(GRU(50, return_sequences=True, input_shape=(time_step, 1)))
    gru_model.add(GRU(50, return_sequences=False))
    gru_model.add(Dense(25))
    gru_model.add(Dense(1))
    gru_model.compile(optimizer='adam', loss='mean_squared_error')
    gru_model.fit(X_train, y_train, batch_size=1, epochs=1)
    
    train_predict = gru_model.predict(X_train)
    test_predict = gru_model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    
    st.write("GRU Model Predictions")
    plot_predictions(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict)

    # CNN Model
    st.subheader("CNN Model")
    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(time_step, 1)))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(50, activation='relu'))
    cnn_model.add(Dense(1))
    cnn_model.compile(optimizer='adam', loss='mean_squared_error')
    cnn_model.fit(X_train, y_train, batch_size=1, epochs=1)
    
    train_predict = cnn_model.predict(X_train)
    test_predict = cnn_model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    
    st.write("CNN Model Predictions")
    plot_predictions(scaler.inverse_transform(y_test.reshape(-1, 1)), test_predict)
else:
    st.write("Please upload a CSV file to proceed.")
