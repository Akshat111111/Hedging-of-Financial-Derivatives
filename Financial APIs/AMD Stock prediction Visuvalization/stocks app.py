import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import gradio as gr

# Load the pre-trained model
model = load_model('AMD_model.h5')

def fetch_and_predict(stock_ticker, start_date, end_date):
    # Fetch stock data
    df = yf.download(stock_ticker, start=start_date, end=end_date)

    # Use only the 'Close' feature for prediction
    data = df[['Close']]

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare the data for the model
    def create_sequences(data, look_back=60):
        X = []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back), 0])
        return np.array(X)

    look_back = 60
    X = create_sequences(scaled_data, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Make predictions
    y_pred = model.predict(X)

    # Inverse transform the predictions to get actual values
    predicted_prices = scaler.inverse_transform(np.concatenate((np.zeros((len(y_pred), data.shape[1]-1)), y_pred), axis=1))[:, -1]

    # Plot the results
    plt.figure(figsize=(18, 12))

    # Plot actual vs predicted stock prices
    plt.subplot(2, 2, 1)
    plt.plot(data.index[look_back:], predicted_prices, color='red', label='Predicted Stock Price')
    plt.plot(data.index, data['Close'], color='blue', label='Actual Stock Price')
    plt.title(f'Stock Price Prediction for {stock_ticker}')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()

    # Plot moving average
    moving_avg = data['Close'].rolling(window=30).mean()
    plt.subplot(2, 2, 2)
    plt.plot(data.index, data['Close'], color='blue', label='Stock Price')
    plt.plot(data.index, moving_avg, color='orange', label='30-Day Moving Average')
    plt.title('30-Day Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()

    # Plot bar chart of monthly average prices
    df_monthly = data.resample('M').mean()
    plt.subplot(2, 2, 3)
    df_monthly['Close'].plot(kind='bar', color='green')
    plt.title('Monthly Average Stock Price')
    plt.xlabel('Month')
    plt.ylabel('Average Stock Price')

    # Plot pie chart of price distribution
    last_week_prices = data['Close'].resample('W').mean().tail(4)
    plt.subplot(2, 2, 4)
    plt.pie(last_week_prices, labels=last_week_prices.index.strftime('%Y-%m-%d'), autopct='%1.1f%%', colors=plt.cm.Paired(np.arange(len(last_week_prices))))
    plt.title('Last 4 Weeks Stock Price Distribution')

    plot_path = 'prediction_plots.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path

# Define the Gradio interface with a simulate button
def run_simulation(stock_ticker, start_date, end_date):
    # Call the prediction function and return the plot
    return fetch_and_predict(stock_ticker, start_date, end_date)

iface = gr.Interface(
    fn=run_simulation,
    inputs=[
        gr.Textbox(label="Stock Ticker", value="AMD"),
        gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2023-01-01"),
        gr.Textbox(label="End Date (YYYY-MM-DD)", value="2023-12-31")
    ],
    outputs=gr.Image(type="filepath", label="Stock Price Analysis"),
    live=False
)

# Add a "Simulate" button
iface.launch(share=True, inbrowser=True)
