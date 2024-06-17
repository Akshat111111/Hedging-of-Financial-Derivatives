import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn.hmm import GaussianHMM

"""
This script demonstrates the application of Hidden Markov Models (HMM) for stock price analysis and trading strategy development.

The script performs the following steps:
1. Load and preprocess the stock price dataset.
2. Perform exploratory data analysis (EDA) on the dataset.
3. Fit a Gaussian HMM to the log returns of the stock prices.
4. Predict the hidden states of the stock prices using the trained HMM.
5. Develop a trading strategy based on the predicted hidden states.
6. Visualize the stock prices, hidden states, and trading signals.

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- hmmlearn

"""

# Set the aesthetics for the plots
sns.set(style="whitegrid")

# Load the dataset
data = pd.read_csv('Dataset/US Stock Market Data & Technical Indicators/AAPL.csv')
df_Stock = data.rename(columns={'Close(t)': 'Close'})

"""
Exploratory Data Analysis (EDA)
"""

# Display basic information about the dataset
print(df_Stock.info())

# Summary statistics for numerical features
print(df_Stock.describe())

# Plot histograms
df_Stock.hist(bins=50, figsize=(20, 15))
plt.show()

# Compute the correlation matrix and generate a heatmap
numeric_df = df_Stock.select_dtypes(include=[np.number])
corr = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

# Plotting the closing prices
df_Stock['Close'].plot(figsize=(10, 7))
plt.title("Stock Price", fontsize=17)
plt.ylabel('Price', fontsize=14)
plt.xlabel('Time', fontsize=14)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()

"""
Hidden Markov Model (HMM)
"""

# Prepare data for HMM
df_Stock['Log_Returns'] = np.log(df_Stock['Close'] / df_Stock['Close'].shift(1))
df_Stock.dropna(inplace=True)

# Fit HMM
model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000)
model.fit(df_Stock[['Log_Returns']])

"""
The GaussianHMM class from the hmmlearn library is used to initialize an HMM with the following parameters:
- n_components: The number of hidden states in the model (set to 4 in this case).
- covariance_type: The type of covariance matrix for each hidden state (set to "diag" for diagonal covariance matrices).
- n_iter: The maximum number of iterations for the Baum-Welch algorithm used for training the HMM (set to 1000).

The fit() method is called to train the HMM on the log returns of the stock prices.
"""

# Predict hidden states and plot them
hidden_states = model.predict(df_Stock[['Log_Returns']])

"""
The predict() method is used to predict the most likely sequence of hidden states for the given log returns.
The predicted hidden states are stored in the hidden_states variable.
"""

plt.figure(figsize=(10, 7))
for i in range(model.n_components):
    state = (hidden_states == i)
    plt.plot(df_Stock.index[state], df_Stock['Close'][state], '.', label=f'State {i+1}')
plt.title('Stock Price Segmented by Hidden States')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

"""
The above code plots the stock prices segmented by the predicted hidden states.
Each hidden state is represented by a different color and marker.
"""

# Define the trading strategy
def apply_trading_strategy(hidden_states, data):
    buy_signals = []
    sell_signals = []
    portfolio = []
    cash = 10000  # initial cash
    position = 0  # initial stock position

    for i in range(1, len(hidden_states)):
        if hidden_states[i] != hidden_states[i-1]:
            if hidden_states[i] == 0 and position == 0:  # Regime 0 might be interpreted as bullish
                # Buy signal
                shares_to_buy = cash // data['Close(t)'][i]
                cash -= shares_to_buy * data['Close(t)'][i]
                position += shares_to_buy
                buy_signals.append(data['Close(t)'].index[i])
            elif hidden_states[i] == 1 and position > 0:  # Regime 1 might be interpreted as bearish
                # Sell signal
                cash += position * data['Close(t)'][i]
                position = 0
                sell_signals.append(data['Close(t)'].index[i])

        portfolio.append(cash + position * data['Close(t)'][i])

    return buy_signals, sell_signals, portfolio

"""
The apply_trading_strategy() function implements a simple trading strategy based on the predicted hidden states.
The strategy assumes that:
- Regime 0 (hidden state 0) is interpreted as a bullish market, and a buy signal is generated when the hidden state changes to 0.
- Regime 1 (hidden state 1) is interpreted as a bearish market, and a sell signal is generated when the hidden state changes to 1.

The function takes the predicted hidden states and the stock price data as input and returns the buy signals, sell signals, and the portfolio value over time.
"""

# Apply the trading strategy
buy_signals, sell_signals, portfolio = apply_trading_strategy(hidden_states, data)

# Plot the results
plt.figure(figsize=(15, 5))
plt.plot(data['Close(t)'], label='Close Price')
plt.plot(data['Close(t)'].index[buy_signals], data['Close(t)'][buy_signals], '^', markersize=10, color='g', lw=0, label='Buy signals')
plt.plot(data['Close(t)'].index[sell_signals], data['Close(t)'][sell_signals], 'v', markersize=10, color='r', lw=0, label='Sell signals')
plt.title('Trading Strategy')
plt.legend()
plt.show()

"""
The above code applies the trading strategy to the stock price data and plots the results.
The stock prices are plotted as a line plot, and the buy and sell signals are indicated by green triangles and red inverted triangles, respectively.
"""
