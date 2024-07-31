# Financial Time Series Forecasting

## 🎯 Goal
The main goal of this project is to develop a model for predicting financial time series data using deep learning algorithms. This project uses LSTM and GRU models to predict stock prices based on historical data.

## 🧵 Dataset
The dataset used for this project is `financial_time_series_data.csv`. It contains synthetic financial data for the period from 2010-01-01 to 2020-12-31, including columns such as `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.

## 🧾 Description
This project involves the implementation of LSTM and GRU models to forecast stock prices. The models are trained on historical financial data, and their performance is evaluated based on metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## 🧮 What I have done!
1. Generated a synthetic financial time series dataset.
2. Performed EDA to understand the distribution and characteristics of the data.
3. Implemented and trained LSTM and GRU models.
4. Evaluated the models based on their MSE and MAE.
5. Visualized the model predictions against true values.

## 🚀 Models Implemented
- **Long Short-Term Memory (LSTM):** LSTMs are well-suited for time series forecasting due to their ability to learn long-term dependencies.
- **Gated Recurrent Unit (GRU):** GRUs are a variant of RNNs that are effective for sequence learning and can capture dependencies over time.

## 📚 Libraries Needed
- numpy
- pandas
- matplotlib
- seaborn
- sklearn
- tensorflow

## 📊 Exploratory Data Analysis Results
### Dataset Distribution
- Number of samples: 2886
- Number of features: 5 (Open, High, Low, Close, Volume)

### Visualizations
![Closing Prices Over Time](images/closing_prices_over_time.png)

## 📈 Model Performance
- **LSTM:**
  - MSE: 0.000897
  - MAE: 0.021141

- **GRU:**
  - MSE: 0.000804
  - MAE: 0.020083

## 📢 Conclusion
In this project, we explored the use of LSTM and GRU models for financial time series forecasting. Both models performed well, with the GRU model achieving slightly better results based on MSE and MAE metrics.

## Contributor
Ashish Kumar Patel
