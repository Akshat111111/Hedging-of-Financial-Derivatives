# Derivative Pricing Models Using Deep Learning

## 🎯 Goal
The main goal of this project is to develop a model for pricing derivatives, specifically options, using deep learning algorithms. This project uses LSTM, GRU, and DNN models to predict option prices based on historical financial data.

## 🧵 Dataset
The dataset used for this project is `derivative_pricing_data.csv`. It contains synthetic option pricing data generated using the Black-Scholes model, including columns such as `Date`, `Strike_Price`, `Underlying_Price`, `Volatility`, `Time_to_Maturity`, `Risk_Free_Rate`, and `Option_Price`.

## 🧾 Description
This project involves the implementation of LSTM, GRU, and DNN models to forecast option prices. The models are trained on historical financial data, and their performance is evaluated based on metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## 🧮 What I have done!
1. Generated a synthetic financial time series dataset.
2. Performed EDA to understand the distribution and characteristics of the data.
3. Implemented and trained LSTM, GRU, and DNN models.
4. Evaluated the models based on their MSE and MAE.
5. Visualized the model predictions against true values.

## 🚀 Models Implemented
- **Long Short-Term Memory (LSTM):** LSTMs are well-suited for time series forecasting due to their ability to learn long-term dependencies.
- **Gated Recurrent Unit (GRU):** GRUs are a variant of RNNs that are effective for sequence learning and can capture dependencies over time.
- **Deep Neural Network (DNN):** DNNs are powerful models for capturing complex relationships in the data.

## 📚 Libraries Needed
- numpy
- pandas
- matplotlib
- seaborn
- sklearn
- tensorflow

## 📊 Exploratory Data Analysis Results
### Dataset Distribution
- Number of samples: 1578
- Number of features: 6 (Strike_Price, Underlying_Price, Volatility, Time_to_Maturity, Risk_Free_Rate, Option_Price)

### Visualizations
![Pairplot](images/pairplot.png)

## 📈 Model Performance
- **LSTM:**
  - MSE: 0.000897
  - MAE: 0.021141

- **GRU:**
  - MSE: 0.000804
  - MAE: 0.020083

- **DNN:**
  - MSE: 0.000762
  - MAE: 0.019721

## 📢 Conclusion
In this project, we explored the use of LSTM, GRU, and DNN models for derivative pricing. All models performed well, with the DNN model achieving the best results based on MSE and MAE metrics.

## Contributor
Ashish Kumar Patel
