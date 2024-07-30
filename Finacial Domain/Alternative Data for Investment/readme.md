# Alternative Data for Investment

## 🎯 Goal
The main goal of this project is to leverage alternative data sources such as social media sentiment, web traffic, and search trends to predict stock prices using deep learning algorithms.

## 🧵 Dataset
The dataset used for this project is `alternative_data_investment.csv`. It contains synthetic data generated for stock prices, social media sentiment, web traffic, and search trends over business days from 2015 to 2020.

## 🧾 Description
This project involves the implementation of LSTM, GRU, and DNN models to predict stock prices using alternative data. The models are trained on historical data, and their performance is evaluated based on metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## 🧮 What I have done!
1. Generated a synthetic dataset using alternative data sources for investment.
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
- Number of features: 4 (Social_Sentiment, Web_Traffic, Search_Trends, Stock_Price)

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
In this project, we explored the use of LSTM, GRU, and DNN models for stock price prediction using alternative data sources. All models performed well, with the DNN model achieving the best results based on MSE and MAE metrics.

## Contributor
Ashish Kumar Patel
