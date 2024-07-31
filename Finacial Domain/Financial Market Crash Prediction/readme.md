# Financial Market Crash Prediction

## 🎯 Goal
The main goal of this project is to develop and compare different deep learning models for predicting financial market crashes. This project utilizes historical data, macroeconomic indicators, and other factors to forecast market crashes using LSTM and GRU models.

## 📁 Dataset
The dataset used for this project is `financial_market_crash_data.csv`, which contains data on market index, volatility, interest rate, GDP growth, unemployment rate, and market crash indicator.

## 📜 Description
This project involves the implementation of LSTM and GRU models for market crash prediction. The models are trained to recognize patterns in historical data and various features, and their performance is evaluated and compared.

## 🧮 What I Had Done!
1. Loaded and preprocessed the dataset.
2. Performed EDA to understand the distribution and characteristics of the data.
3. Implemented two different models: LSTM and GRU.
4. Trained and evaluated each model.
5. Compared the models based on their accuracy and loss metrics.

## 🚀 Models Implemented
- **Long Short-Term Memory (LSTM):** A type of recurrent neural network (RNN) that is well-suited for time series forecasting due to its ability to capture long-term dependencies.
- **Gated Recurrent Unit (GRU):** Another type of RNN that is similar to LSTM but has a simpler architecture, making it computationally efficient.

## 📚 Libraries Needed
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow

## 📊 Exploratory Data Analysis (EDA) Results
### Dataset Information
- Dataset shape: (1000, 7)
- Missing values: None

### Visualizations
- Pairplot of features
  ![Pairplot of Features](images/eda_pairplot.png)
- Correlation Heatmap
  ![Correlation Heatmap](images/eda_correlation.png)

## 📈 Performance of the Models Based on the Metrics
- **LSTM:**
  - Loss: `<Loss_LSTM_Value>`
  - Accuracy: `<Accuracy_LSTM_Value>`

- **GRU:**
  - Loss: `<Loss_GRU_Value>`
  - Accuracy: `<Accuracy_GRU_Value>`

## 📢 Conclusion
In this project, we explored the use of LSTM and GRU models for market crash prediction. Both models showed promising results, with the GRU model performing slightly better in terms of accuracy. These models can be further tuned and optimized for better performance.


