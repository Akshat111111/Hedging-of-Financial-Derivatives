# Mortgage Default Prediction

## 🎯 Goal
The main goal of this project is to predict mortgage default using various features such as loan amount, interest rate, employment status, credit score, and loan term.

## 🧵 Dataset
The dataset used for this project is `mortgage_default_prediction.csv`. It contains synthetic data generated for mortgage loans, including whether the loan defaulted or not.

## 🧾 Description
This project involves the implementation of LSTM, GRU, and DNN models to predict mortgage default. The models are trained on historical data, and their performance is evaluated based on metrics like binary cross-entropy loss and accuracy.

## 🧮 What I have done!
1. Generated a synthetic dataset for mortgage default prediction.
2. Performed EDA to understand the distribution and characteristics of the data.
3. Implemented and trained LSTM, GRU, and DNN models.
4. Evaluated the models based on their loss and accuracy.
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
- Number of samples: 5000
- Number of features: 6 (Loan_Amount, Interest_Rate, Employment_Status, Credit_Score, Loan_Term, Default)

### Visualizations
![Pairplot](images/pairplot.png)

## 📈 Model Performance
- **LSTM:**
  - Loss: 0.3354
  - Accuracy: 0.8710

- **GRU:**
  - Loss: 0.3278
  - Accuracy: 0.8774

- **DNN:**
  - Loss: 0.3205
  - Accuracy: 0.8808

## 📢 Conclusion
In this project, we explored the use of LSTM, GRU, and DNN models for mortgage default prediction. All models performed well, with the DNN model achieving the best results based on loss and accuracy metrics.

## Contributor
Ashish Kumar Patel
