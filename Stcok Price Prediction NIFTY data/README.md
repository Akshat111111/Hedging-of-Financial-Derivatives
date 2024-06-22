# Stock Price Prediction using LSTM, Linear Regression, and RNN

This repository contains the implementation of stock price prediction models using Long Short-Term Memory (LSTM), Linear Regression, and Recurrent Neural Network (RNN) techniques. The project is based on the Nifty50 stock market dataset available on Kaggle and is developed as part of the GirlScript Summer of Code 2024.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
  - [LSTM](#lstm)
  - [Linear Regression](#linear-regression)
  - [RNN](#rnn)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [License](#license)

## Introduction
This project aims to predict stock prices using three different machine learning models: LSTM, Linear Regression, and RNN. The goal is to compare the performance of these models in terms of their prediction accuracy and other evaluation metrics.

## Dataset
The dataset used in this project is the Nifty50 stock market data, which includes historical stock prices for various companies. The dataset can be found on Kaggle: [Nifty50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data).


## Installation
To install the necessary dependencies, run:
```
pip install -r requirements.txt
```

## Usage
To train and evaluate the models, follow these steps:

1. **Data Preprocessing**:
   Run the data preprocessing script to clean and prepare the data.
   ```bash
   python src/data_preprocessing.py
   ```

2. **Train LSTM Model**:
   Train the LSTM model using the preprocessed data.
   ```bash
   python src/train_lstm.py
   ```

3. **Train Linear Regression Model**:
   Train the Linear Regression model using the preprocessed data.
   ```bash
   python src/train_linear_regression.py
   ```

4. **Train RNN Model**:
   Train the RNN model using the preprocessed data.
   ```bash
   python src/train_rnn.py
   ```

## Models Implemented

### LSTM
The LSTM model is a type of recurrent neural network capable of learning long-term dependencies, especially suited for time series prediction tasks.

### Linear Regression
Linear Regression is a simple and interpretable model used for predicting a continuous target variable based on the linear relationship between input features.

### RNN
Recurrent Neural Networks (RNNs) are designed for sequence data and are capable of capturing temporal dependencies in the data.

## Evaluation Metrics
The performance of the models is evaluated using the following metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R-squared (R²)
- Root Mean Squared Error (RMSE)
- Accuracy
- F1 score

## Results
The results section will contain a detailed comparison of the models' performance based on the evaluation metrics mentioned above. Visualizations such as loss curves and predicted vs actual price plots will also be included.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
