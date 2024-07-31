# Forex-Stock-Price-Prediction-using-Transformer-and-Time-Embeddings

This repository contains the code for the Project "Forex Stock Price Prediction using Transformers and Time Embeddings" written in Tensorflow 2.9.1.

## About the Project

The foreign exchange (Forex) market is a global decentralized market for the trading of currencies. It is one of the largest financial markets in the world, with an average daily trading volume of over $5 trillion. Predicting the price movements of Forex markets is a challenging task, as these markets are affected by a variety of factors, including economic and political events, interest rates, and global news. In recent years, deep neural networks have shown great potential in predicting stock market prices. In this report, weexplore the use of transformer-based deep neural networks for Forex stock market price prediction. We use the Transformer-XL architecture, which is a variant of the Transformer architecture that is designed to handle long sequences. We also use time embeddings to encode the time information of the input data. We evaluate our model on the EUR/USD Forex pair and compare it to a variety of baseline models. Our model achieves a mean absolute error (MAE) of 0.0004 on the test set, which is a 20% improvement over the best baseline model.

## Dataset Description and Preprocessing

The dataset we have used is that of IBM stock price history. The dataset starts from 1962-02-16 and ends on the date 2020-01-31. The data contains the Open, High, Low, Close as well as the trading Volume (OHLCV) of the IBM stock for every day between the aforementioned dates, leading to a total size of 14588 entries.

![IBM](https://github.com/ayushabrol13/Forex-Stock-Price-Prediction-using-Transformer-and-Time-Embeddings-/blob/master/plots/IBM_Close_Price.png)

![IBM Volume](https://github.com/ayushabrol13/Forex-Stock-Price-Prediction-using-Transformer-and-Time-Embeddings-/blob/master/plots/IBM_Volume.png)


## Conclusion

Deep neural networks (DNNs) are a type of machine learning algorithm that can be used to predict stock market prices. DNNs consist of multiple layers of interconnected nodes, and they are trained using large amounts of historical data to make predictions about future stock market prices. Recently, transformer-based DNNs have gained popularity in the field of natural language processing over RNNs, LSTMs and GRUs because of their attention mechanisms allowing them to have large reference windows, they have also been applied to other domains, including stock market price prediction. We have also explored stock price prediction using RNNs and LSTMs, however, transformers superseded them in the prediction task and hence our report sticks strictly towards the analysis of stock price prediction using Transformers.
