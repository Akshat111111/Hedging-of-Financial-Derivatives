Stock Price Prediction Using RNN

Overview Of Project:

Stock Price Prediction using machine learning helps to discover the future value of company stock on stock market. The benefit of predicting the value of stock is to gain more profit from the market. For the closing price prediction of Tata Motors, the model is built using Long Short Term Memory Network (LSTM). The model is trained using the past records of the Company from Yahoo Finance.

Libraries/Frameworks used:

pandas_datareader: It is used to extract the dataset of stock prices.

numpy: It is used to handle the arrays.

pandas: It is used to handle data frames.

MinMaxScaler: Is used to scale the data.

Sequential: To build the ML model

Dense: It is a layer for the model, in which each neuron receives input from all the neurons of the previous layer.

LSTM: It is a layer for the model, in which each neuron receiver input from all the past previous layers.

Dropout: It is a layer for the model, it helps to randomly select the neurons and helps to reduce the overfitting of the model.

matplotlib.pylot: Is used for plotting the graphs for values.