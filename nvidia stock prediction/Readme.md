## Nvidia Stock Price Prediction with LSTM and Swish Activation
This project allows us to predict future Nvidia stock prices using a Long Short-Term Memory (LSTM) neural network built with TensorFlow and Keras. The model leverages the Swish activation function, known for its potential benefits over traditional ReLU activation.

### What's Included?

* nvidia_stock_prediction.py: The core Python script for building, training, and evaluating the LSTM model.


* Data Preparation:
The script has a CSV file containing historical daily data on Nvidia stock prices. 





### The script performs the following actions:

* Loads historical data from the CSV file.
* Preprocesses the data (handling missing values, converting data types, scaling prices).
* Splits the data into training and testing sets for model training and evaluation.
* Constructs the LSTM model architecture with multiple LSTM layers and utilizes the Swish activation function.
* Trains the model on the training data for a set number of epochs.
* Evaluates the model's performance on the testing data using metrics like mean squared error.


### Further Resources

TensorFlow: https://www.tensorflow.org/\
Keras: https://keras.io/\
LSTM Networks: https://en.wikipedia.org/wiki/Long_short-term_memory


