# Predicting IPO Performance Using Deep Learning

## 🎯 Goal
The main goal of this project is to develop a model to predict the performance of Initial Public Offerings (IPOs) based on historical data and market conditions. This project will involve implementing and comparing the performance of three different deep learning algorithms: Basic Neural Networks (NN), Convolutional Neural Networks (CNN), and Recurrent Neural Networks (RNN).

## 🧵 Dataset
The dataset used for this project is `ipo_performance_data.csv`. It contains historical data of IPOs, including various features such as industry, financial metrics, and market conditions, as well as the IPO performance.

## 🧾 Description
This project involves the implementation of three different deep learning algorithms to predict IPO performance: Basic Neural Networks (NN), Convolutional Neural Networks (CNN), and Recurrent Neural Networks (RNN). Each model is trained to predict the performance of IPOs, and their performance is evaluated and compared based on Mean Squared Error (MSE) and Mean Absolute Error (MAE).

## 🧮 What I Had Done!
1. Loaded and preprocessed the dataset.
2. Performed EDA to understand the distribution and characteristics of the data.
3. Implemented three different models: Basic NN, CNN, and RNN.
4. Trained and evaluated each model.
5. Compared the models based on their MSE and MAE scores.

## 🚀 Models Implemented
- **Basic Neural Network (NN):** A simple neural network model with fully connected layers.
- **Convolutional Neural Network (CNN):** Chosen for its effectiveness in handling grid-like data and extracting spatial features.
- **Recurrent Neural Network (RNN):** Specifically LSTMs, used for sequence learning, making them useful for predicting sequences of financial data.

## 📚 Libraries Needed
- pandas
- numpy
- matplotlib
- seaborn
- tensorflow
- sklearn

## 📊 Exploratory Data Analysis Results
- Dataset shape: (1000, 10)
- Missing values in the dataset: None

### Visualizations
#### Distribution of IPO Performance
![Distribution of IPO Performance](images/ipo_performance_distribution.png)

#### Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)

## 📈 Performance of the Models based on the MSE and MAE Scores

- **Basic NN:**
  - MSE: 117.01535807363456
  - MAE: 9.484842688226093

- **CNN:**
  - MSE: 53.287359829332296
  - MAE: 5.351262890476877

- **RNN:**
  - MSE: 168.49246983074522
  - MAE: 11.557391764757833

## 📢 Conclusion
In this project, we explored the use of Basic NN, CNN, and RNN models for predicting IPO performance. The CNN model achieved the lowest MSE and MAE scores, making it the most effective model for this task. The Basic NN model also performed reasonably well, while the RNN model had the highest error scores.

Based on MSE and MAE scores, the CNN model is the best-fitted model for predicting IPO performance among the three developed models.

## Contributor
Ashish Kumar Patel
