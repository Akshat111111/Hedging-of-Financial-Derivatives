# Walmart Stock Price Prediction

## 🎯 Goal
The main purpose of this project is to **Predict the Stock prices** from the dataset (mentioned below) using various models (LSTM, SGD And Random Forest) and comparing their accuracy.

## 🧵 Dataset

The link to the dataset is given below :-

**Link :- https://www.kaggle.com/datasets/meetnagadia/walmart-stock-price-from-19722022**

## 🧾 Description

This project involves the comparative analysis of Stock Price Prediction Of Walmart by Using the models, namely **LSTM** , **SGD** , **Random Forest** , applied to a specific dataset. The dataset consists of the data taken from the year of 1972 to 2022 so that we can accurately know what the actual prices, and the objectives include training and evaluating these models to compare their accuracy scores and performance metrics. Additionally, exploratory data analysis (EDA) techniques are employed to understand the dataset's characteristics, explore class distributions, detect imbalances, and identify areas for potential improvement. The methodology encompasses data preparation, model training, evaluation, comparative analysis of accuracy and performance metrics, and visualization of EDA insights.

## 🧮 What I had done!

### 1. Data Preprocessing:
    Load and clean historical stock price data for Walmart.
    Scaled the features to ensure they are on a similar scale. For LSTM, MinMaxScaler is commonly used to normalize the data between 0 and 1.
    Divided the dataset into training and testing sets. Typically, 80% of the data is used for training, and 20% for testing.
    Addressed any missing values and normalize the dataset to ensure consistency and accuracy.

### 2. Exploratory Data Analysis (EDA):
    Plotted the historical stock prices to understand trends and patterns over time.
    Checked the correlation between different features to understand relationships.
    Identified any seasonal patterns or long-term trends in the stock prices.
    Analyzed how trading volume correlates with price movements.

### 3. Data Analysis:
    Calculated moving averages to smooth out short-term fluctuations and highlight longer-term trends.
    Decomposed the data into trend, seasonal, and residual components.

### 4. Model Training:
    Builded an LSTM model using a framework like TensorFlow or Keras, specifying the number of layers and neurons.
    Adjusted hyperparameters such as batch size, epochs, and learning rate to improve performance.
    Used a simple linear regression model trained with the SGD optimizer.
    Fitted the model to the training data, ensuring the optimization process converges properly.

### 5. Model Evaluation:
    Used the trained models to predict stock prices on the test set.
    Compared the performance of LSTM, SGD, and Random Forest Regressor to determine which model provides the best predictions.
    Checked the Stock Open Price vs Closed Price.

## 🚀 Models Implemented

Trained the dataset on various models , each of their summary is as follows :-

### LSTM

While implementing the model for LSTM in code, We have Observed the dataset which defines and plots (based on the columns) like stock date price, volume etc.

**Images**:

![image](https://github.com/vivekvardhan2810/Hedging-of-Financial-Derivatives/assets/91594529/6b3f7bbd-6446-4158-b573-fea8102ba28d)

![image](https://github.com/vivekvardhan2810/Hedging-of-Financial-Derivatives/assets/91594529/2d1581ac-014a-4bee-bf7d-e839c223f484)

![image](https://github.com/vivekvardhan2810/Hedging-of-Financial-Derivatives/assets/91594529/4de17948-2acc-4147-9eb8-93e629bd7434)


Added the epoches upto 100, so that it beccomes productive

The accuracy became: [0.0005664046038873494, 0.02379925549030304]

### Stochastic Gradient Descent (SGD)

While implementing the SGD in code, We have Observed SGD randomly selects a subset (mini-batch) of the training data to compute the gradient, introducing stochasticity into the optimization process.

**Accuracy**: 

Mean Squared Error: 0.25953534275012946

R-squared: 0.9998106265355644

## Random Forest

While implementing the Random Forest in code, We have Observed During prediction, Random Forest aggregates the predictions of all trees (regression trees in the case of regression tasks) to produce the final output.

**Accuracy**: 

RMSE: 0.4214316254053856

MAE: 0.20164004768581736

## 📚 Libraries Needed

1. **NumPy:** Fundamental package for numerical computing.
2. **pandas:** Data analysis and manipulation library.
3. **scikit-learn:** Machine learning library for classification, regression, and clustering.
4.  **Matplotlib:** Plotting library for creating visualizations.
5.  **Keras:** High-level neural networks API, typically used with TensorFlow backend.
6. **seaborn:** Statistical data visualization library based on Matplotlib.

## 📊 Exploratory Data Analysis Results

### Bar Chart :-
 A bar chart showing the distribution of labels in the training dataset. It visually represents the frequency of each label category, providing an overview of how the labels are distributed across the dataset.

![image](https://github.com/vivekvardhan2810/Hedging-of-Financial-Derivatives/assets/91594529/8e67c98e-18bf-4c97-8ac0-122b8fc937a9).

### Plotting:

![image](https://github.com/vivekvardhan2810/Hedging-of-Financial-Derivatives/assets/91594529/490c6612-1e3a-4bdb-850e-da1eec29e263)

### Heatmaps:

![image](https://github.com/vivekvardhan2810/Hedging-of-Financial-Derivatives/assets/91594529/15086ae0-a76b-4845-833d-c79c513a6cca)

![image](https://github.com/vivekvardhan2810/Hedging-of-Financial-Derivatives/assets/91594529/5f362014-2ace-465d-a8aa-a451106d50e9)

## 📈 Performance of the Models based on the Accuracy Scores

| Models      |       Accuracy Scores|
|------------ |------------|
|LSTM  |0.56% ( Validation Accuracy: 0.056)|
|SGD  | 99% (Validation Accuracy: 0.9998) |
|Random Forest  | 42% (Validation Accuracy:0.4214) |

## 📢 Conclusion

According to the accuracy scores it can be concluded that SGD , Random Forest were able to perform good on this dataset, but LSTM was not that much good.

## ✒️ Your Signature

Full name:- Vivek Vardhan                      
Github Id :- https://github.com/vivekvardhan2810 
Email ID :- vivekvardhan43862@gmail.com  
LinkdIn :- https://www.linkedin.com/in/vivek-vardhan-23682521b/ </br>
Participant Role :- Contributor / GSSOC (Girl Script Summer of Code ) - 2024
