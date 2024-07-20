# GDP Prediction Model

This project aims to predict the Gross Domestic Product (GDP) using a dataset from [Kaggle](https://www.kaggle.com/rutikbhoyar/gdp-prediction-dataset). Four different machine learning regressors were evaluated to determine the best model for accurate GDP prediction.

## Dataset

The dataset used in this project is available on Kaggle: [GDP Prediction Dataset](https://www.kaggle.com/rutikbhoyar/gdp-prediction-dataset). It includes various features that can influence GDP, such as economic indicators, demographic information, and other relevant data points.

## Models Tested

The following machine learning regressors were used to predict GDP:

1. **Linear Regression**
2. **Support Vector Machine (SVM)**
3. **Random Forest**
4. **Gradient Boosting**

## Evaluation Metrics

The models were evaluated using the following metrics:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in a set of predictions, without considering their direction.
- **Root Mean Squared Error (RMSE)**: Measures the square root of the average of squared differences between prediction and actual observation. It gives higher weight to large errors.
- **R-Squared (R²) Score**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Results

Based on the evaluation metrics, the Random Forest Regressor provided the best prediction performance. The results are summarized below:

### Random Forest Regressor

- **Mean Absolute Error (MAE)**: 2125.24
- **Root Mean Squared Error (RMSE)**: 3051.71
- **R-Squared (R²) Score**: 0.8873

### Performance Comparison

The performance of the models in descending order is as follows:

1. **Random Forest**
2. **Gradient Boosting**
3. **Linear Regression**
4. **Support Vector Machine (SVM)**

## Usage

### Prerequisites

Ensure you have the following libraries installed:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Conclusion

The Random Forest Regressor provided the most accurate predictions for GDP, as evidenced by its superior performance across multiple evaluation metrics. This model can be further tuned and optimized to improve accuracy and reliability.

For further details on model training, evaluation, and implementation, please refer to the `gdp_prediction.py` script.

