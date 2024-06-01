# Loan Status Prediction with Multiple Machine Learning Algorithms

## Table of Contents
1. [Introduction](#introduction)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Model Selection](#model-selection)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Model Evaluation](#model-evaluation)
6. [Conclusion](#conclusion)
7. [Installation and Usage](#installation-and-usage)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction
The Loan Status Prediction project aims to predict the status of loans using multiple machine learning algorithms. By incorporating a diverse set of models and comparing their performance, we can improve the accuracy and reliability of the predictions. This project includes Exploratory Data Analysis (EDA), model selection, hyperparameter tuning, and model evaluation.

## Exploratory Data Analysis (EDA)
EDA is a crucial step in understanding the dataset and preparing it for model training. The following steps were undertaken:

1. **Data Cleaning and Preprocessing:** 
   - Handled missing values.
   - Encoded categorical variables.
   - Scaled numerical features.

2. **Visualization:**
   - Visualized the distribution of numerical features.
   - Analyzed the relationships between different features using scatter plots and correlation matrices.
   - Identified patterns and anomalies in the data.

## Model Selection
A diverse set of machine learning models were chosen for evaluation:

1. **Logistic Regression:**
   - Suitable for binary classification problems.
   - Provides probabilities for class membership.

2. **Decision Trees:**
   - Non-parametric model.
   - Captures non-linear relationships.

3. **Random Forest:**
   - Ensemble method using multiple decision trees.
   - Reduces overfitting and improves generalization.

4. **Gradient Boosting Machines (GBM):**
   - Sequential ensemble technique.
   - Optimizes performance through gradient descent.

5. **Support Vector Machines (SVM):**
   - Effective in high-dimensional spaces.
   - Uses hyperplanes to separate classes.

6. **Voting Classifier:**
    - Combines multiple machine learning models to improve predictive performance.
    - Can use both hard voting (majority rule) and soft voting (average predicted       probabilities).
    - Utilized Logistic Regression, Decision Trees, Random Forest, as base models in the Voting Classifier.

## Hyperparameter Tuning
Hyperparameter tuning was performed to optimize the performance of each model. Techniques such as Grid Search and Random Search were used to find the best parameters.

## Model Evaluation
The performance of each model was evaluated using various metrics:

1. **Confusion Matrix:**
   - Visual representation of the true positives, true negatives, false positives, and false negatives.

2. **Precision, Recall:**
   - Precision: Ratio of true positives to the sum of true and false positives.
   - Recall: Ratio of true positives to the sum of true positives and false negatives.

3. **ROC-AUC Curve:**
   - Plots the true positive rate against the false positive rate.
   - Measures the ability of the model to distinguish between classes.

The best-performing model was selected based on these evaluation metrics.

## Conclusion
Incorporating multiple machine learning algorithms and comparing their performance significantly improved the accuracy and reliability of loan status predictions. The Random Forest model emerged as the best-performing model, providing a good balance of precision, recall, and overall predictive power.

## Installation and Usage
To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/binguliki/loan-status-prediction.git
   cd loan-status-prediction
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project:**
   ```bash
   python main.py
   ```

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a pull request.

Please ensure your pull request adheres to the [contribution guidelines](CONTRIBUTING.md).
