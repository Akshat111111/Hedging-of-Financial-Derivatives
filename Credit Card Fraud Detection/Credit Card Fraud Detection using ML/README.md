# Credit Card Fraud Detection using Logistic Regression

Welcome to the Credit Card Fraud Detection project! This project aims to detect fraudulent transactions using a Logistic Regression model. This README file provides a comprehensive guide on how to set up and run the project.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Training](#model-training)
6. [Evaluation](#evaluation)
7. [License](#license)

## Introduction

Credit card fraud is a significant issue in the financial sector. Detecting fraudulent transactions promptly can prevent substantial financial losses. This project utilizes machine learning, specifically Logistic Regression, to identify and flag potential fraudulent transactions.

## Dataset

The dataset used in this project is a commonly used credit card fraud detection dataset, which contains transactions made by European cardholders in September 2013. It includes 284,807 transactions, among which 492 are fraudulent. The dataset is highly imbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

You can download the dataset from [Kaggle][https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)


## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. **Create a virtual environment**:
    ```sh
    python3 -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Data Preprocessing

Before training the model, you need to preprocess the data. Run the following command to start the data preprocessing:

```sh
python src/data_preprocessing.py
```

### Model Training

To train the Logistic Regression model, use the `model_training.ipynb` notebook or run the following command:

```sh
python src/model.py
```

### Evaluation

Evaluate the model's performance using the `model_evaluation.ipynb` notebook or run the following command:

```sh
python src/evaluation.py
```

## Model Training

The Logistic Regression model is trained using the following steps:

1. **Load and preprocess the dataset**.
2. **Split the dataset** into training and testing sets.
3. **Train the Logistic Regression model** on the training set.
4. **Evaluate the model** on the testing set using appropriate metrics.

## Evaluation

The model is evaluated using various metrics, including:

- **Accuracy**
- **Precision**

These metrics provide a comprehensive view of the model's performance, especially in the context of imbalanced datasets.


Please ensure that your code adheres to the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
