# Corporate Earnings Prediction Using Deep Learning

## Overview

This project involves predicting corporate earnings using deep learning techniques. The goal is to develop a model that can accurately forecast future earnings based on historical financial data.

## Table of Contents

- [Project Description](#project-description)
- [Data](#data)
- [Dependencies](#dependencies)
- [Model](#model)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)

## Project Description

The aim of this project is to leverage deep learning to predict corporate earnings. We use historical earnings data to train a model and evaluate its performance. The dataset includes features related to past earnings and other financial metrics.

## Data

The dataset used in this project is `corporate_earnings_prediction.csv`. It contains the following columns:

- `Date`: The date of the earnings report.
- `Feature_1`, `Feature_2`, ..., `Feature_N`: Financial metrics and indicators.
- `Target_Earnings`: The earnings to be predicted.

**Sample Data:**

| Date       | Feature_1 | Feature_2 | ... | Target_Earnings |
|------------|-----------|-----------|-----|-----------------|
| 2020-01-01 | 100.0     | 200.0     | ... | 150.0           |
| 2020-02-01 | 110.0     | 210.0     | ... | 160.0           |
| ...        | ...       | ...       | ... | ...             |

## Dependencies

This project requires the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow` (or `keras`)

## Model
The deep learning model used in this project is a Sequential Neural Network implemented using TensorFlow/Keras. The model architecture includes:

1. Input Layer
2. Dense Hidden Layers with ReLU Activation
3. Output Layer with a Linear Activation Function

Training
The model is trained using the following configuration:

Epochs: 50
Batch Size: 1
Optimizer: Adam
Loss Function: Mean Squared Error (MSE)

Results-
Epoch 1/50
1/1 [==============================] - 2s 2s/step - loss: 361613328384.0000 - val_loss: 338603704320.0000
...
Epoch 50/50
1/1 [==============================] - 0s 59ms/step - loss: 361499623424.0000 - val_loss: 338476302336.0000

The final model performance is evaluated based on the loss function. The loss decreased over 50 epochs, with the final training loss at approximately 361499623424.0000 and validation loss at 338476302336.0000.

Test Loss: 338458083328.0

# Contributor- 
Ashish Kumar Patel
