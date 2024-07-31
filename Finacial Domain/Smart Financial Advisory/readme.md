# Smart Financial Advisory

## Overview

The Smart Financial Advisory project aims to provide data-driven financial advice using deep learning algorithms. The project involves generating a synthetic dataset, performing exploratory data analysis (EDA), and applying deep learning models to predict financial outcomes.

## Dataset

A synthetic dataset `financial_advisory_dataset.csv` is generated with the following features:
- **Client_ID**: Unique identifier for the client.
- **Age**: Age of the client.
- **Income**: Annual income of the client.
- **Investment_Experience**: Years of investment experience.
- **Risk_Tolerance**: Risk tolerance level ('Low', 'Medium', 'High').
- **Investment_Amount**: Amount invested.
- **Financial_GOAL**: Financial goal ('Retirement', 'Education', 'Buying Home', 'Travel').
- **Returns**: Annual returns on investment.

## Exploratory Data Analysis (EDA)

The EDA includes:
- Age distribution
- Income vs Investment Amount
- Returns distribution by Financial Goal

## Deep Learning Models

### Feedforward Neural Network (NN)
A neural network with two hidden layers is used to predict the returns based on client features.

### Optional Models
- **Convolutional Neural Network (CNN)**: Typically used for image or temporal data. Not applied in this example.
- **Long Short-Term Memory (LSTM)**: Suitable for time-series data. Applied if temporal features are included.




