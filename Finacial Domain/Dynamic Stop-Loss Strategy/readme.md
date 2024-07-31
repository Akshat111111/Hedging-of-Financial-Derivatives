# Dynamic Stop-Loss Strategy

This project implements a dynamic stop-loss strategy using historical stock data, exploratory data analysis (EDA), and a genetic algorithm to optimize the strategy parameters.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Genetic Algorithm](#genetic-algorithm)
- [Installation](#installation)
- [Usage](#usage)


## Introduction
A dynamic stop-loss strategy is a risk management technique designed to limit losses and protect profits. This project uses historical stock data to develop and optimize such a strategy using a genetic algorithm.

## Dataset
The dataset is obtained using historical stock data from Yahoo Finance. The data includes daily prices for a selected stock (e.g., Apple Inc.).

## Exploratory Data Analysis
EDA is performed to understand the data structure, visualize key features, and prepare for the application of the genetic algorithm. Key steps include:
- Plotting the closing price over time.
- Analyzing the distribution of daily returns.

## Genetic Algorithm
A genetic algorithm is used to optimize the stop-loss and take-profit percentages, maximizing the final capital. The steps involved include:
- **Fitness Function**: Evaluates the performance of a given solution.
- **Population Initialization**: Creates an initial population of potential solutions.
- **Selection**: Selects the best-performing solutions to be parents for the next generation.
- **Crossover**: Combines pairs of parents to produce offspring.
- **Mutation**: Introduces random variations to the offspring to maintain diversity.




## Usage
1. Generate the dataset using historical stock data from Yahoo Finance.
2. Perform exploratory data analysis (EDA) to understand the data.
3. Run the genetic algorithm to optimize the stop-loss and take-profit parameters.


