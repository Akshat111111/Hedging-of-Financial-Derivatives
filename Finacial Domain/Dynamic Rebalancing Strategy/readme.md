# Dynamic Rebalancing Strategy using Genetic Algorithm

## Description
This project aims to develop a dynamic rebalancing strategy for a portfolio using a genetic algorithm. The strategy optimizes the portfolio weights to maximize the risk-adjusted return. The project involves generating synthetic financial data, applying a genetic algorithm to optimize the rebalancing strategy, and performing exploratory data analysis (EDA).

## Dataset
The dataset `dynamic_rebalancing_strategy.csv` contains synthetic financial data with the following columns:
- Date: The date of the observation
- Asset_1: The price of the first asset
- Asset_2: The price of the second asset
- Asset_3: The price of the third asset

## Key Techniques
1. **Fitness Function**: Defined using the Sharpe ratio of the portfolio.
2. **Population Initialization**: Random initialization of the population.
3. **Selection**: Selecting the best individuals based on fitness scores.
4. **Crossover**: Combining parents to produce offspring.
5. **Mutation**: Introducing randomness to maintain genetic diversity.
6. **Genetic Algorithm**: Combining the above steps to evolve the population.

## Exploratory Data Analysis (EDA)
- Summary statistics of the dataset.
- Line plots of asset prices over time.
- Pairplot of the dataset to visualize relationships between assets.

## Results
The genetic algorithm was run for 50 generations with a population size of 20. The best fitness score was tracked over generations to monitor the algorithm's performance.

## How to Run
1. Ensure you have the necessary libraries installed:

numpy
pandas
matplotlib
seaborn

## Contributor
Ashish Kumar Patel
