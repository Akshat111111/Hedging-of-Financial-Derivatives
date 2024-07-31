# Market Regime Detection using Genetic Algorithm

## Description
This project aims to detect market regimes using a genetic algorithm. Market regimes, such as bull and bear markets, can significantly impact trading strategies. By identifying these regimes, traders can adapt their strategies accordingly. This project involves generating synthetic financial data, applying a genetic algorithm to detect market regimes, and performing exploratory data analysis (EDA).

## Dataset
The dataset `market_regime_detection.csv` contains synthetic financial data with the following columns:
- Date: The date of the observation
- Price: The price of the asset
- Volume: The trading volume of the asset
- Volatility: The volatility of the asset

## Key Techniques
1. **Fitness Function**: Defined using a simple moving average crossover strategy.
2. **Population Initialization**: Random initialization of the population.
3. **Selection**: Selecting the best individuals based on fitness scores.
4. **Crossover**: Combining parents to produce offspring.
5. **Mutation**: Introducing randomness to maintain genetic diversity.
6. **Genetic Algorithm**: Combining the above steps to evolve the population.

## Exploratory Data Analysis (EDA)
- Summary statistics of the dataset.
- Line plots of price, volume, and volatility over time.
- Pairplot of the dataset to visualize relationships between features.

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
